"""Run long-term (1->2..10) and short-term (i->i+1) Dataset A experiments."""

import argparse
import csv
import json
import math
import os
import platform
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data import load_batch, scale_pair
from .model import MPCCNN, MultilevelCenterLoss, gaussian_mmd


# Supporting Information Tables S2 and S3: epoch, LR, batch size, center weight.
LONG_CONFIG = {
    2: (800, 5e-4, 90, 2.0), 3: (800, 6e-4, 90, 0.1),
    4: (800, 6e-4, 80, 1.2), 5: (800, 8e-4, 64, 1.0),
    6: (800, 2e-4, 64, 1.0), 7: (800, 5e-4, 50, 1.1),
    8: (800, 1.5e-4, 60, 1.0), 9: (800, 1e-4, 64, 1.0),
    10: (800, 2e-4, 64, 1.0),
}
SHORT_CONFIG = {
    1: (800, 5e-4, 90, 2.0), 2: (800, 6e-4, 128, 1.0),
    3: (800, 5e-4, 64, 1.0), 4: (800, 8e-4, 160, 2.0),
    5: (800, 6e-4, 160, 1.0), 6: (800, 8e-4, 256, 1.0),
    7: (800, 5e-4, 160, 2.0), 8: (800, 1e-3, 90, 2.0),
    9: (800, 5.5e-3, 90, 0.2),
}
PAPER_MPC_LONG = [85.06, 84.79, 80.10, 75.58, 95.16, 71.28, 64.04, 62.12, 66.37]
PAPER_MPC_SHORT = [85.06, 92.88, 96.13, 94.71, 74.31, 76.86, 90.20, 88.38, 40.10]
PAPER_CNN_LONG = [70.74, 75.78, 67.19, 64.90, 68.37, 61.25, 35.37, 37.93, 41.01]
PAPER_CNN_SHORT = [70.74, 77.71, 78.59, 64.38, 51.42, 70.85, 56.22, 66.74, 21.46]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def next_batch(loader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def training_weights(model, source_levels, target):
    """Estimate target MMD without changing BatchNorm's source running statistics."""
    was_training = model.extractor.training
    with torch.no_grad():
        model.extractor.eval()
        try:
            target_levels = model.extractor(target)
            weights, _ = model.weights(tuple(z.detach() for z in source_levels), target_levels)
        finally:
            model.extractor.train(was_training)
    return weights


def pretrain(model, source_loader, target_loader, epochs, lr, device):
    if epochs == 0:
        return []
    optimizer = torch.optim.Adam(model.extractor.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    for epoch in range(epochs):
        model.train()
        source_iter, target_iter = iter(source_loader), iter(target_loader)
        steps = max(len(source_loader), len(target_loader))
        loss_sum = 0.0
        for _ in range(steps):
            (xs, _), source_iter = next_batch(source_loader, source_iter)
            (xt,), target_iter = next_batch(target_loader, target_iter)
            source_levels = model.extractor(xs.to(device))
            target_levels = model.extractor(xt.to(device))
            loss = torch.stack([gaussian_mmd(s, t) for s, t in zip(source_levels, target_levels)]).sum()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite MMD loss in pretrain epoch {epoch + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        history.append(loss_sum / steps)
    return history


def fit(model, centers, source_loader, target_loader, epochs, lr, center_weight, center_beta, device, variant):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    history = []
    for epoch in range(epochs):
        model.train()
        target_iter = iter(target_loader)
        loss_sum, correct, count = 0.0, 0, 0
        for xs, ys in source_loader:
            xs, ys = xs.to(device), ys.to(device)
            source_levels = model.extractor(xs)
            if variant == "mpc":
                (xt,), target_iter = next_batch(target_loader, target_iter)
                weights = training_weights(model, source_levels, xt.to(device))
                probabilities = model.probabilities(source_levels, weights)
                task_loss = F.nll_loss(probabilities.clamp_min(1e-10).log(), ys)
                loss = task_loss + center_weight * centers(source_levels, ys)
            else:
                logits = model.logits(source_levels)[-1]
                probabilities = F.softmax(logits, dim=1)
                loss = F.cross_entropy(logits, ys)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite train loss in epoch {epoch + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if centers:
                centers.update(source_levels, ys, center_beta)
            loss_sum += loss.item() * len(ys)
            correct += (probabilities.argmax(dim=1) == ys).sum().item()
            count += len(ys)
        scheduler.step()
        history.append({"epoch": epoch + 1, "train_loss": loss_sum / count,
                        "train_accuracy": correct / count})
    return history


@torch.no_grad()
def evaluate(model, source, target, target_labels, device, batch_size, variant, seed):
    model.eval()
    if variant == "mpc":
        rng = np.random.default_rng(seed)
        source_index = rng.choice(len(source), min(len(source), 256), replace=False)
        target_index = rng.choice(len(target), min(len(target), 256), replace=False)
        source_levels = model.extractor(torch.from_numpy(source[source_index]).to(device))
        target_levels = model.extractor(torch.from_numpy(target[target_index]).to(device))
        weights, distances = model.weights(source_levels, target_levels)
    else:
        weights = distances = None
    predictions = []
    for (xt,) in DataLoader(TensorDataset(torch.from_numpy(target)), batch_size=batch_size):
        levels = model.extractor(xt.to(device))
        if variant == "mpc":
            probability = model.probabilities(levels, weights)
        else:
            probability = F.softmax(model.logits(levels)[-1], dim=1)
        predictions.append(probability.argmax(dim=1).cpu().numpy())
    predicted = np.concatenate(predictions)
    result = {"accuracy": float((predicted == target_labels).mean()),
              "correct": int((predicted == target_labels).sum()),
              "n_target": int(len(target_labels))}
    if variant == "mpc":
        result.update(mmd_weights=weights.cpu().tolist(), mmd_distances=distances.cpu().tolist())
    return result


def task_list(protocol, source, target):
    if (source is None) != (target is None):
        raise ValueError("provide both --source and --target")
    if source is not None:
        if not (1 <= source <= 9 and 2 <= target <= 10):
            raise ValueError("batch ids must be source 1..9 and target 2..10")
        kind = protocol if protocol != "all" else ("short" if target == source + 1 else "long")
        if kind == "short" and target != source + 1:
            raise ValueError("short protocol requires target = source + 1")
        if kind == "long" and source != 1:
            raise ValueError("long protocol requires source = 1")
        return [(kind, source, target)]
    tasks = []
    if protocol in ("long", "all"):
        tasks += [("long", 1, target_id) for target_id in range(2, 11)]
    if protocol in ("short", "all"):
        tasks += [("short", source_id, source_id + 1) for source_id in range(1, 10)]
    return tasks


def run_task(args, kind, source_id, target_id, device, output):
    seed_everything(args.seed)
    x_source, y_source = load_batch(args.data_dir / f"batch{source_id}.dat")
    x_target, y_target = load_batch(args.data_dir / f"batch{target_id}.dat")
    x_source, x_target, mean, std = scale_pair(x_source, x_target, args.scaling)
    paper_epochs, paper_lr, paper_batch, paper_center_weight = (
        LONG_CONFIG[target_id] if kind == "long" else SHORT_CONFIG[source_id])
    epochs = args.epochs if args.epochs is not None else paper_epochs
    pretrain_epochs = args.pretrain_epochs if args.pretrain_epochs is not None else 100
    batch_size = args.batch_size if args.batch_size is not None else paper_batch
    lr = args.learning_rate if args.learning_rate is not None else paper_lr
    center_weight = (args.center_weight if args.center_weight is not None
                     else paper_center_weight)
    source_loader = DataLoader(TensorDataset(torch.from_numpy(x_source), torch.from_numpy(y_source)),
                               batch_size=batch_size, shuffle=True, drop_last=False)
    target_loader = DataLoader(TensorDataset(torch.from_numpy(x_target)), batch_size=batch_size,
                               shuffle=True, drop_last=False)
    model = MPCCNN(dropout=args.dropout, architecture=args.architecture).to(device)
    if args.model == "cnn":
        def initialize(module):
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, a=0.01, nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        model.apply(initialize)
    centers = MultilevelCenterLoss().to(device) if args.model == "mpc" else None
    pretrain_history = (pretrain(model, source_loader, target_loader, pretrain_epochs, lr, device)
                        if args.model == "mpc" else [])
    train_history = fit(model, centers, source_loader, target_loader, epochs, lr,
                        center_weight, args.center_beta, device, args.model)
    metrics = evaluate(model, x_source, x_target, y_target, device, batch_size, args.model, args.seed)
    paper_values = (PAPER_MPC_LONG if args.model == "mpc" else PAPER_CNN_LONG) if kind == "long" else (
        PAPER_MPC_SHORT if args.model == "mpc" else PAPER_CNN_SHORT)
    paper_accuracy = paper_values[target_id - 2 if kind == "long" else source_id - 1] / 100
    row = {"protocol": kind, "source": source_id, "target": target_id, "model": args.model,
           "accuracy": metrics["accuracy"], "paper_accuracy": paper_accuracy,
           "gap_percentage_points": 100 * (metrics["accuracy"] - paper_accuracy),
           "correct": metrics["correct"], "n_target": metrics["n_target"],
           "epochs": epochs, "pretrain_epochs": len(pretrain_history), "batch_size": batch_size,
           "learning_rate": lr, "center_weight": center_weight,
           "center_beta": args.center_beta, "scaling": args.scaling, "seed": args.seed,
           "architecture": args.architecture}
    stem = f"{kind}_{source_id}_to_{target_id}_{args.model}"
    torch.save({"model_state": model.state_dict(), "centers": centers.state_dict() if centers else None,
                "source_mean": torch.from_numpy(mean), "source_std": torch.from_numpy(std),
                "config": row}, output / f"{stem}.pt")
    (output / f"{stem}.json").write_text(json.dumps({"result": row, "metrics": metrics,
        "pretrain_loss": pretrain_history, "training": train_history}, indent=2), encoding="utf-8")
    print(f"{kind} {source_id}->{target_id} {args.model}: {100*metrics['accuracy']:.2f}% "
          f"(paper {100*paper_accuracy:.2f}%; gap {row['gap_percentage_points']:+.2f} pp)", flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs") / "mpc")
    parser.add_argument("--protocol", choices=("long", "short", "all"), default="all")
    parser.add_argument("--model", choices=("mpc", "cnn"), default="mpc")
    parser.add_argument("--architecture", choices=("figure3", "legacy"), default="figure3",
                        help="Figure 3 head/conv activation layout; legacy reads earlier results")
    parser.add_argument("--source", type=int)
    parser.add_argument("--target", type=int)
    parser.add_argument("--epochs", type=int, help="override the paper's 800 formal epochs")
    parser.add_argument("--pretrain-epochs", type=int, help="default 100; unspecified in SI")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--dropout", type=float, default=0.5, help="unspecified in paper")
    parser.add_argument("--center-beta", type=float, default=0.5, help="Eq. 10 update rate, unspecified in paper")
    parser.add_argument("--center-weight", type=float, help="override SI center-loss weight for a diagnostic run")
    parser.add_argument("--scaling", choices=("source", "pooled"), default="source",
                        help="z-score statistics from source only or pooled unlabeled source+target")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="one pair, one pretrain and formal epoch")
    parser.add_argument("--skip-existing", action="store_true", help="reuse completed per-task JSON/checkpoints")
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.pretrain_epochs is not None and args.pretrain_epochs < 0:
        parser.error("--pretrain-epochs must be nonnegative")
    if args.center_weight is not None and args.center_weight < 0:
        parser.error("--center-weight must be nonnegative")
    if args.smoke:
        args.epochs, args.pretrain_epochs = 1, 1
        args.source, args.target = 1, 2
        args.protocol = "short"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    environment = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                   "python": platform.python_version(), "platform": platform.platform(),
                   "torch": torch.__version__, "numpy": np.__version__,
                   "device": str(device), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   "seed": args.seed, "command_config": vars(args) | {"data_dir": str(args.data_dir),
                       "output_dir": str(args.output_dir)}}
    (args.output_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    rows = []
    for kind, source, target in task_list(args.protocol, args.source, args.target):
        stem = f"{kind}_{source}_to_{target}_{args.model}"
        result_path = args.output_dir / f"{stem}.json"
        checkpoint_path = args.output_dir / f"{stem}.pt"
        if args.skip_existing and result_path.exists() and checkpoint_path.exists():
            row = json.loads(result_path.read_text(encoding="utf-8"))["result"]
            expected_epochs = args.epochs if args.epochs is not None else 800
            expected_pretrain = (args.pretrain_epochs if args.pretrain_epochs is not None else 100) if args.model == "mpc" else 0
            paper_config = LONG_CONFIG[target] if kind == "long" else SHORT_CONFIG[source]
            expected_center_weight = (args.center_weight if args.center_weight is not None
                                      else paper_config[3])
            expected_batch_size = args.batch_size if args.batch_size is not None else paper_config[2]
            expected_lr = args.learning_rate if args.learning_rate is not None else paper_config[1]
            if (row["epochs"] != expected_epochs or row["pretrain_epochs"] != expected_pretrain
                    or row["seed"] != args.seed or row.get("scaling") != args.scaling
                    or row.get("architecture", "legacy") != args.architecture
                    or row["center_weight"] != expected_center_weight
                    or row["batch_size"] != expected_batch_size
                    or row["learning_rate"] != expected_lr
                    or row["center_beta"] != args.center_beta):
                raise ValueError(f"existing run uses different settings: {result_path}")
            print(f"Reusing {stem}", flush=True)
        else:
            row = run_task(args, kind, source, target, device, args.output_dir)
        rows.append(row)
        with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        summary = {"run_type": "pilot" if any(row["epochs"] < 800 for row in rows) else "paper_formal_epochs",
                   "completed_tasks": len(rows), "expected_tasks": len(task_list(args.protocol, args.source, args.target)),
                   "protocols": {}}
        for group in ("long", "short"):
            selected = [row for row in rows if row["protocol"] == group]
            if selected:
                summary["protocols"][group] = {
                    "tasks": len(selected),
                    "mean_accuracy": float(np.mean([row["accuracy"] for row in selected])),
                    "paper_mean_accuracy": float(np.mean([row["paper_accuracy"] for row in selected])),
                    "mean_gap_percentage_points": float(np.mean([row["gap_percentage_points"] for row in selected]))}
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Completed {len(rows)} tasks; mean accuracy {100*np.mean([r['accuracy'] for r in rows]):.2f}%")


if __name__ == "__main__":
    main()
