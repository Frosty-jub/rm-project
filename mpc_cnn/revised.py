"""Controlled MPC-CNN experiments; all departures from the paper are recorded.

Target labels are used only in reporting. Final-epoch checkpoints are evaluated;
the runner never selects a checkpoint or updates a parameter using target labels.
"""

import argparse
import hashlib
import json
import platform
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import load_batch
from .model import MPCCNN, MultilevelCenterLoss
from .train import LONG_CONFIG, SHORT_CONFIG, PAPER_MPC_LONG, PAPER_MPC_SHORT, seed_everything


@dataclass
class Recipe:
    seed: int = 42
    scaling: str = "domain"
    transform: str = "none"
    row_l2: bool = False
    head: str = "linear"
    dropout: float = 0.1
    initialize: str = "kaiming"
    pretrain_epochs: int = 600
    pretrain_objective: str = "mmd"
    pretrain_mmd_weight: float = 1.0
    pretrain_steps: str = "max"
    epochs: int = 800
    learning_rate: float = 0.0
    batch_size: int = 0
    center_multiplier: float = 0.01
    center_reduction: str = "mean"
    center_normalize: bool = False
    center_init: str = "source"
    center_beta: float = 0.5
    center_warmup: int = 50
    weight_decay: float = 1e-4
    objective: str = "ensemble"
    ensemble_gradient: str = "detached"
    normalize_weights: bool = True
    formal_mmd_weight: float = 0.0
    mmd_mode: str = "consistent_eval"
    bandwidth_mode: str = "median_detached"
    fixed_bandwidth: float = 1.0
    schedule: str = "cosine"
    log_every: int = 100

    def validate(self):
        choices = {"scaling": ("source", "pooled", "domain"),
                   "transform": ("none", "signed_log"),
                   "head": ("linear", "leaky", "relu"),
                   "initialize": ("kaiming", "torch"),
                   "pretrain_objective": ("mmd", "ce_mmd"),
                   "pretrain_steps": ("max", "source"),
                   "center_reduction": ("mean", "sum"),
                   "center_init": ("source", "zero"),
                   "objective": ("ensemble", "heads"),
                   "ensemble_gradient": ("detached", "differentiable"),
                   "mmd_mode": ("consistent_eval", "mixed"),
                   "bandwidth_mode": ("median_detached", "median_differentiable", "fixed"),
                   "schedule": ("cosine", "step")}
        for key, permitted in choices.items():
            if getattr(self, key) not in permitted:
                raise ValueError(f"{key} must be one of {permitted}")
        for key in ("row_l2", "center_normalize", "normalize_weights"):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f"{key} must be boolean")
        for key in ("pretrain_epochs", "epochs", "center_warmup", "log_every"):
            value = getattr(self, key)
            if type(value) is not int or value < (1 if key in ("epochs", "log_every") else 0):
                raise ValueError(f"{key} must be a nonnegative integer; epochs and log_every must be positive")
        if type(self.batch_size) is not int or self.batch_size < 0:
            raise ValueError("batch_size must be a nonnegative integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for key in ("dropout", "pretrain_mmd_weight", "learning_rate", "center_multiplier",
                    "center_beta", "weight_decay", "formal_mmd_weight", "fixed_bandwidth"):
            value = getattr(self, key)
            if not isinstance(value, (float, int)) or not np.isfinite(value):
                raise ValueError(f"{key} must be finite")
            if value < 0 or (key == "fixed_bandwidth" and value == 0):
                raise ValueError(f"{key} must be nonnegative; fixed_bandwidth must be positive")
        if self.dropout >= 1 or self.center_beta > 1:
            raise ValueError("dropout must be below 1 and center_beta at most 1")
        if self.pretrain_objective == "mmd" and self.pretrain_mmd_weight == 0 and self.pretrain_epochs:
            raise ValueError("MMD-only pretraining needs a positive MMD weight")


def task_parameters(source_id, target_id):
    if type(source_id) is not int or type(target_id) is not int:
        raise ValueError("batch IDs must be integers")
    if source_id == 1 and target_id in LONG_CONFIG:
        return LONG_CONFIG[target_id], PAPER_MPC_LONG[target_id - 2]
    if source_id in SHORT_CONFIG and target_id == source_id + 1:
        return SHORT_CONFIG[source_id], PAPER_MPC_SHORT[source_id - 1]
    raise ValueError("valid tasks are long 1->2..10 or short i->i+1")


def revised_mmd(source, target, recipe):
    """Biased Gaussian MMD with an explicit bandwidth differentiation policy."""
    joined = torch.cat((source, target), 0)
    squared = torch.cdist(joined, joined).square()
    if recipe.bandwidth_mode == "fixed":
        bandwidth = squared.new_tensor(recipe.fixed_bandwidth)
    else:
        candidates = squared[squared.detach() > 0]
        bandwidth = candidates.median().clamp_min(1e-6) if candidates.numel() else squared.new_tensor(1.0)
        if recipe.bandwidth_mode == "median_detached":
            bandwidth = bandwidth.detach()
    kernel = torch.exp(-squared / bandwidth)
    n = len(source)
    squared_mmd = kernel[:n, :n].mean() + kernel[n:, n:].mean() - 2 * kernel[:n, n:].mean()
    return squared_mmd.clamp_min(1e-12).sqrt()


class RevisedMPC(MPCCNN):
    def __init__(self, recipe):
        super().__init__(dropout=recipe.dropout, architecture="figure3")
        self.head = recipe.head
        self.recipe = recipe

    def weights(self, source_levels, target_levels):
        distances = torch.stack([revised_mmd(s, t, self.recipe)
                                 for s, t in zip(source_levels, target_levels)])
        raw = torch.sigmoid(distances).reciprocal()
        return (raw / raw.sum() if self.recipe.normalize_weights else raw), distances

    def logits(self, levels):
        def activate(z):
            if self.head == "relu":
                return F.relu(z)
            if self.head == "leaky":
                return F.leaky_relu(z, negative_slope=0.01)
            return z
        return tuple(head(activate(z)) for head, z in zip(self.classifiers, levels))


def prepare(source, target, recipe):
    recipe.validate()
    source, target = source.reshape(len(source), -1), target.reshape(len(target), -1)
    if source.shape[1] != 128 or target.shape[1] != 128 or not len(source) or not len(target):
        raise ValueError("source and target must be nonempty arrays with 128 features")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("source and target features must be finite")
    if recipe.transform == "signed_log":
        source = np.sign(source) * np.log1p(np.abs(source))
        target = np.sign(target) * np.log1p(np.abs(target))
    def stats(x):
        mean, std = x.mean(0), x.std(0)
        return mean, np.where(std < 1e-8, 1.0, std)
    if recipe.scaling == "source":
        sm, ss = stats(source)
        tm, ts = sm, ss
    elif recipe.scaling == "pooled":
        sm, ss = stats(np.concatenate((source, target)))
        tm, ts = sm, ss
    elif recipe.scaling == "domain":
        sm, ss = stats(source)
        tm, ts = stats(target)
    else:
        raise ValueError(recipe.scaling)
    xs = ((source - sm) / ss).reshape(-1, 1, 16, 8).astype(np.float32)
    xt = ((target - tm) / ts).reshape(-1, 1, 16, 8).astype(np.float32)
    if recipe.row_l2:
        xs = xs / np.maximum(np.linalg.norm(xs.reshape(len(xs), -1), axis=1), 1e-12)[:, None, None, None]
        xt = xt / np.maximum(np.linalg.norm(xt.reshape(len(xt), -1), axis=1), 1e-12)[:, None, None, None]
    if not np.isfinite(xs).all() or not np.isfinite(xt).all():
        raise ValueError("preprocessing produced nonfinite features")
    return xs, xt, {"source_mean": sm.tolist(), "source_std": ss.tolist(),
                    "target_mean": tm.tolist(), "target_std": ts.tolist()}


def transform_with_saved(raw, preprocessing, recipe, domain):
    """Apply exactly the saved feature transform for checkpoint inference."""
    if domain not in ("source", "target"):
        raise ValueError("domain must be source or target")
    flat = np.asarray(raw, dtype=np.float32).reshape(len(raw), -1)
    if flat.shape[1] != 128 or not np.isfinite(flat).all():
        raise ValueError("raw features must be finite with 128 columns")
    if recipe.transform == "signed_log":
        flat = np.sign(flat) * np.log1p(np.abs(flat))
    mean = np.asarray(preprocessing[f"{domain}_mean"], dtype=np.float32)
    std = np.asarray(preprocessing[f"{domain}_std"], dtype=np.float32)
    scaled = ((flat - mean) / std).astype(np.float32)
    if recipe.row_l2:
        scaled = scaled / np.maximum(np.linalg.norm(scaled, axis=1, keepdims=True), 1e-12)
    if not np.isfinite(scaled).all():
        raise ValueError("checkpoint preprocessing produced nonfinite features")
    return scaled.reshape(-1, 1, 16, 8)


@torch.no_grad()
def predict_checkpoint(checkpoint, raw, domain="target", device="cpu"):
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    recipe = Recipe(**saved["recipe"])
    recipe.validate()
    model = RevisedMPC(recipe).to(device)
    model.load_state_dict(saved["model_state"])
    model.eval()
    x = torch.from_numpy(transform_with_saved(raw, saved["preprocessing"], recipe, domain)).to(device)
    weights = torch.as_tensor(saved["evaluation_weights"], device=device)
    probabilities = torch.cat([model.probabilities(model.extractor(batch), weights)
                               for batch in x.split(512)])
    return probabilities.cpu().numpy()


def array_digest(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(str(array.dtype).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def provenance(data_dir, source_id, target_id, raw_s, labels_s, raw_t, labels_t):
    code_files = (Path(__file__), Path(__file__).with_name("model.py"),
                  Path(__file__).with_name("data.py"), Path(__file__).with_name("train.py"))
    return {"code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files},
            "data_files": {str(batch): {"path": str((data_dir / f"batch{batch}.dat").resolve()),
                                         "sha256": hashlib.sha256((data_dir / f"batch{batch}.dat").read_bytes()).hexdigest()}
                           for batch in (source_id, target_id)},
            "source_features_sha256": array_digest(raw_s),
            "source_labels_sha256": array_digest(labels_s),
            "target_features_sha256": array_digest(raw_t),
            "target_labels_sha256": array_digest(labels_t)}


@contextmanager
def temporary_mode(module, training, dropout=None):
    states = [(child, child.training) for child in module.modules()]
    module.train(training)
    if dropout is not None:
        for child in module.modules():
            if isinstance(child, nn.Dropout):
                child.train(dropout)
    try:
        yield
    finally:
        for child, state in states:
            child.training = state


def initialize(model, recipe):
    if recipe.initialize == "kaiming":
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, a=0.01, nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


def permutations(n, batch_size, generator, device):
    order = torch.randperm(n, generator=generator).to(device)
    return list(order.split(batch_size))


def center_features(levels, recipe):
    return tuple(F.normalize(z, dim=1) for z in levels) if recipe.center_normalize else levels


def center_loss(centers, levels, labels, recipe):
    levels = center_features(levels, recipe)
    loss = centers(levels, labels)
    return loss * len(labels) if recipe.center_reduction == "sum" else loss


def mmd_weights(model, xs, xt, source_levels, recipe):
    gradient_context = torch.no_grad() if recipe.ensemble_gradient == "detached" else nullcontext()
    with gradient_context:
        if recipe.mmd_mode == "mixed":
            with temporary_mode(model.extractor, False):
                used_source = (tuple(z.detach() for z in source_levels)
                               if recipe.ensemble_gradient == "detached" else source_levels)
                return model.weights(used_source, model.extractor(xt))[0]
        with temporary_mode(model.extractor, False):
            return model.weights(model.extractor(xs), model.extractor(xt))[0]


@torch.no_grad()
def evaluate(model, xs, ys, xt, yt, seed):
    model.eval()
    rng = np.random.default_rng(seed)
    si = rng.choice(len(xs), min(len(xs), 256), replace=False)
    ti = rng.choice(len(xt), min(len(xt), 256), replace=False)
    weights, distances = model.weights(model.extractor(xs[si]), model.extractor(xt[ti]))
    result = {"weights": weights.cpu().tolist(), "mmd": distances.cpu().tolist()}
    for name, data, labels in (("source", xs, ys), ("target", xt, yt)):
        predictions, positives = [], []
        for batch in data.split(512):
            levels = model.extractor(batch)
            predictions.append(model.probabilities(levels, weights).argmax(1))
            positives.append(torch.stack([(z > 0).float().mean() for z in levels]) * len(batch))
        predicted = torch.cat(predictions)
        result[name + "_accuracy"] = float((predicted == labels).float().mean())
        result[name + "_predicted_counts"] = torch.bincount(predicted, minlength=6).cpu().tolist()
        result[name + "_positive_feature_fractions"] = (torch.stack(positives).sum(0) / len(data)).cpu().tolist()
    return result


def run(recipe, source_id, target_id, output, data_dir, device, pretrained=None):
    recipe.validate()
    (_, si_lr, si_batch, si_center), reference = task_parameters(source_id, target_id)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but CUDA is unavailable")
    lr, batch_size = recipe.learning_rate or si_lr, recipe.batch_size or si_batch
    config = asdict(recipe) | {"source": source_id, "target": target_id,
                               "effective_lr": lr, "effective_batch_size": batch_size,
                               "effective_center_weight": si_center * recipe.center_multiplier}
    raw_s, labels_s = load_batch(data_dir / f"batch{source_id}.dat")
    raw_t, labels_t = load_batch(data_dir / f"batch{target_id}.dat")
    evidence = provenance(data_dir, source_id, target_id, raw_s, labels_s, raw_t, labels_t)
    runtime = {"python": platform.python_version(), "platform": platform.platform(),
               "torch": str(torch.__version__), "numpy": np.__version__,
               "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
               "device": str(device),
               "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    evidence["runtime"] = runtime
    evidence["pretrained_checkpoint_sha256"] = (hashlib.sha256(Path(pretrained).read_bytes()).hexdigest()
                                                 if pretrained else None)
    pretrain_keys = ("seed", "scaling", "transform", "row_l2", "head", "dropout", "initialize", "pretrain_epochs",
                    "pretrain_objective", "pretrain_mmd_weight", "pretrain_steps", "learning_rate", "batch_size",
                    "weight_decay", "bandwidth_mode", "fixed_bandwidth")
    pretrain_config = {k: config[k] for k in pretrain_keys} | {
        "effective_lr": lr, "effective_batch_size": batch_size,
        "source": source_id, "target": target_id,
        "code_sha256": evidence["code_sha256"], "runtime": runtime,
        "source_features_sha256": evidence["source_features_sha256"],
        "source_labels_sha256": evidence["source_labels_sha256"],
        "target_features_sha256": evidence["target_features_sha256"]}
    final_path = output / "result.json"
    if final_path.exists():
        previous = json.loads(final_path.read_text(encoding="utf-8"))
        checkpoint = output / "model.pt"
        if (previous.get("config") != config or previous.get("provenance") != evidence
                or not checkpoint.is_file()
                or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != previous.get("model_sha256")):
            raise ValueError(f"Output directory contains incompatible or incomplete artifacts: {output}")
        print("REUSE", output, previous["metrics"]["target_accuracy"], flush=True)
        return previous
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory has an incomplete run: {output}")
    saved_pretrained = None
    if pretrained:
        saved_pretrained = torch.load(pretrained, map_location="cpu", weights_only=True)
        if saved_pretrained.get("pretrain_config") != pretrain_config:
            raise ValueError("Pretraining recipe, data, code, or runtime mismatch")
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    seed_everything(recipe.seed)
    torch.set_num_threads(2)
    xs, xt, preprocessing = prepare(raw_s, raw_t, recipe)
    xs, xt = torch.from_numpy(xs).to(device), torch.from_numpy(xt).to(device)
    ys, yt = torch.from_numpy(labels_s).to(device), torch.from_numpy(labels_t).to(device)
    config_path = output / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    environment = runtime | {"provenance": evidence, "selection": "final_epoch",
                   "target_feature_policy": "unlabeled target features used for MMD and optional scaling",
                   "target_labels_training": False}
    (output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    model = RevisedMPC(recipe).to(device)
    initialize(model, recipe)
    history = []
    history_path = output / "history.jsonl"
    def record(item):
        history.append(item)
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item) + "\n")
    if pretrained:
        model.load_state_dict(saved_pretrained["model_state"])
        record({"stage": "pretrain_reused", "checkpoint": str(Path(pretrained).resolve()),
                "sha256": evidence["pretrained_checkpoint_sha256"]})
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=recipe.weight_decay)
        generator = torch.Generator().manual_seed(recipe.seed + 101)
        for epoch in range(recipe.pretrain_epochs):
            model.train()
            s_batches = permutations(len(xs), batch_size, generator, device)
            t_batches = permutations(len(xt), batch_size, generator, device)
            steps = max(len(s_batches), len(t_batches)) if recipe.pretrain_steps == "max" else len(s_batches)
            total = 0.0
            for step in range(steps):
                si, ti = s_batches[step % len(s_batches)], t_batches[step % len(t_batches)]
                # Concatenate domains to use the same batch normalization; disable
                # dropout for the MMD estimate so stochastic masks are not domain drift.
                with temporary_mode(model.extractor, True, dropout=False):
                    joined = model.extractor(torch.cat((xs[si], xt[ti])))
                sl, tl = tuple(z[:len(si)] for z in joined), tuple(z[len(si):] for z in joined)
                mmd = torch.stack([revised_mmd(s, t, recipe) for s, t in zip(sl, tl)]).sum()
                loss = recipe.pretrain_mmd_weight * mmd
                if recipe.pretrain_objective == "ce_mmd":
                    loss = loss + torch.stack([F.cross_entropy(z, ys[si]) for z in model.logits(sl)]).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite pretraining loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach())
            record({"stage": "pretrain", "epoch": epoch + 1, "loss": total / steps})
            if (epoch + 1) % recipe.log_every == 0:
                print(output.name, "pretrain", epoch + 1, round(total / steps, 5), flush=True)
        torch.save({"model_state": model.state_dict(), "pretrain_config": pretrain_config}, output / "pretrained.pt")
    centers = MultilevelCenterLoss().to(device)
    if recipe.center_init == "source":
        with torch.no_grad(), temporary_mode(model.extractor, False):
            levels = center_features(model.extractor(xs), recipe)
            for z, center in zip(levels, centers.centers):
                for label in ys.unique():
                    center[label] = z[ys == label].mean(0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=recipe.weight_decay)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=recipe.epochs)
                 if recipe.schedule == "cosine" else torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5))
    generator = torch.Generator().manual_seed(recipe.seed + 202)
    torch.manual_seed(recipe.seed + 303)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(recipe.seed + 303)
    for epoch in range(recipe.epochs):
        model.train()
        s_batches = permutations(len(xs), batch_size, generator, device)
        t_batches = permutations(len(xt), batch_size, generator, device)
        ce_total, center_total, count = 0.0, 0.0, 0
        center_coefficient = si_center * recipe.center_multiplier * min(1.0, (epoch + 1) / max(1, recipe.center_warmup))
        for step, si in enumerate(s_batches):
            ti = t_batches[step % len(t_batches)]
            sl = model.extractor(xs[si])
            weights = mmd_weights(model, xs[si], xt[ti], sl, recipe)
            probability = model.probabilities(sl, weights)
            ce = F.nll_loss(probability.clamp_min(1e-10).log(), ys[si])
            if recipe.objective == "heads":
                ce = torch.stack([F.cross_entropy(z, ys[si]) for z in model.logits(sl)]).mean()
            cl = center_loss(centers, sl, ys[si], recipe)
            loss = ce + center_coefficient * cl
            if recipe.formal_mmd_weight:
                with temporary_mode(model.extractor, False):
                    es, et = model.extractor(xs[si]), model.extractor(xt[ti])
                loss = loss + recipe.formal_mmd_weight * torch.stack([revised_mmd(s, t, recipe) for s, t in zip(es, et)]).sum()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite formal loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            centers.update(center_features(sl, recipe), ys[si], recipe.center_beta)
            ce_total += float(ce.detach()) * len(si)
            center_total += float(cl.detach()) * len(si)
            count += len(si)
        scheduler.step()
        item = {"stage": "formal", "epoch": epoch + 1, "ce": ce_total / count,
                "center": center_total / count, "center_coefficient": center_coefficient}
        if (epoch + 1) % recipe.log_every == 0 or epoch + 1 == recipe.epochs:
            item["metrics"] = evaluate(model, xs, ys, xt, yt, recipe.seed)
            print(output.name, "formal", epoch + 1,
                  "source", round(100 * item["metrics"]["source_accuracy"], 2),
                  "target", round(100 * item["metrics"]["target_accuracy"], 2), flush=True)
        record(item)
    metrics = evaluate(model, xs, ys, xt, yt, recipe.seed)
    result = {"config": config, "metrics": metrics, "paper_accuracy_percent": reference,
              "gap_percentage_points": 100 * metrics["target_accuracy"] - reference,
              "seconds": time.monotonic() - start, "selection": "final_epoch",
              "training_target_labels_used": False, "provenance": evidence}
    checkpoint = output / "model.pt"
    torch.save({"model_state": model.state_dict(), "centers": centers.state_dict(), "config": config,
                "recipe": asdict(recipe), "preprocessing": preprocessing,
                "evaluation_weights": metrics["weights"], "provenance": evidence}, checkpoint)
    result["model_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    final_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--source", type=int, default=1)
    parser.add_argument("--target", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--pretrained", type=Path)
    args = parser.parse_args()
    recipe = Recipe(**json.loads(args.recipe.read_text(encoding="utf-8")))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    result = run(recipe, args.source, args.target, args.output, args.data_dir, device, args.pretrained)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
