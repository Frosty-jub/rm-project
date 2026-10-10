"""Read saved checkpoints and probe feature/gradient collapse without training.

Run from the repository root: python scripts/audit_reproduction.py
"""

import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpc_cnn.data import load_batch
from mpc_cnn.model import MPCCNN

CHECKPOINTS = [
    "runs/full_1_to_2_bnfix/short_1_to_2_mpc.pt",
    "runs/full_figure3/long_1_to_2_mpc.pt",
    "runs/diagnostic_figure3_no_pretrain/short_1_to_2_mpc.pt",
    "runs/diagnostic_figure3_no_center/short_1_to_2_mpc.pt",
    "runs/diagnostic_figure3_cnn/short_1_to_2_cnn.pt",
]


def inspect_checkpoint(relative_path):
    path = ROOT / relative_path
    saved = torch.load(path, map_location="cpu", weights_only=True)
    config = saved["config"]
    model = MPCCNN(dropout=0.0, architecture=config.get("architecture", "legacy"))
    model.load_state_dict(saved["model_state"])
    model.eval()
    domains = {}
    for domain, batch_id in (("source", config["source"]), ("target", config["target"])):
        x, y = load_batch(ROOT / "dataset" / f"batch{batch_id}.dat")
        x = (x.reshape(len(x), -1) - saved["source_mean"].numpy()) / saved["source_std"].numpy()
        with torch.no_grad():
            levels = model.extractor(torch.from_numpy(x.reshape(-1, 1, 16, 8)))
        domains[domain] = (levels, torch.from_numpy(y))
    if config["model"] == "mpc":
        rng = np.random.default_rng(config["seed"])
        selected = []
        for levels, labels in domains.values():
            indices = rng.choice(len(labels), min(len(labels), 256), replace=False)
            selected.append(tuple(z[indices] for z in levels))
        with torch.no_grad():
            weights, _ = model.weights(*selected)
    else:
        weights = None
    report = {"checkpoint": relative_path, "config": config,
              "saved_environment": json.loads((path.parent / "environment.json").read_text())}
    for domain, (levels, labels) in domains.items():
        feature_stats = []
        for z in levels:
            head_input = F.relu(z) if model.architecture == "figure3" else z
            feature_stats.append({
                "positive_fraction_before_head": float((z > 0).float().mean()),
                "mean_feature_std_after_head_activation": float(head_input.std(0).mean()),
                "all_zero_rows_after_head_activation": int((head_input == 0).all(1).sum()),
            })
        detached = tuple(z.detach().requires_grad_(True) for z in levels)
        probabilities = (model.probabilities(detached, weights) if weights is not None
                         else F.softmax(model.logits(detached)[-1], dim=1))
        loss = F.nll_loss(probabilities.clamp_min(1e-10).log(), labels)
        gradients = torch.autograd.grad(loss, detached, allow_unused=True)
        predictions = probabilities.argmax(1)
        report[domain] = {
            "n": len(labels), "accuracy": float((predictions == labels).float().mean()),
            "predicted_counts_raw_ids_1_to_6": torch.bincount(predictions, minlength=6).tolist(),
            "features": feature_stats,
            "task_gradient_norm_wrt_each_level": [None if g is None else float(g.norm()) for g in gradients],
        }
    reference = json.loads(path.with_suffix(".json").read_text())["result"]["accuracy"]
    report["saved_target_accuracy"] = reference
    report["target_accuracy_matches_saved"] = abs(report["target"]["accuracy"] - reference) < 1e-6
    return report


def mode_probe():
    torch.manual_seed(42)
    model = MPCCNN(dropout=0.5)
    x = torch.randn(32, 1, 16, 8)
    with torch.no_grad():
        model.train()
        source_train = model.extractor(x)
        model.eval()
        target_eval = model.extractor(x)
        _, mixed = model.weights(source_train, target_eval)
        source_eval = model.extractor(x)
        _, consistent = model.weights(source_eval, target_eval)
    return {"identical_input_mmd_train_vs_eval": mixed.tolist(),
            "identical_input_mmd_eval_vs_eval": consistent.tolist(),
            "interpretation": "Train/eval mode differences inject MMD even for identical input domains."}


def main():
    torch.set_num_threads(4)
    report = {"audit_runtime": {"python": platform.python_version(), "torch": torch.__version__,
                                "numpy": np.__version__},
              "checkpoints": [inspect_checkpoint(p) for p in CHECKPOINTS],
              "mode_probe": mode_probe()}
    output = ROOT / "runs" / "audit_20260929" / "checkpoint_review.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for row in report["checkpoints"]:
        print(row["checkpoint"], "source", round(100 * row["source"]["accuracy"], 2),
              "target", round(100 * row["target"]["accuracy"], 2),
              "matches saved", row["target_accuracy_matches_saved"])
    print(json.dumps(report["mode_probe"], indent=2))
    print(output)


if __name__ == "__main__":
    main()
