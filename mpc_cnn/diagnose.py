"""Inspect source/target counts and confusion using raw class IDs 1 through 6."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import RAW_ID_TO_PAPER_GAS, load_batch
from .model import MPCCNN


@torch.no_grad()
def predict(model, features, weights, variant, batch_size=256):
    outputs = []
    for start in range(0, len(features), batch_size):
        levels = model.extractor(torch.from_numpy(features[start:start + batch_size]))
        probabilities = (model.probabilities(levels, weights) if variant == "mpc"
                         else F.softmax(model.logits(levels)[-1], dim=1))
        outputs.append(probabilities.argmax(1).numpy())
    return np.concatenate(outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    source, source_y = load_batch(args.data_dir / f"batch{config['source']}.dat")
    target, target_y = load_batch(args.data_dir / f"batch{config['target']}.dat")
    mean = checkpoint["source_mean"].numpy()
    std = checkpoint["source_std"].numpy()
    source = ((source.reshape(len(source), -1) - mean) / std).reshape(-1, 1, 16, 8)
    target = ((target.reshape(len(target), -1) - mean) / std).reshape(-1, 1, 16, 8)
    model = MPCCNN(dropout=0.0, architecture=config.get("architecture", "legacy"))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    if config["model"] == "mpc":
        rng = np.random.default_rng(config["seed"])
        source_idx = rng.choice(len(source), min(len(source), 256), replace=False)
        target_idx = rng.choice(len(target), min(len(target), 256), replace=False)
        weights, distances = model.weights(
            model.extractor(torch.from_numpy(source[source_idx])),
            model.extractor(torch.from_numpy(target[target_idx])))
    else:
        weights = distances = None
    source_pred = predict(model, source, weights, config["model"])
    target_pred = predict(model, target, weights, config["model"])
    confusion = np.zeros((6, 6), dtype=np.int64)
    np.add.at(confusion, (target_y, target_pred), 1)
    report = {"raw_class_ids_in_matrix_order": [1, 2, 3, 4, 5, 6],
              "paper_gas_names_in_matrix_order": [RAW_ID_TO_PAPER_GAS[i] for i in range(1, 7)],
              "source_accuracy": float((source_pred == source_y).mean()),
              "target_accuracy": float((target_pred == target_y).mean()),
              "target_true_counts": np.bincount(target_y, minlength=6).tolist(),
              "target_predicted_counts": np.bincount(target_pred, minlength=6).tolist(),
              "target_confusion_rows_true_columns_predicted": confusion.tolist()}
    if weights is not None:
        report["mmd_weights"] = weights.tolist()
        report["mmd_distances"] = distances.tolist()
    output = args.checkpoint.with_suffix(".diagnostics.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
