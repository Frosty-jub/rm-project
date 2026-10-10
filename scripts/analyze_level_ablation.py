"""Replay each saved checkpoint and measure the three heads and their fusion.

Target labels are read only for this post-training diagnostic. No model or
weight selection is performed using the target labels.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mpc_cnn.data import load_batch
from mpc_cnn.revised import Recipe, RevisedMPC, transform_with_saved


@torch.no_grad()
def inspect(checkpoint, data_dir):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    recipe = Recipe(**saved["recipe"])
    model = RevisedMPC(recipe)
    model.load_state_dict(saved["model_state"])
    model.eval()
    weights = np.asarray(saved["evaluation_weights"], dtype=np.float64)
    config = saved["config"]
    output = {"source_batch": config["source"], "target_batch": config["target"],
              "weights": weights.tolist()}
    for domain in ("source", "target"):
        raw, labels = load_batch(data_dir / f"batch{config[domain]}.dat")
        x = torch.from_numpy(transform_with_saved(raw, saved["preprocessing"], recipe, domain))
        probabilities = []
        for batch in x.split(512):
            levels = model.extractor(batch)
            probabilities.append(torch.stack([torch.softmax(logits, dim=1)
                                               for logits in model.logits(levels)], dim=0))
        heads = torch.cat(probabilities, dim=1).numpy()
        ensemble = np.einsum("h,hnd->nd", weights, heads)
        uniform = heads.mean(axis=0)
        predicted = ensemble.argmax(axis=1)
        output[f"{domain}_heads_percent"] = [round(100 * float((p.argmax(axis=1) == labels).mean()), 4)
                                                for p in heads]
        class_counts = np.bincount(labels, minlength=6)
        output[f"{domain}_class_counts"] = class_counts.tolist()
        output[f"{domain}_head_recalls"] = [
            [round(float(((p.argmax(axis=1) == class_id) & (labels == class_id)).sum() / count), 4)
             if count else None for class_id, count in enumerate(class_counts)]
            for p in heads]
        output[f"{domain}_ensemble_percent"] = round(100 * float((predicted == labels).mean()), 4)
        output[f"{domain}_uniform_percent"] = round(100 * float((uniform.argmax(axis=1) == labels).mean()), 4)
        output[f"{domain}_counts"] = np.bincount(predicted, minlength=6).tolist()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suites", type=Path, nargs="+")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for suite in args.suites:
        folders = [suite] if (suite / "model.pt").is_file() else sorted(suite.glob("batch*_to_*"))
        for folder in folders:
            if not (folder / "model.pt").is_file():
                continue
            row = inspect(folder / "model.pt", args.data_dir)
            result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
            if abs(row["target_ensemble_percent"] - 100 * result["metrics"]["target_accuracy"]) > 0.0002:
                raise AssertionError(f"ensemble replay mismatch: {folder}")
            row["suite"] = str(suite)
            row["checkpoint"] = str((folder / "model.pt").resolve())
            rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for row in rows:
        print(f"{row['suite']} {row['source_batch']}->{row['target_batch']} "
              f"heads={row['target_heads_percent']} ensemble={row['target_ensemble_percent']} "
              f"uniform={row['target_uniform_percent']}")


if __name__ == "__main__":
    main()
