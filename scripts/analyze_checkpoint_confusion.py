"""Inspect final-epoch target class errors from a saved MPC-CNN checkpoint."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mpc_cnn.data import load_batch
from mpc_cnn.revised import predict_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="task folder containing model.pt and result.json")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.loads((args.run / "result.json").read_text(encoding="utf-8"))
    target = result["config"]["target"]
    raw, labels = load_batch(args.data_dir / f"batch{target}.dat")
    predictions = predict_checkpoint(args.run / "model.pt", raw, domain="target").argmax(axis=1)
    confusion = np.zeros((6, 6), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    accuracy = float(np.trace(confusion) / confusion.sum())
    if abs(accuracy - result["metrics"]["target_accuracy"]) > 1 / len(labels) / 10:
        raise AssertionError("checkpoint predictions do not match saved target accuracy")
    counts = confusion.sum(axis=1)
    report = {"source": result["config"]["source"], "target": target,
              "class_index_policy": "zero-based indices 0..5; add 1 for UCI file labels",
              "accuracy_percent": 100 * accuracy, "target_class_counts": counts.tolist(),
              "predicted_class_counts": confusion.sum(axis=0).tolist(),
              "recall_by_class_index_0_to_5": np.divide(confusion.diagonal(), counts,
                                                          out=np.zeros(6, dtype=float), where=counts != 0).tolist(),
              "confusion_true_rows_predicted_columns": confusion.tolist()}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
