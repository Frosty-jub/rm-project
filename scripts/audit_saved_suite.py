"""Replay saved MPC-CNN predictions and validate suite artifacts against raw batches."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mpc_cnn.data import load_batch
from mpc_cnn.revised import predict_checkpoint


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    standalone = (args.suite / "result.json").is_file()
    if standalone:
        saved = json.loads((args.suite / "result.json").read_text(encoding="utf-8"))
        rows = [{"source": saved["config"]["source"], "target": saved["config"]["target"],
                 "target_accuracy_percent": 100 * saved["metrics"]["target_accuracy"]}]
        expected_count = 1
    else:
        summary = json.loads((args.suite / "summary.json").read_text(encoding="utf-8"))
        with (args.suite / "tasks.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != summary["completed_count"]:
            raise AssertionError("CSV row count does not match suite summary")
        expected_count = summary["requested_pair_count"]
    audited = []
    for row in rows:
        source, target = int(row["source"]), int(row["target"])
        folder = args.suite if standalone else args.suite / f"batch{source}_to_{target}"
        result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        checkpoint = folder / "model.pt"
        if digest(checkpoint) != result["model_sha256"]:
            raise AssertionError(f"checkpoint hash mismatch: {folder}")
        if result["selection"] != "final_epoch" or result["training_target_labels_used"]:
            raise AssertionError(f"unexpected training or selection policy: {folder}")
        for batch in (source, target):
            raw_path = args.data_dir / f"batch{batch}.dat"
            if digest(raw_path) != result["provenance"]["data_files"][str(batch)]["sha256"]:
                raise AssertionError(f"data hash mismatch: {raw_path}")
        metrics = {}
        for domain, batch in (("source", source), ("target", target)):
            raw, labels = load_batch(args.data_dir / f"batch{batch}.dat")
            probabilities = predict_checkpoint(checkpoint, raw, domain=domain, device="cpu")
            if probabilities.shape != (len(labels), 6) or not np.isfinite(probabilities).all():
                raise AssertionError(f"invalid probabilities: {folder} {domain}")
            expected_sum = sum(result["metrics"]["weights"])
            if result["config"]["normalize_weights"] and not np.isclose(expected_sum, 1, atol=1e-5):
                raise AssertionError(f"normalized weights do not sum to one: {folder}")
            if not np.allclose(probabilities.sum(axis=1), expected_sum, atol=1e-5):
                raise AssertionError(f"score sum does not match saved weights: {folder} {domain}")
            predicted = probabilities.argmax(axis=1)
            accuracy = float((predicted == labels).mean())
            expected = result["metrics"][f"{domain}_accuracy"]
            if abs(accuracy - expected) > 1 / len(labels) / 10:
                raise AssertionError(f"accuracy mismatch: {folder} {domain}: {accuracy} != {expected}")
            counts = np.bincount(predicted, minlength=6).tolist()
            if counts != result["metrics"][f"{domain}_predicted_counts"]:
                raise AssertionError(f"prediction counts mismatch: {folder} {domain}")
            metrics[f"{domain}_accuracy_percent"] = 100 * accuracy
        if abs(metrics["target_accuracy_percent"] - float(row["target_accuracy_percent"])) > 0.001:
            raise AssertionError(f"CSV accuracy mismatch: {folder}")
        with (folder / "history.jsonl").open(encoding="utf-8") as stream:
            last = json.loads(list(stream)[-1])
        if last["stage"] != "formal" or last["epoch"] != result["config"]["epochs"]:
            raise AssertionError(f"history did not reach final epoch: {folder}")
        audited.append({"source": source, "target": target, **metrics,
                        "checkpoint_sha256": result["model_sha256"]})
    report = {"suite": str(args.suite.resolve()), "audited_count": len(audited),
              "expected_count": expected_count,
              "complete": len(audited) == expected_count, "tasks": audited}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("audited_count", "expected_count", "complete")}, indent=2))


if __name__ == "__main__":
    main()
