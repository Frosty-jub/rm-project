"""Combine completed long and short Dataset A protocols into one comparison."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long-dir", type=Path, required=True)
    parser.add_argument("--short-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    long_rows = [row for row in read_rows(args.long_dir / "results.csv") if row["protocol"] == "long"]
    candidates = [row for directory in args.short_dir
                  for row in read_rows(directory / "results.csv") if row["protocol"] == "short"]
    unique_short = {}
    for row in candidates:
        pair = (int(row["source"]), int(row["target"]))
        if pair in unique_short and abs(float(unique_short[pair]["accuracy"]) - float(row["accuracy"])) > 1e-8:
            raise ValueError(f"duplicate short task has different accuracy: {pair}")
        unique_short[pair] = row
    short_rows = [unique_short[(source, source + 1)] for source in range(1, 10)
                  if (source, source + 1) in unique_short]
    if len(long_rows) != 9 or len(short_rows) != 9:
        raise ValueError(f"expected 9 tasks per protocol, got {len(long_rows)} long and {len(short_rows)} short")
    if {(int(row["source"]), int(row["target"])) for row in long_rows} != {(1, t) for t in range(2, 11)}:
        raise ValueError("long-term task set is incomplete")
    if {(int(row["source"]), int(row["target"])) for row in short_rows} != {(s, s + 1) for s in range(1, 10)}:
        raise ValueError("short-term task set is incomplete")
    rows = long_rows + short_rows
    if len({(row["seed"], row["scaling"], row["epochs"], row["pretrain_epochs"],
             row.get("architecture", "legacy")) for row in rows}) != 1:
        raise ValueError("runs use inconsistent seed, scaling, architecture, or training budget")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for protocol, selected in (("long", long_rows), ("short", short_rows)):
        actual = np.mean([float(row["accuracy"]) for row in selected])
        paper = np.mean([float(row["paper_accuracy"]) for row in selected])
        summary[protocol] = {"tasks": 9, "mean_accuracy": float(actual),
                             "paper_mean_accuracy": float(paper),
                             "gap_percentage_points": float(100 * (actual - paper))}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for protocol, item in summary.items():
        print(f"{protocol}: {100*item['mean_accuracy']:.2f}% vs paper "
              f"{100*item['paper_mean_accuracy']:.2f}% "
              f"({item['gap_percentage_points']:+.2f} pp)")


if __name__ == "__main__":
    main()
