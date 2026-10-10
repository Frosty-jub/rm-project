"""Produce a checked long/short comparison only from a complete 17-pair suite."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


LONG = tuple((1, target) for target in range(2, 11))
SHORT = tuple((source, source + 1) for source in range(1, 10))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.suite / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "complete" or summary["completed_count"] != 17:
        raise ValueError("suite is incomplete; refusing to calculate full protocol means")
    with (args.suite / "tasks.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = set(LONG + SHORT)
    by_pair = {(int(row["source"]), int(row["target"])): row for row in rows}
    if len(rows) != 17 or set(by_pair) != expected:
        raise ValueError("CSV does not contain exactly the 17 unique protocol pairs")
    report = []
    for protocol, pairs in (("long-term", LONG), ("short-term", SHORT)):
        selected = [by_pair[pair] for pair in pairs]
        target_mean = float(np.mean([float(row["target_accuracy_percent"]) for row in selected]))
        source_mean = float(np.mean([float(row["source_accuracy_percent"]) for row in selected]))
        paper_mean = float(np.mean([float(row["paper_accuracy_percent"]) for row in selected]))
        report.append((protocol, target_mean, source_mean, paper_mean, target_mean - paper_mean))
    lines = ["# Full MPC-CNN suite comparison", "",
             "Final-epoch, seed 42, exploratory modified MPC-CNN selected using target scores. "
             "The 17 distinct trained pairs fill 9 long-term and 9 short-term protocol rows; "
             "1→2 appears in both means.", "",
             "| Protocol | Target macro | Source macro | Paper macro | Gap |",
             "|---|---:|---:|---:|---:|"]
    for name, target, source, paper, gap in report:
        lines.append(f"| {name} | {target:.2f}% | {source:.2f}% | {paper:.2f}% | {gap:+.2f} pp |")
    lines.extend(("", "| Pair | Protocol | Target | Source | Paper | Gap |",
                  "|---|---|---:|---:|---:|---:|"))
    for pair in LONG + tuple(pair for pair in SHORT if pair not in LONG):
        row = by_pair[pair]
        membership = "long + short" if pair in LONG and pair in SHORT else "long" if pair in LONG else "short"
        lines.append(f"| {pair[0]}→{pair[1]} | {membership} | "
                     f"{float(row['target_accuracy_percent']):.2f}% | "
                     f"{float(row['source_accuracy_percent']):.2f}% | "
                     f"{float(row['paper_accuracy_percent']):.2f}% | "
                     f"{float(row['gap_percentage_points']):+.2f} pp |")
    lines.extend(("", "See `tasks.csv`, per-pair `result.json` and `model.pt`, and `audit.json` "
                  "for the underlying artifacts. The recipe is intentionally modified from the paper; "
                  "one seed and target-informed recipe selection do not establish paper reproduction.", ""))
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
