"""Run a fixed, explicitly exploratory MPC-CNN task suite.

Run from the repository root, e.g.:
  python scripts/run_revised_suite.py --recipe recipes/revised.json --tasks representative --output runs/revised_probe
"""

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mpc_cnn.revised import Recipe, run, task_parameters


LONG = tuple((1, target) for target in range(2, 11))
SHORT = tuple((source, source + 1) for source in range(1, 10))
REPRESENTATIVE = ((1, 2), (1, 6), (4, 5))
STUDY_LABELS = ("exploratory modified MPC-CNN; target-informed recipe selection",
                "exploratory modified MPC-CNN; precommitted recipe")


def select_pairs(tasks=None, pairs=None):
    if (tasks is None) == (pairs is None):
        raise ValueError("provide exactly one of tasks or pairs")
    if pairs is not None:
        try:
            selected = tuple(tuple(int(part) for part in item.split(":")) for item in pairs.split(","))
        except ValueError as exc:
            raise ValueError("pairs must look like 1:2,4:5") from exc
    else:
        selected = {"representative": REPRESENTATIVE, "long": LONG,
                    "short": SHORT, "all": LONG + SHORT}[tasks]
    for pair in selected:
        if len(pair) != 2:
            raise ValueError("pairs must look like 1:2,4:5")
        task_parameters(*pair)
    return tuple(dict.fromkeys(selected))


def summarize(rows, pairs, recipe, study_label):
    summary = {"study_label": study_label, "selection": "final_epoch_no_target_checkpoint_selection",
               "recipe": asdict(recipe), "pairs": [list(pair) for pair in pairs],
               "requested_pair_count": len(pairs), "completed_count": len(rows),
               "status": "complete" if len(rows) == len(pairs) else "incomplete",
               "unique_task_count": len(pairs), "protocols": {}}
    for name, protocol in (("long", LONG), ("short", SHORT)):
        selected = [row for row in rows if (row["source"], row["target"]) in protocol]
        summary["protocols"][name] = {
            "task_count": len(selected), "full_protocol_task_count": len(protocol),
            "macro_target_accuracy_percent": float(np.mean([row["target_accuracy_percent"] for row in selected]))
            if selected else None,
            "macro_paper_accuracy_percent": float(np.mean([row["paper_accuracy_percent"] for row in selected]))
            if selected else None}
    return summary


def write_progress(output, rows, pairs, recipe, study_label):
    summary = summarize(rows, pairs, recipe, study_label)
    fields = ("source", "target", "long_protocol", "short_protocol", "target_accuracy_percent",
              "source_accuracy_percent", "paper_accuracy_percent", "gap_percentage_points", "result")
    csv_path = output / "tasks.csv"
    csv_temp = output / "tasks.csv.tmp"
    with csv_temp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    csv_temp.replace(csv_path)
    summary_temp = output / "summary.json.tmp"
    summary_temp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_temp.replace(output / "summary.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    tasks = parser.add_mutually_exclusive_group(required=True)
    tasks.add_argument("--tasks", choices=("representative", "long", "short", "all"))
    tasks.add_argument("--pairs", help="comma-separated source:target task list")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--study-label", choices=STUDY_LABELS, default=STUDY_LABELS[0])
    args = parser.parse_args()
    recipe = Recipe(**json.loads(args.recipe.read_text(encoding="utf-8")))
    recipe.validate()
    pairs = select_pairs(args.tasks, args.pairs)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "summary.json"
    expected = summarize([], pairs, recipe, args.study_label)
    if summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        for key in ("recipe", "pairs", "study_label"):
            if previous.get(key) != expected[key]:
                raise ValueError("summary.json belongs to another suite")
    rows = []
    write_progress(args.output, rows, pairs, recipe, args.study_label)
    for source, target in pairs:
        result = run(recipe, source, target, args.output / f"batch{source}_to_{target}",
                     args.data_dir, device)
        rows.append({"source": source, "target": target,
                     "long_protocol": (source, target) in LONG,
                     "short_protocol": (source, target) in SHORT,
                     "target_accuracy_percent": 100 * result["metrics"]["target_accuracy"],
                     "source_accuracy_percent": 100 * result["metrics"]["source_accuracy"],
                     "paper_accuracy_percent": result["paper_accuracy_percent"],
                     "gap_percentage_points": result["gap_percentage_points"],
                     "result": str((args.output / f"batch{source}_to_{target}" / "result.json").resolve())})
        summary = write_progress(args.output, rows, pairs, recipe, args.study_label)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
