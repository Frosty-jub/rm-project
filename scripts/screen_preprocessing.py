"""Exploratory preprocessing diagnosis; this is not an MPC-CNN reproduction.

Only source labels are passed to SVC.fit. Target labels are used to report
accuracy and rank preprocessing candidates, so this screen is target-informed
and its selected results are not an untouched test benchmark.
"""

import os

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import csv
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mpc_cnn.data import load_batch


PREPROCESSORS = (
    "source_zscore", "pooled_zscore", "domain_zscore", "row_l2",
    "source_zscore_row_l2", "signedlog_source_zscore",
    "signedlog_pooled_zscore", "signedlog_domain_zscore", "source_zscore_coral",
)


def zscore_fit(x):
    mean, std = x.mean(axis=0), x.std(axis=0)
    return mean, np.where(std < 1e-8, 1.0, std)


def row_l2(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def matrix_power_psd(covariance, exponent):
    values, vectors = np.linalg.eigh(covariance)
    return (vectors * np.maximum(values, 1e-8) ** exponent) @ vectors.T


def transform_pair(source, target, mode):
    """Uses feature matrices only, with no source or target labels."""
    source, target = source.copy(), target.copy()
    if mode.startswith("signedlog_"):
        source = np.sign(source) * np.log1p(np.abs(source))
        target = np.sign(target) * np.log1p(np.abs(target))
        mode = mode.removeprefix("signedlog_")
    if mode == "row_l2":
        return row_l2(source), row_l2(target)
    if mode.startswith("source_zscore"):
        mean, std = zscore_fit(source)
        source, target = (source - mean) / std, (target - mean) / std
    elif mode == "pooled_zscore":
        mean, std = zscore_fit(np.concatenate((source, target)))
        source, target = (source - mean) / std, (target - mean) / std
    elif mode == "domain_zscore":
        source_mean, source_std = zscore_fit(source)
        target_mean, target_std = zscore_fit(target)
        source, target = (source - source_mean) / source_std, (target - target_mean) / target_std
    else:
        raise ValueError(mode)
    if mode.endswith("_row_l2"):
        source, target = row_l2(source), row_l2(target)
    elif mode.endswith("_coral"):
        # Fixed 10% diagonal shrinkage avoids fitting an unstable inverse to a
        # source batch with only 161 observations and 128 features.
        cs, ct = np.cov(source, rowvar=False), np.cov(target, rowvar=False)
        identity = np.eye(source.shape[1])
        cs = 0.9 * cs + 0.1 * np.trace(cs) / len(cs) * identity
        ct = 0.9 * ct + 0.1 * np.trace(ct) / len(ct) * identity
        source = source @ matrix_power_psd(cs, -0.5) @ matrix_power_psd(ct, 0.5) + target.mean(axis=0)
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Non-finite preprocessing output")
    return source, target


def run(data_root, output_root, screen_only=False):
    output_root.mkdir(parents=True, exist_ok=True)
    data = {i: load_batch(data_root / f"batch{i}.dat") for i in range(1, 11)}
    data = {i: (x.reshape(len(x), -1).astype(np.float64), y) for i, (x, y) in data.items()}
    rows, cache = [], {}

    def evaluate(source_id, target_id, preprocessing, phase):
        key = (source_id, target_id, preprocessing)
        if key not in cache:
            xs, ys = data[source_id]
            xt, yt = data[target_id]
            xs, xt = transform_pair(xs, xt, preprocessing)
            start = time.perf_counter()
            model = SVC(C=10.0, kernel="rbf", gamma="scale", cache_size=256, random_state=42)
            model.fit(xs, ys)
            source_accuracy = 100 * accuracy_score(ys, model.predict(xs))
            target_predictions = model.predict(xt)
            target_accuracy = 100 * accuracy_score(yt, target_predictions)
            cache[key] = {
                "source": source_id, "target": target_id, "preprocessing": preprocessing,
                "model": "RBF_SVC_C10_gamma_scale", "source_count": len(ys), "target_count": len(yt),
                "source_accuracy": source_accuracy, "target_accuracy": target_accuracy,
                "target_predicted_classes": len(np.unique(target_predictions)),
                "seconds": time.perf_counter() - start,
            }
        row = {"phase": phase, **cache[key]}
        rows.append(row)
        with (output_root / "results.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(row), flush=True)
        return row

    representatives = [(1, 2), (1, 6), (4, 5)]
    for preprocessing in PREPROCESSORS:
        for source_id, target_id in representatives:
            evaluate(source_id, target_id, preprocessing, "representative_screen")
    ranking = sorted(({
        "preprocessing": preprocessing,
        "representative_mean_target_accuracy": float(np.mean([
            cache[s, t, preprocessing]["target_accuracy"] for s, t in representatives
        ])),
    } for preprocessing in PREPROCESSORS), key=lambda entry: -entry["representative_mean_target_accuracy"])
    selected = [entry["preprocessing"] for entry in ranking[:3]]
    print(json.dumps({"ranking": ranking, "selected": selected}), flush=True)
    if not screen_only:
        for preprocessing in selected:
            for target_id in range(2, 11):
                evaluate(1, target_id, preprocessing, "long_exploratory")
            for source_id in range(1, 10):
                evaluate(source_id, source_id + 1, preprocessing, "short_exploratory")
    averages = {}
    for preprocessing in selected:
        averages[preprocessing] = {}
        for phase in ("long_exploratory", "short_exploratory"):
            values = [r["target_accuracy"] for r in rows if r["preprocessing"] == preprocessing and r["phase"] == phase]
            if values:
                averages[preprocessing][phase] = float(np.mean(values))
    metadata = {
        "purpose": "Preprocessing diagnosis for MPC-CNN; SVC is a diagnostic substitute, not a reproduced MPC-CNN.",
        "selection_disclosure": "TARGET-INFORMED EXPLORATORY: target labels score/rank candidates. No target labels used in transforms or model.fit. Selected scores are not untouched benchmark results.",
        "python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__,
        "model": {"type": "SVC", "C": 10.0, "kernel": "rbf", "gamma": "scale", "random_state": 42},
        "representatives": representatives, "ranking": ranking, "selected": selected, "averages": averages,
    }
    (output_root / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"averages": averages}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--output-root", type=Path, default=Path("runs/preprocessing_screen"))
    parser.add_argument("--screen-only", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.data_root, args.output_root, args.screen_only)
