"""LIBSVM-like Gas Sensor Array Drift loader and source-only scaling.

Raw label IDs are kept consistent across batches. Do not infer gas names from
the column order of paper Table 2: that table uses a different ordering.
"""

from pathlib import Path

import numpy as np


EXPECTED_COUNTS = {1: 445, 2: 1244, 3: 1586, 4: 161, 5: 197,
                   6: 2300, 7: 3613, 8: 294, 9: 470, 10: 3600}
# Inferred by matching raw-ID counts against each gas column in paper Table 2
# for all ten batches. The UCI concentration-extension page uses another order.
RAW_ID_TO_PAPER_GAS = {1: "Acetone", 2: "Acetaldehyde", 3: "Ethanol",
                       4: "Ethylene", 5: "Ammonia", 6: "Toluene"}


def load_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return features as (N, 1, 16, 8) and zero-based gas labels."""
    features, labels = [], []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            if not fields:
                continue
            label = int(fields[0])
            if label not in range(1, 7):
                raise ValueError(f"{path}:{line_number}: invalid class {label}")
            row = np.zeros(128, dtype=np.float32)
            seen = set()
            for field in fields[1:]:
                key, value = field.split(":", 1)
                index = int(key)
                if index not in range(1, 129) or index in seen:
                    raise ValueError(f"{path}:{line_number}: invalid/duplicate feature {index}")
                seen.add(index)
                row[index - 1] = float(value)
            if len(seen) != 128 or not np.isfinite(row).all():
                raise ValueError(f"{path}:{line_number}: expected 128 finite features")
            features.append(row.reshape(1, 16, 8))
            labels.append(label - 1)
    x = np.stack(features)
    y = np.asarray(labels, dtype=np.int64)
    batch_id = int(Path(path).stem.removeprefix("batch"))
    expected = EXPECTED_COUNTS.get(batch_id)
    if expected is not None and len(y) != expected:
        raise ValueError(f"{path}: got {len(y)} samples, expected {expected}")
    return x, y


def scale_from_source(source: np.ndarray, target: np.ndarray):
    """Fit each of 128 z-score statistics on labeled source data only."""
    source_flat = source.reshape(len(source), -1)
    target_flat = target.reshape(len(target), -1)
    mean = source_flat.mean(axis=0)
    std = source_flat.std(axis=0)
    std[std < 1e-8] = 1.0
    xs = ((source_flat - mean) / std).reshape(-1, 1, 16, 8).astype(np.float32)
    xt = ((target_flat - mean) / std).reshape(-1, 1, 16, 8).astype(np.float32)
    if not np.isfinite(xs).all() or not np.isfinite(xt).all():
        raise ValueError("non-finite value after source-only scaling")
    return xs, xt, mean, std


def scale_pair(source: np.ndarray, target: np.ndarray, mode: str):
    """Scale a pair using source or pooled unlabeled features."""
    if mode == "source":
        xs, xt, mean, std = scale_from_source(source, target)
        return xs, xt, mean, std
    if mode != "pooled":
        raise ValueError(f"unknown scaling mode: {mode}")
    source_flat = source.reshape(len(source), -1)
    target_flat = target.reshape(len(target), -1)
    pooled = np.concatenate((source_flat, target_flat), axis=0)
    mean = pooled.mean(axis=0)
    std = pooled.std(axis=0)
    std[std < 1e-8] = 1.0
    xs = ((source_flat - mean) / std).reshape(-1, 1, 16, 8).astype(np.float32)
    xt = ((target_flat - mean) / std).reshape(-1, 1, 16, 8).astype(np.float32)
    if not np.isfinite(xs).all() or not np.isfinite(xt).all():
        raise ValueError("non-finite value after pooled scaling")
    return xs, xt, mean, std
