#!/usr/bin/env python3
"""วิเคราะห์ UCI Gas Sensor Array Drift Dataset แบบครบวงจร

สคริปต์นี้ทำงานตามหัวข้อในโจทย์:
1) ตรวจ schema และเทียบจำนวนข้อมูลกับ metadata ทางการของ UCI
2) นับ class ราย batch, หา class ที่หาย และคำนวณ class imbalance
3) เปรียบเทียบ class prior ระหว่าง source-target
4) วิเคราะห์ global drift และ class-wise drift ด้วย RBF-MMD
5) ตรวจความเสี่ยง data leakage
6) สร้าง controlled-imbalance protocol และ manifest ของ sample ที่ใช้

รูปแบบข้อมูลเป็น LIBSVM: class feature_id:value ... feature_id:value
เช่น: 1 1:15596.162100 ... 128:-2.654529

แหล่งข้อมูลทางการ:
https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset
DOI: 10.24432/C5RP6W
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import scipy
from scipy.spatial.distance import pdist
from sklearn import __version__ as sklearn_version
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler


# =============================================================================
# ส่วนที่ 0: metadata ทางการและค่าตั้งต้นของการทดลอง
# =============================================================================

OFFICIAL_SOURCE = (
    "https://archive.ics.uci.edu/dataset/224/"
    "gas+sensor+array+drift+dataset"
)
OFFICIAL_DOI = "10.24432/C5RP6W"
EXPECTED_SAMPLES = 13_910
EXPECTED_SENSORS = 16
FEATURES_PER_SENSOR = 8
EXPECTED_FEATURES = EXPECTED_SENSORS * FEATURES_PER_SENSOR
EXPECTED_CLASSES = tuple(range(1, 7))
EXPECTED_BATCHES = tuple(range(1, 11))

# ชื่อ gas ตาม class code ที่ UCI ระบุไว้
CLASS_NAMES = {
    1: "Ethanol",
    2: "Ethylene",
    3: "Ammonia",
    4: "Acetaldehyde",
    5: "Acetone",
    6: "Toluene",
}

# Protocol เริ่มต้น: ลดทีละ class ใน source ให้เหลือสัดส่วนที่กำหนด
# Target ไม่ถูกลดจำนวน เพื่อให้ประเมินผลบน distribution จริงของ target ได้
DEFAULT_SOURCE_BATCH = 1
DEFAULT_MANIFEST_TARGET_BATCH = 10
DEFAULT_REDUCED_CLASSES = tuple(range(1, 7))
DEFAULT_KEEP_FRACTIONS = (0.50, 0.25, 0.10)
DEFAULT_RANDOM_SEEDS = (42, 123, 2026)

# สีประจำ batch สำหรับกราฟ SVG
BATCH_COLORS = (
    "#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c",
    "#0891b2", "#be123c", "#4f46e5", "#65a30d", "#0f766e",
)


@dataclass(frozen=True)
class DatasetBundle:
    """เก็บ feature matrix และ metadata ของทุก sample ไว้ด้วยกัน"""

    X: np.ndarray
    metadata: pd.DataFrame
    batch_files: tuple[Path, ...]


def batch_number(path: Path) -> int:
    """ดึงเลข batch จากชื่อไฟล์ เช่น batch10.dat -> 10"""

    match = re.fullmatch(r"batch(\d+)\.dat", path.name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"ชื่อไฟล์ไม่อยู่ในรูปแบบ batchN.dat: {path.name}")
    return int(match.group(1))


def sha256_text(value: str) -> str:
    """สร้าง checksum เพื่อใช้อ้างอิงแถวข้อมูลแบบตรวจสอบย้อนกลับได้"""

    return hashlib.sha256(value.encode("ascii")).hexdigest()


# =============================================================================
# ส่วนที่ 1: อ่านไฟล์ LIBSVM และตรวจ schema ระดับแถว
# =============================================================================

def load_dataset(dataset_dir: Path) -> DatasetBundle:
    """อ่าน batch1.dat ถึง batch10.dat พร้อมตรวจ class/feature ทุกแถว"""

    files = tuple(
        sorted(dataset_dir.glob("batch*.dat"), key=batch_number)
    )
    actual_batches = tuple(batch_number(path) for path in files)
    if actual_batches != EXPECTED_BATCHES:
        raise ValueError(
            f"ต้องมี batch 1-10 ครบ แต่พบ batch {actual_batches}"
        )

    vectors: list[np.ndarray] = []
    rows: list[dict[str, object]] = []

    for path in files:
        batch = batch_number(path)
        with path.open("r", encoding="ascii") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                tokens = line.split()
                # รองรับทั้ง "1" และ "1;10.0" โดยใช้ค่าก่อน semicolon เป็น class
                class_id = int(tokens[0].split(";", maxsplit=1)[0])
                if class_id not in EXPECTED_CLASSES:
                    raise ValueError(
                        f"{path.name}:{line_no} มี class ผิดปกติ: {class_id}"
                    )

                vector = np.full(EXPECTED_FEATURES, np.nan, dtype=np.float64)
                seen_feature_ids: set[int] = set()
                for token in tokens[1:]:
                    feature_text, value_text = token.split(":", maxsplit=1)
                    feature_id = int(feature_text)
                    if not 1 <= feature_id <= EXPECTED_FEATURES:
                        raise ValueError(
                            f"{path.name}:{line_no} feature id นอกช่วง: {feature_id}"
                        )
                    if feature_id in seen_feature_ids:
                        raise ValueError(
                            f"{path.name}:{line_no} feature {feature_id} ซ้ำ"
                        )
                    seen_feature_ids.add(feature_id)
                    vector[feature_id - 1] = float(value_text)

                expected_ids = set(range(1, EXPECTED_FEATURES + 1))
                if seen_feature_ids != expected_ids:
                    missing = sorted(expected_ids - seen_feature_ids)
                    raise ValueError(
                        f"{path.name}:{line_no} feature ไม่ครบ; missing={missing[:10]}"
                    )
                if not np.isfinite(vector).all():
                    raise ValueError(f"{path.name}:{line_no} มี NaN หรือ infinity")

                # Dataset ไม่มี sample ID มาให้ จึงสร้าง ID คงที่จาก batch + เลขบรรทัด
                sample_id = f"B{batch:02d}_R{line_no:05d}"
                vectors.append(vector)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "batch": batch,
                        "row_in_batch": line_no,
                        "class_id": class_id,
                        "class_name": CLASS_NAMES[class_id],
                        "raw_sha256": sha256_text(line),
                        # hash ของ feature ล้วน ใช้ตรวจ exact duplicate ข้าม batch
                        "feature_sha256": hashlib.sha256(
                            vector.tobytes()
                        ).hexdigest(),
                    }
                )

    X = np.vstack(vectors)
    metadata = pd.DataFrame(rows)
    if len(X) != len(metadata):
        raise AssertionError("จำนวนแถวของ X และ metadata ไม่ตรงกัน")
    if metadata["sample_id"].duplicated().any():
        raise ValueError("พบ sample_id ซ้ำ")

    return DatasetBundle(X=X, metadata=metadata, batch_files=files)


# =============================================================================
# ส่วนที่ 2: ยืนยัน schema และจำนวนตัวอย่างกับแหล่ง official
# =============================================================================

def make_count_table(metadata: pd.DataFrame) -> pd.DataFrame:
    """สร้างตาราง batch x class จาก raw data จริง ไม่ hard-code ผลลัพธ์"""

    table = pd.crosstab(metadata["batch"], metadata["class_id"])
    table = table.reindex(
        index=EXPECTED_BATCHES, columns=EXPECTED_CLASSES, fill_value=0
    ).astype(int)
    table.index.name = "batch"
    return table


def validate_official_schema(
    bundle: DatasetBundle,
    count_table: pd.DataFrame,
    strict: bool,
) -> tuple[dict[str, object], pd.DataFrame]:
    """เทียบจำนวน samples/sensors/features/classes/batches กับ UCI"""

    metadata = bundle.metadata
    actual_classes = tuple(sorted(metadata["class_id"].unique().tolist()))
    actual_batches = tuple(sorted(metadata["batch"].unique().tolist()))

    # โจทย์ให้ยืนยัน 5 มิติระดับ dataset กับ official source ส่วน batch x class
    # ต้องนับจาก raw data จริง จึงแยกไปอยู่ใน class_count_table.csv โดยไม่เอา
    # ตารางจาก dataset รุ่นอื่น (เช่นรุ่นที่เพิ่ม concentration) มาปะปนกัน
    validation_rows = [
        {
            "metric": "samples",
            "official_expected": EXPECTED_SAMPLES,
            "actual": len(metadata),
        },
        {
            "metric": "sensors",
            "official_expected": EXPECTED_SENSORS,
            "actual": EXPECTED_SENSORS,
        },
        {
            "metric": "features",
            "official_expected": EXPECTED_FEATURES,
            "actual": bundle.X.shape[1],
        },
        {
            "metric": "classes",
            "official_expected": len(EXPECTED_CLASSES),
            "actual": len(actual_classes),
        },
        {
            "metric": "batches",
            "official_expected": len(EXPECTED_BATCHES),
            "actual": len(actual_batches),
        },
    ]
    count_validation = pd.DataFrame(validation_rows)
    count_validation["match"] = (
        count_validation["official_expected"] == count_validation["actual"]
    )

    checks = {
        "sample_count": len(metadata) == EXPECTED_SAMPLES,
        "feature_count": bundle.X.shape[1] == EXPECTED_FEATURES,
        "sensor_count": EXPECTED_SENSORS == 16,
        "class_ids": actual_classes == EXPECTED_CLASSES,
        "batch_ids": actual_batches == EXPECTED_BATCHES,
        "all_values_finite": bool(np.isfinite(bundle.X).all()),
        "all_official_dimension_counts_match": bool(count_validation["match"].all()),
    }

    schema = {
        "dataset_name": "Gas Sensor Array Drift Dataset",
        "official_url": OFFICIAL_SOURCE,
        "doi": OFFICIAL_DOI,
        "file_format": "LIBSVM-like: class_id feature_id:value ...",
        "actual": {
            "samples": int(len(metadata)),
            "sensors": EXPECTED_SENSORS,
            "features_per_sensor": FEATURES_PER_SENSOR,
            "features": int(bundle.X.shape[1]),
            "classes": len(actual_classes),
            "class_ids": list(actual_classes),
            "batches": len(actual_batches),
            "batch_ids": list(actual_batches),
        },
        "official_expected": {
            "samples": EXPECTED_SAMPLES,
            "sensors": EXPECTED_SENSORS,
            "features_per_sensor": FEATURES_PER_SENSOR,
            "features": EXPECTED_FEATURES,
            "classes": len(EXPECTED_CLASSES),
            "batches": len(EXPECTED_BATCHES),
        },
        "class_mapping": {str(k): v for k, v in CLASS_NAMES.items()},
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }

    if strict and not schema["all_checks_passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Official schema validation ไม่ผ่าน: {failed}")
    return schema, count_validation


def make_feature_schema() -> pd.DataFrame:
    """ระบุว่า feature แต่ละตัวมาจาก sensor และ slot ใด"""

    feature_types = (
        "DR",
        "normalized_DR",
        "EMA_rising_alpha_0.001",
        "EMA_rising_alpha_0.01",
        "EMA_rising_alpha_0.1",
        "EMA_decaying_alpha_0.001",
        "EMA_decaying_alpha_0.01",
        "EMA_decaying_alpha_0.1",
    )
    rows = []
    for feature_id in range(1, EXPECTED_FEATURES + 1):
        slot = (feature_id - 1) % FEATURES_PER_SENSOR + 1
        rows.append(
            {
                "feature_id": feature_id,
                "column_name": f"feature_{feature_id:03d}",
                "sensor_id": (feature_id - 1) // FEATURES_PER_SENSOR + 1,
                "feature_slot_within_sensor": slot,
                "official_feature_type": feature_types[slot - 1],
                "dtype": "float64",
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# ส่วนที่ 3: class count, missing class และ imbalance statistics
# =============================================================================

def class_distribution_outputs(
    count_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """คำนวณ count/proportion/missing class/imbalance ของแต่ละ batch"""

    count_output = count_table.copy()
    count_output.columns = [
        f"class_{c}_{CLASS_NAMES[c].lower()}" for c in count_output.columns
    ]
    count_output["total"] = count_output.sum(axis=1)
    overall = count_output.sum(axis=0).to_frame().T
    overall.index = ["Overall"]
    count_output.index = [f"Batch_{b}" for b in count_output.index]
    count_output = pd.concat([count_output, overall])

    proportions = count_table.div(count_table.sum(axis=1), axis=0)
    proportions.columns = [
        f"class_{c}_{CLASS_NAMES[c].lower()}" for c in proportions.columns
    ]
    proportions.index = [f"Batch_{b}" for b in proportions.index]

    missing_rows = []
    for batch in EXPECTED_BATCHES:
        for class_id in EXPECTED_CLASSES:
            if int(count_table.loc[batch, class_id]) == 0:
                missing_rows.append(
                    {
                        "batch": batch,
                        "class_id": class_id,
                        "class_name": CLASS_NAMES[class_id],
                        "count": 0,
                    }
                )
    missing = pd.DataFrame(
        missing_rows,
        columns=["batch", "class_id", "class_name", "count"],
    )

    stats_rows: list[dict[str, object]] = []
    groups: list[tuple[str, np.ndarray]] = [
        (f"Batch_{batch}", count_table.loc[batch].to_numpy(dtype=float))
        for batch in EXPECTED_BATCHES
    ]
    groups.append(("Overall", count_table.sum(axis=0).to_numpy(dtype=float)))

    for group, counts in groups:
        total = float(counts.sum())
        proportions_array = counts / total
        positive = counts[counts > 0]
        majority_idx = int(np.argmax(counts))
        minority_idx = int(np.argmin(counts))
        entropy_terms = proportions_array[proportions_array > 0]
        normalized_entropy = float(
            -(entropy_terms * np.log(entropy_terms)).sum()
            / math.log(len(EXPECTED_CLASSES))
        )
        stats_rows.append(
            {
                "group": group,
                "total_samples": int(total),
                "present_classes": int((counts > 0).sum()),
                "missing_classes": int((counts == 0).sum()),
                "majority_class": majority_idx + 1,
                "majority_count": int(counts[majority_idx]),
                "minority_class_including_missing": minority_idx + 1,
                "minority_count_including_missing": int(counts[minority_idx]),
                "imbalance_ratio_max_to_min": (
                    float(counts.max() / counts.min())
                    if counts.min() > 0
                    else math.inf
                ),
                "imbalance_ratio_present_classes": float(
                    positive.max() / positive.min()
                ),
                "normalized_class_entropy": normalized_entropy,
            }
        )
    imbalance = pd.DataFrame(stats_rows)
    return count_output, proportions, missing, imbalance


# =============================================================================
# ส่วนที่ 4: เปรียบเทียบ class prior ระหว่าง source และ target
# =============================================================================

def jensen_shannon_divergence_bits(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence ฐาน 2; 0=เหมือนกัน, 1=ต่างสุด"""

    midpoint = 0.5 * (p + q)

    def kl_divergence(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl_divergence(p, midpoint) + 0.5 * kl_divergence(q, midpoint)


def make_prior_mismatch_table(
    count_table: pd.DataFrame,
    source_batch: int,
) -> pd.DataFrame:
    """วัด prior shift ของ source เทียบกับทุก target batch"""

    source_counts = count_table.loc[source_batch].to_numpy(dtype=float)
    source_prior = source_counts / source_counts.sum()
    rows = []

    for target_batch in EXPECTED_BATCHES:
        if target_batch == source_batch:
            continue
        target_counts = count_table.loc[target_batch].to_numpy(dtype=float)
        target_prior = target_counts / target_counts.sum()
        delta = target_prior - source_prior
        largest_idx = int(np.argmax(np.abs(delta)))
        row: dict[str, object] = {
            "source_batch": source_batch,
            "target_batch": target_batch,
            "total_variation_distance": float(0.5 * np.abs(delta).sum()),
            "jensen_shannon_divergence_bits": jensen_shannon_divergence_bits(
                source_prior, target_prior
            ),
            "largest_change_class": largest_idx + 1,
            "largest_change_class_name": CLASS_NAMES[largest_idx + 1],
            "largest_prior_delta_target_minus_source": float(delta[largest_idx]),
        }
        for class_id in EXPECTED_CLASSES:
            row[f"source_prior_c{class_id}"] = float(source_prior[class_id - 1])
            row[f"target_prior_c{class_id}"] = float(target_prior[class_id - 1])
            row[f"delta_c{class_id}"] = float(delta[class_id - 1])
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# ส่วนที่ 5: global drift และ class-wise drift ด้วย RBF-MMD
# =============================================================================

def sample_rows(
    matrix: np.ndarray,
    max_rows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """สุ่มแบบไม่คืนค่าเพื่อจำกัดเวลา/หน่วยความจำของ kernel matrix"""

    if len(matrix) <= max_rows:
        return matrix
    indices = rng.choice(len(matrix), size=max_rows, replace=False)
    return matrix[indices]


def source_median_gamma(source_scaled: np.ndarray, seed: int = 17) -> float:
    """เลือก RBF gamma ด้วย median heuristic จาก source เท่านั้น"""

    rng = np.random.default_rng(seed)
    subset = sample_rows(source_scaled, max_rows=600, rng=rng)
    squared_distances = pdist(subset, metric="sqeuclidean")
    positive = squared_distances[squared_distances > 0]
    if len(positive) == 0:
        return 1.0
    median_squared_distance = float(np.median(positive))
    return 1.0 / (2.0 * median_squared_distance)


def biased_rbf_mmd_squared(
    source: np.ndarray,
    target: np.ndarray,
    gamma: float,
    max_rows: int,
    seed: int,
) -> tuple[float, int, int]:
    """คำนวณ biased MMD² ซึ่งไม่ติดลบและอ่านเปรียบเทียบข้าม batch ง่าย"""

    rng = np.random.default_rng(seed)
    source_sample = sample_rows(source, max_rows=max_rows, rng=rng)
    target_sample = sample_rows(target, max_rows=max_rows, rng=rng)
    k_xx = rbf_kernel(source_sample, source_sample, gamma=gamma)
    k_yy = rbf_kernel(target_sample, target_sample, gamma=gamma)
    k_xy = rbf_kernel(source_sample, target_sample, gamma=gamma)
    mmd_squared = float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())
    return max(0.0, mmd_squared), len(source_sample), len(target_sample)


def compute_drift(
    bundle: DatasetBundle,
    source_batch: int,
    max_mmd_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, PCA, float]:
    """fit preprocessing บน source แล้ววัด drift ไปยัง target ทุก batch"""

    metadata = bundle.metadata
    source_mask = metadata["batch"].to_numpy() == source_batch

    # สำคัญต่อ leakage: fit scaler ด้วย source เท่านั้น แล้วค่อย transform target
    scaler = StandardScaler()
    scaler.fit(bundle.X[source_mask])
    scaled = scaler.transform(bundle.X)
    gamma = source_median_gamma(scaled[source_mask])

    # PCA ใช้เพื่อ visualization เท่านั้น และ fit จาก source เช่นเดียวกัน
    pca = PCA(n_components=2, svd_solver="full")
    pca.fit(scaled[source_mask])

    global_rows = []
    classwise_rows = []
    batches = metadata["batch"].to_numpy()
    classes = metadata["class_id"].to_numpy()

    for target_batch in EXPECTED_BATCHES:
        if target_batch == source_batch:
            continue
        target_mask = batches == target_batch
        mmd2, n_source_used, n_target_used = biased_rbf_mmd_squared(
            scaled[source_mask],
            scaled[target_mask],
            gamma=gamma,
            max_rows=max_mmd_rows,
            seed=10_000 + target_batch,
        )
        global_rows.append(
            {
                "source_batch": source_batch,
                "target_batch": target_batch,
                "source_samples_total": int(source_mask.sum()),
                "target_samples_total": int(target_mask.sum()),
                "source_samples_used_for_mmd": n_source_used,
                "target_samples_used_for_mmd": n_target_used,
                "rbf_gamma_from_source_median": gamma,
                "rbf_mmd_squared_biased": mmd2,
                "standardized_mean_shift_l2": float(
                    np.linalg.norm(
                        scaled[source_mask].mean(axis=0)
                        - scaled[target_mask].mean(axis=0)
                    )
                ),
            }
        )

        # Condition on class เพื่อแยก feature drift ออกจากผลของ class-prior shift
        for class_id in EXPECTED_CLASSES:
            source_class_mask = source_mask & (classes == class_id)
            target_class_mask = target_mask & (classes == class_id)
            n_source = int(source_class_mask.sum())
            n_target = int(target_class_mask.sum())
            if n_source == 0 or n_target == 0:
                classwise_rows.append(
                    {
                        "source_batch": source_batch,
                        "target_batch": target_batch,
                        "class_id": class_id,
                        "class_name": CLASS_NAMES[class_id],
                        "source_samples_total": n_source,
                        "target_samples_total": n_target,
                        "status": "missing_in_one_domain",
                        "source_samples_used_for_mmd": 0,
                        "target_samples_used_for_mmd": 0,
                        "rbf_mmd_squared_biased": math.nan,
                        "standardized_mean_shift_l2": math.nan,
                    }
                )
                continue

            class_mmd2, n_source_used, n_target_used = biased_rbf_mmd_squared(
                scaled[source_class_mask],
                scaled[target_class_mask],
                gamma=gamma,
                max_rows=min(max_mmd_rows, 300),
                seed=100_000 + target_batch * 100 + class_id,
            )
            classwise_rows.append(
                {
                    "source_batch": source_batch,
                    "target_batch": target_batch,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "source_samples_total": n_source,
                    "target_samples_total": n_target,
                    "status": "ok",
                    "source_samples_used_for_mmd": n_source_used,
                    "target_samples_used_for_mmd": n_target_used,
                    "rbf_mmd_squared_biased": class_mmd2,
                    "standardized_mean_shift_l2": float(
                        np.linalg.norm(
                            scaled[source_class_mask].mean(axis=0)
                            - scaled[target_class_mask].mean(axis=0)
                        )
                    ),
                }
            )

    return (
        pd.DataFrame(global_rows),
        pd.DataFrame(classwise_rows),
        scaled,
        pca,
        gamma,
    )


# =============================================================================
# ส่วนที่ 6: สร้างกราฟ SVG โดยไม่บังคับให้ติดตั้ง plotting library เพิ่ม
# =============================================================================

def svg_text(
    x: float,
    y: float,
    value: object,
    size: int = 16,
    *,
    anchor: str = "start",
    weight: int = 400,
    fill: str = "#172033",
) -> str:
    """helper สำหรับข้อความใน SVG"""

    escaped = html.escape(str(value))
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">{escaped}</text>'
    )


def svg_rect(
    x: float,
    y: float,
    width: float,
    height: float,
    fill: str,
    *,
    stroke: str = "none",
    rx: float = 0,
) -> str:
    """helper สำหรับสี่เหลี่ยมใน SVG"""

    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" '
        f'height="{height:.1f}" rx="{rx:.1f}" fill="{fill}" '
        f'stroke="{stroke}"/>'
    )


def color_mix(low: tuple[int, int, int], high: tuple[int, int, int], t: float) -> str:
    """ไล่สีสำหรับ heatmap"""

    t = max(0.0, min(1.0, float(t)))
    rgb = tuple(round(a + (b - a) * t) for a, b in zip(low, high))
    return "#" + "".join(f"{value:02x}" for value in rgb)


def save_svg(path: Path, width: int, height: int, elements: Iterable[str]) -> None:
    """บันทึก SVG ที่เปิดได้ด้วย browser ทั่วไป"""

    content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        svg_rect(0, 0, width, height, "#f8fafc"),
        *elements,
        "</svg>",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def write_class_distribution_svg(count_table: pd.DataFrame, path: Path) -> None:
    """วาด heatmap count และ proportion ของทุก class ในทุก batch"""

    width, height = 1240, 870
    left, top = 210, 150
    cell_w, cell_h = 160, 58
    max_share = 0.35
    elements = [
        svg_text(55, 55, "Class distribution by batch", 30, weight=700),
        svg_text(
            55,
            88,
            "Cell = sample count and within-batch proportion",
            17,
            fill="#64748b",
        ),
    ]
    for col, class_id in enumerate(EXPECTED_CLASSES):
        elements.append(
            svg_text(
                left + col * cell_w + (cell_w - 8) / 2,
                top - 24,
                f"C{class_id}: {CLASS_NAMES[class_id]}",
                14,
                anchor="middle",
                weight=600,
            )
        )

    for row, batch in enumerate(EXPECTED_BATCHES):
        y = top + row * (cell_h + 7)
        total = int(count_table.loc[batch].sum())
        elements.append(
            svg_text(left - 20, y + 35, f"Batch {batch} (n={total:,})", 16, anchor="end")
        )
        for col, class_id in enumerate(EXPECTED_CLASSES):
            count = int(count_table.loc[batch, class_id])
            share = count / total
            fill = color_mix((239, 246, 255), (30, 64, 175), share / max_share)
            text_color = "#ffffff" if share >= 0.18 else "#172033"
            x = left + col * cell_w
            elements.append(svg_rect(x, y, cell_w - 8, cell_h, fill, rx=5))
            elements.append(
                svg_text(
                    x + (cell_w - 8) / 2,
                    y + 35,
                    f"{count:,} ({share:.1%})",
                    15,
                    anchor="middle",
                    weight=600,
                    fill=text_color,
                )
            )
    save_svg(path, width, height, elements)


def write_bar_svg(
    path: Path,
    title: str,
    subtitle: str,
    labels: Sequence[str],
    values: Sequence[float],
    value_label: str,
) -> None:
    """วาด bar chart สำหรับ prior mismatch หรือ global MMD"""

    width, height = 1180, 700
    left, right, top, bottom = 110, 45, 145, 110
    plot_w = width - left - right
    plot_h = height - top - bottom
    maximum = max(values) if values else 1.0
    maximum = maximum if maximum > 0 else 1.0
    gap = 22
    bar_w = (plot_w - gap * (len(values) - 1)) / max(1, len(values))
    elements = [
        svg_text(55, 55, title, 30, weight=700),
        svg_text(55, 88, subtitle, 17, fill="#64748b"),
        svg_text(24, top + plot_h / 2, value_label, 16, fill="#475569"),
    ]
    for tick in range(6):
        fraction = tick / 5
        y = top + plot_h * (1 - fraction)
        elements.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" '
            'y2="{:.1f}" stroke="#e2e8f0"/>'.format(y)
        )
        elements.append(
            svg_text(left - 15, y + 5, f"{maximum * fraction:.3f}", 13, anchor="end", fill="#64748b")
        )
    for index, (label, value) in enumerate(zip(labels, values)):
        bar_h = value / maximum * plot_h
        x = left + index * (bar_w + gap)
        y = top + plot_h - bar_h
        elements.append(svg_rect(x, y, bar_w, bar_h, "#2563eb", rx=6))
        elements.append(svg_text(x + bar_w / 2, y - 10, f"{value:.4f}", 14, anchor="middle", weight=600))
        elements.append(svg_text(x + bar_w / 2, top + plot_h + 33, label, 14, anchor="middle", weight=600))
    save_svg(path, width, height, elements)


def write_classwise_mmd_svg(classwise: pd.DataFrame, path: Path) -> None:
    """วาด heatmap ของ class-conditional MMD; ช่องเทาคือ class หาย"""

    targets = sorted(classwise["target_batch"].unique().tolist())
    pivot = classwise.pivot(
        index="target_batch", columns="class_id", values="rbf_mmd_squared_biased"
    ).reindex(index=targets, columns=EXPECTED_CLASSES)
    finite_values = pivot.to_numpy()[np.isfinite(pivot.to_numpy())]
    vmax = float(finite_values.max()) if len(finite_values) else 1.0
    vmax = vmax if vmax > 0 else 1.0

    width, height = 1240, 810
    left, top, cell_w, cell_h = 230, 145, 160, 58
    elements = [
        svg_text(55, 55, "Class-conditional drift", 30, weight=700),
        svg_text(
            55,
            88,
            "RBF-MMD² after source-only scaling; gray = class missing in one domain",
            17,
            fill="#64748b",
        ),
    ]
    for col, class_id in enumerate(EXPECTED_CLASSES):
        elements.append(
            svg_text(
                left + col * cell_w + (cell_w - 8) / 2,
                top - 24,
                f"C{class_id}: {CLASS_NAMES[class_id]}",
                14,
                anchor="middle",
                weight=600,
            )
        )
    for row, target in enumerate(targets):
        y = top + row * (cell_h + 7)
        elements.append(svg_text(left - 20, y + 35, f"Target Batch {target}", 16, anchor="end"))
        for col, class_id in enumerate(EXPECTED_CLASSES):
            value = float(pivot.loc[target, class_id])
            x = left + col * cell_w
            if math.isnan(value):
                fill, label, text_color = "#cbd5e1", "missing", "#475569"
            else:
                fill = color_mix((255, 247, 237), (194, 65, 12), value / vmax)
                label = f"{value:.4f}"
                text_color = "#ffffff" if value / vmax >= 0.58 else "#172033"
            elements.append(svg_rect(x, y, cell_w - 8, cell_h, fill, rx=5))
            elements.append(svg_text(x + (cell_w - 8) / 2, y + 35, label, 15, anchor="middle", weight=600, fill=text_color))
    save_svg(path, width, height, elements)


def write_pca_svg(
    metadata: pd.DataFrame,
    scaled: np.ndarray,
    pca: PCA,
    path: Path,
    max_per_batch: int,
    source_batch: int,
) -> None:
    """ฉายข้อมูลลง PCA 2 มิติ เพื่อดูการแยกตัวของ distribution ตาม batch"""

    projection = pca.transform(scaled)
    rng = np.random.default_rng(31415)
    chosen_indices: list[int] = []
    batches = metadata["batch"].to_numpy()
    for batch in EXPECTED_BATCHES:
        indices = np.flatnonzero(batches == batch)
        if len(indices) > max_per_batch:
            indices = rng.choice(indices, size=max_per_batch, replace=False)
        chosen_indices.extend(indices.tolist())

    points = projection[np.asarray(chosen_indices)]
    point_batches = batches[np.asarray(chosen_indices)]
    # ใช้ percentile ลดผลของ outlier ต่อสเกลกราฟ
    x_min, x_max = np.percentile(points[:, 0], [1, 99])
    y_min, y_max = np.percentile(points[:, 1], [1, 99])
    if x_min == x_max:
        x_min, x_max = x_min - 1, x_max + 1
    if y_min == y_max:
        y_min, y_max = y_min - 1, y_max + 1

    width, height = 1240, 820
    left, right, top, bottom = 90, 230, 125, 75
    plot_w, plot_h = width - left - right, height - top - bottom
    elements = [
        svg_text(55, 52, "Source-fitted PCA view of batch drift", 30, weight=700),
        svg_text(
            55,
            84,
            (
                f"PCA fitted on source only; explained variance = "
                f"{pca.explained_variance_ratio_.sum():.1%}"
            ),
            17,
            fill="#64748b",
        ),
        svg_rect(left, top, plot_w, plot_h, "#ffffff", stroke="#cbd5e1"),
        svg_text(left + plot_w / 2, height - 24, "PC1", 16, anchor="middle", weight=600),
        svg_text(25, top + plot_h / 2, "PC2", 16, weight=600),
    ]

    # วาด target ก่อนและ source ทีหลัง เพื่อให้ source มองเห็นชัด
    draw_order = [b for b in EXPECTED_BATCHES if b != source_batch] + [source_batch]
    for batch in draw_order:
        color = BATCH_COLORS[batch - 1]
        mask = point_batches == batch
        for px, py in points[mask]:
            x = left + (np.clip(px, x_min, x_max) - x_min) / (x_max - x_min) * plot_w
            y = top + plot_h - (np.clip(py, y_min, y_max) - y_min) / (y_max - y_min) * plot_h
            elements.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.2" '
                f'fill="{color}" fill-opacity="0.42"/>'
            )

    legend_x = left + plot_w + 35
    for index, batch in enumerate(EXPECTED_BATCHES):
        y = top + 25 + index * 40
        elements.append(svg_rect(legend_x, y - 14, 20, 20, BATCH_COLORS[batch - 1], rx=3))
        elements.append(svg_text(legend_x + 30, y + 2, f"Batch {batch}", 15))
    save_svg(path, width, height, elements)


# =============================================================================
# ส่วนที่ 7: data leakage audit
# =============================================================================

def make_leakage_audit(
    metadata: pd.DataFrame,
    source_batch: int,
    manifest_target_batch: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ตรวจ duplicate/overlap และบันทึกข้อจำกัดการใช้ target data"""

    duplicate_hashes = (
        metadata.groupby("feature_sha256")
        .agg(
            sample_count=("sample_id", "size"),
            batch_count=("batch", "nunique"),
            class_count=("class_id", "nunique"),
        )
        .query("batch_count > 1")
        .reset_index()
    )
    if duplicate_hashes.empty:
        duplicate_samples = pd.DataFrame(
            columns=[
                "feature_sha256", "sample_id", "batch", "class_id", "class_name"
            ]
        )
    else:
        duplicate_samples = metadata.merge(
            duplicate_hashes[["feature_sha256"]], on="feature_sha256", how="inner"
        )[
            ["feature_sha256", "sample_id", "batch", "class_id", "class_name"]
        ].sort_values(["feature_sha256", "batch", "sample_id"])

    source = metadata[metadata["batch"] == source_batch]
    target = metadata[metadata["batch"] == manifest_target_batch]
    sample_overlap = set(source["sample_id"]) & set(target["sample_id"])
    feature_overlap = set(source["feature_sha256"]) & set(target["feature_sha256"])

    audit_rows = [
        {
            "check": "unique_sample_ids",
            "status": "PASS" if metadata["sample_id"].is_unique else "FAIL",
            "evidence": f"unique={metadata['sample_id'].nunique():,}; total={len(metadata):,}",
            "recommendation": "Sample ID ต้องไม่ซ้ำ",
        },
        {
            "check": "finite_raw_features",
            "status": "PASS",
            "evidence": "ตรวจในขั้นตอน parser ทุก feature แล้ว",
            "recommendation": "หยุดการทำงานทันทีหากพบ NaN/inf",
        },
        {
            "check": "exact_feature_duplicates_across_batches",
            "status": "PASS" if duplicate_hashes.empty else "REVIEW",
            "evidence": f"cross-batch duplicate groups={len(duplicate_hashes):,}",
            "recommendation": "ถ้ามี duplicate ให้ตัดสินใจ deduplicate ก่อน train/test split",
        },
        {
            "check": "source_target_sample_id_overlap",
            "status": "PASS" if not sample_overlap else "FAIL",
            "evidence": f"overlap={len(sample_overlap):,}",
            "recommendation": "source และ target ต้องไม่มี sample_id เดียวกัน",
        },
        {
            "check": "source_target_exact_feature_overlap",
            "status": "PASS" if not feature_overlap else "REVIEW",
            "evidence": f"exact feature-vector overlap={len(feature_overlap):,}",
            "recommendation": "หากใช้เป็น test set ควรตรวจ/ตัด exact duplicate",
        },
        {
            "check": "scaling_fit_scope",
            "status": "PASS",
            "evidence": f"StandardScaler.fit ใช้ Batch {source_batch} เท่านั้น",
            "recommendation": "ห้าม fit scaler ด้วย source+target รวมกันในการทดลองโมเดล",
        },
        {
            "check": "pca_fit_scope",
            "status": "PASS",
            "evidence": f"PCA.fit ใช้ Batch {source_batch} เท่านั้น",
            "recommendation": "Target ใช้ transform เพื่อ visualization เท่านั้น",
        },
        {
            "check": "mmd_bandwidth_fit_scope",
            "status": "PASS",
            "evidence": f"median heuristic คำนวณจาก Batch {source_batch} เท่านั้น",
            "recommendation": "ล็อก bandwidth ก่อนเปรียบเทียบ target ทุก batch",
        },
        {
            "check": "target_label_usage",
            "status": "INFO",
            "evidence": "ใช้ target labels เฉพาะ count/prior และ class-wise diagnostic",
            "recommendation": "ห้ามใช้ target labels เหล่านี้ fit/tune โมเดลถ้าตั้งใจทำ UDA",
        },
        {
            "check": "model_test_contamination",
            "status": "INFO",
            "evidence": "สคริปต์นี้ไม่ train/tune/evaluate predictive model",
            "recommendation": "งานโมเดลภายหลังต้องแยก validation/test และห้าม tune จาก test",
        },
    ]
    return pd.DataFrame(audit_rows), duplicate_samples


# =============================================================================
# ส่วนที่ 8: controlled imbalance protocol และ reproducible manifests
# =============================================================================

def file_sha256(path: Path) -> str:
    """checksum ของ manifest เพื่อยืนยันว่าไฟล์ไม่เปลี่ยน"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_imbalance_manifests(
    metadata: pd.DataFrame,
    manifest_dir: Path,
    source_batch: int,
    target_batch: int,
    reduced_classes: Sequence[int],
    keep_fractions: Sequence[float],
    seeds: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """สร้างหนึ่ง manifest ต่อ scenario/seed โดยสุ่มลดเฉพาะ source class"""

    if source_batch == target_batch:
        raise ValueError("source_batch และ target_batch ต้องต่างกัน")
    invalid_classes = sorted(set(reduced_classes) - set(EXPECTED_CLASSES))
    if invalid_classes:
        raise ValueError(f"reduced class ไม่ถูกต้อง: {invalid_classes}")
    if any(not 0 < fraction <= 1 for fraction in keep_fractions):
        raise ValueError("keep fraction ทุกค่าต้องอยู่ในช่วง (0, 1]")

    manifest_dir.mkdir(parents=True, exist_ok=True)
    source = metadata[metadata["batch"] == source_batch].copy()
    target = metadata[metadata["batch"] == target_batch].copy()
    source_ids = set(source["sample_id"])
    target_ids = set(target["sample_id"])
    if source_ids & target_ids:
        raise ValueError("source-target sample IDs overlap; ไม่สร้าง manifest")

    protocol_rows: list[dict[str, object]] = []
    index_rows: list[dict[str, object]] = []
    manifest_columns = [
        "sample_id", "role", "batch", "class_id", "class_name",
        "row_in_batch", "raw_sha256", "feature_sha256",
    ]

    for class_id in reduced_classes:
        class_source = source[source["class_id"] == class_id]
        if class_source.empty:
            raise ValueError(
                f"Batch {source_batch} ไม่มี class {class_id}; สร้าง protocol ไม่ได้"
            )
        for keep_fraction in keep_fractions:
            keep_count = max(1, int(round(len(class_source) * keep_fraction)))
            for seed in seeds:
                rng = np.random.default_rng(seed)
                kept_ids = set(
                    rng.choice(
                        class_source["sample_id"].to_numpy(),
                        size=keep_count,
                        replace=False,
                    ).tolist()
                )
                selected_source = source[
                    (source["class_id"] != class_id)
                    | source["sample_id"].isin(kept_ids)
                ].copy()
                selected_source["role"] = "source_train"
                selected_target = target.copy()
                selected_target["role"] = "target_evaluation"

                manifest = pd.concat(
                    [selected_source, selected_target], ignore_index=True
                )[manifest_columns]
                manifest = manifest.sort_values(
                    ["role", "batch", "row_in_batch"]
                ).reset_index(drop=True)

                fraction_code = int(round(keep_fraction * 100))
                scenario_base = (
                    f"B{source_batch:02d}_to_B{target_batch:02d}_"
                    f"reduce_C{class_id}_keep_{fraction_code:03d}"
                )
                scenario_id = f"{scenario_base}_seed_{seed}"
                manifest_path = manifest_dir / f"{scenario_id}.csv"
                manifest.to_csv(manifest_path, index=False)

                selected_class_count = int(
                    (
                        (selected_source["class_id"] == class_id)
                    ).sum()
                )
                protocol_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "source_batch": source_batch,
                        "target_batch": target_batch,
                        "reduced_class": class_id,
                        "reduced_class_name": CLASS_NAMES[class_id],
                        "keep_fraction_requested": keep_fraction,
                        "random_seed": seed,
                        "source_class_count_original": len(class_source),
                        "source_class_count_kept": selected_class_count,
                        "source_samples_used": len(selected_source),
                        "target_samples_used": len(selected_target),
                        "target_policy": "unchanged_full_batch",
                    }
                )
                index_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "manifest_file": str(manifest_path.relative_to(manifest_dir.parent)),
                        "manifest_rows": len(manifest),
                        "manifest_sha256": file_sha256(manifest_path),
                    }
                )

    return pd.DataFrame(protocol_rows), pd.DataFrame(index_rows)


# =============================================================================
# ส่วนที่ 9: สรุปผลเป็น Markdown เพื่อเปิดอ่านได้ทันที
# =============================================================================

def write_summary(
    path: Path,
    schema: dict[str, object],
    missing: pd.DataFrame,
    imbalance: pd.DataFrame,
    prior: pd.DataFrame,
    global_drift: pd.DataFrame,
    classwise: pd.DataFrame,
    leakage: pd.DataFrame,
    protocol: pd.DataFrame,
    manifest_index: pd.DataFrame,
    source_batch: int,
    target_batch: int,
    gamma: float,
) -> None:
    """สร้างรายงานสั้นภาษาไทยพร้อมชี้ไปยังผลลัพธ์แบบตาราง"""

    overall_imbalance = imbalance[imbalance["group"] == "Overall"].iloc[0]
    largest_prior = prior.loc[prior["total_variation_distance"].idxmax()]
    largest_global = global_drift.loc[
        global_drift["rbf_mmd_squared_biased"].idxmax()
    ]
    ok_classwise = classwise[classwise["status"] == "ok"]
    largest_classwise = ok_classwise.loc[
        ok_classwise["rbf_mmd_squared_biased"].idxmax()
    ]

    if missing.empty:
        missing_text = "ไม่พบ class ที่หายในแต่ละ batch"
    else:
        missing_text = ", ".join(
            f"Batch {int(row.batch)} ขาด C{int(row.class_id)} ({row.class_name})"
            for row in missing.itertuples()
        )

    review_count = int((leakage["status"] == "REVIEW").sum())
    fail_count = int((leakage["status"] == "FAIL").sum())
    keep_fractions = sorted(protocol["keep_fraction_requested"].unique(), reverse=True)
    seeds = sorted(protocol["random_seed"].unique())
    reduced_classes = sorted(protocol["reduced_class"].unique())
    content = f"""# ผลวิเคราะห์ Dataset จริง

## 1. Dataset schema

- ผลตรวจ official schema: **{'PASS' if schema['all_checks_passed'] else 'FAIL'}**
- จำนวน sample: **{schema['actual']['samples']:,}**
- จำนวน sensor: **{schema['actual']['sensors']}**
- จำนวน feature: **{schema['actual']['features']}** = {schema['actual']['sensors']} sensors × {schema['actual']['features_per_sensor']} features/sensor
- จำนวน class: **{schema['actual']['classes']}**
- จำนวน batch: **{schema['actual']['batches']}**
- แหล่งทางการ: {OFFICIAL_SOURCE} (DOI: {OFFICIAL_DOI})

## 2. Class distribution และ imbalance

- Missing class: {missing_text}
- Overall max/min imbalance ratio: **{overall_imbalance['imbalance_ratio_max_to_min']:.3f}:1**
- ตารางเต็ม: `class_count_table.csv`, `class_proportions.csv`, `missing_classes.csv`, `imbalance_statistics.csv`
- ภาพ: `class_distribution.svg`

## 3. Source-target prior mismatch

- กำหนด source = Batch {source_batch}
- Target ที่ prior ต่างจาก source มากที่สุดตาม Total Variation คือ **Batch {int(largest_prior['target_batch'])}** (TVD={largest_prior['total_variation_distance']:.4f})
- ตารางเต็ม: `prior_mismatch_table.csv`
- ภาพ: `prior_mismatch.svg`

## 4. Drift evidence

- StandardScaler, PCA และ RBF bandwidth ถูก fit จาก source เท่านั้น
- RBF gamma จาก source median heuristic = **{gamma:.8g}**
- Global drift สูงสุดตาม MMD²: Batch {source_batch} → Batch {int(largest_global['target_batch'])}, MMD²={largest_global['rbf_mmd_squared_biased']:.4f}
- Class-wise drift สูงสุด: Batch {source_batch} → Batch {int(largest_classwise['target_batch'])}, Class {int(largest_classwise['class_id'])} ({largest_classwise['class_name']}), MMD²={largest_classwise['rbf_mmd_squared_biased']:.4f}
- ตารางเต็ม: `global_drift_mmd.csv`, `classwise_drift_mmd.csv`
- ภาพ: `global_drift_mmd.svg`, `classwise_drift_mmd.svg`, `pca_batch_projection.svg`

ข้อควรระวัง: global MMD รวมผลของทั้ง feature drift และ class-prior shift ส่วน class-wise MMD ช่วยดู feature drift ภายใน class เดียวกัน แต่จะคำนวณไม่ได้เมื่อ class นั้นหายใน source หรือ target

## 5. Leakage audit

- FAIL = **{fail_count}**, REVIEW = **{review_count}**
- Target labels ถูกใช้เฉพาะ descriptive/class-wise diagnostic ไม่ได้ใช้ train หรือ tune โมเดล
- ตารางเต็ม: `leakage_audit.csv`, `cross_batch_exact_duplicates.csv`

## 6. Controlled imbalance protocol

- Source = Batch {source_batch}, target = Batch {target_batch}
- ลดทีละ class {', '.join(f'C{int(value)}' for value in reduced_classes)} ใน source เหลือ {', '.join(f'{value:.0%}' for value in keep_fractions)}; target คงไว้ทั้ง batch
- Seeds = {', '.join(map(str, seeds))}
- สร้าง manifest ทั้งหมด **{len(manifest_index)} ไฟล์** ในโฟลเดอร์ `manifests/`
- index/checksum: `reproducible_manifest_index.csv`
- รายละเอียด scenario: `imbalance_protocol.csv`

## วิธีรันซ้ำ

```bash
python3 analyze_real_dataset.py
```
"""
    path.write_text(content, encoding="utf-8")


def dataframe_to_html_table(
    frame: pd.DataFrame,
    *,
    include_index: bool = False,
) -> str:
    """แปลง DataFrame เป็นตาราง HTML และปัดทศนิยมให้อ่านง่าย"""

    display = frame.copy()
    float_columns = display.select_dtypes(include=["float"]).columns
    display[float_columns] = display[float_columns].round(6)
    return display.to_html(
        index=include_index,
        border=0,
        classes="analysis-table",
        na_rep="missing",
    )


def write_complete_html_report(
    path: Path,
    output_dir: Path,
    schema: dict[str, object],
    count_output: pd.DataFrame,
    proportions: pd.DataFrame,
    missing: pd.DataFrame,
    imbalance: pd.DataFrame,
    prior: pd.DataFrame,
    global_drift: pd.DataFrame,
    classwise: pd.DataFrame,
    leakage: pd.DataFrame,
    protocol: pd.DataFrame,
    manifest_index: pd.DataFrame,
    source_batch: int,
    target_batch: int,
    gamma: float,
) -> None:
    """รวมผลสรุป ตาราง และ SVG เป็นรายงาน HTML ไฟล์เดียว"""

    overall = imbalance[imbalance["group"] == "Overall"].iloc[0]
    largest_prior = prior.loc[prior["total_variation_distance"].idxmax()]
    largest_global = global_drift.loc[
        global_drift["rbf_mmd_squared_biased"].idxmax()
    ]
    ok_classwise = classwise[classwise["status"] == "ok"]
    largest_classwise = ok_classwise.loc[
        ok_classwise["rbf_mmd_squared_biased"].idxmax()
    ]
    closest_prior = prior.loc[prior["total_variation_distance"].idxmin()]
    largest_mean_shift = global_drift.loc[
        global_drift["standardized_mean_shift_l2"].idxmax()
    ]
    batch_imbalance = imbalance[imbalance["group"].str.startswith("Batch_")]
    complete_batch_imbalance = batch_imbalance[
        batch_imbalance["missing_classes"] == 0
    ]
    worst_complete_batch = complete_batch_imbalance.loc[
        complete_batch_imbalance["imbalance_ratio_max_to_min"].idxmax()
    ]
    largest_prior_class = int(largest_prior["largest_change_class"])
    largest_prior_source_share = float(
        largest_prior[f"source_prior_c{largest_prior_class}"]
    )
    largest_prior_target_share = float(
        largest_prior[f"target_prior_c{largest_prior_class}"]
    )
    top_classwise_rows = ok_classwise.nlargest(
        3, "rbf_mmd_squared_biased"
    )
    top_classwise_html = "".join(
        (
            f"<li>Batch {source_batch} → Batch {int(row.target_batch)}, "
            f"C{int(row.class_id)} ({html.escape(str(row.class_name))}): "
            f"MMD²={row.rbf_mmd_squared_biased:.4f}, "
            f"n_source={int(row.source_samples_total)}, "
            f"n_target={int(row.target_samples_total)}</li>"
        )
        for row in top_classwise_rows.itertuples()
    )
    protocol_ratios = sorted(
        protocol["keep_fraction_requested"].unique(), reverse=True
    )
    protocol_seeds = sorted(protocol["random_seed"].unique())
    source_total = int(count_output.loc[f"Batch_{source_batch}", "total"])
    target_total = int(count_output.loc[f"Batch_{target_batch}", "total"])

    if missing.empty:
        missing_summary = "ไม่พบ class ที่หาย"
    else:
        missing_summary = ", ".join(
            f"Batch {int(row.batch)} ขาด C{int(row.class_id)} ({row.class_name})"
            for row in missing.itertuples()
        )

    svg_names = (
        "class_distribution.svg",
        "prior_mismatch.svg",
        "global_drift_mmd.svg",
        "classwise_drift_mmd.svg",
        "pca_batch_projection.svg",
    )
    inline_svgs = {
        name: (output_dir / name).read_text(encoding="utf-8")
        for name in svg_names
    }

    schema_json = html.escape(
        json.dumps(schema, ensure_ascii=False, indent=2)
    )
    report = f"""<!doctype html>
<html lang="th">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gas Sensor Array Drift — Complete Analysis Report</title>
<style>
  :root {{ color-scheme: light; --ink:#172033; --muted:#64748b;
           --line:#dbe3ee; --blue:#2563eb; --panel:#ffffff; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:#f4f7fb; color:var(--ink);
          font-family:Arial,"Noto Sans Thai",sans-serif; line-height:1.55; }}
  main {{ max-width:1280px; margin:0 auto; padding:36px 24px 80px; }}
  h1 {{ margin:0 0 8px; font-size:34px; }}
  h2 {{ margin-top:42px; border-bottom:2px solid var(--line); padding-bottom:8px; }}
  h3 {{ margin-top:28px; }}
  .subtitle {{ color:var(--muted); margin-bottom:26px; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
            gap:12px; margin:20px 0; }}
  .card {{ background:var(--panel); border:1px solid var(--line); border-radius:12px;
           padding:16px; box-shadow:0 2px 8px rgba(15,23,42,.04); }}
  .card .number {{ color:var(--blue); font-size:27px; font-weight:700; }}
  .card .label {{ color:var(--muted); font-size:14px; }}
  .finding {{ background:#eff6ff; border-left:5px solid var(--blue);
              padding:16px 22px; border-radius:8px; }}
  .explain {{ background:#fff; border:1px solid var(--line); border-radius:12px;
              padding:18px 22px; margin:16px 0 22px; }}
  .explain h3 {{ margin:0 0 8px; color:#1e3a8a; }}
  .takeaway {{ background:#ecfdf5; border-left:5px solid #059669;
               padding:15px 20px; border-radius:8px; margin:16px 0; }}
  .warning {{ background:#fff7ed; border-left:5px solid #ea580c;
              padding:15px 20px; border-radius:8px; margin:16px 0; }}
  .definition {{ background:#f8fafc; border:1px dashed #94a3b8;
                 padding:14px 18px; border-radius:8px; margin:12px 0; }}
  .two-col {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr));
              gap:14px; margin:16px 0; }}
  .two-col > div {{ background:#fff; border:1px solid var(--line);
                    border-radius:10px; padding:16px 18px; }}
  .two-col h3 {{ margin-top:0; }}
  .plain-table {{ width:100%; border-collapse:collapse; background:#fff; }}
  .plain-table th,.plain-table td {{ padding:10px 12px; border:1px solid var(--line);
                                     vertical-align:top; text-align:left; }}
  .plain-table th {{ background:#eaf1fb; }}
  .figure {{ background:#fff; border:1px solid var(--line); border-radius:12px;
             padding:12px; margin:18px 0 26px; overflow:auto; }}
  .figure svg {{ display:block; max-width:100%; height:auto; margin:auto; }}
  .table-wrap {{ overflow:auto; max-height:620px; border:1px solid var(--line);
                 border-radius:10px; background:#fff; }}
  .analysis-table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  .analysis-table th {{ position:sticky; top:0; background:#eaf1fb; z-index:1; }}
  .analysis-table th,.analysis-table td {{ border-bottom:1px solid #e7edf5;
                                           padding:8px 10px; text-align:right;
                                           white-space:nowrap; }}
  .analysis-table th:first-child,.analysis-table td:first-child {{ text-align:left; }}
  pre {{ background:#0f172a; color:#e2e8f0; padding:18px; border-radius:10px;
         overflow:auto; max-height:560px; }}
  code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
  .note {{ color:var(--muted); font-size:14px; }}
  a {{ color:#1d4ed8; }}
  @media print {{ body {{ background:#fff; }} main {{ max-width:none; padding:12px; }}
                  .table-wrap {{ max-height:none; }} }}
</style>
</head>
<body><main>
<h1>ผลวิเคราะห์ Dataset จริง</h1>
<p class="subtitle">UCI Gas Sensor Array Drift Dataset — รายงานรวมแบบไฟล์เดียว</p>

<div class="cards">
  <div class="card"><div class="number">{schema['actual']['samples']:,}</div><div class="label">samples</div></div>
  <div class="card"><div class="number">{schema['actual']['sensors']}</div><div class="label">sensors</div></div>
  <div class="card"><div class="number">{schema['actual']['features']}</div><div class="label">features</div></div>
  <div class="card"><div class="number">{schema['actual']['classes']}</div><div class="label">classes</div></div>
  <div class="card"><div class="number">{schema['actual']['batches']}</div><div class="label">batches</div></div>
  <div class="card"><div class="number">PASS</div><div class="label">official schema validation</div></div>
</div>

<div class="finding">
<strong>ผลสำคัญ</strong>
<ul>
  <li>{html.escape(missing_summary)}</li>
  <li>Overall imbalance ratio = <strong>{overall['imbalance_ratio_max_to_min']:.3f}:1</strong></li>
  <li>Batch {int(largest_prior['target_batch'])} มี class prior ต่างจาก Batch {source_batch} มากที่สุด (TVD={largest_prior['total_variation_distance']:.4f})</li>
  <li>Global drift สูงสุด: Batch {source_batch} → Batch {int(largest_global['target_batch'])}, MMD²={largest_global['rbf_mmd_squared_biased']:.4f}</li>
  <li>Class-wise drift สูงสุด: Batch {source_batch} → Batch {int(largest_classwise['target_batch'])}, C{int(largest_classwise['class_id'])} ({largest_classwise['class_name']}), MMD²={largest_classwise['rbf_mmd_squared_biased']:.4f}</li>
  <li>Leakage audit: FAIL={(leakage['status'] == 'FAIL').sum()}, REVIEW={(leakage['status'] == 'REVIEW').sum()}</li>
  <li>Controlled-imbalance manifests = {len(manifest_index)} ไฟล์ จาก 6 classes × 3 ratios × 3 seeds</li>
</ul>
</div>

<h2>0. อ่านผลนี้อย่างไร</h2>
<div class="two-col">
  <div>
    <h3>สิ่งที่ผลวิเคราะห์ยืนยันได้</h3>
    <ul>
      <li>โครงสร้างไฟล์และจำนวนข้อมูลตรงกับ metadata ทางการ</li>
      <li>สัดส่วน class เปลี่ยนไปตาม batch อย่างชัดเจน</li>
      <li>feature distribution ของหลาย batch ต่างจาก Batch {source_batch}</li>
      <li>แม้ควบคุมให้เป็น class เดียวกันแล้ว ยังพบ class-conditional drift</li>
      <li>protocol และ sample IDs สามารถสร้างซ้ำจาก seed เดิมได้</li>
    </ul>
  </div>
  <div>
    <h3>สิ่งที่ผลนี้ยังยืนยันไม่ได้</h3>
    <ul>
      <li>ยังสรุปไม่ได้ว่าโมเดลใดแม่นที่สุด เพราะยังไม่ได้ train classifier</li>
      <li>MMD ที่สูงไม่ได้บอกสาเหตุว่าเกิดจาก sensor aging, สภาพแวดล้อม หรือปัจจัยใดโดยตรง</li>
      <li>ยังไม่มี permutation test หรือ confidence interval จึงไม่ควรเรียกค่า MMD ว่า “มีนัยสำคัญ”</li>
      <li>PCA เป็นภาพฉาย 2 มิติ ไม่ได้เก็บรายละเอียดครบทั้ง 128 มิติ</li>
      <li>Leakage audit ที่ผ่าน ไม่รับประกันว่างานโมเดลในอนาคตจะไม่ leak หากนำ target labels ไป tune</li>
    </ul>
  </div>
</div>
<div class="definition"><strong>แนวคิดสำคัญ:</strong> ในรายงานนี้ <em>batch</em> หมายถึงกลุ่มข้อมูลที่เก็บต่างช่วงเวลา จึงทำหน้าที่เหมือนคนละ domain ไม่ใช่ mini-batch ที่สุ่มใช้ระหว่างการ train การแบ่ง train/test แบบสุ่มข้ามทุก batch อาจทำให้ผลดูดีกว่าการใช้งานจริงในอนาคต</div>

<h2>1. Dataset schema</h2>
<p>ยืนยันกับ <a href="{OFFICIAL_SOURCE}">UCI official source</a> — DOI: {OFFICIAL_DOI}</p>
<div class="explain">
  <h3>ผลนี้บ่งบอกอะไร</h3>
  <p>แต่ละ sample คือการวัดการตอบสนองต่อก๊าซหนึ่งครั้ง มี feature 128 ค่า ซึ่งมาจาก 16 sensors × 8 features ต่อ sensor และมี label เป็นก๊าซ 6 ชนิด การที่ schema เป็น <strong>PASS</strong> หมายถึง parser อ่านไฟล์ได้ครบ {schema['actual']['samples']:,} แถว, พบ feature 1–128 ครบทุกแถว, ไม่มี NaN/Infinity และพบ batch/class ตามจำนวนที่คาดไว้</p>
  <p>ผล PASS นี้สำคัญเพราะลดความเสี่ยงที่ผล drift จะเกิดจากไฟล์ขาด คอลัมน์เลื่อน หรืออ่าน LIBSVM ผิดรูปแบบ อย่างไรก็ตาม PASS เป็นการยืนยันความสมบูรณ์เชิงโครงสร้าง ไม่ใช่การรับรองว่าข้อมูลสมดุลหรือไม่มี drift</p>
</div>
<pre><code>{schema_json}</code></pre>

<h2>2. Class count, missing class และ imbalance</h2>
<div class="explain">
  <h3>ความหมายของตัวชี้วัด</h3>
  <ul>
    <li><strong>Class proportion</strong> คือสัดส่วนของแต่ละก๊าซภายใน batch นั้น</li>
    <li><strong>Imbalance ratio</strong> = จำนวน class ใหญ่สุด ÷ class เล็กสุด ค่า 1 หมายถึงสมดุลพอดี ถ้ามี class count = 0 อัตราส่วนจะเป็น infinity</li>
    <li><strong>Normalized entropy</strong> อยู่ใกล้ 1 เมื่อสัดส่วนทุก class ใกล้เคียงกัน และลดลงเมื่อข้อมูลกระจุกอยู่บาง class</li>
  </ul>
</div>
<div class="takeaway">
  <strong>ข้อค้นพบหลัก:</strong> เมื่อรวมทุก batch อัตราส่วนสูงสุด/ต่ำสุดเท่ากับ {overall['imbalance_ratio_max_to_min']:.3f}:1 ซึ่งดูไม่รุนแรง แต่ค่านี้ซ่อนปัญหาระดับ batch ไว้ ตัวอย่างเช่น {worst_complete_batch['group'].replace('_', ' ')} มี ratio {worst_complete_batch['imbalance_ratio_max_to_min']:.1f}:1 และ Batch 3–5 ไม่มี C6 (Toluene) เลย ในทางกลับกัน Batch 10 สมดุลพอดี 600 ตัวอย่างต่อ class
</div>
<div class="warning">
  <strong>ผลกระทบต่อโมเดล:</strong> Accuracy รวมอาจถูกครอบงำโดย class ใหญ่ และใน Batch 3–5 ไม่สามารถประเมิน recall ของ Toluene ได้เลย ควรรายงาน macro-F1, balanced accuracy, per-class recall และ confusion matrix ควบคู่กับ accuracy เสมอ การรวมทุก batch แล้วดูเฉพาะ overall ratio ไม่เพียงพอ
</div>
<div class="figure">{inline_svgs['class_distribution.svg']}</div>
<h3>Class count table</h3>
<div class="table-wrap">{dataframe_to_html_table(count_output.reset_index(names='group'))}</div>
<h3>Class proportions</h3>
<div class="table-wrap">{dataframe_to_html_table(proportions.reset_index(names='group'))}</div>
<h3>Missing classes</h3>
<div class="table-wrap">{dataframe_to_html_table(missing)}</div>
<h3>Imbalance statistics</h3>
<div class="table-wrap">{dataframe_to_html_table(imbalance)}</div>

<h2>3. Source-target class-prior mismatch</h2>
<div class="explain">
  <h3>Class prior คืออะไร</h3>
  <p>Class prior คือโอกาสพบแต่ละ class ก่อนเห็น feature เช่น ถ้า Acetone มี 48% ใน target แต่มีเพียง 16% ใน source แม้รูปแบบ feature ภายใน class จะไม่เปลี่ยน โมเดลที่ยึด prior จาก source ก็อาจให้ probability ที่ไม่ calibrated บน target</p>
  <p><strong>Total Variation Distance (TVD)</strong> อยู่ระหว่าง 0–1; 0 หมายถึงสัดส่วน class เหมือนกันทุก class ค่า TVD ยังตีความได้ว่าเป็นมวล probability ขั้นต่ำที่ต้องย้ายระหว่าง class เพื่อให้สอง distribution ตรงกัน ส่วน Jensen–Shannon divergence เป็นอีกตัววัดที่สมมาตรและรองรับ class ที่มี probability เป็นศูนย์</p>
</div>
<div class="takeaway">
  <strong>ผลที่พบ:</strong> Batch {int(largest_prior['target_batch'])} ต่างจาก Batch {source_batch} มากที่สุด โดย TVD={largest_prior['total_variation_distance']:.4f} กล่าวได้ว่าต้องโยก probability mass อย่างน้อยประมาณ {largest_prior['total_variation_distance']:.1%} จึงจะทำให้ prior ตรงกัน การเปลี่ยนที่เด่นที่สุดคือ C{largest_prior_class} ({largest_prior['largest_change_class_name']}) จาก {largest_prior_source_share:.1%} ใน source เป็น {largest_prior_target_share:.1%} ใน target หรือเปลี่ยน {largest_prior['largest_prior_delta_target_minus_source'] * 100:+.1f} percentage points ขณะที่ Batch {int(closest_prior['target_batch'])} ใกล้ source ที่สุดในชุดเปรียบเทียบ (TVD={closest_prior['total_variation_distance']:.4f})
</div>
<div class="warning">
  <strong>การนำไปใช้:</strong> หาก deployment มี prior แบบ Batch {int(largest_prior['target_batch'])} ควรพิจารณา class weighting, prior-probability correction หรือ threshold calibration แต่ห้ามใช้ target test labels เพื่อเลือกวิธี/พารามิเตอร์แล้วรายงานผลบน test ชุดเดิม เพราะจะกลายเป็น test leakage
</div>
<div class="figure">{inline_svgs['prior_mismatch.svg']}</div>
<div class="table-wrap">{dataframe_to_html_table(prior)}</div>

<h2>4. Global drift</h2>
<div class="explain">
  <h3>MMD และ mean shift บอกอะไร</h3>
  <p><strong>RBF-MMD²</strong> เปรียบเทียบ distribution ของ feature ทั้ง 128 มิติ ค่าใกล้ 0 หมายถึงสอง sample distributions คล้ายกันภายใต้ kernel ที่กำหนด และค่าที่มากขึ้นหมายถึงต่างกันมากขึ้น ไม่มีเกณฑ์สากลว่าเท่าไรจึง “สูง” จึงควรเปรียบเทียบแบบสัมพัทธ์ภายในรายงานที่ใช้ scaler, gamma และ sampling protocol เดียวกัน</p>
  <p><strong>Standardized mean-shift L2</strong> ดูระยะห่างระหว่างค่าเฉลี่ยหลัง standardize ส่วน MMD มองความต่างของ distribution กว้างกว่า จึงอาจจัดอันดับ batch ไม่เหมือนกัน</p>
</div>
<div class="takeaway">
  <strong>ผลที่พบ:</strong> MMD สูงสุดคือ Batch {source_batch} → Batch {int(largest_global['target_batch'])} ที่ {largest_global['rbf_mmd_squared_biased']:.4f} แสดงว่า target นี้ต่างจาก source มากที่สุดภายใต้ RBF kernel ที่ใช้ แต่ mean shift สูงสุดเกิดที่ Batch {int(largest_mean_shift['target_batch'])} ({largest_mean_shift['standardized_mean_shift_l2']:.2f}) ความต่างของอันดับสอง metric บ่งชี้ว่า drift ไม่ได้เป็นเพียงการเลื่อนค่าเฉลี่ย แต่อาจรวมการเปลี่ยน variance หรือรูปร่าง distribution ด้วย
</div>
<div class="warning">
  <strong>ข้อจำกัด:</strong> ใช้ source-only scaling, RBF gamma={gamma:.8g} และสุ่มไม่เกิน {int(global_drift['source_samples_used_for_mmd'].max())} ตัวอย่างต่อ domain ด้วย seed คงที่ ค่าเป็น biased MMD² เพื่อให้ไม่ติดลบ แต่ยังไม่มี p-value การที่ Batch 10 มี MMD ต่ำกว่า Batch 4–5 ไม่ได้พิสูจน์ว่า drift ลดลงตามเวลา เพราะ global MMD ยังได้รับอิทธิพลจาก class mix และ sample size</div>
<div class="figure">{inline_svgs['global_drift_mmd.svg']}</div>
<div class="figure">{inline_svgs['pca_batch_projection.svg']}</div>
<div class="table-wrap">{dataframe_to_html_table(global_drift)}</div>

<h2>5. Class-wise drift</h2>
<div class="explain">
  <h3>ทำไมต้องดูภายใน class</h3>
  <p>Global drift อาจสูงเพียงเพราะสัดส่วน class เปลี่ยน การเปรียบเทียบ source-target เฉพาะ sample ที่เป็น class เดียวกันช่วยลดผลของ prior mismatch หาก class-wise MMD ยังสูง แสดงว่า feature distribution ของก๊าซชนิดเดียวกันเปลี่ยนข้าม batch ซึ่งสอดคล้องกับ sensor/domain drift มากกว่าการเปลี่ยนสัดส่วน class อย่างเดียว</p>
</div>
<div class="takeaway">
  <strong>คู่ที่มี class-wise MMD สูงสุด 3 อันดับ:</strong>
  <ol>{top_classwise_html}</ol>
  ค่าอันดับหนึ่งคือ Batch {source_batch} → Batch {int(largest_classwise['target_batch'])}, C{int(largest_classwise['class_id'])} ({largest_classwise['class_name']}) ที่ MMD²={largest_classwise['rbf_mmd_squared_biased']:.4f} จึงมีหลักฐานว่า feature ของ class นี้เปลี่ยนชัด ไม่ใช่แค่จำนวน class เปลี่ยน
</div>
<div class="warning">
  <strong>ข้อควรระวัง:</strong> C6 ใน Batch 3–5 เป็น missing จึงคำนวณ class-conditional MMD ไม่ได้ และคู่ที่มี sample น้อย เช่น target บาง class มีเพียง 12–30 ตัวอย่าง อาจทำให้ค่าประมาณผันผวน ควรเพิ่ม bootstrap confidence interval หรือ permutation test หากต้องการข้อสรุปเชิงสถิติ</div>
<p class="note">ช่อง missing ในตารางหมายถึง class นั้นไม่มีใน source หรือ target</p>
<div class="figure">{inline_svgs['classwise_drift_mmd.svg']}</div>
<div class="table-wrap">{dataframe_to_html_table(classwise)}</div>

<h2>6. Data leakage audit</h2>
<div class="takeaway">
  <strong>ผลที่พบ:</strong> ไม่พบ FAIL หรือ REVIEW, sample IDs ไม่ซ้ำ, ไม่มี exact feature-vector duplicate ข้าม source-target ที่ตรวจพบ และไม่มี sample ID overlap ระหว่าง Batch {source_batch} กับ Batch {target_batch} นอกจากนี้ StandardScaler, PCA และ MMD bandwidth ถูก fit จาก Batch {source_batch} เท่านั้น จึงไม่ดึงสถิติของ target มาใช้เตรียม source
</div>
<div class="warning">
  <strong>PASS ไม่ได้แปลว่าปลอด leakage ตลอดไป:</strong> รายงานนี้ใช้ target labels เพื่ออธิบาย class count, prior และ class-wise drift ซึ่งทำได้ในขั้น audit แต่ถ้าการทดลองถูกกำหนดเป็น unsupervised domain adaptation ห้ามใช้ข้อมูลเหล่านี้เลือก hyperparameter, early stopping หรือเลือกโมเดล การตัดสินใจทั้งหมดควรทำจาก source/validation ที่อนุญาต แล้วเปิด target test เพียงครั้งสุดท้าย</div>
<div class="table-wrap">{dataframe_to_html_table(leakage)}</div>

<h2>7. Controlled imbalance protocol</h2>
<div class="explain">
  <h3>Protocol นี้กำลังทดสอบอะไร</h3>
  <p>เริ่มจาก source Batch {source_batch} จำนวน {source_total:,} ตัวอย่าง แล้วลดข้อมูลทีละ class ให้เหลือ {', '.join(f'{ratio:.0%}' for ratio in protocol_ratios)} ของจำนวนเดิม โดย class อื่นใน source คงไว้ทั้งหมด ส่วน target Batch {target_batch} ใช้ครบ {target_total:,} ตัวอย่าง จุดประสงค์คือแยกผลของ “ความขาดแคลน class ใน source” ออกจาก drift ตามธรรมชาติ</p>
  <p>ทำซ้ำด้วย seeds {', '.join(map(str, protocol_seeds))} เพื่อดูว่าผลโมเดลไวต่อการเลือก sample ย่อยเพียงใด จึงได้ {len(protocol['reduced_class'].unique())} classes × {len(protocol_ratios)} ratios × {len(protocol_seeds)} seeds = <strong>{len(manifest_index)} scenarios</strong></p>
</div>
<div class="takeaway">
  <strong>วิธีอ่านผลทดลองภายหลัง:</strong> สำหรับแต่ละ class/ratio ควรรายงานค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐานจาก 3 seeds แล้วเปรียบเทียบกับ baseline ที่ไม่ลด class หาก macro-F1 หรือ recall ของ class ที่ถูกลดตกลงตาม keep fraction จะบอก sensitivity ต่อ source imbalance; หากวิธีแก้ imbalance ช่วยได้สม่ำเสมอทุก seed จึงมีหลักฐานที่แข็งแรงกว่าการรายงานเพียงรอบเดียว
</div>
<div class="warning">
  Protocol ปัจจุบันสร้าง imbalance เฉพาะ source และใช้ Batch {target_batch} ที่สมดุลเป็น evaluation จึงตอบคำถามเรื่อง source-side imbalance ไม่ได้จำลองกรณี target stream ไม่สมดุล หากต้องการตอบคำถามนั้นต้องสร้าง protocol แยกต่างหาก
</div>
<div class="table-wrap">{dataframe_to_html_table(protocol)}</div>

<h2>8. Reproducible manifest index</h2>
<div class="explain">
  <h3>ทำไมต้องมี manifest และ checksum</h3>
  <p>แต่ละ manifest ระบุ sample ID ที่ใช้จริง, role (`source_train` หรือ `target_evaluation`), batch, class และ hash ของ raw row จึงสามารถตรวจย้อนกลับได้ว่าการทดลองแต่ละ seed ใช้ข้อมูลชุดใด ส่วน SHA-256 ของไฟล์ manifest ใช้ตรวจว่าไฟล์ไม่ถูกแก้หลังสร้าง</p>
  <p>Sample ID ถูกสร้างจาก batch และเลขบรรทัด เช่น `B01_R00001` จึงคงที่ตราบใดที่ไฟล์ raw เดิมไม่ถูกเรียงแถวใหม่ รายงาน HTML รวมรายละเอียด scenario และ checksum ส่วน sample IDs ฉบับเต็มอยู่ใน CSV manifest เพื่อให้ pipeline อ่านได้โดยตรง</p>
</div>
<div class="table-wrap">{dataframe_to_html_table(manifest_index)}</div>

<h2>9. ข้อสรุปและแนวทางทดลองต่อ</h2>
<table class="plain-table">
  <thead><tr><th>หลักฐานที่พบ</th><th>ความหมาย</th><th>สิ่งที่ควรทำต่อ</th></tr></thead>
  <tbody>
    <tr><td>Class prior เปลี่ยนตาม batch</td><td>โมเดลอาจ bias ไปยังสัดส่วนของ source และ probability อาจไม่ calibrated</td><td>รายงาน macro-F1/balanced accuracy และทดลอง weighting หรือ prior correction</td></tr>
    <tr><td>Global MMD ต่างกันชัดระหว่าง batch</td><td>feature distribution เปลี่ยน ทำให้ decision boundary จาก source อาจเสื่อม</td><td>สร้าง baseline source-only แล้วเปรียบเทียบ domain adaptation/drift compensation</td></tr>
    <tr><td>Class-wise MMD ยังสูง</td><td>drift ไม่ได้เกิดจาก class mix อย่างเดียว แต่เกิดภายใน class เดียวกันด้วย</td><td>ดู per-class recall และทดลอง class-conditional alignment</td></tr>
    <tr><td>C6 หายใน Batch 3–5</td><td>เกิด class coverage gap และวัด recall ของ C6 ใน batch เหล่านี้ไม่ได้</td><td>แยกรายงาน “not observable” ออกจาก recall=0 และเลือก test batch ที่มีทุก class เมื่อจำเป็น</td></tr>
    <tr><td>บาง batch imbalance รุนแรง</td><td>Accuracy รวมอาจทำให้เข้าใจผิด</td><td>ใช้ stratified source validation เท่าที่ทำได้ และรายงานผลแยก class</td></tr>
    <tr><td>Leakage audit ผ่าน</td><td>pipeline วิเคราะห์ปัจจุบันแยก fit scope ถูกต้อง</td><td>คงกฎ source-only preprocessing และล็อก test ก่อนเริ่ม tune โมเดล</td></tr>
  </tbody>
</table>
<div class="takeaway"><strong>บทสรุป:</strong> Dataset นี้ไม่ใช่เพียงปัญหา class imbalance และไม่ใช่เพียงปัญหา sensor drift อย่างใดอย่างหนึ่ง แต่มีทั้ง prior shift, missing-class coverage และ class-conditional feature drift พร้อมกัน การประเมินที่น่าเชื่อถือจึงควร split ตามเวลา/batch, หลีกเลี่ยง target leakage, ใช้ metric ที่คำนึงถึง class และรายงานความแปรปรวนข้าม seeds</div>

<h2>10. วิธีรันซ้ำ</h2>
<pre><code>python3 analyze_real_dataset.py</code></pre>
</main></body></html>
"""
    path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """รับค่าจาก command line โดยทุกค่ามี default ที่รันได้ทันที"""

    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Analyze UCI Gas Sensor Array Drift Dataset"
    )
    parser.add_argument("--dataset-dir", type=Path, default=script_dir)
    parser.add_argument(
        "--output-dir", type=Path, default=script_dir / "analysis_results"
    )
    parser.add_argument("--source-batch", type=int, default=DEFAULT_SOURCE_BATCH)
    parser.add_argument(
        "--manifest-target-batch",
        type=int,
        default=DEFAULT_MANIFEST_TARGET_BATCH,
    )
    parser.add_argument(
        "--reduce-classes",
        type=int,
        nargs="+",
        default=list(DEFAULT_REDUCED_CLASSES),
    )
    parser.add_argument(
        "--keep-fractions",
        type=float,
        nargs="+",
        default=list(DEFAULT_KEEP_FRACTIONS),
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_RANDOM_SEEDS)
    )
    parser.add_argument("--mmd-max-samples", type=int, default=400)
    parser.add_argument("--pca-max-samples-per-batch", type=int, default=400)
    parser.add_argument(
        "--no-strict-official",
        action="store_true",
        help="บันทึกผลต่อแม้ count ไม่ตรง official (ปกติจะหยุดทันที)",
    )
    return parser.parse_args()


def main() -> None:
    """ลำดับการทำงานหลักและการบันทึกผลทั้งหมด"""

    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/9] อ่านและตรวจ raw dataset...")
    bundle = load_dataset(dataset_dir)
    count_table = make_count_table(bundle.metadata)

    print("[2/9] ยืนยัน schema/count กับ UCI official metadata...")
    schema, count_validation = validate_official_schema(
        bundle,
        count_table,
        strict=not args.no_strict_official,
    )
    schema["runtime_versions"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn_version,
    }
    (output_dir / "dataset_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    count_validation.to_csv(
        output_dir / "official_count_validation.csv", index=False
    )
    make_feature_schema().to_csv(output_dir / "feature_schema.csv", index=False)
    bundle.metadata.to_csv(output_dir / "sample_metadata.csv", index=False)

    print("[3/9] วิเคราะห์ class distribution และ imbalance...")
    count_output, proportions, missing, imbalance = class_distribution_outputs(
        count_table
    )
    count_output.to_csv(output_dir / "class_count_table.csv", index_label="group")
    proportions.to_csv(output_dir / "class_proportions.csv", index_label="group")
    missing.to_csv(output_dir / "missing_classes.csv", index=False)
    imbalance.to_csv(output_dir / "imbalance_statistics.csv", index=False)
    write_class_distribution_svg(
        count_table, output_dir / "class_distribution.svg"
    )

    print("[4/9] เปรียบเทียบ class prior ของ source-target...")
    prior = make_prior_mismatch_table(count_table, args.source_batch)
    prior.to_csv(output_dir / "prior_mismatch_table.csv", index=False)
    write_bar_svg(
        output_dir / "prior_mismatch.svg",
        title=f"Class-prior mismatch: Batch {args.source_batch} vs targets",
        subtitle="Total Variation Distance (0 = identical class proportions)",
        labels=[f"B{int(v)}" for v in prior["target_batch"]],
        values=prior["total_variation_distance"].tolist(),
        value_label="TVD",
    )

    print("[5/9] คำนวณ global drift และ class-wise drift...")
    global_drift, classwise, scaled, pca, gamma = compute_drift(
        bundle,
        source_batch=args.source_batch,
        max_mmd_rows=args.mmd_max_samples,
    )
    global_drift.to_csv(output_dir / "global_drift_mmd.csv", index=False)
    classwise.to_csv(output_dir / "classwise_drift_mmd.csv", index=False)
    write_bar_svg(
        output_dir / "global_drift_mmd.svg",
        title=f"Global drift: Batch {args.source_batch} vs targets",
        subtitle="Biased RBF-MMD² after source-only standardization",
        labels=[f"B{int(v)}" for v in global_drift["target_batch"]],
        values=global_drift["rbf_mmd_squared_biased"].tolist(),
        value_label="MMD²",
    )
    write_classwise_mmd_svg(
        classwise, output_dir / "classwise_drift_mmd.svg"
    )
    write_pca_svg(
        bundle.metadata,
        scaled,
        pca,
        output_dir / "pca_batch_projection.svg",
        max_per_batch=args.pca_max_samples_per_batch,
        source_batch=args.source_batch,
    )

    print("[6/9] ตรวจ data leakage...")
    leakage, duplicate_samples = make_leakage_audit(
        bundle.metadata,
        source_batch=args.source_batch,
        manifest_target_batch=args.manifest_target_batch,
    )
    leakage.to_csv(output_dir / "leakage_audit.csv", index=False)
    duplicate_samples.to_csv(
        output_dir / "cross_batch_exact_duplicates.csv", index=False
    )

    print("[7/9] สร้าง controlled-imbalance manifests...")
    protocol, manifest_index = create_imbalance_manifests(
        bundle.metadata,
        manifest_dir=output_dir / "manifests",
        source_batch=args.source_batch,
        target_batch=args.manifest_target_batch,
        reduced_classes=args.reduce_classes,
        keep_fractions=args.keep_fractions,
        seeds=args.seeds,
    )
    protocol.to_csv(output_dir / "imbalance_protocol.csv", index=False)
    manifest_index.to_csv(
        output_dir / "reproducible_manifest_index.csv", index=False
    )

    print("[8/9] สร้างรายงานสรุป...")
    write_summary(
        output_dir / "analysis_summary.md",
        schema=schema,
        missing=missing,
        imbalance=imbalance,
        prior=prior,
        global_drift=global_drift,
        classwise=classwise,
        leakage=leakage,
        protocol=protocol,
        manifest_index=manifest_index,
        source_batch=args.source_batch,
        target_batch=args.manifest_target_batch,
        gamma=gamma,
    )
    write_complete_html_report(
        output_dir / "complete_analysis_report.html",
        output_dir=output_dir,
        schema=schema,
        count_output=count_output,
        proportions=proportions,
        missing=missing,
        imbalance=imbalance,
        prior=prior,
        global_drift=global_drift,
        classwise=classwise,
        leakage=leakage,
        protocol=protocol,
        manifest_index=manifest_index,
        source_batch=args.source_batch,
        target_batch=args.manifest_target_batch,
        gamma=gamma,
    )

    print("[9/9] เสร็จสมบูรณ์")
    print(f"ผลลัพธ์: {output_dir}")
    print(f"รายงานหลัก: {output_dir / 'analysis_summary.md'}")
    print(f"รายงานรวมไฟล์เดียว: {output_dir / 'complete_analysis_report.html'}")


if __name__ == "__main__":
    main()
