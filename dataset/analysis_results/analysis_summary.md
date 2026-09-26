# ผลวิเคราะห์ Dataset จริง

## 1. Dataset schema

- ผลตรวจ official schema: **PASS**
- จำนวน sample: **13,910**
- จำนวน sensor: **16**
- จำนวน feature: **128** = 16 sensors × 8 features/sensor
- จำนวน class: **6**
- จำนวน batch: **10**
- แหล่งทางการ: https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset (DOI: 10.24432/C5RP6W)

## 2. Class distribution และ imbalance

- Missing class: Batch 3 ขาด C6 (Toluene), Batch 4 ขาด C6 (Toluene), Batch 5 ขาด C6 (Toluene)
- Overall max/min imbalance ratio: **1.834:1**
- ตารางเต็ม: `class_count_table.csv`, `class_proportions.csv`, `missing_classes.csv`, `imbalance_statistics.csv`
- ภาพ: `class_distribution.svg`

## 3. Source-target prior mismatch

- กำหนด source = Batch 1
- Target ที่ prior ต่างจาก source มากที่สุดตาม Total Variation คือ **Batch 8** (TVD=0.3739)
- ตารางเต็ม: `prior_mismatch_table.csv`
- ภาพ: `prior_mismatch.svg`

## 4. Drift evidence

- StandardScaler, PCA และ RBF bandwidth ถูก fit จาก source เท่านั้น
- RBF gamma จาก source median heuristic = **0.0029796342**
- Global drift สูงสุดตาม MMD²: Batch 1 → Batch 5, MMD²=0.4874
- Class-wise drift สูงสุด: Batch 1 → Batch 5, Class 1 (Ethanol), MMD²=1.0246
- ตารางเต็ม: `global_drift_mmd.csv`, `classwise_drift_mmd.csv`
- ภาพ: `global_drift_mmd.svg`, `classwise_drift_mmd.svg`, `pca_batch_projection.svg`

ข้อควรระวัง: global MMD รวมผลของทั้ง feature drift และ class-prior shift ส่วน class-wise MMD ช่วยดู feature drift ภายใน class เดียวกัน แต่จะคำนวณไม่ได้เมื่อ class นั้นหายใน source หรือ target

## 5. Leakage audit

- FAIL = **0**, REVIEW = **0**
- Target labels ถูกใช้เฉพาะ descriptive/class-wise diagnostic ไม่ได้ใช้ train หรือ tune โมเดล
- ตารางเต็ม: `leakage_audit.csv`, `cross_batch_exact_duplicates.csv`

## 6. Controlled imbalance protocol

- Source = Batch 1, target = Batch 10
- ลดทีละ class C1, C2, C3, C4, C5, C6 ใน source เหลือ 50%, 25%, 10%; target คงไว้ทั้ง batch
- Seeds = 42, 123, 2026
- สร้าง manifest ทั้งหมด **54 ไฟล์** ในโฟลเดอร์ `manifests/`
- index/checksum: `reproducible_manifest_index.csv`
- รายละเอียด scenario: `imbalance_protocol.csv`

## วิธีรันซ้ำ

```bash
python3 analyze_real_dataset.py
```
