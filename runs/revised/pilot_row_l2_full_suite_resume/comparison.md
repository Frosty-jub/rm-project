# Full MPC-CNN suite comparison

Final-epoch, seed 42, exploratory modified MPC-CNN selected using target scores. The 17 distinct trained pairs fill 9 long-term and 9 short-term protocol rows; 1→2 appears in both means.

| Protocol | Target macro | Source macro | Paper macro | Gap |
|---|---:|---:|---:|---:|
| long-term | 62.72% | 100.00% | 76.06% | -13.33 pp |
| short-term | 76.69% | 100.00% | 82.07% | -5.38 pp |

| Pair | Protocol | Target | Source | Paper | Gap |
|---|---|---:|---:|---:|---:|
| 1→2 | long + short | 71.54% | 100.00% | 85.06% | -13.52 pp |
| 1→3 | long | 72.64% | 100.00% | 84.79% | -12.15 pp |
| 1→4 | long | 59.01% | 100.00% | 80.10% | -21.09 pp |
| 1→5 | long | 75.13% | 100.00% | 75.58% | -0.45 pp |
| 1→6 | long | 83.61% | 100.00% | 95.16% | -11.55 pp |
| 1→7 | long | 48.63% | 100.00% | 71.28% | -22.65 pp |
| 1→8 | long | 45.92% | 100.00% | 64.04% | -18.12 pp |
| 1→9 | long | 52.98% | 100.00% | 62.12% | -9.14 pp |
| 1→10 | long | 55.06% | 100.00% | 66.37% | -11.31 pp |
| 2→3 | short | 96.60% | 100.00% | 92.88% | +3.72 pp |
| 3→4 | short | 64.60% | 100.00% | 96.13% | -31.53 pp |
| 4→5 | short | 95.94% | 100.00% | 94.71% | +1.23 pp |
| 5→6 | short | 73.83% | 100.00% | 74.31% | -0.48 pp |
| 6→7 | short | 77.14% | 100.00% | 76.86% | +0.28 pp |
| 7→8 | short | 91.50% | 100.00% | 90.20% | +1.30 pp |
| 8→9 | short | 82.77% | 100.00% | 88.38% | -5.61 pp |
| 9→10 | short | 36.28% | 100.00% | 40.10% | -3.82 pp |

See `tasks.csv`, per-pair `result.json` and `model.pt`, and `audit.json` for the underlying artifacts. The recipe is intentionally modified from the paper; one seed and target-informed recipe selection do not establish paper reproduction.
