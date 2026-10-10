# Full MPC-CNN suite comparison

Final-epoch, seed 42, exploratory modified MPC-CNN selected using target scores. The 17 distinct trained pairs fill 9 long-term and 9 short-term protocol rows; 1→2 appears in both means.

| Protocol | Target macro | Source macro | Paper macro | Gap |
|---|---:|---:|---:|---:|
| long-term | 66.85% | 100.00% | 76.06% | -9.20 pp |
| short-term | 70.06% | 100.00% | 82.07% | -12.01 pp |

| Pair | Protocol | Target | Source | Paper | Gap |
|---|---|---:|---:|---:|---:|
| 1→2 | long + short | 75.00% | 100.00% | 85.06% | -10.06 pp |
| 1→3 | long | 77.55% | 100.00% | 84.79% | -7.24 pp |
| 1→4 | long | 70.81% | 100.00% | 80.10% | -9.29 pp |
| 1→5 | long | 69.04% | 100.00% | 75.58% | -6.54 pp |
| 1→6 | long | 92.35% | 100.00% | 95.16% | -2.81 pp |
| 1→7 | long | 69.31% | 100.00% | 71.28% | -1.97 pp |
| 1→8 | long | 49.32% | 100.00% | 64.04% | -14.72 pp |
| 1→9 | long | 52.55% | 100.00% | 62.12% | -9.57 pp |
| 1→10 | long | 45.75% | 100.00% | 66.37% | -20.62 pp |
| 2→3 | short | 88.84% | 100.00% | 92.88% | -4.04 pp |
| 3→4 | short | 91.93% | 100.00% | 96.13% | -4.20 pp |
| 4→5 | short | 83.76% | 100.00% | 94.71% | -10.95 pp |
| 5→6 | short | 64.17% | 100.00% | 74.31% | -10.14 pp |
| 6→7 | short | 75.12% | 100.00% | 76.86% | -1.74 pp |
| 7→8 | short | 67.35% | 100.00% | 90.20% | -22.85 pp |
| 8→9 | short | 62.98% | 100.00% | 88.38% | -25.40 pp |
| 9→10 | short | 21.39% | 100.00% | 40.10% | -18.71 pp |

See `tasks.csv`, per-pair `result.json` and `model.pt`, and `audit.json` for the underlying artifacts. The recipe is intentionally modified from the paper; one seed and target-informed recipe selection do not establish paper reproduction.
