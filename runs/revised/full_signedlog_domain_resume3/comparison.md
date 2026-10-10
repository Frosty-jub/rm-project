# Full MPC-CNN suite comparison

Final-epoch, seed 42, exploratory modified MPC-CNN selected using target scores. The 17 distinct trained pairs fill 9 long-term and 9 short-term protocol rows; 1→2 appears in both means.

| Protocol | Target macro | Source macro | Paper macro | Gap |
|---|---:|---:|---:|---:|
| long-term | 65.26% | 100.00% | 76.06% | -10.79 pp |
| short-term | 66.07% | 100.00% | 82.07% | -16.00 pp |

| Pair | Protocol | Target | Source | Paper | Gap |
|---|---|---:|---:|---:|---:|
| 1→2 | long + short | 73.55% | 100.00% | 85.06% | -11.51 pp |
| 1→3 | long | 76.54% | 100.00% | 84.79% | -8.25 pp |
| 1→4 | long | 64.60% | 100.00% | 80.10% | -15.50 pp |
| 1→5 | long | 74.11% | 100.00% | 75.58% | -1.47 pp |
| 1→6 | long | 87.30% | 100.00% | 95.16% | -7.86 pp |
| 1→7 | long | 61.80% | 100.00% | 71.28% | -9.48 pp |
| 1→8 | long | 47.28% | 100.00% | 64.04% | -16.76 pp |
| 1→9 | long | 48.09% | 100.00% | 62.12% | -14.03 pp |
| 1→10 | long | 54.08% | 100.00% | 66.37% | -12.29 pp |
| 2→3 | short | 88.46% | 100.00% | 92.88% | -4.42 pp |
| 3→4 | short | 78.88% | 100.00% | 96.13% | -17.25 pp |
| 4→5 | short | 56.85% | 100.00% | 94.71% | -37.86 pp |
| 5→6 | short | 64.17% | 100.00% | 74.31% | -10.14 pp |
| 6→7 | short | 77.14% | 100.00% | 76.86% | +0.28 pp |
| 7→8 | short | 66.67% | 100.00% | 90.20% | -23.53 pp |
| 8→9 | short | 64.26% | 100.00% | 88.38% | -24.12 pp |
| 9→10 | short | 24.64% | 100.00% | 40.10% | -15.46 pp |

See `tasks.csv`, per-pair `result.json` and `model.pt`, and `audit.json` for the underlying artifacts. The recipe is intentionally modified from the paper; one seed and target-informed recipe selection do not establish paper reproduction.
