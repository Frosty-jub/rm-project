# Reproduction results (Dataset A, original implementation)

**Recovered historical report:** This document was restored from local patch history after the original folder was deleted. The old result CSV, JSON diagnostics and `.pt` checkpoints linked below were not recovered. The tables preserve the measurements recorded before deletion; their file links are historical references rather than currently openable artifacts. New experiments are recorded in `runs/revised/` and [the current explanation](WHY_FIRST_RUN_DIFFERS.md).

All 18-task results below came from the original implementation, now represented by the `legacy` architecture option: it had an extra LeakyReLU after the fourth convolution and omitted the ReLU shown before each classifier FC in Figure 3. They therefore do not measure an exact Figure 3 architecture. They use seed 42, Python 3.12.13, NumPy 2.3.5, PyTorch 2.14.0+cpu on Windows 11, and the supplied 13,910-sample Dataset A. The machine has an NVIDIA RTX 2060, but this run used the CPU PyTorch build. The exact environment and arguments are saved in each run directory. No target labels were used for training, weighting, or scaling.

## Full 800-epoch results

The [combined CSV](runs/full_combined/results.csv) contains all 18 transfer tasks. Each uses the paper's task-specific formal learning rate, batch size, center weight, and 800 epochs; 100 pretraining epochs, source-only z-score scaling, seed 42, and the documented choices in [reproduction_note.md](reproduction_note.md).

| Protocol | Our macro mean | Paper macro mean | Gap |
|---|---:|---:|---:|
| Long-term, 9 tasks | 36.75% | 76.06% | -39.30 pp |
| Short-term, 9 tasks | 49.13% | 82.07% | -32.94 pp |

| Protocol | Transfer | Ours | Paper | Gap |
|---|---|---:|---:|---:|
| Long | 1→2 | 44.29% | 85.06% | -40.77 pp |
| Long | 1→3 | 55.61% | 84.79% | -29.18 pp |
| Long | 1→4 | 37.89% | 80.10% | -42.21 pp |
| Long | 1→5 | 42.13% | 75.58% | -33.45 pp |
| Long | 1→6 | 43.83% | 95.16% | -51.33 pp |
| Long | 1→7 | 30.61% | 71.28% | -40.67 pp |
| Long | 1→8 | 20.41% | 64.04% | -43.63 pp |
| Long | 1→9 | 13.40% | 62.12% | -48.72 pp |
| Long | 1→10 | 42.58% | 66.37% | -23.79 pp |
| Short | 1→2 | 44.29% | 85.06% | -40.77 pp |
| Short | 2→3 | 65.70% | 92.88% | -27.18 pp |
| Short | 3→4 | 80.12% | 96.13% | -16.01 pp |
| Short | 4→5 | 14.21% | 94.71% | -80.50 pp |
| Short | 5→6 | 54.52% | 74.31% | -19.79 pp |
| Short | 6→7 | 56.16% | 76.86% | -20.70 pp |
| Short | 7→8 | 87.76% | 90.20% | -2.44 pp |
| Short | 8→9 | 18.30% | 88.38% | -70.08 pp |
| Short | 9→10 | 21.11% | 40.10% | -18.99 pp |

## Pilot coverage

The [18-task pilot](runs/pilot_all_5_bnfix/results.csv) used 5 formal epochs and 1 pretraining epoch to check every data path. Its long and short macro means were 13.03% and 19.62%. These are execution checks, not paper-comparable results.

## Full-budget task 1 → 2

| Z-score population | Our accuracy | Paper accuracy | Gap | Source accuracy at evaluation |
|---|---:|---:|---:|---:|
| [Source only](runs/full_1_to_2_bnfix/short_1_to_2_mpc.json) | 44.29% | 85.06% | −40.77 pp | 100% |
| [Pooled unlabeled source + target](runs/full_1_to_2_pooled_bnfix/short_1_to_2_mpc.json) | 51.85% | 85.06% | −33.21 pp | 100% |

Both used 100 MMD pretraining epochs and the paper's formal settings of 800 epochs, 5×10⁻⁴ learning rate, batch size 90, and center weight 2. Figure 4(a) shows the paper's Batch 1/2 pretraining example through approximately 600 epochs, so our 100-epoch choice does not match that figure. A per-task pretraining schedule and z-score population are not fully specified. The pooled run is a sensitivity analysis, not a selected benchmark setting.

The [source-only confusion report](runs/full_1_to_2_bnfix/short_1_to_2_mpc.diagnostics.json) shows 520 of 532 target samples with raw label 5 (Ammonia) predicted as raw label 1 (Acetone). On long 1→6, the [confusion report](runs/full_all/long_1_to_6_mpc.diagnostics.json) shows all target samples predicted as only raw labels 1 or 2 (Acetone or Acetaldehyde), despite 100% source accuracy. On short 4→5, the [confusion report](runs/full_short/short_4_to_5_mpc.diagnostics.json) shows source accuracy only 62.73% and target predictions limited to two classes. These gas names are inferred by matching raw-ID counts to paper Table 2 across all ten batches. The earlier prose assigned gas names using a conflicting mapping; that naming was wrong, but the numeric predictions and accuracy were unaffected. These failures, alongside architecture differences and unspecified preprocessing, Gaussian bandwidth, dropout, regularization, learning-rate schedule, and center update rate, are candidate explanations for the gap; they do not establish which choice caused it.

An earlier run let unlabeled target batches update BatchNorm running statistics during formal training; its source evaluation accuracy fell to 20.2%. The runs linked above instead evaluate target features without modifying those running statistics. The paper does not specify this policy, so the earlier behavior is not by itself a proven paper violation. The current policy introduces a train/eval mismatch between source and target features, demonstrated in the [2026-09-29 audit](paper_fidelity_audit.md).

## Figure 3 activation audit: Batch 1 → Batch 2

The default `figure3` architecture addresses the two visible activation differences in the original implementation, interpreting ReLU/FC as ReLU before FC. Each completed run below used 800 formal epochs, the SI task learning rate and batch size, and seed 42. Their environment was Python 3.12.4, PyTorch 2.13.0+cpu, NumPy 1.26.4, different from the legacy runs above; the legacy-to-figure3 comparison is therefore not controlled for software versions. These are focused diagnostics, not a completed 18-task reproduction for the corrected architecture. The [2026-09-29 audit](paper_fidelity_audit.md) reproduced the saved accuracies and found all-zero source inputs to classifier levels 2 and 3 in the collapsed checkpoint.

| Run | MMD pretraining epochs | Center weight | Z-score fit | Source accuracy | Target accuracy |
|---|---:|---:|---|---:|---:|
| [MPC-CNN, Figure 3](runs/full_figure3/long_1_to_2_mpc.json) | 100 | 2 | Source | 22.02% | 26.85% |
| [MPC-CNN, no pretraining](runs/diagnostic_figure3_no_pretrain/short_1_to_2_mpc.json) | 0 | 2 | Source | 74.61% | 36.98% |
| [MPC-CNN, no center loss](runs/diagnostic_figure3_no_center/short_1_to_2_mpc.json) | 100 | 0 | Source | 100.00% | 42.85% |
| [CNN baseline](runs/diagnostic_figure3_cnn/short_1_to_2_cnn.json) | 0 | 0 | Source | 100.00% | 53.78% |
| [CNN baseline, pooled scaling](runs/diagnostic_figure3_cnn_pooled/short_1_to_2_cnn.json) | 0 | 0 | Source + unlabeled target | 100.00% | 48.95% |

For CNN rows, 0 denotes the effective center-loss contribution: the CNN path does not construct or use class centers, although its saved task configuration also records the SI center weight.

The paper reports 85.06% for MPC-CNN and 70.74% for CNN on this task. The corrected-layout MPC-CNN predicts one class for all 1,244 target samples and also fails to fit source. Removing either the chosen pretraining schedule or center-loss contribution improves source fit. These controlled changes support an interaction between our pretraining and center-loss implementation; they do not identify the unpublished choices needed to reproduce the paper. The CNN baseline fits source but remains 16.96 percentage points below the paper on target, so the gap also involves transfer behavior. Pooled scaling reduces this CNN's accuracy by 4.82 points, while it improved the earlier legacy MPC-CNN run by 7.56 points; preprocessing effects depend on the training configuration. Ablations can change random-number consumption and therefore are diagnostic, not a causal proof from a single seed.
