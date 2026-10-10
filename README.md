# MPC-CNN reproduction (Gas Sensor Array Drift Dataset)

**Recovery note (9 October 2026):** The original workspace directory was accidentally deleted. Source code was recovered from local patch history; its reviewed SHA-256 values match the 29 September snapshot. Dataset A was restored from UCI with its original 10 batches and byte hashes for batches 1/2. Historical `runs/` checkpoints and CSV files were not recovered. New checkpoints are in `runs/revised/`. See [why the first run differed](WHY_FIRST_RUN_DIFFERS.md) and the [recovery experiment plan](experiment_plan_20261008.md).

Implementation of the **Dataset A classification** experiments in Jiang et al., *ACS Sensors* 2025, DOI [10.1021/acssensors.4c03655](https://doi.org/10.1021/acssensors.4c03655). The local `dataset/batch1.dat` through `batch10.dat` files contain all 13,910 samples. The model follows Figure 3: 16 × 8 input, four 2D convolution layers (8, 16, 32, 16 channels), FC 128 → 64 → 32, and three six-class heads. It uses three Gaussian MMD distances for unsupervised pretraining and ensemble weights, then cross-entropy plus three center losses for formal training.

The original 18-task results survive only as numbers in [reproduction_results.md](reproduction_results.md); the original CSV and checkpoints were deleted with the workspace. Those checkpoints used the `legacy` activation layout. The current default `figure3` option corrects the two visible Figure 3 activation differences. The implementation does not match the paper's reported accuracy; the report documents the gaps and diagnostics.

## Environment

The paper used Python 3.9, PyTorch, a GTX 4060, and 16 GB RAM. The historical reproduction used Python 3.12.13, NumPy 2.3.5, PyTorch 2.14.0+cpu, and seed 42. The restored GPU environment uses Python 3.12.4, NumPy 1.26.4, PyTorch 2.7.1+cu126, and a local RTX 2060. Each run saves exact versions, hardware choice, seed, and arguments to `environment.json`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements_mpc.txt
```

To recreate the tested CPU package versions, install `requirements_mpc.lock.txt` instead. For GPU acceleration, use a PyTorch build compatible with the installed CUDA driver.

## Run

```powershell
# Check one transfer task: one unsupervised and one formal epoch
.\.venv\Scripts\python.exe -m mpc_cnn.train --smoke --output-dir runs/smoke

# Full Dataset A protocol: nine long-term and nine adjacent transfers
.\.venv\Scripts\python.exe -m mpc_cnn.train --protocol all --output-dir runs/full

# Run an individual long-term task with the paper's per-task settings
.\.venv\Scripts\python.exe -m mpc_cnn.train --protocol long --source 1 --target 10

# Source-only CNN ablation (paper Tables 4–5)
.\.venv\Scripts\python.exe -m mpc_cnn.train --model cnn --protocol all --output-dir runs/cnn

# Resume a previously interrupted full run
.\.venv\Scripts\python.exe -m mpc_cnn.train --protocol all --output-dir runs/full --skip-existing

# Inspect predictions and class confusion in a saved checkpoint
.\.venv\Scripts\python.exe -m mpc_cnn.diagnose runs/full/short_1_to_2_mpc.pt
```

The default formal training epoch count, learning rate, batch size, and center-loss weight come from Supporting Information Tables S2 and S3. `--epochs`, `--pretrain-epochs`, `--batch-size`, and `--learning-rate` allow explicit overrides. Target labels do not enter gradients, MMD weights, or scaling. The runner reports final accuracy. The later exploratory study has examined target scores when choosing preprocessing and recipes, so its results are target-informed.

Results are saved as `results.csv`, per-task JSON histories, and model `.pt` checkpoints. The CSV includes the paper's accuracy and the gap in percentage points. The long-term protocol is Batch 1 → Batches 2–10. The short-term protocol is Batch *i* → Batch *i+1*, *i*=1…9.

Use `--scaling pooled` for a separate sensitivity run that estimates z-score statistics from source and unlabeled target features. The default is `source` because the paper does not state the normalization population.

Use `--architecture legacy` only to reproduce the original run's activation layout. The default is `figure3`; do not combine checkpoints or results from the two layouts.

## Reproduction limits

Figure 4(a) shows approximately 600 pretraining epochs for the Batch 1/2 example, while this implementation defaults to 100. The paper does not fully specify a per-task pretraining schedule, Gaussian bandwidth, dropout rate, L2 coefficient, learning-rate decay schedule, center update rate, or the population used for z-score statistics. The code also differs from the center-loss reduction written in Eq. 8 and compares train-mode source features with eval-mode target features for formal MMD weights. See the [fidelity audit](paper_fidelity_audit.md) and [reproduction notes](reproduction_note.md). Passing execution tests does not establish paper fidelity or successful learning.

## Revised experiments (GPT-6 Sol implementation, GPT-6 Astra review)

`mpc_cnn/revised.py` provides controlled alternatives for MMD mode/bandwidth, ensemble gradients, center reduction, preprocessing, and classifier activation. The original runner remains available; its historical checkpoints were lost. A fresh legacy 1→2 run in the restored GPU environment is saved separately in `runs/recovered_legacy_1_to_2/`. See the [experiment plan](experiment_plan_20260929.md) for explicit changes to the paper and the target-informed selection disclosure.

The separate `.venv_gpu` environment uses Python 3.12.4, PyTorch 2.7.1+cu126 and NumPy 1.26.4. CUDA forward/backward was verified on the local RTX 2060 (6 GB). Recreate it with:

```powershell
python -m venv .venv_gpu
.\.venv_gpu\Scripts\python.exe -m pip install -r requirements_mpc_gpu.lock.txt
.\.venv_gpu\Scripts\python.exe -m unittest discover -s tests -v

# A reduced-budget exploratory pilot: 3 representative tasks
.\.venv_gpu\Scripts\python.exe scripts/run_revised_suite.py --recipe configs/pilot_signedlog_domain.json --tasks representative --device cuda --output runs/revised/pilot_signedlog_domain

# Completed 600+800 suite; compatible checkpoints are reused
.\.venv_gpu\Scripts\python.exe scripts/run_revised_suite.py --recipe configs/full_signedlog_domain.json --tasks all --device cuda --output runs/revised/full_signedlog_domain_resume3

# Completed source z-score + row-L2 50+150 suite
.\.venv_gpu\Scripts\python.exe scripts/run_revised_suite.py --recipe configs/pilot_source_row_l2.json --tasks all --device cuda --output runs/revised/pilot_row_l2_full_suite_resume

# Recompute source and target predictions from every saved checkpoint
.\.venv_gpu\Scripts\python.exe scripts/audit_saved_suite.py runs/revised/full_signedlog_domain_resume3 --output runs/revised/full_signedlog_domain_resume3/audit.json

# Literal-equation sensitivity on one task (600 pretraining + 800 formal epochs)
.\.venv_gpu\Scripts\python.exe scripts/run_revised_suite.py --recipe configs/equation_sensitivity.json --pairs 1:2 --device cuda --output runs/revised/equation_sensitivity
```

Each task saves its configuration, preprocessing, environment, code/data hashes, losses, final metrics and model. Compatible completed tasks can be reused; incompatible or incomplete output directories are rejected. Suite summaries report each protocol's task count so a partial run cannot be mistaken for the nine-task macro. The shared 1->2 task is trained once and counted once in each protocol. See the [trained-artifact audit](trained_artifact_audit.md) for CPU checkpoint replay, hash checks, and completed-run status. The [Thai discrepancy report](WHY_FIRST_RUN_DIFFERS.md) separates demonstrated causes and controlled sensitivity tests from unresolved paper settings.

Three complete fixed-recipe studies are available: [signed-log 50+150](runs/revised/pilot_signedlog_full_suite/comparison.md) scores 66.85% long / 70.06% short; [row-L2 50+150](runs/revised/pilot_row_l2_full_suite_resume/comparison.md) scores 62.72% / 76.69%; [signed-log 600+800](runs/revised/full_signedlog_domain_resume3/comparison.md) scores 65.26% / 66.07%. All three checkpoint suites passed independent replay 17/17. Paper reference macros are 76.06% / 82.07%. These exploratory recipes differ from the written method and were evaluated after target-informed choices.

After the earlier workspace deletion, a verified snapshot is kept outside Downloads. Refresh it with `python scripts/backup_reproduction.py C:\Users\ASUS\Documents\mpc-cnn-progress-20261010.zip`; the archive excludes virtual environments and incomplete task folders.

## Accuracy ablation and repository contents

The [controlled center-loss ablation](ACCURACY_ABLATION_2026-10-10.md) identifies a major loss of transfer accuracy on Batch 4→5: center coefficient 0.02 gives 56.85%, 0.002 gives 67.51%, and zero gives 80.20%, with the same pretrained checkpoint. The [full 600+800 task table](runs/revised/full_signedlog_domain_resume3/comparison.md) and compact `runs/` JSON/CSV reports are included here. The Supporting Information sets `w=2` for Batch 4→5; applying that value to the current mean-center implementation yields 34.01%. This is a controlled diagnostic, not a complete paper reproduction.

GitHub contains source, configs, dataset, reports, and compact metrics. Model checkpoints (`*.pt`), epoch histories (`*.jsonl`), local Python environments, and the paper's PDF are omitted to keep the repository small. Recreate a checkpoint with the commands above before running `scripts/audit_saved_suite.py`; saved JSON reports include the original checkpoint hashes and runtime provenance.
