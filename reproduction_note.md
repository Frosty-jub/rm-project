# MPC-CNN reproduction note

## Sources and scope

- Main paper: Jiang et al., *Gas Sensor Drift Compensation Using Semi-Supervised Ensemble Classifiers with Multi-Level Features and Center Loss*, *ACS Sensors* **10** (2025), 2906–2918, DOI [10.1021/acssensors.4c03655](https://doi.org/10.1021/acssensors.4c03655). The user-provided PDF is outside this repository.
- Supporting Information: [ACS Figshare file](https://acs.figshare.com/articles/journal_contribution/Gas_Sensor_Drift_Compensation_Using_Semi-Supervised_Ensemble_Classifiers_with_Multi-Level_Features_and_Center_Loss/28752293), Tables S2 and S3.
- Dataset A only. Dataset B is a separate 14-feature regression dataset and is not present in this workspace.

## Implemented protocol

| Step | Implementation | Source |
|---|---|---|
| Data | 10 LIBSVM-like files, 13,910 samples, 128 features, six labels; reshape to 16 × 8 | Main paper Table 2 |
| Backbone | Conv2d 8/16 channels, 2 × 2, padding 1; 2 × 2 max pool; Conv2d 32/16 channels, 3 × 3, padding 1; FC 128/64/32 | Main paper Figure 3 |
| MMD pretraining | Gaussian-kernel empirical MMD at each of three FC levels; sum as loss | Main paper Eqs. 3–4, 11 |
| Ensemble | Reciprocal sigmoid of each MMD distance; three classifier probabilities combined by weights | Main paper Eqs. 5–7 |
| Formal training | Labeled source cross-entropy + weighted center loss; unlabeled target used for MMD weights | Main paper Eqs. 8–12 |
| Center update | Three 6-class centers updated from mini-batches | Main paper Eq. 10, Figure 3 |
| Long-term | Batch 1 → each Batch 2–10 | Main paper Table 3 |
| Short-term | Batch *i* → Batch *i+1*, *i*=1…9 | Main paper Table 3 |
| Formal settings | Per-task 800 epochs, LR, batch size and center weight | Supporting Information Tables S2–S3 |

## Necessary choices where the paper is incomplete

See the [2026-09-29 fidelity audit](paper_fidelity_audit.md) for verified differences, checkpoint probes, and corrections to earlier explanations.

- The default `figure3` architecture follows the visible activation blocks in Figure 3: no activation after the fourth convolution, and it interprets the classifier label ReLU/FC as ReLU before FC. The slash notation does not fully specify executable ordering. Older results used `legacy`, which had LeakyReLU after that convolution and omitted the classifier ReLU. Saved older checkpoints are read with `legacy` automatically. The old and new results must not be pooled.

- Standardization fits the mean and standard deviation on the source batch only. The paper states z-score scaling but does not specify the fitted population; source-only fitting avoids target information leaking into the preprocessing estimates.
- `--scaling pooled` is provided as a prespecified sensitivity analysis using source and unlabeled target features; it never uses target labels. Its results must be reported separately from the source-only setting.
- The Gaussian bandwidth uses the median nonzero squared pairwise distance of the current source and target embedding batches. The paper does not give a bandwidth. The biased empirical MMD estimator is used so that the distance stays nonnegative.
- Eq. 7 does not normalize the sum of weights, although cross-entropy needs valid probabilities. We divide the three reciprocal-sigmoid weights by their sum.
- MMD weights are detached during formal training. This prevents supervised loss from changing the domain-distance estimate through the weight path; the paper does not specify this derivative. Source features use train mode while target features use eval mode; the audit demonstrates a nonzero MMD even for identical inputs due to this difference.
- Pretraining defaults to 100 epochs, but Figure 4(a) visibly shows the Dataset A Batch 1/2 example through approximately 600 epochs. Thus the default does not reproduce that illustrated run; the paper does not provide a per-task pretraining schedule. The earlier assertion that no pretraining duration was given overlooked this figure.
- Dropout 0.5, center update rate β=0.5, zero-initialized centers, Adam weight decay 1e-4, and cosine learning-rate decay are implementation choices. Formal epoch/LR/batch/weight settings follow the SI tables.
- Training uses batchwise MMD weights. At evaluation, weights are calculated from at most 256 unlabeled source and target samples chosen with the run seed and held constant across the target batch. The paper does not specify evaluation batch composition.
- Center loss and cross-entropy are each averaged over a mini-batch. This differs from the written objective: Eq. 8 sums center loss while Eq. 12 averages cross-entropy. At batch size 90, the implemented center penalty is 1/90 of the literal Eq. 8 penalty for the same loss weight. Increasing it is not a demonstrated fix for collapse.

## Interpreting the reported gap

The raw class IDs in Batch 1 have counts `[90, 98, 83, 30, 70, 74]`, while the paper's Table 2 presents the counts in a different gas-name order. Counts alone are insufficient to infer an ID-to-gas mapping. The [official UCI Dataset 224 page](https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset) explicitly documents the raw file coding as 1 Ethanol, 2 Ethylene, 3 Ammonia, 4 Acetaldehyde, 5 Acetone, 6 Toluene. Older diagnostic prose inferred a different mapping from counts and should not be used for gas-name claims. All model results in this repository use numeric labels; applying the same label permutation to every batch would leave accuracy unchanged.

The paper reports mean MPC-CNN accuracies of **76.06% long-term** and **82.07% short-term** in Tables 4–5. The 18 individual reference values are embedded in `mpc_cnn/train.py` and written alongside each measured value. A result gap is evidence of a difference in this implementation, training budget, seed, preprocessing, or unspecified settings; it is not evidence that one choice alone caused the gap. Runs with reduced epochs must be labeled as pilot runs.
