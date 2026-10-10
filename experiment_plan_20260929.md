# Revised MPC-CNN experiment plan, 2026-09-29

Implementation: GPT-6 Sol. Independent code and scientific review: GPT-6 Astra. The completed runner is reviewed before substantive training.

## Scope and disclosure

Dataset A, same 10 supplied batches and Figure 3 convolution/FC dimensions. This project has already examined target scores. Any selected recipe and its final suite are **exploratory, target-informed modified MPC-CNN results**, not an untouched reproduction of the paper. Target labels are never used in gradients, preprocessing fits, center updates, or checkpoint selection. Every reported checkpoint is the final epoch.

## Bounded first stage

Three representative tasks: 1->2, 1->6, 4->5. Three predefined preprocessing candidates: domain z-score, signed-log plus domain z-score, source z-score plus row L2. The diagnostic SVM screen motivated these candidates; SVM accuracy is not MPC-CNN accuracy.

All three candidates use the same exploratory core: linear classifier heads; 50 epochs of source CE plus 0.1*MMD pretraining; 150 formal epochs; SI learning rate and batch size; mean center loss at 0.01 times SI weight, source-mean centers, 50-epoch warmup; dropout 0.1; Kaiming initialization; cosine schedule; same-mode MMD; differentiable median bandwidth; detached normalized ensemble weights. These are reduced-budget pilots and contain explicit departures from the paper.

Continuation requires finite losses and healthy source learning/features. Target accuracy may guide the exploratory candidate choice, which is disclosed. A selected recipe will be frozen before a full long/short suite. No per-task best-target checkpoint or per-task winning recipe will be assembled into a benchmark.

## Equation sensitivity

Run 1->2 with 600 MMD-only pretraining epochs and 800 formal epochs, SI settings, sum center loss, zero centers, ReLU heads, source z-score, same-mode MMD, and raw differentiable reciprocal-sigmoid weights. This tests one literal reading of Eqs. 7-12. Eq. 12 calls the output a probability while Eq. 7 omits normalization, so this is an equation sensitivity, not a uniquely established implementation of the authors' code. Differentiable raw weights add a domain-distance gradient through the weight-sum term. Median bandwidth, beta, initialization, dropout and normalization population are still assumptions.

## Evidence and limitations

Each run stores expanded configuration, environment, data/code hashes, losses, source/target predictions, feature diagnostics, checkpoint and final metrics. The suite counts 1->2 in each of the two protocols while training it only once. Macro accuracy gives each of nine tasks equal weight. GPU and CPU results are not presented as a controlled single-factor comparison with historical runs.

Paper references: long macro 76.06%, short macro 82.07%, 1->2 85.06%. Matching a single task does not reproduce either macro. Several seeds are needed to establish robustness beyond the initial seed-42 experiments.
