# Independent Astra review: MPC-CNN revision

Date: 2026-09-29. Coding owner: GPT-6 Sol. Independent reviewer: GPT-6 Astra. Review role was read-only for implementation; only this report was written by the reviewer. Bounded CPU probes and temporary test artifacts were used; no substantive training was performed by the reviewer.

**Verdict: ready for the planned CUDA pilots. No outstanding implementation correctness blocker was found in the reviewed revision. This is approval to test a transparent experimental implementation, not a claim that the paper's results have been reproduced.**

## Evidence and scope

Reviewed the local main-paper extraction, Supporting Information, prior fidelity audit and reproduction notes/results, original `model.py`/`train.py`/`data.py`, saved historical checkpoint audit, preprocessing screen, completed `revised.py`, suite runner, tests, and the four predefined experiment configurations. The experiment plan and configurations agree on the exploratory changes and target-informed selection disclosure.

The reviewer independently ran `python -m unittest tests.test_revised -v`: **7/7 passed** after the final runtime fix. Sol and the parent separately report **12/12 project tests passed**; the parent's pass also used the CUDA environment. Independent reviewer probes used Python 3.12.4, PyTorch 2.13.0+cpu, NumPy 1.26.4. These checks do not establish CUDA numerical equivalence or trained accuracy.

Reviewed SHA-256 snapshot:

| File | SHA-256 |
|---|---|
| `mpc_cnn/revised.py` | `ede6262ca2833653a270561b407741abaa8d65ba2ea46a916ac2f1ead40ac8fb` |
| `scripts/run_revised_suite.py` | `d2cb4fe3c6362f7875416011f91972ec0f742f2a97eaecc6ae15b1a58d088b95` |
| `tests/test_revised.py` | `f3585552be88f9fd5c41cf786e2a61c4523fe51a933bfe70d438955f32e5a7a8` |

## What is proven about the historical discrepancy

1. **Loss reduction differs from the written equations.** Eq. 8 sums center loss over the minibatch, while Eq. 12 averages classification loss. The old code averaged both. For 90 samples and the same coefficient, its center penalty was 1/90 of the literal sum. Increasing that penalty is a fidelity correction, not a demonstrated remedy for collapse.
2. **The old pretraining budget differs from Figure 4(a).** The illustrated Batch 1/2 pretraining run reaches approximately 600 epochs; the old default used 100. The paper does not provide a separate pretraining schedule for every task. SI Tables S2/S3 specify 800 formal epochs and the task settings.
3. **The completed Figure 3 checkpoint has lost source discrimination.** The historical checkpoint audit reproduces 22.02% source and 26.85% target accuracy with all predictions in raw class 2. Classifier levels 2/3 receive entirely zero source vectors after ReLU, and their task gradients with respect to those features vanish in the evaluation probe. This demonstrates collapse at that checkpoint; it does not by itself isolate the training cause.
4. **Mixed train/eval feature modes distort MMD.** Historical identical-input probes give roughly 0.7 MMD when the two domains use different modes. Same-mode evaluation reduces this to floating-point residuals. The change's effect on trained accuracy still requires experiments.

The old complete 18-row results use the legacy activation layout: 36.75% long and 49.13% short macro accuracy, versus 76.06% and 82.07% in the paper. They must not be presented as results from the revised runner or the Figure 3 layout.

The classifier's ReLU/FC ordering, z-score fitting population, bandwidth, dropout rate, center initialization/update rate, precise learning-rate schedule, MMD evaluation sampling, and gradient policy through weights remain incompletely specified. The absence of stop-gradient in the paper does not prove which executable derivative its authors used.

## A new bounded gradient finding

The adaptive median bandwidth deserves a controlled sensitivity. When source and target features are jointly multiplied by a scalar, Gaussian MMD with a recomputed median squared-distance bandwidth is scale invariant above its numerical floor. Detaching that bandwidth computes a surrogate derivative that need not preserve this invariance.

Independent probe: seed 73, source shape 16x8, target shape 17x8 shifted by 0.7, global scale 1.0. Both adaptive modes give loss 0.584693. Detached median gives radial derivative **+0.120144**, while finite differences are approximately zero (-0.0000298). Differentiable median gives radial derivative **-0.0000000447**. The old `gaussian_mmd` gives the same +0.120144 surrogate derivative. A fixed-bandwidth control agrees with finite differences.

Thus the detached choice can encourage shrinking feature scale even where its recomputed forward loss is unchanged. This is evidence of a mathematical mechanism, **not proof that it caused the historical collapse**. The revised configuration records the choice explicitly, and the predefined pilots use the differentiable median.

## Completed implementation audit

| Area | Finding |
|---|---|
| Architecture | Figure 3 convolution and FC dimensions are retained. Linear/leaky/ReLU heads are explicit recipe choices. Linear heads are exploratory. |
| Pretraining | MMD-only and source CE plus MMD are distinct options. The latter explicitly changes the paper's entirely unsupervised pretraining. Both domains share a concatenated BN pass; dropout is disabled for this pass. These mode choices are recorded in code provenance and are implementation assumptions. |
| Formal objective | Sum/mean center reduction, SI multiplier, warmup, source/zero centers and optional normalized center features are explicit. With multiplier 1, sum reduction and no warmup/feature normalization, the minibatch center expression matches Eq. 8. Center updates retain Eq. 10's denominator. |
| Ensemble gradients | Detached and differentiable policies operate as named. Independent probes confirm finite nonzero source and target gradients through differentiable weights. Same-mode MMD preserves BN buffers and restores every module's prior mode. |
| Target labels | Read only for metrics and provenance. A synthetic CE+MMD pretrain plus two formal epochs with dropout, nonzero sum-center loss and differentiable weights produced bitwise identical model state and centers after permuting target labels. This is stronger than checking an inference-only path, while remaining a bounded test. |
| Replay | Saved source/target preprocessing statistics, recipe, model state and fixed evaluation weights are sufficient for inference. All 12 combinations of source/pooled/domain scaling, signed-log on/off and row-L2 on/off replayed float32 transforms exactly. A synthetic trained checkpoint replayed probabilities bitwise exactly. |
| Pretrain reuse | Separate formal shuffle/dropout seeds make reuse comparable. A cached pretrain produced bitwise identical final model state and centers in the independent probe. External checkpoint digest is recorded; incompatible recipe/data/code/runtime is rejected before output files are created. |
| Final reuse | Guards compare full configuration, code/data/array hashes, runtime identity, external pretrain digest and final checkpoint hash. Independent probes reject a changed Torch version, a CPU result requested as CUDA, and a tampered final checkpoint. |
| Aggregation | There are 17 unique transfer pairs and 18 protocol memberships; 1->2 legitimately contributes to each nine-task macro. Macro averages weight tasks equally. Suite progress distinguishes requested/completed tasks and incomplete/complete status. |

Two findings were corrected by Sol during review: float64 reconstruction of saved float32 preprocessing statistics, and missing runtime identity in cache/reuse checks. The fixes were independently checked. The reviewer did not edit implementation files.

## Fidelity and interpretation limits

Eq. 7 omits normalization, whereas Eq. 12 describes class probabilities. Raw reciprocal-sigmoid weights produce an output whose class sum exceeds one. For detached weights, dividing by the sum preserves argmax and gradients through classifier outputs. For differentiable weights, the raw objective contains an additional `-log(sum(weights))` term and therefore a different domain-distance gradient. The configured raw-weight run is appropriately called **equation sensitivity**, not a uniquely resolved faithful implementation.

The representative exploratory pilots deliberately combine linear heads, supervised CE+MMD pretraining, smaller mean-center penalty, source-mean centers, warmup, altered dropout and preprocessing. Better results from them would demonstrate improvement of a modified MPC-CNN, not establish those changes as the paper's missing settings.

Batch 4 has 161 source samples while SI batch size is 160. The runner retains the final singleton minibatch. Spatial BatchNorm2d remains defined, but literal sum-center strength changes with batch size. This batching choice is reconstructible from configuration/data and should be kept fixed in comparisons.

Prediction weights use a deterministic sample of at most 256 source and target examples. Saved weights make replay independent of reconstructing this sample, but the sampling rule is an implementation assumption. Same-domain MMD can retain roughly 0.0002-0.0003 numerical residuals in float32; exact zero should not be claimed.

Prior target scores, including the SVC screen over all tasks, have already informed preprocessing candidates. Consequently the later frozen full suite remains **target-informed exploratory evaluation**. A final-epoch policy prevents best-target-checkpoint selection but does not undo recipe-selection exposure. Domain scaling uses unlabeled target features legitimately within this transductive setup; that is distinct from using target labels to select a recipe.

## Bounded experiment recommendation

The recorded plan is suitable: three predefined preprocessing candidates on 1->2, 1->6 and 4->5, each with 50 pretraining and 150 formal epochs; continuation based on finite optimization and source/feature health, with any target-based recipe choice disclosed. Freeze one recipe before its 600/800 full suite. Run the separately labeled 600/800 equation sensitivity on 1->2. A matched single-change ablation on the selected recipe can test the suspected mechanism without attributing an aggregate gain to several changes at once.

Report final trained results separately from this implementation verdict. One seed and three representative pairs cannot establish the paper's full macro result or robustness. A full frozen suite can measure the remaining gap; repeated seeds are needed before a robustness claim.

## Trained-artifact follow-up

At the time of this implementation verdict, new substantive CUDA training had not yet been independently reviewed. This section may be extended with checkpoint replay and report consistency evidence after the parent supplies completed artifacts.
