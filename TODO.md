# TODO

## Test gaps against https://cs231n.github.io/neural-networks-3/

- [ ] Single precision limits the gradient check to ~1e-5 relative error; tolerance is 1e-3. Tighter checks need f64, which means making Layer and Network generic over the float type.
- [ ] Every new Activation or Objective variant needs its own gradient check case; the current checks only cover RELU, IDENTITY and MSE.
- [ ] When regularization is added: gradient check the data loss with regularization off and the regularization term on its own, and add the sanity check that raising the regularization strength raises the loss.
- [ ] When dropout is added: fix the seed during gradient checks.
- [ ] Trainer::run returns no loss history, so nothing can test that the loss curve is sane (full-batch GD with a small learning rate should be non-increasing). Feature first, then test.
- [ ] The overfit test uses batch size 1, so a full epoch never goes through aggregate_batch. Add a full-batch variant (batch size 8, ~1500-2000 epochs).
- [ ] No test that run() surfaces a wrong-dimension sample as TrainerError::BadForwardPass instead of panicking.
- [ ] Updater::TBD is a placeholder whose update arm does nothing. Test or remove once it is real.

## Categorical output head (NLL / softmax)

- [ ] Max-subtraction overflow fix in both `compute` and `gradient`. Test that logits like (500, 500, 500) produce finite uniform probabilities and finite loss.
- [ ] Full MLP end-to-end categorical gradient check with NLL objective through the whole network.
- [ ] Deliberate-error gradient rejection test: confirm the checker catches an intentionally wrong gradient.
- [ ] Large-logit-shift behavior test: a shift large enough to destroy logit differences through f32 rounding. Representation limit, distinct from the mathematical invariance.
- [ ] Invalid-input policy for `evaluate` and `run`: both panic on empty datasets via public-field mutation. Write tests for the chosen contract.

## Sequence baselines

- [ ] Define category IDs, BOS token, sequence boundaries, and train/dev/test split roles before creating windowed examples.
- [ ] Hand-written conditional tables: one-observation and two-observation histories on the same eligible positions.
- [ ] Corresponding window MLPs trained and compared on those positions.
- [ ] IID / no-history control baseline.
- [ ] Jitter binning: unsigned bins, full-alphabet histogram, lag-34 report.
- [ ] Bounded window-MLP jitter dev run, or explicit compute-limit note if unfinished.

## Deferred

- [ ] Explicit `is_finite()` assertions in loss tests.
- [ ] Meaningful relative total-decrease check.
- [ ] Absolute-plus-relative gradient checking policy.
- [ ] Log-softmax simplification in `compute`. Numerical improvement only.
- [ ] Shift-invariance / zero-sum-gradient connection. Understood but not independently reproduced.
- [ ] Retrieval check: NLL gradient at p_target = 0.9 vs 0.1.
- [ ] Retrieval check: what breaks with ReLU on the last layer.
- [ ] LeCun §2 pp. 1-3.
