# ADR 0004: Q2 (classes 0-4) corrects the label shift with the published class mix

- **Status:** Accepted (decided before the Q2 run)
- **Date:** 2026-11-09
- **Deciders:** Daneesh Walika (choice among four options), Claude Code (implementation)

## Context

The training cut-offs (`k5`, 117,031 rows) contain the positive classes 1-4 as
48.6 / 24.2 / 11.6 / 15.6 % of the positives; the official validation has
16 / 14 / 30 / 76 trucks, i.e. 11.8 / 10.3 / 22.1 / 55.9 % (class 0 is 97.2 % vs. 97.3 %). A multi-class
model learns the class prior of its training rows, so its probabilities for classes 3-4 are too low for
the validation mix, and the cost matrix punishes exactly the miss of class 4 (cost 500) the most.
Details: `docs/problem_framing.md` ("open point for 07.11") and `reports/class_mix_check_q1.csv`.

## Options considered

1. **Prior-shift correction with the published mix** (chosen).
2. No correction, report the shift.
3. Class weights in training (importance weighting).
4. Label-free EM / black-box shift estimate of the validation prior.

## Decision

Option 1: the calibrated class probabilities of the multi-class model are multiplied by
`pi_target(c) / pi_train(c)` and renormalised; the class is the one with the lowest **expected cost**
under the 5 x 5 cost matrix. `pi_train` is counted on the training rows the model is fitted on.
`pi_target` is the class mix **published in the dataset paper** for the validation split
(4,910 / 16 / 14 / 30 / 76, stored in `configs/config.yaml`), an aggregate number, never a per-row label
used for fitting. The test counts are not used.

## Rules fixed before the run

- Nothing is tuned on the validation: hyper-parameters are the Q1 winner's (not re-tuned), the decision
  rule follows from the cost matrix and the two priors.
- The corrected model is reported **next to** the uncorrected one (same model, expected-cost decision
  with the training prior), and next to the constant predictors (always class k), which are the
  baselines. The best constant predictor, not only "always class 0", is the bar to beat.
- The paired bootstrap of the validation trucks (500 resamples, seed 42) gives the spread of the cost
  difference. A saving smaller than that spread is reported as "not distinguishable from noise".
- The result is reported whatever it is. The test split is not touched.

## Consequences and caveats

- The correction assumes `p(x | class)` is the same in training and validation (label shift). The
  class-mix check showed that the validation classes may come from another reference point than the
  one used for the training labels, so the assumption is only partly supported. The correction may
  therefore help less than the arithmetic suggests.
- The validation class counts are used for the correction and the cost is measured on the same
  validation trucks: the corrected cost is optimistic by construction (a calibration-style use of
  validation information). It is stated next to every number. The honest out-of-sample number would be
  the test split, which is used once at the very end and only after approval.
- Multi-class probabilities of the boosted model are checked for calibration on out-of-fold training
  predictions; the correction is only as good as these probabilities.
