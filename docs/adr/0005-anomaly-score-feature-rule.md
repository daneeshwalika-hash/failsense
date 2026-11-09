# ADR 0005: Rule for the anomaly score as an extra feature (fixed before the run)

- **Status:** Accepted (decided before the Q4 run)
- **Date:** 2026-11-09
- **Deciders:** Claude Code under the autonomous working mode of 09.11 (rule only; no scope change)

## Context

Q4 asks whether a model that only knows healthy trucks can flag the trucks that are repaired later
(IsolationForest, LOF; labels are used only to evaluate). A follow-up question of the plan: does the
anomaly score help the supervised Q1 model as an extra feature? The official validation is the only
independent evaluation we have, and it holds just 136 positive trucks, so the result is noisy. Without
a rule fixed in advance, any small gain could be read as success (the same situation as ADR for
`matched` in `docs/results.md` section 7).

## Decision

- **Detectors** are fitted on **healthy training trucks only** (never repaired during the study,
  `k5` cut-offs, class 0). The repair labels are used only to evaluate the scores. Validation and test
  are never used for fitting.
- **Evaluation of the detectors:** PR-AUC and ROC-AUC of the anomaly score against "repair within 48
  steps" on the official validation (last readout per truck), recall among the 5 % most anomalous
  trucks, and a 95 % bootstrap interval (500 resamples of trucks, seed 42). The bar is the random
  ranking (PR-AUC = prevalence 0.027) and the supervised model (PR-AUC 0.065).
- **The detector used as a feature is fixed in advance:** `IsolationForest` on the full preprocessed
  inputs (not chosen by the validation result).
- **Feature leakage rule:** the training rows' scores are **cross-fitted by truck** (5 folds,
  StratifiedGroupKFold on the healthy flag, seed 42): a row is scored by a detector that never saw
  its truck. The validation rows are scored by the detector fitted on all healthy training trucks. So
  train and validation scores are both out-of-sample.
- **Adoption rule:** the extra feature is adopted only if the PR-AUC on the official validation of the
  model with the feature minus the model without it is **positive and larger than the std of its paired
  bootstrap difference** (500 resamples). Each variant is fitted with 3 seeds and the probabilities are
  averaged, so seed noise does not decide. Otherwise the registered Q1 model stays unchanged.
- The result is reported whatever it is. The test split is not touched. Nothing is re-registered.

## Consequences

- A weak or negative result is a valid outcome and is documented as such.
- If the feature were adopted, the API would also have to serve the detector; that is not done in this
  step, so no registered model changes today.
