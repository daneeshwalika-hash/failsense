# Plan: one frozen evaluation on the test splits (Q1, Q2, RUL)

**Status: plan only. Nothing here has been run, and no test file has been read.** The run needs Daneesh's
separate approval. This document says what is frozen, what still has to be frozen, and how the single run is
protected against a second look.

## 1. Principle

The test data are used **once**. Everything that could change a number is decided and stored **before** the run,
the evaluation script reads **only saved artifacts** (no training, no fitting, no threshold search, no
calibration), refuses to run if any artifact differs from its recorded hash, and refuses a second run. The
result is reported whatever it is; nothing is tuned afterwards.

## 2. What is frozen

| | Q1 (binary, SCANIA) | Q2 (classes 0-4, SCANIA) | RUL (C-MAPSS FD001) |
|---|---|---|---|
| Model | registered `failsense-q1` **version 1** (run `a40dd5b3...`), alias `candidate` | multi-class XGBoost, the Q1 recipe (`multi:softprob`), fitted on the `k5` training rows | LightGBM + rolling mean/std (windows 5, 15), parameters of `rul.lightgbm`, fitted on all 100 training engines; **baselines frozen the same way (training engines only): Ridge on the raw settings + sensors (alpha 1.0) and the mean predictor** |
| Status | **exists** | **does not exist as an artifact** (`q2_classes.run` fits it in memory) | **does not exist as an artifact** (only cross-validation was run) |
| Calibration / correction | sigmoid slope 0.5258518583, intercept -1.9573859341, clip eps 1e-6 (`calibration_q1_summary.json`, `deployment_choice`) | prior shift with `pi_train` (counted on the training rows) and `pi_target` = the class mix **published for the validation** (4,910 / 16 / 14 / 30 / 76). The test class counts are **not** used (ADR 0004) | none; the RUL cap 125 of the training target |
| Decision rule | alert if calibrated risk >= **0.0353238554** | class with the lowest expected cost under the 5 x 5 matrix | point prediction at the last cycle of each test engine |
| Cost / error | SCANIA matrix: primary mapping (false alarm 7, miss 200-500, alert on a failing truck 0) **and** the literal mapping | 5 x 5 matrix, total cost | RMSE in cycles |
| Features | the 239 registered input columns (`registry_q1.json`), built by the same shared function as training and validation (last readout per truck, deltas from the previous readout) | same 239 columns | 3 settings + 21 sensors + rolling features of the informative sensors (the 15 non-constant sensors of the training engines (frozen list)), past cycles of the same engine only |
| Seeds | fixed (42) | fixed (42) | fixed (42) |

## 3. The freeze step (must be approved and run first, it does not touch test data)

`failsense.freeze` (`make freeze`) trains **on training data only** and writes `artifacts/frozen/`
(git-ignored, hashes committed):

0. **Q1 model:** a copy of the registered model (version 1) is stored in `artifacts/frozen/`, so the evaluation
   does not depend on the MLflow registry.
1. **Q2 model:** the multi-class pipeline of `q2_classes.run`, fitted on all `k5` training rows, saved (skops /
   MLflow, same trusted types as `register`); plus `prior_train`, `prior_target`, the cost matrix.
2. **RUL models:** the LightGBM pipeline, the Ridge baseline and the mean predictor, all fitted on all training engines, saved; plus the list of informative
   sensors, the windows and the cap.
3. **Manifest** `reports/frozen_manifest.json` (committed; the frozen files themselves stay in the git-ignored
   `artifacts/frozen/`) with: the git commit (the working tree must be clean), the
   SHA-256 of every frozen file (Q1 registry artifact reference + version, calibration summary, registry
   file, Q2 and RUL models, config sections used, the cost matrix), the SHA-256 of the **five** test files (SCANIA readouts, specifications, labels; C-MAPSS `test_FD001.txt`,
   `RUL_FD001.txt`; hashed only: the bytes are not parsed), the library versions (`uv.lock` hash), and the decisions of section 5.

## 4. The evaluation script (`scripts/evaluate_test_once.py`, to be written after this plan is approved)

* Refuses to start unless (`failsense.freeze.verify_manifest`): the flag `--approved-single-run` is given, the working tree is clean and no code or
  config file (`src/`, `configs/`, `pyproject.toml`, `uv.lock`, the registry and calibration files) differs from the
  manifest commit (documentation may change), every hash matches, and `reports/test_evaluation.lock` does not exist. It creates the lock
  **before** reading any test file, so even a crash cannot be "retried with a fix" silently.
* **Q1:** test readouts -> last readout per truck -> registered model -> frozen calibrator -> frozen threshold.
  Reported: PR-AUC and ROC-AUC (95 % bootstrap interval over trucks) **with the test prevalence next to them**
  (and the lift PR-AUC / prevalence), total cost at the frozen threshold under both mappings **next to the constant
  baselines: "never alert" and "alert every truck" (the primary baseline) under both mappings**, the saving against
  each, alert rate, recall and precision, recall per class. The test cost is genuinely out-of-sample (nothing is
  cross-fitted).
* **Q2:** test last readouts -> Q2 model -> corrected and uncorrected probabilities -> expected-cost decision.
  Reported: total cost, the five constant predictors and the best constant (frozen from the validation:
  "always class 4"), macro F1, confusion matrices, paired bootstrap of the correction and of the saving vs. the
  best constant.
* **RUL:** the last cycle of each of the 100 test engines (rolling features from its past cycles) -> prediction ->
  RMSE against `RUL_FD001.txt`. **Primary: against the RUL capped at 125** (the training target); also reported
  against the raw RUL (comparable with the literature) and the asymmetric NASA score. **The frozen baselines (mean
  predictor and Ridge on the raw sensors) are scored on the same 100 last cycles and reported next to LightGBM**,
  with a paired bootstrap of the difference to Ridge (RUL compared with a baseline). 100 engines: the interval is
  wide and is reported.
* Writes `reports/test_evaluation_final.json`, one MLflow run tagged `stage = test, single run`, and the sections
  for `docs/results.md`. **No other test access exists in the repository** (a test asserts that the test paths are
  read only by this script).
* Script tests (mini fixtures only, never the real test files): refuses without the flag, with a dirty tree, with a
  changed hash, with an existing lock; the lock is created before the first read; the metrics match hand-computed
  values on a tiny example.

## 5. Decisions needed before the freeze

1. **RUL model:** freeze LightGBM + rolling features (best cross-validation, 18.1 RMSE) rather than Ridge. *Proposal: yes.*
2. **RUL reference:** capped at 125 as primary, raw as secondary. *Proposal: as written above.*
3. **Q2 target mix:** the validation mix published in the paper, not the test mix. *Proposal: yes (ADR 0004).*
4. **Q1 headline mapping:** primary mapping as the headline, literal mapping next to it. *Proposal: both, primary first.*
5. **Not evaluated on test:** the MLP, LightGBM for Q1, the anomaly detectors, the anomaly feature, the
   retrain challenger (none was selected). *Proposal: leave them out.*
6. **After the run:** report the numbers as they are, including a worse-than-validation result, in
   `docs/results.md` and the model card; no re-tuning, no second run.

## 6. What is deliberately not frozen

The test labels' class mix and the test `time_step` distribution are unknown to us and stay so; the threshold and
the Q2 correction are therefore applied as learned on the validation, and a shift between validation and test shows
up in the result instead of being corrected away.
