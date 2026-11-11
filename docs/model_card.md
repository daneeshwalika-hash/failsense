# Model Card: FailSense Q1 (`failsense-q1`, alias `candidate`, version 1)

Format after Mitchell et al., "Model Cards for Model Reporting" (2019). Numbers come from the files named
in brackets; nothing here is a target that was set afterwards. Numbers of 09.11.2026, text revised 11.11.2026.
The **test split has not been used**; the final test evaluation is pending (scheduled after course review,
17.11.2026).

## 1. Model details

| | |
|---|---|
| Task | Binary classification: will **Component X** of a truck need a repair within the next **48 time steps**? |
| Model | XGBoost (`xgboost-cpu`), 300 rounds, depth 4, learning rate 0.05, subsample 0.8, column sample 0.8, no class weights; inside one scikit-learn Pipeline (median imputer, one-hot of the 8 specifications) `[reports/registry_q1.json]` |
| Inputs | 239 columns of the latest readout of one truck: `time_step`, 105 raw readout columns (8 counters, 97 histogram bins), 8 specifications, plus engineered features from the readout and the readout immediately before it: counter delta / rate (16), histogram shares (97), weighted mean bin and top-bin share (12) |
| Output | Calibrated probability (`risk`) and an alert flag (`risk >= 0.0353`) |
| Calibration | Sigmoid (Platt): `logit(risk) = -1.957 + 0.526 * logit(score)`, fitted on out-of-fold scores of a validation-like set built from **training** trucks (no validation labels) `[reports/calibration_q1_summary.json]` |
| Alert threshold | 0.0353238554, the cost-optimal threshold on the whole official validation set (primary cost mapping) |
| Registry | MLflow model `failsense-q1`, version 1, alias `candidate`; the API refuses to serve a version the calibrator was not fitted for |
| Code / date | `src/failsense/`, trained 05.11.2026 on the `k5` training sample |

## 2. Intended use

* **Intended:** ranking and flagging trucks of the SCANIA Component X fleet for a workshop check of that
  component, as **decision support** for a service planner (the alert is a suggestion, the planner decides).
* **Out of scope:** safety-critical automatic action, other components or fleets without re-evaluation,
  any claim about the physical cause of a failure (the 14 variables are anonymised; the explanations
  `POST /explain` say what the *model* uses, not what causes a failure).

## 3. Data

* **Training:** SCANIA Component X, training split, 23,549 trucks; `k5` = 5 random cut-offs per truck, 117,031
  rows, 2.8 % positive (Kharazian et al., *Scientific Data* 12, 493, 2025; dataset Version 3, DOI
  10.5878/bnh5-ka77, licence CC BY 4.0).
* **Evaluation:** the official validation split, one row per truck (its last readout): 5,046 trucks, **136
  positives (2.70 %)**. Used for the threshold and for the numbers below, never for fitting.
* **Test split:** not used for any model, threshold, calibrator or feature choice; to be evaluated once at the
  very end. (`make parquet`, `make validate` and the EDA notebook read test files only for format conversion and
  schema checks; the test labels have never been loaded for an analysis. Details: `docs/test_evaluation_plan.md`,
  section 1a.)
* The data are real fleet data with anonymised variables; no personal data.
* **Attribution (data licence).** This model, including the bundled file `models/failsense-q1/model.skops`, was
  trained on the *SCANIA Component X Dataset* (Lindgren, T., Steinert, O., Andersson Reyna, O., Kharazian, Z., &
  Magnússon, S., 2025, Version 3, Scania CV AB, https://doi.org/10.5878/bnh5-ka77), which is licensed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The model file is adapted material in the sense of
  that licence: it holds learned parameters (tree splits, imputation medians, category names), not rows of the
  data. The authors and Scania CV AB do not endorse this model. The code of the project is MIT-licensed; the
  dataset keeps its own licence.

## 4. Performance (official validation, 5,046 trucks, 136 positives)

| Metric | Value |
|---|---|
| PR-AUC | **0.065** (95 % interval 0.048-0.102); base rate 0.027 `[calibration_q1_summary.json]` |
| ROC-AUC | 0.70 |
| Total cost, SCANIA matrix, primary mapping | **28,220** (threshold cross-fitted by truck) vs. never alert 57,400 and alert every truck 34,370; saving vs. alert-every-truck 6,150, better in 5/5 folds |
| At that decision | 2,976 of 5,046 trucks alerted (59 %); recall **85.3 %** (116 of 136 failures found); precision **3.9 %** (2,860 false alarms) |
| Recall by urgency class (1 / 2 / 3 / 4) | 87.5 % / 85.7 % / 73.3 % / 89.5 % |

**How to read this honestly:** the model ranks trucks only weakly better than chance (PR-AUC 2.4x the base
rate) and the cost advantage comes from the cost matrix: a false alarm costs 7, a miss 200-500, so the
cheapest decision alerts more than half of the fleet. The saving depends on the primary cost mapping (an
alert on a failing truck is free); under the stricter reading "alert = predicted class 1" the model saves
only 148 vs. never alerting. The cost is the cross-fitted one; the in-sample cost (27,565) is optimistic.

**The served risk is about 1.6x too high on the validation:** the mean calibrated risk is 0.04275, the true repair
rate 0.02695 `[reports/calibration_q1_scores.csv]`. The calibrator was fitted on a validation-like set of training
trucks whose positives are mostly early warnings (class 1), while the validation positives are mostly close to the
failure (class 4). The alert decision is not affected (the threshold was learned on the same calibrated values), but
the number itself is: **use the risk for ranking and for the cost decision, not as an exact probability.**

Other questions on the same data (`docs/results.md`): classes 0-4 (Q2) do not resolve the urgency (88 % of
healthy trucks and all positives are put into class 4); anomaly detection on healthy trucks is close to a
random ranking (Q4); a neural network, LightGBM and random forest are not better than this XGBoost recipe.

## 5. Factors and subgroups

* The label distribution differs between the training cut-offs and the validation (positives: 49 / 24 / 12 /
  16 % in classes 1-4 vs. 12 / 10 / 22 / 56 %). The calibrated probabilities of imminent failures may be too
  low; the threshold was learned on the validation to absorb part of this.
* Truck configuration (the 8 specifications) is the dominant structure of the data (clusters, section 10 of
  `docs/results.md`) and the model uses it; no per-configuration evaluation was done.
* Performance on other fleets, other years or other operating conditions is **unknown**.

## 6. Ethical considerations and risks

* No people are scored (vehicles only). A false alarm costs a workshop visit; a miss can cost a breakdown.
* **Automation bias:** a risk of a few percent can raise an alert (threshold 3.5 %); planners must not read
  an alert as "this truck will fail". The dashboard states the threshold and the cost logic.
* The German LLM report phrases the model's factors. The first real answer of the local LLM (09.11.2026) passed
  the schema, factor and language checks and still contained invented numbers (a wrong difference, log-odds
  written as percent). Since then every percentage must be a given fact, comparisons are written by code, and a
  text that fails the checks twice is replaced by a deterministic template. A report must still be read as a
  phrasing of the model's output, not as a diagnosis.

## 7. Caveats and recommendations

* **Monitoring:** `make drift-check` compares the answered risks with the validation reference (PSI + KS). It
  **misses a x3 scaling of the counters with 200 rows** (PSI 0.075, KS p 0.012, although the alert rate rose
  from 55 % to 70 %); only x10 and above were flagged. It sees output drift only, no input drift, and no
  labels exist in production, so a loss of accuracy is invisible. Evidently was considered and rejected (ADR 0006).
* **Other limits of the same project:** the classes 0-4 model (Q2) does **not** resolve the urgency (88 % of
  healthy trucks and all failures are put into class 4); the anomaly detectors (Q4) are close to a random
  ranking; the German report was produced by a real local LLM only three times (09.11.2026), so its text quality
  is checked by rules, not by a user study.
* **Before relying on the numbers:** evaluate once on the test split, estimate real workshop costs with the
  service department (the matrix is the challenge's, not this fleet's), and re-check calibration on recent
  labelled data.
* **The risk is not an exact probability:** on average about 1.6x too high on the validation (0.04275 vs.
  0.02695, section 4); the Brier score is no better than the base rate.
* **Retraining:** `make retrain` compares a challenger with this recipe on a fixed holdout and promotes only
  with 95 % confidence (ADR 0007); a promoted version must be recalibrated (`make calibrate`) before it is
  served.
