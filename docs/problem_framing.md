# Problem Framing — FailSense

> Status: **v1.8 (06.11.2026, `matched` vs. `k5` on the official validation: keep `k5`)**: `matched` has the
> better PR-AUC (0.0913 vs. 0.0653, difference +0.0260 > spread 0.0114) but its cost saving (2,165) is
> below its spread (3,285), so the pre-set rule keeps `k5`; "none vs. balanced" closed (no class weights).
> v1.7 (06.11.2026, SHAP and permutation importance of the candidate, §6)**: grouped
> permutation finds only the histogram groups 459 and 291 clearly above the spread; grouped SHAP ranks
> 397, 459, 666, 167, 291; 397 is #1 by SHAP and gain but its shuffle does not hurt the validation PR-AUC.
> v1.6 (06.11.2026, open point: label shift for Q2, decision on 07.11)**.
> v1.5 (06.11.2026, class-mix check: the calibrated evaluation set's class mix is an
> artefact of rank matching; open point, §6, §8)**.
> v1.4 (06.11.2026, scope: Bosch CNC cut from v1.0, see ADR 0003). Q4 (anomaly
> detection) is evaluated on SCANIA only; the Bosch CNC part is future work.
> v1.3 (06.11.2026, calibration and cost threshold on the official validation).
> The candidate is scored once on the official validation (136 of 5,046 trucks positive): PR-AUC
> 0.065 (95 % bootstrap interval 0.048–0.102), ROC-AUC 0.70. A sigmoid calibrator fitted on
> out-of-fold scores of the validation-like set (no validation labels) cuts the Brier score from
> 0.038 to 0.026 = the base-rate predictor (0.026): the probabilities are usable but add no skill
> over the base rate. With the SCANIA costs the best policy is to alert about 59 % of the trucks
> (cross-fitted cost 28,220 vs. 34,370 for always-alert, saving 6,150 = 1.22 per truck, better in
> 5/5 folds); alerting only the riskiest 5 % costs 50,231, *more* than always-alert (§6).
> v1.2 (05.11.2026, hyperparameter search and the candidate model). Optuna on
> XGBoost and LightGBM found nothing clearly better than the untuned engineered XGBoost
> (+0.0015 ± 0.0040, 3/5 folds, calibrated set); that model is registered as `failsense-q1`
> (alias `candidate`). Calibrated set: PR-AUC 0.057, ROC-AUC 0.66, Brier 0.028 (worse than
> predicting the base rate: the probabilities are too high); alerting the riskiest 5 % of trucks
> finds 15 % of the failures with 8 % precision (about 3 × the base rate). Calibration and the
> cost threshold come on 06.11 (§6).
> v1.1 (05.11.2026, evaluation protocol fixed before tuning). The old CV scores were
> too optimistic: with a validation-like evaluation set XGBoost drops from 0.1335 to 0.053
> (calibrated) or 0.090 (literal procedure), the logistic regression from 0.110 to 0.044 /
> 0.078; the family ranking is unchanged. The validation `time_step` shift is a property of the
> data (the official validation histories are longer), not of the cutting procedure (§6, §8).
> v1.0 (05.11.2026, plan day 04.11 caught up). Tree ensembles clearly beat the
> linear reference: random forest, histogram boosting, XGBoost and LightGBM all reach about
> 0.131–0.134 PR-AUC against 0.110 (better in 5 / 5 folds), with the engineered features worth
> about +0.010 more on top. The candidates for tuning are XGBoost and LightGBM (§6).
> v0.9 (05.11.2026, plan day 03.11 caught up). kNN, Gaussian naive Bayes and
> single decision trees are compared with the logistic-regression reference in one table on
> the same folds: none beats it (best: kNN 0.106 vs. 0.110); single trees overfit and are
> far behind (§6). Boosting is added to the same table next.
> v0.8 (05.11.2026, RUL baseline added). C-MAPSS FD001 (Q3): a linear model
> halves the RMSE of the mean predictor (21.6 vs. 41.7 cycles); Ridge and Lasso are
> identical (§6, RUL baseline).
> v0.7 (05.11.2026, plan day 02.11 caught up). Regularization (L2, L1,
> ElasticNet) did not improve the linear reference: all linear variants stay at about 0.11
> PR-AUC, so the limit is bias and non-linear models come next (§6). The reference stays
> L2, C = 1. L1 shows that 80 of 196 coefficients are enough.
> v0.6 (05.11.2026, plan day 30.10 caught up). Engineered features, imbalance
> handling, outliers and the `time_step` check are measured (§3, §6, §8). Result in one line:
> `log1p` on the skewed raw features is the first change that helps the linear baseline in
> every fold; engineered features and SMOTE do not; the training cut-offs sit much earlier in
> a truck's life than the validation readouts (KS = 0.43).
> v0.5 (29.10): all SCANIA files are checked by a data contract, the
> truck specifications are part of the Pipeline, and scaling and the specifications were
> compared fold by fold: neither changes the linear baseline (see change log).
> v0.4 (28.10): `k = 5` cut-offs per truck decided by cross validation, baselines measured
> and logged in MLflow, learning curve shows that the linear baseline is limited by bias.
> v0.2 was rewritten after the dataset switch ([ADR 0001](adr/0001-dataset-switch.md)):
> SCANIA Component X is the main dataset, NASA C-MAPSS FD001 stays for RUL.
> Questions, metrics and the cost model (§1–§6) were written **before** any model was
> trained, so that success criteria cannot be chosen after seeing results; measured
> results are added later and marked with their date.
> Numbers marked *(paper)* come from the dataset publication and are verified in the EDA
> (`notebooks/01_eda_scania.ipynb`).

## 1. Business problem

When a component of a heavy-duty truck fails on the road, the truck stops, has to be towed
to a workshop and the delivery is late. This is expensive for the fleet operator.
Replacing components at fixed intervals wastes parts that are still good.

The goal of FailSense is an **early warning based on operational data** that the truck
already records, so that the component is repaired at the next planned workshop visit —
not too early and not too late — and the technician understands *why* the warning was raised.

## 2. Questions and ML task types

| # | Question | ML task | Dataset | Priority |
|---|----------|---------|---------|----------|
| 1 | Will Component X need a repair within the next 48 time steps? | Supervised — binary classification | SCANIA Component X | Must |
| 2 | How close is the repair (class 0–4)? | Supervised — multi-class classification with a cost matrix | SCANIA Component X | Should |
| 3 | How many cycles remain until failure (RUL)? | Supervised — regression on time series | NASA C-MAPSS FD001 | Must |
| 4 | Does a truck behave abnormally, even without failure examples? | Unsupervised — anomaly detection | SCANIA Component X (Must); Bosch CNC cross-machine part **moved to future work** ([ADR 0003](adr/0003-cut-bosch-cnc.md)) | Must |
| 5 | Why did the model raise a warning, and what should the technician do? | Explainable AI (SHAP) + LLM report | Model outputs | Must / Should |

### Why each task type

- **Q1 and Q2 are supervised** because labels exist: the repair records tell us when
  Component X was repaired. From them we derive, for every readout, how far away the repair
  was. Q1 asks "yes/no within 48 time steps" — the decision a workshop planner needs.
  Q2 keeps the five time windows of the SCANIA challenge, because an alarm 40 steps early
  and an alarm 3 steps early lead to different actions.
- **Q3 is regression**, because the answer is a number (cycles left), not a class.
  It uses C-MAPSS because every engine there runs until failure, so the true RUL is known
  for every cycle. In SCANIA many trucks never fail during the study (see censoring, §3).
- **Q4 is unsupervised**, because in practice most components have no or very few failure
  examples. The model learns "normal" from healthy trucks only and flags deviations.
  The repair labels are used **only to evaluate** the anomaly score, never to train it.
- **Q5 is not a prediction task.** It explains the predictions of Q1/Q2 (SHAP) and turns
  them into a short German maintenance note for the technician (local LLM).

### What one prediction means

A truck sends a **readout** at irregular time steps. Each readout contains cumulative
counters and histograms of operating conditions up to that moment.

One prediction is made **for one truck at one readout time `t`**, using only the readouts
up to `t`:

> *"Given everything this truck has reported until now, will Component X need a repair
> within the next 48 time steps?"*

This is a forecast over time, not a classification of an independent snapshot. That is
why time and vehicle are the main leakage risks (§4).

## 3. Data facts that shape the design

*(paper; row and truck counts verified exactly in the EDA on 26.10)*

| Part | Content | Size |
|---|---|---|
| `train_operational_readouts.csv` | `vehicle_id`, `time_step`, 105 feature columns | ~1.12 M rows, 23,550 trucks, ~1.2 GB |
| `train_tte.csv` | `length_of_study_time_step`, `in_study_repair` per truck | ~9.7 % of trucks repaired |
| `train_specifications.csv` | `Spec_0` … `Spec_7` (categorical) per truck | 23,550 rows |
| `validation_*` / `test_*` | readouts up to a random cut-off, specifications, **class 0–4** of the last readout | ~5,046 / ~5,045 trucks |

- **Feature types:** 8 numerical counters (cumulative, e.g. `171_0`) and 6 histogram
  variables with 10–36 bins each (e.g. `167_0` … `167_9`). Variable names are anonymised
  and values are partly scaled → **no physical feature engineering** is possible.
- **Missing values:** at most 0.86 % per feature (EDA); histograms are missing as whole
  blocks → simple imputation inside the Pipeline.
- **Specifications:** one row per truck, joined to every cut-off by `vehicle_id`
  (`src/failsense/features/build.py`, 29.10) and one-hot encoded inside the Pipeline.
  Validation/test contain categories that never occur in training →
  `OneHotEncoder(handle_unknown="ignore")` encodes them as all zeros.
- **Strong imbalance:** classes 1–4 are only about 2.7 % of the validation trucks.
  A model that always predicts class 0 has high accuracy and zero recall →
  **accuracy is not a valid metric**.
- **Censoring:** for trucks with `in_study_repair = 0`, the study ended before any repair.
  We only know that the repair happens **after** `length_of_study_time_step`, not when.
- **Validation and test are cut at a random readout** per truck; training trucks have their
  full history. Training samples must be built so that they look like this situation.

### Building the training labels (implemented in `src/failsense/data/labels.py`, 27.10)

For a truck with study length `L` and a readout at time step `t`, the distance to the end
of the study is `d = L − t`.

| Truck | Condition | Label |
|---|---|---|
| Repaired (`in_study_repair = 1`) | `d > 48` | class 0 |
| Repaired | `24 < d ≤ 48` / `12 < d ≤ 24` / `6 < d ≤ 12` / `0 ≤ d ≤ 6` | class 1 / 2 / 3 / 4 |
| Not repaired (`in_study_repair = 0`) | `d > 48` | class 0 (the repair is more than 48 steps away) |
| Not repaired | `d ≤ 48` (including `d = 48`) | **unknown (`<NA>`) → not used for training** |
| Any | `d < 0` (readout after the end of observation) | error — the pipeline stops |

Q1 label = 1 if class ≥ 1, else 0; unknown stays unknown.

**Window borders (decided 27.10).** The challenge description defines the windows
(48–24, 24–12, 12–6, 6–0) but does **not** say whether a border belongs to the upper or the
lower window. Because `time_step` is a decimal, values close to a border occur. FailSense
uses **upper bound inclusive**: class *k* if `d ≤ bound[k−1]`, class 0 only if `d > 48`,
which matches "more than 48 = class 0". The bounds `[48, 24, 12, 6]` are stored in
`configs/config.yaml` (`labels.class_bounds`), and every border value (6, 12, 24, 48) has a
unit test.

**Implementation choices.** Labels are returned as a **separate table** keyed by
`(vehicle_id, time_step)`, never as extra columns next to the features. Unknown is `<NA>`
(nullable `Int8`), not 0 or −1, so it cannot be used as a class by accident. The constant
`LABEL_COLUMNS` lists every column that carries target information; the leakage test uses it.
In the EDA, about 18 % of the training readouts are unknown.

### Training samples: random cut-offs (decided 27.10, `src/failsense/data/sampling.py`)

Validation and test give one label per truck at the end of a randomly truncated history,
while training trucks have their full history. Training on every readout would create a
train/serving skew: long histories would dominate, rows of one truck are near-copies, and the
age distribution and positive rate would differ from validation/test.

FailSense therefore draws **k random readouts per truck** (only readouts with a known label,
without replacement, seeded) and treats each one as "the moment the model is asked". Features
for a cut-off are built only from readouts up to that cut-off (§4, rule 3).

- `k` and the seed come from `configs/config.yaml` (`sampling`); `null` keeps all known
  readouts.
- **Decision (28.10): `k = 5`.** Compared with `src/failsense/models/compare_cutoffs.py` on a
  **fixed evaluation set** (one cut-off per truck, fixed folds by vehicle, §7), logistic
  regression on the 106 raw features:

  | k | training rows | validation PR-AUC (5 folds) | training time per fold |
  |---|---|---|---|
  | 1 | 23,549 | 0.1001 ± 0.0052 | 0.8 s |
  | **5** | 117,031 | **0.1051 ± 0.0065** | 8.2 s |
  | all | 919,315 | 0.1073 ± 0.0074 | 105 s |

  Compared fold by fold (same folds), `k = 5` beats `k = 1` in 4 of 5 folds (+0.005 on
  average); "all" adds only +0.002 over `k = 5`, less than the variation between folds, at
  13 times the training time. Because every model in week 2 (and the hyperparameter search)
  repeats this cost, `k = 5` is used. **Re-check with the best model on 05.11**, because a
  stronger model may use more data better (`--ks 5 10 all`).
- **`time_step` check (measured 05.11, `python -m failsense.analysis.time_step_check`).**
  `time_step` of the training cut-offs against the **last readout per truck in validation**
  (readouts only, no labels):

  | Sample | n | q05 | q25 | median | q75 | q95 | KS vs. validation |
  |---|---|---|---|---|---|---|---|
  | training cut-offs (`k = 5`) | 117,031 | 8.8 | 44.0 | 83.2 | 144.2 | 260.4 | **0.425** |
  | training, all labelled readouts | 919,315 | 11.2 | 55.4 | 108.0 | 182.8 | 288.2 | 0.318 |
  | validation, last readout per truck | 5,046 | 78.2 | 116.4 | 169.4 | 263.1 | 347.4 | — |

  (KS = largest gap between the two empirical distribution functions; 0 = identical,
  1 = separate. The p-value is 0 and says nothing here: with 100,000 rows any gap is
  "significant".) Figure: `reports/figures/time_step_check_q1.png`.

  **Conclusion: the training cut-offs do not imitate the validation situation.** The median
  validation readout is twice as late as the median cut-off (169 vs. 83), and 95 % of the
  validation trucks are seen after time step 78, where 25 % of the cut-offs lie below 44.
  Even using every labelled readout (`k = all`) only reduces KS to 0.32, so the gap is not
  caused by `k`. Plausible reasons (not yet tested): validation histories are truncated only
  after a minimum length, and the last 48 steps of censored training trucks are excluded
  (§3, labels), which removes late readouts from training. Consequence: the model is trained
  on younger trucks than it is asked about (covariate shift on every cumulative counter).
  Whether this costs score is open. **Open point (decided 05.11): the sampling is not changed
  now.** It is tested in catch-up step 7 together with `compare_cutoffs`, on a **fixed
  evaluation set per fold that imitates validation** (one late readout per truck), so the
  validation rows stay the same while the training sample (e.g. cut-offs drawn from the
  validation `time_step` distribution, or restricted to `time_step >= ~70`) is varied (§7).
  **Update (v1.1): done, and the explanation is different from the two reasons suggested
  above.** The official validation histories are simply *longer* than a uniform cut of
  training-length histories would give (§6, "Evaluation protocol"): the shift is a property of
  the data, not of the cutting procedure.
- Several cut-offs of one truck are safe because all of them stay in the same CV fold
  (`groups = vehicle_id`).

### Data contract (29.10, `src/failsense/data/schema.py`)

Every SCANIA file is checked with **pandera** before it is used
(`uv run python -m failsense.data.schema`). The contract states what valid input looks like,
so bad data stops the pipeline with a clear message instead of silently producing a worse model.

| File | Checks |
|---|---|
| Readouts (all splits) | exactly `vehicle_id`, `time_step` and the 105 feature columns (`strict`: an extra column, e.g. a label column, is an error); `vehicle_id` int64, all other columns float64 (no silent type coercion); no negative values; missing values only in features; `time_step` strictly increasing per truck; not empty |
| Specifications | one row per truck; `Spec_0` … `Spec_7` text matching `Cat<n>` — the category *list* is not fixed, because unseen categories in validation/test are valid data |
| Time-to-event | one row per truck; `length_of_study_time_step` ≥ 0; `in_study_repair` ∈ {0, 1} |
| Across files | every truck with readouts has a specification row; no truck in two splits |

**Result on the real data (29.10):** all 7 files pass (1,122,452 / 196,227 / 198,140 readout
rows, < 1 s from Parquet). Two facts are now guaranteed instead of assumed: readouts are
**sorted by `time_step` within every truck** (needed for the "previous readout" features on
30.10), and pandas 3 reads the specifications as the `str` dtype.

## 4. Data leakage — the most important risk

In SCANIA the risk is not one obvious column (as with AI4I), but **the truck and the time**.

**Rules:**
1. **Label columns are never features:** `length_of_study_time_step`, `in_study_repair`,
   the derived distance `d`, the class and the Q1 label. `vehicle_id` is never a feature.
2. **No truck in more than one split.** The official split is by truck. Inside the training
   set, cross-validation uses `StratifiedGroupKFold(n_splits=5)` with `groups = vehicle_id`.
   Readouts of the same truck are very similar; a random row split would let the model
   recognise the truck instead of learning the failure pattern.
3. **No information from the future.** Features at time `t` (for example the change of a
   counter) are computed only from readouts at time ≤ `t`.
4. All preprocessing (imputation, scaling, encoding, resampling) runs **inside** an sklearn
   `Pipeline` and therefore inside each cross-validation fold. Since 29.10 a
   `ColumnTransformer` routes numeric columns (median imputer → scaler) and specifications
   (`OneHotEncoder`) to their own steps (`src/failsense/models/pipeline.py`).
5. **Automated checks and tests.** `src/failsense/data/split.py` provides
   `assert_no_group_overlap` (stops the pipeline if a truck is in two splits) and
   `make_group_folds` (StratifiedGroupKFold by `vehicle_id`, with an overlap check per fold);
   tests in `tests/data/test_split.py`. On 27.10 the real SCANIA splits were checked:
   **0 shared trucks** (train 23,550 / validation 5,046 / test 5,045). The feature-matrix test
   (no column from `LABEL_COLUMNS`, no `vehicle_id`) follows in `tests/test_leakage.py` (31.10).
   Two further guards since 29.10: the data contract rejects any label column in the readouts
   (`strict` columns), and the numeric column selector inside the Pipeline raises if
   `vehicle_id` or a label column reaches `fit`.
6. C-MAPSS: cycles of the same engine never appear in training and validation at the same
   time → `GroupKFold` by engine id.

**Lesson kept from v0.1 (AI4I 2020):** the failure-mode columns `TWF`, `HDF`, `PWF`, `OSF`,
`RNF` defined the target. A one-line rule "failure if any mode flag is set" — no ML —
reached recall 0.973 and precision 0.948. Leakage can make a useless model look excellent.

## 5. Cost of errors

In maintenance, the errors do not cost the same: a missed failure (truck breaks down on the
road) is far more expensive than an unnecessary workshop check.

Instead of an assumed ratio, FailSense uses the **5 × 5 cost matrix published with the
SCANIA challenge** (IDA 2024). Rows = true class, columns = predicted class.
Source: the challenge description `2024_IDA_challenge_v2.pdf` (CC BY 4.0), shipped with the
dataset.

| true \\ pred | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| **0** | 0 | 7 | 8 | 9 | 10 |
| **1** | 200 | 0 | 7 | 8 | 9 |
| **2** | 300 | 200 | 0 | 7 | 8 |
| **3** | 400 | 300 | 200 | 0 | 7 |
| **4** | 500 | 400 | 300 | 200 | 0 |

Example: true 0 → predicted 1 (unnecessary early check) costs **7**; true 4 → predicted 0
(imminent failure missed) costs **500**.

`Total cost = Σ cost[true, predicted] × number of trucks`

- The full matrix is copied into `configs/config.yaml` (06.11)
  (not hardcoded) and covered by a test that checks its shape and the zero diagonal.
- **Q2** is evaluated directly with this total cost.
- **Q1** (binary) uses costs derived from the same matrix; the derivation is documented
  together with the threshold choice (06.11).
- The decision threshold is chosen to minimise total cost on the **validation** set —
  not fixed at 0.5.
- The matrix reflects the challenge, not necessarily the true costs of a specific fleet.
  In a real project the costs come from the service department.

## 6. Success metrics

| Task | Primary metric | Secondary | Reference point |
|------|----------------|-----------|-----------------|
| Q1 binary | **PR-AUC** (average precision) | Recall and precision at the cost-optimal threshold, total cost, confusion matrix | Random classifier PR-AUC ≈ prevalence (measured in EDA); `DummyClassifier`; `LogisticRegression` |
| Q2 classes 0–4 | **Total cost** (SCANIA matrix) | Macro F1, confusion matrix | `DummyClassifier` (always class 0) |
| Q3 RUL | **RMSE in cycles** | MAE | Predict-the-mean baseline; `Ridge` |
| Q4 anomaly | PR-AUC of the anomaly score vs. classes ≥ 1 | Recall in the top-k most anomalous trucks | Random ranking |
| Q5 explanation | SHAP top factors stable across folds | LLM report is valid JSON, in German, and only uses the top-5 SHAP factors | — |

**Why PR-AUC and not ROC-AUC?** With only a few percent positives, ROC-AUC looks good even
for weak models, because the many true negatives dominate. PR-AUC focuses on the positive class.

**Baselines (measured 28.10, fixed evaluation set, 5 folds by vehicle, `k = 5`).**

| Model | Validation PR-AUC | Training PR-AUC | Gap |
|---|---|---|---|
| `DummyClassifier` (prior) | 0.028 (= share of positives) | 0.028 | 0 |
| `LogisticRegression` (balanced, 106 raw features) | **0.105 ± 0.006** | 0.109 | +0.004 |

In `baseline_cv` (validation on 5 cut-offs per truck) the same model reaches ROC-AUC 0.81
but PR-AUC 0.10, which illustrates the point above.

**Diagnosis: high bias (underfitting).** The learning curve
(`reports/figures/learning_curve_q1_logreg.png`) shows a large gap with little data
(training 0.285 vs. validation 0.065 at 1,872 rows) that closes almost completely at the full
size (0.109 vs. 0.105 at 93,617 rows). Both curves meet at a **low** level, and the last
doubling of the data adds only +0.003. More data will not fix this model; better features
(30.10) and non-linear models (week 2) are the next levers.

**Preprocessing comparison (measured 29.10, `baseline_cv`, `k = 5`, same 5 folds by vehicle,
validation on the sampled cut-offs).** These experiments change the *model*, not the training
sample, so every model sees exactly the same rows and folds (§7).

| Model | Validation PR-AUC | Training PR-AUC | Gap |
|---|---|---|---|
| `DummyClassifier` (prior) | 0.0276 | 0.0276 | 0 |
| LogReg, numeric only, StandardScaler (as on 27./28.10) | **0.1014 ± 0.0047** | 0.1094 | +0.008 |
| LogReg + specifications, StandardScaler | 0.1013 ± 0.0069 | 0.1147 | +0.013 |
| LogReg + specifications, MinMaxScaler | 0.1001 ± 0.0041 | 0.1101 | +0.010 |

| Paired comparison (per fold) | Mean difference | Better in |
|---|---|---|
| + specifications vs. numeric only | −0.0000 ± 0.0047 | 3 / 5 folds |
| MinMaxScaler vs. StandardScaler | −0.0012 ± 0.0042 | 3 / 5 folds |

- The numeric-only model reproduces the 27./28.10 result exactly (0.1014, same fold scores):
  the refactoring into a `ColumnTransformer` changed nothing.
- **Specifications do not help the linear model.** The validation score is unchanged while the
  training score rises, i.e. the model learns something that does not transfer to new trucks.
  A linear model adds one effect per category; if the specifications matter, it is more likely
  through interactions (e.g. with load or age), which tree models can learn.
  **Decision:** keep them in the Pipeline and repeat the paired comparison with boosting on
  04.11; if they do not help there either, remove them (simpler model and API).
- **Scaler: no real difference** (−0.0012, far inside the fold noise of ±0.004).
  **Decision:** keep `StandardScaler` as default: it does not compress all values when one
  extreme value occurs, and tree models in week 2 need no scaling at all.
  Note: MinMax wins 3 of 5 folds but has the lower mean — the number of wins alone is not
  enough; mean and spread of the paired differences are reported together.
- The diagnosis stays **high bias**: preprocessing does not fix a model that is too simple.

**Feature engineering, imbalance and outliers (30.10 tasks, measured 05.11, `baseline_cv`,
`k = 5`, same 5 folds).** All rows are logistic regression + specifications unless stated.
`logreg_specs_standard` (class weight "balanced", raw features) is the reference.

| Model | Validation PR-AUC | Training | Gap | Mean predicted probability |
|---|---|---|---|---|
| reference: balanced, raw | 0.1013 ± 0.0069 | 0.1147 | +0.013 | 0.351 |
| + engineered features | 0.1026 ± 0.0063 | 0.1261 | +0.024 | 0.332 |
| without `time_step` | 0.1016 ± 0.0068 | 0.1149 | +0.013 | 0.351 |
| engineered, without `time_step` | 0.1028 ± 0.0057 | 0.1262 | +0.023 | 0.332 |
| class weight none | 0.1033 ± 0.0044 | 0.1292 | +0.026 | **0.028** |
| SMOTE (inside the CV pipeline) | 0.1007 ± 0.0085 | 0.1161 | +0.015 | 0.346 |
| **`log1p` on raw counters and histogram counts** | **0.1099 ± 0.0055** | 0.1254 | +0.016 | 0.331 |

| Paired comparison (per fold) | Mean difference | Better in |
|---|---|---|
| engineered vs. reference | +0.0013 ± 0.0056 | 3 / 5 folds |
| without `time_step` vs. reference | +0.0002 ± 0.0003 | 4 / 5 folds |
| engineered without `time_step` vs. engineered | +0.0002 ± 0.0008 | 3 / 5 folds |
| balanced vs. none | −0.0020 ± 0.0029 | 2 / 5 folds |
| SMOTE vs. balanced | −0.0006 ± 0.0017 | 2 / 5 folds |
| **`log1p` vs. reference** | **+0.0086 ± 0.0053** | **5 / 5 folds** |

- **Engineered features** (counter Δ and Δ/Δtime_step from the previous readout of the same
  truck, histogram shares, weighted mean bin, top-bin share with `features.top_fraction = 0.2`):
  +0.0013 is far inside the fold noise (±0.0056), while the train/validation gap doubles
  (+0.013 → +0.024). The 125 extra columns are fitted on training trucks and do not transfer.
  **Decision:** keep the functions, keep the logistic-regression reference without them;
  re-test with boosting on 04.11. Counter `171_0` decreases in 7 of 1.12 million readouts
  (the other 7 counters never); kept, not clipped.
- **`time_step`:** with and without differ by +0.0002 ± 0.0003. For the linear model it is no
  signal and no obvious selection shortcut. **Decision:** keep it until the boosting re-check
  on 04.11 (trees can use it differently), and see the distribution shift in §3.
- **Imbalance:** the three strategies are indistinguishable in PR-AUC (differences smaller than
  their own noise), but they differ completely in the **predicted probabilities**: with class
  weight "none" the mean probability is 0.028 (true rate 0.0276), i.e. the model is
  **calibrated on average**; "balanced" and SMOTE inflate every probability to about 0.35.
  PR-AUC only uses the ranking and cannot see this; the cost-based threshold and the LLM
  report need real probabilities. **Decision:** keep "balanced" as the reference (earlier
  numbers stay comparable), no SMOTE (same quality, twice the runtime, fractional values in
  one-hot columns). **"none" vs. "balanced" + calibration is decided on 06.11** with Brier
  score and reliability curve.
- **Missing values:** `SimpleImputer(strategy="median")` inside the Pipeline, fitted per fold.
  A column that is entirely missing in a training part is dropped by the imputer (the model
  gets one feature less, announced by a warning); this sklearn default is kept on purpose
  because zero-filling would feed a constant to the model. NaN at prediction time is imputed.

**Outliers (measured 05.11, `python -m failsense.analysis.outliers`, training cut-offs only;
`reports/outliers_q1_*.csv|md`, figures in `reports/figures/`).** Outlier = above Q3 + 3 × IQR
of the feature; nothing is removed and no threshold is chosen from validation scores.

- The raw counters and histogram counts are extremely right-skewed. Skewness of 342 for several
  histogram-397 bins equals √n (√117,031 = 342), the maximum possible value: **one single row
  dominates** the feature. It is truck 613 at `time_step` 356.6, histogram-397 total 3.1 × 10¹¹
  (p99 of that total: 1.6 × 10⁸; the next-largest bin value is 4,000 × smaller). In total 11 of
  117,031 cut-offs have a histogram bin above 10⁹; truck 8596 repeats the identical value
  1.17 × 10¹⁰ in four consecutive readouts (a frozen value). These rows are the likely
  **data errors**; they are too few (0.01 %) to matter for a count, but one such row inflates
  the variance of a `StandardScaler`.
- The wide tails are mostly **failure signal, not error.** Repair rate in outlier rows vs.
  normal rows for the 10 features with the largest outlier share (overall rate 0.0276):
  `167_9` 5.3 % vs. 2.3 % (×2.3), `309_0__rate` 4.9 % vs. 2.6 % (×1.9), `370_0` 4.1 % vs.
  2.5 % (×1.6), `459_16` 3.8 % vs. 2.5 % (×1.5); the weakest is `370_0__rate` (×1.1). Clipping
  these rows would delete the signal.
- Zero-heavy histogram bins have IQR = 0: the fence is undefined and they are reported as such.
- **`log1p` before the scaler** (inside the Pipeline, 105 raw counter and histogram columns;
  no threshold, no row removed, order of values unchanged) is the **first change that helps
  the linear model in every fold** (+0.0086 ± 0.0053, 5 / 5; smallest per-fold gain +0.0008,
  largest +0.0143). The gain is modest relative to the fold spread, and the gap grows slightly
  (+0.016), but it is consistent and costs nothing at prediction time. **Decision:** `log1p`
  becomes part of the logistic-regression reference from now on; boosting is invariant to it
  and does not need it.

**Regularization (measured 05.11, `python -m failsense.models.regularization_cv`,
`make regularization`, `k = 5`, same 5 folds, reference = specs + balanced + `log1p`, L2,
C = 1).** scikit-learn 1.9 no longer wants `penalty=` (deprecated since 1.8); the penalty is
set by `l1_ratio` (0 = L2, 1 = L1, in between = ElasticNet). L2 stays on `lbfgs`, so the
reference reproduces exactly (0.1099). L1 and ElasticNet use `saga`, the only solver of the
solver table that supports all three penalties; it needs scaled inputs, which the Pipeline
provides. The model sees 196 columns after preprocessing (321 with engineered features).

| Model | Validation PR-AUC | Training | Gap | Non-zero coefficients (over folds) | Runtime |
|---|---|---|---|---|---|
| L2, C = 1 (**reference**) | 0.1099 ± 0.0055 | 0.1254 | +0.016 | 195 of 196 | 21 s |
| L2, C = 0.1 | 0.1117 ± 0.0051 | 0.1253 | +0.014 | 195 | 14 s |
| L2, C = 0.01 | 0.1130 ± 0.0063 | 0.1230 | +0.010 | 195 | 12 s |
| L1, C = 0.01 | 0.1117 ± 0.0074 | 0.1186 | +0.007 | **80** (77–87) | 62 s |
| L1, C = 0.1 † | 0.1118 ± 0.0049 | 0.1251 | +0.013 | 151 (147–156) | 202 s |
| L1, C = 1 † | 0.1099 ± 0.0053 | 0.1254 | +0.016 | 189 (184–192) | 246 s |
| ElasticNet (l1_ratio 0.5), C = 0.1 | 0.1118 ± 0.0051 | 0.1254 | +0.014 | 166 (162–171) | 149 s |
| L1, C = 0.1, + engineered features † | 0.1114 ± 0.0056 | 0.1292 | +0.018 | 278 of 321 | 319 s |

† **Not converged:** `saga` reached `max_iter = 1000` in at least one fold (MLflow tag
`converged = false`, param `n_iter_max`). The scores are kept for this exploratory step but
are not fully optimised solutions; they were not re-run. All other models converged.

| Paired comparison (per fold) | Mean difference | Better in |
|---|---|---|
| L2 C = 0.01 (best L2) vs. reference | +0.0032 ± 0.0042 | 3 / 5 folds |
| L1 C = 0.1 (best L1 by mean) vs. reference | +0.0019 ± 0.0016 | 4 / 5 folds |
| L1 C = 0.01 (80 coefficients) vs. reference | +0.0018 ± 0.0077 | 3 / 5 folds |
| ElasticNet vs. reference | +0.0019 ± 0.0014 | 4 / 5 folds |
| ElasticNet vs. L1 C = 0.1 | −0.0000 ± 0.0006 | 2 / 5 folds |
| L1 + engineered vs. L1 C = 0.1 | −0.0004 ± 0.0033 | 3 / 5 folds |

- **No setting gives a proven gain.** All differences (at most +0.003) are smaller than the
  fold-to-fold spread (±0.005). "Best C" was chosen on the same CV that reports the score,
  so even these small gains are optimistic. **Decision:** the logistic-regression reference
  stays **L2, C = 1**. Real tuning (nested or separate evaluation) is not done for `saga`;
  the hyperparameter search in the next step is for the two best models, most likely boosting.
- **80 of 196 coefficients at no measurable cost.** With L1 and C = 0.01 the model keeps
  80 of 196 columns (77–87 depending on the fold) and scores 0.1117 ± 0.0074, i.e. the same
  as the reference within noise (+0.0018 ± 0.0077, 3 / 5 folds), with a smaller
  train/validation gap (+0.007 instead of +0.016). More than half of the inputs are redundant
  for the linear model.
- **Linear models saturate at about 0.11 PR-AUC.** Raw features, specifications, `log1p`,
  engineered features, SMOTE, three penalties and four values of C all end between 0.100 and
  0.113. Stronger regularization lowers the gap but not the score, which is the sign of **high
  bias** (the same diagnosis as the learning curve on 28.10): the model is too simple for the
  problem, not too flexible. **Non-linear models (week 2: trees, boosting) are the next
  step.**
- **Can L1 use the engineered features?** No: 0.1114 vs. 0.1118 without them (−0.0004 ±
  0.0033). Only one engineered feature (`397_19__share`) reaches the top 10.
- **Collinearity: coefficients mean "selected", not "important".** The top features of the
  best L1 model (`reports/regularization_q1_l1_top_features.md`) have coefficients up to
  ±2.6 on standardized inputs, and bins of the same histogram get opposite signs
  (`397_19` +2.60, `397_13` −1.27, `397_9` −1.05): the model uses differences between
  neighbouring, strongly correlated bins. Such coefficients are unstable and must not be
  read as feature importance; **SHAP is the tool for that (06.11)**. The one-hot
  specification columns (5 of the top 10, e.g. `Spec_6_Cat7`) are 0/1 and **not scaled**, so
  their coefficients are not on the same scale as the standardized numeric ones and can only
  be compared with each other.
- **Runtime:** the whole comparison takes about 17 minutes (`saga`, 5 folds in parallel),
  against about 5 minutes for `make cv`. `saga` is the slow part, another reason not to tune
  it further.

**Model families on the same folds (measured 05.11, `python -m failsense.models.compare_models_cv`,
`make compare`, `k = 5`, same 5 folds by vehicle, raw features + specifications).** One
table for all families; boosting models are added to it next (`build_models`). Reference =
logistic regression (specs, balanced, `log1p`, L2, C = 1). The reference reproduces
0.1098 here (0.10985; 0.10988 earlier: a difference in the fifth decimal from the parallel
LBFGS runs, not a change).

| Model | Validation PR-AUC | Training | Gap | Mean predicted probability | Runtime (5 folds) | Paired vs. reference |
|---|---|---|---|---|---|---|
| **logistic regression (reference)** | **0.1098 ± 0.0055** | 0.1254 | +0.016 | 0.331 | 29 s | — |
| kNN, k = 25, uniform | 0.0855 ± 0.0073 | n/a | n/a | 0.026 | 27 s | −0.0244 ± 0.0088 (0 / 5) |
| kNN, k = 101, uniform | 0.0976 ± 0.0093 | n/a | n/a | 0.026 | 28 s | −0.0123 ± 0.0120 (2 / 5) |
| kNN, k = 101, distance-weighted | 0.1057 ± 0.0113 | n/a | n/a | 0.026 | 28 s | −0.0041 ± 0.0131 (2 / 5) |
| Gaussian naive Bayes | 0.0514 ± 0.0008 | 0.0525 | +0.001 | 0.569 | 3 s | −0.0585 ± 0.0053 (0 / 5) |
| decision tree, depth 4 | 0.0721 ± 0.0038 | 0.0824 | +0.010 | 0.357 | 8 s | −0.0378 ± 0.0068 (0 / 5) |
| decision tree, depth 8 | 0.0806 ± 0.0020 | 0.1277 | +0.047 | 0.299 | 11 s | −0.0293 ± 0.0065 (0 / 5) |
| decision tree, depth unlimited | 0.0726 ± 0.0060 | 0.2573 | +0.185 | 0.207 | 14 s | −0.0372 ± 0.0041 (0 / 5) |

Set-up (why): kNN with `log1p` + `StandardScaler` inside the Pipeline (distances need
comparable scales) and k of 25 and 101 (a probability is a fraction with denominator k; small
k gives almost only 0 and 1), no class weights; its training score is **not computed**, because every
training point is its own nearest neighbour. Gaussian naive Bayes with `log1p`, no scaler (it
standardises each feature on its own), default priors = class frequency (it is generative and
has no class weights). Trees: `class_weight = "balanced"`, `min_samples_leaf = 100`, no
scaler and no `log1p`, because a split only compares a value with a threshold and any
monotone transformation gives the same tree.

- **No family beats the logistic regression.** The nearest is kNN with k = 101 and distance
  weights (−0.0041 ± 0.0131, 2 / 5 folds): within noise, with the largest fold-to-fold spread
  of all models (±0.0113). **Decision:** the reference stays logistic regression; the
  other families stay in the table as comparison points.
- **kNN improves with smoother estimates:** k 25 → 101 gives +0.012 and distance weighting
  another +0.008. Its predicted probabilities average 0.026 (true rate 0.0276): without
  class weights it is calibrated on average, like the logistic regression with weight "none"
  (§6, imbalance), while the reference (0.331) is shifted by `class_weight`.
- **Gaussian naive Bayes is the weakest (0.051, about 1.9 × the Dummy)** and predicts a mean
  probability of 0.569 against a true rate of 0.028: the independence and normality
  assumptions fail on correlated histogram bins and 0/1 columns, so its probabilities are
  extreme and useless for a threshold. Train ≈ validation: it is misspecified, not overfit.
- **Single trees overfit and stay far behind:** the gap grows with depth (+0.010, +0.047,
  +0.185) and the unlimited tree reaches 0.257 on training data and 0.073 on validation.
  Depth 8 is best (0.0806 ± 0.0020), still −0.029 under the reference in every fold. A single
  tree has high variance; **ensembles of trees (random forest, boosting) are the next step.**
- **Order on this data:** logistic regression (0.110) > kNN (0.106) > tree (0.081) > naive
  Bayes (0.051). A smooth linear boundary on `log1p` features beats local and rule-based single
  models; the signal is weak and spread over many features, which no single model captures.
- **Top impurity importances of the best tree (depth 8; averaged over the 5 fold models;
  biased, SHAP on 06.11):** `459_9` (0.317), `666_0` (0.167), `459_10` (0.098, in only 2 / 5
  folds), `time_step` (0.049), `459_15`, `837_0`, `397_3`, `397_12`, `459_16`, `397_15` (each
  0.010–0.021). Impurity importance favours features with many distinct values and is
  computed on the training data, so it is a pointer, not an explanation. Two observations:
  histogram 459 dominates (as in the outlier analysis, §6), and `time_step` is used in all
  5 folds, which feeds the selection-effect check on 04.11 (§8).
- **Runtime:** the three kNN models take 27–28 s each for all 5 folds. The one-fold timing
  predicted 5–7 s per model if the folds ran in parallel, or 24–35 s one after another; the
  actual time matches the sequential figure, so brute-force kNN on 196 dimensions did not
  profit from running the folds in parallel. The whole table takes about 2.6 minutes.

**Ensembles and boosting (measured 05.11, `make compare`, comparison set `v2`, `k = 5`, same 5
folds by vehicle, raw features + specifications).** The four new models are added to the
same table. Inputs: median imputer + one-hot specs, **no scaler, no `log1p`** (trees are
invariant to monotone transformations). The three boosting models have the **same fixed
budget, not tuned**: learning rate 0.05, 300 rounds, trees of depth 4 (XGBoost) or up to 31
leaves (HistGradientBoosting, LightGBM; `min_samples_leaf` / `min_child_samples` 100),
row/column sampling 0.8 for XGBoost and LightGBM, **no class weights**. The random forest has 300
trees, `class_weight = "balanced_subsample"`, `min_samples_leaf = 50`.

| Model | Validation PR-AUC | Training | Gap | Mean predicted probability | Runtime (5 folds) | Paired vs. reference |
|---|---|---|---|---|---|---|
| logistic regression (reference) | 0.1098 ± 0.0055 | 0.1254 | +0.016 | 0.331 | 30 s | — |
| kNN k = 101, distance-weighted (best kNN) | 0.1057 ± 0.0113 | n/a | n/a | 0.026 | 28 s | −0.0041 ± 0.0131 (2 / 5) |
| decision tree depth 8 (best tree) | 0.0806 ± 0.0020 | 0.1277 | +0.047 | 0.299 | 11 s | −0.0293 ± 0.0065 (0 / 5) |
| Gaussian naive Bayes | 0.0514 ± 0.0008 | 0.0525 | +0.001 | 0.569 | 3 s | −0.0585 ± 0.0053 (0 / 5) |
| **random forest** | **0.1339 ± 0.0079** | 0.7479 | +0.614 | 0.194 | 113 s | **+0.0240 ± 0.0093 (5 / 5)** |
| HistGradientBoosting | 0.1309 ± 0.0054 | 0.9173 | +0.786 | 0.022 | 35 s | +0.0210 ± 0.0041 (5 / 5) |
| **XGBoost** | 0.1335 ± 0.0080 | 0.5147 | +0.381 | 0.027 | 16 s | +0.0236 ± 0.0044 (5 / 5) |
| LightGBM | 0.1331 ± 0.0040 | 0.9284 | +0.795 | 0.022 | 14 s | +0.0233 ± 0.0044 (5 / 5) |

(The remaining kNN and tree variants are in `reports/compare_models_q1.csv` and §6 above.)

- **Ensembles beat the linear reference in every fold:** +0.021 to +0.024 PR-AUC, about
  twice the fold spread of the reference. This is the first family that clearly does.
  The four ensembles are **indistinguishable from each other**: random forest vs. XGBoost
  +0.0004 ± 0.0077 (2 / 5 folds), HistGradientBoosting −0.0026 ± 0.0031 (1 / 5), LightGBM
  −0.0003 ± 0.0050 (2 / 5). The ranking 0.1339 > 0.1335 > 0.1331 > 0.1309 is not a result.
- **The training scores are very high (0.51–0.93) and the gap is large, yet validation is
  best.** The ensembles fit the training trucks almost perfectly (LightGBM 0.928, 300 rounds
  with 31 leaves) and still rank unseen trucks better than anything else. The gap is the
  price of the fixed, untuned budget (no early stopping, see below); regularisation and
  fewer rounds are what step 7 tunes, with a group-aware inner split.
- **Predicted probabilities:** boosting without class weights gives a mean of 0.022–0.027
  (true rate 0.0276), calibrated on average like kNN; the random forest with balanced
  subsamples gives 0.194.
- **No early stopping, on purpose.** Early stopping needs validation data. A random split of
  the training *rows* puts readouts of the same truck on both sides, so the stopping point
  would be chosen on trucks the model has seen (leakage). scikit-learn's
  HistGradientBoosting switches early stopping **on by default** for more than 10,000 rows
  (`early_stopping="auto"`) with such a random split; it is turned off explicitly and a test
  guards it. Tuning the number of rounds with a group-aware inner split is step 7.
- **Parallelism, one level:** the 5 folds run in parallel and every model gets
  `cores // folds = 3` threads (XGBoost, LightGBM, forest explicitly; HistGradientBoosting
  has no `n_jobs`, joblib's worker processes limit its OpenMP threads to the same share).
  The runs scale as expected (e.g. XGBoost 16 s for 5 folds against a one-fold estimate of
  27 s if run one after another); there was no oversubscription. The whole table takes
  6 min 54 s.
- **Top 10 gain-based importances of XGBoost** (share of the total gain, mean over the fold
  models; biased, SHAP on 06.11): `666_0` (0.073), `459_9` (0.051), `459_10` (0.042),
  `158_6` (0.031), `397_27` (0.014), `Spec_7_Cat4` (0.013), `Spec_1_Cat1` (0.011), `167_5`,
  `459_11`, `Spec_4_Cat0` (about 0.010). Histogram 459 and counter `666_0` lead as in the
  single tree and in the outlier analysis; three of the ten are one-hot specification
  categories. `time_step` is not in the top 10. Gain importance favours features used early
  and with many distinct values and is computed on training data: a pointer, not an
  explanation.

**Re-checks of the best boosting model (XGBoost) on the same folds** (paired against the
unchanged XGBoost; positive = better):

| Variant | Validation PR-AUC | Mean predicted probability | Paired vs. XGBoost | Paired vs. reference |
|---|---|---|---|---|
| XGBoost (base) | 0.1335 ± 0.0080 | 0.027 | — | +0.0236 ± 0.0044 (5 / 5) |
| without the specifications | 0.1266 ± 0.0041 | 0.027 | −0.0069 ± 0.0041 (0 / 5) | +0.0167 ± 0.0030 (5 / 5) |
| without `time_step` | 0.1274 ± 0.0054 | 0.027 | −0.0061 ± 0.0032 (0 / 5) | +0.0175 ± 0.0035 (5 / 5) |
| **with the engineered features** | **0.1435 ± 0.0080** | 0.026 | **+0.0101 ± 0.0078 (4 / 5)** | +0.0337 ± 0.0049 (5 / 5) |
| with positive weight neg / pos = 35.2 | 0.1248 ± 0.0079 | **0.262** | −0.0087 ± 0.0081 (1 / 5) | +0.0149 ± 0.0096 (5 / 5) |

- **The specifications help boosting** (−0.0069 without, in every fold), unlike the linear
  model, where they changed nothing (§6, 29.10); as suspected on 29.10, they matter through
  interactions. **Decision: keep them.**
- **`time_step` helps boosting** (−0.0061 without it, in every fold). That is either real
  wear information or the study-design selection effect of §8 (the training cut-offs are much
  earlier than the validation readouts, §3); this comparison cannot tell which, because the
  validation parts have the same distribution as the training cut-offs. **Decision: keep it
  for now and test it on the validation-like fixed evaluation set in step 7.**
- **The engineered features are the largest effect so far (+0.0101 ± 0.0078, 4 of 5 folds,
  XGBoost 0.1435)**, but the mean is only 1.3 standard deviations of the fold differences and
  one fold is negative, so it is **likely, not proven**. They did nothing for the linear model
  (§6, 30.10): counter rates and histogram shares need interactions to be useful. **Decision:
  carry them into step 7 as an input option.**
- **A positive weight does not help** (−0.0087 ± 0.0081, 1 of 5 folds) and shifts the mean
  predicted probability from 0.027 to 0.262; ranking quality does not improve, and the
  probabilities would need re-calibration. **Decision: no class weights for boosting.** The
  neg / pos ratio of 35.2 is taken once from the training labels (the positive rate is 0.028
  in every fold), not fitted per fold.
- **Candidates for step 7 (hyperparameter search with a group-aware inner split):
  XGBoost and LightGBM** (statistically tied with each other and with the random forest;
  both train in about 15 s for 5 folds, support early stopping on a group-aware split, and
  the random forest is seven times slower and offers no round count to tune). HistGradient-
  Boosting is the weakest of the three by a small margin (−0.0026 ± 0.0031 vs. XGBoost).
  The engineered features enter the search as an option.
- **Dependencies:** XGBoost is installed as `xgboost-cpu` (no NVIDIA libraries, much smaller;
  matters for the Docker image). LightGBM needs the system OpenMP runtime `libgomp1`, which a
  fresh WSL install and `python:*-slim` images lack (`apt install libgomp1`). Saving these
  models with MLflow's default `skops` format may list new trusted types; no model is saved in
  this step.

**Evaluation protocol: the validation-like evaluation set (05.11, `compare_cutoffs --eval
validation-like`, `failsense.models.eval_set`).** Why the old CV evaluation had to be
replaced, how the replacement was found, and what it changes.

*What the dataset documentation says* (Kharazian et al. 2025, *Scientific Data* 12, 493, and
the IDA 2024 challenge document that ships with the data; quotes verbatim): vehicles are split
at random 70 / 15 / 15 into train / validation / test; "the last readout for the validation set
is selected randomly among all readouts for each vehicle"; the challenge document: "the readout
data for validation and test have been selected such that, at random, a readout for a vehicle
has been selected as the last readout". Labels are the windows "(48 to 24), (24 to 12), (12 to
6), and (6 to 0) time_step before the failure"; validation has 4,910 vehicles in class 0 and
136 in classes 1–4 (**2.70 %**), test 4,903 and 142 (2.81 %). **Not stated:** whether "random"
is uniform, how censored vehicles are labelled, and why the histories have the length they
have (the challenge document warns that "repair frequencies and readout frequencies may have
been modified").

*Why matching only the `time_step` distribution failed.* The first attempt paired the sorted
trucks with the sorted validation `time_step` values (rank matching). It reproduced the
distribution (KS 0.012) but sent every repaired truck to its own end: positive rate **9.61 %,
3.6 × the official 2.70 %**, and 99.6 % of the repaired trucks scored as positives. In the
training data `time_step` and the distance to the end of the study are tied together per
truck. The scratch analysis behind the final protocol (KS against the official last-readout
`time_step`, positive rate, median distance to the study end of the evaluation row):

| Method | KS | Positive rate | Median distance to end, repaired / censored |
|---|---|---|---|
| A. rank matching of the `time_step` marginal | **0.012** | **9.61 %** | 2.2 / 51.6 |
| B. rank matching with noise on the reach (σ = 1.0) | 0.170 | 6.21 % | 18.8 / 61.0 |
| D. density-ratio weighted pick per truck | 0.226 | 5.46 % | 41.6 / 80.6 |
| literal: one uniform random readout per truck | 0.310 | 2.61 % | — |
| C. random target ≤ reach | 0.367 | 3.54 % | 67.8 / 100.6 |
| old evaluation: uniform random readout, known labels | 0.428 | 2.80 % | 87.0 / 122.8 |
| **calibrated rank matching (final)** | **0.008** | **2.70 %** | — |
| official validation | 0 | 2.70 % | not observable |

With `time_step` alone no label-free method reaches both properties. Imitating the stated
procedure literally reproduces the positive rate (2.61 % here, 2.72 % with another seed;
official 2.70 % and 2.81 %) but not the distribution.

*The reason: the official validation histories are longer.* The paper's own counts give
196,227 / 5,046 = **38.9 readouts per validation vehicle** (test: 198,140 / 5,045 = 39.3)
against 1,122,452 / 23,550 = **47.7** for a complete training history. The validation vehicles
therefore keep **82 % of a mean training history**; a uniform random cut of training-length
histories keeps only about **51 %** (simulated: 24.2 readouts). The official last readouts match
the *end* of training histories (KS 0.012 against the training trucks' last known-label
readout) while only about 28 % of the repaired validation trucks (136 of an expected ~490, if their repair share is the training 9.65 %) lie within 48 steps of their repair.
**The `time_step` shift between training cut-offs and validation (median 83 vs. 169, §3) is a
property of the data, not of the cutting procedure.**

*The final protocol: calibrated rank matching.* Per training truck one row with a known (or,
for censored near-end readouts, class-0, as in the official labels) label, chosen so that the
evaluation set imitates **both** observable properties of the validation split: (1) the
`time_step` distribution of the official validation last readouts (rank matching of the trucks'
reach to the sorted targets) and (2) the positive rate (2.70 %): a random share of the repaired
trucks, found by bisection (33 %), may end near their repair; the others may only be evaluated
more than 48 steps before it. **Used from the official validation: the `time_step` of its readouts
and one aggregate number of its labels (136 / 5,046 = 2.70 %); no per-row labels; the test set
is untouched.** The set is fixed (seeded, depends on no model or fold), on the same truck-to-fold
assignment as `make cv` (23,549 trucks; one truck with only censored near-end readouts has no
training row and is dropped). Result: **KS 0.008, positive rate 2.70 % (635 of 23,549)**,
median `time_step` 170 (official 169). Figure: `reports/figures/eval_set_time_step_*.png`.
Caveat: the repaired-truck share varies with `time_step` in this set (Q1 16.2 %, Q2 9.5 %, Q3
9.9 %, Q4 3.0 %): late rows come mostly from censored trucks; the official relation is not
observable.

*Results: XGBoost (step-6 settings), training samplings on the fixed evaluation set (5 folds,
same truck-to-fold assignment). Paired differences are against `k5`.*

| Training sampling | Rows | Positive rate | KS of train `time_step` vs. validation | Calibrated: PR-AUC | vs. k5 | Literal: PR-AUC | vs. k5 |
|---|---|---|---|---|---|---|---|
| k5 (current) | 117,031 | 2.76 % | 0.425 | 0.0530 ± 0.0045 | — | 0.0901 ± 0.0161 | — |
| k10 | 227,866 | 2.82 % | 0.417 | 0.0508 ± 0.0053 | −0.0022 ± 0.0029 (2/5) | 0.0842 ± 0.0111 | −0.0059 ± 0.0077 (1/5) |
| k all | 919,315 | 2.80 % | 0.318 | 0.0442 ± 0.0060 | −0.0088 ± 0.0039 (0/5) | 0.0791 ± 0.0112 | −0.0110 ± 0.0139 (0/5) |
| `time_step` ≥ 70 | 107,756 | 4.72 % | 0.258 | 0.0580 ± 0.0077 | +0.0050 ± 0.0063 (4/5) | 0.0866 ± 0.0105 | −0.0034 ± 0.0085 (2/5) |
| matched to validation | 70,909 | **7.84 %** | 0.046 | 0.0737 ± 0.0129 | **+0.0207 ± 0.0138 (5/5)** | 0.0912 ± 0.0102 | **+0.0012 ± 0.0097 (4/5)** |

*Re-checks for the `matched` sampling (paired against `matched`):*

| Variant | Calibrated | Literal |
|---|---|---|
| without `time_step` | 0.0674 ± 0.0076; −0.0062 ± 0.0063 (0/5) | 0.0860 ± 0.0137; −0.0052 ± 0.0048 (1/5) |
| with engineered features | 0.0819 ± 0.0105; **+0.0082 ± 0.0120 (4/5)** | 0.1018 ± 0.0090; **+0.0106 ± 0.0075 (5/5)** |

*Old CV evaluation vs. validation-like evaluation (k5 training, same models):*

| Model | Old CV | Calibrated | Literal |
|---|---|---|---|
| XGBoost | 0.1335 ± 0.0080 | **0.0530 ± 0.0045** | 0.0901 ± 0.0161 |
| logistic regression (reference) | 0.1098 ± 0.0055 | 0.0444 ± 0.0031 | 0.0781 ± 0.0076 |

**Are the conclusions the same on both sets?**
- **Level and ranking: the old scores were too optimistic by a factor of 1.4–2.5, the ranking is
  unchanged.** XGBoost stays ahead of the logistic regression on both sets, but by +0.0086
  (calibrated) and +0.0120 (literal) instead of +0.0237. The calibrated set is the harder one
  (late rows, hard negatives from trucks that fail much later); the mean predicted probability of
  XGBoost on it is 0.055 against a rate of 0.027 (late rows have large counters), a first
  hint for the calibration on 06.11.
- **More uniform rows do not help** (`k10`, `k all` are worse or equal on both sets; `k all` is
  worse in 0/5 folds on both, −0.0088 and −0.0110). Same conclusion.
- **`matched` is the pick of the pre-set rule on both sets, but the evidence is not the same.**
  On the calibrated set it is clearly better (+0.0207, 5/5 folds); on the literal set it is
  indistinguishable from `k5` (+0.0012 ± 0.0097). `time_step` ≥ 70 helps on one (+0.0050)
  and not on the other (−0.0034). **Caution:** the `matched` training sample is built by the
  same rank matching as the calibrated evaluation set and has a positive rate of 7.84 % (75.7 %
  of the rows of repaired trucks are positives, against 28.5 % in `k5`), so part of its gain may
  come from the shared construction and not from a real advantage on the official validation.
  The literal set, which shares no mechanism with it, shows no gain. **Not a result to build
  on yet.**
- **`time_step` helps on both sets** (−0.0062 and −0.0052 without it; 0/5 and 1/5 folds), about
  one standard deviation each: the same direction and size as in the old CV (−0.0061). It stays
  in.
- **The engineered features help on both sets** (+0.0082 and +0.0106) and in the old CV
  (+0.0101): the same direction in three independent evaluations, 4/5, 5/5 and 4/5 folds. The
  best-supported result of this comparison; no single evaluation proves it (each difference is
  0.7–1.4 standard deviations of the fold differences), the agreement does.

**Decisions.** (1) The validation-like evaluation (`calibrated`, with `literal` as second
reference) is the evaluation protocol from step 7b on; old CV numbers are kept for continuity.
(2) Training sampling for step 7b: keep **`k5`** as default (cheap, no shared construction with
the evaluation); carry `matched` as a candidate that is checked once on the official validation
on 06.11. (3) Keep `time_step`. (4) Carry the engineered features into the hyperparameter
search as an option (default on). (5) The official validation split on 06.11 is used for
threshold and calibration and is the real check of this protocol.

**Hyperparameter search and the candidate model (05.11, `make tune`, `make evaluate`,
`make register`; `failsense.models.{tune,evaluate,register}`).** Evaluation protocol = the
validation-like `calibrated` set (main) with `literal` as second reference; training sample `k5`
(step 7a).

*Set-up.* Optuna 5.0.0, TPE sampler (seed 42), `MedianPruner` per fold (after every fold the running
mean is reported; 5 start-up trials, no pruning after the first fold); XGBoost and LightGBM are
tuned **separately**, no class weights. **Objective:** the mean PR-AUC over the 5 folds on the
calibrated evaluation rows. **Early stopping inside every fold** (50 rounds, at most 2,000, metric
average precision) on a truck-grouped inner split of the *training* trucks (`GroupShuffleSplit`
by `vehicle_id`, 15 % of the trucks, fixed seed per fold); the evaluation rows never take part and a
test checks that no truck is in two of the three parts. The number of rounds is therefore not
searched; the final model uses the **median of the per-fold early-stopping rounds** as a fixed
number. Trial 0 is the untuned step-6 setting with engineered features (an anchor). Budget: 40 trials
per model **with a hard cap of 12 minutes per model**; a one-trial timing (24 s XGBoost, 14 s
LightGBM) had projected 25.4 min.

| Parameter | XGBoost | LightGBM |
|---|---|---|
| `learning_rate` | 0.01–0.2 (log) | 0.01–0.2 (log) |
| tree size | `max_depth` 3–8 | `num_leaves` 8–128 (log) |
| minimum leaf size | `min_child_weight` 1–200 (log) | `min_child_samples` 20–400 (log) |
| `subsample`, `colsample_bytree` | 0.5–1.0, 0.4–1.0 | 0.5–1.0 (every round), 0.4–1.0 |
| `reg_lambda`, `reg_alpha` | 0.01–50, 0.001–10 (log) | 0.01–50, 0.001–10 (log) |
| engineered features | yes / no (categorical) | yes / no (categorical) |

*Studies (reproducible with seed 42 and the target of 40 trials; the counts depend on the time cap).*

| Model | Trials finished | Complete / pruned | Anchor (step 6) | Best trial | Best PR-AUC | Time |
|---|---|---|---|---|---|---|
| XGBoost | 24 of 40 (**time cap reached**) | 20 / 4 | 0.0550 | 17 | **0.0585** | 12.0 min |
| LightGBM | 40 of 40 | 33 / 7 | 0.0544 | 28 | **0.0592** | 9.8 min |

Best settings: XGBoost `learning_rate` 0.064, `max_depth` 4, `min_child_weight` 1.9, `subsample` 0.66,
`colsample_bytree` 0.63, `reg_lambda` 8.5, `reg_alpha` 0.003, engineered features, median 163 rounds
(per fold 124 / 260 / 344 / 100 / 163). LightGBM `learning_rate` 0.048, `num_leaves` 15,
`min_child_samples` 70, `subsample` 0.99, `colsample_bytree` 0.76, `reg_lambda` 3.0, `reg_alpha`
0.018, engineered features, median 79 rounds. Engineered features were chosen in 18 of 20 (XGBoost)
and 31 of 33 (LightGBM) complete trials. Complete trials took 34 s (XGBoost) and 16 s (LightGBM) on
average, pruned ones 12 s and 7 s. Trials table: `reports/hpo_q1_*_trials.csv`; best settings:
`reports/hpo_q1_best_params.json`; one MLflow run per study (not per fold) with the counts.

*Tuned vs. untuned (5 folds, same training sample and evaluation rows, `evaluate`).* The untuned
reference is the step-6 XGBoost with the engineered features (300 fixed rounds). The tuned models are
fitted per fold exactly as in the search (and reproduce the search scores to the last digit).

| Eval set | Model | PR-AUC | ROC-AUC | Brier | Mean predicted prob. | Top 5 %: precision / recall / F1 |
|---|---|---|---|---|---|---|
| calibrated | XGBoost tuned | 0.0585 ± 0.0093 | 0.661 | 0.0283 | 0.0535 | 8.1 % / 15.1 % / 0.105 |
| calibrated | LightGBM tuned | 0.0592 ± 0.0138 | 0.651 | 0.0277 | 0.0527 | 7.8 % / 14.6 % / 0.102 |
| calibrated | **XGBoost untuned, engineered** | 0.0570 ± 0.0062 | 0.664 | 0.0284 | 0.0533 | 8.0 % / 14.9 % / 0.104 |
| literal | XGBoost tuned | 0.0923 ± 0.0084 | 0.781 | 0.0251 | 0.0335 | 12.0 % / 23.0 % / 0.157 |
| literal | LightGBM tuned | 0.0860 ± 0.0066 | 0.779 | 0.0250 | 0.0338 | 10.8 % / 20.7 % / 0.142 |
| literal | **XGBoost untuned, engineered** | 0.0943 ± 0.0070 | 0.784 | 0.0251 | 0.0332 | 11.8 % / 22.7 % / 0.155 |

| Paired vs. untuned (per fold) | Calibrated | Literal |
|---|---|---|
| XGBoost tuned | +0.0015 ± 0.0040 (3 / 5 folds) | −0.0020 ± 0.0083 (2 / 5) |
| LightGBM tuned | +0.0022 ± 0.0100 (2 / 5) | −0.0083 ± 0.0070 (1 / 5) |

Confusion matrix when the riskiest 5 % of the trucks of every fold are alerted (pooled over the 5
folds, 23,549 trucks; **provisional, the threshold is chosen by cost on 06.11**), untuned model,
calibrated set: 94 true positives, 1,086 false alarms, 541 missed failures, 21,828 true negatives
(`reports/evaluate_q1_confusion.csv`). There is no total cost today: it needs the cost threshold.

- **Search result: nothing clearly better.** The tuned scores (0.0585, 0.0592) are the *best of N
  trials* selected on the same folds and evaluation rows they are scored on, so they are
  **optimistic**; against the untuned model the differences (+0.0015, +0.0022) are far inside the fold
  noise, and on the `literal` set both tuned models are slightly *worse*. The search found
  shallow, more regularised trees (depth 4 / 15 leaves, `reg_lambda` 3–8), i.e. the same capacity as
  the step-6 setting. **The real check is the official validation on 06.11.**
- **Caveat that favours the untuned model:** early stopping needs held-out trucks, so the tuned
  models are fitted on 85 % of the training trucks in the search and in this comparison, while the
  untuned model uses all of them. The effect is small, but it is not zero.
- **The probabilities are too high on the calibrated set.** The mean predicted probability is 0.053
  against a positive rate of 0.027, and the Brier score (0.0283) is *worse* than the Brier score of
  always predicting the base rate (0.0263): late rows have large cumulative counters and the model
  extrapolates. On the literal set (0.0251 vs. 0.0254 for the constant) it is about equal. Ranking is
  the model's strength (ROC-AUC 0.66 / 0.78), not the probabilities: **calibration is the first job on
  06.11.**
- **Alerting the riskiest 5 % finds 15 % of the failures (calibrated set) at a precision of 8 %**,
  about 3 × the positive rate of 2.7 % (4.5 × on the literal set). With a failure rate this low and a
  weak signal, every alert is mostly a false alarm; whether that is acceptable is a cost question.
- **Winner rule (fixed before the run):** a tuned model is registered only if it is better than the
  untuned engineered XGBoost on the calibrated set by more than one standard deviation of the fold
  differences **and** in at least 4 of 5 folds. Neither is, so **the simpler untuned engineered XGBoost
  is the candidate** (the reason is stored with the model).
- **The candidate model.** `failsense-q1`, version 1, alias `candidate` (the FastAPI service loads
  `models:/failsense-q1@candidate` on 07.11): one scikit-learn Pipeline (median imputer + one-hot
  specifications, then XGBoost with 300 rounds, learning rate 0.05, depth 4, no class weights) fitted
  on all 117,031 `k5` training rows, **no validation or test data**; 239 input columns (106 raw
  numeric, 125 engineered, 8 specifications; list in `reports/registry_q1.json`). Saved with `skops`
  (the XGBoost / LightGBM types are listed as trusted); the reloaded model predicts identically
  (tested on 5,089 rows, and in a test with a temporary registry). **The service must compute the 125
  engineered features from the truck's previous readouts** (§4, rule 3) before calling the model.
- **Limitations.** XGBoost's study stopped at 24 of 40 trials (time cap); an unlimited search might find
  a better trial, but the scores of the best trials are flat (top-5 trials within 0.002) and fold noise
  is ±0.009. Early stopping used the average-precision on random inner-validation rows (not the
  validation-like distribution). The `matched` and `ts70` samplings were not tuned (decision of 7a).

**RUL baseline on NASA C-MAPSS FD001 (measured 05.11, `python -m failsense.rul.baseline`,
`make rul-baseline`, MLflow experiment `failsense-rul`).** Training file only; the official
test file and `RUL_FD001.txt` have not been read (used once at the end).

- **Data:** 100 training engines, 20,631 rows, cycles 1..max without gaps, lifetimes 128–362
  cycles (median 199), no missing values; 3 operational settings + 21 sensors. The data is
  **simulated** (NASA Ames PCoE), the RUL part shows the method, not real engine performance.
- **Target:** `RUL = max cycle of the engine − cycle`, capped at **125** (`rul.cap` in the
  config): early in life the sensors show no degradation, so the true RUL there is not
  recoverable. 125 is the value commonly used for FD001, a convention and not a measurement.
  Capped target: mean 86.8, standard deviation 41.7. `max_cycle` and RUL are never features.
- **Validation:** `GroupKFold(5)` by engine (80 / 20 engines per fold), so no engine is in
  both parts; scaler inside the Pipeline, fitted per fold. Models: `DummyRegressor(mean)`,
  Ridge (alpha 0.1–100), Lasso (alpha 0.001–1); all 24 inputs (settings + sensors).

| Model | Validation RMSE | Training RMSE | Gap | Validation MAE |
|---|---|---|---|---|
| Dummy (mean) | 41.67 ± 0.00 | 41.67 | 0.00 | 36.97 |
| Ridge, alpha 100 (best on this CV) | 21.64 ± 1.38 | 21.44 | +0.19 | 17.72 |
| Lasso, alpha 0.1 (best on this CV) | 21.63 ± 1.38 | 21.44 | +0.19 | 17.72 |
| Lasso, alpha 1 | 21.70 ± 1.37 | 21.53 | +0.17 | 17.81 |
| Ridge alpha 100 + `cycle` (variant) | 20.09 ± 1.38 | 19.94 | +0.15 | 16.34 |

| Paired comparison (per fold; positive = lower RMSE) | Mean difference | Better in |
|---|---|---|
| Ridge vs. Dummy | +20.04 ± 1.38 | 5 / 5 folds |
| Lasso vs. Dummy | +20.04 ± 1.38 | 5 / 5 folds |
| Lasso vs. Ridge | +0.00 ± 0.01 | 3 / 5 folds |
| Ridge + `cycle` vs. Ridge | +1.54 ± 0.41 | 5 / 5 folds |

- **The Dummy's RMSE (41.67) is the standard deviation of the target (41.7)**, as it must be
  for a mean predictor. A linear model on the sensors halves it (about −20 cycles in every
  fold). **Target rule:** later models must beat Ridge's 21.6 on the same folds.
- **The regularization strength does not matter here.** Ridge with alpha 0.1 to 100 differs
  only in the fourth decimal: with 24 standardized inputs and about 16,500 rows per fold the
  penalty is negligible, so the "best alpha" is a tie. Alpha was chosen on the same CV that
  reports the score (optimistic); real tuning is not worth it for the linear model.
- **Constant columns in FD001 (measured):** `setting_3` and sensors 1, 5, 10, 16, 18, 19.
  **Lasso sets exactly these seven coefficients to 0** in every fold for alpha up to 0.1 and
  keeps the other 17. At alpha 1 it also switches off `setting_1`, `setting_2` and `sensor_6`
  and the error rises slightly (21.70). So the linear model uses 17 informative inputs;
  dropping the constant columns is planned for 11.11 (they carry no information and need
  no Lasso).
- **Engine age (`cycle`) adds information** (−1.54 RMSE, 5 / 5 folds) and is a legitimate
  input at prediction time, but it is **not in the main models yet**: training rows cover the
  whole life of an engine, while the official test engines are cut at a random point before
  failure, so its distribution differs between training and test (the same question as
  `time_step` in SCANIA, §8). To be decided with the time-series features on 11.11.
- **Caveat for the final test:** the cross validation scores every row of the validation
  engines; the official test evaluates only the last cycle of each truncated test engine. CV
  RMSE is therefore not comparable with published test RMSE values. The last-cycle
  evaluation (and the use of `RUL_FD001.txt`) happens once, at the end.
- **Next (11.11):** drop constant sensors, rolling mean/std features per engine (past cycles
  only), LightGBM, all with the same folds.

**Target rule.** Every new model must beat the logistic-regression baseline **on the same
folds**, compared fold by fold, not only on the mean. A numeric business target (total cost
on validation) is set on 06.11 after calibration and threshold choice, not before.
Results on real data are expected to be clearly lower than the near-perfect scores often
published for synthetic datasets.

### Calibration and cost threshold on the official validation (06.11, `make calibrate`)

First look at the **official validation** split (5,046 trucks, one last readout each, 136 positives =
2.70 %); the test split is untouched. Model: `failsense-q1@candidate` (v1). Everything is in
`reports/calibration_q1_summary.json` (tables: `calibration_q1_scores.csv`, `threshold_q1.csv`,
figures: `calibration_q1_reliability.png`, `threshold_q1_cost_curve.png`), one MLflow run
`calibrate_threshold_q1`.

**Binary cost mapping** (matrix: `configs/config.yaml` `cost`, copied from Table 1 of the challenge
description; it was *not* in the config before 06.11). Primary: alert = the weakest alarm,
predicted class 1. A false alarm costs 7 (`Cost_0_1`), a missed truck of class 1/2/3/4 costs
200/300/400/500 (`Cost_k_0`), an alert on any truck of class 1–4 costs 0. Sensitivity ("literal"):
an alerted truck of class k costs `Cost_k_1` (0/200/300/400). Theoretical threshold for a calibrated
model: `t* = FP / (FP + mean(FN - TP))`; the mean uses the class mix of the **calibrated
validation-like evaluation set** (training labels only): 607 / 26 / 2 / 0 trucks of class 1/2/3/4,
so mean FN = 205 and **t\* = 0.033**.

| Variant (probabilities on validation) | Brier | Log loss | Mean prob. | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| raw model score | 0.0384 | 0.172 | 0.112 | 0.0653 | 0.703 |
| **sigmoid**, fitted label-free on the evaluation set (primary) | **0.0261** | 0.121 | 0.043 | 0.0653 | 0.703 |
| isotonic, fitted label-free | 0.0262 | 0.122 | 0.044 | 0.0536 | 0.694 |
| sigmoid, cross-fitted on validation (reference) | 0.0258 | 0.117 | 0.027 | 0.0676 | 0.701 |
| isotonic, cross-fitted on validation (reference) | 0.0265 | 0.125 | 0.027 | 0.0533 | 0.685 |
| base-rate predictor (always 0.027) | 0.0262 | 0.124 | 0.027 | – | – |

- **Ranking:** PR-AUC 0.065 on the validation vs. 0.057 ± 0.006 (calibrated set) and 0.094 ± 0.007
  (literal set) predicted on 05.11. The point estimate is close to the calibrated estimate, and the
  protocol is not contradicted; but with 136 positives the 95 % interval (0.048–0.102) also contains
  the literal estimate, so the validation cannot rank the two protocols by itself.
- **Calibration transfers only partly.** The raw score is about 4 × too high on validation (mean
  0.112 vs. 0.027; on the evaluation set 0.053). The label-free sigmoid fixes most of it (mean
  0.043) and reaches the Brier score of the base rate, but it is still 1.6 × too high (reliability
  curve above the diagonal, `p < 0.1` zoom). The registered model is fitted on all training trucks,
  the out-of-fold scores come from fold models with 25 % less data, so the score scales differ a
  little. Isotonic offers no gain: it creates ties and has a lower PR-AUC (0.054).
  **Brier skill over the base rate is about zero for every variant**: the model ranks (ROC-AUC 0.70)
  but its probabilities do not beat "everyone has 2.7 %".
- **Calibration does not change the cost-optimal decision** when the threshold is chosen on the
  validation (a monotone map and a threshold search give the same ranking-based cut; raw and sigmoid
  have identical cross-fitted cost). It matters for the *label-free* theoretical threshold: on raw
  scores it costs 30,305, on sigmoid scores 28,612.

| Policy (validation, primary mapping) | Alerts | Total cost |
|---|---|---|
| never alert | 0 | 57,400 |
| **always alert (primary baseline)** | 5,046 | **34,370** (4,910 × 7) |
| alert the riskiest 5 % | 253 | 50,231 |
| theoretical threshold t\* = 0.033 on sigmoid probabilities | 3,235 | 28,612 |
| **cross-fitted cost-optimal threshold (sigmoid), reported** | 2,976 | **28,220** |
| in-sample optimum (optimistic, threshold tuned on the same trucks) | 3,013 | 27,565 |

Saving of the cross-fitted policy vs. always-alert: **6,150 in total = 1.22 per truck**, per fold
+1,230 ± 750 (std of 5 folds: 875, 696, 1,249, 811, 2,519), better in 5/5 folds. The mean saving
is larger than the fold spread, but the bar is low (one fold carries 41 % of the saving) and the
saving is 18 % of the always-alert cost.

At the cross-fitted decisions: 2,070 trucks are **not** alerted, 20 of the 136 positives are among
them (class 1: 2, class 2: 2, class 3: 8, class 4: 8; recall 87.5 / 85.7 / 73.3 / 89.5 %), recall
85 %, precision 3.9 %, 2,860 false alarms. The class counts of the validation (16 / 14 / 30 / 76)
are documentation only; no threshold was derived from them.

**Sensitivity (literal mapping, `alert = predicted class 1`).** Always-alert now costs 76,570 and
never-alert 57,400; the cross-fitted policy costs 57,252, i.e. 148 less than the best trivial policy:
under this reading the model is practically worthless (an alert on a class-4 truck still costs 400).

**What this means.** (1) The honest benchmark is "alert everybody": at 7 vs. 200–500 per miss, the
break-even precision is only about 3 %, barely above the 2.7 % base rate. (2) The model buys a real
but modest saving by *not* alerting the 40 % of trucks with the lowest scores while missing 15 % of
the positives. (3) The "riskiest 5 %" view used so far is a bad policy under these costs.

**Deployment choice for the service (decided 06.11, `deployment_choice` in the summary JSON).**
Label-free **sigmoid calibrator** (slope 0.526, intercept −1.957 on `logit(score)`, fitted on
out-of-fold scores of the calibrated evaluation set, no validation labels) + **threshold 0.0353
learned on the whole validation set** (primary mapping, TP = 0): alert if the calibrated probability
is at least 0.0353. The cost to quote is the **cross-fitted 28,220** (always-alert 34,370, saving
6,150, better in 5/5 folds); the in-sample 27,565 of the final threshold is optimistic. The test
split is still untouched and is used once at the very end.

> **Caveat that must travel with every result:** this positive result depends on the primary
> mapping (a), "an alert on a truck of class 1–4 costs 0". Under the stricter mapping (b) "alert =
> predicted class 1" the model saves only **148 vs. never-alert** (57,252 vs. 57,400).

Open points: the evaluation set's class mix (95.6 % class 1, 0 % class 4) is very different from
the validation's (56 % class 4); this makes the label-free t\* (0.033) higher than a
validation-mix t\* would be (0.016, documentation only) and may make the evaluation-set calibrator
too pessimistic for imminent failures. The cost matrix reflects the challenge, not a real fleet.

### Class-mix check (06.11, `make class-mix`, training data only)

Why does the calibrated evaluation set have 607 / 26 / 2 / 0 positives in classes 1–4 while the
official validation has 16 / 14 / 30 / 76 (the paper's numbers, used only as a reference)?
Nothing was tuned and no calibration was re-run; `reports/class_mix_check_q1.csv`.

| Set (one row per truck) | Positive rate | Class 1 / 2 / 3 / 4 | Share of class 4 |
|---|---|---|---|
| official validation (reference) | 2.70 % | 16 / 14 / 30 / 76 | 56 % |
| literal (one uniform readout per truck, training) | 2.61 % | 281 / 158 / 71 / 104 | 17 % |
| **calibrated rank matching (our protocol)** | 2.70 % | 607 / 26 / 2 / 0 | 0 % |
| rank matching of the `time_step` marginal only | 9.61 % | 36 / 73 / 200 / 1,953 | 86 % |

1. **Readouts are not denser near the repair.** Readouts per time unit are 0.229 / 0.227 / 0.235 /
   0.283 in classes 1–4 (class 4 about 1.2 × denser); the gap to the previous readout is 3.2–4.5 steps
   at every distance. One uniform readout per repaired truck therefore gives about 48 / 24 / 12 / 15 %
   of the positives, matching the *rate* (2.6–2.8 %) but not the official *mix*
   (χ² ≈ 180–205, 3 degrees of freedom).
2. **The label definition matches the sources**: windows 48–24, 24–12, 12–6, 6–0 before the failure
   (dataset paper, challenge PDF), upper bound inclusive, `d = length_of_study_time_step − time_step`.
   A different reference point does not explain it: shifting the failure by 12 steps still gives only
   32 % class 4 with a rate of 3.4 % (χ² 75); no shift reproduces rate and mix together.
3. **Calibrated rank matching pairs each truck with the validation `time_step` quantile of its
   reach rank.** The training reach is 41–46 steps above the target at every quantile (offset
   45 / 41 / 46 / 42 / 45 at q05 … q95), about the 48-step horizon. Repaired trucks that may be
   evaluated at their end therefore land at `d ≈ 39–44` (quartiles; median 41.8): class 1 (607 of
   635), almost never `d ≤ 12` (2 of 635); the other repaired trucks are cut just beyond 48 (class 0).
4. **So the 607 / 26 / 2 / 0 mix is an artefact of the matching**, not a property of the data.
   The official mix (78 % of the positives in the two narrowest windows) cannot come from a uniform
   random readout under this label definition, so the official last-readout selection (or label
   assignment) differs from the paper's description in a way these training data cannot reveal;
   it would need the per-row validation labels or the generator.
5. **Consequences, protocol unchanged:** the calibrated set measures mostly "class 1 early warning",
   the official validation mostly imminent failures (PR-AUC 0.065 vs. 0.057); the label-free t\*
   (0.033, mean FN 205) is too high for the official mix (0.016); the deployed threshold was learned on
   the validation, the calibrator on the evaluation set, so the calibrated probabilities of imminent
   trucks may be too low. Open point for the next time the protocol is touched.

**Open point for 07.11: Q2 (classes 0–4) faces label shift.** The training cut-offs (and the calibrated
evaluation set) are mostly class 1 among the positives, the official validation mostly class 4
(16 / 14 / 30 / 76; class-mix check above). A multi-class model trained on training cut-offs learns
the wrong class prior, and the cost matrix punishes exactly the confusion of class 4 with class 0 or 1
most. Option to decide tomorrow: **prior-shift correction** of the predicted class probabilities
with the class mix **published in the dataset paper** (aggregate numbers, like 136 / 5,046 before),
never per-row validation labels. The paper's test class counts exist as well; they are not used.
Alternatives to weigh against it: no correction (report the shift), or class weights.

### SHAP and permutation importance (06.11, `make explain`)

Model `failsense-q1@candidate`, official validation (5,046 last readouts, 136 positives), test split
untouched. Files: `reports/explain_q1_summary.json`, `explain_q1_groups.csv`, `explain_q1_columns.csv`,
figures `reports/figures/explain_q1_*.png`, MLflow run `explain_q1`. The explanations describe what the
**model** uses, not what causes a failure (the 14 variables are anonymised).

**Methods and units.** SHAP (`shap` 0.54.1, `TreeExplainer` on the XGBoost step after the Pipeline's
preprocessing; one-hot columns summed back to their spec) gives additive contributions to the **raw
log-odds of the uncalibrated model**: base value −3.60 (p = 0.027) + sum of SHAP = margin. The deployed
sigmoid calibrator is linear in the margin, so the calibrated contribution is exactly 0.526 × SHAP;
risks are always quoted as calibrated probabilities. Permutation importance (needs labels, so it is an
evaluation of the fitted model, never a feature-selection tool) shuffles a column or a **group of
columns** (all raw and engineered columns of one variable, jointly) 20 times on the validation and
reports the drop of PR-AUC (baseline 0.0653) as mean ± std of the repeats. That spread is the shuffle
noise only; the sampling noise of 136 positives comes on top. A drop counts as **reliable** if
`mean − 2·std > 0`; groups inside the spread are not ranked. Gain importance (XGBoost) is shown as the
biased reference: it is computed on the training data, favours continuous columns with many split points,
splits credit arbitrarily between correlated columns, is unsigned and says nothing about held-out
performance.

> **Limitation.** `TreeExplainer` runs with the default `feature_perturbation="tree_path_dependent"` (no
> background data; the trees' own cover statistics stand in for the data distribution). With correlated
> features (histogram bins and their shares, counters and their deltas) this can split credit between
> them in an unintuitive way. The `interventional` variant with a training background sample is the
> alternative; it is not used, for runtime reasons. This is why the main comparison is made on **groups**.

| Group (variable) | Columns | Permutation drop of PR-AUC (mean ± std) | Reliable | Rise of the log loss (calibrated p; baseline 0.1212) | Mean abs. SHAP of the group | Gain share |
|---|---|---|---|---|---|---|
| 397 (histogram) | 74 | −0.0023 ± 0.0067 | no | +0.00132 ± 0.00036 (reliable) | **0.630** (rank 1) | 23.5 % |
| **459 (histogram)** | 42 | **+0.0131 ± 0.0039** | **yes (rank 1)** | +0.00311 ± 0.00054 (reliable) | 0.514 (rank 2) | 18.2 % |
| 666 (counter) | 3 | +0.0090 ± 0.0054 | no | +0.00173 ± 0.00052 (reliable) | 0.211 (rank 3) | 9.1 % |
| 167 (histogram) | 22 | +0.0076 ± 0.0046 | no | +0.00118 ± 0.00034 (reliable) | 0.170 (rank 4) | 7.3 % |
| **291 (histogram)** | 24 | **+0.0077 ± 0.0025** | **yes (rank 2)** | +0.00072 ± 0.00029 (reliable) | 0.137 (rank 5) | 6.6 % |
| 158 (histogram) | 22 | +0.0055 ± 0.0040 | no | +0.00160 ± 0.00028 (reliable) | 0.122 | 8.6 % |
| `time_step` | 1 | +0.0019 ± 0.0027 | no | +0.00058 ± 0.00008 (reliable) | 0.057 (rank 11) | 0.4 % |
| Spec_4 | 1 | +0.0005 ± 0.0001 | yes, but negligible | +0.00004 ± 0.00001 | 0.003 | 0.7 % |
| Spec_0 | 1 | 0 | no | 0 | 0 (never used) | 0 % |

(Full table with all 23 groups: `explain_q1_groups.csv`.)

**Does 397 matter for the probability level although not for the ranking?** Yes. Scored by the log loss of
the calibrated probabilities (same groups, same shuffles), shuffling group 397 raises the log loss by
+0.00132 ± 0.00036, a reliable rise (about 3.7 spreads, rank 4 of 23, behind 459, 666 and 158), while the
PR-AUC does not fall (−0.0023 ± 0.0067): the 397 columns help set the *level* of the risk, not the order of
the trucks. The log loss also separates more groups than PR-AUC (14 of 23 are reliable, against 3 for PR-AUC), because it uses
every truck, not only the few positives.

- **Where grouped permutation and grouped SHAP agree:** the histogram groups **459** and **291**. Both are
  in the top 5 of SHAP, and both are the only groups with a drop clearly above the spread (459: about
  20 % of the PR-AUC; 291: about 12 %). Spec_4 is "reliable" only because its spread is tiny; its effect
  (0.0005) is below 1 % of the PR-AUC and SHAP gives it nothing.
- **Where they disagree:** **397 (histogram)** is rank 1 by grouped SHAP and by gain, yet shuffling it
  does not hurt the validation PR-AUC (−0.0023 ± 0.0067): the model uses these 74 columns heavily, but
  on this validation sample they do not help the ranking (they may fit noise, or the class mix differs,
  see the class-mix check). **666 (counter)** and **167 (histogram)** are in the SHAP top 5 with drops
  of +0.0090 and +0.0076, but their spreads are as large as the drops: 20 of 23 groups, including
  these, are **inside the spread and are not ranked**. Spearman rank correlation of the two measures
  over all 23 groups: 0.57.
- **Gain** has the same top 3 as SHAP (397, 459, 666) but overstates the specs and the 158 / 272
  histograms and understates `time_step` (0.4 % of the gain vs. 2.4 % of the SHAP).
- **Beeswarm:** the typical validation truck gets about +0.3 log-odds from `459_9` and `666_0` alone,
  so the explanations start far above the base value; it is the same shift that makes the raw
  probabilities about 4 × too high on the validation (mean 0.112 vs. base 0.027).
- **`time_step`** (dependence plot): positive SHAP (up to +0.3) at about 50–130 steps, slightly negative
  later; a handful of trucks with `time_step` < 50 get strongly negative values. Not a monotone "older
  means riskier" effect.

**Two stories** (rows chosen by a rule fixed in advance: the true positive with the highest calibrated
probability and the lowest-scoring class-4 truck below the deployed threshold 0.0353; figures
`explain_q1_waterfall_*.png`):

1. *Highest-risk true positive, truck 27806, class 3 (6–12 steps before the repair), calibrated p = 0.188
   (5.3 × the threshold).* From the base value −3.60 the margin rises to +0.94. The strongest pushes up are
   the growth rate of counter `666_0` since the last readout (+0.77), a high value in bin 9 of histogram
   459 (33,001, +0.48), a very high growth rate of counter `427_0` (+0.31), the cumulative `666_0`
   (127,501, +0.25), nothing in the top bin of histogram 158 (+0.17) and specification `Spec_7 = Cat4`
   (+0.12). The top 8 add +2.41, the other 231 input features +2.12 together; nothing in the top 8 pushes down. The signal is
   spread over many features.
2. *Worst miss, truck 16995, class 4 (0–6 steps before the repair), calibrated p = 0.020 (below the
   threshold).* The margin is −3.67, almost exactly the base value −3.60: the evidence cancels out. A
   mid-sized value in histogram 459 bin 9 (9,329, +0.24) and bins 3 and 15 push up; a low share in bin 12
   of histogram 397 (−0.16), a low cumulative `666_0` (63,015, −0.15) and the shares of histograms 291 and
   167 push down. This truck looked like an average truck to the model: the explanation shows that the
   miss is missing signal, not a threshold detail.

### `matched` vs. `k5` on the official validation (06.11, `make matched-check`) and "none vs. balanced"

One-time check left open on 05.11 (`reports/matched_check_q1.json`, MLflow run `matched_vs_k5_validation`).
Same recipe as the registered candidate (XGBoost, 300 rounds, same parameters, engineered features, no class
weights, same 239 columns); only the training rows differ: `k5` = the registered model (117,031 rows, 2.8 %
positives), `matched` = 5 cut-offs per truck following the validation `time_step` (70,909 rows, 7.8 %
positives). **Rule fixed before the run:** switch only if `matched` is better in **both** PR-AUC and the
cross-fitted cost by more than the std of the paired bootstrap difference (500 resamples of the validation
trucks, same trucks and folds for both models); otherwise keep `k5`.

| | `k5` (registered) | `matched` | Difference (matched − k5) | Bootstrap spread (std) | Rule |
|---|---|---|---|---|---|
| PR-AUC (95 % interval) | 0.0653 (0.048–0.102) | **0.0913** (0.068–0.128) | **+0.0260** (95 % interval +0.0044 … +0.0477; better in 98.8 % of the draws) | 0.0114 | above the spread |
| Cross-fitted cost (primary mapping) | 28,220 | **26,055** | saving **2,165** (95 % interval −3,011 … +9,504; cheaper in 79.4 % of the draws) | 3,285 | **not above the spread** |

**Decision: keep `k5`** (the rule needs both). Honest reading: the ranking of `matched` is better on the
official validation, which shares no mechanism with the matching (the interval of the PR-AUC difference
excludes 0); the cost saving (7.7 %) points the same way but is smaller than its noise: 136 positives and a
threshold learned on 4/5 of them cannot resolve it. Reasons that also favour `k5` in practice: the deployed
calibrator and threshold belong to `k5`; `matched` would need a new calibrator, whose label-free fit rests on
the evaluation set built by the same rank matching (the 607 / 26 / 2 / 0 artefact, class-mix check above), a
circular dependence; and the validation is now used for a selection, so its numbers are slightly optimistic.
`matched` stays the first candidate to revisit if the protocol is reworked (open point). The test split is
untouched.

**"none vs. balanced" closed.** The deployed XGBoost uses **no class weights**: a positive weight (neg/pos
35.2) was worse on 04.11 (−0.0087, mean probability 0.26 instead of about 0.03) and, for the logistic
regression, "balanced" and "none" are tied (−0.0020 ± 0.0029); the probability level is handled by the sigmoid
calibration (Brier 0.026 on the validation, §6 calibration).

## 7. Evaluation protocol

- **SCANIA:** official split by truck (train / validation / test). No own random split.
- Model selection: `StratifiedGroupKFold(5)` by `vehicle_id` on the sampled training rows,
  built once by `make_group_folds`; `n_splits` and the seed come from `configs/config.yaml`
  (`cv`), so every model uses exactly the same folds.
- **From step 7b on, models are scored on the validation-like evaluation set** (§6,
  "Evaluation protocol"): one row per truck, fixed and seeded, with the `time_step`
  distribution **and** the positive rate of the official validation (`calibrated`), on the
  same truck-to-fold assignment as `make cv`; paired comparisons on those rows. The literal
  procedure (`literal`) is run as a second reference for every important decision, and the old
  CV number is reported for continuity. **The official validation split (06.11) is the real
  check.**
- Threshold choice and calibration: on the validation set.
- **The test set is used exactly once**, for the final reported result.
- All models use the same folds and the same metrics; every run is logged in MLflow
  (local SQLite tracking, `configs/config.yaml` → `mlflow`; UI with `scripts/mlflow_ui.sh`).
  Per run: parameters, per-fold scores, mean and standard deviation, and the **train/validation
  gap** (`return_train_score=True`) as the over-/underfitting signal. The validation part of a
  fold is logged as `val_*`, never as `test_*`, to avoid confusion with the test split.
- **Fixed evaluation set for data-preparation choices.** When an experiment changes the
  *training sample* (number of cut-offs `k`, training size in a learning curve), the
  validation rows must not change with it. These experiments validate on **one cut-off per
  truck** (like the official validation/test splits) with folds assigned **once per vehicle**
  (`compare_cutoffs.assign_vehicle_folds` / `build_fixed_eval_folds`). Only the training rows
  change, so a difference in the score is caused by the choice being tested.
- Models are compared **fold by fold** on the same folds (paired comparison,
  `baseline_cv.paired_comparison`: mean and standard deviation of the per-fold differences and
  the number of folds won); a difference smaller than the variation between folds is not
  treated as an improvement.
- **Input data is validated by the data contract** (§3) before training; a violation stops
  the run.
- **C-MAPSS:** `GroupKFold` by engine on the training engines; the official test engines with
  `RUL_FD001.txt` are used once at the end.

## 8. Known limitations

- Variables are **anonymised and partly scaled**. SHAP can show *which* feature matters,
  but not its physical meaning; the LLM report must not invent one.
- **Censoring:** trucks without a repair give only partial information; their last 48 time
  steps are excluded from training.
- **`time_step` shift between training and validation (explained 05.11, v1.1).** The training
  cut-offs are much earlier than the validation last readouts (median 83 vs. 169, KS = 0.43,
  §3). It is **not** caused by the cutting procedure: the official validation histories are
  *longer*, they keep 82 % of a mean training history (38.9 of 47.7 readouts per vehicle, from
  the paper's counts), while a uniform cut of training-length histories keeps about 51 %
  (§6, "Evaluation protocol"). The model is therefore evaluated on older trucks with larger
  cumulative counters than it is trained on (covariate shift), which no sampling of the
  training trucks removes completely, because in the training data `time_step` is tied to the
  distance to the end of the study. Consequences: the validation-like evaluation set imitates
  the observable properties of the validation split (`time_step` distribution and positive
  rate, using one aggregate label rate; test untouched) but cannot reproduce every property;
  the `matched` training sample is not trusted on the calibrated set alone (shared
  construction); the official validation (06.11) is the real check. Removing `time_step` costs
  XGBoost 0.006 PR-AUC on every evaluation, so the model does use it; whether that is wear
  information or the study design is not decidable from these data.
- **The validation-like evaluation set is a model of the validation split, not the split.**
  Its repaired-truck share changes with `time_step` (16 % early, 3 % late), which cannot be
  checked against the official split, and it counts censored near-end readouts as class 0, as the
  official labels do (they may be pre-failure readouts).
- **Data errors in the tails:** a few rows (11 of 117,031 with a histogram bin above 10⁹, one
  of them 4,000 × above the rest) look like sensor or logging errors. They are kept and
  handled by `log1p`, not removed; a documented clipping or removal rule would have to be
  fitted on training data only.
- The cost matrix comes from a research challenge, not from a real service department.
- **Class mix of the evaluation set:** the calibrated set has 95.6 % class 1 and no class 4, the
  official validation 56 % class 4 (class-mix check, §6, 06.11). Open point, protocol unchanged.
- One component type in one manufacturer's fleet; results do not transfer automatically to
  other components or vehicles.
- C-MAPSS is **simulated** data; the RUL part demonstrates the method, not real engine
  performance.

## 9. Out of scope

- Real-time control of vehicles; the system only warns and explains.
- Planning or scheduling of workshop orders.
- Identifying what Component X physically is.
- Bosch CNC Machining (cross-machine anomaly detection, the former Should item): cut from v1.0 on
  06.11.2026, future work after the presentation on 20.11 ([ADR 0003](adr/0003-cut-bosch-cnc.md)).

## Change log

- **v1.8 (06.11.2026):** one-time `matched` vs. `k5` check on the official validation with a rule fixed in
  advance (paired bootstrap, PR-AUC and cross-fitted cost): keep `k5`; "none vs. balanced" closed (§6).
- **v1.7 (06.11.2026):** SHAP (TreeExplainer, tree_path_dependent, limitation stated), grouped and single
  permutation importance (20 repeats, reliable = mean − 2 std > 0), gain importance, two waterfall stories,
  `explain/` package with `top_shap_factors` for `/explain` and the LLM report (§6).
- **v1.6 (06.11.2026):** open point for 07.11: Q2 faces label shift (training mostly class 1, official
  validation mostly class 4); option: prior-shift correction with the class mix published in the paper (§6).
- **v1.5 (06.11.2026):** class-mix check on training data (`analysis/class_mix.py`, `make class-mix`):
  readouts are not denser near the repair; the 607 / 26 / 2 / 0 mix of the calibrated set comes from
  the constant ≈ 44-step offset between training reach and validation `time_step` in rank matching;
  open point (§6, §8).
- **v1.4 (06.11.2026):** scope decision: the Should item Bosch CNC is cut from v1.0 and moved to
  future work ([ADR 0003](adr/0003-cut-bosch-cnc.md)); Q4 on SCANIA only (§2, §9).
- **v1.3 (06.11.2026):** `cost` block in the config (matrix from the challenge PDF, it was
  not there before); binary cost mapping primary/literal (`cost.py`); validation frame builder
  (last readout per truck, no future readouts); `calibration.py`, `threshold.py`,
  `make calibrate`: sigmoid/isotonic calibrators, cross-fitted threshold by cost, baselines,
  first measurement on the official validation (§6).
- **v1.2 (05.11.2026, step 7b):** Optuna search (TPE, MedianPruner per fold, early stopping on a
  truck-grouped inner split) for XGBoost (24 of 40 trials, 12-minute cap) and LightGBM (40 trials) on the
  calibrated evaluation set: best 0.0585 / 0.0592, not clearly better than the untuned engineered XGBoost
  (+0.0015 ± 0.0040, 3/5 folds; tuned models slightly worse on the literal set). Metrics added: ROC-AUC,
  Brier (worse than the base rate on the calibrated set), mean predicted probability, top-5 % alert
  precision / recall / F1 and a provisional confusion matrix. The untuned engineered XGBoost is
  registered as `failsense-q1` (alias `candidate`) by the pre-set winner rule.
- **v1.1 (05.11.2026, step 7a, evaluation protocol):** the old CV evaluation was too optimistic
  (XGBoost 0.1335 → 0.053 calibrated / 0.090 literal; logistic regression 0.110 → 0.044 / 0.078,
  ranking unchanged). A rank matching of the `time_step` marginal failed the positive-rate check
  (9.61 % vs. 2.70 %); the scratch comparison A–D and the literal procedure are documented as the
  reason for the final protocol, calibrated rank matching (KS 0.008, positive rate 2.70 %; uses the
  validation readouts' `time_step` and one aggregate label rate, test untouched). Found in the
  paper's own counts: the validation histories are longer (82 % vs. about 51 % expected from a
  uniform cut), so the `time_step` shift is a property of the data. Training samplings, `time_step`
  and engineered re-checks on both sets (§6); `matched` not trusted yet (shared construction,
  7.84 % positives); decisions for step 7b.
- **v1.0 (05.11.2026, tasks of plan day 04.11, SCANIA only):** random forest,
  HistGradientBoosting, XGBoost and LightGBM added to the comparison table (fixed budget, no
  early stopping, no class weights): all about 0.131–0.134 PR-AUC vs. 0.110 for the logistic
  regression, better in 5/5 folds and tied with each other. Re-checks of XGBoost: specifications
  and `time_step` help (−0.007 / −0.006 without), engineered features +0.010 (4/5 folds, likely),
  positive class weight hurts and shifts the probabilities. Candidates for step 7: XGBoost and
  LightGBM (§6).
- **v0.9 (05.11.2026, tasks of plan day 03.11):** one comparison table (`make compare`) for
  logistic regression, kNN (k 25/101, uniform/distance), Gaussian naive Bayes and decision
  trees (depth 4/8/unlimited) on the same 5 folds: nothing beats the reference (best other:
  kNN 0.1057 vs. 0.1098); naive Bayes is badly miscalibrated (mean probability 0.57), trees
  overfit with depth; impurity top-10 and runtimes documented; boosting is added to the same
  table next (§6).
- **v0.8 (05.11.2026, tasks of plan day 02.11, RUL part):** C-MAPSS FD001 baseline (Q3):
  RUL labels with cap 125, `GroupKFold` by engine, Dummy / Ridge / Lasso, RMSE and MAE in
  MLflow experiment `failsense-rul`; linear model halves the Dummy's RMSE (21.6 vs. 41.7),
  Ridge = Lasso, Lasso zeros exactly the 7 constant columns, `cycle` variant measured but not
  adopted; caveat on CV vs. last-cycle test evaluation (§6).
- **v0.7 (05.11.2026, tasks of plan day 02.11):** regularization of the logistic-regression
  reference (L2, L1, ElasticNet, C): no proven gain, reference stays L2 with C = 1; L1 keeps
  80 of 196 coefficients at no measurable cost; linear models saturate at about 0.11 PR-AUC
  (high bias, non-linear models next); three `saga` models did not converge and are marked
  (MLflow tag `converged = false`); collinearity and one-hot scaling caveats for the
  coefficients, SHAP on 06.11 (§6).
- **v0.6 (05.11.2026, tasks of plan day 30.10):** engineered features (counter Δ and rate,
  histogram shares, mean bin, top-bin share), `time_step` with/without, class weight
  none/balanced/SMOTE incl. the calibration note, missing-value handling, outlier analysis and
  `log1p` (first consistent gain: +0.0086 ± 0.0053, 5/5 folds), and the `time_step`
  distribution check (KS = 0.43: training cut-offs are much earlier than the validation last
  readouts) added to §3, §6 and §8; decisions for 04.11 and 06.11 documented; the `time_step` shift is an open point for catch-up step 7 (sampling unchanged).
- **v0.5 (29.10.2026):** pandera data contract for all SCANIA files, verified on the real data
  (§3); specifications joined by `vehicle_id` and one-hot encoded in a `ColumnTransformer`
  (§3, §4); two additional leakage guards (§4); StandardScaler vs. MinMaxScaler and
  with/without specifications compared fold by fold — no real difference, decisions and
  re-check on 04.11 documented (§6); evaluation protocol: paired comparison function and data
  validation (§7).
- **v0.4 (28.10.2026):** `k = 5` cut-offs per truck decided by CV on a fixed evaluation set
  (§3); baselines measured (Dummy 0.028, LogReg 0.105 PR-AUC) and diagnosed as high bias with a
  learning curve (§6); evaluation protocol extended with MLflow tracking, train/validation gap,
  fixed evaluation set and paired comparison (§7); `time_step` distribution check moved to
  30.10; wrong reference "documented in v0.3" for the Q1 cost derivation corrected (§5).
- **v0.3 (27.10.2026):** window borders decided (upper bound inclusive, `labels.py`); full
  5 × 5 cost matrix from the challenge description; training samples by random cut-offs
  per truck (`sampling.py`, `k` chosen by CV on 28.10); split overlap checks and group folds
  (`split.py`), verified on the real splits; EDA numbers for missing values; possible
  selection effect added to the limitations.
- **v0.2 (26.10.2026):** dataset switch to SCANIA Component X (ADR 0001); new questions,
  task-type justification, label construction with censoring, vehicle/time leakage rules,
  SCANIA cost matrix instead of the assumed 10 : 1 ratio.
- **v0.1 (25.10.2026):** first version for AI4I 2020.
