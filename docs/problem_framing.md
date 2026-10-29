# Problem Framing — FailSense

> Status: **v0.5 (29.10.2026)**. All SCANIA files are checked by a data contract, the
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
| 4 | Does a truck behave abnormally, even without failure examples? | Unsupervised — anomaly detection | SCANIA Component X | Must |
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
- The check of the `time_step` distribution at the sampled cut-offs against the last readout
  in validation is moved to 30.10, together with the selection-effect check (§8).
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

**Target rule.** Every new model must beat the logistic-regression baseline **on the same
folds**, compared fold by fold, not only on the mean. A numeric business target (total cost
on validation) is set on 06.11 after calibration and threshold choice, not before.
Results on real data are expected to be clearly lower than the near-perfect scores often
published for synthetic datasets.

## 7. Evaluation protocol

- **SCANIA:** official split by truck (train / validation / test). No own random split.
- Model selection: `StratifiedGroupKFold(5)` by `vehicle_id` on the sampled training rows,
  built once by `make_group_folds`; `n_splits` and the seed come from `configs/config.yaml`
  (`cv`), so every model uses exactly the same folds.
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
- **Possible selection effect (to check on 30.10):** in the EDA, censored trucks are observed
  for at least 121 time units, repaired trucks from 73.4. Together with the excluded censored
  readouts near the end, features such as truck age (`time_step`) might separate the groups
  for reasons of study design rather than wear. Their effect is measured with and without them.
- The cost matrix comes from a research challenge, not from a real service department.
- One component type in one manufacturer's fleet; results do not transfer automatically to
  other components or vehicles.
- C-MAPSS is **simulated** data; the RUL part demonstrates the method, not real engine
  performance.

## 9. Out of scope

- Real-time control of vehicles; the system only warns and explains.
- Planning or scheduling of workshop orders.
- Identifying what Component X physically is.

## Change log

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
