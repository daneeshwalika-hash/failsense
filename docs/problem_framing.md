# Problem Framing — FailSense

> Status: **v0.2 (26.10.2026)**. Rewritten after the dataset switch
> ([ADR 0001](adr/0001-dataset-switch.md)): SCANIA Component X is the main dataset,
> NASA C-MAPSS FD001 stays for RUL. Written **before** any model was trained, so that
> metrics and success criteria cannot be chosen after seeing results.
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

*(paper; to be verified in the EDA)*

| Part | Content | Size |
|---|---|---|
| `train_operational_readouts.csv` | `vehicle_id`, `time_step`, 105 feature columns | ~1.12 M rows, 23,550 trucks, ~1.2 GB |
| `train_tte.csv` | `length_of_study_time_step`, `in_study_repair` per truck | ~9.7 % of trucks repaired |
| `train_specifications.csv` | `Spec_0` … `Spec_7` (categorical) per truck | 23,550 rows |
| `validation_*` / `test_*` | readouts up to a random cut-off, specifications, **class 0–4** of the last readout | ~5,046 / ~5,045 trucks |

- **Feature types:** 8 numerical counters (cumulative, e.g. `171_0`) and 6 histogram
  variables with 10–36 bins each (e.g. `167_0` … `167_9`). Variable names are anonymised
  and values are partly scaled → **no physical feature engineering** is possible.
- **Missing values:** less than 1 % per feature *(paper)* → simple imputation inside the
  Pipeline.
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
| Repaired | `d` in the windows 48–24, 24–12, 12–6, 6–0 | class 1, 2, 3, 4 |
| Not repaired (`in_study_repair = 0`) | `d > 48` | class 0 (the repair is at least 48 steps away) |
| Not repaired | `d ≤ 48` | **unknown → not used for training** |

Q1 label = 1 if class ≥ 1, else 0. The exact window borders (inclusive / exclusive) are
taken from the paper and tested with unit tests.

**Open design decision (27.10):** use all readouts of a truck as training rows (more data,
but strongly correlated rows) or sample readouts per truck like the random cut-off in
validation/test. Both are compared with the same cross-validation.

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
   `Pipeline` and therefore inside each cross-validation fold.
5. **Automated tests** (`tests/test_leakage.py`) fail if a label column or `vehicle_id` is in
   the feature matrix, or if a truck appears in more than one split / fold.
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
Examples *(paper)*: true 0 → predicted 1 (unnecessary early check) costs **7**;
true 4 → predicted 0 (imminent failure missed) costs **500**.

`Total cost = Σ cost[true, predicted] × number of trucks`

- The full matrix is copied from Table 1 of the paper into `configs/config.yaml`
  (not hardcoded) and covered by a test that checks its shape and the zero diagonal.
- **Q2** is evaluated directly with this total cost.
- **Q1** (binary) uses costs derived from the same matrix; the derivation is documented
  in v0.3 together with the threshold choice (06.11).
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

**No target number yet.** Numeric targets are set **after** the baselines are measured
(28.10), not invented now. Results on real data are expected to be clearly lower than the
near-perfect scores often published for synthetic datasets.

## 7. Evaluation protocol

- **SCANIA:** official split by truck (train / validation / test). No own random split.
- Model selection: `StratifiedGroupKFold(5)` by `vehicle_id` on the training set,
  `random_state = 42`.
- Threshold choice and calibration: on the validation set.
- **The test set is used exactly once**, for the final reported result.
- All models use the same folds and the same metrics; every run is logged in MLflow.
- **C-MAPSS:** `GroupKFold` by engine on the training engines; the official test engines with
  `RUL_FD001.txt` are used once at the end.

## 8. Known limitations

- Variables are **anonymised and partly scaled**. SHAP can show *which* feature matters,
  but not its physical meaning; the LLM report must not invent one.
- **Censoring:** trucks without a repair give only partial information; their last 48 time
  steps are excluded from training.
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

- **v0.2 (26.10.2026):** dataset switch to SCANIA Component X (ADR 0001); new questions,
  task-type justification, label construction with censoring, vehicle/time leakage rules,
  SCANIA cost matrix instead of the assumed 10 : 1 ratio.
- **v0.1 (25.10.2026):** first version for AI4I 2020.
