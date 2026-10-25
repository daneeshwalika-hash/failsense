# Problem Framing — FailSense

> Status: v0.1 (25.10.2026). Written **before** any model was trained, so that metrics
> and success criteria cannot be chosen after seeing results.

## 1. Business problem

An unplanned machine stop on a production line stops the whole line and is expensive.
Fixed-interval maintenance replaces parts that are still good. The goal of FailSense is an
**early warning based on sensor data**, so maintenance happens only when it is needed —
and the technician understands *why* the warning was raised.

## 2. Questions and ML task types

| # | Question | ML task (initial) | Dataset | Priority |
|---|----------|-------------------|---------|----------|
| 1 | Will this machine fail? | Supervised — binary classification | AI4I 2020 | Must |
| 2 | Which failure mode is expected? | Supervised — multi-class / multi-label classification | AI4I 2020 | Should |
| 3 | How many cycles remain until failure (RUL)? | Supervised — regression on time series | NASA C-MAPSS FD001 | Must |
| 4 | Is the machine behaving abnormally, even without failure examples? | Unsupervised — anomaly detection | AI4I 2020 | Must |
| 5 | Why did the model predict a failure, and what should the technician do? | Explainable AI (SHAP) + LLM report | Model outputs | Must / Should |

*To be refined on 26.10 (course day 1): justification of each task type.*

**What one prediction means (AI4I).** Each row is an independent snapshot of a process state
(product type, temperatures, speed, torque, tool wear). Question 1 is therefore:
*"Does this operating state lead to a machine failure?"* — not a forecast over time.
Forecasting over time is covered by question 3 (C-MAPSS).

## 3. Data facts that shape the design (from EDA, `notebooks/01_eda.ipynb`)

- 10,000 rows, no missing values, no duplicates. `UDI` and `Product ID` are identifiers → dropped.
- **Strong imbalance:** 339 failures = **3.39 %**. A model that always predicts "no failure"
  reaches **96.61 % accuracy and 0 % recall** → accuracy is not a valid metric here.
- Failure modes overlap (24 rows with more than one mode) → question 2 is closer to multi-label.
- Label inconsistencies are kept as recorded and documented: 9 failures without a mode,
  18 rows with `RNF = 1` but `Machine failure = 0`.
- Failures follow **thresholds and interactions** (e.g. HDF: temp diff < 8.6 K **and**
  speed < 1380 rpm; PWF at both power extremes). Spearman correlations with the target are
  weak (|rho| ≤ 0.17) although the rules are clear → non-linear models and physical
  features are expected to help.

## 4. Data leakage — the most important risk

`TWF`, `HDF`, `PWF`, `OSF`, `RNF` describe *which* failure happened and are only known
**after** the failure. A one-line rule "failure if any mode flag is set" — no ML — reaches
**recall 0.973 and precision 0.948** (TP 330, FP 18, FN 9).

**Rules:**
1. The five mode columns are **never** features. They are only the target of question 2.
2. An automated test (`tests/test_leakage.py`) fails if any of them appears in the features.
3. All preprocessing (scaling, encoding, imputation, resampling) runs **inside** an
   sklearn `Pipeline` and therefore inside each cross-validation fold.
4. C-MAPSS: cycles of the same engine must never be in training and validation at the same
   time → `GroupKFold` by engine id.

## 5. Cost assumption

In maintenance, the two errors do not cost the same:

| Error | Meaning | Relative cost |
|-------|---------|---------------|
| False Negative (FN) | Failure not detected → unplanned line stop | **10** |
| False Positive (FP) | False alarm → unnecessary inspection | **1** |

`Total cost = 10 × FN + 1 × FP`

The ratio 10 : 1 is an **assumption**, not a measured value. It will be stored in
`configs/config.yaml` and can be changed without code changes. The decision threshold is
chosen to minimise total cost on the **validation** set — not fixed at 0.5.

## 6. Success metrics

| Task | Primary metric | Secondary | Reference point |
|------|----------------|-----------|-----------------|
| Q1 binary | **PR-AUC** (average precision) | Recall and precision at the cost-optimal threshold, total cost, confusion matrix | Random classifier PR-AUC ≈ prevalence ≈ 0.034; `DummyClassifier`; `LogisticRegression` |
| Q2 failure mode | Macro F1 per mode | Per-mode recall | `DummyClassifier`; RNF expected ≈ 0 (random by design) |
| Q3 RUL | **RMSE in cycles** | MAE | Predict-the-mean baseline; `Ridge` |
| Q4 anomaly | PR-AUC of the anomaly score vs. real failures | Recall in the top-k most anomalous rows | Random ranking |
| Q5 explanation | SHAP consistent with EDA mechanisms | LLM report is valid JSON, in German, and only uses the top-5 SHAP factors | — |

**Why PR-AUC and not ROC-AUC?** With 3.4 % positives, ROC-AUC looks good even for weak
models, because the many true negatives dominate. PR-AUC focuses on the positive class.

**No target number yet.** Numeric targets are set **after** the baselines are measured
(28.10), not invented now.

## 7. Evaluation protocol

- Fixed split: train 70 % / validation 15 % / test 15 %, stratified, `random_state = 42`, saved to disk.
- Model selection and threshold choice: cross-validation on train + validation set only.
- **The test set is used exactly once**, for the final reported result.
- All models use the same split, the same folds and the same metrics; every run is logged in MLflow.

## 8. Known limitations

- AI4I and C-MAPSS are **synthetic / simulated**. Results do not transfer 1 : 1 to a real
  factory. SECOM (real semiconductor data) is used as a reality check and is expected to
  give clearly weaker results.
- `RNF` failures are random by design and cannot be predicted from sensors.
- `TWF` is only partly predictable (tool replacement vs. failure is random in 200–240 min).
- The cost ratio is an assumption; in a real project it comes from the maintenance department.

## 9. Out of scope

- Real-time control of machines; the system only warns and explains.
- Planning or scheduling of maintenance work orders.
