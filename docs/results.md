# Results — FailSense (status 06.11.2026)

All numbers below come from files in `reports/` or from the MLflow runs (experiments `failsense-q1-binary`,
`failsense-rul`); the source is named next to each table. Design, decisions and the history of every
choice are in [problem_framing.md](problem_framing.md). **The test split has not been used.**

## 1. Summary

1. **Business result (official validation, 5,046 trucks, 136 failing):** with the SCANIA cost matrix the
   deployed alert policy costs **28,220** (cross-fitted), against **34,370** for "alert every truck":
   a saving of **6,150 (18 %, 1.22 per truck), better in 5/5 folds**; never alerting costs 57,400.
2. The policy alerts 2,976 trucks (recall **85 %**, precision 3.9 %) and **skips 2,070 trucks, among
   them 20 of the 136 failing ones** (class 1: 2, class 2: 2, class 3: 8, class 4: 8).
3. **The model ranks, but its probabilities add no skill over the base rate** (PR-AUC 0.065, ROC-AUC 0.70,
   Brier 0.026 = base rate); the honest benchmark is "alert everybody", because a miss costs 200–500 and
   a false alarm 7.
4. **Caveat (mapping a):** the saving assumes an alert on a failing truck costs 0. If an alert is read as
   "predicted class 1" (mapping b), the model saves only **148** vs. never alerting. The positive result
   depends on mapping (a).
5. The test split is untouched and is used once, at the very end. Deployed: sigmoid calibrator fitted
   without validation labels + threshold 0.0353 learned on the whole validation (the cost to quote is the
   cross-fitted 28,220, not the in-sample 27,565).

Source: `reports/calibration_q1_summary.json` (`deployment_choice`, `primary_baselines`,
`primary_decision_cross_fitted`, `sensitivity_literal`), `reports/threshold_q1.csv`.

## 2. Evaluation protocol

**Random-row cross validation was too optimistic.** The first protocol (5-fold CV, folds grouped by truck,
random cut-offs) gave PR-AUC 0.13 for XGBoost, but its cut-offs lie far earlier in a truck's history than
the official validation's last readouts (median `time_step` 83 vs. 169, KS 0.425). On an evaluation set
that imitates the validation, the same models score 1.4–2.5 × lower:

| Model | Old CV (random rows) | Calibrated validation-like set | Literal set | Factor |
|---|---|---|---|---|
| XGBoost | 0.1335 ± 0.0080 | 0.0530 ± 0.0045 | 0.0901 ± 0.0161 | 2.5 / 1.5 |
| Logistic regression | 0.1098 ± 0.0055 | 0.0444 ± 0.0031 | 0.0781 ± 0.0076 | 2.5 / 1.4 |

Source: `reports/eval_old_vs_new_q1_calibrated.csv`, `reports/eval_old_vs_new_q1_literal.csv`,
`reports/time_step_check_q1.csv`.

**Three ways to build an evaluation set** (`reports/eval_set_methods_q1.csv`): *calibrated* rank matching
reproduces both the validation's `time_step` distribution (KS 0.008) and its positive rate (2.696 % vs.
2.695 %); *literal* (one random readout per truck) matches the rate (2.61 %) but not the `time_step`
distribution (KS 0.31); plain rank matching matches the distribution (KS 0.012) but gives a 9.6 % positive
rate. The calibrated set is the main protocol, the literal set the second reference.

**Class-mix finding (06.11).** The calibrated set has 607 / 26 / 2 / 0 positives in classes 1–4, the
official validation 16 / 14 / 30 / 76:

| Set | Positive rate | Class 1 / 2 / 3 / 4 |
|---|---|---|
| official validation (reference) | 2.70 % | 16 / 14 / 30 / 76 |
| literal (one uniform readout per truck) | 2.61 % | 281 / 158 / 71 / 104 |
| **calibrated rank matching** | 2.70 % | **607 / 26 / 2 / 0** |

Source: `reports/class_mix_check_q1.csv`. The matching pairs each truck with the validation `time_step`
quantile of its reach rank; the training reach is about 41–46 steps above the target at every quantile,
about the 48-step horizon, so repaired trucks are placed at 39–44 steps before the repair (class 1). So the
calibrated set measures **mostly class-1 early warning**; the official validation is mostly imminent
failures (56 % class 4). Readouts are not denser near the repair (0.23–0.28 per time unit in every window),
and a shifted reference point does not explain the validation's mix. Consequences are in section 9.

## 3. Model comparison

PR-AUC, 5 folds grouped by truck, same folds for every model, old CV (random cut-offs). The base rate
(a classifier that ignores the features) is 0.0276.

| Model | PR-AUC (mean ± std) | Train PR-AUC | Paired vs. logreg (folds won) |
|---|---|---|---|
| Dummy (prior) | 0.0276 ± 0.0000 | 0.0276 | – |
| Logistic regression (L2, balanced, `log1p`, specs) | 0.1098 ± 0.0055 | 0.125 | reference |
| kNN (k = 101, distance) | 0.1057 ± 0.0113 | – | −0.0041 (2/5) |
| Gaussian naive Bayes | 0.0514 ± 0.0008 | 0.053 | −0.0585 (0/5) |
| Decision tree (depth 8) | 0.0806 ± 0.0020 | 0.128 | −0.0293 (0/5) |
| Random forest | 0.1339 ± 0.0079 | 0.748 | +0.0240 (5/5) |
| HistGradientBoosting | 0.1309 ± 0.0054 | 0.917 | +0.0210 (5/5) |
| XGBoost | 0.1335 ± 0.0080 | 0.515 | +0.0236 (5/5) |
| LightGBM | 0.1331 ± 0.0040 | 0.928 | +0.0233 (5/5) |
| XGBoost + engineered features (deployed recipe) | **0.1435 ± 0.0080** | 0.612 | +0.0337 (5/5) |

Source: `reports/compare_models_q1.csv` (Dummy: MLflow runs `dummy_prior_k5`; engineered row:
`reports/compare_models_q1_recheck.csv`). The four boosting and forest models are tied with each other;
all beat the linear model in 5/5 folds. The large train/validation gaps of the trees show overfitting.
Without the specs (−0.0069, 0/5 folds) or without `time_step` (−0.0061, 0/5) XGBoost is worse; engineered
features help (+0.0101, 4/5).

On the validation-like sets (the protocol from 05.11 on; `reports/evaluate_q1.csv`,
`reports/eval_old_vs_new_q1_*.csv`):

| Model | Calibrated set | Literal set |
|---|---|---|
| Logistic regression | 0.0444 ± 0.0031 | 0.0781 ± 0.0076 |
| XGBoost | 0.0530 ± 0.0045 | 0.0901 ± 0.0161 |
| **XGBoost + engineered (registered `failsense-q1` v1)** | **0.0570 ± 0.0062** | **0.0943 ± 0.0070** |

## 4. Hyperparameter search

Optuna (TPE, MedianPruner per fold, early stopping on a truck-grouped inner split) on the calibrated set;
XGBoost: 24 of 40 trials (12-minute cap), LightGBM: 40 trials. Source: `reports/evaluate_q1.csv`,
`reports/evaluate_q1_paired.csv`, `reports/evaluate_q1_winner.json`, `reports/hpo_q1_best_params.json`.

| Model | Calibrated PR-AUC | Paired vs. untuned (calibrated) | Literal PR-AUC | Paired vs. untuned (literal) |
|---|---|---|---|---|
| XGBoost tuned (best of 24) | 0.0585 ± 0.0093 | +0.0015 ± 0.0040 (3/5) | 0.0923 ± 0.0084 | −0.0020 ± 0.0083 (2/5) |
| LightGBM tuned (best of 40) | 0.0592 ± 0.0138 | +0.0022 ± 0.0100 (2/5) | 0.0860 ± 0.0067 | −0.0083 ± 0.0070 (1/5) |
| **XGBoost untuned, engineered** | 0.0570 ± 0.0062 | reference | 0.0943 ± 0.0070 | reference |

**Winner rule (fixed before the run):** a tuned model is registered only if its mean fold difference is
larger than the standard deviation of the differences **and** it wins in at least 4 of 5 folds on the
calibrated set. Neither model qualified (`"qualifying": []`), so the simpler **untuned engineered XGBoost
(300 rounds, learning rate 0.05, depth 4, no class weights)** is registered. The tuned scores are the best
of N trials chosen on the same folds they are scored on, hence optimistic; on the literal set both tuned
models are slightly worse.

## 5. Calibration and cost

All numbers on the **official validation** (5,046 trucks, 136 positives = 2.70 %).

**Probabilities** (source `reports/calibration_q1_scores.csv`; calibrators fitted label-free on
out-of-fold scores of the calibrated set, except the two `_cv` references):

| Variant | Brier | Log loss | Mean probability (true rate 0.027) | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| raw model score | 0.0384 | 0.172 | 0.112 | 0.0653 | 0.703 |
| **sigmoid, label-free (primary)** | **0.0261** | 0.121 | 0.043 | 0.0653 | 0.703 |
| isotonic, label-free | 0.0262 | 0.122 | 0.044 | 0.0536 | 0.694 |
| sigmoid, cross-fitted on validation | 0.0258 | 0.117 | 0.027 | 0.0676 | 0.701 |
| isotonic, cross-fitted on validation | 0.0265 | 0.125 | 0.027 | 0.0533 | 0.685 |
| base-rate predictor | 0.0262 | 0.124 | 0.027 | – | – |

The PR-AUC of 0.0653 (95 % bootstrap interval 0.048–0.102) lies near the 05.11 calibrated estimate
(0.0570) and inside the interval of the literal one (0.0943), so the validation cannot rank the two
protocols. The raw score is about 4 × too high; the label-free sigmoid fixes most of it but stays 1.6 × too
high, and **no variant beats the base rate in Brier score**. Isotonic gives no gain.

![Reliability curves](../reports/figures/calibration_q1_reliability.png)

**Cost** (primary mapping: false alarm 7, miss 200 / 300 / 400 / 500 for class 1 / 2 / 3 / 4, alert on a
failing truck 0; source `reports/calibration_q1_summary.json`, `reports/threshold_q1.csv`):

| Policy | Alerts | Total cost |
|---|---|---|
| never alert | 0 | 57,400 |
| **always alert (primary baseline)** | 5,046 | **34,370** |
| alert the riskiest 5 % | 253 | 50,231 |
| theoretical threshold t* = 0.033 on sigmoid probabilities | 3,235 | 28,612 |
| **cross-fitted cost-optimal threshold (sigmoid), reported** | 2,976 | **28,220** |
| in-sample optimum (optimistic) | 3,013 | 27,565 |

Cross-fitted saving vs. always alert: **6,150** in total, per fold 875 / 696 / 1,249 / 811 / 2,519
(mean 1,230 ± 750, 5/5 folds). One fold carries 41 % of the saving, so the bar "mean larger than the
spread" is low. The old "riskiest 5 %" policy is **worse than always alerting**. t* uses the class mix of
the calibrated set (training labels only; mean cost of a miss 205); the cross-fitted fold thresholds
(0.0353–0.0362) agree with it.

![Cost over thresholds](../reports/figures/threshold_q1_cost_curve.png)

**Sensitivity — mapping (b), "alert = predicted class 1"** (an alerted truck of class k costs
`Cost_k_1` = 0 / 200 / 300 / 400): always alert costs 76,570, never alert 57,400, the cross-fitted policy
**57,252**, i.e. **148 less than never alerting**. Under this reading the model is practically worthless.

**Deployment choice (decided 06.11):** label-free sigmoid calibrator (slope 0.526, intercept −1.957 on the
logit of the raw score) and **threshold 0.0353** on the calibrated probability, learned on the whole
validation (`deployment_choice` in the summary JSON). The test split is used once at the very end.

## 6. Explainability

Model `failsense-q1@candidate`, official validation; source `reports/explain_q1_groups.csv`,
`reports/explain_q1_summary.json` (MLflow run `explain_q1`). SHAP values are contributions to the raw
log-odds of the **uncalibrated** model (base value −3.60, p = 0.027); the calibrated contribution is
0.526 × SHAP. Permutation importance shuffles a **group** of columns (all raw and engineered columns of
one variable) 20 times; "reliable" means mean − 2·std > 0. The spread is the shuffle noise only; the
sampling noise of 136 positives comes on top.

| Group | Columns | PR-AUC drop (baseline 0.0653) | Log-loss rise (baseline 0.1212) | Mean \|SHAP\| (rank) | Gain share |
|---|---|---|---|---|---|
| 397 (histogram) | 74 | −0.0023 ± 0.0067 (no) | +0.00132 ± 0.00036 (**yes**) | **0.630 (1)** | 23.5 % |
| **459 (histogram)** | 42 | **+0.0131 ± 0.0039 (yes)** | +0.00311 ± 0.00054 (yes) | 0.514 (2) | 18.2 % |
| 666 (counter) | 3 | +0.0090 ± 0.0054 (no) | +0.00173 ± 0.00052 (yes) | 0.211 (3) | 9.1 % |
| 167 (histogram) | 22 | +0.0076 ± 0.0046 (no) | +0.00118 ± 0.00034 (yes) | 0.170 (4) | 7.3 % |
| **291 (histogram)** | 24 | **+0.0077 ± 0.0025 (yes)** | +0.00072 ± 0.00029 (yes) | 0.137 (5) | 6.6 % |
| `time_step` | 1 | +0.0019 ± 0.0027 (no) | +0.00058 ± 0.00008 (yes) | 0.057 (11) | 0.4 % |
| Spec_0 | 1 | 0 | 0 | 0 (never used) | 0 % |

- **Permutation by PR-AUC** finds only two groups clearly above the spread, **459 and 291**; the other 20
  groups are inside it and are not ranked. **Grouped SHAP** ranks 397, 459, 666, 167, 291. They agree on 459
  and 291; the Spearman correlation over all 23 groups is 0.57. Gain has the same top three as SHAP but is
  biased (training data, favours columns with many split points, splits credit between correlated columns).
- **The 397 case:** group 397 is #1 by SHAP and gain, yet shuffling it does not lower the PR-AUC
  (−0.0023 ± 0.0067). Scored by the log loss of the calibrated probabilities it **does** matter
  (+0.00132 ± 0.00036, rank 4 of 23): it sets the *level* of the risk, not the *order* of the trucks. By
  log loss 14 of 23 groups are reliable, against 3 by PR-AUC.
- **`time_step`** (`explain_q1_shap_dependence_time_step.png`): positive SHAP (up to +0.3) at about 50–130
  steps, slightly negative later, strongly negative for a few trucks below 50: not a monotone "older means
  riskier" effect.

![Mean absolute SHAP](../reports/figures/explain_q1_shap_bar.png)

**Two stories** (rows chosen by a rule fixed in advance; `explain_q1_summary.json` → `stories`):

1. **Highest-risk true positive — truck 27806, class 3 (6–12 steps before the repair), calibrated
   p = 0.188 (5.3 × the threshold).** The margin rises from −3.60 to +0.94. Biggest pushes up: the growth
   rate of counter `666_0` (+0.77), a high value in bin 9 of histogram 459 (+0.48), a very high growth rate
   of counter `427_0` (+0.31), the cumulative `666_0` (+0.25), an empty top bin in histogram 158 (+0.17),
   `Spec_7 = Cat4` (+0.12). The top 8 add +2.41, the other 231 inputs +2.12; nothing in the top 8 pushes
   down: the signal is spread over many features.

   ![Waterfall, true positive](../reports/figures/explain_q1_waterfall_true_positive.png)

2. **Worst miss — truck 16995, class 4 (0–6 steps before the repair), calibrated p = 0.020 (below the
   threshold).** The margin is −3.67, almost the base value: the evidence cancels out. Up: a mid-sized
   value in bin 9 of histogram 459 (+0.24); down: a low share in bin 12 of histogram 397 (−0.16), a low
   cumulative `666_0` (−0.15), the shares of histograms 291 and 167. The truck looked average to the model:
   the miss is missing signal, not a threshold detail.

   ![Waterfall, missed positive](../reports/figures/explain_q1_waterfall_missed_positive.png)

## 7. Sampling finding: the cut-off design matters more than tuning

**Like for like, on the same evaluation set (calibrated set, paired per fold):** changing the training
cut-offs from `k5` to `matched` gains **+0.0207** PR-AUC (5/5 folds), the whole Optuna search only
**+0.0015 ± 0.0040** (3/5 folds). On the literal set, the second reference, `matched` gains
+0.0012 ± 0.0097 (4/5 folds), so the large effect is not confirmed there.

| Change (same set, paired per fold) | Effect on PR-AUC | Where |
|---|---|---|
| `matched` instead of `k5` training cut-offs | +0.0207 (5/5 folds) | calibrated set, `compare_sampling_q1_calibrated.csv` |
| Optuna search (24 / 40 trials), tuned vs. untuned | +0.0015 ± 0.0040 (3/5 folds) | calibrated set, `evaluate_q1_paired.csv` |
| `matched` instead of `k5` training cut-offs | +0.0012 ± 0.0097 (4/5 folds) | literal set, `compare_sampling_q1_literal.csv` |

**Independent confirmation of the ranking only (official validation).** On the official validation, which
shares no mechanism with the rank matching, `matched` raises the PR-AUC by **+0.0260** (95 % interval
+0.0044 … +0.0477; `reports/matched_check_q1.json`). This confirms the *ranking* effect. The cost side did
not pass the pre-registered rule (below), so it is not a confirmation of the business result.

`matched` = 5 cut-offs per truck whose `time_step` follows the validation last readouts (rank matching with
noise; 70,909 rows, 7.8 % positives vs. 117,031 rows, 2.8 % for `k5`). More uniform rows do not help
(`k10` −0.0022, all readouts −0.0088 on the calibrated set). The choice of *which readouts become training
examples* moves the score by 0.02–0.03, the whole hyperparameter search by 0.0015.

**Kept `k5`, by a pre-registered rule.** The rule was fixed before the run: switch only if `matched` is
better in **both** PR-AUC and the cross-fitted cost by more than the std of the paired bootstrap
difference (500 resamples of the validation trucks). Result (`matched_check_q1.json`): PR-AUC 0.0913 vs.
0.0653 (difference +0.0260 > spread 0.0114: **above**), cross-fitted cost 26,055 vs. 28,220 (saving 2,165
< spread 3,285: **not above**; cheaper in 79 % of the draws). Keep `k5`; revisit `matched` later. A rule
relaxed after the result is no longer a test, and `matched` would need a new calibrator whose label-free
fit rests on the evaluation set built by the same rank matching (circular).

## 8. RUL baseline so far (NASA C-MAPSS FD001, Q3)

100 engines, 20,631 rows; RUL = max cycle − cycle, capped at 125; 5-fold `GroupKFold` by engine; RMSE in
cycles (lower is better). Source: `reports/rul_baseline_fd001.csv`. The test file and `RUL_FD001.txt` are
untouched.

| Model | RMSE (mean ± std) | MAE |
|---|---|---|
| Dummy (mean) | 41.67 ± 0.00 | 36.97 |
| **Ridge (alpha 1)** | **21.64 ± 1.38** | 17.72 |
| Lasso (alpha 0.1) | 21.63 ± 1.38 | 17.72 |
| Ridge + `cycle` as a feature (not adopted) | 20.09 ± 1.38 | 16.34 |

The alpha of Ridge and Lasso is a tie. Time-series features and boosting come on 11.11.

## 9. Limitations and open points

- **Class mix and label shift (Q2).** The training cut-offs and the calibrated evaluation set are mostly
  class 1 among the positives, the official validation mostly class 4. This (a) makes the evaluation-set
  scores mostly a class-1 measure, (b) makes the label-free t* (0.033) too high for the official mix
  (about 0.016 with the validation's mix, computed from the class counts published with the dataset;
  documentation only, not used for any decision), and (c) means a multi-class model (Q2) would learn the wrong class
  prior. Decision on 07.11: prior-shift correction with the class mix **published in the dataset paper**
  (aggregate numbers, like 136 / 5,046), never per-row validation labels.
- **Only 136 positive trucks** in the validation: PR-AUC (interval 0.048–0.102), the cost saving and the
  per-class recall (16 / 14 / 30 / 76 trucks) are noisy; the fold spread is on the order of the saving's
  size in one fold. The validation is now used for the threshold and for one selection (`matched` vs.
  `k5`), so numbers on it are slightly optimistic; the test split is the only clean check.
- **Mapping (a) vs. (b)** decides whether the model beats the trivial policies (section 5). The cost matrix
  comes from a research challenge, not from a real service department.
- **Probabilities add no skill over the base rate** (Brier 0.0261 vs. 0.0262); they are only calibrated on
  average.
- **Anonymised data:** the 14 variables have no physical meaning in this dataset; explanations describe
  what the model uses, not what causes a failure.
- **Censoring:** trucks without a repair have unknown labels near the end of observation (excluded from
  training; counted as class 0 in the evaluation sets, as the official labels do).
- **SHAP:** `TreeExplainer` runs with the default `tree_path_dependent` perturbation (no background data).
  With correlated features (histogram bins and their shares) this can split credit unintuitively; the
  `interventional` variant with a training background sample is the alternative and is not used, for runtime
  reasons. The group-level comparison is the safer reading. Permutation importance is an evaluation of the
  fitted model, not a tool for choosing features.
- **Not done yet:** Q2 (classes 0–4), anomaly detection (Q4), the LLM report (Q5), the API and dashboard.
  Bosch CNC is cut from v1.0 ([ADR 0003](adr/0003-cut-bosch-cnc.md)).

## 10. Unsupervised view of the training trucks (PCA, KMeans; 09.11)

Descriptive analysis on the 23,550 **training** trucks (last readout, the 239 model inputs;
`make unsupervised`, `notebooks/02_unsupervised.ipynb`, files `reports/unsupervised_*`). Labels
only colour the map and cross-tabulate the clusters. These are not predictive numbers.

| Finding | Value |
|---|---|
| Variance in PC1 / PC2 | 22.4 % / 15.0 % (without `log1p` of the counters: 14.7 % / 10.8 %) |
| Components for 90 % of the variance | 50 of 239 inputs |
| Best k by silhouette (sample 5,000) | k = 2, silhouette 0.27 (k >= 3: about 0.08-0.13; stable over 4 seeds, 0.26-0.27) |
| Repair rate cluster 0 / cluster 1 | 5.4 % (n = 4,840) / 10.7 % (n = 18,710); overall 9.6 % |
| Normalised mutual information cluster vs repaired | 0.007 |

The repaired trucks lie inside the dense blob of the others, so the first two components do
not separate them. The two clusters are mainly a **truck configuration split** (`Spec_0 =
Cat0`: 90 % of cluster 1 vs 50 % of cluster 0), not a wear or failure split. Unsupervised
grouping therefore does not replace the supervised model; it is a descriptive tool (fleet
segments). Caveat: a repaired truck's last readout lies just before its repair, a non-repaired
truck's at the end of observation, which differs from the random cut-offs of the validation.
