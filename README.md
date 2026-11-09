# FailSense

[![CI](https://github.com/daneeshwalika-hash/failsense/actions/workflows/ci.yml/badge.svg)](https://github.com/daneeshwalika-hash/failsense/actions/workflows/ci.yml)

End-to-end predictive maintenance system on real truck fleet data: predicts component failures, estimates remaining useful life (RUL), detects anomalies and explains its predictions.

> Work in progress — first release planned for November 2026.

## Requirements

- Linux or WSL2 (Ubuntu), Python 3.12 (uv installs it from `.python-version`), [uv](https://docs.astral.sh/uv/).
- **System package `libgomp1` (OpenMP runtime) is required for LightGBM** on Linux:

  ```bash
  sudo apt install libgomp1
  ```

  Without it `import lightgbm` fails with `OSError: libgomp.so.1: cannot open shared object file`.
  That breaks the model-comparison tests, `make test` and the pre-commit pytest hook. A fresh
  WSL Ubuntu does not have the package; GitHub's `ubuntu-latest` runner already does.

## Quickstart

```bash
make setup      # uv sync + pre-commit hooks
make data       # download NASA C-MAPSS, prepare SCANIA Component X (manual download, see below)
make parquet    # CSV -> Parquet copies (faster to load)
make validate   # check all SCANIA files against the data contract
make test       # pytest (needs no raw data)
make cv         # cross validation of the baseline models, logged to MLflow
make train      # fit the reference model on the training split, logged to MLflow
make help       # all targets
```

`make leakage` runs only the data-leakage tests (`pytest -m leakage`).

Dependency notes:

- **XGBoost is installed as `xgboost-cpu`** (CPU-only build, same `import xgboost`). The plain
  `xgboost` package pulls NVIDIA NCCL libraries on Linux, which a CPU project does not need;
  this matters for the size of the Docker image.
- **LightGBM needs the system package `libgomp1`** (see Requirements above).
- **Note for 07.11 (Dockerfile):** `python:*-slim` images do not contain `libgomp1` either, so the
  Docker image must install it (`apt-get update && apt-get install -y --no-install-recommends
  libgomp1`), or `import lightgbm` fails inside the container. Use `xgboost-cpu` (already the
  dependency) to keep the image small.

## Results so far

**Full results with tables, figures and limitations: [docs/results.md](docs/results.md)** (status 06.11.2026).
The sections below are the short version.

Q1 (will Component X need a repair within 48 time steps?) on the SCANIA **training** split:
5-fold cross validation, folds grouped by truck, 5 random cut-offs per truck, PR-AUC
(average precision). The test split has not been used.

| Model | Validation PR-AUC | Training PR-AUC | Gap |
|---|---|---|---|
| Dummy (always the prior, = share of positives) | 0.0276 | 0.0276 | 0 |
| Logistic regression, 106 raw features | 0.1014 ± 0.0047 | 0.1094 | +0.008 |
| + truck specifications | 0.1013 ± 0.0069 | 0.1147 | +0.013 |
| + `log1p` on counters and histogram counts (**linear reference**) | 0.1099 ± 0.0055 | 0.1254 | +0.016 |
| XGBoost, 300 rounds, untuned (no early stopping) | **0.1335 ± 0.0080** | 0.5147 | +0.381 |
| XGBoost + engineered features | **0.1435 ± 0.0080** | 0.6124 | +0.469 |

- The linear reference is about 4× better than the Dummy, but its score is **low**: a linear
  model is limited by bias (learning curve, `docs/problem_framing.md` §6). Tree ensembles
  (random forest, HistGradientBoosting, XGBoost, LightGBM) are clearly better (+0.021 to
  +0.024, better in 5 of 5 folds) and indistinguishable from each other; they are not tuned
  yet, and their large train/validation gap shows it.
- `log1p` is the only change so far that wins in every fold (+0.0086 ± 0.0053, 5/5 folds).
  Engineered features (counter rates, histogram shares), SMOTE, and the specifications gave
  no real gain for the linear model.
- **The scores above come from a random-row cross validation and are too optimistic.** On a
  validation-like evaluation set (same `time_step` distribution and positive rate as the
  official validation) XGBoost scores about **0.05–0.09** PR-AUC and the logistic regression
  0.04–0.08, with the same ranking (`docs/problem_framing.md` §6, "Evaluation protocol"). The
  official validation histories are longer than a uniform cut of training histories would give
  (82 % vs. about 51 %), so the `time_step` shift is a property of the data. From here on,
  models are scored on the validation-like set; the official validation is the real check.
- A **candidate model** is registered in the local MLflow registry as `failsense-q1` (alias
  `candidate`): XGBoost with 125 engineered features, untuned (a hyperparameter search with Optuna
  found nothing clearly better). On the validation-like evaluation set it reaches PR-AUC 0.057 and
  ROC-AUC 0.66; alerting the riskiest 5 % of the trucks finds 15 % of the failures at 8 % precision
  (about 3 × the base rate).
- **First look at the official validation** (136 of 5,046 trucks positive, test untouched):
  PR-AUC 0.065 (95 % interval 0.048–0.102), ROC-AUC 0.70. The raw probabilities are about 4 × too
  high; a sigmoid calibrator fitted without validation labels brings the Brier score to the level of
  the base rate (0.026) but adds no skill over it. With the SCANIA cost matrix, "alert every truck"
  costs 34,370 (4,910 × 7); the cross-fitted cost-optimal threshold costs 28,220 (saving 6,150 =
  1.22 per truck, better in 5/5 folds), alerting only the riskiest 5 % costs 50,231. Under the
  stricter reading "alert = predicted class 1" the model is no better than never alerting
  (`make calibrate`, `docs/problem_framing.md` §6).
- **Deployment choice (06.11):** label-free sigmoid calibrator + alert threshold 0.0353 learned on the
  whole validation set; quote the cross-fitted cost 28,220, not the in-sample one. The saving depends
  on the primary cost mapping (an alert on a failing truck is free); under "alert = predicted class
  1" the model saves only 148 vs. never alerting. The test split is still untouched.
- **Explanations (SHAP, `make explain`):** on the validation only two histogram groups (459, 291) lose
  PR-AUC clearly when shuffled; the group with the largest SHAP and gain (397) does not. SHAP values are
  in raw log-odds of the uncalibrated model; `top_shap_factors` returns the top features per truck.
- Not done yet: the other questions (Q2–Q5). No target for the total
  cost is set.

**Q3 (remaining useful life) on NASA C-MAPSS FD001**, training engines only, 5-fold
`GroupKFold` by engine, RUL capped at 125 cycles, RMSE in cycles (lower is better); the
test file is untouched:

| Model | Validation RMSE | Training RMSE | MAE |
|---|---|---|---|
| Dummy (mean) | 41.67 ± 0.00 | 41.67 | 36.97 |
| Ridge (alpha 100) | 21.64 ± 1.38 | 21.44 | 17.72 |
| Lasso (alpha 0.1) | 21.63 ± 1.38 | 21.44 | 17.72 |

A linear model halves the error of the mean predictor. Ridge and Lasso are identical; Lasso
only switches off the 7 constant columns. Details in the docs; this is simulated data.

Details, decisions and the evaluation protocol: [docs/problem_framing.md](docs/problem_framing.md).

## Prediction API

```bash
make serve          # scripts/serve.sh: loads models:/failsense-q1@candidate, docs at /docs
```

* `GET /health` - status and the loaded model (name, registry version, alias).
* `POST /predict` - body: `current` readout (`time_step` + the 105 raw columns, `null` allowed),
  optional `previous` (the readout of the **same truck immediately before** `current`:
  `time_step` + the 8 counters), and the truck `specifications` (`Spec_0` ... `Spec_7`).
  Response: calibrated `risk`, `alert` (risk >= the cost-based `threshold`), model version.
  The raw model score is never returned. Without `previous` the deltas are NaN, as for the
  first readout of a truck in training. Label columns are rejected (HTTP 422).

The service refuses to start if the calibration summary was made for another model than
`models:/<tuning.registered_model>@<tuning.alias>`.

### Docker

```bash
docker build -t failsense-api .
docker run --rm -p 8000:8000 -v "$PWD/mlflow.db:/app/mlflow.db" \
  -v "$PWD/mlartifacts:$PWD/mlartifacts" failsense-api
```

Multi-stage image (uv, `--locked --no-dev`), `libgomp1` for LightGBM/XGBoost, CPU-only
`xgboost-cpu`, runs as a non-root user (uid 10001). It contains code, config and the calibration
summary, **not** the model or any data: the model is loaded from the MLflow registry at startup.
MLflow stores artifact paths as absolute URIs, hence the second mount at the host path;
`docker compose` with an MLflow server (planned) removes this detail. CI builds the image and
imports the app inside it.

## Dashboard

```bash
make serve       # terminal 1: the API
make dashboard   # terminal 2: Streamlit on http://127.0.0.1:8501
```

Pick a demo truck (a fixed sample of the **validation** split, never the test split), see the
calibrated risk, the cost-based alert threshold, the risk of the latest readouts and the
known outcome. The page never loads the model: every number comes from `POST /predict`
(`FAILSENSE_API_URL` changes the API address). Streamlit lives in the `dashboard` dependency
group (installed by `uv sync`, not part of the API image).

### Prediction log and stream simulator

Every `POST /predict` is appended to a DuckDB file (`logs/predictions.duckdb`, git-ignored, set by
`api.prediction_log`; the raw score is not stored). Only the API process opens the file;
`GET /predictions/latest?limit=20` returns the newest rows and feeds the dashboard table.

```bash
make simulate    # 200 readings, 1 per second, to the running API
uv run python scripts/simulate_stream.py --interval 0.2 --drift-start 60 --drift-factor 2
```

The simulator replays the last readouts of validation trucks (each with its previous readout) in
random order; `--drift-start` scales the counters up gradually, a fleet used harder than the
training data, which the drift check (planned) should detect. The test split is never used.

## Future work

- **Bosch CNC Machining: cross-machine anomaly detection on real vibration data.** Cut from v1.0 on
  06.11.2026 to protect the Must items; planned only after the presentation on 20.11.2026, with a
  split by machine (never random), FFT features and the same anomaly-detection setup as for SCANIA.
  Reasons and the plan to resume: [ADR 0003](docs/adr/0003-cut-bosch-cnc.md). The dataset is not
  used yet, so it is not listed under Data.

## Data

| Dataset | Used for | Source | Licence / terms |
| --- | --- | --- | --- |
| SCANIA Component X (real truck fleet) | Failure classification, cost-based threshold, anomaly detection | Swedish National Data Service (SND), [doi:10.5878/jvb5-d390](https://doi.org/10.5878/jvb5-d390) | Open access; cite the dataset and the paper below |
| Turbofan Engine Degradation Simulation (C-MAPSS, FD001) | Remaining useful life (RUL) | NASA Prognostics Data Repository, [PCoE data set page](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) (also [data.nasa.gov](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)) | **No licence stated** on the NASA pages (the catalog entry says "License not specified"); users are asked to acknowledge the repository and the data donators, and use the data at their own risk. Simulated data, not real engines |

Raw data is never committed; `scripts/download_data.sh` prepares it in `data/raw/`.

**SCANIA Component X (~1.5 GB) needs one manual download:**
open the [SND catalogue page](https://researchdata.se/en/catalogue/dataset/2024-34),
download all files as a zip, copy the zip into `data/raw/scania/` and run `make data`.
The script extracts the archive, checks that all nine files exist and verifies the SHA-256
of `train_operational_readouts.csv`.

Why real data instead of the popular synthetic AI4I 2020 dataset: see
[ADR 0001](docs/adr/0001-dataset-switch.md).

### Citations

- Kharazian, Z., Lindgren, T., Magnússon, S., Steinert, O., & Andersson Reyna, O. (2025).
  SCANIA Component X dataset: a real-world multivariate time series dataset for predictive
  maintenance. *Scientific Data*, 12, 493. https://doi.org/10.1038/s41597-025-04802-6
- Lindgren, T., Steinert, O., Andersson Reyna, O., Kharazian, Z., & Magnússon, S. (2024).
  SCANIA Component X Dataset: A Real-World Multivariate Time Series Dataset for Predictive
  Maintenance [Data set]. Scania CV AB. https://doi.org/10.5878/jvb5-d390
- Saxena, A., & Goebel, K. (2008). *Turbofan Engine Degradation Simulation Data Set*. NASA
  Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA
  (citation text as given on the NASA PCoE data set page; no DOI is stated).
- Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). Damage propagation modeling for
  aircraft engine run-to-failure simulation. *International Conference on Prognostics and
  Health Management (PHM 2008)*.

Acknowledgment: the C-MAPSS data was provided by the NASA Ames Prognostics Center of
Excellence (PCoE) through the NASA Prognostics Data Repository; thanks to NASA and the
donators of the data. NASA states that users employ the data at their own risk.
