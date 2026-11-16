.PHONY: help setup data parquet validate train cv regularization compare mlp tune evaluate register calibrate serve dashboard simulate compose-up compose-down publish-model retrain freeze freeze-verify rehearse drift-reference drift-check coverage explain matched-check rul-baseline rul-timeseries analysis unsupervised class-mix error-overlap q2 anomaly figures test leakage lint

help:   ## Show this help
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-15s %s\n", $$1, $$2}'

setup:  ## Install the environment and git hooks
	uv sync
	uv run pre-commit install

data:   ## Download NASA C-MAPSS and prepare SCANIA Component X in data/raw/
	bash scripts/download_data.sh

parquet:  ## Convert SCANIA CSV files to Parquet in data/processed/scania/
	uv run python -m failsense.data.convert

validate:  ## Check all SCANIA Parquet files against the data contract (needs make parquet)
	uv run python -m failsense.data.schema

train:  ## Fit the logistic-regression reference on the training split, log it to MLflow
	uv run python -m failsense.models.train

cv:     ## 5-fold CV of all baseline models on the training split, paired comparisons, MLflow
	uv run python -m failsense.models.baseline_cv

regularization:  ## L2 / L1 / ElasticNet grid, non-zero coefs, top L1 features (slow: ~17 min, saga)
	uv run python -m failsense.models.regularization_cv

compare:  ## One table: logreg / kNN / GaussianNB / trees on the same folds (kNN is slow), MLflow
	uv run python -m failsense.models.compare_models_cv

mlp:  ## MLPClassifier vs logreg and XGBoost on the same folds, paired (writes compare_mlp_q1.csv, ~3 min)
	uv run python -m failsense.models.mlp_cv

tune:  ## Optuna search for XGBoost and LightGBM on the validation-like eval set (slow, see make help)
	uv run python -m failsense.models.tune

evaluate:  ## Tuned vs. untuned models on both eval sets: PR-AUC, ROC-AUC, Brier, top-5 % alerts
	uv run python -m failsense.models.evaluate

register:  ## Fit the winner on the training split and register it as failsense-q1 (alias candidate)
	uv run python -m failsense.models.register

calibrate:  ## Calibrate failsense-q1@candidate and choose the alert threshold by cost on the OFFICIAL validation
	uv run python -m failsense.models.calibrate

matched-check:  ## One-time check matched vs k5 training sample on the OFFICIAL validation (PR-AUC + cross-fitted cost, bootstrap)
	uv run python -m failsense.models.matched_check

serve:  ## Start the prediction API (loads failsense-q1@candidate; docs at /docs)
	scripts/serve.sh

dashboard:  ## Start the Streamlit dashboard (needs `make serve` running)
	scripts/dashboard.sh

simulate:  ## Replay validation readouts to the running API, 1 per second (see --help for drift)
	uv run python scripts/simulate_stream.py --max-readings 200

compose-up:  ## docker compose: MLflow server + API + dashboard (publishes the model first); needs Docker
	scripts/compose_up.sh

compose-down:  ## Stop the compose stack (volumes are kept)
	docker compose down

publish-model:  ## Copy failsense-q1@candidate from the local registry to the MLflow server (http://127.0.0.1:5000)
	uv run python -m failsense.models.publish

retrain:  ## Champion vs challenger on a fixed holdout (ADR 0007), dry run; `uv run python -m failsense.models.retrain --promote` registers a winner
	uv run python -m failsense.models.retrain

freeze:  ## Freeze Q1/Q2/RUL for the single test evaluation (trains on TRAINING data only, clean git tree, writes reports/frozen_manifest.json)
	uv run python -m failsense.freeze

freeze-verify:  ## Verify the frozen artifacts, source files, config, test-file hashes and git state against the manifest
	uv run python -m failsense.freeze --verify

rehearse:  ## REHEARSAL of the test evaluation on the VALIDATION split: no lock, no test file, result in the log and in the temp folder
	uv run python scripts/evaluate_test_once.py --rehearse-on-validation

drift-reference:  ## Risks of the deployed model on the validation last readouts = reference for the drift check
	uv run python -m failsense.monitoring.drift reference

drift-check:  ## Compare the API's recent predictions with the reference (PSI + KS); exit 0 ok, 1 warning, 2 alert, 3 too few rows
	uv run python -m failsense.monitoring.drift check

coverage:  ## Test suite with a coverage report (terminal + coverage.xml)
	uv run pytest -q --cov --cov-report=term --cov-report=xml

explain:  ## SHAP, permutation and gain importance of failsense-q1@candidate on the official validation (needs make calibrate)
	uv run python -m failsense.explain.explain_q1

rul-baseline:  ## C-MAPSS FD001 RUL: Dummy / Ridge / Lasso baselines, GroupKFold by engine, MLflow
	uv run python -m failsense.rul.baseline

rul-timeseries:  ## C-MAPSS FD001: rolling mean/std features, Ridge vs LightGBM, GroupKFold by engine, MLflow
	uv run python -m failsense.rul.timeseries

analysis:  ## Outlier analysis and time_step check (writes CSV/Markdown/figures to reports/)
	uv run python -m failsense.analysis.outliers
	uv run python -m failsense.analysis.time_step_check

unsupervised:  ## PCA map + KMeans clusters of the training trucks (labels only for colour), figures/CSV in reports/
	uv run python -m failsense.analysis.unsupervised

error-overlap:  ## Validation only: overlap of the missed failures and false alarms of logreg, XGBoost and MLP (reports/error_overlap_q1*.csv)
	uv run python -m failsense.analysis.error_overlap

class-mix:  ## Why the calibrated eval set has another class mix than the validation (training data only)
	uv run python -m failsense.analysis.class_mix

q2:  ## Q2 classes 0-4: multi-class model, prior-shift correction (ADR 0004), 5x5 total cost on the validation (~7 min)
	uv run python -m failsense.models.q2_classes

anomaly:  ## Q4: IsolationForest + LOF on healthy trucks, anomaly score as an extra feature (ADR 0005, ~6 min)
	uv run python -m failsense.models.anomaly

figures:  ## Headline result figures (PNG in reports/figures) from the saved reports; nothing is refitted
	uv run python -m failsense.report_figures

test:   ## Run the test suite (no raw data needed)
	uv run pytest

leakage:  ## Run only the data-leakage tests
	uv run pytest -m leakage

lint:   ## Lint and format check
	uv run ruff check .
	uv run ruff format --check .
