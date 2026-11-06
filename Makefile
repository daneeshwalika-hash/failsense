.PHONY: help setup data parquet validate train cv regularization compare tune evaluate register calibrate rul-baseline analysis test leakage lint

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

tune:  ## Optuna search for XGBoost and LightGBM on the validation-like eval set (slow, see make help)
	uv run python -m failsense.models.tune

evaluate:  ## Tuned vs. untuned models on both eval sets: PR-AUC, ROC-AUC, Brier, top-5 % alerts
	uv run python -m failsense.models.evaluate

register:  ## Fit the winner on the training split and register it as failsense-q1 (alias candidate)
	uv run python -m failsense.models.register

calibrate:  ## Calibrate failsense-q1@candidate and choose the alert threshold by cost on the OFFICIAL validation
	uv run python -m failsense.models.calibrate

rul-baseline:  ## C-MAPSS FD001 RUL: Dummy / Ridge / Lasso baselines, GroupKFold by engine, MLflow
	uv run python -m failsense.rul.baseline

analysis:  ## Outlier analysis and time_step check (writes CSV/Markdown/figures to reports/)
	uv run python -m failsense.analysis.outliers
	uv run python -m failsense.analysis.time_step_check

test:   ## Run the test suite (no raw data needed)
	uv run pytest

leakage:  ## Run only the data-leakage tests
	uv run pytest -m leakage

lint:   ## Lint and format check
	uv run ruff check .
	uv run ruff format --check .
