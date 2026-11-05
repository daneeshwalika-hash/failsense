.PHONY: help setup data parquet validate train cv analysis test leakage lint

help:   ## Show this help
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-10s %s\n", $$1, $$2}'

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
