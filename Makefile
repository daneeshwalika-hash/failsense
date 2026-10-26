.PHONY: setup data test lint

setup:  ## Install the environment and git hooks
	uv sync
	uv run pre-commit install

data:   ## Download NASA C-MAPSS and prepare SCANIA Component X in data/raw/
	bash scripts/download_data.sh

test:   ## Run the test suite
	uv run pytest

lint:   ## Lint and format check
	uv run ruff check .
	uv run ruff format --check .
