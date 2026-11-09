# syntax=docker/dockerfile:1
# FailSense prediction API image (multi-stage, non-root).
#
# The image holds code + config + the calibration summary, NOT the model: the model is read
# from the MLflow registry at startup (mount mlflow.db and mlartifacts, or point to a tracking
# server - see the README). Data and secrets never enter the image (.dockerignore).
#
# Build:  docker build -t failsense-api .
# Run:    docker run --rm -p 8000:8000 -v "$PWD/mlflow.db:/app/mlflow.db" \
#           -v "$PWD/mlartifacts:$PWD/mlartifacts" failsense-api

# ---- stage 1: build the virtual environment ------------------------------------------------
FROM python:3.12-slim AS builder

# uv pinned to the version the project uses (uv_build backend, see pyproject.toml).
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv

# COMPILE_BYTECODE: faster start; LINK_MODE=copy: the venv is copied to the next stage, so no
# hardlinks into the uv cache; PYTHON_DOWNLOADS=never: use the image's Python (3.12).
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# 1) dependencies only: this layer is cached until pyproject.toml / uv.lock change.
#    --locked: fail if uv.lock is out of date; --no-default-groups: no pytest/jupyter/ruff
#    (dev) and no Streamlit (dashboard) in the API image.
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --locked --no-default-groups --no-install-project

# 2) the project itself, installed as a normal (non-editable) package.
COPY src ./src
RUN uv sync --locked --no-default-groups --no-editable

# ---- stage 2: slim runtime image ------------------------------------------------------------
FROM python:3.12-slim AS runtime

# LightGBM and XGBoost need the OpenMP runtime, which python:*-slim does not ship.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Fixed uid so mounted files can be given matching permissions.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app

WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
# pyproject.toml is the repo-root marker for failsense.config; configs/ and the two small
# report files are what the service reads at startup.
COPY --chown=app:app pyproject.toml ./
COPY --chown=app:app configs ./configs
COPY --chown=app:app reports/calibration_q1_summary.json reports/registry_q1.json ./reports/

ENV PATH="/app/.venv/bin:$PATH" \
    FAILSENSE_ROOT=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER app
EXPOSE 8000

# /health answers 503 until the model is loaded; python is used because slim has no curl.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=4)"

# 0.0.0.0 inside the container; publish the port with -p to choose who can reach it.
CMD ["uvicorn", "failsense.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
