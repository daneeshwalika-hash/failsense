#!/usr/bin/env bash
# Start the FailSense stack (MLflow server, API, dashboard) with docker compose.
# 1. build and start the MLflow server, 2. copy the registered model into it (verified by reloading),
# 3. build and start the API and the dashboard.
# Needs: Docker running, a local registry with failsense-q1@candidate (make register), data/ for the dashboard.
set -euo pipefail

cd "$(dirname "$0")/.."

docker compose up -d --build mlflow
echo "Publishing failsense-q1@candidate to the MLflow server ..."
uv run python -m failsense.models.publish --target http://127.0.0.1:5000
docker compose up -d --build api dashboard

echo "API:       http://127.0.0.1:8000/docs"
echo "Dashboard: http://127.0.0.1:8501"
echo "MLflow:    http://127.0.0.1:5000"
echo "Stop with: docker compose down   (add -v to delete the volumes)"
