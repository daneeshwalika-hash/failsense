#!/usr/bin/env bash
# Start the FailSense stack (MLflow server, seed, API, dashboard) with docker compose.
# A clean machine works: the one-shot `seed` service puts the bundled model into an empty registry.
# Needs: Docker running; the three validation Parquet files in ./data/processed/scania (make parquet),
# which are the only data files mounted into the dashboard container (read-only).
# Optional: .env (see .env.example) for OLLAMA_BASE_URL.
set -euo pipefail

cd "$(dirname "$0")/.."

docker compose up -d --build

echo "API:       http://127.0.0.1:8000/docs"
echo "Dashboard: http://127.0.0.1:8501"
echo "MLflow:    http://127.0.0.1:5000"
echo "Logs:      docker compose logs -f api"
echo "Stop with: docker compose down   (add -v to delete the volumes)"
