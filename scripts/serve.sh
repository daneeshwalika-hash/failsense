#!/usr/bin/env bash
# Start the FailSense prediction API (loads models:/<tuning.registered_model>@<tuning.alias>).
# Host and port come from configs/config.yaml (api.host / api.port); override with
# FAILSENSE_HOST / FAILSENSE_PORT. Needs an MLflow registry with the model (make register).
set -euo pipefail

cd "$(dirname "$0")/.."

read -r HOST PORT < <(uv run python -c \
  "from failsense.config import load_config; a = load_config()['api']; print(a['host'], a['port'])")
HOST="${FAILSENSE_HOST:-$HOST}"
PORT="${FAILSENSE_PORT:-$PORT}"

echo "Serving FailSense API on http://${HOST}:${PORT} (docs: /docs)"
exec uv run uvicorn failsense.api.main:app --host "$HOST" --port "$PORT"
