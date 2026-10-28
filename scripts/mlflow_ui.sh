#!/usr/bin/env bash
# Start the local MLflow UI on the project's SQLite tracking database.
# Usage: ./scripts/mlflow_ui.sh [port]      (default port: 5000)
# Stop with Ctrl+C.
set -euo pipefail

# Always run from the repo root, so the database path matches configs/config.yaml
# (mlflow.db_path) no matter where the script is called from.
cd "$(dirname "$0")/.."

PORT="${1:-5000}"
echo "MLflow UI: http://127.0.0.1:${PORT}"
exec uv run mlflow ui --backend-store-uri "sqlite:///$(pwd)/mlflow.db" --host 127.0.0.1 --port "$PORT"
