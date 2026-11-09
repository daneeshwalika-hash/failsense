#!/usr/bin/env bash
# Start the Streamlit dashboard (needs the API: scripts/serve.sh).
# The port can be changed with FAILSENSE_DASHBOARD_PORT (default 8501).
set -euo pipefail

cd "$(dirname "$0")/.."

PORT="${FAILSENSE_DASHBOARD_PORT:-8501}"
exec uv run streamlit run src/failsense/dashboard/app.py \
  --server.port "$PORT" --server.address 127.0.0.1 --browser.gatherUsageStats false
