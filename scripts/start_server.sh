#!/usr/bin/env bash
# ==============================================================================
# RallyIQ Server Launcher
# ==============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

VENV_PYTHON="$PROJECT_DIR/venv/bin/python3"
PORT="${PORT:-8000}"

echo "======================================================================"
echo " Starting RallyIQ Application Server on http://localhost:$PORT"
echo "======================================================================"

exec "$VENV_PYTHON" -m uvicorn backend.api.main:app --host 0.0.0.0 --port "$PORT" --log-level info
