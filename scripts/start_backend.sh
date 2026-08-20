#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}/backend"

PORT="${APP_PORT:-7777}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-react-langchain}"

if [ -x ".venv/bin/uvicorn" ]; then
  echo "Starting backend with backend/.venv"
  exec .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port "${PORT}"
fi

if command -v uv >/dev/null 2>&1; then
  echo "backend/.venv is unavailable; starting with uv"
  exec uv run uvicorn app.main:app --host 127.0.0.1 --port "${PORT}"
fi

if command -v conda >/dev/null 2>&1; then
  if conda run -n "${CONDA_ENV_NAME}" python -c "import fastapi, uvicorn, lightrag" >/dev/null 2>&1; then
    echo "backend/.venv and uv are unavailable; starting with conda env ${CONDA_ENV_NAME}"
    exec conda run --no-capture-output -n "${CONDA_ENV_NAME}" \
      python -m uvicorn app.main:app --host 127.0.0.1 --port "${PORT}"
  fi
fi

echo "No project environment found; trying system python3"
exec python3 -m uvicorn app.main:app --host 127.0.0.1 --port "${PORT}"
