#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"${ROOT_DIR}/scripts/check_ollama.sh"
"${ROOT_DIR}/scripts/check_9router.sh"
"${ROOT_DIR}/scripts/start_backend.sh"
