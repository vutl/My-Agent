#!/usr/bin/env bash
set -euo pipefail

API_BASE="${OPENAI_API_BASE:-http://localhost:20128/v1}"
API_KEY="${OPENAI_API_KEY:-any}"
MODEL="${ROUTER_MODEL:-${DEFAULT_MODEL:-cx/gpt-5.5}}"

echo "Checking 9router at ${API_BASE%/}"
MODELS_JSON="$(
  curl -fsS "${API_BASE%/}/models" \
    -H "Authorization: Bearer ${API_KEY}"
)"

if grep -Fq "\"id\":\"${MODEL}\"" <<<"${MODELS_JSON}"; then
  echo "9router model is available: ${MODEL}"
else
  echo "9router is reachable, but the configured router model is missing: ${MODEL}"
  echo "Start/configure 9router on ${API_BASE%/} with model ${MODEL}"
  exit 2
fi
