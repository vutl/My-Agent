#!/usr/bin/env bash
set -euo pipefail

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-embeddinggemma:300m}"
VISION_MODEL="${VISION_MODEL:-qwen3-vl:4b}"

echo "Checking Ollama at ${OLLAMA_HOST}"
TAGS_JSON="$(curl -fsS "${OLLAMA_HOST%/}/api/tags")"

for MODEL in "${EMBEDDING_MODEL}" "${VISION_MODEL}"; do
  MODEL_FOUND=false
  if grep -Fq "\"name\":\"${MODEL}\"" <<<"${TAGS_JSON}"; then
    MODEL_FOUND=true
  elif [[ "${MODEL}" != *:* ]] && grep -Fq "\"name\":\"${MODEL}:" <<<"${TAGS_JSON}"; then
    MODEL_FOUND=true
  fi

  if [ "${MODEL_FOUND}" = true ]; then
    echo "Ollama model is available: ${MODEL}"
  else
    echo "Ollama is reachable, but a required embedding/VLM model is missing: ${MODEL}"
    echo "Run: ollama pull ${MODEL}"
    exit 2
  fi
done
