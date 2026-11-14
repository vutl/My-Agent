#!/usr/bin/env bash
# Start Aya's services if they are not running: 9router (:20128), the backend
# (:7777, which spawns the TTS sidecar on :7801) and the desktop UI (:5173).
# Optional: local SearXNG metasearch on :8888 when installed in ~/.aya/searxng.
# Each runs in its own session (setsid) so it outlives the terminal or agent
# session that started it. Logs go to data/logs/. Usage: scripts/aya-up.sh [status]
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGS="$ROOT/data/logs"
mkdir -p "$LOGS"

listening() { lsof -ti "tcp:$1" -sTCP:LISTEN >/dev/null 2>&1; }

detach() {  # detach <log> <cwd> <command...>: new session, no stdin, survives the caller
  local log="$1" cwd="$2"; shift 2
  # exec: no wrapper shell stays behind holding the caller's stdout.
  ( cd "$cwd" && exec nohup python3 -c 'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' "$@" \
    >"$log" 2>&1 </dev/null ) >/dev/null 2>&1 &
}

wait_port() {  # wait_port <port> <seconds>
  for _ in $(seq 1 "$2"); do listening "$1" && return 0; sleep 1; done; return 1
}

status() {
  for item in "20128 9router" "11434 ollama" "8888 searxng" "7777 backend" "7801 tts" "5173 ui"; do
    set -- $item
    if listening "$1"; then echo "  ✓ $2 (:$1)"; else echo "  ✗ $2 (:$1)"; fi
  done
}

if [ "${1:-}" = "status" ]; then status; exit 0; fi

if ! listening 11434; then echo "Ollama is not running (embeddings): start the Ollama app."; fi

SEARXNG="$HOME/.aya/searxng"
if [ -x "$SEARXNG/.venv/bin/python" ] && ! listening 8888; then
  echo "starting SearXNG…"
  detach "$LOGS/searxng.log" "$SEARXNG" env SEARXNG_SETTINGS_PATH="$SEARXNG/settings.yml" "$SEARXNG/.venv/bin/python" -m searx.webapp
  wait_port 8888 30 || echo "  SearXNG did not come up; see $LOGS/searxng.log"
fi

if ! listening 20128; then
  echo "starting 9router…"
  # 9router exits at once without a stdin; keep one open.
  detach "$LOGS/9router.log" "$ROOT" sh -c 'tail -f /dev/null | 9router -n -l -p 20128 -H 127.0.0.1 --skip-update'
  wait_port 20128 30 || echo "  9router did not come up; see $LOGS/9router.log"
fi

if ! listening 7777; then
  echo "starting backend…"
  detach "$LOGS/backend-7777.log" "$ROOT" env NO_PROXY='*' backend/.venv/bin/python -m uvicorn app.main:app \
    --host 127.0.0.1 --port 7777 --app-dir backend
  wait_port 7777 90 || echo "  backend did not come up; see $LOGS/backend-7777.log"
fi

if ! listening 5173; then
  echo "starting UI…"
  detach "$LOGS/vite-5173.log" "$ROOT/apps/desktop" env NO_PROXY='*' npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
  wait_port 5173 60 || echo "  UI did not come up; see $LOGS/vite-5173.log"
fi

# The TTS sidecar is started by the backend; give it a moment.
wait_port 7801 30 >/dev/null || true
status
