#!/usr/bin/env bash
# Restore Aya's DB from the Cloudflare R2 replica into a NEW file (never over the live DB).
# Usage: scripts/litestream-restore.sh [output-path] [-timestamp 2026-11-14T10:00:00Z]
# Then stop the backend, move the restored file to data/sqlite/app.db and start it again.
# Needs backend/.env R2 settings and ~/.aya/litestream-sse.key (the SSE-C key).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/data/sqlite/app.restored.db}"; shift || true
[ ! -e "$OUT" ] || { echo "$OUT exists; choose another path" >&2; exit 1; }
set -a
eval "$(grep -E '^(LITESTREAM_ACCESS_KEY_ID|LITESTREAM_SECRET_ACCESS_KEY|AYA_R2_ENDPOINT|AYA_R2_BUCKET)=' "$ROOT/backend/.env")"
AYA_DB_PATH="$ROOT/data/sqlite/app.db"
set +a
mkdir -p "$(dirname "$OUT")"
"$HOME/.aya/bin/litestream" restore -config "$ROOT/scripts/litestream.yml" -o "$OUT" "$@" "$AYA_DB_PATH"
sqlite3 "$OUT" "PRAGMA integrity_check;" | head -1
echo "restored: $OUT"
