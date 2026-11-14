#!/usr/bin/env bash
# Snapshot Aya's local data/ into ~/Documents/Aya-Backup/<timestamp>/.
# - SQLite databases are copied with the online backup API (consistent while the backend runs).
# - Everything else is rsynced; unchanged files are hard-linked to the previous snapshot,
#   so each snapshot costs only what changed.
# - Re-downloadable or bulky runtime parts are skipped: models, logs, caches.
# - Keeps the newest $AYA_BACKUP_KEEP snapshots (default 14).
# Usage: scripts/backup-data.sh          (run it before any merge into main, too)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/data"
DEST="${AYA_BACKUP_DIR:-$HOME/Documents/Aya-Backup}"
KEEP="${AYA_BACKUP_KEEP:-14}"
PY="$ROOT/backend/.venv/bin/python"

[ -d "$SRC" ] && [ ! -L "$SRC" ] || { echo "data/ is missing or is a symlink; refusing to back up" >&2; exit 1; }
mkdir -p "$DEST"
chmod 700 "$DEST"
STAMP="$(date +%Y%m%d-%H%M%S)"
TARGET="$DEST/$STAMP"
PREVIOUS="$(ls -1d "$DEST"/2* 2>/dev/null | tail -1 || true)"

rsync -a --delete \
  --exclude 'models/' --exclude 'logs/' --exclude '__pycache__/' --exclude '*.tmp' \
  --exclude '*.db' --exclude '*.db-wal' --exclude '*.db-shm' --exclude '*.sqlite*' \
  ${PREVIOUS:+--link-dest="$PREVIOUS"} "$SRC/" "$TARGET/"

# Consistent copies of every SQLite database, even while it is being written.
find "$SRC" \( -name '*.db' -o -name '*.sqlite' -o -name '*.sqlite3' \) -type f -not -path '*/models/*' -print0 |
while IFS= read -r -d '' db; do
  out="$TARGET/${db#"$SRC"/}"
  mkdir -p "$(dirname "$out")"
  "$PY" - "$db" "$out" <<'EOF'
import sqlite3, sys
src = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
dst = sqlite3.connect(sys.argv[2])
src.backup(dst)
dst.close(); src.close()
EOF
done

COUNT="$(ls -1d "$DEST"/2* 2>/dev/null | wc -l | tr -d ' ')"
if [ "$COUNT" -gt "$KEEP" ]; then
  ls -1d "$DEST"/2* | head -n "$((COUNT - KEEP))" | while read -r old; do rm -rf "$old"; done
fi
echo "backup: $TARGET ($(du -sh "$TARGET" | cut -f1))"
