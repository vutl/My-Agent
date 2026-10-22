#!/bin/zsh
# Release rollback drill on a COPY of the production data (never the live DB).
#
#   scripts/release_rollback_drill.sh <release_sha> <baseline_sha> <work_dir>
#
# 1. back up data/sqlite/app.db, data/lancedb, data/artifacts (APFS clones) and
#    record their checksums;
# 2. start <release_sha> on a copy: health, startup migrations, one smoke turn;
# 3. code-only rollback: start <baseline_sha> on the migrated copy;
# 4. full rollback: restore the backup, check the checksums, start <baseline_sha>.
# Writes <work_dir>/drill.jsonl (one JSON line per step) and exits non-zero on
# any failed step.
set -u
cd "$(dirname "$0")/.."
REPO=$PWD; RELEASE=$1; BASELINE=$2; WORK=$3; PORT=7796
QUESTION=${DRILL_QUESTION:-"KST là viết tắt của gì?"}
mkdir -p "$WORK"; REPORT=$WORK/drill.jsonl; : > "$REPORT"
step() { print -r -- "$1" >> "$REPORT"; print -r -- "$1"; }
fail() { step "{\"step\":\"$1\",\"ok\":false,\"detail\":\"$2\"}"; stop_backend; exit 1; }
stop_backend() { kill $(lsof -ti tcp:$PORT -sTCP:LISTEN) 2>/dev/null; sleep 2; }
checksums() { (cd "$1" && find sqlite/app.db lancedb artifacts -type f -print0 | sort -z | xargs -0 shasum -a 256 | shasum -a 256 | cut -c1-16); }

tree_for() {  # a detached worktree per sha, with the local .env
  local dir=$WORK/tree-$1
  [ -d "$dir" ] || git worktree add --detach "$dir" "$1" >/dev/null 2>&1 || return 1
  cp backend/.env "$dir/backend/.env"
  print -r -- "$dir"
}

start_backend() {  # <code tree> <data dir> <log>
  stop_backend
  NO_PROXY='*' APP_DATA_DIR="$2" LIGHTRAG_ENABLED=false VISUAL_RETRIEVAL_ENABLED=false \
    DUAL_QUERY_RRF_ENABLED=false PYTHONPATH="$1" nohup "$REPO/backend/.venv/bin/python" -m uvicorn \
    app.main:app --host 127.0.0.1 --port $PORT --app-dir "$1/backend" > "$3" 2>&1 < /dev/null &
  for _ in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:$PORT/health >/dev/null && return 0; sleep 2; done
  return 1
}

probe() {  # <step name> <data dir>: health + documents + one smoke turn
  sleep 15  # startup migrations (title refresh) run in a background thread
  "$REPO/backend/.venv/bin/python" - "$1" "$2" "$PORT" "$QUESTION" <<'PY' >> "$REPORT"
import json, sqlite3, sys, urllib.request
sys.path.insert(0, "scripts")
from evaluate_agent_conversations import _run_turn
name, data_dir, port, question = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
base = f"http://127.0.0.1:{port}"
health = json.load(urllib.request.urlopen(f"{base}/health", timeout=10))
db = sqlite3.connect(f"file:{data_dir}/sqlite/app.db?mode=ro", uri=True)
documents = db.execute("select count(*) from documents").fetchone()[0]
marked = db.execute("select count(*) from documents where metadata_json like '%title_rules%'").fetchone()[0]
result = _run_turn(base_url=base, case={"case_id": name, "message": question, "expected_document_ids": [],
                   "expected_route": "file_qa"}, conversation_id=None, model=None, timeout=300, debug_trace=False)
events = [e.__dict__ if hasattr(e, "__dict__") else e for e in result["events"]]
names = [e.get("event") for e in events]
cited = sorted({c.get("filename") for e in events if e.get("event") == "answer.citations"
                for c in (e.get("data") or {}).get("citations", []) if c.get("filename")})
ok = health.get("status") == "ok" and "run.completed" in names and bool(str(result.get("answer") or "").strip())
print(json.dumps({"step": name, "ok": ok, "health": health.get("status"), "documents": documents,
                  "title_rules_marked": marked, "completed": "run.completed" in names, "cited": cited,
                  "answer_head": str(result.get("answer") or "")[:160]}, ensure_ascii=False))
PY
  tail -1 "$REPORT"
  grep -q '"ok": true' <(tail -1 "$REPORT")
}

# 1. backup
rm -rf "$WORK/backup" "$WORK/release-data" "$WORK/restored-data"
mkdir -p "$WORK/backup/sqlite"
cp -c data/sqlite/app.db "$WORK/backup/sqlite/app.db" && cp -cR data/lancedb data/artifacts "$WORK/backup/" \
  || fail backup "copy failed"
BEFORE=$(checksums "$WORK/backup")
step "{\"step\":\"backup\",\"ok\":true,\"checksum\":\"$BEFORE\",\"release\":\"$RELEASE\",\"baseline\":\"$BASELINE\"}"

# 2. release on a copy
cp -cR "$WORK/backup" "$WORK/release-data"
RTREE=$(tree_for "$RELEASE") || fail release "worktree"
start_backend "$RTREE" "$WORK/release-data" "$WORK/release.log" || fail release "backend did not start"
probe release "$WORK/release-data" || fail release "probe failed"
step "{\"step\":\"release_migration_log\",\"ok\":true,\"detail\":$(grep -c catalog_titles_refreshed "$WORK/release.log")}"

# 3. code-only rollback on the migrated data
BTREE=$(tree_for "$BASELINE") || fail rollback_code "worktree"
start_backend "$BTREE" "$WORK/release-data" "$WORK/rollback-code.log" || fail rollback_code "baseline did not start on migrated data"
probe rollback_code_only "$WORK/release-data" || fail rollback_code "probe failed"

# 4. full rollback: restore the backup
cp -cR "$WORK/backup" "$WORK/restored-data"
AFTER=$(checksums "$WORK/restored-data")
[ "$AFTER" = "$BEFORE" ] || fail restore "checksum $AFTER != $BEFORE"
step "{\"step\":\"restore\",\"ok\":true,\"checksum\":\"$AFTER\"}"
start_backend "$BTREE" "$WORK/restored-data" "$WORK/rollback-full.log" || fail rollback_full "baseline did not start on restored data"
probe rollback_full "$WORK/restored-data" || fail rollback_full "probe failed"
stop_backend
git worktree remove --force "$RTREE" >/dev/null 2>&1; git worktree remove --force "$BTREE" >/dev/null 2>&1
step "{\"step\":\"done\",\"ok\":true}"
