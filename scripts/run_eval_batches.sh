#!/bin/zsh
# Run an Aya evaluation dataset in conversation batches with a 9router quota
# wait, re-running batches that aborted on provider failures, then merge the
# rows and write answers.jsonl for tools/harness/judge.py.
#
#   scripts/run_eval_batches.sh <dataset.jsonl> <base_url> <out.json> [batch_size]
set -u
cd "$(dirname "$0")/.."
DATASET=$1; BASE_URL=$2; OUT=$3; SIZE=${4:-8}
BD=${OUT%.json}-batches; mkdir -p "$BD"
KEY=$(grep -E '^OPENAI_API_KEY=' backend/.env | cut -d= -f2-)
MODEL=$(grep -E '^DEFAULT_MODEL=' backend/.env | cut -d= -f2-)
wq() {
  until curl -s -m 60 http://127.0.0.1:20128/v1/chat/completions -H 'Content-Type: application/json' \
      -H "Authorization: Bearer $KEY" -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"max_tokens\":5}" \
      | grep -q '"choices"'; do
    echo "quota wait $(date +%T)"; sleep 120
  done
}
backend/.venv/bin/python - "$DATASET" "$BD" "$SIZE" <<'PY'
import collections, json, sys
dataset, batch_dir, size = sys.argv[1], sys.argv[2], int(sys.argv[3])
conversations = collections.OrderedDict()
for line in open(dataset, encoding="utf-8"):
    if line.strip():
        row = json.loads(line)
        conversations.setdefault(row["conversation"], []).append(line.strip())
names = sorted(conversations)
for index in range(0, len(names), size):
    with open(f"{batch_dir}/b{index // size:02d}.jsonl", "w", encoding="utf-8") as handle:
        handle.write("".join("\n".join(conversations[name]) + "\n" for name in names[index:index + size]))
PY
for attempt in 1 2 3; do
  for batch in "$BD"/b*.jsonl; do
    out=${batch%.jsonl}.json
    [ -f "$out" ] && ! grep -q '"aborted"' "$out" && continue
    wq; echo "== $(basename "$batch") $(date +%T)"
    backend/.venv/bin/python scripts/evaluate_aya_acceptance.py --dataset "$batch" --base-url "$BASE_URL" \
      --output "$out" 2>&1 | grep -E '^\[|ABORT'
  done
done
backend/.venv/bin/python - "$BD" "$OUT" <<'PY'
import glob, json, sys
batch_dir, out = sys.argv[1], sys.argv[2]
rows = []
for path in sorted(glob.glob(f"{batch_dir}/b*.json")):
    report = json.load(open(path, encoding="utf-8"))
    rows += report["rows"]
    if report.get("aborted"):
        print("ABORTED", path)
json.dump({"rows": rows}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
with open(out.replace(".json", ".answers.jsonl"), "w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps({
            "conversation": row["conversation"], "turn": row["turn"], "message": row["message"],
            "answer": row["answer"], "attached_figures": len(row.get("retrieved_figures") or []),
            "attached_tables": len(row.get("retrieved_tables") or []),
        }, ensure_ascii=False) + "\n")
print("merged", len(rows), "turns;", sum(row["passed"] for row in rows), "structural pass")
PY
