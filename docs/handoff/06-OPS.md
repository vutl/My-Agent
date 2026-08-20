# 06 — Ops (updated 2026-08-20)

## Ports / providers

| Service | Port | Ghi chú |
|---------|------|---------|
| Backend FastAPI | `7777` | Agent, history, files/index APIs |
| 9router | `20128` | Mặc định `cx/gpt-5.6-sol`; UI hiện thêm Terra/Luna |
| Ollama | `11434` | Chỉ embedding `embeddinggemma:300m` trong env hiện tại |

## Start khuyến nghị

```bash
# 9router
9router -n -l -p 20128

# Backend — dùng .venv có lightrag
cd backend
NO_PROXY='*' .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 7777

# Desktop
cd ../apps/desktop
npm run dev   # hoặc tauri dev7
```

Health:

```bash
curl -sS http://127.0.0.1:7777/health
curl -sS http://127.0.0.1:20128/v1/models -H 'Authorization: Bearer any'
```

`/health` phải có vision/model-policy block tương ứng `openai_compatible` /
`cx/gpt-5.6-sol`, `model_fallback_enabled=false`, cùng L2/L3-outbox pending/retry status.
Sau khi sửa `.env`, restart uvicorn; không tin process reload nếu trước đó chạy lẫn conda.

## Env chốt (`backend/.env`)

```text
DEFAULT_MODEL=cx/gpt-5.6-sol
LLM_PROVIDER=openai_compatible
OPENAI_API_BASE=http://localhost:20128/v1
OPENAI_API_KEY=any
ROUTER_MODEL=cx/gpt-5.6-sol
ROUTER_LLM_PROVIDER=openai_compatible
EMBEDDING_MODEL=embeddinggemma:300m
EMBEDDING_DIM=768
EMBEDDING_MAX_TOKEN_SIZE=2048
EMBEDDING_QUERY_PREFIX="task: search result | query: "
EMBEDDING_DOCUMENT_PREFIX="title: none | text: "
VISION_PROVIDER=openai_compatible
VISION_MODEL=cx/gpt-5.6-sol
LIGHTRAG_LLM_MODEL=cx/gpt-5.6-sol
LIGHTRAG_LLM_FALLBACK_MODELS=
LIGHTRAG_LLM_MAX_ASYNC=1
LIGHTRAG_LLM_TIMEOUT_SECONDS=300
LIGHTRAG_LLM_TIMEOUT_RETRIES=1
LIGHTRAG_CHUNK_TOKEN_SIZE=600
LIGHTRAG_CHUNK_OVERLAP_TOKEN_SIZE=80
MEMORY_FOLD_DEBOUNCE_SECONDS=12
MEMORY_WORKER_SHUTDOWN_TIMEOUT_SECONDS=3
RERANK_ENABLED=false
```

Startup hiện fail-closed nếu fixed answer/router/vision/LightRAG roles nằm ngoài
`cx/gpt-5.6-sol`, `cx/gpt-5.6-terra`, `cx/gpt-5.6-luna`; nếu provider lệch khỏi
OpenAI-compatible 9router; hoặc nếu LightRAG có fallback list. Mặc định là Sol.

## Index/enrichment workflow

- Parse/re-index lớn có thể chạy `--skip-vision` để hoàn tất staged extraction trước, rồi enrich visual sau.
- Chỉ mark document vision fingerprint hiện tại khi các eligible visuals đã được process; rate limit/partial run không được đánh dấu nhầm là complete.
- Gặp `429`, connection error hoặc provider timeout: dừng batch, giữ partial
  success, chờ provider ổn định; không tự đổi model.
- Re-index staged phải giữ old document/index slice nếu parse fail.
- LightRAG stale prune sẽ bị chặn nếu canonical index rỗng hoặc canonical docs chưa `processed`.

Privacy boundary:

- Full-corpus visual enrich gửi figure crop + page image qua 9router.
- Full canonical LightRAG ingest gửi corpus text qua 9router.
- User đã approve hai thao tác bulk ngày 2026-07-23. Visual enrichment bằng
  GPT-5.6 Sol đã hoàn tất; canonical LightRAG ingest cũng hoàn tất 21/21 ngày
  2026-07-29, không dùng model fallback.

Post-ingest LightRAG:

- 21 status records: 21 processed / 0 failed; không pending/processing/unready.
- Graph: 8.559 nodes / 14.173 edges; 7 duplicate tombstones đã prune.
- `/rag/lightrag/insert-all` preflight gateway/model, dọn canonical interrupted
  queue, xử lý từng canonical paper và abort lỗi đầu tiên.
- Adapter retry đúng một lần cùng Sol cho OpenAI/httpx timeout/network error;
  không retry 429/status/mismatch và không đổi Terra/Luna/Ollama.
- Extract role timeout 660s (worker 1.320s, health 1.335s); query role 240s.
- Audit nhanh: `GET /rag/lightrag/status`; query smoke dùng
  `POST /rag/lightrag/query`.
- Sau khi đổi embedding, rebuild VDB từ graph/KV gốc bằng
  `scripts/rebuild_lightrag_vectors.py --yes`; script dùng đúng app adapter và
  không gọi LLM/9router.

## Tests / build

```bash
cd backend
.venv/bin/python -m pytest -q

cd ../apps/desktop
npm run build
```

Full backend suite gần nhất: **622 passed**. Các suite cho durable memory,
runtime concurrency, L0/L3, model policy, conversation state/scope, evidence
guard, retrieval/table/figure, evidence cards và public/conversational evaluator
đều green. Desktop production build green.

## E2E agent conversation

```bash
./backend/.venv/bin/python scripts/e2e_agent_conversation.py
```

Harness giữ một `conversation_id`, kiểm routes/focus/sources/figures/tables, persisted history, đồng thời GET `image_url` để xác nhận HTTP 200, `Content-Type: image/*` và body không rỗng.

Strict pass gần nhất (2026-07-19): conversation `b08d1f90-c050-4d74-8c2f-b04125ba3796`, 7/7 runs completed, 14 messages, không fallback; quantitative turns check đủ Accuracy/F1/CCC. Turn 6 giữ đúng một Figure 1 architecture và image 200; turn 7 giữ `table_id=a7b425c9-d142-53bb-8945-3c11150b52de`, `table_index=1`. DB cuối chuỗi có L1 ASPIRE, recent WhiSER, `memory.revision=summary_revision=7`, recent beats 5/6/7.

## Data paths

- SQLite: `data/sqlite/app.db` — L0 messages/FTS, L1 metadata, durable L2 turns/jobs/summary và versioned L3 items
- LightRAG working dir: `data/lightrag/`
- LightRAG Obsidian snapshot: `obsidian-lightrag-vault/`; rebuild bằng
  `backend/.venv/bin/python scripts/export_lightrag_obsidian.py`
- LanceDB/vector data: theo `lancedb_path` trong settings
- Extracted artifacts: `data/artifacts/`

## Inspect L1/L2/L3 nhanh

```bash
sqlite3 "data/sqlite/app.db" \
  "SELECT id, substr(summary,1,120), substr(metadata_json,1,400) FROM conversations ORDER BY updated_at DESC LIMIT 3;"

sqlite3 "data/sqlite/app.db" \
  "SELECT status, COUNT(*), SUM(dirty_through_seq-summary_through_seq) FROM conversation_memory_jobs GROUP BY status;"

sqlite3 "data/sqlite/app.db" \
  "SELECT kind, scope, memory_key, status, confidence, source_conversation_id, source_turn_seq FROM memory_items ORDER BY updated_at DESC LIMIT 20;"

sqlite3 "data/sqlite/app.db" \
  "SELECT status, COUNT(*), SUM(attempt_count) FROM conversation_memory_l3_outbox GROUP BY status;"
```

Khi verify E2E, kiểm thêm message count/saved retrieval sources, active document,
recent document thread, job dirty/summary cursors, pending turns và L3 provenance;
không chỉ nhìn text answer.
