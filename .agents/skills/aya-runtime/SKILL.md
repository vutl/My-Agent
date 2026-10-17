---
name: aya-runtime
description: Start, restart and check Aya's services (FastAPI backends on :7777/:7790/:7791/:7792, desktop frontend on :5173, 9router on :20128, Ollama embeddings). Use before running evals or reproducing a chat bug, when a port is down, when 9router returns 401/429, or when you need the isolated eval backend environment.
---

# Aya runtime

Repository root: the directory containing `AGENTS.md`. Run commands from it.

## Services

| Port | What | Data |
|---|---|---|
| 20128 | 9router (Aya's only LLM gateway: answers `cx/gpt-6.1-sol`, router `cx/gpt-6-luna`) | `~/.9router/db/data.sqlite` |
| 11434 | Ollama, embeddings only (`embeddinggemma:300m`) | — |
| 7777 | main backend (uses `backend/.env`) | `data/sqlite/app.db` |
| 7790 | eval clone | `data/retrieval_eval/internal/runtime/e2e-foundation-20261005` |
| 7791 | sealed v1 | `data/retrieval_eval/sealed/v1/runtime` |
| 7792 | sealed v2 | `data/retrieval_eval/sealed/v2/runtime` |
| 5173 | desktop frontend (Vite) | — |

Health: `curl -s http://127.0.0.1:<port>/health`.

## Start an isolated eval backend

```bash
NO_PROXY='*' APP_DATA_DIR="$PWD/<runtime dir>" AGENT_DEBUG_TRACE_ENABLED=true \
LIGHTRAG_ENABLED=false VISUAL_RETRIEVAL_ENABLED=false DUAL_QUERY_RRF_ENABLED=false \
nohup backend/.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port <port> \
  --app-dir backend > <runtime dir>/../backend-<port>.log 2>&1 &
```

Restart after every code change (`kill $(lsof -ti tcp:<port> -sTCP:LISTEN)` first);
a backend started before a commit runs the old code.

Frontend: `cd apps/desktop && VITE_BACKEND_URL=http://127.0.0.1:7790 npm run dev -- --host 127.0.0.1 --port 5173 --strictPort`.

## 9router

- Off → start it yourself (user's standing instruction); if it asks for a
  password use `123456`. Headless (plain nohup exits at once):
  `nohup sh -c 'tail -f /dev/null | 9router -n -l -p 20128 -H 127.0.0.1 --skip-update' > <log> 2>&1 &`
- Test calls need `Authorization: Bearer $OPENAI_API_KEY` from `backend/.env`.
- Account health without reading tokens:
  `sqlite3 -readonly ~/.9router/db/data.sqlite "select email,isActive,updatedAt from providerConnections where provider='codex'"`;
  per-call results in table `usageHistory`. `401` = the account needs a
  re-login (user action); `429 usage limit … reset after Nm` = wait.
- Development agents do not use 9router (see the agent-harness skill).

## Tests and graph

- `cd backend && PYTHONPATH=.. .venv/bin/python -m pytest -q` (PYTHONPATH is required).
- After code changes: `~/.local/share/uv/tools/graphifyy/bin/graphify update .`.
- Debug a single chat turn: POST `/agent/run/stream` with `"debug_trace": true`
  (see `scripts/evaluate_agent_conversations.py::_run_turn`), then
  GET `/agent/runs/<run_id>/debug-trace` for the final prompt and routing.
