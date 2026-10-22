# CODEBASE_MAP

Where things live in Aya (updated 2026-10-23). For status and history read
`docs/handoff/README.md`; for navigation questions the graph in
`graphify-out/` (`graphify query "…"`) is faster than browsing.

## Top level

```text
apps/desktop/     Tauri + React + Vite desktop app (the UI)
backend/          FastAPI server: chat, RAG, assistant tools, ingest, memory
docs/handoff/     Current status, plans, release records (read 00-STATUS first)
docs/archive/     Old plans and handoff history
evals/            Evaluation sets in git (acceptance, sealed v1–v5 questions/verdicts, release evidence)
scripts/          Eval runners, ops and maintenance scripts
tools/harness/    Agent harness: Codex/Claude roles, pipeline, batch judge
data/             Runtime data (SQLite, LanceDB, artifacts, eval corpora) — not in git
_personal/        The user's own files kept out of git
```

## Backend (`backend/app`)

| Package | What it does |
|---|---|
| `main.py` | App factory; lifespan starts memory folding, document processing workers, the reminder worker, the Telegram bot (if configured) and the catalog title refresh |
| `api/agent/` | The chat endpoint `/agent/run/stream` (SSE) and runs/approvals. `routes.py` endpoints · `stream.py` turn orchestration (`run_agent_stream`) · `scope.py` document scope and vague references · `retrieval.py` hybrid/whole-document retrieval · `answer.py` answer generation and streaming · `verification.py` post-verifier and number checks · `tables.py` direct table answers · `citations.py` [S#] citations · `assistant.py` agent-route tool turn · `deps.py` patchable dependencies · `common.py`, `models.py` |
| `api/assistant.py` | `/assistant/notifications`, `/assistant/reminders` |
| `api/{chat,files,catalog,rag,health,lightrag_routes}.py` | History, ingest jobs, catalog, RAG utilities, health |
| `agents/graph.py` | Prompt assembly for chat and RAG answers (persona, citation rules) |
| `agents/tool_loop.py` | Assistant loop: model ↔ tools, approvals (`APPROVALS`), untrusted-output wrapping, [W#] sources |
| `tools/` | First-party tools: `registry.py` · `web.py` (search/fetch, SSRF guard) · `files.py` + `policy.py` (paths, secrets) · `code.py` (Seatbelt sandbox) · `reminders.py` · `library.py` |
| `channels/telegram.py` | Telegram bot, owner only, client of the agent API |
| `services/` | Router (`tool_decision_service.py`), document scope, RAG service, indexing and document processing, memory (conversation + long-term), answer verifier, numeric grounding, reminders, legacy evidence validator |
| `rag/` | Parsing (Docling, OCR, tables, figures), chunking, retrieval, fusion, context composition, vision summaries |
| `catalog/` | Document cards and titles (+ startup title refresh) |
| `llm/openai_client.py` | 9router client: chat, streaming, `chat_with_tools`, retries |
| `db/sqlite.py` | Schema and migrations |
| `core/config.py` | Settings (env: models, ports, workspace, Telegram, CORS) |

Request flow of one chat turn: `POST /agent/run/stream` (registered in `routes.py`) →
`stream.run_agent_stream` → scope (`services/document_scope_service`) →
router (`services/tool_decision_service`) → one of:
`chat` → `answer._stream_graph_and_answer`;
`file_qa/research` → `retrieval` → `answer` (+ `verification`);
`agent` → `assistant._stream_assistant_tool_turn` → `agents/tool_loop`.

## Desktop app (`apps/desktop/src`)

| Path | What it does |
|---|---|
| `App.tsx` | Shell: sidebar (brand, nav, history slot), top bar, chat/library routes, notification toasts. Fluid layout: columns use `clamp()` against the window width |
| `routes/ChatPage.tsx` | Chat screen: streaming, tool cards, avatar state, trace; renders the history list into the sidebar (portal) and Aya's stage on the right |
| `routes/FilesPage.tsx` | Library (folders, ingest jobs, documents) |
| `components/chat/` | Messages, Markdown + KaTeX + citation chips, tool-call cards, composer |
| `components/avatar/` | `AyaStage.tsx` (right column: Aya, speech bubble, upcoming reminders, collapsible) · `Live2DAvatar.tsx` (Mao, expressions, lip sync, idle life; `fill` follows its container) · SVG fallback |
| `components/notifications/` | Reminder toasts |
| `lib/sse.ts`, `lib/api.ts` | Backend calls; `lib/types/` shared types |
| `public/live2d/` | Cubism Core and the Live2D models (see its LICENSE.md) |

## Tests and checks

- Backend: `cd backend && PYTHONPATH=.. .venv/bin/python -m pytest -q`
- Frontend: `cd apps/desktop && npx tsc --noEmit -p . && npm run build`
- RAG regression: acceptance (`evals/aya_acceptance_v1.jsonl`) and vague-ref
  variants (`evals/sealed/v1/variants`); see `docs/handoff/12-RELEASE-RAG-V1.md`.
- Release drill: `scripts/release_rollback_drill.sh <release> <baseline> <dir>`.
