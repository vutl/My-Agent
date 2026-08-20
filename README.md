# My Agent

Local-first desktop AI agent app for macOS first.

**Agent / session handoff (đọc trước khi code tiếp):** [`docs/handoff/README.md`](docs/handoff/README.md) · [`AGENTS.md`](AGENTS.md)

Current implementation status:

- FastAPI app on `localhost:7777`
- `/health` backend, 9router/model-policy, embedding, vision, and memory status
- `/chat` and `/agent/run/stream` use `cx/gpt-5.6-sol` through 9router by default;
  the desktop selector also exposes `cx/gpt-5.6-terra` and `cx/gpt-5.6-luna`
- `/chat/stream` SSE chat endpoint
- SQLite chat history with conversations and messages
- `/agent/run/stream` LangGraph-backed agent stream
- Agent run and tool-call logging in SQLite
- `/files/index-folder` manual local folder indexing
- `/catalog/scan-folder` shallow file catalog and `/catalog/search`
- `/files/resolve` and `/files/read` for transient single-file access
- Logical RAG collections for selected files
- `/rag/index-file`, `/rag/index-selected-files`, `/rag/search-in-collection`
- `/rag/search`, `/rag/search-debug`, and document/chunk inspection endpoints
- Text RAG over `.txt`, `.md`, `.pdf`, and `.docx` using SQLite FTS5
- LanceDB vector indexing for document cards and text chunks
- Hybrid retrieval via `/rag/search-hybrid` using LanceDB + SQLite FTS5 RRF merge
- macOS dev scripts under `scripts/`

Current agent includes sticky paper focus, table/figure-aware RAG, evidence
validation, and durable L0/L1/L2/L3 memory.

Last verified on 2026-08-20:

- `backend/.venv/bin/python -m pytest -q` => 622 passed
- `npm run build` in `apps/desktop` => passed
- LightRAG EmbeddingGemma runtime query, production LanceDB schema/counts and
  retrieval eval passed; GPT-5.6 Sol remains the default answer/extraction LLM

## Repository boundary and secrets

This repository contains source code, tests, scripts, architecture documents,
lockfiles, and the safe configuration template `.env.example`.

It intentionally does **not** contain local paper PDFs, extracted pages and
figures, SQLite databases, LanceDB/LightRAG indexes, evaluation outputs, model
weights, Graphify output, Obsidian vaults, virtual environments, build output,
or real `.env` files. These are machine-local runtime assets and can contain
private documents, chat history, absolute paths, or provider credentials.

Create local configuration from the example and keep the resulting file
untracked:

```bash
cp .env.example backend/.env
```

Before publishing changes, verify that no ignored or secret material is staged:

```bash
git status --short
git check-ignore -v backend/.env data/rag.db graphify-out/graph.json
```

## Prerequisites

- Python 3.11+
- 9router on port `20128`
- Ollama for embeddings:

```bash
ollama pull embeddinggemma:300m
```

## Backend Dev

Recommended in this workspace:

```bash
9router -n -l -p 20128
cd backend
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 7777
```

Using `uv` when available:

```bash
cd backend
uv sync
uv run uvicorn app.main:app --reload --port 7777
```

Using `venv` and `pip`:

```bash
cd backend
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 7777
```

The launcher also honors:

```bash
CONDA_ENV_NAME=react-langchain ./scripts/start_backend.sh
```

Check the backend:

```bash
curl http://localhost:7777/health
```

Call chat:

```bash
curl -N http://localhost:7777/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"message":"Say hello briefly","model":"cx/gpt-5.6-sol"}'
```

## Desktop Dev

The current desktop app scaffold uses NPM because `pnpm` is not installed on this Mac yet.

```bash
cd apps/desktop
npm install
npm run dev
```

Open the Vite URL shown in the terminal, usually:

```bash
http://127.0.0.1:5173
```

Tauri native launch requires Rust/Cargo:

```bash
cd apps/desktop
npm run tauri -- dev
```
