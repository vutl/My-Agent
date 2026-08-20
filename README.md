# Aya

**Aya is a local-first AI research agent for reading, understanding, and discussing technical documents.**

Aya is built for conversations that move naturally between papers, side topics,
follow-up questions, tables, figures, and comparisons—without losing track of
which document the user is talking about.

Unlike a basic “chat with PDF” interface, Aya combines a persistent
conversation state, structured document retrieval, graph navigation, visual
understanding, and evidence validation in one desktop agent.

## What Aya can do

### Talk naturally about documents

Ask questions in Vietnamese or English without repeatedly restating the full
paper name. Aya keeps track of the active paper, recently discussed documents,
topics, and multi-paper references such as “the previous paper”, “those two”,
or “compare it with ASPIRE”.

You can leave the paper discussion, talk about something else, and return later
without manually rebuilding the context.

### Retrieve the right kind of evidence

Aya does not treat every question as plain text search. It can retrieve:

- passages and section-level context;
- canonical tables and individual result rows;
- figures, architecture diagrams, and page images;
- dataset, method, contribution, training, ablation, and limitation evidence;
- evidence from multiple papers for comparison or synthesis.

Exact table and figure requests are resolved against canonical document
artifacts instead of guessed from model-generated text.

### Understand papers beyond text chunks

Documents are parsed into structural parent–child chunks, tables, figures, and
page-aware artifacts. Figure descriptions are enriched with the paper summary,
page context, nearby captions, and surrounding content so a cropped panel or a
publisher logo is less likely to be mistaken for a meaningful research figure.

Aya uses LightRAG as a navigation graph and grounds final answers back to
canonical document evidence. Graph entities help find relationships; they are
not silently promoted into factual evidence.

### Compare multiple papers without mixing them up

Multi-paper questions are retrieved and validated per document. Aya preserves
document ownership for metrics, tables, methods, and claims, then synthesizes
the comparison only after the required papers are covered.

This prevents a baseline mentioned inside one paper from being mistaken for the
active paper, and prevents a number from one result table from being assigned to
another model or document.

### Refuse unsupported quantitative claims

Aya is designed to abstain when the retrieved evidence cannot support a number,
metric, parameter count, or comparison. Its evidence validator understands
scientific-table structures, metric aliases, hierarchical headers, and common
Markdown/Docling extraction formats.

The goal is simple: **retrieve the evidence, preserve its ownership, then
answer—never invent a plausible-looking result.**

### Remember across conversations

Aya's memory is split into complementary layers:

- **L0 — full history:** durable messages and searchable past episodes;
- **L1 — working state:** active papers, recent document threads, and topic;
- **L2 — conversation memory:** rolling summary, pending turns, and relevant
  recent context;
- **L3 — long-term memory:** validated semantic, episodic, and procedural
  memories with provenance.

Memory folding runs in the background and is designed to survive restarts,
concurrent messages, quota errors, and delayed summaries without dropping the
raw turns that have not been folded yet.

## How Aya works

```text
User
  │
  ▼
Conversation state + intent + document scope
  │
  ├── Canonical tables / figures / pages
  ├── Hybrid text retrieval (FTS5 + LanceDB)
  ├── Parent–child structural context
  ├── Evidence cards by paper facet
  └── LightRAG graph navigation
  │
  ▼
Per-document evidence validation
  │
  ▼
GPT-5.6 Sol through 9router
  │
  ▼
Grounded answer + sources + visual/table artifacts
```

The default generation model is `cx/gpt-5.6-sol`. Aya also exposes
`cx/gpt-5.6-terra` and `cx/gpt-5.6-luna`, but it does not silently switch to a
different answer model when the selected provider fails.

Local EmbeddingGemma embeddings power the current vector index. Ollama is used
for embedding infrastructure, not as an automatic replacement for the main
agent model.

## Architecture

| Layer | Technology |
|---|---|
| Desktop | Tauri, React, TypeScript, Vite |
| Agent API | FastAPI, LangGraph, SSE streaming |
| Main model gateway | 9router, default `cx/gpt-5.6-sol` |
| Retrieval | SQLite FTS5, LanceDB, LightRAG |
| Embeddings | EmbeddingGemma-300M |
| Document processing | Docling-based text, table, page, and figure pipeline |
| Memory and provenance | SQLite |

Aya is currently a controlled research-agent pipeline rather than an
unrestricted autonomous tool loop. This keeps document scope, evidence
boundaries, and model/provider behavior explicit while more tools are added.

## Run Aya locally

Requirements:

- Python 3.11+
- Node.js and npm
- 9router on port `20128`
- Ollama with `embeddinggemma:300m`

```bash
# Terminal 1 — model gateway
9router -n -l -p 20128

# Terminal 2 — backend
cd backend
cp ../.env.example .env
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 7777

# Terminal 3 — desktop web UI
cd apps/desktop
npm install
npm run dev
```

Aya keeps local papers, extracted artifacts, indexes, chat databases, model
weights, and real environment files outside version control.

## Project status

The current implementation includes document-aware conversation routing,
L0–L3 memory, structured and graph-assisted RAG, table/figure delivery,
multi-paper evidence boundaries, validated streaming, debug traces, and an
evaluation harness for conversational and multimodal document QA.

The latest verified backend suite passes **622 tests**, and the desktop
production build is green.
