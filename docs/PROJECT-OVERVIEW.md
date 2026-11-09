# Aya: project overview

*Status as of 2026-11-09. Every number on this page is measured, and the evidence lives in the
repository: [`evals/results/`](../evals/results/README.md), [`docs/handoff/`](handoff/) and the
test suites.*

Aya is a personal AI assistant for the desktop. It started as "chat with my papers" and grew
into an agent that does three things:
- answers questions over a private document library, refusing to state a number it cannot
  trace to a source;
- carries out multi-step tasks with 30 tools, in the background, across restarts and with
  approvals for anything risky;
- does all this as a character: an animated Live2D anime girl who reads along as she answers
  and speaks in one of 174 voices.

It was designed, specified, evaluated and released by one developer. The developer led a team
of AI coding agents through a plan → implement → test → review harness, and nothing shipped
until it passed a sealed acceptance test.

## At a glance

| | |
|---|---|
| Document QA accuracy | **85.3% / 86.5%** correct on two sealed, independently written test sets (95 + 104 questions, multi-format corpora), **0 fabricated numbers** |
| Agent task success | **27 / 30 (90%)** on a sealed multi-step task suite, **0 unauthorised side effects**, prompt-injection tasks included |
| Tools | 30, among them web search and reading, files, sandboxed Python and shell, reminders, Apple Calendar, Reminders and Notes, apps and links, Spotify, long-term memory, notifications, asking the user, and searching the user's library |
| Voices | 174 (25 Vietnamese, 13 Vietnamese with anime-character timbre, 136 Japanese VOICEVOX), streamed sentence by sentence |
| Automated tests | 2,268 backend + 41 voice-sidecar tests |
| Code | ≈ 70k lines of Python in the backend, ≈ 41k lines of backend tests, ≈ 24k lines of evaluation and ops scripts, ≈ 10k lines of TypeScript/CSS in the desktop app; 668 tracked files, 251 commits since 2026-08-20 |
| Latency | streamed answers and early tool cards; document-free turns can skip intent routing; first text 2–3 s instead of 5–7 s in testing (behind a flag) |

## What makes it different

1. **Measured honesty.** "Don't make things up" is a test, not a promise:
   - a number in an answer must be traceable to the cited evidence, and a claimed action must
     have a matching tool event;
   - quality is measured on sealed test sets. An independent agent writes each set, the
     developer never sees it before the single run, and a model other than the answer model
     judges the answers;
   - every set is used once and then retired to regression duty.
2. **Documents, not chunks.** Papers are parsed into structure (sections, parent–child chunks,
   canonical tables, figures, page images). A question about "Table 3" or "the architecture
   figure" gets the real artifact, not a paraphrase. Evidence is validated per document, so a
   comparison of two papers cannot borrow a number from the wrong one.
3. **Conversation that keeps its place.** Aya tracks which paper is under discussion. She
   resolves "that one", "the first paper" and "go back to it", and asks when a reference is
   ambiguous. She can take a detour to a side topic and come back without losing the thread.
4. **An agent that finishes.** Goals become persistent tasks with a plan, step budgets,
   approvals, a resume after a backend restart, and a report at the end. A reply to Aya's
   question continues the waiting task instead of starting over.
5. **A presence.** A Live2D character turns her head to read the answer as it streams. She
   lip-syncs to the real audio level and nods. She reacts with expressions for thinking,
   working, waiting for approval and finishing. A voice picker offers Vietnamese and Japanese
   voices.

## Capabilities in depth

### Grounded document question answering (RAG v1, released 2026-10-22)
- **Ingestion:** Docling-based parsing of PDF, DOCX, Markdown and text into structural
  parent–child chunks, canonical tables (row level), figures with quality checks, and page
  images. Figure descriptions come from a vision model, enriched with the caption, page context
  and paper summary. Per-document cards hold title, facets and language.
- **Retrieval:**
  - hybrid search (SQLite FTS5 + LanceDB vectors with EmbeddingGemma-300M) and fusion;
  - optional reranking;
  - a LightRAG knowledge graph used only for navigation, never promoted to evidence;
  - Vietnamese text normalisation;
  - query rewriting for vague follow-ups.
- **Scope:** a catalog resolves documents by title, file name or description. A
  conversation-state machine keeps sticky focus. The intent router decides chat / document /
  agent, and asks when "the other one" could mean several documents.
- **Answering:** evidence obligations per question type (dataset, method, results, ablation,
  limitation…). Context composition is per document with citation labels `[S#]`. Answers attach
  tables and figures and render formulas (KaTeX).
- **Grounding:** a numeric grounding layer normalises numbers (Vietnamese decimal commas,
  percentages, thousands) and checks them against the cited evidence. An answer verifier
  repairs unsupported claims instead of letting them through.
- **Release discipline:** sealed v4/v5 acceptance, 60/60 on the acceptance regression set, a
  release record, a database backup and a rehearsed rollback drill.

### Agent core (P1, gate passed 2026-10-29)
- **Tasks:** stored in SQLite with a state machine, leases and an event log. They run as
  plan → execute → re-plan with step, tool-call and time budgets. A background worker continues
  tasks across restarts and notifies the user on completion (in-app, macOS, Telegram).
- **Interaction primitives:**
  - `ask_user` with option cards; the answer resumes the same task;
  - `remember` / `recall` / `forget` for long-term memory;
  - `notify`;
  - an artifact store, so long tool outputs stay readable without flooding the context.
- **Tool loop:**
  - argument validation and safe repair of malformed JSON;
  - per-tool rate limits;
  - approval cards for risky actions;
  - untrusted-content wrapping, so a web page or file cannot instruct Aya;
  - streamed answers, and tool cards that appear while the model is still writing the call.
- **Capability awareness:** Aya's description of what she can do is generated from the live
  tool registry, so it stays true as tools are added.
- **Router bypass** (behind a flag): a deterministic gate sends document-free turns straight to
  the main model. Two hand-off tools return document questions (`ask_library`) or multi-step
  goals (`plan_task`) to the established pipelines. It was validated by:
  - replaying 370 evaluation turns through the gate;
  - a 60-probe routing check;
  - a full RAG re-run with no accuracy loss.

### Tools (30)
- **Web:** search; fetch with readable-text extraction and an SSRF guard.
- **Files** (workspace-scoped; secrets never readable): list, read, search, write, edit, move,
  trash (reversible).
- **Code:** Python and shell in a macOS Seatbelt sandbox, with no network and no access to
  secrets.
- **Reminders:** Aya's own reminders, a durable scheduler that survives restarts; plus Apple
  Reminders, Apple Calendar and Apple Notes.
- **Apps:** open apps and links; Spotify control.
- **Assistant:** ask the user, remember / recall / forget, notify, read stored artifacts,
  search the library.
- **Channels:** the desktop app, macOS notifications, and a Telegram bot that answers only its
  owner.

### Memory
- L0: full history.
- L1: the working state (active documents and topic).
- L2: rolling conversation summaries, folded in the background.
- L3: long-term memory with provenance and consent-aware tools.

### Character and voice
- **Character:**
  - Live2D (Cubism 4, Mao model) with fill-to-stage layout, zoom and pan, and supersampled
    rendering;
  - gaze that follows the text being written, then spoken;
  - mouth driven by the audio level through the model's lip-sync parameters, with vowel shapes;
  - idle animations, dozing, and reactions to clicks.
- **Voice sidecar** (its own process and environment):
  - VieNeu-TTS v3 Turbo on ONNX/CPU for Vietnamese; VOICEVOX core for Japanese, with the
    answer translated before it is spoken;
  - voice cloning for Vietnamese speech in the timbre of anime characters, with each
    character's credit and terms respected;
  - ready about 10–13 s after start, against minutes for the previous design;
  - a segmenter speaks sentence by sentence while the answer streams, and never cuts inside
    code, math or URLs.

## Architecture

```
Desktop (Tauri · React · TypeScript)
  chat · Live2D stage · voice picker · task cards · approvals · notifications
        │  SSE
        ▼
FastAPI backend ───────────────────────────────────────────────────────────────
  intent router / bypass gate ─► document pipeline (scope → retrieval → evidence → answer → grounding)
                              └► agent tool loop / task runner (plan → tools → approvals → report)
  memory L0–L3 · task worker · reminder scheduler · Telegram channel · TTS supervisor
        │                                │                         │
        ▼                                ▼                         ▼
LLM gateway (9router):           Local data:                   Voice sidecar:
  cx/gpt-6.1-sol answers           SQLite (history, tasks,        VieNeu-TTS (ONNX)
  cx/gpt-6-luna routing            memory, FTS5), LanceDB,        VOICEVOX core
                                   LightRAG graph, artifacts,
                                   Ollama EmbeddingGemma
```

**Local-first, precisely.** The documents, indexes, embeddings, memory, task state, reminders
and voice all run on the user's Mac. Only the language-model calls go through the gateway to a
hosted model.

## How it was built and verified

- **Specification first:**
  - `docs/plans/agent/` holds the roadmap (P1–P5), the agent core, the tool catalog, skills,
    safety, presence, evaluation and execution plans;
  - `docs/handoff/` keeps a running status, a done list, a TODO/issues list and a failure
    ledger;
  - every work package has an approved plan with tests and acceptance criteria.
- **Multi-agent engineering harness (`tools/harness/`):**
  - a head agent (Claude) writes plans, reads every diff and runs the gates;
  - Codex agents (gpt-6.1-sol / astra / luna) and Claude Sonnet/Opus implement and review in
    isolated git worktrees, in plan → execute → test → review rounds;
  - the harness rotates between two accounts, parks an account on auth errors and persists
    its logs;
  - nothing merges itself.
- **Evaluation as a product feature:**
  - five generations of sealed RAG sets and an acceptance set with hard and fuzzy variants;
  - numeric-grounding probes, a sealed agent suite and a routing probe set;
  - judges that are independent of the answer model;
  - Wilson confidence intervals;
  - failure classes (retrieval / routing / generation) that drive fixes;
  - a rule: fix generic mechanisms only, never hard-code an answer.
- **Operations:**
  - `scripts/aya-up.sh` starts and supervises all services;
  - isolated evaluation backends never touch production data;
  - release records with a backup and a rollback drill;
  - a regression test is required to fail without its fix.

## Timeline

- **2026-08-20:** first commit.
- **August–October:** document pipeline, memory, routing and grounding, iterated against
  sealed sets v1 to v5.
- **2026-10-22:** RAG v1 released to production after sealed v4/v5 (85.3% / 86.5%, 0
  fabricated).
- **Late October:** assistant tools, sandbox, reminders, macOS integrations, Live2D stage,
  voice sidecar (174 voices), lip sync and reading gaze.
- **2026-10-25 → 10-29:** agent core P1 (tasks, background worker, resume, `ask_user`, memory
  tools, budgets, streaming, eval runner). The P1 gate passed at 27/30 with 0 unauthorised
  effects.
- **2026-10-28/29:** router bypass, validated without RAG regression.

## What is next

- **Deep research** ([plan](plans/agent/08-DEEP-RESEARCH.md)), on the level of ChatGPT, Gemini
  and Claude:
  - clarifying questions and an editable research plan;
  - parallel research workers over web, papers (arXiv, Semantic Scholar, OpenAlex), PDFs and
    the user's library;
  - evidence cards with verbatim quotes;
  - expansion rounds driven by gaps and contradictions;
  - a long cited report with claim-level verification;
  - sources kept as a queryable collection;
  - gated by a sealed 20-question suite.
- **P2–P5:**
  - browser automation, documents (docx/xlsx/pdf), mail and messages, MCP;
  - skills and scheduled workflows (daily brief, paper digest, inbox triage);
  - voice input and emotion-aware presence;
  - an audit UI, undo, and packaging.
- **Known open issues:** see [`handoff/05-TODO-AND-ISSUES.md`](handoff/05-TODO-AND-ISSUES.md).
  Examples: an intermittent stall on a gateway call, Japanese voice latency (a translation step
  per sentence), and answers to questions about documents missing from the library that
  sometimes come from the web instead of a "not in your library" reply.

## Repository map

| Path | What |
|---|---|
| `backend/app/rag/`, `services/rag_service.py` | parsing, chunking, retrieval, evidence, vision |
| `backend/app/api/agent/` | the chat/agent endpoint: routing, scope, document pipeline, bypass gate |
| `backend/app/agents/` | tool loop, task runner, hand-offs, LangGraph answer graph |
| `backend/app/services/` | tasks, worker, memory, grounding, verifier, reminders, speech |
| `backend/app/tools/` | the 30 tools and the registry |
| `services/tts/` | voice sidecar |
| `apps/desktop/` | Tauri + React app (chat, Live2D stage, voice picker, task cards) |
| `evals/`, `scripts/` | evaluation sets, runners, judges, ops scripts |
| `tools/harness/` | multi-agent development harness |
| `docs/plans/`, `docs/handoff/` | plans, status, decisions, failure ledger |
