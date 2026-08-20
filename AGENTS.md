# AGENTS.md — My Agent

Trước khi sửa code hoặc debug RAG/chat/memory, **đọc**:

1. [`docs/handoff/README.md`](docs/handoff/README.md) — mục lục
2. [`docs/handoff/00-STATUS.md`](docs/handoff/00-STATUS.md) — snapshot
3. File chuyên đề liên quan (`02-MEMORY`, `03-RAG-AND-FOCUS`, `05-TODO-AND-ISSUES`)

## Non-negotiables (user)

- Approved 9router models: **`cx/gpt-5.6-sol` / `cx/gpt-5.6-terra` / `cx/gpt-5.6-luna`**; mặc định `sol`. Không chuyển local Ollama cho router/answer trừ khi user yêu cầu rõ.
- Giữ sticky paper focus (L1); không để follow-up benchmark nhảy sang PDF khác.
- Không hardcode mapping paper → figure number.
- Không bịa số liệu từ paper khi retrieval thiếu evidence.
- Persona: **Aya**.

## Stack nhanh

Tauri/React desktop + FastAPI (`:7777`) + 9router (`:20128`) + Ollama embed/VLM + LightRAG/LanceDB + SQLite history.

## Khi xong việc lớn

Cập nhật `docs/handoff/00-STATUS.md`, `04-DONE.md`, `05-TODO-AND-ISSUES.md` (ngày + bullet ngắn).

## graphify (AUTOMATIC & MANDATORY)

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

CRITICAL RULE: For ANY question about codebase structure, architecture, data flow, or code behavior, Agent MUST automatically query Graphify FIRST when `graphify-out/graph.json` exists. Do NOT wait for the user to type `/graphify`. Do NOT manually grep/read multiple source files before querying the graph.

Rules:
- For ALL codebase questions, immediately run `graphify query "<question>" --budget 1200` (or Graphify MCP `query_graph` with `token_budget=1200`). Increase budget only for broad architecture questions.
- Use `graphify path "<A>" "<B>"` for entity relationships and `graphify explain "<concept>"` for focused concepts.
- Treat the graph as the primary navigation layer: query it first to find the exact sub-graph / minimal set of files needed. Read raw source files ONLY to verify exact code edits, debug runtime failures, or if graph evidence is insufficient.
- `obsidian-vault/` is the human-facing view of the same generated graph. Do not bulk-read its generated notes as code context.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, fast, no API cost).
