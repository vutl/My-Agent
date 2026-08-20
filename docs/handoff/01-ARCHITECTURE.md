# 01 — Architecture

## High-level

```text
Desktop (ChatPage)
    → POST /agent/run/stream (SSE)
        → serialize same conversation; advertise global foreground busy
        → pack L1 + L3 + stable L2 + pending turns + L0 episodes + raw recent
        → IntentRouterService (GPT) → chat | file_qa | research
        → [optional] QueryRewrite + retrieve (LightRAG / LanceDB)
        → LangGraph (plan + final_prompt) — không phải ReAct loop đầy đủ
        → stream answer (GPT)
        → persist answer + L1 update (nếu RAG) + durable L2 turn/job
        → SSE complete immediately

Background after idle:
    durable L2 worker (one fold globally, selected approved GPT-5.6 only)
        → rolling summary cursor
        → atomic durable outbox for validated L3 ops
        → receipt-backed semantic/episodic/procedural materialization
        → retry/recover from SQLite after quota failure or restart
```

**Không phải ReAct đầy đủ.** Tool decision + retrieve **trước** graph; graph chủ yếu chuẩn bị plan/prompt rồi generate.

## File then chốt

| Vai trò | Path |
|---------|------|
| Agent orchestration + SSE | `backend/app/api/agent.py` |
| Intent router | `backend/app/services/tool_decision_service.py` |
| Query rewrite / follow-up heuristics | `backend/app/services/query_rewrite_service.py` |
| L1 working state | `backend/app/services/conversation_state.py` |
| L2 durable summary worker | `backend/app/services/conversation_memory.py` |
| Foreground concurrency gate | `backend/app/services/conversation_runtime.py` |
| L0 history search + L3 store | `backend/app/services/long_term_memory.py` |
| Chat history SQLite | `backend/app/services/chat_history.py` |
| LightRAG bridge | `backend/app/lightrag/` |
| RAG hybrid / visual | `backend/app/services/rag_service.py` |
| LangGraph nodes | `backend/app/agents/graph.py` |
| Config / env | `backend/app/core/config.py`, `backend/.env` |
| Desktop chat | `apps/desktop/src/routes/ChatPage.tsx` |
| SSE client | `apps/desktop/src/lib/sse.ts` |

## Plans gốc (tham khảo dài)

- `pdf/PROJECT_PLAN_ADDENDUM_RAG_AGENT_MEMORY_VISUAL.md` — memory §13, visual RAG
- `pdf/PROJECT_PLAN_ADDENDUM_STORAGE_CATALOG_LANCEDB.md` — catalog / LanceDB
- `CODEBASE_MAP.md`, `RAG_LOGIC.md` — map cũ (có thể lệch so với code hiện tại; ưu tiên handoff này)

## Reference ngoài (không phải code production)

- `NOT_OUR_RAG_JUST_REF/demo_chunking` — ReAct + Qdrant demo
- Học được: rerank, visual tool; **chưa** port full ReAct
