# Agent handoff index

**Đọc khi bắt đầu session mới về My Agent.**
Cập nhật lần cuối: **2026-10-25** (UTC+7).

| File | Nội dung |
|------|----------|
| [00-STATUS.md](./00-STATUS.md) | Snapshot hiện tại — đọc trước |
| [01-ARCHITECTURE.md](./01-ARCHITECTURE.md) | Stack, luồng agent, file then chốt |
| [02-MEMORY.md](./02-MEMORY.md) | L0/L1/L2/L3 memory, durability, concurrency, recovery |
| [03-RAG-AND-FOCUS.md](./03-RAG-AND-FOCUS.md) | RAG, sticky focus, bug cross-paper |
| [04-DONE.md](./04-DONE.md) | Việc đã làm trong các session gần |
| [05-TODO-AND-ISSUES.md](./05-TODO-AND-ISSUES.md) | Việc sẽ làm + vấn đề tồn đọng |
| [08-AYA-FINISH-PLAN.md](./08-AYA-FINISH-PLAN.md) | **Kế hoạch chốt để hoàn tất Aya — đọc và làm theo** |
| [06-OPS.md](./06-OPS.md) | Chạy local, env, ports, lệnh hay dùng |
| [07-NUMERIC-GROUNDING-REVIEW.md](./07-NUMERIC-GROUNDING-REVIEW.md) | Numeric correctness, facts jobs, giới hạn và report 2026-10-09 |
| [09-SEALED-EVAL-PLAN.md](./09-SEALED-EVAL-PLAN.md) | Sealed evaluation method |
| [10-AGENT-HARNESS.md](./10-AGENT-HARNESS.md) | Agent harness: roles, accounts, pipeline |
| [11-FAILURE-LEDGER.md](./11-FAILURE-LEDGER.md) | Failure mechanisms of RAG and their generalisation status |
| [12-RELEASE-RAG-V1.md](./12-RELEASE-RAG-V1.md) | RAG v1 release record: evidence, rollback drill, rollout |
| [13-ASSISTANT-TOOLS-PLAN.md](./13-ASSISTANT-TOOLS-PLAN.md) | Assistant phase: tools, sandbox, skills, Live2D Aya — plan and progress |
| [../plans/agent/README.md](../plans/agent/README.md) | **Kế hoạch "Aya làm được mọi thứ"**: roadmap P1–P5, agent core, tools, skills/workflows, safety, presence, eval, work packages |

## Quy tắc cứng (user đã chốt)

1. **Answer LLM + Intent Router = GPT qua 9router** (AGENTS.md, 2026-10-18):
   answers `cx/gpt-6.1-sol` (mặc định), router `cx/gpt-6-luna`; `cx/gpt-5.6-*`
   chỉ để tái lập eval cũ; `VISION_MODEL` giữ `cx/gpt-5.6-sol`. **Không** đẩy
   router sang Ollama local hoặc tự đổi model khi lỗi.
2. Embedding production: Ollama `embeddinggemma:300m`, 768d, asymmetric
   query/document prefix. LanceDB và LightRAG VDB đã cùng re-index; không đổi
   model/prefix nếu chưa rebuild cả hai index.
3. Persona: **Aya**. App: Tauri/React + FastAPI, local-first.
4. Không hardcode “ASPIRE → Fig 1” kiểu paper-specific; ưu tiên sticky L1 + ranking theo caption/VLM text.
5. Không bịa số liệu paper khi excerpt thiếu.

## Chat transcript liên quan

Cursor agent transcript (ASPIRE / memory / sticky):
`~/.cursor/projects/Users-vutl2004-Documents-My-Agent/agent-transcripts/a782f2fd-8583-43e7-8481-fbc24061a42c/`

Older history: `docs/archive/handoff-history/`.
