# Agent handoff index

**Đọc khi bắt đầu session mới về My Agent.**
Cập nhật lần cuối: **2026-07-24** (UTC+7).

| File | Nội dung |
|------|----------|
| [00-STATUS.md](./00-STATUS.md) | Snapshot hiện tại — đọc trước |
| [01-ARCHITECTURE.md](./01-ARCHITECTURE.md) | Stack, luồng agent, file then chốt |
| [02-MEMORY.md](./02-MEMORY.md) | L0/L1/L2/L3 memory, durability, concurrency, recovery |
| [03-RAG-AND-FOCUS.md](./03-RAG-AND-FOCUS.md) | RAG, sticky focus, bug cross-paper |
| [04-DONE.md](./04-DONE.md) | Việc đã làm trong các session gần |
| [05-TODO-AND-ISSUES.md](./05-TODO-AND-ISSUES.md) | Việc sẽ làm + vấn đề tồn đọng |
| [06-OPS.md](./06-OPS.md) | Chạy local, env, ports, lệnh hay dùng |

## Quy tắc cứng (user đã chốt)

1. **Answer LLM + Intent Router = GPT qua 9router**. Approved:
   `cx/gpt-5.6-sol`, `cx/gpt-5.6-terra`, `cx/gpt-5.6-luna`; mặc định
   **Sol**. **Không** đẩy router sang Ollama local hoặc tự đổi model khi lỗi.
2. Embedding production: Ollama `embeddinggemma:300m`, 768d, asymmetric
   query/document prefix. LanceDB và LightRAG VDB đã cùng re-index; không đổi
   model/prefix nếu chưa rebuild cả hai index.
3. Persona: **Aya**. App: Tauri/React + FastAPI, local-first.
4. Không hardcode “ASPIRE → Fig 1” kiểu paper-specific; ưu tiên sticky L1 + ranking theo caption/VLM text.
5. Không bịa số liệu paper khi excerpt thiếu.

## Chat transcript liên quan

Cursor agent transcript (ASPIRE / memory / sticky):
`~/.cursor/projects/Users-vutl2004-Documents-My-Agent/agent-transcripts/a782f2fd-8583-43e7-8481-fbc24061a42c/`
