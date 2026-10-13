# RAG Foundation Completion — Correctness Pass

## Tóm tắt

Giữ kiến trúc local-first nhưng sửa các điểm chưa hoàn chỉnh trước khi đánh giá retrieval hay mở rộng agent. FTS5 và LanceDB tạo canonical local-search readiness; LightRAG, evidence cards và VLM tự động chạy nền cho folder được phê duyệt nhưng không chặn tài liệu sử dụng. Mọi projection phải gắn với đúng revision và chỉ active sau validation nguyên tử.

## 1. Hoàn thiện revision lifecycle và durable processing

- Chuyển mọi ingestion path sang cùng lifecycle: `discover → staged revision → canonical/FTS/Lance validation → atomic activation`.
- Giữ endpoint thủ công để tương thích UI, nhưng Parse chỉ tạo staged revision; Lance hoàn tất mới được activate. Không còn đường activate ngay sau parse.
- Job có idempotency key theo folder/options/snapshot fingerprint; rescan trùng không tạo job hoặc revision thừa.
- Bổ sung heartbeat, lease renewal, retry có exponential backoff, error classification và startup recovery.
- Stage receipt dùng fingerprint thực từ source/parser/chunker/embedding/config, không dùng chuỗi `stage:v1`.
- Chia trạng thái rõ: `staged`, `local_ready`, `enriching`, `ready`, `partial`, `failed`; Ollama offline giữ revision cũ active và revision mới retryable.
- Thêm validation trước activation: expected/actual counts và content identity của canonical artifacts, FTS và Lance phải khớp.

## 2. Làm revision isolation thực sự

- Chuyển FTS reader sang `rag_chunks_fts_v2`, query bắt buộc khớp `documents.active_revision_id`; legacy FTS chỉ tồn tại trong migration window.
- Thêm `revision_id` thành cột vật lý trong mọi Lance table; lọc active revision ngay trong Lance query trước top-k, không lọc sau retrieval.
- Migration Lance chạy trong isolated index, kiểm count/recall rồi atomic path switch; cleanup theo revision thay vì `metadata_json LIKE`.
- Version hóa document cards và paper evidence cards theo `(document_id, revision_id)`; staged card không được ghi đè card của active revision.
- Mọi retrieval path—FTS, vector, table, figure, parent expansion và card routing—dùng chung active-revision contract.
- Giữ logical `document_id` ổn định cho chat/citation; runtime artifact ID được version theo revision.

## 3. LightRAG và enrichment tự động nhưng không canonical

- LightRAG external ID dùng `document_id + revision_id`; ingest chỉ đọc chunks của đúng revision.
- Provenance graph lưu revision, parent identity và stage fingerprint; resolver từ chối graph cũ hoặc không khớp active revision.
- Sau local activation, tự enqueue VLM/figure enrichment, evidence cards và LightRAG cho folder đã được phê duyệt gửi qua 9router.
- Enrichment lỗi/timeout được retry độc lập và không rollback local-ready revision.
- Khi graph chưa sẵn sàng, Aya dùng text/table/figure hybrid; khi graph sẵn sàng, graph chỉ hỗ trợ navigation/multi-hop và mọi claim vẫn phải resolve về canonical evidence.
- Không fallback sang model khác khi Ollama hoặc 9router lỗi.

## 4. Parser, readiness và evaluation gates

- Hoàn thiện DOCX ordered paragraph/table/image, Markdown heading/table/code và local-image security dưới approved root.
- PDF giữ Docling structural parse; selective OCR chỉ chạy trên page thiếu text/readability, lưu provenance `native/ocr/visual`.
- Không ép table-image thành canonical table khi confidence thấp; giữ page/image evidence và đánh trạng thái `partial` hoặc `visual_only`.
- Sửa evaluator để mô phỏng đúng production decomposition và active-revision filtering.
- Chạy lại audited `dev-v2.1`; sửa A0 theo failure taxonomy trước khi đánh giá A1/B. Dual-query tiếp tục tắt.
- Sau Gate C mới xây visual retrieval A/B và public stratified corpus; không promote ColVision nếu không tăng visual recall khách quan.
- Tạo `shadow-v3` từ unseen documents sau khi freeze Release Candidate; không dùng lại shadow-v2.

## Interfaces và trạng thái

- `POST /files/process-folder` trả về hoặc tái sử dụng durable job theo idempotency key.
- Job/status API bổ sung revision, stage fingerprint, attempts, next retry, provider/model, progress counts và failure reason.
- Document status trả riêng `local_search_state`, `graph_state`, `card_state`, `figure_state`, `visual_state`; `ready` không được suy ra từ một stage bị `skipped`.
- Endpoint cũ giữ response tương thích nhưng thực thi qua lifecycle an toàn; đánh dấu deprecated trong API metadata, chưa thay Files UI ở pha này.

## Kiểm thử và nghiệm thu

- Rescan, duplicate bytes và concurrent enqueue không tạo duplicate revision/vector.
- Rename/move giữ logical document; tombstone chỉ sau full scan thành công.
- Crash tại parse/FTS/Lance/activation không làm mất active revision.
- Ollama offline, 9router timeout và process restart đều retry đúng, không fallback.
- FTS/Lance/cards/LightRAG không trả evidence từ staged hoặc retired revision.
- Lance top-k không bị stale vectors chiếm chỗ.
- DOCX image/table, Markdown local image và mixed-scan PDF giữ đúng provenance.
- Backend suite, lifecycle crash tests, isolated real-Lance migration, desktop build và HTTP/SSE regression đều pass.
- Gate C: Candidate R@30 ≥ 0,95; Document R@5 ≥ 0,90; Evidence R@10 ≥ 0,85; từng modality ≥ 0,80; multi-paper ≥ 0,95; P50 ≤ 500 ms; P95 ≤ 1,5 s; zero scope leak/model call/fallback.
- Chỉ sau isolated RC và shadow-v3 pass mới migration production và cleanup projection cũ.

## Giả định đã khóa

- Embedding tiếp tục là local Ollama `embeddinggemma:300m`.
- Router/answer/VLM dùng các model 9router được duyệt, mặc định `cx/gpt-5.6-sol`.
- LightRAG được tự động sử dụng cho approved folder, nhưng là derived navigation layer và không chặn local readiness.
- Không sửa Files UI, không bật dual-query và không mutate production trong đợt triển khai này.
- Worktree hiện đang có toàn bộ thay đổi foundation chưa commit; implementation phải bảo tồn chúng và chia commit theo migration, lifecycle, revision isolation và validation.
