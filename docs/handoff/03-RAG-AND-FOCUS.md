# 03 — RAG, sticky focus, figures/tables

## Retrieval stack

- LightRAG và legacy/Lance retrieval được merge; LightRAG scoped rỗng có legacy fallback an toàn.
- LanceDB giữ text/table/figure chunks; collection scope áp dụng cho mọi engine.
- Embedding query/index hiện dùng Ollama `embeddinggemma:300m` (768d), với
  query prefix `task: search result | query: ` và document prefix
  `title: none | text: `. LanceDB và LightRAG entity/relation/chunk VDB cùng
  nằm trong vector space này.
- Canonical text chunking v3 dùng structural parent–child: child tối đa 384
  lexical tokens, overlap 48; parent tối đa 1.536 tokens. Child được retrieve
  trước rồi mở rộng lên parent để answer có đủ ngữ cảnh mà không làm vector quá
  lớn. Tokenizer là deterministic Unicode lexical tokenizer chạy offline, không
  có first-run download.
- Embedding text được prepend context từ filename/title/page/section và summary
  ngắn của DocumentCard; raw content trong SQLite vẫn sạch để citation/render.
- Hybrid RRF là fast path mặc định. Embedding-cosine reranker cũ tắt mặc định vì
  benchmark v3 cho thấy vừa chậm vừa giảm MRR/nDCG. Local cross-encoder là chế độ
  opt-in, yêu cầu explicit model directory và fail-closed nếu asset/dependency
  thiếu; runtime không tự download model.
- Figure VLM lúc index/enrich mặc định dùng **9router `cx/gpt-5.6-sol`**, không chạy VLM mỗi query.
- Answer + intent router cũng mặc định dùng **9router `cx/gpt-5.6-sol`**; UI
  cho chọn thêm Terra/Luna nhưng không có automatic fallback.
- Result/table intent không bị pha architecture terms; focused table ask ưu tiên legacy/Lance, truyền `min_tables=1` và context builder reserve ít nhất một table.
- Count/list table intent (`mấy bảng`, `bao nhiêu bảng`, `how many tables`, …)
  lấy toàn bộ canonical `document_tables` trong paper đang focus rồi mới compose
  context; không suy tổng số bảng từ một top-1 similarity hit.
- Retrieval cache version `v14`; table query từ chối cache entry không còn table
  source. Bump `v14` loại cache comparison cũ tạo bởi LightRAG sau khi scoped
  comparison chuyển sang local hybrid.

## Staged retrieval / broad comparison

- Specific QA vẫn đi fast hybrid path.
- Compare intent có ít nhất hai focused canonical documents được decomposition
  thành một branch/document, chạy đồng thời và merge evidence có provenance.
- Mỗi decomposed branch có đúng một canonical paper dùng local hybrid, tránh hai
  LightRAG provider calls cho một comparison đã biết rõ paper. Graph vẫn giữ cho
  discovery/unscoped/cross-document reasoning khi chưa có canonical focus.
- Branch luôn giữ document scope, không recurse decomposition và không mở rộng
  sang paper ngoài L1/recent thread.
- Final prompt round-robin excerpt theo document và reserve first-source budget
  cho từng paper. Trusted coverage metadata cấm model báo “paper/source vắng mặt”
  nếu canonical evidence đã retrieve; thiếu facet nào thì chỉ được nói facet đó.
- Đây là controlled multi-hop cho paper comparison; chưa phải một ReAct loop
  tự do trên toàn corpus.

## Latency + streaming policy (2026-08-02)

- Planner cho mọi grounded local route là deterministic; retrieval/tool routing đã
  hoàn tất trước LangGraph nên không gọi Sol chỉ để viết lại kế hoạch 2–4 bước.
- Yêu cầu hiển thị chính xác một canonical table (`table_id` + `document_id` +
  Markdown/provenance rõ) render trực tiếp, validate bằng cùng claim guard rồi
  stream; không gọi planner/answer LLM. Inventory, analysis/evaluation hoặc nhiều
  candidate không đi fast path.
- Câu RAG định tính stream theo Markdown paragraph: block hoàn tất được validate
  trước khi hiện; dòng số unsupported được sanitize/suppress. Yêu cầu bảng/số vẫn
  buffer toàn answer để không rò row/metric chưa kiểm.
- Quantitative draft sai không regenerate ngay: loại dòng unsupported, validate
  lại và chỉ retry Sol nếu requested metric không còn đủ. Retry vẫn tối đa một lần,
  sau đó fallback không số.
- Trace tách `router`, `graph`, `first_validated_token` và `generation_total`;
  desktop hiện đúng nhãn **First Validated Token**, không gọi nhầm raw first token.

## Sticky focus và hội thoại tự nhiên

Luồng hiện tại:

1. Rewrite/router nhận L1 active document, recent document thread, L2 summary và recent beats.
2. Nếu user không topic-switch và không nêu paper mới, paper QA/deepen/resume giữ `L1.active_document_ids`.
3. Casual detour không xóa paper focus; câu “quay lại paper trước” có thể lấy từ document thread.
4. `_scope_documents_to_focus` lọc evidence theo focus; không fallback kiểu `scoped or documents` gây leak toàn corpus.
5. L1 chỉ commit sau grounded answer hoàn tất; run fail/invalid không ghi đè state.

Conversation state tự repair stale UUID sau re-index khi filename hoặc topic match duy nhất. L2 dùng compare-and-swap revision để fold async cũ không ghi đè context mới hơn.

## Figures/tables khi index

Pipeline extraction V9:

1. Docling raw assets → geometry grouping.
2. Ghép multi-panel/composite khi các crop thuộc cùng figure.
3. Gắn caption/ref sentences, page context, section và paper summary.
4. Quality gate phân loại `accepted`, `needs_review`, `rejected`; loại logo/branding/page fallback không phải nội dung.
5. Gửi crop (high detail) + full page (low detail) và context sang model GPT-5.6
   được chọn qua 9router (mặc định Sol).
6. Lưu stable document/figure/table IDs và provenance VLM; vector hóa accepted visual chunks.

Không dùng `index + 1` để suy Figure N, không hardcode paper-specific mapping. Ranking figure dùng caption/VLM/figure type và query intent:

- Query architecture/pipeline/framework boost `architecture`/`diagram`, demote plot/photo.
- Query quantitative boost chart/plot/table.
- Query xin một/best/most-relevant figure chỉ trả top 1 sau scope, rank và quality curation; không hardcode paper → figure number.
- Từ “mô hình” trong câu hỏi nội dung thông thường không tự kích hoạt visual intent.
- Desktop chỉ render curated order/backend metadata, không tự đoán Figure N từ câu hỏi.

Full-corpus figure/page enrichment bằng GPT-5.6 Sol đã hoàn tất sau khi user
approve privacy: 60 assets enriched, 76 quality-skipped, 0 failed. Live quality
là 59 accepted / 22 needs-review / 55 rejected; crop thiếu context vẫn được hạ
needs-review dù VLM đã mô tả, không promote chỉ để tăng recall.

Vision client coi completion rỗng từ 9router là lỗi; không lưu provenance model
giả cho asset chưa thực sự được mô tả.

## Evidence guard

- Validator loại evidence khác active focus và mixed-document evidence không hợp lệ.
- RAG answer được buffer để kiểm quantitative claims trước khi stream ra UI.
- Metric và parameter count được đối chiếu evidence, gồm magnitude `k/m/b` và header `P (M)`/`Params(M)`.
- Parser hỗ trợ row-oriented/repeated-header Markdown tables, model-vs-dataset ownership, comparison owners, spaced decimals, prose trong table cell và Docling token dính như `accuracy of75.86%`.
- Fail-closed signal bắt metric/value layout chưa parse; bounded attempt diagnostics chỉ lưu nội bộ, không copy unsupported draft numbers ra SSE.
- Quantitative prompt yêu cầu bind model/dataset/metric rõ ràng; retry dùng layout đơn giản và vẫn fallback không số nếu evidence thực sự thiếu.
- Nếu claim không được support: retry một lần với constraint; vẫn thiếu evidence thì trả lời không số thay vì bịa.

## LightRAG provenance/migration

- Chunk có basename trùng hoặc separator provenance mơ hồ không được relabel thành focused document.
- Bridge path mapping có thể nối processed graph IDs cũ với canonical document IDs hiện tại.
- User đã approve và canonical LightRAG re-ingest đã hoàn tất ngày 2026-07-29:
  **21/21 processed**, 0 failed/unready; graph **8.559 nodes / 14.173 edges**.
  Lượt cuối prune 7 duplicate tombstones.
- Stale prune chỉ được chạy khi canonical index không rỗng và tất cả canonical LightRAG docs đã `processed`; nếu không sẽ trả lý do blocked và xóa 0.
- Full-sync preflight đúng gateway/model, cô lập từng canonical document khỏi
  upstream failed backlog, kiểm `doc_status=processed` sau `ainsert` và abort
  ngay lỗi đầu tiên. Vì vậy provider outage không còn tạo false success hoặc
  kéo cả corpus chạy song song.
- Adapter yêu cầu native streaming từ 9router rồi collect thành text, tránh lỗi
  non-stream conversion 502. OpenAI/httpx timeout/connection được retry một lần
  với chính model Sol; quota/status/model mismatch fail ngay. Extract role có
  timeout 660s để LightRAG watchdog chứa trọn provider timeout 300s + retry;
  query role vẫn 240s. Chunking canonical dùng 600 token, overlap 80 và
  extraction concurrency 1.

## Các bug đã đóng

- “benchmark / Acc / F1 / CCC” không còn bị coi là paper entity mới làm rơi sticky focus.
- Deepen/visual follow-up không còn bám history paper cũ thay vì L1 active.
- LightRAG chunk thiếu/ambiguous provenance không lọt vào focused answer.
- ASPIRE architecture query không còn ưu tiên plot chỉ vì lexical overlap; ranking có figure-type intent.
- Last-source cache chỉ tái dùng khi collection, mode, index fingerprint và required table shape còn hợp lệ.
- Markdown/model/dataset subject parsing không còn làm benchmark đúng bị false reject; baseline đứng sau không chiếm nhầm số model chính.
- Model dạng sentence-cased hyphen như `Pitch-fusion` được giữ làm owner của
  các ô metric; không còn bị thay bằng dataset/filename rồi false-reject.
- “thế/vậy” không tạo topic switch nếu model/entity đã xuất hiện trong recent
  paper context; single-paper model comparison vẫn giữ sticky L1.
- Parent expansion lấy passage qua `parent_chunk_id`, không còn chỉ ghép neighbor
  mù; stable child/parent IDs giữ nguyên khi rebuild cùng source.
