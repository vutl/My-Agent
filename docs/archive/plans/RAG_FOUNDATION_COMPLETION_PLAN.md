# RAG Foundation Completion Plan — folder corpus first

**Status:** implementation plan, chưa phải báo cáo hoàn thành  
**Ngày chốt:** 2026-08-25  
**Phạm vi release này:** các file đã có hoặc được phát hiện trong folder; chưa làm drag/drop UX  
**Đích:** đóng một bản RAG v1 có vòng đời dữ liệu đúng, retrieval đo được và đủ ổn định để bắt đầu xây tool/sub-agent

## Implementation checkpoint — 2026-08-28

- Packet A/B lifecycle, revision isolation and durable folder processing are
  implemented and covered by the backend suite.
- Gate C r15 on the isolated 21-paper clone is green for the measured A0/A1
  baseline (`0.97` candidate recall, `0.91` evidence recall, `1.00` document
  recall). Dual-query B-full activated `0/60`, so the flag remains off; this is
  not evidence that dual-query improves retrieval.
- Gate D isolated matrix passes native/mixed/table-image PDF, DOCX, Markdown
  and TXT, including rescan, duplicate-byte, changed-content and rename
  invariants. Corrupt input is explicitly `partial`.
- Gate E is intentionally still an isolated decision, not a required visual
  dependency of local RAG. The first resource-controlled run measured about
  8.5 minutes of CPU ColSmol for one 53-page image-only PDF and was stopped
  before aggregation; no visual feature flag is promoted. MMLongBench-Doc and
  SPIQA are public stress/evaluation corpora, not production documents.
- A follow-up document-reuse probe completed seven rows (three questions on
  the same image-only PDF and three on a mixed PDF) before native-PDF
  preparation became too slow. This confirms the evaluator now avoids duplicate
  cold indexing, but still does not provide a visual promotion score.
- Public retrieval/visual runners now preflight local EmbeddingGemma before
  cold ingestion and emit `provider_unavailable` without a score when Ollama
  is offline. Failed checkpoint rows are retried and replaced on resume.
- A five-case isolated SPIQA text/table diagnostic reports phrase recall
  `0.60`, artifact recall `0.70` and modality-aware support recall `0.80`;
  this is explicitly non-promotable and does not close the public arbitrary-
  document gate.

## 0. Quyết định kiến trúc

Không biến thao tác hiện tại thành một chuỗi mù quáng `parse -> LanceDB -> LightRAG` rồi gọi đó là production. Pipeline mới phải tách rõ:

1. **Canonical local path** — parse, provenance, FTS và LanceDB; provider-free, có thể tự chạy nền.
2. **Optional enrichment path** — figure VLM, evidence cards, LightRAG và visual answer; có privacy/cost/model gate riêng.
3. **Evaluation path** — index cô lập, qrels có kiểm duyệt, không được mutate production.

Các quyết định giữ nguyên trong toàn bộ kế hoạch:

- Answer/router/VLM dùng đúng model được duyệt qua 9router; không fallback model.
- Embedding production tiếp tục là `embeddinggemma:300m`, 768d, prefix hiện tại.
- `DUAL_QUERY_RRF_ENABLED=false` cho đến khi vượt retrieval gate và agent E2E.
- LightRAG là lớp navigation/multi-hop tùy chọn, không phải canonical evidence store.
- Docling là parser/layout extractor; OCR chỉ là recovery cho trang không có text dùng được; ColVision là retrieval ảnh trang, không thay parser hay OCR.
- Không re-index production hoặc gửi dữ liệu ra provider trong lúc thử nghiệm nếu chưa có policy/authorization tương ứng.

## 1. Những lỗi gốc đã xác nhận trong repo

### 1.1 Identity và dedup hiện phụ thuộc đường dẫn

- `files.source_path` và `documents.source_path` đang `UNIQUE`.
- Cùng byte content ở hai đường dẫn khác nhau sẽ tạo hai canonical document và hai bộ derived artifacts.
- Rename/move chưa được nhận diện như cùng một source; file biến mất khỏi folder chưa có scan reconciliation/tombstone an toàn.
- `files.content_hash` là SHA-256 của file bytes, nhưng `documents.content_hash` hiện là SHA-256 của **extracted text**. Một PDF chỉ đổi ảnh/bảng nhưng text không đổi có thể bị skip nhầm.

### 1.2 Một document ID chưa đại diện được content revision

- Re-index cùng path giữ stable `document_id`, có lợi cho focus/citation.
- SQLite canonical rows được thay sau khi parse/chunk thành công, nhưng LanceDB và LightRAG có vòng đời riêng.
- LanceDB chưa có shared activation receipt với SQLite.
- LightRAG thấy cùng `document_id` ở trạng thái `processed` sẽ skip; nó chưa kiểm content/parser/chunker revision. Nội dung mới có thể không đi vào graph.

### 1.3 Readiness đã có nhưng orchestration chưa có

- `DocumentReadinessReport` đã phân biệt `ready/partial/failed` và native/OCR/visual-only page.
- Chưa có durable job graph chung cho parse, local vector, visual enrichment, card và LightRAG.
- `paper_evidence_build_jobs` chỉ quản lý riêng evidence card; chưa phải ingestion coordinator.
- Sau crash/restart chưa có một receipt duy nhất để trả lời “revision nào đã sẵn sàng ở SQLite/FTS/Lance/LightRAG?”.

### 1.4 Parsing và visual coverage chưa đồng đều

- PDF: Docling + table structure + page/picture render; OCR fallback chỉ kích hoạt ở document-level rất sparse. Mixed PDF có vài trang scan có thể vẫn còn visual-only.
- DOCX: hiện chỉ lấy paragraph text; bỏ tables và embedded images.
- Markdown/TXT có text path tốt nhưng linked local images và richer block provenance chưa phải contract chung.
- Chưa có ColPali/ColQwen/ColSmol page-image retrieval trong live pipeline.

### 1.5 Retrieval chưa đạt gate

- Dev-v2.1 scope-faithful A0 hiện khoảng Candidate R@30 `0.87`, Evidence R@10 `0.72`, Document R@5 `0.95`.
- Table/figure trong suite hiện mạnh; text và intra-document multi-hop còn yếu.
- Direct `RagService` evaluator không mô phỏng đầy đủ production compare path có per-paper decomposition.
- Public MMLongBench/SPIQA mới là smoke nhỏ; chưa đủ để tuyên bố arbitrary-document quality.

## 2. Target data model: source, blob, document và revision phải tách nhau

Không phá schema live ngay. Thực hiện migration additive, backfill và chạy isolated trước.

### 2.1 Bốn identity khác nhau

| Entity | Vai trò | Identity authority |
|---|---|---|
| `source_instance` | Một path được folder scanner nhìn thấy | normalized absolute path + folder membership |
| `content_blob` | Bytes bất biến của file | raw SHA-256 + size + MIME sniff |
| `logical_document` | Tài liệu ổn định cho chat focus/citation | stable `document_id` |
| `document_revision` | Một lần materialize cụ thể | raw hash + parser/chunker/extraction fingerprints |

Quy tắc:

- Cùng path + cùng raw hash + cùng fingerprints: skip thật, không parse/re-embed.
- Cùng path + raw hash mới: tạo revision mới; revision cũ vẫn active cho tới khi local revision mới sẵn sàng.
- Path khác + raw hash giống: thêm source alias/collection membership; không tạo lại chunks/vectors/LightRAG.
- Rename trong cùng folder scan: nếu raw hash khớp duy nhất, rebind source alias và giữ `document_id`.
- Không dedup “gần giống”; chỉ exact raw hash được tự động dedup.
- Extracted-content hash dùng kiểm integrity/qrel, không dùng thay raw hash để phát hiện file change.

### 2.2 Schema additive dự kiến

- `content_blobs(raw_sha256 PK, size_bytes, mime_type, managed_path NULL, created_at)`.
- `document_sources(id, document_id, folder_id, source_path UNIQUE, raw_sha256, status, first_seen_at, last_seen_at, metadata_json)`.
- `document_revisions(id, document_id, raw_sha256, extracted_hash, parser_fingerprint, chunker_fingerprint, artifact_fingerprint, created_at, status, metadata_json)` với unique materialization fingerprint.
- `document_stage_receipts(document_id, revision_id, stage, fingerprint, status, started_at, finished_at, error, metadata_json, UNIQUE(revision_id, stage, fingerprint))`.
- `documents.active_revision_id`; giữ `documents.id` là stable logical identity.
- `chunks`, tables, figures, cards, FTS rows và Lance metadata đều mang `revision_id`; legacy rows được backfill vào revision đầu tiên.

Không copy file trong folder ở phase này. Source path là authority và `managed_path=NULL`. Thiết kế future upload dùng content-addressed managed storage nhưng chưa triển khai UI.

### 2.3 Atomic activation

```text
discover bytes
  -> stage immutable revision
  -> parse + canonical artifacts
  -> build FTS + Lance projection for revision
  -> validate counts/hashes/readiness
  -> atomically set documents.active_revision_id
  -> asynchronously retire old projections
```

FTS và Lance đều được build dưới revision mới trong khi revision cũ vẫn query được. Query filter theo một `active_revision_id` do SQLite quản lý; vì vậy activation chỉ cần đổi một pointer sau khi cả hai local projections đã có receipt hợp lệ. Nếu parse/vector mới fail, revision cũ tiếp tục phục vụ. Không delete old canonical/vector trước khi replacement đã được kiểm tra.

LightRAG dùng revision-scoped external ID, ví dụ `doc:<document_id>:rev:<revision_id>`. Chỉ graph receipt khớp active revision mới được query. Nếu graph mới chưa ready, Aya dùng local hybrid path và báo graph enrichment pending; không dùng graph cũ cho revision mới.

## 3. Folder reconciliation và durable processing

### 3.1 Scan transaction

Mỗi lần scan folder có `scan_id`:

1. Snapshot toàn bộ supported files và raw hashes.
2. Resolve unchanged/new/changed/renamed/duplicate.
3. Chỉ sau một **full scan thành công** mới tombstone source không còn xuất hiện.
4. Partial scan, permission error hoặc process crash không được xóa source cũ.
5. Một logical document chỉ retired khi không còn source alias/collection reference nào.

Thêm test cho symlink policy, case-sensitive path, permission failure và file thay đổi trong lúc hash/parse.

### 3.2 Durable job coordinator

Tạo `DocumentProcessingService` dùng SQLite durable queue, theo pattern đã có ở conversation memory nhưng scope riêng:

- unique job/receipt theo `(revision_id, stage, fingerprint)`;
- `pending/running/complete/partial/failed/cancelled`;
- lease owner + lease expiry + heartbeat;
- bounded retry theo error class; provider quota/auth fail-closed;
- startup recovery chuyển expired `running` về retryable state;
- idempotent stage handlers và per-stage diagnostic;
- bounded concurrency phù hợp M1 16 GB; foreground chat được ưu tiên tài nguyên.

### 3.3 Stage policy

| Stage | Mặc định | Provider | Có chặn local search không? |
|---|---|---|---|
| discover/hash/MIME | auto | none | có |
| parse/readiness/canonical stage | auto | none | có |
| FTS + LanceDB | auto | local EmbeddingGemma | có |
| activate local revision | auto sau validation | none | có |
| selective OCR | auto theo page-quality policy | local | chỉ chặn page liên quan |
| figure/page VLM summary | policy-gated background | 9router Sol | không |
| paper evidence card | policy-gated background | 9router Sol | không |
| LightRAG ingest | policy-gated background | 9router Sol + local embedding | không |
| visual multi-vector index | feature-gated local background | local model | không |

Trạng thái cuối phải nói rõ capability, không chỉ một chữ `ready`:

- `local_search_ready`;
- `visual_search_ready/skipped/pending`;
- `figure_enrichment_ready/skipped/pending`;
- `graph_ready/skipped/pending`;
- `partial_reasons[]` và `failed_stage`.

Trong phase này cung cấp CLI/API status và worker; chưa làm drag/drop/progress UI.

## 4. Parser contract và format coverage

Tạo một adapter contract chung trả về:

- ordered blocks/sections;
- per-page or logical-location text;
- tables, figures và page images;
- page/block/bbox/caption/hash provenance;
- text origin (`native`, `ocr`, `derived_visual`);
- warnings và readiness diagnostics.

### 4.1 PDF hardening

- Docling giữ vai trò structural parser chính; pypdf/PyMuPDF là fallback/verification.
- Đổi OCR activation từ chỉ document-level sang **page-level quality recovery**:
  - giữ native text tốt;
  - chỉ OCR trang empty/sparse/low-signal;
  - so quality trước/sau và giữ provenance;
  - không OCR toàn paper chỉ vì một trang ảnh.
- Table-image classification phải dựa trên parser item/layout/caption/structure hoặc optional VLM output; không dựa tên benchmark/file.
- Figure grouping, logo/publisher rejection và crop-fragment detection tiếp tục là generic quality policy.
- Nếu table/figure không extract chắc chắn, giữ page evidence thay vì gán artifact type sai.

### 4.2 DOCX support phải đúng với nhãn “supported”

- Giữ document order giữa paragraph, table và image anchors.
- Extract native Word tables thành canonical table markdown/cells.
- Extract embedded images qua relationship ID, SHA-256 và surrounding paragraph/caption.
- Không bịa page number vì DOCX reflow; dùng ordered block/section locator. Optional render-to-PDF là một visual enrichment riêng, không phải canonical parse requirement.

### 4.3 Markdown/TXT

- Giữ heading/code/table provenance.
- Markdown local-image links chỉ được đọc khi nằm trong approved root; hash và tạo figure artifact alias.
- Unsupported/corrupt inputs fail rõ ràng, không tạo document “ready” rỗng.

Release v1 chỉ tuyên bố hỗ trợ `pdf/docx/md/txt`. Format khác bị reject minh bạch cho tới khi có adapter + eval riêng.

## 5. Retrieval baseline trước, Dual-Query sau

Không tune theo một caption/paper cụ thể.

### 5.1 Làm sạch evaluation gate

- Hoàn tất human audit dev-v2.1: câu hỏi thực sự answerable, qrel alternatives, language, conversation state, zero gold leakage.
- Giữ canonical Evidence identity cho text/table/figure/page.
- Thêm production compare runner mô phỏng đúng per-paper decomposition và context reservation.
- Tách metrics first-stage candidate pool khỏi final evidence sau rerank/projection.

### 5.2 Failure-driven baseline fixes

Với từng miss, phân loại trước khi sửa:

- parser/qrel;
- document routing/scope;
- FTS-only miss;
- vector-only miss;
- artifact generation miss;
- candidate cutoff/fusion loss;
- parent/neighbor expansion loss;
- final projection/rerank loss;
- graph provenance/scope leak.

Ưu tiên đạt candidate recall bằng A0 đơn giản:

- exact filename/section/table/figure/numeric anchors qua FTS;
- semantic text/artifact candidates qua EmbeddingGemma;
- canonical parent/neighbor expansion sau child retrieval;
- per-document quota và explicit compare decomposition;
- artifact identity parity giữa SQLite, FTS, Lance và EvidenceBundle.

Chỉ benchmark cross-encoder khi evidence đã nằm trong candidate pool và có latency/RAM budget rõ. Không dùng reranker để che candidate-generation miss.

### 5.3 Dual-query promotion

- Tiếp tục paired A0/A1/B-full/B-anchor trên audited cases.
- Raw branch chỉ chạy khi mang exact signal mà standalone query làm mất/thay đổi.
- Equal RRF trước; chỉ thử weighted RRF khi ranking audit chứng minh fusion loss.
- Nếu candidate union thắng nhưng dual-query không tăng thêm có ý nghĩa, chỉ promote union.
- Không bật dual-query mặc định trước shadow + E2E.

## 6. Visual page retrieval là một isolated experiment

ColVision không được nhập thẳng vào production vì “SOTA”. Tạo interface `VisualRetrievalProvider` và index cô lập theo revision/page.

### 6.1 Arms

- `T`: current text/table/figure hybrid.
- `V`: page-image multi-vector retrieval.
- `T+V`: document shortlist + page-level late fusion.

Shortlist local-first:

- thử model khoảng 500M trước (ColSmol class);
- chỉ thử model khoảng 2B nếu 500M không đạt quality gate và M1 16 GB còn đủ RAM/latency;
- không tải/promote model 4B chỉ vì leaderboard.

Model shortlist phải khóa model card/license/revision SHA. Storage/index của multi-vector branch tách khỏi Lance single-vector production cho tới khi prototype chứng minh schema phù hợp.

### 6.2 Routing

- Focused document: visual page search có thể chạy song song trong bounded page set.
- Corpus-wide: document shortlist bằng cards/FTS/text vector trước, rồi visual search trong candidate documents.
- Sparse/scan/visual-heavy readiness tăng ưu tiên visual branch.
- Không chỉ dựa vào từ `hình`, `chart` vì nhiều visual question không nói rõ modality.

### 6.3 Promotion metrics

- page/evidence recall theo text/table/chart/diagram/scan;
- answer quality với ảnh thật được đưa cho VLM;
- query P50/P95;
- cold index time/document;
- bytes/page và toàn corpus;
- peak RSS/thermal trên M1 16 GB;
- incremental update/delete correctness.

Promote `T+V` chỉ khi tăng recall/answer quality trên visual subsets, không làm rơi text/table exact cases và nằm trong resource gate. Nếu không thắng, giữ feature flag off; đây là kết quả hợp lệ, không phải lý do trì hoãn vô hạn.

## 7. Evaluation programme để tránh “con hát mẹ khen hay”

### 7.1 Lifecycle suite — bắt buộc trước quality benchmark

Test ít nhất các tình huống:

- scan lại không đổi;
- duplicate bytes ở path khác;
- rename/move;
- content text đổi;
- chỉ image/table bytes đổi;
- parser/chunker fingerprint đổi;
- file bị xóa;
- scan lỗi giữa chừng;
- crash giữa parse/vector/activation;
- Ollama embedding offline;
- 9router quota/auth/timeout;
- restart và resume;
- collection alias/delete reference;
- rollback revision cũ.

Acceptance:

- zero duplicate canonical chunks/vectors cho exact duplicate;
- zero stale active revision;
- failed replacement không làm mất revision đang phục vụ;
- mọi job về terminal/retryable state có giải thích;
- no silent fallback/model switch.

### 7.2 Public document-quality dev set

Ingest qua **đúng production adapters và coordinator**, nhưng ở runtime root cô lập:

- native-text scientific PDF;
- mixed native + scanned PDF;
- image-only scan/form;
- report có table-image;
- charts/diagrams/slides;
- multi-page/multi-hop documents;
- DOCX có paragraph/table/image;
- Markdown/TXT;
- Việt/Anh và negative/unanswerable.

Nguồn có thể lấy stratified từ MMLongBench-Doc, SPIQA, DocVQA và fixture tự tạo chỉ cho lifecycle invariants. Official question/qrel/scorer được ưu tiên; fixture tự tạo không dùng để tuyên bố external quality.

Chạy ba mode khi hợp lệ:

- gold evidence — generation ceiling;
- full extracted/visual document với coverage được báo cáo;
- Aya production pipeline.

### 7.3 Metrics/gates

Retrieval-only:

- Candidate Recall@30 >= 0.95;
- Document Recall@5 >= 0.90;
- Evidence Recall@10 >= 0.85;
- từng supported modality >= 0.80;
- multi-paper coverage >= 0.95;
- warm retrieval P50 <= 500 ms, P95 <= 1.5 s;
- zero unresolved qrel, zero forbidden scope leak, zero model call.

Ingestion:

- 100% source/revision receipts nhất quán;
- zero document được ghi `local_search_ready` nhưng không query được;
- page/table/figure provenance resolve sau isolated re-index với UUID mới;
- dedup/move/delete/crash suite pass 100%.

Agent E2E:

- dev-20 + regression-v1 giữ pass rate hiện tại qua HTTP/SSE với Sol;
- zero transport/tool/model fallback;
- zero unsupported numeric claim;
- negative cases abstain đúng;
- báo cold/warm latency, token/visual coverage và provider cost riêng.

Không lấy vài smoke case hoặc unit-test pass làm quality claim.

## 8. Thứ tự implementation cho Luna/Terra

### Packet A — lifecycle correctness

1. Additive schema + migration/backfill cho source/blob/revision/receipt.
2. Raw-byte change detection và exact-content dedup.
3. Folder scan reconciliation/rename/delete semantics.
4. Revision-scoped Lance/LightRAG IDs + active-revision filtering.
5. Lifecycle/crash tests.

**Gate A:** lifecycle suite pass; production behavior cũ vẫn đọc được qua compatibility layer.

### Packet B — durable local pipeline

1. SQLite coordinator, lease/recovery/idempotent handlers.
2. Auto local stages: parse/readiness/FTS/Lance/activate.
3. Policy-gated optional stage receipts.
4. CLI/API status; chưa sửa Files UI.

**Gate B:** folder rescan tự hội tụ đúng sau crash/offline; không cần bấm ba bước để đạt local-search-ready.

### Packet C — đóng retrieval trên corpus hiện tại

1. Audit dev-v2.1 and production-compare runner.
2. Chạy taxonomy trên A0 sau migration để bắt schema/projection regression.
3. Sửa baseline miss tổng quát cho text/multi-hop.
4. Chạy dual-query/candidate-union ablations.
5. Optional bounded cross-encoder experiment only if justified.

**Gate C:** current 21-paper retrieval metrics pass và agent E2E không regression. Đây là mốc sớm nhất chứng minh nền paper-RAG hiện tại đã đóng.

### Packet D — parser coverage và arbitrary-format gate

1. Per-page selective OCR for mixed PDFs.
2. Generic table-image/page fallback.
3. DOCX ordered block/table/image extraction.
4. MD linked-image security/provenance.
5. Ingest public format-stratified dev set qua chính coordinator.
6. Rerun retrieval metrics theo từng format/modality.

**Gate D:** supported-format readiness/provenance và retrieval gates cùng pass; format chưa đạt bị ghi unsupported/partial chứ không được quảng bá là ready.

### Packet E — isolated visual retrieval

1. Provider interface + revision/page index.
2. Text vs visual vs fusion A/B on public visual subset.
3. Measure RAM/index/latency on actual M1 16 GB.
4. Promote or explicitly reject the branch based on gates.

**Gate E:** visual subset passes without text/exact regression, or visual feature remains honestly disabled.

### Packet F — Release Candidate

1. Freeze Git SHA and source/index/parser/chunker/embedding/visual/rewrite fingerprints.
2. Run lifecycle + public dev + agent E2E.
3. Build unseen shadow set by document family and run once.
4. Stage production migration with backup/health/rollback.

## 9. Future drop-in storage contract — thiết kế bây giờ, code UI sau

Khi có drag/drop:

- Copy upload vào content-addressed managed store, ví dụ `data/library/blobs/<sha-prefix>/<sha>.<ext>`.
- Original filename/path chỉ là source alias; một blob có nhiều aliases/collections.
- Same SHA không lưu/index lại; chỉ tăng reference/membership.
- Delete một alias không xóa blob/canonical document nếu còn reference khác.
- MIME sniff, filename sanitization, size/page limits, quarantine unsupported/corrupt files.
- Conversation citation giữ logical document + revision + source snapshot, không phụ thuộc path có còn tồn tại.
- Managed store, DB, artifacts và visual indexes tiếp tục nằm ngoài Git.

Thiết kế này tương thích folder mode hiện tại vì folder source dùng cùng source/blob/document/revision registry nhưng `managed_path=NULL`.

## 10. Definition of Done để chuyển sang agent system

RAG v1 được coi là **đủ đóng** khi:

1. Gate A/B/C/D pass trên isolated production path.
2. Visual Gate E được quyết định bằng số đo: promote nếu thắng, hoặc giữ off với giới hạn sản phẩm được ghi rõ.
3. Shadow-by-document/family và Sol E2E pass trên frozen RC.
4. `EvidenceBundle`, `DocumentReadinessReport`, source/revision receipts và retrieval API được version hóa thành stable tool contract.
5. Rollback production được thử thật; handoff/status/Graphify đồng bộ.

Sau điểm này, không tiếp tục “đánh bóng RAG vô hạn”. Bug mới đi vào regression suite; roadmap chính chuyển sang authorization, web/file/code tools, ReAct loop và sub-agent orchestration trên cùng `EvidenceBundle` contract.

## 11. Những việc cố ý không làm trong plan này

- Không làm drag/drop/progress UI ngay.
- Không đổi embedding production.
- Không coi LightRAG/VLM là stage bắt buộc để local search ready.
- Không nhập ColVision vào production trước A/B.
- Không re-index production trước isolated migration/rollback smoke.
- Không hardcode paper, caption, figure/table number hoặc benchmark filename.
- Không mở rộng thêm agent tools trước khi Gate D và RC pass.

## Tài liệu kỹ thuật tham chiếu

- [Docling pipeline options](https://docling-project.github.io/docling/reference/pipeline_options/)
- [ColPali/ColVision engine and supported model families](https://github.com/illuin-tech/colpali)
- [ColPali paper](https://arxiv.org/abs/2407.01449)
- [MMLongBench-Doc](https://papers.nips.cc/paper_files/paper/2024/file/ae0e43289bffea0c1fa34633fc608e92-Paper-Datasets_and_Benchmarks_Track.pdf)
- [SPIQA](https://github.com/google/spiqa)
- [DocVQA challenges](https://www.docvqa.org/challenges/2026)
