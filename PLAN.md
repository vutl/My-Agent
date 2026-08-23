# PLAN.md v2 — Document Intelligence Gate trước khi mở rộng Aya Agent

## Kết luận sau phản biện

Giữ bốn trụ cột của kế hoạch cũ, nhưng chỉnh năm điểm để Luna/Terra không triển khai lệch:

- `EvidenceBundle` phải được định nghĩa **trước** dual-query để tránh migrate interface hai lần.
- OCR không dùng riêng ngưỡng “<100 ký tự/trang”; Aya hiện đã có sparse-page OCR nhưng cần readiness đa tín hiệu.
- Full-document baseline không được gọi là “full” nếu âm thầm head+tail truncate.
- Original+rewrite RRF là thử nghiệm có feature flag, không mặc định luôn tốt hơn. Nghiên cứu production năm 2026 cho thấy fusion có thể tăng raw recall nhưng mất lợi ích sau rerank/truncation. [RAG Fusion deployment study](https://arxiv.org/abs/2603.02153)
- `EvidenceBundle` dùng tagged union theo modality, không ép text, table, figure và web result thành một object phẳng.

## Pha 1 — Contract và evaluation foundation

### Typed contracts

Định nghĩa trước khi sửa retrieval:

```text
EvidenceCandidate
├── TextEvidence
├── TableEvidence
├── FigureEvidence
└── PageEvidence

EvidenceBundle
├── request/query origins
├── candidate documents
├── selected evidence
├── facet/modality coverage
├── document readiness
└── retrieval/ambiguity diagnostics
```

Mọi candidate chứa document/page/section ID, parent-child link, provenance, extraction confidence, lexical/vector/RRF/rerank scores và query origin.

Các tool tương lai dùng chung envelope `EvidenceItem`, nhưng payload vẫn là tagged union theo source/modality.

### Evaluation v2

- Tạo `dev-v2`: 60 câu nhìn thấy, cân bằng 6 loại câu × Việt/Anh.
- Tạo `shadow-v2`: 24 câu sealed, document/family không trùng dev.
- Retrieval-only chạy toàn bộ không cần Sol; Sol E2E chạy dev khi có thay đổi lớn và shadow một lần mỗi release candidate.
- Mỗi câu có qrels theo document, page, table/figure và canonical evidence span.
- Public gate bổ sung MMLongBench-Doc, SPIQA và ViDoRe V2 cho visual/page retrieval. [ViDoRe V2](https://arxiv.org/abs/2505.17166)

### Sửa phép đo ba chế độ

Runner hiện giới hạn `90.000` ký tự và head+tail truncate tại [evaluate_public_multimodal_three_mode.py](</Users/vutl2004/Documents/My Agent/scripts/evaluate_public_multimodal_three_mode.py:494>).

Thay bằng:

- `gold_evidence`.
- `full_document`: chỉ dùng nhãn này khi toàn bộ extracted text vừa budget.
- `windowed_long_context`: dùng khi vượt budget, bắt buộc ghi overflow/truncation.
- `aya_pipeline`.

Dùng tokenizer, mặc định tối đa 96.000 input tokens và dành ít nhất 16.000 tokens cho system/output. Không chọn page theo câu hỏi trong full-document mode; page images lấy uniform, tối đa 8 và báo visual coverage.

## Pha 2 — DocumentReadiness v2 và OCR

### Page-level readiness

Mỗi trang được phân loại:

- `native_text_ready`
- `ocr_text_ready`
- `visual_only`
- `unresolved`

Document:

- `ready`: có retrievable units và không còn trang unresolved.
- `partial`: có dữ liệu dùng được nhưng còn trang/modality unresolved.
- `failed`: parser lỗi hoặc không tạo được text/table/figure/page evidence nào.

Readiness report chứa page count, native/OCR text coverage, sparse/unresolved pages, table count, figure quality counts, parser fingerprint, warnings và failure reasons.

### OCR policy

Aya hiện đã OCR sparse PDF theo document ratio và page threshold tại [parsers.py](</Users/vutl2004/Documents/My Agent/backend/app/rag/parsers.py:301>). Không thay bằng một ngưỡng 100 ký tự đơn giản.

Policy mới:

1. Giữ native text nếu đủ ký tự và tỷ lệ alphanumeric hợp lệ.
2. OCR riêng trang sparse, corrupt hoặc có raster area lớn nhưng thiếu text.
3. Giữ RapidOCR hiện tại làm baseline.
4. A/B Docling `PDF_AWARE_LAYOUT_REGIONS`; chỉ dùng full-page OCR khi regional OCR vẫn unresolved. Docling hỗ trợ các OCR mode và full-page option chính thức. [Docling OCR options](https://docling-project.github.io/docling/reference/pipeline_options/)
5. OCR không tạo được text thì giữ page image dưới trạng thái `visual_only`; không giả thành parser success.
6. Re-index atomic, stable IDs/fingerprints và không xóa index cũ trước readiness gate.

## Pha 3 — Candidate generation và dual-query RRF

### Query policy

Chạy dual query chỉ khi standalone rewrite khác query gốc và vẫn giữ đủ explicit entities, metric/model tokens và document references:

- `q0`: original query.
- `q1`: standalone Aya rewrite.

Không sinh 3–5 query mới bằng Sol. RAG-Fusion có thể tăng coverage nhưng cũng có nguy cơ query drift. [RAG-Fusion paper](https://arxiv.org/abs/2402.03367)

### Two-stage retrieval hoàn chỉnh

Current hybrid path đã dùng document cards và global FTS tại [rag_service.py](</Users/vutl2004/Documents/My Agent/backend/app/services/rag_service.py:228>), nhưng dense chunk search có thể bị khóa vào card-selected documents.

Thuật toán mới:

1. Với mỗi query, chạy song song document-card top-12 và global FTS top-20.
2. Candidate-document union:
   - Explicit focus: chỉ dùng scope đã resolve.
   - Unscoped: tối đa 8 card documents + 4 document mới từ global FTS.
   - Nếu union rỗng: global dense recovery trên toàn corpus.
3. Dense search trong document union:
   - Text top-24.
   - Table top-12.
   - Figure/page top-12.
4. RRF theo query và channel, mặc định `k=60`; giữ toàn bộ component scores.
5. Candidate pool tối đa 30 trước rerank.
6. Compare query reserve tối thiểu 6 candidate/paper; final evidence reserve tối thiểu 3/paper.
7. Exact Table/Figure N, evidence card và canonical artifact path không chạy hai lần.
8. Nếu facet/document coverage còn thiếu, dùng second-hop deterministic hiện có; không thêm foreground Sol call.

Feature flag: `DUAL_QUERY_RRF_ENABLED=false` mặc định cho tới khi A/B pass cả retrieval và answer quality.

## Pha 4 — Evidence assembly và contextual retrieval

- Final bundle tối đa 12 evidence units, có diversity theo document, page, modality và subquery.
- Child chunk dùng để match; parent section dùng để đọc.
- Table giữ caption, header và query-relevant rows.
- Figure giữ crop, full-page context, caption và surrounding text.
- Validator tiếp tục đọc canonical ledger riêng; ledger không đi vào prompt.

Aya đã có contextual prefix và parent expansion, nên không tạo thêm 50–100 token context bằng Sol cho mỗi chunk. Cách hiện tại tiếp tục được A/B như một phiên bản local/deterministic của contextual retrieval. [Anthropic Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval)

Late chunking chỉ là nhánh nghiên cứu sau blind-v2 vì cần embedding model hỗ trợ long context; không áp trực tiếp lên EmbeddingGemma hiện tại. [Late Chunking paper](https://arxiv.org/abs/2409.04701)

## Pha 5 — Reranker và visual retrieval, chỉ khi cần

### Reranker gate

Chỉ thử reranker khi Candidate Recall@30 đạt `≥0.95`.

- Baseline đầu tiên dùng cross-encoder adapter hiện có.
- Candidate tối đa 20, input tối đa 512 tokens/candidate.
- Warm rerank timeout 800 ms; timeout trả first-stage ranking kèm diagnostic, không silent failure.
- Qwen3-Reranker-0.6B là experiment thứ hai, yêu cầu adapter riêng và chỉ giữ nếu thắng blind-v2 trên M1. Model hỗ trợ multilingual, 32K context và Apache 2.0. [Qwen3 Embedding/Reranker](https://qwenlm.github.io/blog/qwen3-embedding/)
- Không promote 4B/8B trên M1 16 GB.

### Visual branch

Nếu figure/page Evidence Recall@5 vẫn dưới `0.80`, dựng isolated page-image multi-vector experiment theo ColPali/ViDoRe; không thay text/table pipeline ngay. [ColPali](https://arxiv.org/abs/2407.01449)

## Acceptance gates

- Candidate Recall@30 ≥ `0.95`.
- Document Recall@5 ≥ `0.90`.
- Evidence Recall@10 ≥ `0.85`.
- Không modality nào dưới `0.80`.
- Multi-paper required-document coverage ≥ `0.95`.
- Supported numeric claims ≥ `0.98`; zero unsupported number được emit.
- Unanswerable abstention ≥ `0.90`.
- Warm local retrieval, bao gồm dual local searches:
  - P50 ≤ `500 ms`
  - P95 ≤ `1.5 s`
- Rewrite, retrieval, rerank, validation và Sol generation phải có clock riêng.
- Feature chỉ promote nếu tăng blind-v2 và không regression dev-20/regression-v1.
- Không dùng model-authored answer hay top-level agent pass thay cho evidence qrels.

## Thứ tự Luna/Terra triển khai

1. Commit bản PLAN.md đã hiệu chỉnh và freeze dataset manifests/hashes.
2. Thêm typed EvidenceBundle + evaluation instrumentation.
3. Implement page/document readiness và OCR diagnostics.
4. Implement candidate-document union và dual-query RRF dưới feature flag.
5. Migrate context composer/validator sang EvidenceBundle.
6. Chạy retrieval-only dev-v2; sửa candidate generation trước reranker.
7. A/B reranker và visual branch nếu acceptance còn thiếu.
8. Chạy Sol E2E, shadow-v2 và public multimodal gates.
9. Atomic re-index/promote với backup và rollback.
10. Chỉ sau khi document gate pass mới thêm web/MCP/shell/sub-agent tools.

Giữ nguyên trong vòng đầu: EmbeddingGemma, LightRAG graph, Sol answer policy và production index. Không thêm tool, không bulk re-embed và không thêm foreground LLM retrieval call.
