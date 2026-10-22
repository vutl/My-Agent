# Paper Evidence Cards & Held-out Conversational Evaluation Plan

**Trạng thái:** Review-only, chưa implement
**Ngày audit:** 2026-08-12
**Phạm vi:** canonical evidence cards, multi-paper retrieval, selective raw fallback, progressive validation/streaming, held-out conversational eval
**Model policy:** chỉ `cx/gpt-5.6-sol`, `cx/gpt-5.6-terra`, `cx/gpt-5.6-luna`; mặc định `cx/gpt-5.6-sol`; không fallback sang Ollama/local cho router hoặc answer

## 1. Kết luận audit

Năm việc được đề xuất là đúng hướng, nhưng không nên nối trực tiếp một lớp “LLM summary card” vào prompt. Cách đó có thể nhanh hơn retrieval hiện tại nhưng biến hallucination lúc index thành “evidence” tồn tại lâu dài.

Thiết kế nên là:

1. Giữ `document_cards` hiện tại cho catalog/discovery.
2. Thêm `PaperEvidenceCard` riêng, chuẩn hóa theo năm facet cốt lõi: `task`, `architecture`, `dataset_setup`, `benchmark_results`, `contributions`.
3. Mỗi ý trong card phải trỏ về canonical chunk/table/figure, có quote/hash/page và trạng thái validation. Card là lớp nén/navigation, không phải nguồn sự thật tự thân.
4. Khi chat, load card của toàn bộ paper đã resolve bằng một batch DB read. Chỉ chạy raw retrieval song song cho đúng các `(paper, facet)` còn thiếu/stale/không hợp lệ.
5. Stream theo section của từng paper sau khi section đó được validate; không đợi toàn bộ bài trả lời nếu các section trước đã an toàn.
6. Đóng băng một held-out set 60 turn trước khi tuning feature này và chỉ chạy full set khi dev gate đã qua.

```text
User turn
   -> deterministic route + canonical document scope + requested facets
   -> batch-load valid evidence cards for all scoped papers
   -> coverage matrix: (paper x facet)
   -> parallel raw fallback for missing cells only
   -> fair per-paper context bundles
   -> one structured Sol generation stream
   -> validate complete paper section
   -> emit validated section
   -> validate/emit synthesis after all required papers are covered
```

## 2. Hiện trạng đã kiểm tra

### 2.1 Có thể tái sử dụng

- Canonical SQLite đã có `documents`, `chunks`, `document_tables`, `document_figures`, content hash, stable document ID và FK cascade.
- Reindex giữ document identity và thay canonical derived artifacts theo document.
- Scope resolver đã resolve paper trước router, giữ sticky focus và `must_cover_all` cho multi-paper.
- Query facet detection hiện đã có `architecture`, `training_method`, `benchmark_results`, `dataset_setup`, `ablation`, `limitations`, `visual_evidence`.
- Multi-paper retrieval đã decomposed thành một branch/document và chạy bằng `asyncio.gather`.
- Context composer đã giới hạn budget, giữ provenance và ưu tiên table/figure khi query yêu cầu.
- Exact canonical table path có thể trả bảng không cần model generation.
- `validate_answer_claims` đã kiểm metric/value, owner, document scope và multi-document coverage.
- Non-quantitative paper QA đã có progressive validated Markdown blocks.
- Public HTTP/SSE E2E harness đã có, nhưng mới là một flow 7 turn cố định.

### 2.2 Khoảng trống thật sự

- `document_cards` hiện tại chỉ có title guess, phần đầu văn bản làm summary, tags và keyword frequency. Nó không phải evidence card.
- Live DB cho thấy title guess có thể là một câu giữa abstract thay vì title paper. Vì vậy không nên mở rộng card này thành nguồn evidence.
- Chưa có schema chuẩn hóa `task/architecture/dataset/results/contribution` và chưa có evidence provenance ở cấp từng claim.
- Facet coverage hiện suy ra từ heading/artifact metadata; chưa có coverage ledger bền vững theo từng paper.
- Retrieval vẫn đưa raw excerpt cho mọi query, kể cả facet đã có thể được nén an toàn.
- Multi-paper raw retrieval đã parallel, nhưng chưa có card-first/missing-facet-only policy.
- `must_cover_all` và quantitative request hiện buffer cả answer để bảo toàn correctness; chưa có validator theo paper section.
- `scripts/evaluate_retrieval.py` chỉ chấm một expected document/query, không chấm multi-paper, facet, evidence ID, streaming hay conversational state.
- Bộ 8 dev query và 30 title-ablated diagnostic không phải held-out conversational evaluation.

## 3. Nguyên tắc không được phá

1. **Canonical evidence remains authoritative.** Summary/synopsis trong card không được dùng để hợp thức hóa claim nếu source refs không tồn tại hoặc đã stale.
2. **No paper-specific mapping.** Không hardcode paper -> section/table/figure/facet.
3. **Fail closed.** Card thiếu, stale, parser mismatch, source hash mismatch hoặc invalid JSON thì raw fallback; nếu raw retrieval vẫn thiếu thì Aya nói thiếu evidence.
4. **No online card generation.** Chat request không chờ Sol xây card. Xây card là indexing/backfill job; online path chỉ load hoặc fallback.
5. **No model fallback.** 9router timeout/quota/unavailable thì job dừng ở trạng thái có thể resume; không đổi model ngoài allowlist và không chuyển local.
6. **Scope before retrieval.** Card hoặc raw fallback không được thay đổi canonical document scope đã resolve.
7. **Exact artifact intent bypasses summary.** User xin “Table 2”, “Figure 3”, quote, row hay số liệu chi tiết thì dùng canonical artifact/raw evidence path.
8. **Coverage and rhetoric are independent.** `must_cover_all` là nghĩa vụ phủ paper, không ép mọi multi-paper request thành intent `compare`.

## 4. Canonical facet taxonomy

Tạo một source of truth mới: `backend/app/rag/paper_facets.py`.

### 4.1 Core facets

| Canonical key | Chứa gì | Không chứa gì |
|---|---|---|
| `task` | Problem, input/output, target setting, research question | Marketing/background chung |
| `architecture` | Components, feature flow, fusion, encoder/decoder, training-relevant structure | Result numbers không giải thích kiến trúc |
| `dataset_setup` | Dataset, split, labels, modalities, protocol, preprocessing dùng trong experiment | Claim hiệu năng |
| `benchmark_results` | Main comparison, metric/value, baseline, ablation result khi cần | Số không gắn owner/metric/dataset |
| `contributions` | Novelty/claimed contributions có evidence từ paper | Đánh giá chủ quan của Aya |

### 4.2 Auxiliary facets giữ lại

`training_method`, `ablation`, `limitations`, `visual_evidence` tiếp tục tồn tại cho query planning. Chúng có thể map vào core card khi hợp lý nhưng không bị xóa khỏi retrieval layer.

Ví dụ:

- `ablation` có thể bổ sung `benchmark_results` nhưng vẫn mang tag riêng.
- `visual_evidence` là evidence kind, không phải lúc nào cũng là nội dung `architecture`.
- `training_method` có thể đứng riêng để tránh nhồi quá nhiều vào `architecture`.

Taxonomy, marker tiếng Việt/Anh, query expansion, extraction labels và eval labels phải import từ cùng module; không giữ nhiều regex list lệch nhau.

## 5. Data model đề xuất

Không sửa nghĩa của `document_cards`. Thêm ba bảng canonical mới.

### 5.1 `paper_evidence_cards`

Một row/document/version hợp lệ:

- `id`
- `document_id` (FK cascade, unique cho version active)
- `document_content_hash`
- `parser_name`, `parser_version`
- `schema_version`
- `prompt_version`
- `generator_provider`, `generator_model`
- `status`: `building | complete | partial | failed | stale`
- `coverage_json`
- `created_at`, `updated_at`
- `metadata_json`

### 5.2 `paper_evidence_facets`

Một row/card/facet:

- `id`, `card_id`, `document_id`
- `facet`
- `synopsis`
- `status`: `complete | partial | unavailable | invalid`
- `confidence`
- `source_count`
- `facet_hash`
- `created_at`, `updated_at`
- unique `(card_id, facet)`

### 5.3 `paper_evidence_items`

Một item là một ý nhỏ có thể kiểm chứng:

- `id`, `facet_id`, `document_id`, `ordinal`
- `claim_text`
- `evidence_refs_json`
- `validation_status`
- `validation_reason`
- `created_at`

Mỗi evidence ref phải có:

- `source_kind`: `chunk | table | figure`
- `source_id`
- `page`
- `section_title/heading_path` nếu có
- exact/normalized `quote`
- `source_content_hash`
- optional `table_row_keys`, `metric_keys`, `figure_label`

Do `source_id` có thể trỏ tới ba bảng khác nhau, application validator phải kiểm `(source_kind, source_id, document_id)` trong transaction; không chấp nhận ID chỉ vì nó tồn tại ở paper khác.

### 5.4 Invalidation

Card chỉ hợp lệ khi đồng thời khớp:

- current `documents.content_hash`;
- parser identity/version;
- evidence-card schema version;
- extraction prompt version;
- tất cả referenced source IDs + source hashes.

Reindex không được xóa active card trước khi card mới build xong. Build vào staging rows, validate đầy đủ, rồi swap active card trong một transaction. Nếu build lỗi, canonical document mới vẫn dùng raw retrieval và card cũ được đánh `stale`, không được đưa vào prompt.

## 6. Card build pipeline

### 6.1 Candidate selection không dùng LLM

Với mỗi paper:

1. Đọc canonical chunk headings, table captions/types, figure captions/types và nearby page context.
2. Gán candidate facets bằng taxonomy chung.
3. Chọn candidate pool có diversity theo section và artifact, không chỉ top vector score.
4. Ưu tiên exact structured artifacts cho result/dataset và architecture figure khi chất lượng figure đã qua gate.
5. Không lấy logo, author block, reference-only mention hoặc figure panel bị cắt làm evidence chính.

Candidate cap phải cấu hình được; khởi điểm đề xuất 3-5 source/facet, tối đa khoảng 18 source/paper trước extraction.

### 6.2 Một Sol extraction call/paper

- Một request chứa năm core facets và candidate sources của đúng một paper.
- Strict JSON schema, temperature thấp, source IDs là enum từ candidate set.
- Không gọi một model request/facet vì sẽ tăng cost/latency gấp năm.
- Build nhiều paper song song với bounded concurrency mặc định `2`; không mở concurrency không giới hạn trên M1/9router.
- Timeout/quota/provider error ghi `failed`, giữ resume cursor và dừng theo policy; không fallback model.

### 6.3 Deterministic validation sau extraction

Trước khi publish card:

1. JSON/schema validation.
2. Tất cả source IDs thuộc đúng document và candidate set.
3. Quote xuất hiện trong normalized canonical source.
4. Numeric result đi qua metric/value/owner validator hiện có.
5. `benchmark_results` không được chỉ dựa vào prose generated nếu paper có structured table tương ứng nhưng source ref không trỏ table/chunk hợp lệ.
6. Không cho một source của paper A vào card paper B.
7. Item fail thì loại item; facet còn source hợp lệ có thể `partial`; không fail cả paper một cách không cần thiết.

`synopsis` chỉ được materialize từ các item đã valid. Runtime prompt luôn nhận cả synopsis và compact provenance, không nhận synopsis trần.

## 7. Runtime: card-first, missing-facet-only fallback

### 7.1 Coverage matrix

Sau scope resolution và facet extraction:

```text
required = ordered_document_ids x requested_facets
```

Nếu query không gọi facet cụ thể:

- direct “paper làm gì?” -> `task + contributions`;
- compare tổng quát -> `task + architecture + dataset_setup + benchmark_results + contributions`, nhưng context budget có thể chọn compact synopsis của cả năm;
- exact table/figure -> canonical artifact path, không dùng default facet expansion để thay thế artifact.

Mỗi cell có trạng thái `card_valid`, `card_partial`, `missing`, `stale`, `raw_recovered`, `unavailable`.

### 7.2 Load nhiều paper

- Dùng một batch SQLite query cho toàn bộ ordered document IDs; một query nhanh và nhất quán hơn tạo nhiều thread DB read giả-parallel.
- Sau batch read, assemble logical per-paper bundles độc lập.
- Chỉ các cell không `card_valid` mới tạo raw retrieval branches.
- Raw branches chạy song song bằng cơ chế `asyncio.gather` hiện có, giữ exact document scope và global hop cap.
- Merge theo round-robin paper/facet để paper đầu không ăn hết context budget.

### 7.3 Khi nào bắt buộc raw excerpt

- facet thiếu/stale/invalid;
- user xin exact quote/page/table/figure/row;
- user hỏi chi tiết vượt synopsis, ví dụ optimizer, split cụ thể, per-class metric;
- validator không thể chứng minh numeric claim từ card refs;
- card chỉ có `partial` và phần thiếu liên quan trực tiếp query.

Không bổ sung raw excerpt chỉ vì “có thể hữu ích”. Diagnostics phải ghi rõ `card_hit`, `raw_fallback_reason`, `paper_facet_coverage`.

### 7.4 Không thêm LanceDB table ở phase đầu

Phase 1 lưu evidence cards trong SQLite vì document scope và facet key đã biết; exact lookup vừa nhanh vừa tránh re-embed toàn corpus.

Chỉ cân nhắc vector hóa facet cards sau held-out eval nếu unscoped discovery vẫn là bottleneck. Khi đó phải dùng table riêng, fingerprint riêng và atomic replacement; không trộn vào `document_cards` hiện tại.

## 8. Progressive per-paper validation và streaming

### 8.1 Output contract

Prompt yêu cầu một section có machine-readable boundary cho từng canonical document, theo đúng scope order, sau đó mới có synthesis section.

Boundary không được gửi ra UI. Ví dụ nội bộ:

```text
<paper document_id="..."> ... </paper>
<paper document_id="..."> ... </paper>
<synthesis> ... </synthesis>
```

### 8.2 State machine

1. `evidence.paper.ready`: coverage bundle của một paper đã sẵn sàng; đây là progress event, không phải answer claim.
2. Buffer đến khi đóng một `<paper>` section.
3. Validate section với evidence chỉ của paper đó:
   - đúng document identity;
   - required facets được nhắc hoặc section nói rõ facet nào thiếu evidence;
   - numeric claims đúng source;
   - không mượn owner/value từ paper khác.
4. Section valid -> emit `answer.paper.validated`, rồi stream section thành `message.delta` chunks.
5. Section invalid -> sanitize hoặc một correction call chỉ cho section đó; không regenerate các section đã valid.
6. Nếu model kết thúc mà thiếu paper, append deterministic insufficiency section cho paper đó. Không im lặng bỏ paper.
7. Synthesis chỉ emit sau khi mọi required paper đã có valid/insufficient section. Synthesis không được thêm numeric fact mới; cross-paper delta chỉ do deterministic arithmetic hoặc được validate riêng.

Như vậy `must_cover_all` vẫn đúng ở cuối response nhưng user có thể thấy paper đầu sau khi section đó hoàn tất, thay vì chờ toàn answer.

### 8.3 Safety boundaries

- Exact multi-paper quantitative tables vẫn có thể dùng deterministic canonical renderer khi mỗi table đã resolve không mơ hồ.
- Không token-stream một metric row chưa hoàn chỉnh.
- Không coi heading của paper là coverage nếu phần dưới nói về paper khác.
- UI phải phân biệt `retrieval/card progress` với `validated answer text`.
- Nếu delimiter malformed, fallback về whole-answer buffer; không lộ draft chưa validate.

## 9. API, settings và observability

### 9.1 Endpoint dự kiến

- `POST /rag/evidence-cards/build` cho một document.
- `POST /rag/evidence-cards/build-all` với `limit`, `force`, `resume`, `max_concurrency`.
- `GET /rag/documents/{document_id}/evidence-card` để audit.
- `GET /rag/evidence-cards/status` để xem complete/partial/stale/failed.

Backfill endpoint phải dùng privacy gate hiện có. Không gửi corpus qua 9router nếu gate chưa được bật cho job đó.

### 9.2 Settings/feature flags

- `PAPER_EVIDENCE_CARDS_ENABLED=false` ban đầu.
- `PAPER_EVIDENCE_CARD_BUILD_ENABLED=false` ban đầu.
- `PAPER_EVIDENCE_CARD_MODEL=cx/gpt-5.6-sol`.
- `PAPER_EVIDENCE_CARD_MAX_CONCURRENCY=2`.
- `PAPER_EVIDENCE_CARD_SCHEMA_VERSION=v1`.
- `PAPER_SECTION_STREAMING_ENABLED=false` ban đầu.

### 9.3 Trace fields

- requested facets;
- per-paper card version/status;
- card-hit and raw-fallback cells;
- source IDs included in each bundle;
- build/extraction/validation timing;
- first evidence-ready, first paper-validated, first message delta;
- per-section validation result and correction attempts;
- prompt chars saved versus raw-only baseline.

Không ghi full raw prompt hoặc unbounded paper text vào normal trace metadata.

## 10. Held-out conversational evaluation

### 10.1 Dataset layout

Tạo trước khi tuning:

- `data/retrieval_eval/conversational-dev-v1.jsonl`: 20 turn để phát triển.
- `data/retrieval_eval/conversational-heldout-v1.jsonl`: **60 held-out turn**, nhóm thành 15 conversation x 4 turn.
- `data/retrieval_eval/conversational-heldout-v1.manifest.json`: corpus fingerprint, schema version, labeler/review state, checksum và freeze timestamp.

Không dùng held-out failures để sửa regex/case từng câu. Nếu phải tune sau khi mở set, retire cả version và tạo v2 từ câu mới.

### 10.2 Phân bố tối thiểu cho 60 turn

- 36 tiếng Việt, 24 tiếng Anh.
- Tất cả paper trong corpus xuất hiện ít nhất hai lần; không để ASPIRE/ViSEC chiếm đa số.
- 12 single-paper facet asks.
- 12 multi-paper compare/extract/list asks.
- 10 sticky follow-up qua casual detour rồi quay lại paper.
- 8 correction/referent turns.
- 6 exact table/figure/result turns.
- 6 hard-negative/general-chat alias traps.
- 6 unanswerable/partial-evidence/abstention turns.

Các nhóm có thể overlap, nhưng manifest phải báo coverage thực tế và cấm duplicate paraphrase giữa dev/held-out.

### 10.3 Label schema

Mỗi turn cần:

- `conversation_id`, `turn_index`, `message`, prior-turn fixture;
- expected route/intent;
- ordered expected document IDs;
- `must_cover_all`;
- expected facets per document;
- acceptable evidence source IDs/kinds hoặc alternative source groups;
- forbidden document IDs;
- expected table/figure IDs khi exact;
- expected abstention;
- allowed/forbidden numeric claims nếu phù hợp;
- latency class, không hardcode provider wall time vào correctness label.

### 10.4 Evaluator

Tạo `scripts/evaluate_agent_conversations.py`, gọi đúng public HTTP/SSE API như UI và chấm:

1. route/intent;
2. exact document scope + order;
3. sticky/correction/referent state;
4. facet coverage matrix;
5. evidence source precision/recall;
6. must-cover-all omissions;
7. unsupported quantitative claims;
8. table/figure identity;
9. abstention correctness;
10. SSE order, first evidence-ready, first validated paper, first answer delta, total latency;
11. exact model identity and no fallback.

LLM judge chỉ là diagnostic phụ cho chất lượng văn phong/qualitative completeness; không được thay deterministic provenance/claim gates và không được là ground truth duy nhất.

### 10.5 Acceptance gates

Correctness gates:

- 0 cross-paper evidence leak.
- 0 unsupported quantitative claim.
- 0 omission trong explicit `must_cover_all`.
- 100% exact document scope cho explicit named/correction cases.
- >= 95% expected facet coverage overall.
- >= 90% abstention correctness.
- 100% exact table/figure identity trong exact-artifact cases.
- model allowlist/fallback violations = 0.

Latency/resource gates:

- local batch card load P95 <= 50 ms trên corpus hiện tại;
- card hit không được kích hoạt embedding/raw retrieval;
- missing-facet fallback không được retrieve facet đã complete;
- retrieval P95 không regression quá 15% so với cùng held-out turn ở raw-only baseline;
- first validated paper và total latency báo paired distribution; rollout chỉ tiếp tục nếu median tốt hơn baseline và P95 không tệ hơn đáng kể;
- backfill concurrency không vượt configured cap.

## 11. Implementation phases

### Phase 0 — Freeze contract và baseline

1. Chốt taxonomy/module API.
2. Viết 20 dev + 60 held-out labels, freeze checksum.
3. Chạy raw-only baseline một lần và lưu trace/result.
4. Review plan + label schema bằng Antigravity/another model và manual sample audit.

**Exit:** manifest frozen, baseline reproducible, không có code path mới bật production.

### Phase 1 — Schema + validator + deterministic candidate selector

Files chính:

- `backend/app/db/sqlite.py`
- new `backend/app/rag/paper_facets.py`
- new `backend/app/services/paper_evidence_service.py`
- tests schema, stale detection, cross-document ref rejection, quote/hash validation.

**Exit:** có thể tạo/load một card synthetic hoàn toàn không gọi model; invalid/stale card fail closed.

### Phase 2 — Sol builder + resumable backfill

Files chính:

- new `backend/app/services/paper_evidence_builder.py`
- `backend/app/api/rag.py`
- `backend/app/core/config.py`
- indexing lifecycle hook trong `backend/app/services/indexing_service.py`.

**Exit:** build một paper và bounded parallel build nhiều paper; timeout/quota resume được; không fallback model; card publish atomic.

### Phase 3 — Runtime card-first + selective fallback

Files chính:

- `backend/app/services/retrieval_agent_service.py`
- `backend/app/api/agent.py`
- `backend/app/rag/context.py`
- `backend/app/agents/state.py`
- `backend/app/agents/graph.py`.

**Exit:** matrix `(paper x facet)` audit được; complete cell không raw retrieve; missing cell fallback đúng document/facet; exact table/figure path không regression.

### Phase 4 — Per-paper progressive validation/streaming

Files chính:

- new `backend/app/services/paper_section_stream.py`
- `backend/app/services/evidence_validator.py`
- `backend/app/api/agent.py`
- desktop SSE/trace rendering trong `apps/desktop/src`.

**Exit:** first valid paper section xuất hiện trước generation end; malformed boundary fail closed; missing paper có explicit insufficiency section; synthesis không thêm số mới.

### Phase 5 — Eval runner + gates

Files chính:

- new `scripts/evaluate_agent_conversations.py`
- new dev/held-out JSONL + manifest;
- CI/local report artifact dưới `data/retrieval_eval/`.

**Exit:** dev pass trước; held-out chạy đúng một lần cho release candidate; report chứa correctness, provenance, streaming, latency và model identity.

### Phase 6 — Rollout

1. Feature flags off -> shadow card coverage only.
2. Bật card-first cho single-paper qualitative asks.
3. Bật multi-paper card-first.
4. Bật per-paper streaming.
5. Chỉ sau held-out gate mới đặt default on.

Rollback chỉ cần tắt flags; canonical raw retrieval path vẫn tồn tại.

## 12. Test matrix bắt buộc

### Unit/integration

- five-facet taxonomy VI/EN và no false generic alias;
- one/many paper card batch load preserves scope order;
- stale hash/parser/prompt/schema versions;
- source ID from wrong paper rejected;
- numeric result source validation;
- partial facet triggers only that raw branch;
- complete facet never triggers raw search;
- exact table/figure bypass;
- parallel raw fallback bounded, timeout isolated, no silent widening;
- context fairness across 2/3 papers;
- per-paper delimiter split across arbitrary token chunks;
- malformed/missing/duplicated section IDs;
- one section correction does not regenerate valid sections;
- final synthesis cannot introduce unsupported metrics;
- SSE event order and desktop progressive render;
- privacy gate and allowed-model enforcement;
- reindex atomic swap and failed-build recovery.

### Regression

Giữ toàn bộ existing suites về:

- document scope/correction grammar;
- sticky focus/history;
- retrieval decomposition/LightRAG provenance;
- canonical table/figure;
- evidence validator;
- answer claim guard/streaming;
- model policy;
- vector index replacement;
- desktop build.

## 13. Chi phí và performance kỳ vọng

- Backfill tốn khoảng **một Sol extraction call/paper**, không phải năm call/facet. Đây là chi phí một lần cho mỗi content/prompt/schema version.
- Online card-hit path không gọi model, embedding hay LightRAG; chỉ batch SQLite read + final answer generation.
- Raw fallback chỉ tốn cho facet thiếu hoặc exact-detail request.
- Một final answer vẫn cần Sol, nên evidence cards giảm retrieval/prompt work nhưng không xóa hoàn toàn provider generation latency.
- Per-paper section streaming cải thiện thời điểm user thấy nội dung đã validate; nó không làm Sol sinh token nhanh hơn.
- Không re-embed toàn corpus ở phase đầu.

## 14. Những điểm cần reviewer phản biện trước khi implement

1. Có đồng ý giữ `document_cards` cho catalog và tạo schema evidence riêng không?
2. Core card chỉ năm facet hay đưa `training_method`, `ablation`, `limitations` thành core ngay v1?
3. Có đồng ý một Sol call/paper + concurrency 2, thay vì one call/facet hoặc unbounded parallel không?
4. Per-paper streaming có chấp nhận mô hình “validated section hoặc explicit insufficiency section”, thay vì atomic whole-answer buffer không?
5. Có đồng ý không vector hóa evidence cards ở phase đầu không?
6. Held-out 60 turn có được freeze trước implementation và chỉ dùng 20-turn dev để tuning không?
7. Acceptance thresholds ở mục 10.5 có cần tăng/giảm trước khi đóng băng manifest không?

## 15. Quyết định khuyến nghị

Khuyến nghị duyệt nguyên thiết kế với các lựa chọn sau:

- evidence schema riêng;
- five core facets + auxiliary facet tags;
- one Sol call/paper, bounded concurrency 2;
- SQLite-only card lookup ở v1;
- raw fallback theo missing cell;
- one structured Sol answer stream, validate/release per paper section;
- 20 dev + 60 frozen held-out turns;
- rollout bằng feature flags và raw path luôn giữ làm rollback.

Sau khi reviewer chấp thuận hoặc sửa các quyết định trên mới bắt đầu Phase 0/1; không backfill corpus và không bật production flag trong giai đoạn review.
