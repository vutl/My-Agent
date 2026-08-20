# 05 — TODO và vấn đề tồn đọng (2026-08-20)

## Baseline blocker

Không còn blocker baseline từ quota/E2E. Strict sequence ASPIRE architecture → casual coffee/tea → resume benchmark → WhiSER → previous paper → architecture figure/no logo → benchmark table/no invented values đã pass ngày 2026-07-19 với conversation `b08d1f90-c050-4d74-8c2f-b04125ba3796`. DB/source/history audit cũng đã pass; xem `00-STATUS.md` và `04-DONE.md`.

## Bulk corpus checkpoint

User đã approve rõ ngày 2026-07-23 việc gửi toàn bộ figure/page images và
extracted paper text qua 9router.

- GPT-5.6 Sol full-corpus visual enrichment: **đã xong** (60 enriched,
  76 quality-skipped, 0 failed).
- Canonical LightRAG ingest: **đã xong 21/21 processed**, 0 failed/unready;
  graph 8.559 nodes / 14.173 edges. Lượt cuối prune 7 duplicate tombstones.

## Index / data còn theo dõi

1. CRAB/source legacy còn thiếu hoặc provenance chưa hoàn chỉnh; cần source/canonical re-index riêng khi có file đúng.
2. Theo dõi các `needs_review` visuals sau GPT enrichment; không tự promote crop nhỏ/logo chỉ để tăng recall.
3. Embedding đã promote sang `embeddinggemma:300m` ngày 2026-08-02 và rebuild
   cả LanceDB lẫn LightRAG entity/relation/chunk VDB. Lần đổi sau vẫn phải
   rebuild cả hai, không chỉ sửa env.
4. Retrieval v3 mới có 8 labeled development queries, nên Hit@3=1.0 chưa phải
   bằng chứng production. Cần freeze held-out **>=60 câu** trước khi tune tiếp,
   split theo paper/family để tránh leakage; bao phủ 21 documents, Vietnamese/
   English, true multi-hop, hard-negative generic hubs, table/figure,
   unanswerable, provenance/path validity và answer-grounding.
5. Document title extraction của một số paper (ví dụ ASPIRE/KST) vẫn lấy nhầm
   câu body thay vì title thật. Filename/context prefix đang bảo vệ retrieval,
   nhưng nên sửa parser/card metadata để document routing bền hơn.
6. EmbeddingGemma đã sửa mạnh raw unscoped Vietnamese discovery trên diagnostic
   hiện có: VI Hit@3 13/15, Hit@6 15/15. Tuy nhiên bộ 30 câu không held-out;
   vẫn cần freeze slice >=60 câu và đo full-agent query rewrite/graph/table/
   figure trước khi coi đây là quality ceiling. Xem
   `data/retrieval_eval/RAG_SYSTEM_EVALUATION_20260731.md`.
7. Explicit document routing đã có corpus-wide regression trên 21 documents,
   nhưng alias ngắn trùng nhiều file (hiện thấy `9router`) cố ý fail-closed; UI
   nên hỏi user chọn file nếu muốn xử lý ambiguity thay vì tự chọn.

## Memory / agent roadmap

| # | Việc | Layer |
|---|------|-------|
| 7 | SSE/WebSocket hoặc polling event `agent.memory.updated` sau background commit | Optional UX |
| 8 | LongMemEval-style corpus lớn: extraction, multi-session, temporal update, abstention | Memory evaluation |
| 9 | Theo dõi precision của auto L3 ops trước khi tăng limit/confidence recall | L3 quality |
| 10 | Optional semantic embedding cho L0 sau khi FTS/episode baseline có số đo | L0 retrieval |
| 11 | Audit thêm natural resume/topic-switch edge cases | L1/L2 |

Đã xong ngày 2026-07-19: durable/coalesced L2, pending-turn injection,
same-thread serialization, startup recovery, L0 episode search, explicit
cross-thread recall và versioned semantic/episodic/procedural L3.

## RAG / architecture roadmap

| # | Việc | Ghi chú |
|---|------|--------|
| 12 | Full ReAct tool loop trong LangGraph | Hiện vẫn là controlled pipeline |
| 13 | Tools: `read_file`, `web_search`, `read_table`, calendar, shell… | Chỉ thêm theo scope/quyền phù hợp |
| 14 | Chọn/tải và benchmark cross-encoder multilingual local; cân nhắc ONNX quantization | Backend PyTorch/local-only đã có; chưa bật production |
| 15 | Enclosing absolute request deadline + measured generation reserve | Hop-2 branch đã timeout 45s; chưa có absolute whole-request budget |
| 16 | Optional LLM subquery planner A/B | Default deterministic planner đã xong; chỉ thử nếu đo được gain/cost/latency |

## Rủi ro vận hành

- Table fast path chỉ áp dụng cho exact `Table/Bảng N`, hoặc một canonical
  main-result table duy nhất có caption comparison/performance rõ và không chứa
  ablation/distribution/statistics. Candidate phải cùng resolved `document_id`;
  câu phân tích/đánh giá vẫn cần Sol. Docling extraction là evidence, không được
  coi mặc định là ground-truth hoàn hảo.
- Research comparison dài đã giảm từ 76,48 s xuống 24,43 s trong live smoke, nhưng
  19,21 s vẫn là một Sol answer generation. Không hứa 2–5 s total cho câu dài;
  cần đo thêm output-length control hoặc provider latency trước khi tối ưu tiếp.
- Explicit/focused structured lookup đã bypass provider router và direct-render
  unique canonical result table. Không mở rộng rule này sang cross-paper compare
  hoặc ambiguous equal-score tables; các trường hợp đó vẫn cần planner/model hoặc
  user chọn rõ bảng.
- Non-quantitative RAG stream theo validated paragraph, nên first visible output
  phụ thuộc model đóng paragraph đầu tiên. Quantitative/table analysis vẫn cố ý
  buffer whole answer để không lộ số chưa validate; UI giờ báo/hiển thị theo
  `buffered_validation` hoặc `validated_reveal`, không gọi đó là token stream thật.
- Router, answer, memory và figure VLM phụ thuộc 9router/quota GPT-5.6; policy
  hiện fail-closed và không silently đổi model. Health expose memory retry error.
- Stable summary có thể chưa fold xong ở turn kế tiếp; durable pending full turns
  + raw recent + L0 retrieval là lớp bảo vệ, không còn chỉ dựa recent beats.
- 101 legacy conversations đang `dormant`; cố ý không bulk summarize. Thread nào
  có turn mới sẽ tự wake và fold backlog của đúng thread đó.
- LightRAG helper upstream chỉ trả text, không expose `response.model`; app pin
  request model đã chọn (mặc định `cx/gpt-5.6-sol`), bỏ model chain và dùng
  native stream. Chưa thể attest response envelope như answer/router/vision.
- LightRAG extraction chạy tuần tự (`max_async=1`) với chunk 600/overlap 80.
  Timeout/connection chỉ retry một lần cùng Sol. Extract role timeout 660s
  (worker watchdog 1.320s); foreground query vẫn 240s để provider outage không
  treo chat lâu.
- Adaptive retrieval có thể phát sinh tối đa 3 query ở hop 2 cho compare,
  infer-structure hoặc multi-facet ask khi hop 1 thiếu coverage. Các branch chạy
  song song nhưng vẫn là upstream LightRAG calls nếu policy chọn graph; quota/
  auth/provider error nổi ra và dừng run, không đổi model/local fallback.
- 47/382 legacy LightRAG chunks chưa map parent vì fuzzy ownership không đủ rõ.
  Đây là intentional fail-closed; raw fallback chỉ được dùng cho chunk chưa từng
  map và có canonical document provenance duy nhất, không dùng cho stale mapping.
- Graphify/Obsidian là code navigation cache, không thay thế source of truth.
  Sau khi sửa code phải chạy `graphify update .`; với behavior/security-sensitive
  work vẫn phải mở source nhỏ nhất mà graph chỉ ra để xác minh.
- `obsidian-lightrag-vault/` là snapshot generated của LightRAG document graph,
  không tự đồng bộ khi ingest thay đổi. Chạy lại
  `scripts/export_lightrag_obsidian.py` sau một corpus ingest mới; không nhập
  lẫn vault này với `obsidian-vault/` của source code.
- `uvicorn --reload` hoặc conda env thiếu `lightrag` có thể tạo process sai dependency; ưu tiên `backend/.venv`.
- Staged re-index phải giữ document cũ nếu parse mới fail; không prune graph/vector trước khi replacement sẵn sàng.
- LanceDB v3 hiện live; rollback v2 nằm ở
  `data/lancedb-v2-pre-chunk-v3-20260730/`. Không xóa backup trước khi chạy
  full 21-document + Vietnamese/table/figure eval.
- `RERANK_ENABLED=false` là chủ ý theo số đo. Không bật lại `embedding` chỉ vì
  tên “reranker”; cross-encoder phải có model local rõ ràng và A/B tốt hơn fast
  hybrid RRF trước khi promote.

## Cố ý không làm

- [x] Promote EmbeddingGemma-300M vào production ngày 2026-08-02; LanceDB và
  LightRAG VDB đã rebuild, runtime smoke + dev-8 + diagnostic 30 câu pass.
- [ ] Xây held-out multilingual set lớn hơn và đo full retrieval pipeline để
  tránh overfit vào diagnostic 30 câu hiện tại.
- [x] Đã thêm isolated Hugging Face provider và chấm GTE, Nomic v2, Jina v5.
  Jina v3 được bỏ khỏi full gate vì v5-small đã thay thế.
- [ ] Trước khi promote Jina v5-small: tạo held-out multilingual/table/figure
  gate, chạy full-agent E2E và xác nhận CC BY-NC phù hợp mục đích sử dụng.
- [ ] Direct figure/page embedding: đánh giá riêng v5-omni-small (2B) theo kiểu
  vision + text shared space; không re-index text bằng v4.

- Hardcode “paper X luôn lấy Figure N”.
- Đưa router/answer/figure VLM về local model để né quota.
- Bịa metric/parameter count khi evidence thiếu.
- Bulk gửi corpus/visuals ra provider khi user chưa approve privacy.
- Tự động bulk summarize 94 conversation lịch sử chỉ để lấp L2; L0 đã giữ source
  và lazy wake tránh tốn quota vô ích.

## Follow-up sau generalized scope pass — 2026-08-09

- [ ] Chạy live multi-turn E2E bằng `cx/gpt-5.6-sol` khi 9router online: ít nhất
  A/B compare → casual detour → `bảng kết quả của chúng` → explicit C correction.
  Local live probe hiện đã chứng minh router/rewrite ~2 ms và retrieval cover đủ
  MSF-SER + wav2small, nhưng generation cố ý fail-closed vì gateway đang offline.
- [ ] Xây held-out conversational eval lớn hơn corpus-pair metamorphic tests:
  unknown aliases, collection exclusions, stale referents, table/figure asks và
  qualitative/quantitative follow-ups. 21/21 và 210/210 là invariance coverage,
  không được quảng bá thành end-to-end answer quality.
- [ ] CMDM Table 2 hiện bị Docling gộp vật lý vào Markdown của Table 1 và không
  có standalone artifact. Runtime cố ý trả không có bảng thay vì clone/suy theo
  vị trí. Muốn phục hồi phải sửa generic extraction/splitting rồi re-index, kèm
  provenance và regression trên nhiều PDF; không hardcode riêng CMDM/Table 2.
- Debug trace trong `backend/.env` hiện bật cho local debugging, nhưng mỗi run vẫn
  phải opt in từ UI và endpoint chỉ phục vụ loopback. Tắt
  `AGENT_DEBUG_TRACE_ENABLED` khi không cần xem raw redacted milestones.
- Long-form cross-paper answers vẫn bị chi phối bởi provider generation latency;
  scope/rewrite/router tối ưu không thể biến một Sol generation dài thành fast
  path. `must_cover_all` hiện buffer toàn draft trước khi reveal để không phát
  answer thiếu paper rồi mới phát hiện; đây là correctness-preserving TTFT cost.
  Không mở direct-render sang compare/analysis chỉ để giảm thời gian.

## Evidence-card rollout gate — updated 2026-08-14

- [x] User đã authorize build/persist card trong isolated 21-paper clone và gửi
  excerpt qua local 9router tới đúng `cx/gpt-5.6-sol`.
- [x] Backfill hoàn tất 21/21: 184 valid item/230 refs/0 invalid; mọi job complete,
  không fallback, không mutate production. Card đều `partial`, nên missing facet
  vẫn đi raw fallback thay vì bịa coverage.
- [x] Full dev-20 pass 20/20 sau fix; cache-cleared targeted card-path pass 5/5
  với 3 card coverage event. Production flags/DB chưa được promote.
- [x] User đã đổi policy ngày 2026-08-14 và cho chạy toàn bộ held-out-60 để tối
  ưu sản phẩm cá nhân. Baseline đã chạy 60/60 và bộ này được retire thành
  regression-v1; không còn dùng để tuyên bố blind release score.
- [x] Đã sửa sáu root-family tổng quát trong regression-v1 report và thêm
  positive/negative/metamorphic variants; backend 622/622. Không hardcode
  câu/paper/table từ 60 case.
- [ ] Khi 9router online, rerun đủ regression-v1 60-turn đúng
  `cx/gpt-5.6-sol`, so sánh per-family với baseline 37/60 (adjusted floor 40/60),
  kiểm zero fallback/transport và persist report mới. Không sửa tiếp theo output
  model trước khi phân loại retrieval/scope/evidence/evaluator rõ ràng.
- [ ] Nếu cần blind release score về sau, tạo shadow/held-out v2 từ unseen
  PDFs/public samples sau khi regression-v1 ổn; không tái gọi v1 là held-out.
- [ ] Sau release gate, rollout production theo staged backup/health/rollback;
  authorization hiện tại chỉ cho isolated clone, không tự cho phép persist sang
  production DB.
- [ ] Card freshness phụ thuộc content/parser/schema/prompt fingerprints; corpus
  re-index tạo pending rebuild nhưng không tự gửi provider. Monitor pending,
  stale, failed counts trong `/health` và evidence-card status API.

## Public benchmark rollout — 2026-08-12

- [ ] Chạy đủ 50 WildBench routing negatives chỉ khi chấp nhận chi phí 50 router
  + answer calls; smoke hiện mới 3 case ban đầu và 1 targeted rerun. Đây không
  phải quality score theo official WildBench checklist.
- [x] Đã dựng isolated FTS5/BM25 index/runner cho MTRAG/MTRAG-UN, chạy full
  777+332 retrieval cases và 12-turn official-reference Sol smoke; không chèn
  corpus benchmark vào SQLite/LanceDB/LightRAG production.
- [x] Đã thêm isolated EmbeddingGemma/hybrid candidate-rerank diagnostic trên
  hai slice 40 câu; Human rewrite tăng mạnh nhưng MTRAG-UN questions không tăng
  MRR, nên không promote toàn cục.
- [x] Đã A/B actual context-aware Aya rewrite trên 40 case và giữ per-case audit;
  không kết luận từ aggregate alone.
- [~] Đã thêm isolated MTRAG end-to-end generation adapter, per-case
  retrieval/token-F1/ROUGE-L/abstention/numeric/latency audit và semantic
  sufficiency guarded mode. Chưa chạy official LLM judges/RAGAS/RAD-Bench và
  không promote từ model-authored target/token-F1. Cần internal dev gate +
  larger conversation-group-held-out slice trước khi công bố quality score.
- [ ] Original+Aya-rewrite RRF đã thắng MRR trên public 40-case, nhưng chưa bật
  runtime. A/B trên internal dev-20/regression-v1 với local raw-retrieval branch,
  cache key, table/figure/focus validation và latency. Chỉ
  parallelize hai local searches khi `rewrite_used=true`; không tự nhân đôi
  LightRAG/card/exact-artifact path.
- [ ] Dựng isolated index/runner cho các split ChatRAG-Bench; giữ cùng isolation
  guard như MTRAG.
- [x] Đã dựng isolated three-mode runner cho SPIQA Test C và MMLongBench-Doc,
  chạy smoke 1+2 case, official MMLong scoring và diagnostic-only SPIQA F1.
- [ ] Mở rộng thành stratified public dev set đủ lớn theo text/table/chart/image,
  single/multi-page và answer type; không dùng 1+2 smoke để tuyên bố production.
- [ ] Tích hợp official SPIQA L3 semantic scorer/environment trước khi công bố
  SPIQA quality number. Token-F1/ROUGE-L hiện chỉ là diagnostic.
- [ ] Tối ưu/background cold page OCR cho image-only PDF và UI indexing progress:
  smoke cold ingest khoảng 455 s dù cached query chỉ 4.588 s.
- [ ] Full-document baseline hiện dùng extracted text + bounded page images vì
  endpoint 9router không nhận native PDF object. Muốn so công bằng kiểu upload
  PDF cần native-PDF provider path hoặc full-page coverage có budget rõ.
- [ ] Tích hợp official checklist/scorer cho WildBench và official metrics cho
  từng benchmark trước khi công bố quality number. MultiChallenge chỉ research/
  eval đến khi upstream làm rõ license.

### Findings mới từ MTRAG E2E — 2026-08-13

- [ ] Retrieval vẫn là bottleneck trên live stratified supported slice: dual
  original+Aya-rewrite RRF Hit@5 chỉ `0.50`. Candidate EmbeddingGemma từng tăng
  Human official-rewrite Hit@5 `0.325 → 0.70`, nhưng MTRAG-UN lại không tăng;
  phải A/B đúng Aya standalone query trên split theo conversation trước khi
  promote. Không full re-index MTRAG thành paper.
- [ ] Semantic evidence sufficiency giải quyết unsupported inference nhưng thêm
  `4.59–6.40s` P50 bằng Sol. Chỉ cân nhắc production dưới feature flag/guarded
  policy sau khi có larger held-out false-accept/false-reject và latency budget;
  không bật default hiện tại.
- [x] Fresh paper-agent dev/smoke HTTP E2E đã chạy sau explicit authorization:
  20/20 pass trên clone, Sol/no fallback, production corpus không mutate.
