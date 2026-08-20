# 04 — Already done (updated 2026-08-20)

## Codex developer context

- [x] Graphify runtime `0.9.37` + MCP extra cài bằng `uv tool`; graph hiện có:
  3.167 nodes, 8.896 edges, 135 communities, zero model calls
- [x] Project skill cài ở `.agents/skills/graphify`; project MCP ở
  `.codex/config.toml`; `codex mcp list`, MCP initialize/list_tools,
  `graph_stats` và `query_graph` đều pass
- [x] `AGENTS.md` áp dụng graph-first budget 1.200 token và chỉ mở source đầy đủ
  khi cần verify/debug/edit/security hoặc graph không đủ tin cậy
- [x] Source-code Obsidian export: 3.273 generated notes + `graph.canvas`; có HTML graph,
  `GRAPH_REPORT.md` và `.graphifyignore` loại corpus/runtime/reference repo
- [x] LightRAG Obsidian export riêng: 8.559 entity notes, 37.016 wikilinks,
  11 type MOCs, canvas 60 hub/214 edge; reusable exporter ở
  `scripts/export_lightrag_obsidian.py`

## LLM / product

- [x] Answer model và intent router dùng **9router**, mặc định
  `cx/gpt-5.6-sol`; UI hiện thêm `cx/gpt-5.6-terra` và `cx/gpt-5.6-luna`
- [x] Figure VLM mặc định dùng OpenAI-compatible 9router `cx/gpt-5.6-sol`
- [x] OpenAI-compatible vision gửi crop + full page với paper/page/section/ref/table context
- [x] Figure provenance lưu provider/model; health API báo vision configuration
- [x] Streaming client đọc error body đúng cách; không còn `ResponseNotRead` che lỗi 9router

## RAG / index / visual quality

- [x] Chunking v3 token-aware structural parent–child: child 384/overlap 48,
  parent 1.536; deterministic offline tokenizer và stable UUID5 IDs
- [x] Contextual embedding prefix từ DocumentCard/filename/page/section; raw
  chunk content vẫn sạch, future LightRAG ingest dedupe theo parent passage
- [x] Parent-aware small-to-big expansion thay cho neighbor-only context
- [x] Safe text-only migration 21/21 documents; không gọi VLM/9router, không
  thay table/figure IDs; SQLite và LanceDB v2 đều có rollback backup
- [x] LanceDB v3 live: 20 cards + 1.054 text + 60 table + 59 figure vectors
- [x] Cross-encoder local backend/config/downloader fail-closed; không implicit
  network/model fallback
- [x] Controlled compare decomposition chạy parallel per focused document;
  retrieval cache bump `v12`
- [x] Cosine reranker cũ tắt mặc định sau A/B thật cho thấy regression
- [x] Collection/focus scope áp dụng cho mọi retrieval engine
- [x] Merge LightRAG + legacy/Lance; scoped LightRAG rỗng fallback an toàn
- [x] LightRAG provenance ambiguous không bị relabel theo focus
- [x] Last-source cache validate collection/mode/index fingerprint
- [x] Table intent không bị architecture dilution; focused table retrieval reserve `min_tables=1`, cache `v11` reject entry thiếu table
- [x] Lance full prune và replace complete document slice
- [x] Delete document đồng bộ LightRAG/Lance trước canonical store
- [x] Stable deterministic document/figure/table IDs qua re-index
- [x] Extraction V9: geometry grouping, caption binding, multi-panel composite, page/paper context
- [x] Visual quality gate loại logo/branding/publisher marks/page fallback
- [x] Figure ranking theo query intent + figure type; không hardcode paper → Figure N
- [x] Best/single-figure request trả đúng top 1; từ “mô hình” đơn thuần không bị coi là visual request
- [x] Vision empty completion gây lỗi thay vì ghi provenance GPT-5.5 giả
- [x] Full-corpus GPT-5.6 Sol visual enrichment: 60 enriched, 76 quality-skipped, 0 failed
- [x] Live figure quality sau enrich: 59 accepted, 22 needs-review, 55 rejected
- [x] Live index snapshot: 20 document cards + 1,146 text + 60 table + 60 figure chunks; SQLite có 21 documents
- [x] LightRAG stale-prune safety gate khi canonical docs chưa processed
- [x] LightRAG ingest preflight 9router/model trước graph mutation; không retry/fallback provider
- [x] LightRAG full-sync cô lập từng canonical paper khỏi failed backlog, bắt false-success `ainsert` và abort lỗi đầu tiên
- [x] LightRAG native stream collector tránh 9router non-stream 502; timeout/connection retry đúng một lần cùng model, quota không retry
- [x] LightRAG bắt cả raw `httpx.ReadTimeout`/network error khi consume stream;
  extract-only watchdog 660s chứa trọn same-Sol retry, foreground query giữ 240s
- [x] LightRAG chunking 600/80 và `max_async=1` để paper/table dày vẫn resume ổn định qua cache
- [x] Canonical LightRAG re-ingest hoàn tất 21/21: 21 processed, 0 failed,
  graph 8.559 nodes / 14.173 edges; prune 7 duplicate tombstones
- [x] Durable LightRAG `source_id -> canonical parent` provenance có hash
  verification, exact/normalized/fuzzy-safe alignment, overlap ranking, stale
  invalidation và delete cascade
- [x] Provider-free live provenance backfill 21/21: 335/382 chunks, 931 parent
  links, 47 ambiguous rejected; SQLite backup + integrity/foreign-key checks pass
- [x] Graph-as-navigation bridge: entity/relation prose không còn là answer
  evidence; parent passages giữ page/section/heading metadata và canonical scope
- [x] Generic structural/high-cardinality hub suppression có cardinality/reason
  diagnostics; per-document quota + result cap chống một paper crowd-out
- [x] Deterministic `RETRIEVAL_ENGINE=auto`: focused/table dùng local hybrid,
  discovery/cross-document dùng LightRAG; mọi path báo engine + policy reason
- [x] Evidence-conditioned hop 2 không dùng planner LLM: tối đa 2 hop, 3 scoped
  subquery chạy parallel, per-branch timeout, canonical dedupe/no-progress discard
- [x] Retrieval cache `v14` fingerprint graph-parent provenance và adaptive policy;
  rebuild timestamp-only không gây churn nhưng mapping/content/overlap change invalidate

## Grounding / evidence

- [x] Retrieval evidence validator chặn cross-focus và mixed-document evidence
- [x] Quantitative claim guard kiểm metric, retry một lần, rồi fallback không số
- [x] Parameter-count validation hỗ trợ `k/m/b` magnitude
- [x] Numeric parser hỗ trợ row-oriented/repeated-header tables, table model ownership, spaced/jammed decimals và prose table cells
- [x] Fail-closed unparsed signals + bounded internal attempt diagnostics; correction prompt dùng layout metric rõ ràng
- [x] Không emit answer số chưa kiểm ra UI trước khi validation hoàn tất

## Memory / focus

- [x] L1 active paper + recent document thread, hỗ trợ natural resume sau casual detour
- [x] Không commit L1 từ failed/invalid run; chỉ sau grounded generation thành công
- [x] L2 durable full-turn queue + dirty/summary cursors; pending turns luôn vào prompt khi stable summary chưa kịp fold
- [x] Per-conversation coalescing, một global fold, 12s idle debounce và foreground answer priority
- [x] Restart/interrupted-job/completed-run recovery; provider lỗi retry cùng model, không heuristic/local summary
- [x] Direct `/chat` crash-gap recovery; L3 durable outbox + replay-safe receipts + exact source cursor
- [x] Foreground/shutdown cancellation giữ provider backoff; health expose L3 retry/error
- [x] L0 FTS + complete-episode historical retrieval; cross-thread chỉ khi có explicit history cue
- [x] L3 semantic/episodic/procedural versioned memory với provenance, validity, supersede/forget và idempotent ops
- [x] Legacy memory migration dùng dormant jobs, tránh bulk 94 GPT summary calls khi startup
- [x] Conversation state repair stale re-index UUID theo filename/topic duy nhất
- [x] Canonical filename được phục hồi khi resume history cũ

## Desktop

- [x] Conversation sidebar/list và resume thread sau refresh
- [x] SSE abort/stop và run status handling
- [x] Health states + indexing controls; bỏ hardcoded library path
- [x] Persist/restore retrieval sources trong history
- [x] History projection giữ `table_id`, `table_index` và `figure_type`
- [x] UI không tự suy requested Figure N; tin curated backend order
- [x] Production build green

## Tests / tools

- [x] Backend full suite gần nhất: 364 passed; compileall green
- [x] RAG audit 2026-07-31: 344 passed/17.90s, SQLite integrity/FK clean,
  335 mapped LightRAG chunks revalidated with 0 stale, live scoped ASPIRE
  figure/table + explicit-empty-scope smoke pass
- [x] Thêm title-ablated multilingual diagnostic 15 intent x EN/VI và report
  trung thực: English Hit@3 0.933, Vietnamese Hit@3 0.333; không gắn nhãn
  held-out/production
- [x] Retrieval v3 production eval (8 labels): Hit@3/5/6 100%, MRR@6 0.9375,
  nDCG@6 0.9539, P50 149.99 ms, P95 244.0 ms
- [x] Evaluation harness có Hit@k, MRR, nDCG, P50/P95/P99, explicit
  rerank/cross-encoder/LanceDB staging switches và stale-label repair
- [x] Focused LightRAG ingest + model-policy sau batch hardening: 32 passed
- [x] Regression tests cho focus/memory, LightRAG scope/prune, figure ranking/VLM, evidence guard và OpenAI stream error
- [x] Regression graph/multi-hop cho `<SEP>` provenance, stale/raw fail-closed,
  parent quota, structured facets, scoped parallel branches, timeout/provider
  propagation, SSE payload, no-progress và cache invalidation
- [x] Regression burst/coalescing/restart/quota/no-fallback/L0 cross-thread/L3 temporal/concurrency
- [x] E2E harness hội thoại nhiều turn: `scripts/e2e_agent_conversation.py`

## Model policy / research

- [x] Unknown provider fail-closed thay vì âm thầm tạo Ollama client
- [x] LightRAG model fallback chain bị vô hiệu; startup chỉ chấp nhận bộ
  `cx/gpt-5.6-{sol,terra,luna}` qua 9router, mặc định Sol
- [x] Answer stream/nonstream và vision từ chối gateway-reported model mismatch
- [x] Chấp nhận đúng de-namespaced alias `cx/gpt-5.5` → `gpt-5.5`, vẫn chặn model/version khác
- [x] Chuyển model policy/UI sang ba GPT-5.6 preset; không tự fallback giữa Sol/Terra/Luna
- [x] Malformed/empty gateway response fail-closed; LightRAG quota nổi ngay, không retry ba lần
- [x] Bỏ UI Qwen/Ollama preset không được backend hỗ trợ; custom model ghi rõ chạy qua 9router
- [x] Router quota/provider error dừng run và surface lỗi thay vì đoán route rồi đi tiếp
- [x] Tải 9 paper memory chính thức + design notes vào `docs/research/agent-memory/`

## Trạng thái E2E

- [x] 2026-08-03: sửa cross-paper baseline-name trap từ live ASPIRE → MSF-SER:
  explicit `bài/paper X`/`Table N X` thắng sticky history; multi-entity correction
  không mất acronym thứ hai; resolver dùng riêng current target; direct renderer
  từ chối foreign `document_id`.
- [x] 2026-08-03: generic result-table retrieval bỏ anchor `ablation`, ưu tiên
  main experimental comparison/performance. Fast-path chỉ nhận canonical caption
  main-result và loại ablation/distribution/statistics. Live Sol chọn đúng Table 3
  MSF-SER, cache-hit wall 4,4 s; backend **373 passed**.
- [x] 2026-08-03: generalize paper targeting ra toàn corpus: 21/21 exact filename,
  title dài, acronym/model alias, correction, compare và negative baseline trap.
  Exact Table N luôn inject canonical `document_tables` record theo resolved paper,
  nên không còn phụ thuộc vector top-k có rank table hay chỉ rank text quanh bảng.
  Live Sol pass ASPIRE → ViSEC/Pitch-fusion và LPMN title dài → correction KST;
  backend **388 passed**.
- [x] 2026-08-03: sửa UI nhận SSE delta nhưng chỉ paint một lần. Bỏ
  `useDeferredValue`/double-RAF coalescing khi streaming; visible SSE output được
  frame-paced và delta lớn chia thành slice 64 code point. Giữ fail-closed
  quantitative validation và gắn mode rõ `token_stream`/`validated_blocks`/
  `buffered_validation`/`validated_reveal`. Desktop build green, backend 388 pass.
- [x] 2026-08-03: structured paper fast path bỏ router + generation call cho
  explicit/focused table-result-dataset lookup. Generic multi-table selector chọn
  unique performance coverage hoặc prior-method comparison theo intent, tie vẫn
  fail-closed. Live ASPIRE dataset + results giảm 65,45 s → **2,397 s** cold;
  router 0 ms, generation 8,8 ms deterministic, backend **394 passed**.

- [x] 2026-08-01: thêm task-prefix đúng vai trò cho embedding, sửa document card
  embedding, tạo isolated staging index/eval; A/B Nomic raw/prefixed/BGE-M3 mà
  không chạm production LanceDB.
- [x] 2026-08-01: benchmark Qwen3-Embedding-0.6B đúng query instruction; harden
  Ollama embedding bằng ordered split cho batch 400 và retry transient EOF.
- [x] 2026-08-01: benchmark EmbeddingGemma-300M và Snowflake Arctic Embed 2 trên
  cùng staging/eval; EmbeddingGemma đạt 30/30 recall@6 và đang dẫn shortlist.
- [x] 2026-08-01: thêm isolated Hugging Face/Sentence Transformers provider với
  prefix, Jina task adapter, revision pin và warm latency; chấm official GTE.
- [x] 2026-08-02: kiểm lại Jina v4/v5 từ official catalog, bỏ full-index v3;
  benchmark pinned Jina v5-small retrieval 20/20 docs, 30 câu Anh/Việt đạt
  Hit@3/6 `0.967/1.0`, VI Hit@3/6 `1.0/1.0`, không chạm production.
- [x] 2026-08-02: promote Ollama EmbeddingGemma-300M vào production với
  asymmetric query/document prefix; atomic-swap LanceDB và giữ backup Nomic.
- [x] 2026-08-02: rebuild LightRAG VDB từ graph/KV gốc bằng app embedding
  adapter, đủ 8.559 entity / 14.173 relation / 382 chunk, missing `0/0`; direct
  entity/chunk query pass và không gọi 9router.
- [x] 2026-08-02: production re-eval EmbeddingGemma: dev-8 Hit@3 `1.0`, MRR
  `0.9375`; title-ablated 30 câu Hit@3/6 `0.933/1.0`, MRR `0.8444`, P50
  `198.44ms`.
- [x] 2026-08-02: sửa ViSEC Table 2 false rejection: hyphenated model subject,
  sticky “thế … pitch fusion”, và canonical table inventory cho count/list.
  Payload cũ validate 12/12; live Sol two-turn trả bảng, không fallback.
- [x] 2026-08-02: latency hardening theo audit nhưng giữ fail-closed grounding:
  grounded planner deterministic; scoped comparison branch local hybrid; exact
  canonical table direct-render; sanitize-before-retry; validated block streaming;
  validator hiểu slash metrics/model version; trace có router/first validated token.
- [x] 2026-08-02: cross-paper prompt round-robin + trusted coverage metadata,
  tránh source lớn của paper đầu crowd-out paper sau. Live ViSEC–ASPIRE không còn
  false “ASPIRE chưa có source”.
- [x] 2026-08-02 live Sol benchmark trên isolated fresh backend: ViSEC Table 2
  2,37 s/attempts 0; comparison 24,43 s wall, first validated generation block
  4,74 s, một generation; trace cũ cùng comparison là 76,48 s.

- [x] Strict 7-turn E2E pass: conversation `b08d1f90-c050-4d74-8c2f-b04125ba3796`, 7/7 completed, 14 messages, no fallback, one Figure 1, table identity persisted, Acc/F1/CCC claims checked.
- [x] GPT-5.6 Sol strict 7-turn E2E pass:
  `data/logs/e2e-gpt-5.6-sol-20260723-final.json`, conversation
  `f12c8042-927f-41ce-9c10-0e53bccb3549`, 7/7 completed, figure HTTP 200.
- [x] DB audit: L1 ASPIRE, recent WhiSER, memory/summary revision 7, recent beats 5/6/7; figure image HTTP 200.
- [x] Canonical LightRAG re-ingest hoàn tất ngày 2026-07-29: 21/21 processed,
  `unready=[]`, graph 8.559 nodes / 14.173 edges; query smoke WHiSER trả đúng
  canonical paper ở top result.
- [x] 2026-08-09: thay các paper-name heuristic phân tán bằng catalog-driven
  `DocumentScopeResolution` trước router; pass 21/21 biến thể tên và 210/210 cặp
  corpus, ambiguous/weak aliases fail-closed.
- [x] 2026-08-09: thêm durable ordered multi-document referent vào L1, hỗ trợ
  `chúng/cả hai/both/them` qua casual detour; explicit current target vẫn thắng.
- [x] 2026-08-09: carry `must_cover_all` qua decomposition, retrieval validation,
  retry và L1 commit; evidence thiếu một paper không còn được coi hợp lệ.
- [x] 2026-08-09: generalized hierarchical numeric-table validator, multi-paper
  direct-table guard, và exact Table N canonical reservation. Live MSF-SER Table
  3 direct-render đúng trong 12.5 ms, không gọi 9router.
- [x] 2026-08-09: opt-in redacted debug trace riêng theo run (64 KiB, 72 giờ,
  newest 25, loopback-only), có UI Debug tab; normal run payload/SSE giữ nguyên.
- [x] 2026-08-09: scope grammar độc lập surface phrase (`A và/với/and/vs/+/, B`,
  quantified pair), correction span roles, plural referents và shared comparison
  predicate; coverage cardinality không ép mọi A+B thành `compare`.
- [x] 2026-08-09: answer-level per-document coverage + bounded mandatory retry;
  generic `Ours/Baseline`/shared metric không còn cover giả hai paper.
- [x] 2026-08-09: weak topical/year/page aliases bị loại nhưng identity aliases
  Pitch-fusion/FM-MOE/KS-Transformer/LPMN và full canonical stems vẫn hoạt động.
- [x] 2026-08-09: page-local unique table-caption reconciliation phục hồi CMDM
  Table 6 không hardcode; live canonical scan 60/60 render + validate. Table 2
  bị merge vật lý tiếp tục fail-closed.
- [x] 2026-08-09: debug trace hardening actual-peer gate, startup TTL/count purge,
  snake/kebab/camel-case secret redaction; backend **514 passed** và desktop
  production build passed.
- [x] 2026-08-09 independent metamorphic audit: core grammar 6.300/6.300,
  correction/replacement 11.340/11.340, additive 1.260/1.260, ambiguity
  252/252, comparison intent 4.200/4.200 và extract/list 2.940/2.940; zero
  residual trong các matrix đã yêu cầu.
- [x] 2026-08-12: implement canonical paper evidence-card schema/service/builder,
  provenance validation và resumable bounded-concurrency jobs; pin Sol và
  fail-closed khi quota/provider lỗi, không model fallback.
- [x] 2026-08-12: card-first multi-paper retrieval theo coverage matrix, batch
  load song song và raw fallback chỉ cho missing facets; exact-artifact path giữ.
- [x] 2026-08-12: progressive per-paper validated streaming + desktop status,
  giữ mỗi paper trong evidence boundary riêng trước synthesis.
- [x] 2026-08-12: tạo dev-20 + sealed held-out-60 conversational eval, manifest
  checksum/corpus fingerprint, fixture tests và public HTTP/SSE evaluator.
- [x] 2026-08-12: backend **525 passed*1*, desktop production build passed;
  retrieval cache bumped `v18`.
- [x] 2026-08-12: tải/pin 7 nguồn public evaluation official (~1,9 GB), full
  checksum/count/ZIP validation pass; dựng catalog 15.892 case nhưng cách ly
  hoàn toàn khỏi production corpus.

- [x] 2026-08-13: thêm dataset-neutral external-passage contract và MTRAG Aya
  E2E runner: real QueryRewriteService → isolated FTS/RRF → real AgentService
  graph/stream → retrieval/generation/abstention/numeric/latency audit; qrels và
  targets chỉ dùng sau generation, production corpus không bị mutate.
- [x] 2026-08-13: chạy live 15-strata MTRAG bằng `cx/gpt-5.6-sol`; dual RRF
  Hit@5/10 `0.50/0.625`, MRR@10 `0.3646`; lưu per-case answer/evidence/latency
  dưới `data/retrieval_eval/public/results/`.
- [x] 2026-08-13: chứng minh BM25/overlap confidence không generalize bằng split
  theo conversation (held-out AUC `0.495`), nên không đưa threshold giả an toàn
  vào runtime.
- [x] 2026-08-13: thêm generalized semantic evidence-sufficiency service và
  guarded MTRAG A/B. Unsupported 4/4 abstain đúng; supported top-5 relevant không
  bị gán `insufficient`; ambiguity đi conditional/clarification thay vì hard
  reject. Giữ opt-in vì thêm P50 `4.59–6.40s` Sol latency.
- [x] 2026-08-13: fast-path pure social acknowledgements kể cả khi còn document
  focus; boundary run chat 3/3, routing ~`0.02ms`, total `1.6–2.2s`.
- [x] 2026-08-13: loopback sidecar URL normalize `localhost → 127.0.0.1` cho
  settings runtime để tránh macOS `::1`/IPv4 bind mismatch, không model/provider
  fallback. Backend **570 passed**, desktop build passed.
- [x] 2026-08-12: tạo 50-case WildBench real-user local-routing-negative suite;
  Sol smoke bắt một false local-RAG activation và generalized local-scope gate
  đã sửa/rerun pass, zero retrieved local document.
- [x] 2026-08-12: thêm public dataset adapters, schema guard, exact `--case-id`
  eval selection và regression tests; Antigravity synthetic drafts không còn
  được hiểu nhầm là benchmark official/runnable.
- [x] 2026-08-12: verification sau public-eval/local-scope patch: backend
  **535 passed**, desktop production build passed; full 50-case adversarial
  routing gate và targeted live Sol rerun đều pass.
- [x] 2026-08-13: dựng isolated immutable MTRAG FTS5/BM25 index 366.438 passage;
  chạy full 777 Human + 332 MTRAG-UN retrieval cases ở sáu query mode, lưu
  manifest/checksum/metrics mà không chạm production corpus.
- [x] 2026-08-13: chuẩn bị 12 official-reference multi-turn cases và chạy live
  `cx/gpt-5.6-sol` trên backend/data directory cách ly: 12/12 pass, zero local
  retrieval, 4/4 unsupported abstain, median TTFT 2.60 s.
- [x] 2026-08-13: sửa language fidelity tổng quát ở final prompt: explicit
  English/Vietnamese request thắng persona default; English prompt không còn
  nhúng Vietnamese pronoun/opening/style. Targeted live Sol regression pass.
- [x] 2026-08-13: verification sau MTRAG/language pass: backend **543 passed**,
  desktop production build passed.
- [x] 2026-08-13: thêm isolated EmbeddingGemma top-50 candidate-rerank runner,
  cache path guard và qrel-non-injection tests; cold/warm Human 40-case metrics
  tái lập và MTRAG-UN 40-case cross-check bắt được regression thay vì overclaim.
- [x] 2026-08-13: final verification sau candidate-rerank: backend **546 passed**;
  desktop production build vẫn green từ cùng checkpoint.
- [x] 2026-08-13: thêm resumable MTRAG adapter cho actual QueryRewriteService,
  pin Sol/no fallback và source-fingerprinted cache; 40-case equal-domain eval
  cho Aya rewrite Hit@10 `0.600`, dual-query RRF MRR `0.402` so với raw `0.325`.
- [x] 2026-08-13: sửa generic “benchmark” bị paper-SER metric expansion ngoài
  authoritative document scope; regression test IBM Cloud + 87 focused scope/
  rewrite tests pass.
- [x] 2026-08-13: final backend verification sau rewrite/RRF pass:
  **550 passed**; desktop production build vẫn green trong cùng checkpoint.
- [x] 2026-08-14: user-authorized full agent dev-20 trên isolated 21-paper clone
  qua local 9router, đúng `cx/gpt-5.6-sol`, no fallback/no production mutation:
  **20/20 pass**, zero transport/provider/tool fallback; median total 10.812s,
  median client-first-delta 4.538s.
- [x] 2026-08-14: generalized evidence validator cho separator-equivalent
  scientific metrics (`CCC_A`, `macro_F1`), Markdown owner subheading,
  multi-metric comparison cell và unique owner-free measurement. Duplicate
  metric/value ở nhiều owner/document vẫn fail-closed. ASPIRE–ViSEC replay giảm
  33.026s/two attempts xuống 11.421s/one attempt; 17 claims supported.
- [x] 2026-08-14: final verification **594 backend tests passed** và desktop
  `tsc && vite build` passed. Final report:
  `data/retrieval_eval/internal/aya-agent-dev20-20260813/agent-dev20-sol-v21-validator-final.json`.
- [x] 2026-08-14: user-authorized evidence-card backfill trong isolated clone
  hoàn tất 21/21 bằng đúng `cx/gpt-5.6-sol`: 184 valid item, 230 canonical
  reference, 0 invalid, 21 durable job complete; production DB/index không đổi.
- [x] 2026-08-14: sửa streaming-JSON/retry/quote/summary validation của generic
  evidence-card builder và bounded task↔contribution facet companion. Fresh
  cache-cleared card-path regression pass 5/5; retrieval card khoảng 4–5 ms.
- [x] 2026-08-14: full agent dev-20 sau card/facet fix pass 20/20, exact Sol,
  zero fallback/transport error; median total 11.223 s, client-first-delta
  5.665 s. Sealed held-out-60 không chạy.
- [x] 2026-08-14: thêm isolated MMLongBench-Doc/SPIQA three-mode runner, gold
  non-injection, official MMLong scorer, multimodal artifact adapter và report
  limitations. Smoke 1 MMLong + 2 SPIQA hoàn tất mà không chạm production.
- [x] 2026-08-14: generic sparse/image-only PDF page-OCR fallback phục hồi live
  MMLong slide deck từ 0 lên 92 text chunks và evidence-page recall 1.0; cached
  Aya query 4.588 s. SPIQA visual adapter nâng diagnostic F1 0.536 → 0.763.
- [x] 2026-08-14: corrected SPIQA B/C per-question flattening; prepared catalog
  15.892 case và full revision/hash/count validation pass. Final verification
  **612 backend tests passed**, desktop production build passed.
- [x] 2026-08-14: theo explicit user policy change, mở/chạy 60-turn blind set
  một lần qua isolated HTTP/SSE backend và retire thành regression-v1. Exact Sol,
  zero fallback/transport error; raw 37/60, evaluator-adjusted floor 40/60 sau
  ba abstention false negative. Root-cause report đã persist; chưa tune/sửa theo
  case trong baseline run.
- [x] 2026-08-20: sửa sáu regression-v1 root families bằng resolver/facet/
  evidence/evaluator primitives dùng chung; không hardcode paper, table number
  hay câu benchmark. Partial-title và mixed conversational scope có negative +
  metamorphic coverage; grouped metrics và per-document insufficiency fail-closed.
- [x] 2026-08-20: verification sau generalized fixes: backend **622 passed**,
  desktop `tsc && vite build` passed. Live regression-v1 Sol rerun chưa chạy khi
  gateway `:20128` offline; không dùng fallback model.
