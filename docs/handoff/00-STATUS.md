# 00 — Status snapshot (2026-08-20)

## Product

**My Agent** — desktop AI assistant (persona **Aya**), local-first:

- UI: `apps/desktop` (React + Vite + Tauri)
- API: `backend` (FastAPI `:7777`)
- Answer + intent router: **9router** `:20128` → mặc định `cx/gpt-5.6-sol`;
  UI hiện thêm `cx/gpt-5.6-terra` và `cx/gpt-5.6-luna`
- Figure VLM lúc index/enrich: **9router** → mặc định `cx/gpt-5.6-sol`
- Ollama: chỉ dùng embedding `embeddinggemma:300m` trong cấu hình hiện tại
- RAG: LightRAG + LanceDB hybrid/text/table/figure
- Chat history/state: SQLite (`data/sqlite/app.db`)

## Trạng thái hiện tại

| Khối | Trạng thái |
|------|------------|
| Chat UI + agent stream SSE | OK; có stop/abort, resume conversation và health/index controls |
| Intent router + answer LLM | GPT-5.6 Sol/Terra/Luna qua 9router; mặc định Sol, không tự fallback |
| Sticky paper focus | L1 active + recent document thread; explicit current-turn `bài/paper X` và `Table N X` thắng tên baseline xuất hiện trong history |
| Conversation memory | Durable/coalesced L2 + L3 outbox/receipts + pending-turn protection + L0 episode search; restart/quota/concurrency safe |
| Retrieval scope | Collection/focus áp dụng cho mọi engine; LightRAG provenance mơ hồ không được relabel bừa |
| Chunking/retrieval v3 | Token-aware structural parent–child + contextual embedding; hybrid RRF là fast path, compare query có per-paper decomposition |
| Graph + multi-hop retrieval | Graph chỉ điều hướng; answer evidence được ground về canonical parent; `auto` routing deterministic; hop 2 tối đa 3 scoped branch |
| Figures/tables | V9 extraction + quality gate; best/single figure trả đúng top 1; focused table ask reserve ít nhất một table source |
| Numeric evidence | Chặn cross-paper/mixed-doc; parser chịu Docling/Markdown, slash/underscore metrics, hierarchical/multi-metric cells và owner scope; unique owner-free measurement chỉ được nhận khi khớp đúng một owner/document, còn ambiguity vẫn fail-closed |
| RAG latency | Exact canonical table và một main-result table có caption semantic rõ render deterministic; grounded planner không gọi LLM; non-quantitative answer stream theo validated Markdown block |
| Desktop history | Persist/restore sources cùng `table_id`, `table_index`, `figure_type`; không tự suy Figure N |
| Codex code context | Graphify AST graph + project skill/MCP + Obsidian vault; graph-first, source-file second |

## Graphify / Obsidian cho Codex

- Graphify `0.9.37` runtime (`0.9.27` project skill) đã update local sau
  regression-v1 generalized fixes: **3.167 nodes, 8.896 edges, 135
  communities**; code AST
  update không gọi model/9router.
- Codex repo skill nằm ở canonical `.agents/skills/graphify/`; `.codex/config.toml`
  đăng ký Graphify MCP bằng absolute interpreter path và đã pass handshake/tool call.
- `AGENTS.md` bắt buộc graph-first với budget mặc định 1.200 token; chỉ đọc source
  đầy đủ khi verify/debug/edit/security hoặc graph thiếu/stale/ambiguous.
- Obsidian source-code vault ở `obsidian-vault/`: **3.273 generated notes** +
  `graph.canvas`;
  HTML graph ở `graphify-out/graph.html`, report ở
  `graphify-out/GRAPH_REPORT.md`.
- LightRAG document graph được export riêng ở `obsidian-lightrag-vault/`:
  8.559 entity notes, 37.016 validated wikilinks, 11 type MOCs và canvas 60
  hub/214 edge. Rebuild bằng `scripts/export_lightrag_obsidian.py`; không trộn
  với source-code vault.
- Exclude khỏi code graph: runtime/index corpus `data/`, `pdf/`, generated output,
  binaries và `NOT_OUR_RAG_JUST_REF/`. Benchmark Graphify ước tính giảm context
  trung bình **6,2×** so với đọc corpus thô.

## Index/corpus thực tế

- SQLite hiện có 21 documents (17 PDF theo document type); artifact version: 19 V9, 1 V7, 1 chưa có version.
- Chunking v3 đã migrate 21/21 documents: **1.055 child text chunks**, mỗi
  child có stable parent link, tối đa 384 lexical tokens; 1.146 legacy vector
  chunks giảm còn **1.054 live vectors** vì document demo không có card/vector.
- LanceDB live snapshot: **20 document cards + 1.054 text + 60 table + 59
  figure vectors**; SQLite vẫn giữ đủ 60 table records và 136 extracted figure
  records. Table/figure IDs và VLM artifacts không bị rebuild trong migration.
- LanceDB đã re-embed toàn bộ bằng EmbeddingGemma-300M (768d) ngày 2026-08-02;
  backup vector space Nomic ở `data/lancedb-nomic-pre-embeddinggemma-20260802/`.
- Context embedding prepend filename/title/page/section và paper summary; retrieval
  child nhỏ rồi expand parent passage tối đa 1.536 lexical tokens.
- Backup rollback: `data/sqlite/app.db.pre-chunk-v3-20260730-232109.bak` và
  `data/lancedb-v2-pre-chunk-v3-20260730/`.
- Corpus hiện có 136 figures và 60 tables; figure quality: **59 accepted, 22 needs-review, 55 rejected**.
- Stable figure IDs đã kiểm: **133/133** giữ nguyên qua re-index V9.
- Full-corpus GPT-5.6 Sol figure enrichment đã chạy xong: 136 total, **60
  enriched**, 76 quality-skipped, 0 failed; 60 assets có provenance
  `9router/cx/gpt-5.6-sol`. Một crop panel thiếu ngữ cảnh đã được hạ từ
  accepted xuống needs-review thay vì cố promote.
- CMDM vẫn loại publisher marks/screenshots và giữ 7 figure nội dung; ASPIRE 5 figure hợp lệ; WhiSER 3 figure hợp lệ.

## LightRAG migration

- Bridge hiện map an toàn path/chunk cũ về canonical document ID; scoped query đã hoạt động.
- SQLite có durable many-to-many
  `lightrag_chunk_parent_provenance`: `source_id` LightRAG được batch-resolve về
  `(document_id, parent_chunk_id)`, kiểm hash lại lúc đọc và fail-closed nếu stale.
- Provider-free live backfill ngày `2026-07-31`: **21/21 documents**, **335/382**
  LightRAG chunks map an toàn thành **931 parent links**; 47 chunk mơ hồ bị loại.
  Có 709 exact/normalized links với `overlap_chars > 0`; fuzzy top-1 giữ overlap 0.
- Entity/relation descriptions chỉ là navigation metadata. Canonical parent text
  mới vào answer context; generic Figure/Table/Section, logo/journal-like hub và
  short high-cardinality node bị suppress với diagnostics.
- Canonical re-ingest hoàn tất ngày `2026-07-29`: doc status chỉ còn
  **21 `processed` / 0 `failed` / 0 pending**, đúng **21/21 canonical**.
- Lượt resume cuối insert 6 tài liệu còn lại, skip 15 tài liệu đã processed và
  prune 7 duplicate tombstones; `unready_document_ids=[]`.
- LightRAG graph đã persist **8.559 nodes / 14.173 edges**.
- Ba LightRAG VDB đã rebuild từ graph/KV gốc bằng EmbeddingGemma: **8.559
  entity / 14.173 relation / 382 chunk**, consistency thiếu `0/0`. Backup Nomic
  ở `data/lightrag-nomic-pre-embeddinggemma-20260802/`; rebuild không gọi LLM.
- Stale prune đã có safety gate: không xóa graph cũ nếu canonical index rỗng hoặc canonical document chưa `processed`.
- User đã approve gửi toàn corpus text/visual qua 9router. Toàn bộ canonical
  extraction/merge chạy bằng `cx/gpt-5.6-sol`; không dùng Terra/Luna/Ollama fallback.
- Batch ingest giờ preflight gateway/model trước mutation, dọn canonical queue
  bị ngắt để upstream không tự resume cả backlog, xác minh status thật sau
  `ainsert`, và dừng tại runtime/provider error đầu tiên.
- LightRAG adapter dùng native `stream=True` rồi collect về text để tránh lỗi
  9router convert stream→JSON; OpenAI SDK `max_retries=0`, chỉ retry đúng một
  lần cùng Sol cho OpenAI/httpx timeout hoặc connection error, không retry
  429/status/mismatch. Extraction pin `max_async=1`, chunk `600`, overlap `80`.
- Extract role dùng timeout 660s (LightRAG worker watchdog 1.320s) để chứa trọn
  provider timeout 300s + một same-model retry; foreground query vẫn giữ 240s.

## Memory migration + hardening (2026-07-19 → 2026-07-23)

- SQLite thêm durable turn/job cursors, FTS toàn L0 và typed/versioned L3.
- Prompt ghép stable summary + pending full turns + relevant historical episodes
  + relevant L3 + raw recent; không còn phụ thuộc ba beat để tránh mất context.
- Production L2 debounce 12 giây, một fold toàn process, global foreground
  priority; provider lỗi giữ pending và retry đúng model đã chọn, không đổi model.
- L2 summary commit và validated L3 ops dùng durable outbox; receipt chống replay
  `forget` cũ xóa version mới. Direct `/chat` crash gap cũng được startup recovery.
- Live DB: 597 messages = 597 FTS rows; 290 durable turns; 20 jobs idle; 101
  legacy jobs dormant (198 unfurled turns), nên startup không bulk GPT.
- L3 hiện có 3 validated items và 3 delivered outbox receipts; operations lưu
  provenance, validity và supersede history.
- Backup trước migration: `data/sqlite/app.db.pre-memory-v2-20260719-1037.bak`.
- Backup trước hardening/live recovery:
  `data/sqlite/app.db.pre-memory-v3-20260723-1036.bak`.

## Paper-target/table regression fix (2026-08-03)

- Root cause từ live run: tên model/paper mới đã xuất hiện như baseline trong bảng
  của active paper bị heuristic coi là same-paper follow-up. Ví dụ `MSF-SER`
  trong Table 2 ASPIRE làm câu `bài MSF-SER` giữ sai focus ASPIRE rồi reuse cache
  đúng key của scope sai.
- Current-turn explicit document target giờ thắng sticky/history; correction có
  nhiều tên nhưng không phải compare dùng target explicit cuối. Entity extraction
  không còn dừng sau alias corpus đầu tiên.
- Direct table renderer bắt buộc `document_id` khớp resolved focus. Generic
  `bảng result` thêm main-results/performance anchors, bỏ bias `ablation`; chỉ
  fast-path khi canonical caption là comparison/performance/main results và loại
  ablation/distribution/statistics.
- Live Sol sequence ASPIRE → MSF-SER đã resolve đúng `MSF-SER.pdf`. Generic
  MSF-SER result chọn Table 3 IEMOCAP; cache-hit retrieval 0 ms, render/validate
  19,9 ms, wall 4,4 s (router 4,25 s).
- Generalization audit cùng ngày không chỉ test MSF-SER: catalog resolver đã
  pass toàn bộ **21/21 filename**, mọi multi-word stem hiện có, alias
  LPMN/Pitch-fusion/KS-Transformer/Mamba, correction, compare, ambiguous alias
  fail-closed và baseline-name negative. Resolver dùng exact title/filename +
  document-card keyword, không hardcode paper → ID/figure/table.
- Live Sol E2E pass các chuỗi ASPIRE → Pitch-fusion/ViSEC, title dài LPMN →
  correction KST. Audit này bắt thêm lỗi đúng-paper nhưng thiếu canonical table
  do vector top-k; exact Table N giờ inject record từ `document_tables` theo
  resolved document ID + table number, không phụ thuộc embedding rank.
- Full backend suite sau generalization fix: **388 passed** (sẽ cập nhật nếu số
  test thay đổi trong lần verify cuối của cùng checkpoint).

## UI streaming/render fix (2026-08-03)

- Root cause của trace có nhiều `message.delta` nhưng UI hiện cả answer một lần:
  validated/direct-table routes phát nhiều delta lớn trong cùng browser turn;
  `requestAnimationFrame` accumulator + React `useDeferredValue` tiếp tục gộp
  chúng trước paint.
- SSE client giờ nhường một visible browser frame giữa output delta và chia riêng
  delta lớn thành presentation slices tối đa 64 Unicode code points. Token delta
  nhỏ giữ nguyên; trace/control events không bị throttle; tab hidden không bị RAF
  throttling.
- Chat state commit trực tiếp mỗi visible slice; bỏ double RAF accumulator và bỏ
  deferred Markdown trong lúc stream. Quantitative/table guard vẫn buffer trước
  validation, sau đó UI hiển thị dạng `validated_reveal`, không giả là raw token
  stream. Timing metadata phân biệt `token_stream`, `validated_blocks`,
  `buffered_validation`, `validated_reveal`.
- Verify: desktop `tsc && vite build` green; backend **388 passed**; live KST
  exact-table SSE có `streaming_mode=validated_reveal` và canonical output.

## Structured paper fast path (2026-08-03)

- Trace ASPIRE `dataset nào + bảng kết quả` cũ mất 65,45 s: router khoảng 5 s,
  retrieval chỉ 411 ms nhưng Sol generation/buffered quantitative validation mất
  56,70 s. Đây là orchestration waste, không phải retrieval bottleneck.
- Explicit paper/file + structured facet (table/result/dataset/metric/figure/
  architecture) và structured follow-up trên active paper giờ route deterministic
  `file_qa`; không gọi provider router. Cross-paper comparison vẫn giữ LLM router.
- Generic canonical result selector phân biệt performance/dataset coverage với
  prior-method comparison bằng caption semantics; unique best mới fast-path, tie
  vẫn fail-closed. Không hardcode ASPIRE/Table 1.
- Live nguyên văn query `Bài ASPIRE dùng dataset nào thế? Cho tôi bảng kết quả đi`:
  router **0 ms**, rewrite **4,3 ms**, cold retrieval **2,299 s**, direct validated
  render **8,8 ms**, wall **2,397 s**; chọn đúng Table 1 IEMOCAP + MSP-Podcast,
  không có answer-model generation. Backend full suite: **394 passed**.

## E2E / verification

### Authorized 21-paper agent dev regression (2026-08-14)

- User đã authorize gửi excerpt và persist evidence cards trong **isolated
  21-paper clone** qua local 9router, đúng `cx/gpt-5.6-sol`; không fallback model
  và không mutate production corpus.
- Evidence-card backfill hoàn tất **21/21**: 21 job `complete`, 21 card `partial`
  hợp lệ, 184/184 item valid với 230 canonical evidence reference, 0 invalid.
  `partial` là trạng thái terminal có coverage thật nhưng không đủ mọi facet;
  runtime chỉ fallback raw retrieval cho facet còn thiếu.
- Full public HTTP/SSE dev-20 sau fix đạt **20/20**, route `19 file_qa + 1 chat`,
  zero transport/provider error, zero tool/model fallback. Median total
  `11.223s`, median client-first-delta `5.665s`, max `18.725s`.
- Vì full rerun được retrieval cache phục vụ, card path được xác minh riêng sau
  khi xóa 140 cache row **chỉ trong clone**: 5/5 case pass, 3 card coverage event;
  WhiSER và MSF-SER dùng `paper_evidence_cards` với retrieval khoảng 4–5 ms.
  Generic facet selector giờ coi task/contribution là một cặp ngữ nghĩa bounded,
  nhưng dataset-only vẫn không bị kéo thêm facet ngoài yêu cầu.
- Seven quantitative cases đều validate; không case nào cần correction
  generation. ASPIRE–ViSEC lấy đúng hai canonical tables, 17/17 claims supported
  trong một attempt. Parser fix là schema-general: separator-equivalent metric
  aliases, local owner subheading, multi-metric cells và unique owner-free
  measurement; duplicate value across documents vẫn fail-closed.
- Artifacts chính:
  `agent-dev20-sol-cards-v3.json` và
  `agent-dev-facet-companion-cardpath-sol-v3.json` dưới clone evaluation.
- Tại checkpoint này held-out-60 vẫn chưa mở. Policy đã được user thay đổi ngày
  2026-08-14; xem mục “Held-out 60 retired into regression-v1” bên dưới.
- Verification cuối checkpoint: backend **612 passed**, desktop production
  build pass.

### Public multimodal three-mode smoke (2026-08-14)

- Đã implement isolated runner cho `gold_evidence`,
  `full_extracted_document` và `aya_pipeline`; gold answer chỉ xuất hiện sau
  generation để chấm điểm. Runner bắt buộc explicit public-upload approval và
  từ chối path ngoài `data/retrieval_eval/public/`.
- Một MMLongBench-Doc case qua official scorer: Gold `1.0`, extracted-full `0.0`,
  Aya `1.0`; Aya evidence-page recall `1.0`. PDF là slide deck image-only nên
  bắt được lỗi parser thật: 0 text chunk trước fix, 92 text chunk sau generic
  full-page RapidOCR fallback. Cold local OCR ingest khoảng 455 s; cached Aya
  query `4.588 s` total, `1.196 s` retrieval và `2.609 s` first token.
- Hai SPIQA Test-C case dùng **diagnostic** Token-F1/ROUGE-L, không gọi là
  official SPIQA score. Aya ban đầu `0.536`; sau khi index toàn bộ official
  figure/table artifacts độc lập gold referred IDs, Aya đạt `0.763` và retrieval
  đúng Table 3/page 6 ở case trước đó bị miss.
- Đây chỉ là dev/smoke cực nhỏ (1 MMLong + 2 SPIQA), đủ để bắt lỗi adapter,
  OCR, artifact retrieval và answer-format contract; **không** phải bằng chứng
  chất lượng production hay so sánh tổng quát với ChatGPT.
- Public catalog đã rebuild đúng **15.892 case / 6 runner mode**; full validation
  pass 17 checksum, 3 ZIP, toàn bộ revision/count. Production corpus không đổi;
  held-out-60 vẫn còn sealed trong chính smoke này và được mở ở baseline sau đó.

### Held-out 60 retired into regression-v1 (2026-08-14)

- User đã đổi mục tiêu từ giữ blind release score sang tối ưu sản phẩm cá nhân và
  cho phép mở/chạy toàn bộ 60 turn. Input hash khớp frozen manifest trước chạy.
- Isolated HTTP/SSE baseline hoàn tất 60/60 bằng đúng `cx/gpt-5.6-sol`, zero
  fallback/transport error. Raw strict result **37/60**; median total `10.05 s`,
  P95 `23.86 s`; median client-first-delta `4.46 s`, P95 `13.68 s`.
- Ba abstention case h15 là evaluator false negative rõ ràng; adjusted floor chỉ
  sau correction rubric là **40/60**, không phải semantic quality score.
- Weakness chính: multi-document strict chỉ **4/19**. Root families gồm 8 case
  partial/long title identity, 5 case anaphor+explicit scope collapse, 2 case
  recent ambiguous pair/correction, 3 facet `task`, 2 evidence validation và 3
  evaluator abstention/tool rubric.
- Bộ này từ nay là **regression-v1**, không còn là held-out mù. Muốn release score
  độc lập phải tạo shadow/held-out v2 từ PDF/public case chưa dùng.
- Báo cáo chi tiết:
  `data/retrieval_eval/internal/aya-agent-dev20-20260813/HELDOUT60_REGRESSION_V1_BASELINE_20260814.md`.

### Regression-v1 generalized root fixes (2026-08-20)

- Đã sửa sáu family từ baseline ở tầng cơ chế, không map câu hỏi/paper/table cụ
  thể: unique partial-title/filename phrase, truncated filename tail, mixed
  anaphor + explicit document, descriptor chỉ trong L1 recent/referent scope,
  bounded ambiguous-pair correction và accent-insensitive contextual aliases.
- Paper facet nhận `task/tasks` theo word boundary; `multitask` không bị match
  nhầm. Grouped metric vector giữ đúng owner cho tới metric kế tiếp, nên dạng
  `CCC a/b/c; F1 x; UAR y` không cho F1 mượn nhầm số CCC.
- Multi-document quantitative coverage chấp nhận một insufficiency section chỉ
  khi section đó gắn rõ identity của đúng document; chỉ nhắc title hoặc generic
  `Ours/Baseline` không đủ để lấp nghĩa vụ evidence.
- Conversational evaluator nhận diện thêm abstention EN/VI và kiểm tool theo
  required-minimum + optional tool có semantics; exact contract vẫn dùng được
  qua `expected_tools_exact`.
- Regression tests gồm negative topical alias, postposed pronoun không connector,
  title dài/truncated, mixed L1 scope, grouped metric và abstention/tool variants.
  Full backend **622 passed**; desktop production build pass. Chưa claim score
  regression-v1 mới: live 60-turn Sol rerun còn chờ 9router online và tuyệt đối
  không fallback sang model khác.

### External-corpus E2E + grounding checkpoint (2026-08-13)

- MTRAG passage corpus giờ đi qua adapter evaluation-only tổng quát ở ranh giới
  evidence: external collection/passage ID → `AgentService` document contract.
  Không đăng ký 366.438 passage thành paper, không sửa SQLite/LanceDB/LightRAG
  production, và qrels/reference target không thể đi vào prompt.
- Stratified live Sol run gồm 15 strata hiện có (4 domain × answerability):
  answerable/partial retrieval `dual original+Aya-rewrite RRF` đạt Hit@5 `0.50`,
  Hit@10 `0.625`, MRR@10 `0.3646`; tốt hơn từng nhánh trên slice nhưng vẫn là
  bottleneck. Token-F1/ROUGE-L với model-authored target chỉ là diagnostic; ít
  nhất một reference answer về privacy nationwide mâu thuẫn evidence/law nên
  không được coi là correctness ground truth.
- General fast social-act routing (`Thank you`/`OK`/`Cảm ơn` variants) bỏ RAG
  kể cả khi L1 còn focus: boundary run route đúng chat 3/3, routing P50 `0.024ms`,
  total khoảng `1.6–2.2s`; document focus không bị xóa.
- BM25 score/overlap answerability gate bị bác bỏ: group-held-out 33 conversation
  chỉ AUC `0.495` dù dev AUC `0.673`. Không promote threshold tune theo benchmark.
- Semantic evidence-sufficiency service có schema `sufficient/partial/
  insufficient/ambiguous`, fail-closed và không chứa tên paper/dataset. Trên
  boundary slice: unsupported abstention tăng `0/4 → 4/4`, social vẫn `3/3`;
  mọi relevant top-5 supported case được giữ ở `partial/ambiguous`, còn hai
  `insufficient` đều là top-5 retrieval miss. Tuy nhiên Sol guard thêm P50
  `4.59–6.40s`, nên hiện chỉ là opt-in guarded evaluation, chưa bật production.
- Grounded prompt mặc định đã yêu cầu exact entity+relation support thay vì
  keyword overlap; generic ambiguity được phép trả lời có điều kiện/clarify,
  chỉ `insufficient` mới hard-stop trong guarded mode.
- Verify local mới nhất: backend **570 passed**, desktop production build pass.
  Fresh internal paper-agent HTTP/SSE run chưa được thực hiện vì privacy gate
  yêu cầu authorization cụ thể cho việc gửi excerpt 21-paper clone qua 9router.

- Backend full suite gần nhất: **394 passed**; `compileall` green; desktop
  production build gần nhất green.
- Latency hardening live trên backend code mới + `cx/gpt-5.6-sol` ngày
  `2026-08-02`:
  - explicit ViSEC Table 2: **2,37 s wall**, router 2,394 s, cached retrieval
    0 ms, deterministic render+validation 13,6 ms, `attempts=0`; không gọi
    LangGraph planner/answer generation. Cold local retrieval cùng query đo
    khoảng 1,4 s thay vì provider-backed generation.
  - ViSEC–ASPIRE comparison: **24,43 s wall** so với trace cũ **76,48 s**;
    cache-miss decomposed local retrieval 1,52 s + hop 2 229 ms, deterministic
    graph 15 ms, first validated block sau 4,74 s generation, đúng một answer
    generation (`attempts=1`). Total còn bị chi phối bởi 19,21 s Sol generation.
  - Cross-paper prompt round-robin + canonical coverage metadata sửa lỗi retrieved
    ASPIRE nhưng prompt greedy chỉ chứa hai ViSEC figures rồi model báo thiếu source.
- Audit lại ngày `2026-07-31`: **344 passed in 17.90s**, SQLite integrity/FK
  sạch, toàn bộ 335 LightRAG chunk đã map revalidate với **0 stale**. Scoped
  live visual smoke trả đúng accepted ASPIRE Figure 1, table có canonical ID,
  và empty scope trả 0 evidence.
- Retrieval eval live (8 labeled queries) sau promote v3, hybrid RRF không dùng
  cosine reranker cũ: **Hit@3/5/6 = 1.0, MRR@6 = 0.9375, nDCG@6 = 0.9539,
  P50 = 149.99 ms, P95 = 244.0 ms**. Baseline v2 + cosine rerank là
  Hit@3 0.875, MRR 0.9062, nDCG 0.9288, P50 1.197 s.
- Diagnostic mới bỏ tên paper, 15 intent x EN/VI (không phải held-out): tổng
  Hit@3 `0.633`; English `0.933`, Vietnamese chỉ `0.333`. 9/15 câu Việt không
  thấy đúng document trong top 6 và tài liệu 9Router tiếng Việt hút top-1 ở 8
  failure case. Đây là raw `search_hybrid`; cần live full-agent query-rewrite
  E2E trước khi kết luận mức ảnh hưởng cuối. Report:
  `data/retrieval_eval/RAG_SYSTEM_EVALUATION_20260731.md`.
- Cosine embedding reranker được tắt mặc định vì benchmark cô lập cho thấy làm
  ASPIRE tụt hạng và tăng latency. Cross-encoder local đã có backend/config
  fail-closed nhưng chỉ bật sau khi model cụ thể được tải và benchmark.
- Compare intent với từ hai focused papers trở lên được tách thành các retrieval
  branch scoped theo canonical document ID rồi chạy song song.
- Retrieval mặc định `auto`: focused direct/table dùng local hybrid; mỗi branch
  của known-paper comparison cũng dùng scoped local hybrid; discovery/cross-document
  chưa decomposition mới dùng LightRAG. Policy không thêm router-model call và luôn
  ghi `selected_engine` + `policy_reason`; retrieval cache hiện `v16` và fingerprint
  cả graph-parent provenance lẫn max-hop/max-subquery policy.
- Adaptive hop 2 chỉ chạy khi evidence/structured-facet coverage thiếu, tối đa
  **2 hop / 3 subquery**, cùng-hop chạy parallel, branch giữ canonical scope và
  timeout riêng 45 giây. Hop 1 luôn được giữ; hop 2 không thêm canonical evidence
  bị discard với `no_new_evidence`.
- Provider-free live bridge smoke đã resolve graph entity về scoped canonical
  parent, không đưa graph prose vào evidence. Chưa dùng smoke này để tuyên bố
  retrieval quality production; tại thời điểm checkpoint này held-out >=60 câu
  vẫn là gate. Policy và baseline mới nằm ở mục regression-v1 phía trên.
- Lifecycle integration bằng DB tạm pass: foreground chặn fold, idle chạy đúng
  một GPT call, commit summary revision 1 và apply L3 operation; dormant startup
  chạy 0 GPT calls.
- Strict E2E đã pass với conversation `b08d1f90-c050-4d74-8c2f-b04125ba3796`: 7/7 runs `completed`, 14 messages, không `run.failed`, không generic quantitative fallback.
- Chuỗi đã kiểm ASPIRE → casual → resume benchmark → WhiSER → previous paper → one-best architecture figure → benchmark table. Turn 3 và 7 đều có claim validator kiểm đủ `accuracy`, `f1`, `ccc`.
- Turn 6 trả đúng một Figure 1 accepted/content/complete, image HTTP 200; turn 7 giữ canonical `table_id` và `table_index=1` qua history projection.
- DB cuối chuỗi: L1 = ASPIRE, recent paper thread = WhiSER, `memory.revision = summary_revision = 7`, recent beats revisions 5/6/7.
- E2E trước migration xác nhận answer, intent router và vision dùng
  `cx/gpt-5.5` qua 9router; cấu hình hiện đã chuyển sang GPT-5.6 Sol mặc định.
  Ollama chỉ phục vụ embedding.
- GPT-5.6 Sol strict E2E mới pass tại
  `data/logs/e2e-gpt-5.6-sol-20260723-final.json`, conversation
  `f12c8042-927f-41ce-9c10-0e53bccb3549`: 7/7 completed, figure HTTP 200,
  table/figure identity giữ nguyên và quantitative Acc/F1/CCC được kiểm.
- Client streaming đọc body lỗi trước `raise_for_status`, nên lỗi 9router không bị che bởi `ResponseNotRead`.
- Model/provider policy fail-closed: unknown provider không còn rơi sang Ollama;
  LightRAG không còn model chain; gateway-reported model mismatch bị từ chối cho
  answer/nonstream/stream và vision provenance.
- Live GPT smoke ngày `2026-07-23 10:40 +07` pass (test conversations đã cleanup):
  health `ok`, router chọn
  `chat`, answer đúng `OK`, run `completed`; L2 cursor `0→1`, job về `idle`.
  9router nhận alias `cx/gpt-5.5` nhưng envelope báo canonical `gpt-5.5`;
  matcher chỉ chấp nhận đúng de-namespaced alias, vẫn từ chối model/version khác.
- Model migration smoke ngày `2026-07-23` pass: health quảng cáo đủ
  Sol/Terra/Luna, mọi fixed role mặc định là `cx/gpt-5.6-sol`,
  `model_fallback_enabled=false`; direct Sol completion trả đúng `OK` và
  response envelope báo canonical `gpt-5.6-sol`.
- Gateway/client fail-closed với malformed/empty response; LightRAG không retry
  quota ba lần và không chuyển model. UI không còn quảng cáo local Qwen path mà
  backend thực tế không dùng.
- Post-ingest status audit: `processed=21`, mọi trạng thái khác bằng 0. LightRAG
  query smoke về WHiSER trả 106 mapped results; top result và 4/6 top sources
  đầu trỏ đúng canonical `WhiSER.pdf`.

## Env chốt (`backend/.env`)

```text
DEFAULT_MODEL=cx/gpt-5.6-sol
LLM_PROVIDER=openai_compatible
OPENAI_API_BASE=http://localhost:20128/v1
ROUTER_MODEL=cx/gpt-5.6-sol
ROUTER_LLM_PROVIDER=openai_compatible
EMBEDDING_MODEL=embeddinggemma:300m
EMBEDDING_DIM=768
EMBEDDING_MAX_TOKEN_SIZE=2048
EMBEDDING_QUERY_PREFIX="task: search result | query: "
EMBEDDING_DOCUMENT_PREFIX="title: none | text: "
RERANK_ENABLED=false
RERANK_MODE=cross_encoder
RERANK_MAX_CANDIDATES=20
AGENTIC_RETRIEVAL_DECOMPOSITION_ENABLED=true
AGENTIC_RETRIEVAL_MAX_HOPS=2
AGENTIC_RETRIEVAL_MAX_SUBQUERIES=3
AGENTIC_RETRIEVAL_HOP_TIMEOUT_SECONDS=45
RETRIEVAL_ENGINE=auto
VISION_PROVIDER=openai_compatible
VISION_MODEL=cx/gpt-5.6-sol
LIGHTRAG_LLM_MODEL=cx/gpt-5.6-sol
LIGHTRAG_LLM_FALLBACK_MODELS=
LIGHTRAG_LLM_MAX_ASYNC=1
LIGHTRAG_LLM_TIMEOUT_SECONDS=300
LIGHTRAG_LLM_TIMEOUT_RETRIES=1
LIGHTRAG_CHUNK_TOKEN_SIZE=600
LIGHTRAG_CHUNK_OVERLAP_TOKEN_SIZE=80
```

## Quy tắc giữ nguyên

- Không đưa router/answer về Ollama nếu user không yêu cầu rõ.
- Không hardcode paper → figure number.
- Không cập nhật L1 từ run fail/invalid; chỉ commit focus sau grounded generation thành công.
- Không bịa metric/parameter count khi retrieval thiếu evidence.
- Bulk external processing phải giữ explicit approval/provenance; approval cho
  corpus hiện tại đã được user cấp ngày 2026-07-23.

## Embedding A/B và production promotion — 2026-08-01 → 2026-08-02

- Production đã promote sang Ollama `embeddinggemma:300m`; `data/lancedb` và
  cả ba LightRAG VDB dùng cùng vector space/prefix.
- Đã thêm query/document prefix support và sửa card indexing dùng document
  embedding. Prefix production là `task: search result | query: ` cho query và
  `title: none | text: ` cho document.
- Staging A/B 30 câu title-ablated: BGE-M3 đạt Hit@3 `0.800`, Hit@6 `0.933`,
  MRR `0.7417`; Nomic prefixed lần lượt `0.633`, `0.733`, `0.6217`.
- Lát cắt tiếng Việt: BGE-M3 Hit@6 `0.933` so với Nomic prefixed `0.467`;
  nhưng BGE-M3 giảm dev-8 và latency P50 `268 ms` so với Nomic `128 ms`.
- Chưa promote trước khi có held-out multilingual gate lớn hơn. Báo cáo đầy đủ:
  `data/embedding_eval/EMBEDDING_MODEL_EVALUATION_20260801.md`.
- Qwen3-Embedding-0.6B staging tiếp tục thắng tổng thể: title-ablated Hit@3
  `0.867`, Hit@6 `0.933`, MRR `0.8317`, P50 `214.85 ms`; EN MRR `1.0`, VI MRR
  `0.6633`. Dev-8 giữ Hit@3 `1.0`. Full index 20 docs mất `390.10s`.
- EmbeddingGemma-300M hiện thắng gate cân bằng: Hit@3 `0.933`, Hit@6 `1.0`,
  MRR `0.8444`, P50 `197.39 ms`; VI Hit@6 `1.0`, zero miss; index `128.75s`.
- Snowflake Arctic Embed 2: Hit@3 `0.933`, Hit@6 `0.967`, MRR `0.8722`, nhưng
  P50 `299.32 ms` và index `250.83s`. GTE multilingual không có official Ollama
  manifest; chưa dùng community conversion.
- Qwen3-Embedding-4B bị loại khỏi foreground gate trên M1: 2.5GB/2560d, chỉ
  index 5/20 docs sau ~10.5 phút (ước lượng 40–45 phút full). Partial staging đã
  dừng chủ động và không được dùng tính quality metric.
- Hugging Face direct staging đã hoạt động với pinned model/code revision và
  Transformers `<5`. GTE dense official: index `213.10s`, Hit@3 `0.867`, Hit@6
  `0.967`, MRR `0.7778`, P50 `88.24ms`; nhanh nhưng chưa thắng quality.
- 2026-08-02: dừng staging Jina v3 sau khi kiểm catalog mới. Jina v5 text small
  retrieval revision `6856e76b...` (standard Qwen3, không remote code) full index
  20/20 trong `911.18s`; title-ablated Hit@3/6 `0.967/1.0`, VI Hit@3/6
  `1.0/1.0`, MRR tổng `0.8178`, P50 `205.09ms`. Không promote do dev-8 giảm và
  license CC BY-NC cần gate cho commercial use.
- Jina v4 3.8B không còn là lựa chọn text hợp lý trên M1. V5 Omni-small 2B là
  nhánh tương lai cho direct image/page embedding, tương thích vector space với
  v5-text-small; chưa tải/index hình. Jina v5 là CC BY-NC 4.0.
- Production re-eval sau promote: dev-8 Hit@3/6 `1.0/1.0`, MRR `0.9375`, nDCG
  `0.9539`, P50 `204.27ms`; title-ablated 30 câu Hit@3/6 `0.933/1.0`, MRR
  `0.8444`, nDCG `0.8833`, P50 `198.44ms`. Đây vẫn là development diagnostic,
  không thay cho held-out >=60 câu.

## ViSEC focused-table regression — 2026-08-02

- Lỗi UI không phải retrieval miss: LanceDB đã trả đúng ViSEC Table 2 rank 1,
  nhưng numeric validator không nhận `Pitch-fusion` dạng sentence-cased hyphen
  nên false-reject và thay cả answer bằng fallback không số.
- Subject parser đã nhận model hyphenated; replay đúng payload cũ support 12/12
  ô UA/WA. Live two-turn Sol E2E conversation
  `23095ae1-5e0f-48ea-9b3b-cd599a011eaa` trả nguyên Table 2,
  `fallback_used=false`, giữ đúng một focus ViSEC.
- Discourse marker “thế” với model đã có trong recent context không còn bị coi
  topic switch/cross-paper compare. Query đếm/liệt kê bảng giờ merge toàn bộ
  canonical table inventory của focused paper; ViSEC smoke có đủ Table 1 + 2.

## Generalized document scope, evidence, and debug trace — 2026-08-09

- Document identity is now resolved from the live catalog before the intent
  router. Filename/title aliases are Unicode-, case-, separator-, and compact-
  tolerant; weak generic keywords cannot silently become paper identities.
  Ambiguous aliases such as `9router` fail closed with candidate filenames.
- Scope is one ordered decision with explicit precedence: current-turn target,
  current multi-paper mentions, durable plural referent, then sticky L1. A
  grounded multi-paper turn stores its ordered referent pair, so `chúng`, `cả
  hai`, `hai cái đó`, `tụi nó`, `bọn nó`, `both`, and `them` survive a casual
  detour without collapsing to one PDF.
- Joint scope is span/grammar based rather than tied to `hai bài A và B`:
  `A và/with/and/vs/+/, B`, quantified pairs, compact aliases and full stems all
  use one resolver. Correction discourse assigns selected/rejected roles for
  reset, negation, replacement and preference (`A? No, B`, `replace A with B`,
  `use B instead of A`) so a rejected ambiguous alias cannot poison B.
- One shared comparison-language predicate is used by scope, router and rewrite.
  Coverage cardinality stays separate from rhetoric: extracting abstract/table
  lists for A+B remains `direct_answer`; `đối chiếu/differences/against/contrast`
  remains `compare` while preserving both IDs.
- Multi-document obligations are independent of the router's intent label.
  Retrieval validation reports missing document IDs and cannot pass until every
  requested paper is covered. Figure-only evidence cannot satisfy a required
  document's text/result facet; fast mode may spend one bounded scoped repair
  hop for this correctness obligation. Answer validation also requires each
  paper identity to own evidence from that paper; generic `Ours/Baseline` and a
  shared value cannot cover two documents. The single-paper canonical-table
  fast path remains disabled for multi-paper scopes.
- Exact `Table/Bảng N` requests now reserve only canonical Table N from each
  focused paper; a lexically stronger but unrequested table cannot take its
  context slot. Live no-provider smoke returned MSF-SER Table 3 in 12.5 ms with
  zero model calls and three SSE deltas.
- Numeric evidence validation now understands hierarchical Markdown table
  headers/sections (including CCC A/V/D) without paper-specific mappings and
  rejects wrong section, metric column, or model ownership.
- Table caption reconciliation is provenance-gated: only one structured
  Figure-labelled table plus one unmatched explicit `Table N:` caption from the
  same document/page may be repaired. Live CMDM Table 6 is recovered this way;
  its physically merged Table 2 still abstains because no safe standalone
  artifact exists. Live canonical table scan is 60/60 render + validate.
- Optional per-run debug trace is implemented behind both a loopback-only server
  gate and an off-by-default UI switch. It stores redacted, bounded milestone
  snapshots in a separate TTL/capped table; TTL/count purge runs at startup and
  access checks the actual ASGI peer. Structured and opaque-string redaction
  covers snake/kebab/camel-case secrets while preserving timing/token metrics;
  normal run APIs and SSE never expose raw prompts/drafts.
- Verification: full backend **514 passed**; desktop `tsc && vite build` passed.
  Independent metamorphic audit passed: core ordered-pair grammar 6,300/6,300,
  correction/replacement 11,340/11,340, additive contrast 1,260/1,260,
  ambiguous joint/rejected-alias 252/252, comparison intent 4,200/4,200, and
  multi-document extract/list intent+coverage 2,940/2,940.
  Live current-backend smoke resolved `not 9router; use ASPIRE, show Table 2`
  to ASPIRE, retrieved in ~263 ms and began validated direct reveal ~14 ms after
  retrieval; CMDM Table 6 direct-rendered without a model call. 9router remained
  unavailable, so no new full Sol multi-turn result is claimed and no fallback
  model was used.

## Canonical paper evidence cards — 2026-08-12

- Đã implement card schema/service/builder cho 8 facet dùng chung: task,
  architecture, dataset/setup, benchmark results, contributions, training,
  ablation, limitations và visual evidence. Card chỉ là navigation/index layer;
  mọi claim vẫn resolve về canonical chunk/table/figure trước khi vào prompt.
- Multi-paper retrieval batch-load card theo ordered scope, dựng coverage matrix
  paper × facet và chỉ chạy raw retrieval cho facet còn thiếu. Exact Table,
  Figure, Page và quote tiếp tục đi canonical artifact path.
- Builder pin approved `cx/gpt-5.6-sol`, atomic publish, provenance/hash/quote và
  numeric-claim validation; lỗi/quota dừng job, không fallback model. Re-index
  chỉ queue durable pending job, không âm thầm gửi tài liệu ra provider.
- Progressive multi-paper stream validate từng paper bằng evidence riêng trước
  reveal, rồi mới sanitize synthesis. UI nhận `evidence.paper.ready` và
  `answer.paper.validated`; malformed/missing section thành insufficiency block.
- Đã tạo dev 20 turn và held-out 60 turn (36 VI/24 EN, phủ đủ 21 docs ít
  nhất hai lần), kèm checksum/corpus fingerprint và HTTP/SSE evaluator.
- Feature flags build/runtime/progressive stream vẫn default off ở production;
  isolated eval backend đã bật runtime/progressive với build off sau khi backfill.
  Retrieval cache namespace đã bump `v18`.
- Authorized isolated backfill/dev gate đã hoàn tất ngày 2026-08-14: cards 21/21,
  fresh card-path 5/5 và full agent dev-20 20/20. Production rollout vẫn là gate
  riêng. Dòng này mô tả checkpoint trước khi user cho mở ngày 2026-08-14; suite
  nay đã retire thành regression-v1.

## Official public evaluation data — 2026-08-12

- Đã pin và tải riêng khoảng 1,9 GB từ 7 nguồn official: MTRAG/MTRAG-UN,
  MultiChallenge, WildBench v2, 7 split ChatRAG-Bench, SPIQA Test A/B/C và
  MMLongBench-Doc. Không ingest bất kỳ dữ liệu public nào vào corpus production.
- Manifest ghi source revision/license/selection và SHA-256; full validation
  pass 17 checksum, 3 ZIP và toàn bộ expected row counts.
- Adapter catalog có 15.892 case/6 runner mode. 50 prompt WildBench thật được
  chuẩn hóa thành local-routing-negative suite chạy qua public HTTP/SSE API.
- Sol smoke đầu 2/3: một prompt external “class materials” bị router gọi local
  RAG và retrieve 3 paper sai. Đã thêm policy tổng quát: LLM không được bật local
  retrieval nếu turn không có resolved catalog identity, active/recent local
  document thread hoặc explicit local-library anchor. Rerun case lỗi pass:
  `chat`, zero focus/retrieval. Không hardcode prompt/tag/paper.
- Ba file do Antigravity tự sinh được đánh dấu invalid drafts; evaluator giờ
  validate schema và fail rõ ràng thay vì vỡ `KeyError` hay coi chúng là public
  benchmark. MultiChallenge chưa có license file nên chỉ dùng research/eval.
- Verification checkpoint: backend **535 passed**; desktop production build
  passed. Full 50-case model-free adversarial gate pass; live Sol targeted rerun
  pass. Không chạy 50 Sol answers để tránh tốn quota vô ích ở bước smoke.

## Isolated MTRAG baseline + language fidelity — 2026-08-13

- Đã dựng index evaluation-only SQLite FTS5/BM25 gồm **366.438 passages** từ
  bốn corpus official. Builder bắt buộc output nằm dưới
  `data/retrieval_eval/public/indexes/`, không import/chạm catalog, LanceDB,
  LightRAG hoặc SQLite production.
- Full no-model matrix đã chạy **777 MTRAG Human + 332 MTRAG-UN cases** ở sáu
  query mode. Human official rewrite tốt nhất trong ba input Human:
  Hit@10 `0.5187`, MRR@10 `0.2976`; nối toàn bộ history làm query giảm chất
  lượng và tăng P50 lên `173.9–347.1 ms` tùy suite.
- 12-turn official-reference Sol smoke trên backend/data directory cách ly:
  **12/12 pass**, route chat 12/12, zero local retrieval, 4/4 unsupported turn
  abstain; median TTFT `2.60 s`, median total `4.48 s`. Token-F1 mean `0.2466`
  chỉ là diagnostic, không giả làm official judge/gate.
- Smoke bắt thêm lỗi language fidelity: explicit English vẫn có thể bị persona
  kéo sang tiếng Việt. Prompt composer giờ ưu tiên explicit language request và
  bỏ Vietnamese pronoun/opening/style khỏi English answer prompt. Targeted Sol
  case đổi F1 `0.024 → 0.299`, trả hoàn toàn English, router `0 ms`.
- Verification hiện tại: backend **550 passed**; desktop production build pass.
  Production 21-paper corpus không bị thay đổi.
- Bounded EmbeddingGemma candidate-rerank A/B cho kết quả không đồng nhất:
  Human official rewrite 40 câu tăng Hit@10 `0.55 → 0.80`, MRR
  `0.241 → 0.562`; MTRAG-UN questions 40 câu giữ Hit@10 `0.35` nhưng MRR giảm
  `0.192 → 0.163`. Hybrid UN Hit@10 `0.40` nhưng MRR `0.181`. Candidate recall
  tương ứng `0.875` và `0.600`, cho thấy query rewrite/candidate generation mới
  là bottleneck trước rerank ở UN; không promote embedding rerank toàn cục.
- Candidate vector cache fingerprint gồm model, prefix, truncation và corpus
  source SHA-256; passage thay nội dung nhưng giữ ID sẽ không reuse vector stale.
- Actual Aya QueryRewriteService đã được đánh giá trên 40 MTRAG Human cases cân
  bằng domain: raw last-turn Hit@10/MRR `0.475/0.325`, rewrite-only
  `0.600/0.393`, original+rewrite RRF `0.600/0.402`, official rewrite
  `0.525/0.374`. 21/40 gọi Sol; rewrite latency median `2.54 s`, P95 `5.73 s`.
- Eval bắt một lỗi generalization thật: generic IBM Cloud “benchmark” bị
  paper-result heuristic biến thành Acc/F1/CCC/arousal/valence. Heuristic này
  giờ chỉ chạy khi scope layer cung cấp authoritative `working_topic`; generic
  chat dùng rewrite bình thường. Dual-query chưa promote runtime trước internal
  dev gate vì không được nhân đôi LightRAG/card/exact-artifact mù quáng.
