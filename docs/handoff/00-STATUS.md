# 00 — Status snapshot (2026-10-10)

## 2026-10-15 (Claude) — sealed multi-format evaluation (plan 09)

- Corpus (sub-agent A): 19 public documents, `data/retrieval_eval/sealed/v1/`
  (8 native PDF incl. 2 Vietnamese releases, 2 scanned PDF, 4 DOCX, 3 MD/TXT,
  2 slide decks; 12 en / 7 vi). Questions (sub-agent B, independent, unseen
  by the head before the run): 89 turns / 75 conversations, groups S1–S10.
- Ingest through Aya's own pipeline exposed generic bugs, fixed before the
  run (`b1c3df6b`): surrogate characters dropped a PDF; Vietnamese
  administrative headers became titles; DOCX/setext/year-ending titles.
- **Sealed result** (Aya, cx/gpt-5.6-sol, judged by sub-agent D, head
  spot-checked 13/89 and agreed): **75.3% correct (CI 65–83%), 85.4%
  correct+partial**, 1 fabricated number on 12 unanswerable turns, 12 false
  abstentions. By language: vi 80.0%, en 70.5%. Weakest: cross-document S5
  44%, figures S3 60%. Gates (85% / 0 fabricated / ≤10% false abstention)
  NOT met. This is the honest out-of-sample number; the set is
  regression-only from now on.
- Causes and generic fixes (`f20d37cf`): model-signalled missing context
  (marker) for whole-document re-answer, dominant-document candidate, title
  acronyms (RAG), evidence-backed check of guessed focus, caption-based
  figure answers without pixels, comparability check before cross-document
  arithmetic, full-document cap 160k.

## 2026-10-14 (Claude) — misspelled names, focus, titles, formulas, post-verify

- **Focus on artifact follow-ups fixed.** Dataset/metric names (IEMOCAP,
  LSSED, MPCL) were treated as "new paper" and dropped sticky focus, so
  "Show Table 3…" after an ASPIRE turn returned LPMN's Table 3. An entity now
  switches papers only if it resolves to exactly one other document; ambiguous
  entities no longer pick an arbitrary document.
- **Real titles.** Catalog titles were author lines / body sentences for every
  PDF. New `app/catalog/titles.py` (largest bold page-1 line, rotated arXiv
  stamps skipped, missing spaces restored; Markdown H1; filename fallback).
  `scripts/backfill_document_titles.py --db-path … [--apply]` ran on the eval
  clone and on production `data/sqlite/app.db` (backup:
  `data/sqlite/backups/app-before-title-backfill-20261014.db`).
- **Misspelled / described names.** Catalog fallback: edit-distance name match
  (only for words absent from library text, so "whisper" stays a word) and
  rare-title-word description match. Router prompt lists the library
  ("L<n>: title (file)") and may return `library_document`, used as focus and
  as a local anchor (fixes Vietnamese descriptions of English titles).
- **Wrong table numbers** yield to a clearly described table; the direct table
  answer says "You mentioned Table 1, but … is Table 2". Cache v35.
- **Formulas:** KaTeX rendering (`$$`, `\[ \]`, `$`, `\( \)`; money kept as
  text), prompt rule to write LaTeX only from excerpt text.
- **Step B:** `ANSWER_VERIFY_MODE=post` (default) streams first, flags lines
  with the legacy validator + sourcing check, and one model call keeps /
  rewrites / removes only flagged lines (`message.reset`). `gate` = legacy.
- **Post-verify measured.** Flagging on `numeric_probes_v3`: 47/60 wrong
  answers flagged, 26/60 correct answers pay one extra call. Sol's verdict on
  the 12 hardest misses (period / negation / ambiguous / borrowed numbers) +
  their controls: 23/24 correct (only v3-05-amb-wrong kept). On live
  acceptance turns the legacy validator flagged 6 correct answers; Sol kept
  all 6 unchanged, 0 errors, 0 bad rewrites.
- **Formula retrieval.** `searchable_text` maps Unicode math letters to plain
  letters for FTS + embeddings (`scripts/refresh_search_text.py` applied to
  clone and production; backups `app-before-search-refresh-20261014.db`,
  `lancedb-before-search-refresh-20261014`). Formula questions add the
  `training_method` facet; reserved passages rank by IDF-weighted overlap.
  Per-document cap kept on purpose (removing it drops public-36 to 0.514).
  Cache v36.
- **Round 3 acceptance** (`aya-acceptance-round3-20261014.json`): G1 20/20,
  G2 18/18, G3 10/10, G4 11/12 (c11t3 was a checker false positive: honest
  caveat after a correct answer), G5 15/15. First token median 3.4 s, p95
  12.7 s (slow turns = vague follow-ups: router + LLM rewrite were serial).
  Manual review found one real miss the checker passed: c03t4 (KST table on
  IEMOCAP/LSSED) lost focus — dataset names read as a topic switch.
- **Fixes after round 3** (`37e044e6`, `6e7f9734`):
  - post-verify checks every numeric line (`ANSWER_VERIFY_SCOPE=numeric`,
    default; `flagged` = old). v3 probes: wrong caught 47 → 58/60, no correct
    number changed;
  - verifier sees full sources (was cut at 2.4k chars → deleted correct late
    table rows); mass-removal verdicts ignored;
  - rewrite runs concurrently with the router;
  - router naming the active paper, or a table/figure/formula ask without
    another paper name, keeps sticky focus;
  - checker: table attach needs printed values; refusal detected at opening.
- **2026-10-15:** complex-50 rest: hard_negative_abstention 10/10,
  implicit_comparison 2/2, rag_004 pass; id-referenced figure/table asks
  ('Bảng 2 trong file <uuid>') 0/10 and 1/10 → library document ids in a
  message are explicit references and force retrieval (`5a472996`,
  `2d47dbe3`): figures 10/10, tables 7/10 (rest ask for 'Bảng 0', a
  zero-based label in Codex's generator; Aya says no Table 0 exists).
  MTRAG rerun with the stricter grounding rule (`a6734bb2`): abstention
  0.733 → 0.80, token F1 0.238 → 0.261, recall 0.486 → 0.514.
- **Codex single-turn suites (adapted, `evals/codex_adapted/`):** casual-50
  49/50 → wb_011 fixed ('việc' fuzzy-matched ViSEC; diacritic words are no
  longer typos, `216fac4e`); ragbench-25 all pass except rag_004 (two papers
  described by content; router only sees titles) → described multi-paper
  questions now scan the library (`38929c21`); complex-50 first 18/18 (rest
  re-running). SPIQA/MMLongBench three-mode (3+3): Aya 6/6.
- **Public suites (Codex downloads), 2026-10-14:**
  - WildBench routing (50 real-user chats): 48/50 stay chat. One false
    activation came from my named-term retrieval override (pasted table's
    capitalised words) → fixed `0617a115`; one router pick of a library paper
    for an essay prompt (open).
  - MTRAG e2e (same 15 stratified cases as Codex's v1): token F1 0.198 →
    0.238, recall 0.437 → 0.486, abstention 0.733 (unchanged). Unanswerable
    turns are sometimes answered from general knowledge (e.g. "Mario Mars" →
    Bruno Mars). Aya's rewrite hit@5 0.375 vs original 0.5 on 8 scored cases.
- **Router scope/presentation** (`e48cec53`): intent regexes are now
  fallbacks; evidence of absence (term counts) for library questions,
  ambiguity rule (`369dcb1f`). Targeted hard rerun (6 conversations): 19/28,
  H8 0/7 → 2/7, H9 4/5.
- **Hard suite r2/r4 and quota (2026-10-14 late):** always-on full-document
  context lifted the judge to 56/90 (r2) but exhausted both 9router accounts
  after 9 turns of r3 → `FULL_DOCUMENT_CONTEXT=auto` (default; re-answer
  with the whole paper only when the excerpt answer reports missing info),
  `always`, `off`. Later fixes: verbatim tables only on show requests
  (`3e0633e9`), questions about a table answered by the model, asked deltas
  computed (prompt no longer forbids it), header-aware table matching,
  several misspelled names per query, library scan with verbatim term pass.
  r4 partial (h01–h05, 32 turns): judge 21/32.
- **Codex suites end to end (post mode):** smoke-8 5/8, dev-20 12/20,
  regression-60 36/60 by Codex's checker, but almost every fail is the legacy
  validator flag (`answer_validation:unsupported_metric_values`), which is a
  gate-mode criterion. Numeric audit (`scripts/audit_numeric_faithfulness.py`,
  terra vs DB tables/passages/figure descriptions) + manual check found real
  issues: post-verify turned "Wav2Vec 2.0" into "Wav2Vec 0" (fixed
  `e97d9575`), Figure N of the focused paper not fetched (fixed `eeeb33b9`),
  one overgeneralisation (KST "3 CCAB best on both datasets").
- **Hard suite** `evals/aya_hard_v1.jsonl` (91 turns, independent author,
  traps on paper inconsistencies). First run, terra judge ≈ 43/89:
  H1 12/17, H2 4/7, H3 3/8, H4 9/14, H5 3/7, H6 4/8, H7 7/16, H8 0/7, H9 1/5.
  Causes: details missed by a 16k excerpt window; library-wide questions
  answered from one paper; short Vietnamese words fuzzy-matched to names.
  Fixes: whole focused paper in context (`FULL_DOCUMENT_CONTEXT`, `4aa7197c`),
  per-document library scan, fuzzy/router focus fixes (`d27eeb7a`).
- **Persona restored** (`eeeb33b9`): the prompt banned anime tone in
  technical replies, i.e. nearly every paper answer.
- **Round 7 (after `160fbab3`, current code):** structural G1 20/20, G2 18/18,
  G3 10/10, G4 12/12, G5 15/15 (G5 rerun separately after a quota stop).
  Terra judge: G1 19/20, G2 16/18, G3 10/10, G4 11/12. First token median
  2.7 s, p95 7.6 s. Remaining judge fails are sub-points: c01t2 "three
  parallel streams", c04t4 paper uses both "Contrastive Supervision/Guidance
  Leg", c09t5 sub-figure (a) not described for a loss question, c10t1 core
  idea without dataset names. Both 9router accounts hit quota once
  mid-run (judge now retries per turn).
- **Biggest find (`160fbab3`):** `graph._format_retrieved_docs` capped the
  answer prompt's excerpts at 5,200 chars and silently skipped later
  sources, so the 10-source / 16k context from plan step 1 never reached the
  model (a formula passage ranked 4th was invisible; Aya said "the paper has
  no formula"). Cap is now 24k (composition already bounds the budget).
  Retrieval-level metrics (public-36, local-60) never saw this because they
  stop at composition — judge answers end to end.
- Round 5/6 (before that fix): structural 80/80; terra judge G1 19→18/20,
  G2 12→13/18, G3 8→9/10, G4 8/12. Post-verify now also sees the recent
  conversation (follow-ups answered from an earlier table lost values).
  c08t3 label corrected (A+V fully-supervised = 63.2/60.8 per Table 5).
- **Round 4 + terra content judge** (`scripts/judge_aya_acceptance.py`,
  `aya-acceptance-round4-20261014-judge.json`): structural checks G1 19/20,
  G2 18/18, G3 10/10, G4 12/12, G5 14/15, but content judge G1 18/20,
  G2 13/18, G3 8/10, G4 11/12. Structural checks overstate quality; use the
  judge from now on. Causes fixed in `e2cca92c`: overview anchors on specific
  questions, formula asks getting training anchors, lexical promotion
  counting expanded neighbourhoods + function words, table-number correction
  reading retrieval expansion terms, post-verify reading `content` instead of
  the expanded excerpt (deleted correct CMDM values). public-36 0.653 → 0.681.
- G5 fuzzy acceptance (15 turns): 6/15 → 14/15 → f15 fixed after cache bump.
- Backend 1,392 passed; desktop build pass. A first round-2 run was voided by a
  9router outage (76 ConnectError runs); evaluator now aborts on provider
  failures instead of scoring empty answers.

## 2026-10-13 (Claude, Haiku-executed plan) — Bước A baseline acceptance

- Bộ đo mới commit (`df7bd33e`): `evals/aya_acceptance_v1.jsonl` (60 lượt,
  12 hội thoại; nhãn kiểm trên DB; nhãn công thức và CRAB bảng đánh dấu
  low-confidence) và `evals/numeric_probes_v3.jsonl` (120 dòng, 60/60).
- **Baseline acceptance (code hiện tại, Sol):** G1 19/20 (95%), G2 12/18 (67%),
  G3 9/10 (90%), G4 11/12 (92%). Kết quả thô: `data/retrieval_eval/internal/aya-acceptance-baseline-20261013.json`.
- Review tay 6 lượt G2 fail (`aya-acceptance-g2-review-20261013.json`):
  - Lỗi thật của Aya: c03t4 chọn nhầm paper khi người dùng nêu rõ tên
    (trình bày bảng của paper khác); c08t2 chọn sai bảng (Table 4 yêu cầu
    "fully-supervised" nhưng trả bảng K′); c10t5 có hình trong paper nhưng
    không lấy được, rơi vào fallback "không thể xác định".
  - Không tính là lỗi: c02t3, c05t2, c12t2 (công thức). Aya trả công thức
    LaTeX hoặc nói rõ không có trong tài liệu. Nhãn công thức chưa xác minh
    nên không dùng để chấm.
- Phát hiện lỗi của chính evaluator: bản đầu không lưu answer nên không kiểm
  được; đã sửa để lưu answer và artifact được chọn.
- Probe số v3: cờ tự động 50 ca là false positive của công cụ kiểm (không
  tính thương/hiệu/phần trăm). Đã đọc tay các ca; nhãn đúng.
- Chưa làm: Bước B (bỏ chặn). Chưa thay đường chặn cho đến khi có bộ đo số
  đầy đủ.

## 2026-10-13 (Claude, plan execution) — Bước 0–2 đang chạy

- **Bước 0 (commit sạch):** 30k dòng tách thành 3 commit theo mảng
  (`491cdf10`, `528f25ac`, `19a76bdb`) + quota retry `229fcc56`. `check.txt`
  và `backend/uv.lock` đưa vào `.gitignore`. Suite 1.363 → 1.367 pass.
- **Quota retry:** `stream_chat` retry chỉ lỗi quota (429/`usage limit`/
  `[429]` trong 503), tối đa 2 lần, cùng model, chỉ khi chưa stream nội dung.
  Không fallback model/Ollama. 4 test.
- **Bước 1 (context):** budget trả lời thường 10 source / 16k ký tự (brief 6 /
  8k). public-36 recall **0,403 → 0,653**, đo lặp 2 lần trên code thật, p50
  ~220 ms. `top_k` giữ 10: thử 20 làm giảm xuống 0,611. Local-60 không đổi
  (0,97 / 0,90 / 1,00). Commit `d225e7e9`.
- **Bước 2 (verifier):** module `answer_verifier.py` (kiểm số có trong source +
  sửa tối đa một lần bằng model, chỉ câu bị flag). 8 test. Đã **thêm shadow**
  cạnh validator cũ, chỉ ghi trace (`answer_number_shadow`), không đổi câu trả
  lời. Validator cũ **vẫn chặn**; chưa thay đường chặn.
- **Mất mát cần ghi nhận:** bộ probe mù 172 câu (kiểm độc lập duy nhất) không
  còn trong scratchpad sau khi session đổi, không có trong git. Gate "false
  accept ≤ 5%" trên bộ đó hiện **không đo lại được**. Bộ niêm phong mới ở Bước 3
  phải commit vào repo ngay khi tạo.
- Đang chạy: dev-20 Sol trên clone :7790 với shadow để so verifier và validator
  cũ trên cùng câu trả lời thật.

## 2026-10-10 (Codex) — provenance, sentence binding and honest completeness review

- Pure Paper cells now resolve to a unique canonical document ID while keeping
  separate model ownership; shortened titles, alias collisions and documents
  without acronym tokens have paired controls. Declared model/dataset cells
  support spaced slash as well as dash separators; no model-name split under
  a bare Model header. Trace proof compaction also handles pre-hard-cap clipping.
- Full frozen Sol dev-20 **19/20**, one answer fallback after two attempts;
  median total/visible **9.397 / 3.075 s**. Failed report retained. Its complete
  drafts exposed paragraph-level paper inheritance onto another sentence.
  Sentence-local canonical/model binding and conservative paragraph sanitization
  repaired offline; sanitizer no longer clears a valid table's model cell just
  because a different unsafe sentence starts with that name.
- Final-source affected Sol replay **4/4 internal rubric**, zero answer/model/
  tool/transport fallback and zero retries. **NOT an answer-quality pass**:
  last output was sanitized, dropped later numeric rows and appended an
  insufficiency section. New completeness diagnostic explicitly flags this
  despite validator success; do not promote from the rubric count.
- Visible numeric regression unchanged **77/85 wrong rejected / 79/87 correct
  accepted / 71 paired pass**, **8 false accepts / 8 false rejects remain**.
  Public 36 artifact helper remains **0.40278**, not improved/retested here.
- Final combined backend suite **1,355 passed**, 7 existing warnings. Desktop
  build and diff check pass. Graphify **5,154 nodes / 15,919 edges / 185
  communities**; vault refreshed AST-only, manual note preserved. Existing
  version/empty-JSON/stale-community-label warnings remain; no paid relabel.
- Production DB/index/embedding/LightRAG unchanged; facts and dual-query off.
  Owned isolated :7790 stopped, user 9router untouched. Goal active; broad
  arbitrary-format, visual and independently scored quality gates remain open.
  Reports, failure details and next steps: [07](07-NUMERIC-GROUNDING-REVIEW.md).

## 2026-10-10 (Codex) — trace/owner correctness continuation

- Bounded debug traces preserve drafts/failures by omitting successful proof
  repetition; source-bound typed paper/model rows retain local ownership.
  Repeated occurrences, packed thresholds, OCR decimals and numbered document
  heading scope repaired with generic correct/wrong controls.
- Backend **1,344 passed**, desktop build and diff check pass. Visible numeric
  regression unchanged **77/85 wrong rejected / 79/87 correct accepted**;
  **8 false accepts / 8 false rejects remain**. Sol full-20 **19/20**, final
  comparison answer fallback (**48.943 s**), no transport/tool/model fallback.
  Later packed-cell schema precedence replay **3/4** (42.126 s last turn).
  Separate Paper/Model ownership repaired subsequently with paired tests;
  abbreviated identity/composite scope still open. Prior failures retained,
  not a final-source passing full gate. No further Sol calls this turn.
- Graphify **5,141 nodes / 15,883 edges / 167 communities**, Obsidian **5,247
  notes**; AST-only, manual note preserved, prior warnings remain.
- Facts/dual-query off, production index/DB unchanged. Public 36 artifact
  helper baseline remains **0.40278**, not remeasured/improved by guard fixes.
  Goal remains active, arbitrary-format/visual/independent acceptance pending.
  Full context and report distinctions: [07](07-NUMERIC-GROUNDING-REVIEW.md).
- Owned isolated :7790 stopped; user 9router untouched. No active evaluation jobs.

## 2026-10-10 (Codex) — bounded prose attribution

- Explicit named source subjects now own subsequent quantities; counted noun
  heads distinguish cases from hospitalizations. Names come from source grammar,
  not a paper/domain mapping. Respectively pairing uses this owner dimension.
- Visible 172-case replay: **77/85 wrong rejected, 79/87 correct accepted**,
  **71/85 paired pass**. Three gains/no regressions versus Oct 9. Still **8 false
  accepts / 8 false rejects**; limited grammar, not semantic entailment.
- Backend **1,334 passed**, desktop build/diff check pass; Graphify **5,127
  nodes / 15,844 edges / 172 communities**, Obsidian **5,233 notes**, manual
  note preserved. AST only; existing graph warnings remain.
- Frozen Sol dev05 HTTP/SSE replay **3/4**, last comparison rejected after
  two attempts and answered with insufficiency fallback (**42.500 s**). No
  transport/tool/model fallback; answer fallback occurred. Existing metric/owner
  binding issues reproduce with new prose binding disabled; full draft trace
  exceeded 64-KB hard cap. Do not call this a passing E2E gate.
- Facts and dual-query remain off; no production index/DB change. Owned :7790
  stopped, user 9router left running. Details/next work: [07](07-NUMERIC-GROUNDING-REVIEW.md).

## 2026-10-09 (Codex) — row/condition binding continuation

- Numeric source/answer-table cells now bind peer measurements per row and
  per clause; omitted group/section/owner ambiguity, bullet-owner inheritance
  and literal inclusion/exclusion repaired. The Week/tuần xfail now passes.
- Visible 172-probe regression improves from **65/85 → 74/85 wrong rejected**,
  **78/87 → 79/87 correct accepted**, ten gains/no regressions. Still **11 false
  accepts / 8 false rejects**; cached facts add no benefit, both flags remain off.
- Empty Markdown scaffolds left by numeric sanitization now removed. First
  frozen Sol dev-20 was 20/20 rubric but exposed this layout defect; final-source
  run **19/20 + 1 upstream disconnect**, bounded affected replay **4/4** (one
  answer retry, worst 38.475 s). This is not a passing single full-suite run.
- Backend **1,330 passed**, desktop build/diff check pass. Graphify **5,120
  nodes / 15,825 edges / 184 communities**, vault **5,226 notes**; AST sync with
  existing version/empty-JSON/stale-label warnings. See [07](07-NUMERIC-GROUNDING-REVIEW.md).
- No production index/embedding/LightRAG change; no claim that retrieval recall
  or arbitrary-format/visual acceptance has improved from these validator fixes.

## 2026-10-09 (Codex) — numeric correctness and durable fact continuation

- P0 quantities/occurrence/source-anchor repairs, signed Decimal/unit/comparator
  checks, composite benchmark scope and cross-source caveat handling. Facts
  strengthen canonical checks only; default off. Production index unchanged.
- Durable fingerprinted batch checkpoints, lease/heartbeat/backoff, transactional
  source/revision validation and stale-fact filtering; isolated CLI only.
- Backend **1,311 pass / 1 xfail**; final frozen Sol affected replay **4/4**,
  no fallback/retry, median visible delta **2.663 s**, total **8.482 s**.
- Honest visible 172-probe replay: **65/85** wrong rejected, **78/87** correct
  accepted, with/without old cached facts. Still **20 false accepts / 9 false
  rejects**; facts cannot promote. New blind/balanced suite and full dev-20 pending.
- Details/reports/next work: [07-NUMERIC-GROUNDING-REVIEW.md](07-NUMERIC-GROUNDING-REVIEW.md).
- Graphify AST **5,106 nodes / 15,770 edges / 185 communities**, Obsidian
  **5,211 notes**; preserved manual `path_46.md`. Existing skill-version and
  empty-JSON warnings remain, no paid semantic relabeling.

## 2026-10-09 (Claude) — second blind probe set, ingest-time facts (flagged)

- Second blind set (172 probes, 16 non-SER scenarios, 46% Vietnamese, author
  never saw code/tests/first set). Rules-only grounding after fixes: wrong
  claims rejected **58/85**, correct accepted **87/87** (12 unseen scenarios:
  38/63 and 64/64). Residual misses were prose attribution, unqualified
  claims, negation/inclusion ("chưa gồm VAT") — beyond rules.
- Mechanisms added (all general): bare percentages delegated to table-cell
  grounding (old validator could not bind "coverage 81.2%"); Vietnamese
  dot-thousands; diacritic-aware labels ("Năm" ≠ "Nam", "Nữ" short label);
  document-level condition labels (2023 vs 2024 report); condition ambiguity
  across sections/documents; "respectively" positional pairing; stated
  differences must match the stated numbers.
- New ingest-time structured facts (`fact_extraction_service.py`, tables
  `document_facts`/`document_fact_jobs`, `scripts/build_document_facts.py`,
  flag `FACT_GROUNDING_ENABLED`, default off; model `FACT_EXTRACTION_MODEL`,
  approved 9router only). A fact is kept only if its quote is verbatim in the
  chunk and contains the value. Grounding treats facts as labelled cells.
- With Sol-extracted facts on the probe evidence (113 facts, 0 rejected):
  wrong rejected **76/85 (89%)**, correct accepted **85/87**; unseen 12
  scenarios **55/63**, **62/64**. Tuning on this set stopped here.
- Backend **1,268 passed, 1 xfail**. Building facts for the 10 dev-conversation
  papers on the isolated clone (approved corpus) to measure live behaviour.

## 2026-10-09 (Claude) — metric-agnostic numeric grounding; live check on v34+

- Blind probes (181, non-SER domains, expectations written before running,
  author never saw code/tests) showed the metric validator does NOT
  generalise: 22 wrong numbers accepted (revenue, hazard ratio, torque, EM,
  doanh thu… are not in its metric vocabulary → "no_metric_value_claims").
- New `app/services/numeric_grounding.py`, AND-ed into
  `validate_answer_claims` (can only add rejections, never relax the metric
  validator): every stated number must occur in eligible evidence (decimal
  comma, thousands groups, % ↔ fraction, rounding, in-row/column
  differences); table cells carry row labels + column header path (group
  headers, sub-header row below |---|, repeated headers, section rows,
  parentheticals stripped); the number's clause + number-free clauses +
  section heading must not name a different label of the same dimension;
  labels like "w/o X" match negated mentions; "family without member"
  (revenue vs Revenue H1/H2) is ambiguous only when sibling values differ and
  the metric validator did not already check the value. Ungrounded values are
  reported as `metric="number"` unsupported claims → existing retry/sanitize.
- An attempt to let grounding OVERRIDE metric-validator rejections was
  reverted: it accepted 36 wrong claims in existing tests.
- Results: backend **1,246 passed, 1 xfail** (numeric condition columns such
  as "Week = 12" vs "tuần 12" — known limitation). First probe set (now
  contaminated, used for diagnosis): false accepts **22 → 1**, false rejects
  19 → 19 (all from the old metric validator). A second blind set is being
  written by an independent agent for the honest measurement.
- Live Sol (v34 + grounding), dev02/dev04/dev05: **10/10 pass**, marker leak
  gone, sticky/correction turns correct. dev05-t4 needed a retry (old
  validator rejected Fusion-Gate rows) → 21 s to first text; grounding also
  flagged one value (0.514) in the correction draft — unverified whether that
  was a false reject (trace truncated at 16 KB). Number-free prefix check now
  ignores identifier digits (miF1, wav2vec2).
- 9router Sol quota: several `429 usage limit` windows overnight; likely
  shared with Codex/other clients on the same account.

## 2026-10-08 (Claude) — P1 context selection, catalog-derived aliases, live check

- Table role: VI captions (Kết quả/So sánh/Thống kê), result cues before
  statistics cues ("Results under distribution shift" = main), generic metric
  header fallback (mAP, IoU, EM, top-1, PSNR…) read from header rows only.
- Main-result inventory capped at 2 best-matching tables per document.
  Reference expansion: per-paragraph cap 2, relevant artifacts first, total
  bounded by the context budget (was a fixed 8, query ignored).
- Same-page projection/dedup only for physically paged formats (PDF/images/
  slides); Markdown/DOCX artifacts all report page 1. Facets ignore projected
  sibling text. Direct table answers render only the table block (live run
  showed "[same-page native text context]" leaking into "Show Table 1").
- Removed every hardcoded paper alias (`_COMPOUND_ENTITY_PATTERNS`,
  `_SHORT_MODEL_NAMES`, `_topic_from_history` regex list, validator
  `_entity_aliases` map). Names now come from the catalog: filename words
  unique to one document and not recurring in other documents' content
  keywords, compounds of catalog words ("mamba fusion"), hyphen initials
  ("KS-Transformer"→kst), hyphenated content keywords ("pitch-fusion").
  Function words (cho, của, for…) excluded; conjunctions never compound.
- Facets: bare model/approach/layers are a weak architecture fallback only
  when no explicit facet is present; filenames are ignored for that check.
- Cache **v34**; cache key now includes visual/rerank settings.
- Backend **1,223 passed**. No-LLM gates vs Codex main-role baseline: local
  60 A0 **identical** (0.97/0.90/1.00, Hit@1/3/5 0.66/0.86/0.92, zero changed
  cases); public 36 **0.4028 = baseline** (one tie-case swap: NLI Table 1 vs 2
  both answer the question; one gain on "state-of-the-art").
- Live Sol dev-20 (P0+P1a code): **12/12 pass** before 9router returned
  `429 usage limit` at turn 13; evaluator stopped without fallback. Manual
  read of the 12 answers: grounded and natural; mixed-paper table now labels
  rows per paper. Issues: marker leak (fixed since), and numeric answers are
  buffered for validation → first visible text 7.5–13.7 s.
- NEXT: when quota allows, run dev02/dev04/dev05 on v34 code
  (`--case-id dev02-t2 --case-id dev04-t4 --case-id dev05-t4`), then
  streaming-with-validation design, then sealed multi-format eval.

## 2026-10-08 (Claude takeover) — P0 fail-closed validator and router detector repairs

- Independent review (Claude, 4 parallel reviewers + reproduced probes) found
  the uncommitted validator FAIL-OPEN versus HEAD: four wrong-owner numeric
  claims (cross-metric reference under a section owner; `- Model: ModelB`
  label body) were accepted via the unique-value fallback. Also accepted:
  `w/o`/`không dùng` conditions, unqualified `CCC 0.42` (dominance), `F_1`.
  All now rejected; matching correct-value controls stay supported.
- Rules (general, no paper names): ownerless fallback must agree with the
  answer section owner in the same document (cross-paper caveats still
  allowed); list-label value after ':' owns sub-bullets only if it starts with
  a known evidence owner; one negation lexer (w/o, without, removing, không
  dùng/có, bỏ, loại bỏ…) beats positive cues; unqualified claim is ambiguous
  when the owner reports several qualified values and the text names none;
  LaTeX metric names normalised. Removed unaccented `dat` cue (= "đạt").
- Router/rewrite detectors: bare model/mô hình/design no longer add the
  architecture facet; explanation questions are no longer result-table asks;
  research-topic words (multimodal, vision, visual grounding, image-text,
  "show me the accuracy") are not pixel requests; weak cues (người/text/từ/
  trang/document) need a media scope. Reference parser: "mô hình 2" is not
  Figure 2; Roman labels upper-case I/V/X only; `Fig. 2a` → Figure 2.
- Removed corpus-specific obligation aliases (creators/video views).
  `pyproject` pythonpath now includes repo root, so the documented
  `cd backend && .venv/bin/python -m pytest -q` works again.
- Backend **1,210 passed** (57 s). New `tests/test_validator_fail_closed.py`
  (19, red→green). Running :7790 backend still serves PRE-P0 code; no live
  replay yet. NEXT: P1 context selection (see 05-TODO "Claude review").

## 2026-10-08 KST — primary result evidence and runtime-only evaluation scope

- Separated evidence gathering from direct-table rendering. Analytical benchmark
  comparisons can retrieve active canonical main-result tables outside vector
  top-k; generation is still required for interpretation. Own-caption purpose
  separates main results, ablation and statistics. Explicit artifact numbers and
  metric coverage retain precedence; companion figures remain available. Cache
  **v31**, no paper/table-number mapping, embedding or index change.
- Nine generic selection/gathering fixtures include six red-before-repair
  failures; focused selection/context **130 pass**. Final backend **1,168 pass**
  (45.93 s; six dependency warnings), evaluator **29 pass**, desktop build pass.
- Same public 36-case raw-helper diagnostic **0.375 -> 0.40278**, one improvement
  and zero regressions, zero LLM calls/catalog mutation. Twenty cases still have
  zero selected-artifact recall. This is NOT actual-image/answer/unseen quality.
  Report: public/results/public-agent-context-main-role-20261008.json.
- Full latest agent source Sol dev-20 **20/20 existing rubric**, median total /
  first visible **10.098 / 4.106 s**, zero tool/transport/guard fallback and zero
  guard-reported unsupported claims. Actual selected evidence for dev05-t4 now
  includes MSF-SER Table 3 (IEMOCAP), and the answer preserves its CCC values
  alongside separately owned Mamba Test1/Test2 rows. Report:
  internal/agent-main-role-dev20-20261008.json; source/index fingerprints frozen.
- Manual review is NOT a semantic pass: dev01-t4 places a mixed-paper results
  table below an ASPIRE/IEMOCAP label even though its rows identify other papers
  and datasets. Preserved SHA-bound manual audit and clean structural audit:
  internal/agent-main-role-{manual,structural}-20261008.json. Empty-qrel dev rubric
  and numeric validation do not prove all qualitative claims or complete scope.
- Discovered evaluator gold-scope leakage and unordered final paper reservation:
  final selection was passed `resolved_case.document_ids` (frozenset). Two
  adversarial scope/order tests failed before repair. Selection now receives
  only runtime focus/catalog scope; qrels are scoring-only. Five recent old
  retrieval reports invalidated in evals/retrieval_v2/invalidated_reports.json;
  any report using that old contract also requires replay before promotion.
- Fresh serial runtime-scope local 60: A0 candidate/evidence/document recall
  **0.97/0.90/1.00**, text/table/figure **0.8333/1/1**, multi-paper **1**, P50/P95
  **124.7/341.8 ms**, 60/60 resolved, zero scope leak/model/rewrite calls. New
  report explicitly marks `qrel_scope_used_for_selection=false`:
  internal/retrieval-v22-runtime-scope-main-role-20261008.json. A1/B-full provide
  no gain/dual activation; defaults remain off. Do not use earlier invalidated
  runs as a promotion baseline or compare their hash-dependent ranking metrics.
- Graphify **4,789 nodes / 13,854 edges / 176 communities**; Obsidian **4,895
  notes** synchronized. Existing zero-node JSON, skill/package version and hub
  label warnings retained. Backend isolated :7790 **21260 / PID 24012**; frontend
  :5173, gateway untouched. All benchmark/test handles completed. No production
  migration, dual/visual promotion or shadow opening; full goal remains active.
- NEXT: general parent/reference -> artifact traversal on implicit public asks,
  clearer mixed-paper scope/completeness checks, wider actual-image/format
  semantic evaluation and frozen RC/unseen shadow. Legacy clone SSE does not
  substitute for arbitrary-format lifecycle/revision acceptance.

## 2026-10-08 KST — facet-role selection and typed document ownership

- Own-facet artifact reservations now prefer dataset/count tables for dataset
  requests and architecture figures for architecture requests. A bare task noun
  inside a performance/evaluation context no longer adds a task/contribution
  obligation. Exact task requests still do. Cache **v30**; no paper/case mapping.
- Same fixed public 36-case production-helper frontier: selected artifact recall
  **0.1944 -> 0.375**, **7 improved / 0 regressed**, P50 **206.6 ms**. Catalog
  unchanged, zero LLM calls. This is raw-query/context selection, NOT full-agent
  answer accuracy, actual uploaded-pixel recall, unseen acceptance or promotion.
  Report: public/results/public-agent-context-facet-role-20261008.json.
- Audited local 60/four-arm rerun: A0 candidate/evidence/document recall
  **0.97/0.90/1.00**, text/table/figure **0.8333/1/1**, multi-paper **1**,
  P50/P95 **130/370 ms**, 60/60 qrels resolved, zero scope leak/model calls.
  B-full remains inactive with no gain; dual-query stays off. Report:
  internal/retrieval-v22-facet-role-20261008.json. Frozen rewrites do not test
  runtime conversation rewriting or final prompt completeness.
- Numeric bundles under arbitrary summary headers preserve row owner/split;
  namespaced numeric schemas retain precedence. Title fragments are allowed
  in structural headings/declared owner cells, not incidental dataset/prose
  cells that could reassign a section to another paper. A generic adversarial
  fixture failed before this repair and passes after it; explicit other-paper
  row labels and scoped insufficiency rows remain supported.
- Full Sol dev-20 **20/20 rubric**, median total/visible **8.822/3.158 s**,
  zero transport/tool/model fallback. Manual audit caught a hidden wrong-section
  table in dev05-t4 despite green numeric/structural checks. That full run
  predates the final title-fragment fix; do NOT call it a final semantic pass.
  Preserved report and SHA-bound manual finding: agent-facet-role-dev20 and
  agent-facet-role-manual-20261008.json under internal/.
- Final-source affected Sol replay **4/4 rubric**, comparison one attempt,
  no fallback and supported numeric claims; observed Mamba table now in the
  correct Mamba section. Median total/visible **10.695/5.535 s** over four turns.
  MSF-SER main IEMOCAP benchmark source is still missing; partial ablation
  results/history do not establish comparison completeness. Report:
  internal/agent-title-fragment-replay-20261008.json. No final full-20 rerun.
- Final backend **1,157 passed** (44.56 s; six dependency warnings), **191**
  validator/guard focused pass; desktop build pass. Structural audit now surfaces
  empty data rows for review (not automatic failure for legitimate group rows).
- Graphify **4,779 nodes / 13,824 edges / 180 communities**; Obsidian export
  **4,885 notes**. Existing 13 zero-node JSON/version/hub-label warnings remain.
  Backend isolated :7790 **22252 / PID 19811**, frontend :5173 reachable earlier
  this interval; gateway healthy and untouched in this interval. Active index
  unchanged; dual/visual/LightRAG off. Production unmodified; goal still active.
- NEXT: main-results vs ablation/parent-artifact relationship selection, wider
  actual-image/arbitrary-format semantic quality, then frozen RC/unseen shadow.
  Public helper **0.375** is progress, not adequate acceptance.

## 2026-10-08 KST — artifact purpose, provenance and quantitative scope repairs

- Tables are not classified as benchmark evidence solely because they are
  tables or their captions mention a benchmark; use their own purpose/schema,
  not expanded neighbouring prose. Raw task facets survive decomposition and
  accumulated context selection. Generic fact questions do not acquire an
  invented abstract/contribution obligation. Retrieval cache **v29**.
- Same-page artifact injection copies document/page provenance only, preserves
  its own revision/index/content, and updates the correct original anchor;
  sibling artifacts cannot inherit expanded text, parent IDs or facet labels.
- Numeric parsing preserves each CCC dimension occurrence, separates declared
  composite model/split cells, excludes model-family buckets from experiment
  conditions, and accumulates owner schemas across all excerpts of a document.
  Descriptive bullet bodies cannot become persistent owners. Explicit
  cross-metric references are still numerically checked, without inheriting the
  previous model section; unknown owners/values and additional score assertions
  remain rejected by adversarial fixtures. No paper-name exceptions.
- Prior full suite **1,137 passed** (88.84 s) predates the latest scope/reference
  repairs. Latest four validator/guard suites **185 passed** (1.71 s); full suite
  and fresh HTTP/SSE replay of the latest source are NOT completed yet.
- Offline replay of the old failed comparison draft against active canonical
  tables now supports **34/34 claims**, zero unsupported. This is a diagnostic
  using preserved draft/canonical evidence, NOT proof of the final live prompt
  or a new E2E pass. Earlier affected four-turn run is **4/4 rubric**, but its
  structural audit still finds an empty results table. Keep that failed report.
- Last completed full Sol dev-20: **20/20 existing rubric**, median total/visible
  **9.974/3.362 s**, zero transport/tool/model fallback. Hidden numeric failures
  persisted; qrel-empty rubric is not semantic acceptance. Last local audited
  retrieval: A0 candidate/evidence/document **0.97/0.90/1.00**, P50/P95
  **138/416 ms**. B-full has no activation/gain; dual-query remains off.
- Same public 36-case raw-helper diagnostic regressed **0.2222 -> 0.1944**
  selected-artifact recall, P50 **226 ms**. No model calls/catalog mutation;
  NOT actual-pixel/full-agent quality. Do not promote or call this a win.
- Backend connection refusal was verified after the overnight interruption;
  same isolated 21-paper root restarted as **12305 / PID 3163**, :7790.
  Router catalog reachable with Sol now; earlier completion/stream pass does
  not substitute for a fresh generation. Production unchanged. AST graph sync
  **4,762 nodes / 13,781 edges / 180 communities**, existing warnings retained.
- NEXT: fresh latest-source Sol replay with semantic/structural audit, repair
  public final-artifact selection without gold-guided queries, and finish
  approved actual-pixel/arbitrary-format quality and unseen RC/shadow gates.

## 2026-10-08 KST — facet-directed rewrite and schema-preserving table projection

- Explicit scoped dataset/training/limitation/contribution requests use shared
  facet vocabulary instead of blanket introduction/architecture expansion.
  Dataset + result questions retain both facets. Auto mode can assess a single
  scoped facet gap, but adds a hop only for a proven structural gap; fast and
  unknown-metadata behavior remain bounded. Seven unrelated VI/EN fixtures
  failed before the fix; coverage/retry tests include sufficient/unknown cases.
- Table projection now selects schema/condition/row bundles across repeated
  headers, dataset groups and multiple Markdown blocks. It retains original
  order; never puts a regression row under a classification header. Three
  failing synthetic fixtures expose the prior bug; projected CCC rows validate
  while a deliberately wrong accuracy label still fails. Cache **v27**.
- Audited 60-case/four-arm retrieval rerun completed before the final table
  patch: A0 candidate/evidence/document recall **0.97/0.90/1.00**, modalities
  **0.8333/1/1**, multi-paper **1**, P50/P95 **120/334 ms**, zero scope leak
  and model calls. Same primitive ranking as before, no gain claimed. Frozen
  rewrites are not a runtime rewrite benchmark; final table projection is not
  proved by this earlier gate. `retrieval-v22-facet-query-20261008.json` internal.
- Final source snapshot full backend **1,129 passed**, five dependency warnings,
  **91.32 s**; desktop build pass; AST graph **4,738 nodes / 13,716 edges /
  175 communities**, with existing zero-node/hub-label warnings. A previous
  run collected an earlier synthetic model name (Target) and failed that test;
  the final fixture uses ORION to exercise actual named-owner binding.
- Sol HTTP/SSE full dev-20 is TERMINAL, **20/20 on the existing rubric**, zero
  transport/tool/model/claim-guard fallback. One turn has a suppressed block;
  one turn still requires numeric correction. Median total **9.211 s**, median
  visible delta **2.990 s**, max **28.839 s**. Source/index fingerprints verified
  unchanged after the run. `agent-facet-schema-dev20-20261008.json` internal.
  Do not treat the aggregate as causal latency/semantic acceptance.
- Dataset replay now retrieves the correct experimental table and names ViSEC,
  ASVP-ESD and Thai SER, not unsupported IEMOCAP/ESD. Result comparison retains
  CCC A/V as well as Acc/F1, validates in one attempt, **12.834 s** total. These
  illustrate repairs, not a broad held-out win. Earlier four-turn pre-table
  replay (also 4/4 rubric) still required correction and dropped CCC; preserved.
- Deep audit finds semantic failures despite green rubric and clean structural
  audit: **dev03-t2** describes JAIT as predominantly unimodal although active
  pages 1/8/9 describe four fusion approaches; **dev05-t4** places a Mamba
  training-statistics row under MSF-SER main results and lacks actual comparison
  result rows. Durable manual findings bind report SHA in
  `agent-facet-schema-dev20-manual-20261008.json`; reviewed seven cases only,
  not a full semantic pass rate. Existing rubric's qrel groups are empty.
- Public raw-query helper replay, same 36-case frontier, completes with no LLM
  calls/catalog changes: selected/potential-pixel artifact recall **0.2222**
  unchanged, P50 **209.5 ms**. It does NOT execute runtime rewrite/planner or
  upload pixels; do not claim it evaluates the facet rewrite or full agent.
  `public-agent-context-facet-schema-20261008.json` under public/results.
- Backend :7790 now owned session **13268**, PID **21091**, same isolated root;
  earlier 34197 is stopped. Frontend :5173 and
  9router unchanged. Dual-query/visual remain off; production unchanged.
- NEXT: artifact-purpose and whole-document facet/semantic coverage, especially
  missing fusion/main-results and post-suppression section binding; then public
  actual-agent/pixel quality. Numeric reference-vs-assignment parsing still
  requires an adversarial fix, not a negation whitelist. No RC/shadow acceptance.

## 2026-10-08 KST — live gateway recovery and approved answer pixels

- Authenticated models and real native Sol streaming rechecked: catalog pass,
  exact `9router-stream-ok`, first delta **1.405 s**, total **1.518 s**.
  A diagnostic harness initially used the wrong StreamChunk field (`delta`);
  it was corrected to `content`. That harness exception was NOT a router failure.
- Earlier in this work interval the gateway really hung: catalog/dashboard
  timed out and a tiny Sol completion timed out while its node consumed CPU.
  After verified failure, restarted the same installed 9router **0.5.95** with
  `--skip-update`; no account/model/config changes or model fallback. Current
  owned CLI session **80370**, node **14474**, :20128. Historical reports
  claiming the gateway was untouched apply to earlier intervals, not this one.
- Separated permission to send canonical answer images from the optional visual
  retrieval flag. SQLite-owned approval, active revision/artifact ownership,
  canonical path, stored digest (when available), actual decodable image bytes,
  bounded size and post-read approval recheck govern uploads. Attachment labels
  describe the same accepted bytes; traces store IDs/digests, never image base64.
  Visual intent now requires pixels actually supplied, not merely a UI path.
- Removed unrelated forced table inventory for generic implicit/numeric asks;
  explicit inventory keeps companion figures and injected revision identity.
  Retrieval cache is **v25**. RagService document/table/figure readers preserve
  revision identity in their returned evidence, not just SQL filtering.
- Public production-helper parity replay completes **36/36**, no model calls
  or catalog changes: selected artifact/potential-pixel recall **0.2222**,
  P50 **234.66 ms**. Prior helper diagnostic was **0.25**. NOT an improvement,
  semantic/pixel accuracy or acceptance. Report:
  `data/retrieval_eval/public/results/public-agent-context-inventory-fix-20261007.json`.
- Four-turn Sol HTTP/SSE replay after gateway restart: **3/4 rubric pass**,
  zero transport/tool/model fallback; comparison DOES use a claim-guard fallback.
  One architecture turn actually supplies an approved canonical figure while
  visual retrieval stays off. Report `agent-approved-images-after-router-restart-20261007.json`
  under internal eval. Keep the earlier transport-failure report separately.
- Deep trace audit prevents interpreting the 3/4 as semantic correctness:
  dataset turn invents evaluated dataset names absent from supplied excerpts;
  rewrite retrieved introduction/architecture rather than evaluation dataset
  evidence. Numeric guard reports `no_metric_value_claims` and misses this.
  Comparison correction contains valid result rows, then explicitly warns that
  different dataset/metric values cannot establish superiority; validator
  misbinds the second mentioned number to the first paper's accuracy. Initial
  attempt also has unsupported dimensional/split claims; stored trace is partly
  truncated, so do not infer full prompt completeness from the trace alone.
- Final full backend **1,118 passed**, five dependency warnings, **92.13 s**;
  test-only stale visual-abstention wording assertions corrected. Desktop build
  passed during this interval. AST Graphify **4,728 nodes / 13,688 edges /
  180 communities**, 12 zero-node JSON warnings and hub-label regeneration remain.
- Backend :7790 session **40208**, PID **15274**, same isolated 21-paper root;
  frontend :5173 session **23843**, correct `VITE_BACKEND_URL` targets :7790;
  both health HTTP 200. Public backend :7791 is stopped. Production DB/index
  unchanged; approval metadata writes were ONLY in owned isolated clones.
  Dual-query/visual flags stay off. Goal remains active, no RC/rollout acceptance.
- NEXT: fix facet/relation coverage for dataset queries and numeric reference
  vs asserted metric binding with adversarial fixtures, preserving wrong-owner
  rejection. Then fresh wider agent/public actual-pixel evaluation; no prompt or
  paper-specific exemptions. Canonical image page/caption labeling and broader
  visual-page authorization also need audit before any visual promotion.

## 2026-10-06 — revision-bound VLM context and completed native public-60

- Previous turn was a verified service/process check: authenticated Sol
  completion HTTP 200, `9router-ok`, **1.83 s**, backend/frontend HTTP 200.
  It was not a provider outage; do not restart user-owned 9router/Ollama.
- Public retrieval session **57284** is now terminal (exit 0), **60/60** rows,
  zero ingestion failures/model calls. Report:
  `data/retrieval_eval/public/results/public-foundation-dev-v2-native-20261006.json`.
  Candidate artifact recall **0.9306**, final-context artifact-name recall
  **0.5833**, evidence phrase recall **0.8380**, candidate page recall **0.75**,
  any-gold-page hit **0.7143**, warm P50 **135 ms**, maximum **429 ms**.
  Filename/phrase/any-page scores are diagnostics, NOT semantic/pixel coverage.
  Same frontier as the thumbnail context-fix run: candidate artifacts rose
  **0.9028 -> 0.9306**, final artifacts/page score did not improve and phrase
  recall fell **0.8611 -> 0.8380**. Different input fidelity, not a ranking win.
- Read-only post-run lifecycle check: all **48** documents have valid active
  revision ownership/status, no staged or multiple-active revisions; **255**
  official active image digests match stored SHA-256. This is not a full
  FTS/Lance/RC acceptance proof. Report `public-native-active-image-audit-20261006.json`.
- Added postprocessor `scripts/audit_public_retrieval_frontier.py`: no model,
  query, ranking or index changes. Full boundary audit finds **13 cases** with
  a candidate artifact lost from final context (all implicit artifact asks),
  three candidate-artifact misses, seven candidate-page misses and three
  final-page losses. Annotation markers require review; parent expansion
  recovery is not automatically labeled a serialization bug. A filename hit
  cannot prove supplied pixels. `public-native-frontier-boundary-audit-20261006.json`.
- Fixed FigureEnrichService's real revision-context mismatch AFTER the frozen
  public run ended. Selection now carries `revision_id`; chunks/tables/cards
  bind to that figure's revision; staged cards cannot borrow an active summary;
  page images resolve from the selected asset, not a different materialization.
  Completion is revision-owned, rejects stale cached revision tags and cannot
  mark a newer active revision after a slow VLM call or a partial batch.
- Before/after read-only audit on the same isolated 21-paper clone (**136**
  active figures): missing same-revision section **121 -> 0**, missing nearby
  table context **40 -> 0**, missing returned revision identity **136 -> 0**.
  Reports `figure-context-revision-audit{,-after}-20261006.json` under internal.
  No summaries/images/indexes were changed by this audit. This proves context
  wiring repair, not an answer-quality improvement.
- Real Sol VLM smoke used a separate public clone and the first two authored
  assets of the first public document by filename (not gold-selected): **2/2**
  enriched, **33.73 s**, no fallback; persisted revision/model match, image
  digests and active revision unchanged. `figure-revision-live-smoke-20261006.json`.
- **75 focused / 1,092 full backend tests passed**, desktop build passed.
  AST graph synchronized **4,689 nodes / 13,571 edges / 180 communities**;
  existing 12 zero-node JSON warnings and hub label regeneration remain visible.
  Two-turn Sol HTTP/SSE dev smoke **2/2** on existing rubric, focus preserved,
  native deltas/one grounded architecture image, zero fallback/suppression.
  Totals **23.74/13.65 s**, visible deltas **6.08/10.29 s**; second rewrite
  took **8.20 s**. This is a smoke, not a broader quality or latency win.
- Backend :7790 reloaded on the SAME isolated 21-paper root: session **27739**,
  PID **46224**, health HTTP 200; old **15074/PID 37848** is stopped. Frontend
  :5173 HTTP 200. Production unchanged; dual-query/visual flags remain off.
- NEXT: repair implicit artifact selection/context projection using query,
  canonical content and provenance, not benchmark gold filenames/answers.
  Then measure paired retrieval + actual supplied-image answer quality. Public
  semantic completeness, arbitrary-format quality, visual/unseen RC gates and
  wider agent regression remain incomplete. Goal stays active.

## 2026-10-06 — native image inputs and alias revision rebuild

- Reverified authenticated 9router models and actual Sol completion: HTTP 200,
  `9router-ok`, **1.88 s**. Backend :7790 and frontend :5173 are reachable.
  Initial sandbox loopback failure is not a provider outage.
- SPIQA adapter was reading the optional 224px ZIP, not native image evidence.
  All 48 authored images in the six frozen SPIQA probe documents were distorted
  to 224x224. The official native test-C archive was downloaded at the existing
  pinned revision and checksum-verified (112,026,645 bytes; SHA-256 `e7978cf1...`).
- Reader now fails closed when native images are absent; no thumbnail fallback.
  Probe contracts label the input policy and record actual supplied dimensions,
  bytes and SHA-256. All authored images, not gold-selected assets, participate
  in source identity. Historical reports remain thumbnail-condition
  diagnostics, not original-resolution acceptance or a measured ranking win.
- Same local RapidOCR: the inspected table's native **970x312** image yields
  legible headers/rows; 224x224 yields gibberish. Upsampling thumbnails does not
  recover the native information. OCR was already enabled, not missing.
- Native 12-case / 36-generation Sol probe **completed**, session **34901**
  terminal. No aggregate Aya answer improvement: SPIQA diagnostic token F1
  **0.1666 -> 0.1598**; MMLong score **0.6603 -> 0.4936** even though those
  inputs were unchanged. This is one stochastic diagnostic, not causal proof
  of a ranking regression or production acceptance. Language inventory still
  receives no images; humor comparison still selects a different table than
  required. Report: `public-answer-probe-native-20261006.json`.
- SQLite audit additionally found **six** native probe documents had their
  image bytes changed under the old active revision: the new-root source alias
  bypassed the reindex policy in a second raw-byte dedup lookup. Kept this probe
  as input/answer diagnostic but **invalid for lifecycle acceptance**. Separate
  `public-answer-probe-native-lifecycle-audit-20261006.json` preserves the failure.
- Rebuild now carries an alias's logical document ID into both stage lookups;
  changed policy creates a staged revision instead of returning the old one.
  SPIQA sync independently rejects changed active bytes/captions/membership
  before writing an image. Same-byte derived enrichment stays refreshable.
  Added fresh/existing alias tests and real-Lance crash/retry scenarios.
- Post-repair public-60 retrieval-only session **57284** is live, on a new
  durable clone `public-native-lifecycle-20261006/embed-661e529d954a`; no LLM calls.
  Latest checkpoint **3/60**, no failed rows, heartbeat advances while the next
  document OCRs 15 native table images. Three replaced documents visibly retain
  retired and active revisions; the in-flight document retains active+staged.
  Output pending: `public-foundation-dev-v2-native-20261006.json`.
  Input audit: `public-spiqa-input-resolution-audit-20261006.json` under public
  results. Final focused lifecycle/public **113 passed**, full backend
  **1,065 passed**, plus both real-Lance alias/non-alias crash variants pass
  after extending that test (full suite not rerun after the test-only addition).
  Public source full validation passes 20 checksums / four ZIPs.
- AST graph synchronized **4,657 nodes / 13,478 edges / 163 communities**;
  12 zero-node source warnings and hub label regeneration remain visible.
  Dual-query/visual flags stay off. Public selection/semantic completeness,
  arbitrary-format acceptance, RC and shadow requirements remain incomplete.
- Backend restarted on the SAME isolated 21-paper root to load the tested
  lifecycle fix: :7790 session **15074**, PID **37848**, health HTTP 200.
  Previous 63710/PID 27221 is terminal; frontend :5173 HTTP 200. Production and
  user-started 9router/Ollama were not modified or restarted.

## 2026-10-06 — condition identity, adversarial validation and Gate C rerun

- 9router authenticated models and actual Sol completion pass (HTTP 200,
  `9router-ok`, **1.94 s**). Sandbox loopback denial is not provider outage.
- Fixed schema-driven loss/config vector alignment, encoder/backbone/SSL owner
  roles, explicit owners inside Condition cells and table-local blank owner
  continuation. Unknown owners and wrong dataset/loss/encoder combinations
  remain rejected; ambiguous condition vectors cannot authorize a named loss.
- Adversarial fixtures exposed and repaired positive-vs-negated configuration
  acceptance. Source-declared negative conditions retain polarity. Model/config
  classification works across documents without looking up owners by value.
- Loss-only tables no longer manufacture loss names as model owners. Explicit
  missing-context cells are not experimental conditions; training hours/minutes
  cannot be parsed as a subsequent accuracy metric. Parameter claims still
  require evidence. Uppercase subscript owners now have one canonical identity.
- Audited retrieval-v2.2 rerun completed **60 cases / four arms**, no model or
  rewrite calls, zero unresolved qrel/scope leak, on the isolated clone. A0:
  candidate/evidence/document recall **0.97/0.90/1.00**, text/figure/table
  **0.8333/1.00/1.00**, multi-paper **1.00**, P50/P95 **119/348 ms**. A1/B-full
  add no recall and B-full activates no distinct-query branch; keep dual off.
  Report: `retrieval-v22-condition-binding-20261006.json` under internal eval.
- First five-turn Sol replay is **5/5 on its existing rubric**, NOT semantic
  acceptance: h03-t4 and h15-t1 still have empty tables. Comparison takes
  **101.75 s** with a correction. Preserved report and structural audit:
  `agent-condition-binding-{replay,completeness}-20261006.json`. Saved-draft
  revalidation after the final schema fixes retains the valid encoder rows;
  incompatible loss assertions remain rejected rather than guessed.
- Final focused suite **177 passed**, full backend **1,058 passed** (five
  dependency warnings), desktop build pass. Fresh four-turn SSE replay completed
  **4/4 rubric**, zero model/tool/transport fallback, clean structural audit.
  Comparison now retains four GBAN and two CRAB result rows, **12 supported
  claims**, one attempt, **44.50 s** total/first-visible. This is one visible
  replay, not a latency win or semantic acceptance: its final conclusion covers
  GBAN only and CRAB evidence still emphasizes ablation rather than main results.
  Reports: `agent-condition-binding-final-{replay,completeness}-20261006.json`.
  Backend :7790 session **63710**, PID **27221** serves the same isolated clone;
  frontend :5173 remains available. Production unchanged. AST graph synchronized
  **4,650 nodes / 13,455 edges / 178 communities**, 12 zero-node JSON warnings
  and hub-based label regeneration remain visible.
- Public arbitrary-format/page/implicit-image quality remains below acceptance.
  Explicit missing-artifact behavior and answer semantic completeness remain
  open. No frozen RC, shadow-v3 opening or production/feature promotion.

## 2026-10-06 — wide-table owner and intrinsic-metric validation verified

- The six-turn affected-family replay finished **4/6**, not a pass. h03-t4
  still fails document coverage after guard retries; h15-t1 fails the existing
  abstention rubric and contains an empty table. h06-t1 passed that rubric but
  also had an empty result table. Preserved report and separate structural
  audit: `agent-validator-matrix-{replay,completeness}-20261005.json`.
- Fixed pivoted tables whose model names are in metric headers, with dataset
  rows carrying measurement conditions. Explicit model columns still win;
  unlabelled condition-header legacy tables retain their row owners. Synthetic
  owner swaps/dataset swaps fail closed. No paper-specific table rule.
- Actual ViSEC prompt/tool evidence contained the complete Table 2, but WA
  source metadata acquired redundant `weighted` conditions, incompatible with
  the answer's extra language annotation. Canonical WA/UA/WAR/UAR intrinsically
  encode that weighting. Matching now omits only that redundant condition;
  those four metric identities remain distinct and wrong-metric tests fail.
- Exact saved-table replay checks all **12** results successfully. New real Sol
  HTTP/SSE replay retains **all three dataset rows / twelve UA-WA values**,
  **13 supported claims**, zero suppressed blocks, zero transport/tool/model
  fallback. Total/first-visible **29.60/9.04 s** (one case, not a latency gate).
  `data/retrieval_eval/internal/agent-intrinsic-metric-replay-20261006.json`.
  Its structural audit is clean; that is not full semantic acceptance.
- Full backend **1,030 passed** (five dependency warnings); desktop build pass.
  Graphify AST **4,616 nodes / 13,352 edges / 171 communities**; 12 zero-node
  JSON warnings/hub label regeneration remain visible.
- After interruption backend/frontend listeners were gone and were restarted
  automatically. Current backend :7790 session **65772**, PID **17180**, isolated
  21-paper root unchanged; frontend :5173 session **85711**. Existing 9router
  models/completion pass; direct Sol completion **1.97 s**, no router restart.
- Remaining: generic model-vs-condition binding (loss/config acronyms, owners
  inside Condition cells, flattened rowspan tables), semantic/negative-artifact
  completeness, public implicit image/page evidence misses. Public acceptance
  is still failing; flags off, production unmodified, shadow-v3 unopened.

## 2026-10-05 — completed baselines, public quality boundaries, validator repair

- Latest direct 9router check: models HTTP 200, Sol completion HTTP 200 in
  **1.48 s**. A sandbox loopback denial is not an unavailable-provider result.
  Backend :7790 serves the isolated 21-paper clone (current session 49899,
  PID 3287); frontend :5173 remains pointed at this clone. Production untouched.
- Public visible frontier **60/60** completed (48 documents) before and after
  artifact-context repairs. Final phrase recall **0.7870 → 0.8611**, but final
  artifact recall **0.6111 → 0.5833**; candidate/final page recall remain
  **0.7500/0.7143**. These are not passing arbitrary-format acceptance results.
  Reports: `public-foundation-dev-v2-20261005.json` and
  `public-foundation-dev-v2-contextfix-20261005.json` under public/results.
  Artifact recall includes filename mentions; it is not pure pixel coverage.
- Both frozen 12-case three-mode answer probes completed **36/36 generations**.
  MMLong Aya score stays **0.6603**; SPIQA diagnostic token F1 **0.1793 → 0.1666**.
  No uniform quality improvement is established. Pixel telemetry confirms two
  remaining SPIQA misses: implicit language inventory receives no images;
  an implicit numerical comparison receives the wrong table. Probe bypasses
  agent planner/claim guard; bounded extracted full-document mode is NOT
  ChatGPT upload. Gold/full scores also vary. Reports:
  `public-answer-probe-{baseline,contextfix}-20261005.json`.
- Agent visible regression completed **59/60**, zero transport/tool/model
  fallback; median total/first-visible **11.98/5.12 s**. Dev-20 is **20/20** on
  its existing rubric, but separate audits still identify empty result tables
  (dev05-t4, h06-t1, h15-t1). The previous four-turn h03 replay passed its rubric
  while still leaving an empty GBAN table; that was NOT a semantic fix.
- Fixed general validator schema bugs: explicit row metrics now override stale
  caption inference (split/fold counts no longer become WA); caption-declared
  transposed condition × owner matrices retain per-component and row-condition
  binding, including PDF-flattened subscript labels. Synthetic cross-owner and
  fold-vs-average substitutions fail closed; the four original GBAN averages
  validate against canonical source rows without loosening number guards.
- Full backend suite **1,028 passed**, five dependency warnings; AST graph
  **4,614 nodes / 13,347 edges / 170 communities**, 12 zero-node JSON warnings
  and hub-based community renaming. Six-turn HTTP/SSE affected-family replay is
  currently running (session 2086), not yet an acceptance result.
- Dual-query/visual remain off; shadow-v3 unopened. Public page/figure quality,
  semantic completeness, frozen RC and rollback acceptance remain outstanding.

## 2026-10-05 — live service recovery and durable agent evaluation

- 9router is now reachable: authenticated models HTTP 200 and actual Sol
  completion HTTP 200 (`9router-ok`, latest recheck 1.43 s). Earlier connection
  refused observations below are historical, not a current blocker.
- Created a durable read-only-source snapshot of all 21 documents at
  `data/retrieval_eval/internal/runtime/e2e-foundation-20261005` using SQLite
  online backup plus independent Lance/artifact copies. Copied pending jobs
  are suppressed only in the clone. Backend :7790 and frontend :5173 are live;
  frontend points to the clone, not production :7777. No production migration.
- Agent evaluator now atomically checkpoints every turn and records in-flight
  run/conversation IDs. Resume requires identical cases/options/backend code/
  canonical projection/image bytes, intact HTTP history and matching persisted
  answers. Quality failures remain in the results. In-flight/failed provider
  turns reject silent replay, and failed runs stop without model fallback.
- Verified one actual SSE turn and a read-only resume: preserved original
  14.10 s total / 4.44 s first visible delta, no second Sol generation. Focused
  tests 31 passed; full backend **1,010 passed** (five dependency warnings).
- AST graph synchronized: **4,584 nodes / 13,239 edges / 167 communities**;
  12 JSON zero-node warnings and regenerated community names remain visible.
- Public visible dev frontier is now 60 cases / 48 distinct documents, disjoint
  from previous dev and reserved shadow document families. The original v1
  selection is invalidated because it failed to resolve prior v2 dataset IDs;
  v2 selection resolves both releases and fails closed on unknown IDs.
  Public session 17730 remains running with production OCR and checkpoints:
  `data/retrieval_eval/public/results/public-foundation-dev-v2-20261005.json.partial`.
  Agent dev-20 session 65974 completed **20/20** on its existing rubric, median
  total/first-visible **11.83/4.15 s**, zero transport/tool fallback. Report:
  `data/retrieval_eval/internal/agent-dev20-foundation-20261005.json`.
  A separate structural audit identifies **one** incomplete answer (empty table
  and dangling final heading): `agent-dev20-completeness-20261005.json`.
  The rubric pass is therefore NOT full answer-quality acceptance. The audit
  itself is diagnostic, not entailment or semantic task-completeness scoring.
  Regression-v1 60 (already visible, NOT shadow) is now live, session 69334:
  `data/retrieval_eval/internal/agent-regression60-foundation-20261005.json.partial`.
  Partial public/regression runs are not completed acceptance results.
- New synthetic fixture independently proves context composition drops one of
  two distinct figures sharing a page, even with `min_figures=2`. Do not change
  the running baseline mid-measurement. Fix artifact-vs-page dedup after the
  baseline completes, then run paired regression; retain existing failed cases.
- Dual-query/visual defaults stay false. Shadow-v3 is unopened and RC/arbitrary
  format acceptance remains incomplete; the persistent goal is not complete.

## 2026-10-05 — repaired revision-scoped authored artifact projection

- Verified nine SPIQA sources had canonical tables/figures in the active
  SQLite revision but zero table/figure vectors in the same Lance revision.
  Adapter IDs were revision-independent and warm reuse only checked text/card.
- Trusted local authored attachments now materialize before local projection
  validation/activation. Attachment fingerprints participate in revision/job
  identity; table/figure IDs and image paths are revision-scoped. Identical
  image bytes may reuse VLM summaries without moving/deleting prior revisions.
- Both coordinator/manual-vector activation and public warm reuse compare expected canonical vector
  identities with physical Lance IDs (using actual indexing eligibility, not
  counting rejected/page-placeholder figures as missing searchable vectors).
- Same 21 public dev cases completed with production OCR and no LLM calls:
  candidate artifact recall **0.50 → 1.00**, evidence phrase recall
  **0.7595 → 0.8738**, query-scoped artifact support **0.8333 → 1.00**,
  P50/max retrieval **135.12/620.61 ms** in the final frozen-code replay. Final artifact recall remains
  **0.50**; this visible slice does not close arbitrary-format or shadow gates.
  Report: `data/retrieval_eval/public/results/public-retrieval-revision-artifacts-verified-21-20261005.json`.
- Public/visual checkpoints now share atomic writes and exact code/config/
  selected-source contracts; source checksums include all authored image bytes,
  not only gold attachments. Legacy or changed-contract rows cannot resume.
- Final full backend suite **996 passed** (five dependency deprecation warnings),
  including legacy compatibility, staged-adapter crash and manual activation.
  Desktop production build passed. Graphify AST synchronized:
  **4,530 nodes / 13,102 edges / 167 communities**;
  nine JSON zero-node warnings remain visible.
- Earlier temporary backend/runtime was removed after interruption. The
  regression-60 run has no final report and is NOT a pass. Direct 9router
  recheck returned connection refused on both loopbacks; no listener at
  configured port 20128. Local embedding remains available. No production
  index mutation, shadow opening, dual-query or visual promotion occurred.

## 2026-10-04 — earlier acceptance snapshot (runs since terminated)

- Visual report merge/recompute/answer-quality attachment now share a
  fail-closed eligibility contract: missing original cases, OCR-disabled text
  control, legacy completion metadata and posthoc case filtering cannot become
  promotion evidence. Scored answer IDs must match the declared frontier.
- Full backend run: **991 passed**; the subsequent pairing-ID guard passed
  **25 focused tests**. Desktop production build passed. AST graph synchronized;
  Graphify warns that nine JSON inputs have no extractable nodes.
- Fresh read-only production snapshot was cloned to
  `/private/tmp/aya-e2e-policy-20261004`; SQLite/Lance/artifacts are separate and
  copied pending jobs were suppressed. Backend **:7790** serves this clone,
  with LightRAG/visual/dual-query disabled for the measured baseline.
- Audited Gate C **60 cases**: A0 candidate/evidence/document recall
  **0.97/0.90/1.00**, modality text **0.8333**, figure/table **1.00**,
  multi-paper **1.00**, P50/P95 **240/877 ms**, zero unresolved qrel,
  forbidden scope leak or model call. A0 acceptance checks passed; no dual-query
  promotion because B-full had no treatment activation. Report:
  `data/retrieval_eval/internal/gate-c-policy-20261004.json`.
- Sol HTTP/SSE dev regression **20/20** passed, no transport/tool fallback;
  reported final unsupported claims are empty. Median total/first visible delta
  **14.59/7.75 s** under concurrent public ingestion. This is not independent
  proof that every qualitative claim is correct. Report:
  `data/retrieval_eval/internal/agent-dev20-policy-20261004.json`.
- Separate seven-turn focus/history/figure regression passed **7/7**, with
  the image endpoint HTTP 200. Report:
  `data/retrieval_eval/internal/e2e-policy-20261004.json`.
- Public **21-case** production-OCR run subsequently completed; regression-v1
  **60-case** Sol was interrupted with no final result (see current snapshot).
  Shadow-v3 remains unopened; no production index migration or promotion occurred.

## 2026-10-04 — authenticated router check and lifecycle correctness

- Live authenticated `cx/gpt-5.6-sol` completion through configured 9router
  returned HTTP 200 and `9router-ok` in 5.35 s; model listing also returned
  HTTP 200 with Sol present. Sandbox `Operation not permitted` is not an
  unavailable-provider result.
- Materialization policy (including PDF OCR budgets/options) now participates
  in revision identity and durable folder-job identity. Isolated reuse checks
  compare immutable active-revision policy, not mutable document metadata.
- Backend suite: **982 passed**. Real isolated lifecycle matrix preserved all
  eight invariants and **6/6** retrieval hits; the intentionally corrupt PDF
  correctly kept the folder result partial. Report:
  `data/retrieval_eval/internal/lifecycle-policy-20261004.json`.
- Visual checkpoints are atomic, validate evaluator/config contracts, and
  expose batch progress. The main visual gate rejects incomplete frontiers
  and OCR-disabled text controls for promotion. Current CPU-only visual run
  remains diagnostic; no visual/public quality gate is claimed passed.
- Production indexes and both dual-query/visual defaults remain unchanged.


## 2026-09-24 — visual backpressure hardening, still non-promoted

- The visual path now respects an explicit page/slide hint before semantic
  ColSmol search, so a request such as ``page 7`` can resolve directly from
  the active-revision page projection rather than paying for a visual query.
- ColSmol page scoring is bounded to an exact per-page batch of 16, and visual
  rendering/worker batches are bounded by 2M/4M pixels. These are resource and
  backpressure safeguards for CPU-only cold indexing, not recall tuning and
  not a claim that Gate E passed.
- Backend suite passed **975/975** after the change; Graphify is synchronized
  (**4,491 nodes / 12,969 edges / 166 communities**). The two interrupted
  current visual-gate reports remain non-scoring, so
  ``VISUAL_RETRIEVAL_ENABLED=false`` remains the only honest setting.

## 2026-09-24 — current visual smoke boundary

- A fresh ColSmol smoke (`vidore/colSmol-256M`) was started against the
  current isolated text projection with 7 selected public cases. The worker
  loaded and created a visual index, but the first MMLong case exceeded the
  180-second visual-index budget and the runner aborted fail-closed: **0/1
  completed**, no recall was reported. Report:
  ``data/retrieval_eval/public/results/visual-gate-current-smoke-20260924.json``.
- This is a provider/cold-index latency limitation, not a retrieval score and
  not a reason to block local text/table search. Visual retrieval remains
  optional and disabled; no production index or feature flag was changed.
- A checkpoint resume later completed the first image-only case with T/V/T+V
  recall **1.0/1.0/1.0** after **165 s** of visual indexing, then was stopped
  before the remaining cases; the partial report is explicitly non-scoring.
  This confirms checkpoint reuse works but also confirms cold visual indexing
  is too slow for an interactive path.

## 2026-09-24 — local and enrichment worker lanes

- Folder/canonical processing and optional enrichment now use separate bounded
  durable lanes (one local + one enrichment worker). A cold ColSmol/VLM job
  can no longer consume the worker that activates parse → FTS → LanceDB for a
  newly submitted folder. SQLite lease/idempotency semantics are unchanged.
- Full backend suite passed **972/972**, desktop production build passed, and
  the isolated lifecycle matrix kept every dedup/revision/rename invariant
  plus all six expected retrieval hits. Report:
  ``data/retrieval_eval/internal/format-lifecycle-lanes-20260924.json``.
  This is a scheduling correctness change, not a visual promotion;
  ``VISUAL_RETRIEVAL_ENABLED`` remains false.

## 2026-09-24 — arbitrary-format lifecycle smoke

- Isolated PDF/DOCX/MD/TXT fixtures passed projection and retrieval contracts
  **4/4**. TXT, Markdown and DOCX were ``ready`` with expected retrieval hits;
  the image-only scan PDF was correctly ``partial`` and excluded from the
  text-hit denominator. Report:
  ``data/retrieval_eval/internal/format-lifecycle-smoke-20260924.json``.

## 2026-09-24 — audited Gate C dev-v2.2

- Audited 60-case retrieval evaluation completed with 50 positive and 10
  negative cases, zero unresolved qrels, zero scope leaks and zero model calls.
  Report: ``data/retrieval_eval/internal/gate-c-dev-v2.2-20260924.json``.
- A0 passed candidate/evidence/document recall **0.97/0.90/1.00**, modality
  text/figure/table **0.8333/1.0/1.0**, multi-paper **1.0**, and P50/P95
  **122.697/356.881 ms**. A1 and B-full were identical to A0.
- B-anchor reached evidence **0.91** and text **0.85**, but latency rose to
  P50/P95 **203.213/414.213 ms** without the required treatment gain. The
  evaluator recommends keeping dual-query disabled.

## 2026-09-24 — current-code Sol HTTP/SSE regression

- Replayed the seven-turn conversation against an isolated `:7792` backend
  using the current runtime clone and ``cx/gpt-5.6-sol``. Result: **7/7**, all
  runs completed, focus survived a casual interlude and paper switch, table
  evidence was validated, and the Figure 1 image endpoint returned HTTP 200
  with bytes. Report:
  ``data/retrieval_eval/internal/e2e-agent-conversation-v6-20260924.json``.
- The isolated backend was stopped after the run; production data and feature
  flags were untouched.

## 2026-09-24 — shadow-v3 sealed preflight

- Shadow-v3 validates structurally: **24 questions**, 23 positive qrels plus
  one abstention, six format/modality buckets, and source fingerprints
  present. It remains sealed (`opened=false`) and was not run.

## 2026-09-24 — candidate/final replay and heuristic rollback

- Replayed the same isolated 21-case public frontier after a generic metric
  projection experiment. The experiment reduced final evidence phrase recall
  from **0.8595** to **0.8024**, so it was removed; no paper-specific rule or
  production index change was retained.
- The authoritative post-rollback report is
  ``data/retrieval_eval/public/results/public-retrieval-artifact-v11-reverted-context-20260924.json``:
  candidate page recall **1.000**, candidate evidence recall **0.9833**,
  candidate artifact recall **1.000**, final evidence phrase recall **0.8595**,
  query-scoped artifact support **1.000**, support recall **0.9500**, retrieval
  P50 **109.988 ms**, maximum **540.858 ms**, 21/21 successful.
- This confirms candidate generation is strong on this diagnostic frontier;
  final-context/answer completeness remains the open Gate E issue. Both visual
  retrieval and dual-query flags remain disabled. 9router Sol and Ollama
  EmbeddingGemma provider checks both returned successfully.

## 2026-09-24 — artifact identity/fusion correction and v5 frontier

- Fixed generic structured-evidence selection: inherited page context no longer
  masquerades as artifact lexical overlap; table/figure image aliases are
  deduplicated before ranking, while canonical table/figure identity is kept.
  Visual query markers and comparison/evaluation morphology are recognized
  without paper-specific rules.
- Isolated public frontier completed **21/21** with zero failures using the
  current EmbeddingGemma index. Aggregate page recall is **1.000**, first-stage
  candidate page recall **1.000**, evidence phrase recall **0.8595**, final
  context artifact recall **0.5000**, first-stage candidate artifact recall
  **1.000**, and retrieval-support recall **0.9500**. Warm retrieval P50 is
  **111.43 ms**, maximum **563.76 ms**.
- Report: ``data/retrieval_eval/public/results/public-retrieval-artifact-v6-candidate-21-20260924.json``.
  This changes the diagnosis: canonical table/figure rows are found in the
  candidate frontier; the remaining 0.50 is mostly final-context modality
  projection/attachment, not candidate generation. Evidence recall is near
  the gate but table/figure answer support still needs explicit stratified
  validation. Visual/dual-query flags remain disabled.
- Full backend suite after the correction: **970 passed**, 5 dependency
  deprecation warnings. Graphify was synchronized after the code changes.

## 2026-09-24 — authenticated Sol HTTP/SSE regression

- Live backend health confirms gateway and Ollama are reachable; answer/VLM
  route is ``cx/gpt-5.6-sol`` through 9router, with no model fallback.
- Seven-turn conversation regression completed **7/7** with no transport,
  tool, model or evidence-validation errors. It covered a paper answer,
  casual interlude, table follow-up, cross-paper switch, contribution
  follow-up and figure/table requests while preserving document focus.
- Figure attachment returned HTTP 200 and table turns carried validated
  evidence. Report:
  ``data/retrieval_eval/internal/e2e-agent-conversation-v4-20260924.json``.
- This validates agent wiring and current provenance behavior only; the
  retrieval promotion gate remains open because public artifact/evidence
  recall is still below threshold.

## 2026-09-24 — post-correction E2E replay

- A fresh isolated backend replay against the corrected retrieval code passed
  **7/7** turns over HTTP/SSE with ``cx/gpt-5.6-sol``. Health reported both
  9router and Ollama reachable, no fallback, all runs completed, focus was
  preserved across the casual interlude and paper switch, table evidence was
  validated, and the returned figure attachment was HTTP 200 with bytes.
- Report:
  ``data/retrieval_eval/internal/e2e-agent-conversation-v5-20260924.json``.
  This confirms the patch did not regress agent wiring; it does not close the
  public retrieval gate or justify enabling dual-query/visual retrieval.

## 2026-09-24 — page provenance v4 frontier

- Added revision-scoped ``chunk_type=page`` rows only for trusted labels on
  textless pages. They are excluded from FTS and Lance embeddings, but are
  available to explicit page/slide resolution. Projection validation and
  vector indexing now count only ``chunk_type=text`` for text projections.
- Full isolated public retrieval frontier completed **21/21** with zero
  failures on lifecycle ``foundation-v4-page-provenance``:
  page recall **1.000**, evidence phrase recall **0.7167**, figure **0.625**,
  table **0.7778**, retrieval P50 **127.97 ms**, max **553.70 ms**.
  Authoritative report:
  ``data/retrieval_eval/public/results/public-retrieval-page-row-v4-21-20260924.json``.
- The retrieval gate is still open: artifact recall is **0.30** and evidence
  recall is below **0.85**. Dual-query and visual flags remain disabled.
- Format lifecycle smoke passed projection/contracts for PDF/DOCX/MD/TXT;
  the intentionally image-only PDF remains ``partial`` and correctly does
  not claim a text retrieval hit. Backend suite is **969 passed**, 5 warnings.

## 2026-09-24 — page-label OCR moved out of retrieval

- Printed-page labels are now recovered during PDF ingestion, bounded by
  ``pdf_page_label_ocr_budget`` (default 16), and cached by source SHA-256.
  The active revision persists the labels in page/chunk provenance; retrieval
  only resolves that metadata and falls back to physical-page semantics when
  a stale revision has no label. It never initializes RapidOCR in the query
  path.
- The previous 21-case report that used query-time OCR is superseded for
  latency/promotion: it proved the page mapping but included a cold
  ``19.54 s`` retrieval outlier. A fresh isolated re-index and rerun is
  required before quoting retrieval P95 or Gate E metrics.
- Targeted parser/RAG/visual tests now pass **64**, including a warm-cache
  assertion that page-label OCR is not initialized on the second read. All
  feature flags remain off and no production projection was changed.
- The durable folder processor contract is now
  ``foundation-v4-page-provenance``; this forces one safe staged rebuild for
  existing bytes so the new page rows cannot be mistaken for an already-valid
  v3 projection.

## 2026-09-24 — sparse printed-label inference

- Added a generic parser rule for sparse OCR: when an Arabic printed-page
  sequence is observed at trusted sampled pages, missing labels between the
  endpoints are interpolated from the observed physical/page stride. This
  handles image-only pages without OCR-ing every footer and does not invent
  offsets for a specific PDF; Roman and facing-page labels stay conservative.
- Parser fingerprints were bumped so a future isolated re-index cannot reuse
  revisions created by the pre-inference parser. Added regression coverage for
  sampled pages 11/16 resolving physical page 14 to printed label 13.
- Targeted parser/RAG/visual tests: **63 passed**. Full backend suite:
  **967 passed**, 5 dependency warnings. The public cached-index report has
  not been rerun with the new parser fingerprint; Gate E remains open.

## 2026-09-24 — on-demand page-label replay

- Added a bounded query-time fallback for explicit page hints when the active
  revision has no printed-label metadata: local OCR inspects only a small
  physical neighborhood and caches labels by source SHA-256. It does not use
  qrels, a fixed offset, or a paper-specific mapping.
- Isolated MMLong case 0315 replay through authenticated ``cx/gpt-5.6-sol``
  now resolves physical page 14 for printed page 13. Candidate, retrieved-text
  and final page recall are all ``1.0``. Two warm Sol replays returned ``2``
  and ``3`` respectively, demonstrating generation variance while retrieval
  stayed correct. Cold retrieval was ``9.30 s``; after the cache/engine-warmup
  optimization it is ``166 ms`` (an earlier warmup-inclusive run was ``1.32 s``).
  This is not yet a claim that the full latency/answer gate passes.
- Reports:
  ``data/retrieval_eval/public/results/gate-e-0315-on-demand-label-20260924.json``
  and warm replay ``...-warm-20260924.json``. The full 21-case frontier still
  needs a fresh isolated run; visual and dual-query flags remain off.

## 2026-09-24 — explicit page candidate path verification

- Added a generic production retrieval channel for explicit ``page/slide N``
  queries. It emits active-revision ``page`` evidence candidates even when the
  page has no OCR/text chunk, while preserving the logical document/revision
  identity. It is driven only by the user's page hint and never by qrels.
- Cached-index Sol rerun completed **21/21** with zero failed rows. MMLong
  candidate evidence-page recall increased from ``0.8182`` to ``0.9091``;
  retrieved-text recall is ``0.8636`` and final evidence recall ``0.8636``.
  SPIQA diagnostic token-F1 is ``0.6443`` and ROUGE-L F1 ``0.6086``. Median
  warm retrieval is ``126 ms`` and median cached preparation ``31 ms``.
- The explicit-slide case 0409 now has candidate and final page recall ``1.0``
  without visual retrieval. Case 0315 remains the sole MMLong page-recall miss.
  The source audit shows that the qrel is coherent: the question names visible
  page 13, which is physical PDF page 14, and the public qrel stores physical
  page ``[14]``. Runtime selected physical page 13 because the image-only page
  had no extracted printed label. This is a generic provenance-mapping miss,
  not a bad qrel; audit artifact:
  ``data/retrieval_eval/internal/mmlong-0315-provenance-audit.json``.
- Candidate recall remains below the ``0.95`` gate until that qrel is audited
  and the remaining arbitrary-format strata are rerun. Visual retrieval and
  dual-query flags stay off. Report:
  ``data/retrieval_eval/public/results/gate-e-postfix-pagehint-21-20260924.json``.

## 2026-09-24 — evaluator-contract-corrected frontier

- Corrected the public evaluator so ``evidence_page_recall`` is computed from
  the actual first-stage ``candidate_results`` frontier, not from the already
  truncated answer context. The report also records ``retrieved_text_page_recall``
  and keeps ``final_evidence_page_recall`` (text plus actually selected image
  attachments) separate. This is a measurement correction, not a retrieval
  or prompt tune.
- Cached-index rerun completed **21/21** with zero failed rows and
  ``cx/gpt-5.6-sol`` generation. MMLongBench-Doc v2: official answer accuracy
  ``0.5455`` (generation is nondeterministic), candidate evidence-page recall
  ``0.8182``, retrieved-text recall ``0.7727`` and final evidence recall
  ``0.8636``. Median warm retrieval was ``124 ms``. SPIQA Test C diagnostic
  token-F1 was ``0.6281`` and ROUGE-L F1 ``0.6062``.
- The retrieval gate remains open: candidate recall is below ``0.95`` and
  visual retrieval source rate is zero because the visual feature flag remains
  off. The answer score is reported separately and is not used to claim a
  retrieval failure.
- Two audit findings must precede further retrieval tuning: MMLong case 0315
  needs a generic printed-label resolver for image-only pages (the qrel itself
  is coherent), while case 0409 attached requested slides 12/14 but
  text-candidate recall stayed zero. These are provenance/modality distinctions,
  not paper-specific fixes.
- Authoritative corrected report:
  ``data/retrieval_eval/public/results/gate-e-postfix-metrics-21-20260924.json``.

## 2026-09-23 — post-fix public frontier (authoritative rerun)

- The post-fix isolated public frontier completed **21/21** rows with zero
  failed rows. All answer generation used authenticated ``cx/gpt-5.6-sol``;
  local retrieval used the pinned Ollama ``embeddinggemma:300m`` provider.
  Production indexes and feature flags were not changed.
- MMLongBench-Doc v2 (11 cases) scored official accuracy ``0.7273``;
  candidate text evidence-page recall was ``0.7727`` and final evidence-page
  recall (retrieved text plus actually selected page attachments) was
  ``0.8636``. Median local retrieval was ``112 ms`` while median isolated PDF
  preparation/indexing was ``152.5 s``. SPIQA Test C (10 cases) reached
  diagnostic token-F1 ``0.6535`` and ROUGE-L F1 ``0.6208``; median retrieval
  was ``120 ms`` and median preparation/indexing ``31.8 s``.
- This is stronger evidence than the previous post-OCR report because it
  includes the new candidate-vs-final metric fields, but it is **not a
  promotion**: candidate recall remains below the ``0.95`` retrieval gate,
  visual retrieval source rate is still zero (feature flag intentionally off),
  and the public frontier is not a substitute for the sealed shadow gate.
- The report exposes distinct failure classes rather than collapsing them:
  some rows have correct attached page evidence but low text-candidate recall,
  while other rows have evidence recall but answer-format/normalization misses.
  No prompt or paper-specific locator was tuned from these rows.
- Authoritative report:
  ``data/retrieval_eval/public/results/gate-e-postfix-21-20260923.json``.

## 2026-09-23 — post-OCR/conditional-multihop verification

- The local 9router gateway is confirmed live and authorized for
  ``cx/gpt-5.6-sol``.  A direct authenticated completion returned ``OK``;
  the earlier ``Missing API key`` response was only from an unauthenticated
  manual probe, not a provider outage.  Ollama remains the local
  ``embeddinggemma:300m`` provider only.
- Image-only visual artifacts now receive bounded local OCR during parsing
  when they have no page text/caption.  This preserves visual provenance and
  supplies searchable anchors without a model call or a production-index
  mutation.
- Generic conditional multi-hop replay ``0413`` now answers ``45%`` (reference
  ``45``), with official MMLong score ``1.0`` and evidence-page recall ``1.0``;
  retrieval was about 236 ms.  The fix combines independent anchor/target
  obligations with the image OCR path; it is not a case-specific locator.
- Post-change verification is clean: ``git diff --check``, Python compile and
  the full backend suite all pass (**964 tests**, 5 dependency deprecation
  warnings).  The full 24-case public shadow has not yet been rerun after
  this latest OCR change, so the arbitrary-format gate remains open.
- The isolated post-change folder lifecycle matrix also passes its intended
  contract: all six supported-format retrieval probes hit, unchanged rescan
  is revision-stable, duplicate bytes create an alias rather than a second
  logical document, content changes create a new revision, rename/move keeps
  the logical identity, and every active row has a revision.  The deliberately
  corrupt PDF remains ``partial`` with a classified ``PdfStreamError``.  Report:
  ``data/retrieval_eval/internal/format-lifecycle-matrix-postocr-20260923.json``.
- A fresh backend HTTP/SSE smoke through the live gateway completed the
  ``file_qa`` route for ``Bài ASPIRE dùng dataset nào?``. It emitted multiple
  ``message.delta`` events, kept the active ``ASPIRE.pdf`` focus, emitted
  ``answer.evidence.validated`` and completed without fallback or transport
  error.
- The post-OCR public frontier is now complete: **21/21** isolated Aya runs
  through ``cx/gpt-5.6-sol`` with zero failed rows.  MMLongBench-Doc v2
  (11 cases) reached official accuracy ``0.9091`` and mean evidence-page
  recall ``0.7273``; SPIQA Test C (10 cases) reached diagnostic token-F1
  ``0.6352`` and ROUGE-L ``0.6120``.  Local retrieval medians were about
  117/130 ms, while PDF preparation dominated (about 157 s median for MMLong).
  The evidence recall gate is still below ``0.85`` and visual retrieval source
  rate remains zero because the feature flag is intentionally off.  Report:
  ``data/retrieval_eval/public/results/gate-e-postocr-21-20260923.json``.
- The lower MMLong page-recall rows are explicit and diagnosable: two questions
  were answered from correctly attached requested page images although the
  text-source page metric does not count those attachments, one comparison
  missed one of two gold pages, and one ranked-page question selected a wrong
  page.  We keep this distinction visible rather than inflating text recall;
  the next fix is page/artifact evidence parity, not a prompt tweak.
- Offline audit of the same 21-case rows gives final-page recall ``0.8182``
  when selected attachments are unioned with text pages (versus candidate
  text recall ``0.7273``). This remains below ``0.85``; the new fields will be
  populated on the next benchmark rerun rather than retrofitting the old
  report.
- Added a generic ``page_superlative`` coverage policy for bounded questions
  such as “page with the slowest processing time”; it is language-normalized
  and not tied to any benchmark document. Full suite after this change is
  **964 passed**.
- Isolated Sol replay of MMLong case 0537 validates the policy end-to-end:
  ``coverage_reason=page_superlative``, official score ``1.0``, candidate and
  final page recall both ``1.0`` and retrieval about 114 ms. Report:
  ``data/retrieval_eval/public/results/superlative-0537-20260923.json``.
- Comparison replay 0494 exposed and then validated two generic fixes:
  branch-unique obligation reservation restored the missing operand, while
  deterministic creator/view aliases let comparison planning avoid a Sol
  round-trip.  The answer stayed ``5%`` with official score ``1.0`` and page
  recall ``1.0``; retrieval fell from about 6.33 s to 301 ms. Report:
  ``data/retrieval_eval/public/results/comparison-0494-v3-20260923.json``.

## 2026-09-23 — live gateway and printed-page mapping fix

- Fresh health check confirms the local gateway is live: 9router is reachable
  at ``127.0.0.1:20128``, ``cx/gpt-5.6-sol`` is available and authorized, and
  the backend reports ``completion_status=ok`` with model fallback disabled.
  Ollama is separately reachable for the pinned local
  ``embeddinggemma:300m`` 768-dimensional embeddings.
- Fixed a generic ingestion/retrieval gap for facing-page PDF extraction.
  Header/footer text such as ``14 15`` is now stored as the printed-label
  range ``14-15``; the parser infers the document's dominant page stride and
  keeps physical page numbers canonical.  Printed labels are resolved in FTS
  page coverage, vector payloads, visual-page lookup and multimodal
  attachment filtering.
- Fresh isolated Sol replay of MMLong cases 0336/0394/0413 validated the
  change.  Case 0336's ``Page 14`` now maps to physical page 9 and Aya
  attaches page 9; its answer changed from ``3`` to the reference ``1``.
  The three-case replay completed 3/3 with no fallback and median retrieval
  about 110 ms.  Overall quality remains below the arbitrary-format gate;
  this is a provenance/transport correctness fix, not a promotion claim.
- Targeted parser/visual/evaluator tests: **120 passed**; full backend suite
  still needs a post-change run.  Dual-query and visual retrieval feature
  flags remain off; production corpus/index was not modified.

## 2026-09-23 — current lifecycle and gateway verification

- Re-ran the isolated folder lifecycle matrix with the current local
  ``embeddinggemma:300m`` provider. PDF native/mixed/table-image, DOCX,
  Markdown and TXT all projected and retrieved their expected evidence;
  the intentionally corrupt PDF stayed ``partial``. Unchanged rescan,
  duplicate bytes, content change, rename/move, stable logical identity and
  active-revision invariants all passed. Report:
  ``data/retrieval_eval/internal/format-lifecycle-matrix-current-20260923.json``.
- Public benchmark manifest validation passed with the pinned source
  revisions/counts. Shadow-v3 structural validation passed (24 cases, 23
  positive and 1 abstention) and it remains sealed/unopened.
- A live HTTP/SSE agent smoke through the backend completed ``dev01-t1``
  with ``cx/gpt-5.6-sol``: route ``file_qa``, expected document retrieved,
  valid evidence/claim guard, no fallback, no transport error. Client first
  delta was about 6.2 s and total generation about 19.2 s; this is provider
  latency, not local retrieval latency.
- The earlier 9router health/SSE checkpoint passed through the backend with
  ``cx/gpt-5.6-sol`` and no fallback. A fresh check in the current shell now
  finds ports ``20128`` and ``11434`` closed (connection refused), so no new
  model-backed score is claimed until the user restarts the sidecars. Direct
  calls without the configured API key still must be treated as unauthorized;
  no fallback model is permitted.
- A shadow-v3 preparation attempt was stopped after its isolated checkpoint
  reached 5/24 cases because the next public PDF spent several minutes in
  parser/index preparation. The partial files are retained for diagnosis but
  are **not** a shadow score; shadow-v3 remains unopened and no promotion
  decision is based on that partial run.

## 2026-09-23 — bounded large-PDF parser policy

- Large PDF sources (over 120 pages or 40 MB) now bypass Docling's expensive
  full table/picture conversion during the local-search stage. The fallback
  preserves native text, extracted images, captions and page provenance;
  captioned/drawing-heavy page screenshots remain lazy canonical evidence,
  while the optional visual worker has its own revision-scoped renderer.
- A 170-page public manual dropped from about 114 seconds in full Docling
  conversion to **2.15 seconds** in the bounded fallback, with 278k extracted
  characters and 134 image artifacts. It is not labelled a complete Docling
  table extraction; readiness/provenance remains authoritative.
- This is a general size policy, not a benchmark-specific exception. Targeted
  parser/lifecycle tests remain green; production flags and index are unchanged.

## 2026-09-23 — public shadow diagnosis: duplicate visual identity

- The bounded shadow completed 23/24 rows; one MMLong case failed during
  isolated ingestion with ``UNIQUE constraint failed:
  document_figures.id``. The cause was generic: repeated OCR/caption labels
  such as ``Figure 1`` were used as the whole visual identity.
- Figure identity now keeps the semantic label/group as an anchor **and** the
  parser ordinal inside the revision-scoped identity envelope. Replaying that
  isolated source now indexes successfully (371 text chunks, 53 tables, 77
  figures); this is a lifecycle correctness fix, not a benchmark answer patch.
- The completed shadow report remains ``status=partial`` and is not promoted;
  it must be rerun after the fix to obtain a valid 24/24 aggregate.

## 2026-09-23 — shadow-v3 rerun after identity fix

- Fresh isolated Aya run completed **24/24 with zero failed rows** through
  ``cx/gpt-5.6-sol`` and local ``embeddinggemma:300m`` (768d). Production
  corpus/index was not modified and ``heldout_60_opened`` remains false.
- Retrieval preparation is now stable, but quality is not at the arbitrary-
  format gate: MMLong official accuracy ``0.3258`` (12 cases), mean evidence
  page recall ``0.6818``; SPIQA diagnostic token-F1 ``0.5016`` and ROUGE-L
  ``0.4671`` (12 cases). These are diagnostic smoke results, not a promotion.
- Warm retrieval medians were ``126 ms`` (MMLong) and ``111 ms`` (SPIQA);
  median indexing was ``32.8/53.7 ms`` because the isolated runtime reused
  validated projections. Visual source rate was ``0`` because visual retrieval
  remains intentionally disabled; page-image attachment is not visual
  retrieval.
- Full backend suite after the lifecycle/identity changes: **955 passed**, 5
  deprecation warnings. Dual-query and visual flags remain off.

## 2026-09-23 — generic visual-observation routing replay

- Visual-intent detection now recognizes observable properties plus a generic
  page/document scope (colour, clothing, people, highlighted words, etc.)
  without requiring the literal word ``figure``. Caption/list inventory
  questions remain metadata-only. This is a general modality fix, not a qrel
  or case-specific rule.
- Isolated 24-case Sol replay completed with zero failures. MMLong official
  accuracy rose to ``0.4056`` with evidence-page recall ``0.6818``; SPIQA
  diagnostic token-F1 was ``0.4657``. Still below the gate, so no promotion.

## 2026-09-23 — agent E2E after visual-routing change

- HTTP/SSE regression over **20/20 conversational dev cases** completed with
  ``cx/gpt-5.6-sol``: 19 ``file_qa`` and 1 ``chat``, zero transport errors,
  zero fallback cases and all expected local-search tools completed.
- Median total latency was about ``16.7 s`` and median client first delta about
  ``5.1 s``; the slowest first delta was about ``50.6 s``. This is an agent /
  9router latency concern, separate from local retrieval medians (~0.1 s),
  and does not block retrieval correctness claims.

## 2026-09-23 — bounded OCR for visual-heavy PDFs

- Foreground OCR now accepts a configurable ``pdf_ocr_page_budget`` (default
  12) through Settings, the folder API and ``IndexingService``. When a large
  or image-heavy PDF exceeds the budget, sparse pages are sampled evenly
  across the document; remaining pages stay visual-only for a durable retry.
- This prevents a scan/manual with hundreds of raster pages from blocking the
  folder worker for minutes while preserving explicit ``partial``/visual
  readiness. It is a generic resource policy, not a benchmark or filename
  rule. Targeted parser/lifecycle tests pass 59/59 after the change.

## 2026-09-23 — answer completeness contract and visual-gate frontier

- Tightened the generic public answer contract: list answers keep distinct
  entities atomic, while free-form answers preserve all supported metrics,
  ranges, comparisons and qualifications. No reference answer or case name is
  interpolated into the prompt.
- Four-case Sol replay (12 generations) completed without failures. SPIQA
  external-source quality improved from the prior ``0.54`` diagnostic token
  F1 to ``0.70`` in the replay; the change is diagnostic and does not promote
  visual retrieval. The image-only slide case still has a strict list-scoring
  mismatch even though the answer is semantically supported by pages 12/14.
- A 21-case ColSmol/T/T+V retrieval frontier was attempted in isolation. The
  worker timed out on the second case, so only 1 positive case is valid;
  promotion correctly failed closed (minimum cases, worker health, visual
  gain and answer-quality checks). This confirms ColSmol remains optional/off,
  not a production dependency.
- Full backend suite after the prompt/visual changes is **953/953**;
  production database/index and feature flags were untouched.

## 2026-09-23 — current paired Gate E report after prompt contract

- Replayed all **21 cases / 63 generations** with live ``cx/gpt-5.6-sol`` and
  evaluator fingerprint ``74c39d54356c3c87``. No failed rows and no fallback:
  ``data/retrieval_eval/public/results/gate-e-answer-quality-current-21-20260923-prompt.json``.
- Aya diagnostic answer quality is ``0.734988`` versus full extracted
  ``0.488157`` and gold evidence ``0.724743``. MMLong official accuracy is
  ``9/11 = 0.8182``; SPIQA token-F1/ROUGE-L are ``0.6435/0.6216`` and remain
  diagnostic, not official SPIQA scores. Median Aya retrieval is ``189 ms``
  (MMLong) / ``126 ms`` (SPIQA); these are not end-to-end Sol latency.
- The two MMLong official misses are not both retrieval failures: the slide
  case received explicit pages 12/14 and produced a semantically supported
  grouped list that the strict list scorer rejects, while the smartphone
  percentage case also fails in gold-evidence mode. Gate E therefore remains
  diagnostic/non-promotable despite the large Aya-vs-full improvement.
- ColSmol visual hits were unavailable on several cases due cold-index
  timeouts; page-render fallback stayed fail-closed and did not activate a
  production visual flag. Shadow-v3 remains sealed.

## 2026-09-23 — isolated Lance revision migration validation

- Ran ``scripts/migrate_lancedb_revision.py`` against the current production
  Lance source into an isolated runtime (SQLite authority was read-only). It
  copied and validated ``21 document cards / 1055 text / 60 table / 59 figure``
  rows, all at 768 dimensions, with zero unattributed or stale rows.
- Isolated ``rag_status`` then reported active physical rows exactly
  ``21/1055/60/59``. Production status still truthfully reports legacy Lance
  projection because the atomic path switch is intentionally deferred until
  Gate E + RC/shadow acceptance; production was not mutated.
- Desktop production build passed. Backend remains **953/953** and all flags
  stay off where their gates are not met.

## 2026-09-23 — page/slide visual scope fix and paired public replay

- Added two generic visual contracts: plural visual nouns (``figures``,
  ``slides``, ``charts`` and related forms) activate the shared visual intent
  policy, and an explicit page/slide reference remains the attachment scope
  even when the optional ColSmol ranker times out or misses that page. This
  uses only the user's query; it does not read qrels or map any paper name.
- A focused isolated replay through live ``cx/gpt-5.6-sol`` completed 2/2
  cases and 6/6 generations. The image-only slide case attached the requested
  pages ``[12, 14]`` despite a ColSmol timeout; the mixed scan page-count case
  attached page ``13`` and returned the gold ``2``. No production index or
  feature flag was changed.
- Targeted visual/query tests pass (78/78). The full paired 21-case report
  remains the authoritative broader diagnostic at
  ``data/retrieval_eval/public/results/gate-e-answer-quality-current-21-20260923.json``:
  Aya ``0.646849`` overall versus full extracted ``0.438274``; MMLong official
  accuracy ``0.6364`` and SPIQA token-F1 ``0.6584`` (SPIQA metric remains
  diagnostic). Gate E is still not promotable because the suite contains
  per-case answer regressions and ColSmol cold-index timeouts.
- 9router/backend health was rechecked: completion authorization and Sol
  streaming are healthy; Ollama remains only the local ``embeddinggemma:300m``
  provider. ``VISUAL_RETRIEVAL_ENABLED`` and ``DUAL_QUERY_RRF_ENABLED`` remain
  off.

## 2026-09-23 — readiness-aware visual recovery and page-pixel fallback

- Public visual audit found a generic contract gap: OCR-created text chunks
  could hide a `partial` document readiness state, and a selected canonical
  figure/page image was not exposed when the question did not literally say
  “figure”. The evaluator now carries `document_readiness` through the active
  revision status, requires visual recovery for partial scan documents with no
  native pages, and promotes selected image-backed figures/tables to the pixel
  contract without using qrels or paper names.
- If ColSmol visual indexing times out, page renders selected by the active
  text/OCR projection remain valid bounded visual evidence when readiness
  requires recovery; no arbitrary attachment can bypass the guard.
- Regression evidence: MMLong `0569` changed from `19%` to the gold `14%` with
  canonical figure pixels; MMLong `0845` changed from `Not answerable`/`7` to
  the gold `5` using four active page renders without visual indexing. The
  isolated replay report is
  `data/retrieval_eval/public/results/gate-e-answer-regression-0845-20260923.json`.
  The three-case visual-index replay still records a ColSmol timeout for the
  image-only PDF; that is optional-provider latency, not a fallback model.
- Added readiness/figure/pixel contract tests. Full backend suite is now
  **952/952**; Graphify was synchronized. Production visual and dual-query
  flags remain off and no production projection was mutated.

## 2026-09-23 — clean conversational dev-20 rerun

- Re-ran the full `data/retrieval_eval/conversational-dev-v1.jsonl` suite
  through the live HTTP/SSE backend with `cx/gpt-5.6-sol` after the generic
  comparison-coverage fixes. The run completed **20/20 passed**, with
  `transport_error=null`, zero fallback cases, and routes `19 file_qa + 1
  chat`.
- Total latency was `P50 16.84 s`, `max 28.12 s`; client first visible delta
  was `P50 5.55 s`, `max 28.12 s`. These are Sol/provider timings, not local
  retrieval latency.
- This replaces the earlier conservative `19/20` checkpoint caused by one
  transient `upstream connection lost` at `dev05-t1`; no retrieval or answer
  validator failure remains in the clean aggregate. The raw report is
  `data/retrieval_eval/internal/agent-conversational-dev-sol-current-20260923.json`.
- Gate C remains passed for A0 and dual-query remains off. Gate E/public
  visual answer quality is still diagnostic/non-promotable; shadow-v3 remains
  sealed and no production projection or feature flag was changed.

## 2026-09-23 — comparison coverage hardening and live Sol regression

- Rechecked local 9router through the real backend: `/v1/models` exposes the
  approved Sol/Terra/Luna routes, backend health reports completion authorized,
  and a live SSE completion streamed `router-ok` through `cx/gpt-5.6-sol`.
  Ollama remains the local `embeddinggemma:300m` provider only.
- Added a generic multi-document comparison contract to the answer prompt:
  every focused document gets an identity-bound section before synthesis,
  including an explicit limitation when a requested facet is unavailable.
  No paper-specific alias or case mapping was added.
- Grounded comparisons now use a bounded low temperature and preserve a
  supported, identity-labelled partial draft when the other paper has no
  support; the sanitizer appends a document-bound insufficiency instead of
  replacing the whole answer with a generic fallback. Unsupported numbers are
  still removed and validation remains fail-closed.
- Targeted prompt/guard tests pass; full backend suite is **949/949**.
  Graphify was synchronized after the code changes.
- The two multi-paper dev groups (ASPIRE/ViSEC and LPMN/Wav2Small) replayed
  **12/12** with valid document coverage and no fallback. The latest full
  conversational dev-20 run completed **20/20** with one `upstream connection
  lost` transport failure at `dev05-t1`; a clean replay of that group passed
  **4/4**, so no retrieval or validator failure is claimed for that row.
- Gate C remains passed for A0 on audited `dev-v2.2`; B-full has no gain and
  dual-query remains disabled. Gate E's 21-case Sol answer-quality report is
  complete but diagnostic/non-promotable because per-case public visual rows
  still regress. The authoritative report is
  `data/retrieval_eval/public/results/gate-e-answer-quality-current-21-20260922.json`:
  overall Aya diagnostic score `0.5914` versus full extracted `0.4416`, with
  MMLong official accuracy `0.5401`; SPIQA token-F1/ROUGE are diagnostic only.
  `VISUAL_RETRIEVAL_ENABLED` and `DUAL_QUERY_RRF_ENABLED` remain `false`;
  shadow-v3 is still sealed.

## 2026-09-16 — folder lifecycle deferred-activation propagation

- Fixed the folder ingestion path to propagate `defer_activation` explicitly
  into `_index_file` instead of relying on its default. New/changed folder
  revisions therefore remain staged until the coordinator validates FTS/Lance
  counts and explicitly switches `active_revision_id`.
- This is a lifecycle correctness fix, not benchmark tuning; direct file
  indexing compatibility remains unchanged.
- Identity/lifecycle and processing-service regression suite: **22 passed**;
  `git diff --check` passed. Production index and feature flags were not
  changed.
- Backend/frontend were restarted and are healthy locally. Sol completion
  remains blocked by the gateway's `401 token_revoked`, with no model
  fallback.

## 2026-09-21 — arbitrary-format lifecycle revalidation

- Re-ran the isolated durable folder matrix through the host-side Ollama
  provider after a sandbox-only connection failure. PDF (native, mixed and
  table-image), DOCX, Markdown and TXT all projected and retrieved their
  expected evidence; the intentionally corrupt PDF remained `partial`.
- Duplicate bytes, unchanged rescan, content change, rename/move and active
  revision invariants all passed. Report:
  `data/retrieval_eval/internal/format-lifecycle-matrix-current.json`.
- Two tracked MMLong public PDFs had been removed from the local benchmark
  clone; restoring them from its Git history returned the public loader and
  sampling tests to green. No benchmark loader was weakened to skip missing
  evidence.
- Full backend verification: **932 passed**, 5 dependency deprecation
  warnings. `git diff --check` passed; production Lance/index and retrieval
  flags remain unchanged.

## 2026-09-16 — generic list/table final-cutoff fix

- Fixed a general retrieval loss where a canonical table was present in the
  deep candidate frontier but dropped before the visible final top-k. Final
  structured promotion now runs after candidate generation (and after
  reranking when enabled), not only inside the first-stage merge.
- Added shared enumerated-evidence intent for generic questions such as which
  languages/models/metrics/items; document-discovery questions (which paper,
  author, article) are excluded to avoid noisy table reservation.
- Modality-local vector RRF ranks text/table/figure channels independently,
  preserving sparse artifact hits without paper-specific rules.
- Isolated SPIQA replay for “Which 7 Indian languages do they experiment with?”
  now returns the canonical table in final results and context; no gold label
  or production write was used.
- Backend suite: **927 passed**; desktop production build passed. Flags remain
  `DUAL_QUERY_RRF_ENABLED=false` and `VISUAL_RETRIEVAL_ENABLED=false`.

## 2026-09-16 — public 21-case retrieval replay after cutoff fix

- Replayed the pinned 21-case public frontier in an isolated runtime with the
  current production-shaped parser/index path and local EmbeddingGemma; 21/21
  completed without provider fallback or production writes.
- Diagnostic aggregate: page recall `0.8182`, evidence phrase recall `0.7333`,
  artifact recall `0.3000`, retrieval support recall `0.7500`; warm retrieval
  P50 `182 ms`, maximum `1,187 ms` (cold parse/OCR is reported separately).
- The result is not a Gate-E pass. Visual artifact cases still need a paired
  visual-channel run, and some public annotation/artifact-name diagnostics are
  not equivalent to answer quality.
- A single Sol probe still returns `401 invalidated oauth token`; no paper
  content was sent and no agent E2E aggregate is claimed.

## 2026-09-16 — explicit artifact routing/context preservation

- Fixed a generic retrieval/context loss for numbered artifact questions. An
  explicit `Figure N`/`Table N` query can now recover the owning document from
  active canonical artifact rows before vector routing; generic artifact
  requests remain bounded to the document frontier selected by cards/FTS.
- Context composition now reserves the requested figure label/index before a
  merely similar visual hit. This preserves the exact artifact through the
  final prompt without any paper- or case-specific mapping.
- Isolated MMLong case `mmlongbench-doc-v2:0543` now selects page 118/Figure 89
  (previously page 82) through the same production-shaped path. This is a
  regression proof, not a corpus-wide score claim.
- Added metadata label/index terms to the canonical artifact lexical
  projection for caption-poor table-image/figure rows.
- Full backend verification after the fix: **923 passed** (5 warnings).

## 2026-09-16 — public retrieval replay after artifact fix

- Replayed the cached 9-case MMLong/SPIQA retrieval-only sample with no LLM
  calls and no production writes. Page recall improved from `0.6667` to
  `1.0000`; retrieval P50/max was `145/236 ms`.
- Evidence/artifact recall remained `0.6389`/`0.3333` because several SPIQA
  annotations require image attachments while their natural-language queries
  are answerable from extracted text; this remains a diagnostic limitation,
  not a promotion score.
- The MMLong Figure 89 row now contains page 118/Figure 89 in the composed
  context. Full public coverage and Sol answer-quality replay remain pending
  valid gateway completion authorization.

## 2026-09-16 — Sol smoke checkpoint after gateway restart

- The local 9router process was started and `/v1/models` exposed the approved
  `cx/gpt-5.6-sol`, `cx/gpt-5.6-terra` and `cx/gpt-5.6-luna` identifiers.
- A 3-case Aya-only smoke reached local EmbeddingGemma retrieval, then all
  three generations failed closed because the gateway returned an invalidated
  OAuth token (`401 token_revoked`, wrapped as `503`). The report is retained
  at `data/retrieval_eval/public/results/gate-e-answer-regression-current-sol-20260916.json`
  and is not a quality score.
- A second attempt after the gateway health probe became reachable reproduced
  the same `token_revoked` response immediately; `/v1/models` availability is
  therefore not sufficient evidence of completion authorization.
- No fallback model, production write or held-out opening occurred. Backend
  and frontend remain available for local UI testing; Sol E2E resumes only
  after the gateway token is refreshed and a model completion succeeds.
- Health now reports catalog reachability separately from
  `completion_authorized`/`completion_error`; a revoked token cannot be shown
  as fully ready merely because `/v1/models` responds.
- Full backend verification after this observability fix: **921 passed**.
- Backend was restarted with the new contract; current health is
  `completion_status=unknown` / `completion_authorized=null` until a real
  completion succeeds or fails, which avoids an unrequested probe call.

## 2026-09-16 — public OCR paired diagnostic

- Re-ran the same 9-case stratified public retrieval sample with the normal
  selective OCR path (isolated runtime, no LLM). All 9/9 cases completed.
- Page recall improved from `0.3333` (no-OCR) to `0.6667`, while evidence
  recall stayed `0.6389` and artifact recall `0.3333`; retrieval P50/Pmax was
  `181/264 ms` versus `152/234 ms` without OCR. This supports OCR as a
  targeted scan/page fallback, not as a replacement for Docling or visual
  retrieval, and is not a promotion gate.
- The native-text profile still had a page miss and table artifact recall was
  `0.25` on four cases; no case-specific patch or production mutation was
  applied. Report: `data/retrieval_eval/public/results/public-retrieval-only-stratified-v5-ocr.json`.

## 2026-09-14 — public evaluator gold-independent visual routing

- Removed benchmark `referred_artifacts`/`evidence_sources` from visual boost,
  page reservation and pixel-guard activation. Public routing now follows the
  question's visual intent or an actually text-empty active projection; gold
  annotations remain scoring-only.
- Added coverage for text-backed Markdown with artifact locators so it is not
  incorrectly forced into `Not answerable`; image-only projections still fail
  closed without pixels. Full backend suite: **914 passed**.
- A fresh 21-case Sol replay was started but stopped fail-closed when 9router
  returned HTTP 429 usage-limit exhaustion. Its partial report is retained and
  is not used for promotion; no fallback model or production write occurred.
- The last complete paired frontier remains the authoritative diagnostic until
  quota resets; Gate E and both visual/dual-query flags stay closed.

## 2026-09-16 — services, checkpoint and Gate-C revalidation

- Restarted the local backend (`:7777`) and Vite frontend (`:5173`) for UI
  testing. Backend health is `degraded` only because 9router (`:20128`) is
  currently offline; Ollama and `embeddinggemma:300m` are reachable.
- Resuming the isolated Sol checkpoint failed closed at model preflight because
  the gateway was unreachable; no fallback or model call occurred.
- Re-ran Gate C retrieval-only on `dev-v2.2` with Ollama: A0/A1/B-full
  Candidate/Evidence/Document `0.97/0.91/1.00`; B-anchor `0.97/0.92/1.00`.
  A0 warm P50/P95 was `206/437 ms`; B-anchor was `314/495 ms`. No production
  data changed.
- Full backend suite after the evaluator routing changes: **914 passed**.
- Isolated lifecycle matrix replay passed six expected retrieval hits plus
  duplicate/rescan/change/rename invariants for PDF/DOCX/MD/TXT; corrupt PDF
  remained explicitly `partial`. Report:
  `data/retrieval_eval/internal/format-lifecycle-matrix-20260916.json`.

## 2026-09-16 — Lance revision migration audit

- Read-only migration audit used the backend's actual SQLite DB
  (`data/sqlite/app.db`) and built a separate projection at
  `/private/tmp/aya-lancedb-revision-audit-20260916-correct`.
- All 21 active documents mapped cleanly: `document_cards=21`,
  `text_chunks=1055`, `table_chunks=60`, `figure_chunks=59`; every output
  table has a physical `revision_id`, and a full-row check found zero active
  identity mismatches.
- Production Lance remains legacy (no physical `revision_id`) and was not
  switched or mutated. A first audit attempt against the unrelated empty
  `data/rag.db` was discarded; it is not evidence about the production corpus.
- `scripts/rag_status.py` now reports this distinction explicitly instead of
  printing an ambiguous active-row count. Isolated vector/lifecycle tests:
  **25 passed**; dual-query and visual flags remain disabled.
- Gate-C retrieval-only rerun against that isolated projection preserved the
  production-corpus baseline: A0/A1/B-full `0.97/0.91/1.00`, B-anchor
  `0.97/0.92/1.00`; A0 P50/P95 `138/305 ms`, B-anchor `215/405 ms`.
  This is migration parity evidence, not a promotion or model-quality claim.
- Migration audit is now fail-closed when a non-empty Lance source is paired
  with a SQLite DB that has no active revisions (`active_revision_db_empty`),
  preventing accidental use of an unrelated empty DB.
- Full backend suite after migration hardening: **918 passed** (5 dependency
  deprecation warnings).
- The migration utility now opens the SQLite authority in `mode=ro` and does
  not call schema initialization; missing/wrong authority paths fail closed.
  Correct-DB audit still produced the same 21-document/1,195-vector result.

## 2026-09-16 — public arbitrary-format diagnostic

- Retrieval-only sample (3 MMLong PDF profiles + 6 SPIQA cases, no OCR and no
  LLM) completed **9/9** through the isolated coordinator. It reported page
  recall `0.333`, evidence phrase recall `0.639`, table-stratum evidence
  recall `0.542`, and retrieval P50 `152 ms`. This is intentionally diagnostic:
  visual/page questions are expected to miss without a visual channel and the
  sample is not a promotion gate.
- A matching ColSmol run was started with local-only weights. After ~5 minutes
  it had completed 1 case while the worker used ~6% RAM and >200% CPU, so it
  was stopped cleanly; the partial checkpoint is retained and no visual score
  was emitted. This confirms visual indexing must remain background/optional.

## 2026-09-14 — open-ended metric-prose rescue replay

- Added a generic text-detail frontier: metric/evaluation questions now reserve
  the strongest lexical text-prose candidate when no explicit table/figure is
  requested. Explicit artifact rows are never evicted, and the rule contains
  no paper/case mapping.
- Added regression coverage for the frontier and reran the full backend suite:
  **914 passed** (5 dependency deprecation warnings).
- Replayed Gate C on `dev-v2.2`: A0/A1/B-full remain
  Candidate/Evidence/Document `0.97/0.91/1.00`; B-anchor is `0.97/0.92/1.00`
  with P50/P95 `231/411 ms` and 129 embedding calls. A0 warm P50/P95 is
  `147/361 ms`; no gate regression, but dual-query still has no promotion
  benefit.
- Report: `data/retrieval_eval/internal/retrieval-v2.2-dev-text-detail.json`.
  This remains retrieval-only and made no production writes or model calls.

## 2026-09-14 — active artifact lexical fallback verification

- Added a bounded, revision-scoped lexical fallback for canonical tables and
  figures in `RagService`. It is used only inside an already selected document
  scope, merges through the same evidence identity/RRF shape, and never scans
  the corpus or reads evaluation answers/qrels.
- Fixed the table-side SQL projection to use `NULL` for figure-only columns;
  the new regression test verifies active artifacts are returned while retired
  revision artifacts are excluded.
- Backend suite: **912 passed** (5 dependency deprecation warnings). The
  fallback is a correctness/recovery improvement, not a Gate-C/E promotion
  claim; production dual-query and visual flags remain disabled.
- Backend `:7777`, frontend `:5173`, 9router `:20128` and Ollama `:11434`
  were health-checked. A Sol HTTP/SSE smoke reached the scoped ViSEC paper and
  emitted retrieval/tool events; no production index or database was changed.

## 2026-09-14 — Gate-C replay after artifact gating

- Replayed the same 60-case `dev-v2.2` suite after restricting canonical
  artifact recovery to explicit structured-evidence queries. A0 now reports
  Candidate R@30 **0.97**, Evidence R@10 **0.91**, Document R@5 **1.00**,
  modality text/figure/table **0.85/1.00/1.00**, multi-paper **1.00**, and
  warm latency P50/P95 **147/368 ms**.
- A1 remains equal to A0; B-full remains equal to A0. B-anchor reaches
  Candidate **0.98** and Evidence **0.93**, but costs P50/P95 **231/409 ms**
  and 129 embedding calls. Dual-query therefore remains disabled.
- Report: `data/retrieval_eval/internal/retrieval-v2.2-dev-artifact-gated.json`.
  This is a retrieval-only diagnostic; no LLM calls or production writes were
  made.

## 2026-09-14 — lifecycle matrix replay

- Re-ran the isolated durable folder matrix through the current coordinator
  using local EmbeddingGemma. Native/mixed/table-image PDF, DOCX, Markdown and
  TXT all projected and retrieved their expected tokens; the intentionally
  corrupt PDF remained `partial`.
- Lifecycle invariants stayed true: unchanged rescan reused its revision,
  duplicate bytes created an alias rather than a new logical document, changed
  content created a new revision while retaining logical identity, rename kept
  identity, and active rows all had revision IDs.
- Report: `data/retrieval_eval/internal/format-lifecycle-matrix-20260914.json`.

## 2026-09-14 — Gate-E paired replay after artifact gating

- Replayed the same 21-case public MMLong/SPIQA frontier with 63 Sol
  generations on the isolated cache. All generations completed with no
  fallback; the visual retrieval report remains `T/V/T+V = 0.4762/0.8571/1.0`.
- Aya answer quality was `0.7115` overall versus bounded full context `0.4342`
  (delta `+0.2772`); MMLong official accuracy was `0.8128`. SPIQA diagnostic
  token-F1 remained slightly below full context (`0.6000` vs `0.6119`) and
  three per-case regressions remain, so promotion is correctly **false**.
- Reports: `gate-e-answer-quality-frontier-v2-artifact-gated.json` and
  `gate-e-combined-frontier-v2-artifact-gated.json`. This is a diagnostic
  frontier, not a production-ready claim; visual and dual-query flags stay off.

## 2026-09-13 — generic visual-modality intent coverage

- Added format- and paper-independent visual vocabulary for natural questions
  that say `visual grounding`, `pixel information`, `multimodal`,
  `image-text`, `vision` or related terms without naming a figure/diagram.
  These terms now activate the optional visual route and pixel-sufficiency
  guard; caption/list inventory questions remain metadata-only.
- Added regression tests for visual-grounding and pixel-information phrasing.
- Full backend suite: **911 passed** (5 dependency deprecation warnings).
- Production visual and dual-query flags remain disabled; no index/database
  mutation occurred. Backend/frontend/9router/Ollama remain available for UI
  smoke testing.

## 2026-09-13 — Gate-C v2.2 retrieval-only replay

- Ran the audited `dev-v2.2` suite (60 cases, 50 positive/10 negative) through
  the current SQLite/LanceDB projections with local EmbeddingGemma; compare
  made no LLM calls and did not mutate production data.
- A0: Candidate R@30 **0.97**, Evidence R@10 **0.91**, Document R@5 **1.00**;
  modality text/figure/table **0.85/1.00/1.00**, multi-paper **1.00**,
  warm latency P50/P95 **148/361 ms**.
- A1 is effectively identical to A0. B-full is also identical (no measurable
  recall delta); B-anchor reaches Candidate **0.98** and Evidence **0.93**
  but increases P50/P95 to **222/424 ms** and activates two query branches in
  65% of cases. The promotion rule therefore keeps dual-query disabled.
- Report: `data/retrieval_eval/internal/retrieval-v2.2-dev-current.json`.

## 2026-09-13 — sealed public shadow-v3 manifest

- Generated `evals/retrieval_v2/shadow-v3.manifest.json` and the matching
  answer-free `shadow-v3.questions.jsonl` from genuine MMLongBench-Doc v2 and
  SPIQA Test-C cases. The selection has 24 cases in six format/modality
  buckets, 24 distinct documents, and excludes every document in the visible
  Gate-E frontier.
- Only case IDs, questions, bucket/profile labels and source/content
  fingerprints are tracked; answers and evidence qrels remain outside the
  manifest. `opened=false` is sealed and no shadow score has been run.
- Generated private qrels at
  `data/retrieval_eval/private/shadow-v3.qrels.jsonl` (24 rows: 23 positive,
  1 expected-abstention negative) with a manifest checksum. The private file
  is ignored by Git and is not an authorization to open the suite.
- Sealed-manifest validator passes: 24 questions/24 documents, six buckets at
  4 each, qrel checksum valid, and no tracked answer/evidence fields.

## 2026-09-13 — service and current-code verification

- Backend `:7777`, Vite frontend `:5173`, 9router `:20128` and Ollama
  `:11434` are running and reachable. Backend health confirms the approved
  answer/router/VLM model is `cx/gpt-5.6-sol`; production embedding remains
  local Ollama `embeddinggemma:300m` (768d), with model fallback disabled.
- A real `/chat/stream` smoke returned `message.started`, multiple
  `message.delta` events and `message.completed` through Sol. This confirms
  the UI transport path is live; it is not a RAG quality benchmark.
- Desktop production build passed (`tsc && vite build`). Full backend suite on
  the current tree is **906 passed** (5 dependency deprecation warnings).
- The isolated SPIQA single-case r94 artifact-projection replay completed
  3/3 generations with no failure, but Aya diagnostic token-F1 remained
  `0.1600` versus full extracted `0.4242`. Three visual images were attached;
  this is a diagnostic result only and is not merged into the authoritative
  21-case r92 aggregate.
- Graphify was synchronized after the evaluator changes. Visual and
  dual-query flags remain disabled; no production index/database was changed.

## Multi-page visual context reservation — 2026-09-03

- Centralized `visual_page_reservation_count()` in `RagService` and wired it
  into the live agent, adaptive/decomposed/LightRAG composition paths, and the
  public evaluator. Explicit `page(s)`, `slide(s)` and `trang(s)` hints now
  reserve every matching retrieved document/page identity; unconstrained
  visual turns retain the one-page minimum. No page is synthesized when a hint
  cannot be resolved.
- Regression coverage now verifies both the reservation helper and bounded
  context composition; backend suite is **873 passed**.
- Isolated Sol smoke for MMLong `0409` (“slides 12 and 14”) now carries two
  visual page sources/images and `evidence_page_recall=1.0` (previously only
  page 12 survived context composition). This is a one-case diagnostic, not a
  Gate-E promotion claim; the official list scorer returned 0 despite the
  grouped answer containing the requested stages.
- Production `VISUAL_RETRIEVAL_ENABLED` and `DUAL_QUERY_RRF_ENABLED` remain
  false; no production index or corpus was changed.

## Gate-E paired 20-case run — 2026-09-03

- Completed the same 20-case frontier through both isolated runners:
  `gate-e-answer-quality-r21-postfix-20case.json` (60/60 Sol generations) and
  `visual-gate-r21-aligned.json` (20/20 retrieval rows). The exact case-ID
  join is recorded in `gate-e-combined-r21.json`; no independent samplers are
  being presented as paired evidence.
- Answer quality: Aya primary score `0.7291` vs bounded full context `0.6243`
  (delta `+0.1049`); MMLong official accuracy `0.7451`, SPIQA diagnostic
  token-F1 `0.7051`. Two SPIQA table cases regress at answer level, so the
  explicit answer attestation is `promotion_safe=false`.
- Visual retrieval: T recall `0.5833`, V `0.7778`, T+V `0.8333` on 18 positive
  rows (2 unanswerable). The visual gain is real in this sample, but the
  minimum positive/profile checks and answer-quality no-regression check fail;
  `VISUAL_RETRIEVAL_ENABLED` remains false.
- This is a useful diagnostic, not a production claim: text retrieval misses
  remain in several SPIQA table/artifact cases and unseen `shadow-v3` is still
  required.

## Gate-E page-reservation replay — 2026-09-03

- Replayed the exact 20-case/60-generation frontier with `cx/gpt-5.6-sol`
  after the shared multi-page reservation fix. All rows completed with
  `embeddinggemma:300m` preflight (768d), no fallback and no production writes.
- Answer diagnostics: MMLong Aya official accuracy `0.4951` vs full extracted
  `0.5000`; SPIQA Aya diagnostic token-F1 `0.7253` vs full `0.6919`; overall
  Aya `0.5872` vs full `0.5768`. MMLong native/multi-page misses remain, so
  answer-quality promotion is still closed.
- The strict join with visual report `visual-gate-r23-production-adapter.json`
  is `gate-e-combined-r25-page-reservation.json`: T+V recall `0.8889`, 18
  positive rows (<20 minimum), visual gain and text non-regression true, but
  answer-quality/profile gates false. `VISUAL_RETRIEVAL_ENABLED` remains off.

## Stack status observability — 2026-09-03

- `scripts/rag_status.py` now prints the model/provider split and feature
  flags explicitly: answer/router/VLM use the approved 9router model, while
  production embeddings use local Ollama `embeddinggemma:300m` (768d).
- The command distinguishes physical revision-aware active counts from the
  legacy production schema. Until the isolated RC is promoted, production
  reports `legacy projection (migration pending)` instead of crashing or
  pretending staged/retired rows are active.
- It also reports parser readiness separately from canonical projection totals
  and labels Lance totals versus active-revision counts.

## Gate-E cache-aware visual smoke — 2026-09-03

- Re-ran the bounded, retrieval-only visual comparator against the existing
  isolated ColSmol cache (`visual-gate-latest-cache-smoke.json`): 7/7 rows
  completed with no 9router calls and no production writes.
- Text (`T`) page/evidence recall was `0.9286`; ColSmol visual (`V`) was
  `1.0000`; late fusion (`T+V`) was `1.0000`. The sample covers MMLong native,
  mixed and image-only PDFs plus SPIQA figure/table cases.
- Warm latency: T P50/P95 `1338/2220 ms`; V P50/P95 `105/3188 ms`. The visual
  P95 is an observed CPU-worker outlier, not a production latency guarantee.
- Promotion remains **false**: only 7 positive cases (<20 minimum), answer
  quality is not attached, and 9router is currently unreachable. This result
  validates the isolated visual path only; it does not justify enabling
  `VISUAL_RETRIEVAL_ENABLED`.
- Document status now exposes `readiness_state` and the parser's readiness
  payload separately from `local_search_state`; a healthy local projection no
  longer hides that an image-only/mixed document is partial or visual-only.
  Added regression coverage; the targeted processing suite is **11 passed**.
- Full backend verification after the status-contract change: **864 passed**;
  desktop production build also passed. Production flags and indexes remain
  unchanged.

## Latest verification — 2026-09-03 (visual readiness guard)

- The isolated answer-quality evaluator now treats a projection/readiness
  signal (`visual_context_expected`) as part of the visual fail-closed
  contract. An image-only PDF cannot silently fall back to OCR/text and make
  a visual claim when page pixels were not selected; explicit chart/figure
  questions keep the same guard on mixed documents.
- Added a regression for a generic fact-shaped question on an image-only
  document with no attached page pixels. It returns the evaluator's
  `Not answerable` guard instead of accepting an inferred answer.
- Verification after this change: **862 backend tests passed**, compile and
  `git diff --check` are clean, and the desktop production build passed.
  Graphify was synchronized (4,125 nodes / 11,953 edges / 156 communities).
- A new 20-case Gate-E post-fix run was not scored: 9router went offline after
  preflight and the checkpoint contains zero answer rows. No fallback model
  was used and no production projection/flag changed.
- Re-ran Gate-D smoke `data/retrieval_eval/internal/format-lifecycle-smoke-r18.json`
  through local `embeddinggemma:300m`: TXT/MD/DOCX are `ready` with expected
  text hits; the image-only PDF is honestly `partial`/non-searchable while
  its revision-scoped projection remains valid. `all_retrieval_contracts_ok`
  is true; the raw all-hit aggregate is intentionally false for the scan.
- Closed a revision-isolation gap in paper evidence cards: the new
  `paper_evidence_card_revisions` / facet / item projections are authoritative
  by `(document_id, revision_id)`, while the old one-row tables remain a
  compatibility mirror. A revision switch test confirms an older active card
  is restored without stale evidence. Verification after this migration:
  **863 backend tests passed**; production data was not migrated or mutated.

## Latest verification — 2026-09-02 (format smoke readiness contract)

- Re-ran the isolated generated TXT/MD/DOCX/image-only-PDF smoke through the
  local Ollama `embeddinggemma:300m` provider. All four revision-scoped Lance
  projections validated; TXT/MD/DOCX reached `ready` and their expected text
  hits passed.
- The generated image-only PDF is intentionally reported as `partial` with
  zero native/OCR-searchable pages because RapidOCR recovered only low-signal
  text. Its projection is valid and the non-searchable readiness contract
  passes; the raw text-hit diagnostic remains `false`. This is not silently
  counted as a text retrieval success or a production `ready` document.
- Added fail-closed contract tests for this distinction and for visual
  promotion semantics. The report is
  `data/retrieval_eval/internal/format-lifecycle-smoke-r17.json` with schema
  `format-lifecycle-smoke-v2`; it makes no 9router calls and does not mutate
  production paths.
- Gate D therefore validates lifecycle/provenance/readiness, while Gate E
  remains the separate visual-value and answer-quality gate. ColSmol,
  MMLongBench-Doc and SPIQA are isolated diagnostics, not default production
  dependencies or proof that arbitrary-document quality is complete.
- Hardened the visual promotion contract so retrieval-only success can never
  set `promotion.passed=true`: a separately attached, complete answer-quality
  report must explicitly set `promotion_safe=true` and meet the same minimum
  paired-case count. Existing visual reports therefore remain valid
  diagnostics but correctly stay non-promotable.
- A fresh 6-case Sol/ColSmol answer-quality attempt was interrupted during
  cold visual indexing after 2/18 mode rows; its checkpoint is retained and
  no aggregate was published. The interruption confirms the real cost of
  cold visual indexing, not an answer-quality score.
- Verification after these changes: **860 backend tests passed**, compile and
  diff checks are clean; production flags/indexes remain unchanged.
- A cache-reuse one-case Sol replay (`gate-e-answer-quality-r20-cached-1case`)
  completed 3/3 modes with official MMLong score `1.0` and visual source /
  attachment rates `1.0`. It is a regression smoke only (one unanswerable
  case), not a Gate-E aggregate.

## Latest verification — 2026-09-02 (visual pixel-intent contract)

- Non-coverage visual answers now attach the ranked visual page(s) first and
  deduplicate rendered aliases such as `page-00015.png`/`page-15.png`. A
  singular visual question is limited to one ranked page; comparison/all-page
  wording retains a bounded multi-page allowance. This prevents adjacent OCR
  pages from adding unsupported objects to an otherwise correct image answer.
  Mixed-scan MMLong `0092` replay now attaches one page image (instead of two
  aliases plus adjacent pages); the Sol answer contains the gold object, while
  the official scorer remains conservative on the extra article wording.
- The live agent now shares the same readiness signal: when visual retrieval
  is explicitly enabled and the active scope contains a visual-only document,
  page retrieval activates for fact-shaped questions too. Canonical
  figure-asset expansion remains tied to explicit visual intent/boost, so the
  readiness fallback does not add unnecessary vector calls.
- Image-only PDF projections now trigger the optional visual channel in the
  isolated multimodal evaluator even when the question is phrased as a plain
  fact/table query. The decision comes from canonical `text_chunks == 0`, not
  benchmark answers, qrels or paper-specific mappings. A cached ColSmol + Sol
  replay of MMLong case `0845` returned the exact answer (`5`, official score
  `1.0`), with one visual page source and four attached page images. This is a
  targeted diagnostic; visual retrieval remains feature-off in production.
- A three-profile retrieval-only attempt was intentionally interrupted after
  two rows because the v2 sampler selected PDFs absent from the chosen cache.
  Its checkpoint is retained as audit evidence: image-only `0845`-style cases
  can miss the required page with text/table retrieval alone, while mixed
  image pages can have correct page recall but zero answer-token coverage.
  No aggregate was published from that interrupted run.
- Added a generic exact-value page-companion projection in `RagService`:
  when a date/number/metric/identifier-shaped query lands on a low-signal text
  chunk, context composition keeps one lexical sibling and one numeric sibling
  from the same active page. This fixed the isolated MMLong mixed-scan date
  case (`0146`: Aya `0.0 -> 1.0`) without paper/case mappings or production
  index changes. Full backend suite after the fix: **858 passed**.
- Extended the shared visual policy to distinguish metadata questions (for
  example, a figure caption/inventory) from questions asking for content
  inside/from/on a picture, chart, figure or slide. Relation-based phrasing in
  English and Vietnamese now requires rendered pixels, so OCR/caption text
  cannot silently stand in for visual proof. This is a generic query contract,
  not a benchmark- or paper-specific rule.
- Added regression coverage for the MMLong-style “in the picture on slide …”
  wording and for metadata-only caption/list questions. Targeted visual/query
  tests pass; full backend suite: **852 passed**. Gate-E remains open pending
  a stable Sol run.
- Public multimodal attachment text now carries explicit, unambiguous page
  provenance for rendered `page-N` images, matching the production agent's
  attachment mapping. This changes only evidence presentation, not retrieval
  ranking or feature flags.
- The durable `POST /files/process-folder` path now defaults to all parser-
  supported formats (`PDF/DOCX/MD/TXT`); the deprecated `/index-folder`
  request keeps its legacy text/Markdown default for compatibility.
- Post-fix isolated MMLong smoke `gate-e-mmlong-page-companion-r5.json` now
  passes the previously hallucinated image-only/unanswerable case: Aya,
  gold-context and full-context all score **1.0**. The warm repeat
  `gate-e-mmlong-page-companion-r6.json` also scores **1.0**, with visual
  source/attachment rates `1.0`, visual-index reuse about `0.82 s`, retrieval
  about `4.56 s`, and generation about `2.72 s`. This is diagnostic evidence,
  not a Gate-E promotion; the cold visual index took about `275 s` and larger
  stratified coverage is still required.

## Latest verification — 2026-08-31

- Fixed a generic multimodal context-composition gap found in Gate-E r18:
  visual page hits were allowed to sit behind ordinary text chunks and could
  be evicted by the per-document/source budget, even though the visual index
  reported success. `compose_retrieval_context()` now accepts an explicit
  visual-page reservation, recognises visual-page provenance independently of
  coverage mode, and keeps the selected page/image before text fill. The
  public evaluator requests one reserved visual page whenever the visual
  provider returns a hit. Added regression coverage; full backend suite:
  **834 passed**. This does not change production flags (`VISUAL_RETRIEVAL_ENABLED`
  remains false) or canonical indexes.
- The reservation is modality-aware: an over-retrieved visual page is not
  allowed to displace exact text for a plain-text question; a page is reserved
  only when the case declares figure/chart/image/table evidence. This keeps
  visual recall and text recall from being traded by an evaluator shortcut.
- The same reservation is wired into the agent's hybrid and local-fallback
  composition paths, so the benchmark and UI answer paths share the contract.
- Gate-E rows now record whether visual evidence was expected, how many visual
  page sources reached the composed context, and whether image bytes were
  attached. Aggregate reports expose those rates so a “visual index complete”
  result cannot be mistaken for “the model saw an image”.
- Optional ColSmol retrieval is now also invoked for ordinary explicit visual
  questions (not only document-wide page aggregation). It remains gated by
  `VISUAL_RETRIEVAL_ENABLED`, so the default text path and production index are
  unchanged; full backend suite remains **834 passed**.

## Latest verification — 2026-08-30

- Completed the resumed public Gate-E answer-quality run
  `gate-e-answer-quality-r18.json`: 20 cases / 60 Sol generations across
  MMLongBench-Doc V2 (12; 4 native, 4 mixed, 4 image-only) and SPIQA Test C
  (8), with all three modes paired and all 20 visual indexes complete. Aya's
  diagnostic score was `0.5507` on MMLong and `0.4863` token-F1 on SPIQA;
  overall Aya-vs-full delta was `-0.0005`, but image-only/unanswerable and
  SPIQA strata still contain misses. This remains diagnostic only; Gate-E and
  `VISUAL_RETRIEVAL_ENABLED` stay closed.
- The run exposed and fixed two generic correctness gaps: the SPIQA visual
  adapter now reads artifact kind from canonical `metadata_json` (not a
  non-existent `asset_type` column), and successful Aya rows with a failed
  visual provider are retryable on resume. Visual worker auto-device now
  retries SIGSEGV (`-11`) on CPU as well as SIGABRT. Targeted visual/evaluator
  tests pass and no production projection changed. Full backend suite after
  these fixes: **830 passed**.
- Visual Gate-E runtime was hardened after observing a real long-PDF
  bottleneck: ColSmol now scores one candidate pool per query (the processor
  tiles internally) and its isolated worker can use bounded PyTorch CPU
  threads via `COLSMOL_TORCH_THREADS` (default: host logical CPUs). This keeps
  page identity/ranking unchanged while avoiding repeated query dispatch on
  M1 CPU; the flag remains off and the live public run is still diagnostic.
- Fixed two lifecycle correctness gaps exposed by contract tests: canonical
  stage receipts now use the exact document-scoped revision materialization
  fingerprint (so healthy parsed revisions are not falsely reported stale),
  and the agent-owned optional visual worker cache is keyed by its complete
  model/device/config signature and drained during FastAPI shutdown. Full
  backend suite after these fixes: **828 passed**; flags and production
  projections remain unchanged.
- Extended parser-version drift detection to DOCX, so unchanged Word bytes
  are reprocessed when canonical extraction/chunking versions change instead
  of silently reusing stale table/image projections.

- Added a generic visual-evidence sufficiency contract to the agent answer
  graph. Visual-intent turns now distinguish canonical figure/page/image
  evidence from OCR/table text; when no visual artifact is retrieved the
  prompt explicitly forbids inferring what an image depicts. A post-generation
  guard now buffers that rare no-visual path and emits a localized abstention
  instead of exposing a hallucinated visual claim. The same context reaches
  primary and local-fallback generation, with regression coverage. Visual-page
  telemetry also counts page artifacts correctly. Full backend suite after the
  guard: **828 passed**; feature flags and
  production indexes remain unchanged.

- Live Gate-C rerun `retrieval-v2.2-live-r16.json` completed on the
  revision-aware isolated 21-paper clone with Ollama `embeddinggemma:300m`:
  A0/A1/B-full are `Candidate/Evidence/Document = 0.97/0.91/1.00`; the
  diagnostic B-anchor is `0.98/0.93/1.00`. B-full activated no second query
  on this suite, so the measured treatment delta is zero and
  `DUAL_QUERY_RRF_ENABLED` remains false. Warm retrieval P50/P95 were
  approximately 139/309 ms for B-full and the run made zero 9router calls.
  This confirms the baseline gate, not a dual-query improvement.
- A fresh public Sol three-mode smoke (`public-three-mode-sol-smoke-r16.json`)
  completed 2 paired cases (6 generations) in an isolated runtime. The
  SPIQA extracted case matched full-context answer quality (diagnostic F1
  1.0); the image-only MMLong unanswerable case was answered correctly by
  gold/full context but not by Aya (`0.0` vs `1.0`). Aya preparation took
  about 248 s for that 48-page image-only PDF because native local OCR/indexing
  ran before generation. This is an honest Gate-E diagnostic, not a promotion
  score; it keeps visual retrieval/answer-quality work open and does not
  change production flags.
- The public answer-quality evaluator now exposes `--disable-pdf-ocr` only for
  isolated visual diagnostics. Its configuration fingerprint and report record
  the OCR choice, so OCR-on and OCR-off checkpoints cannot be mixed; normal
  production/arbitrary-format evaluation still keeps page-level OCR enabled.
- The same 2-case Sol smoke was repeated with OCR disabled in the isolated
  runtime (`public-three-mode-sol-smoke-r17-noocr.json`). Preparation dropped
  from about 248 s to 28 s for the image-only PDF, while the answer remained
  wrong without visual retrieval; this cleanly attributes the cost to local
  parsing/OCR and the quality gap to missing visual evidence, not to 9router
  generation. The switch is diagnostic only and does not alter production.

- The public three-mode evaluator now emits a paired, case-complete answer
  quality diagnostic for `gold_evidence`, `full_extracted_document` and
  `aya_pipeline`. It reports Aya-vs-full and Aya-vs-gold deltas overall and by
  suite/stratum; missing or failed modes are marked partial rather than
  dropped, and the diagnostic never promotes a visual or production flag.
- Full backend suite after this evaluator change: **810 passed**; compileall,
  `git diff --check` and Graphify synchronization passed. No live provider
  calls were made and production flags remain off.
- Final public evaluator reports are written atomically alongside their
  per-row checkpoints, so an interrupted visual/LLM run cannot publish a
  truncated JSON report.
- Visual retrieval reports explicitly label their result scope as
  `isolated_visual_trial`; `production_eligible` remains false until the
  separate answer-quality, RC and shadow checks are attached.
- Paired answer-quality output now also groups by document profile
  (`native_text`, `mixed_text_image`, `image_only`, or `spiqa_extracted`), so
  file-format failures cannot be hidden inside one aggregate.
- Static audit of the proposed Gate-C v2.2 suite passes all 60-case,
  language/context, qrel-shape and scope checks with zero issues
  (`data/retrieval_eval/internal/dev-v2.2-static-audit-r16.json`). It remains
  `promotion_eligible=false` until independent semantic review is run.
- Semantic qrel review now has an atomic, fingerprinted batch checkpoint;
  successful case reviews resume without duplicate Sol calls and failed
  batches remain retryable. Provider preflight failures are emitted as
  structured non-promotable reports (latest offline check:
  `data/retrieval_eval/internal/dev-v2.2-semantic-audit-provider-check-r16.json`).
  Stale failure entries are cleared after a successful retry, and
  `--no-resume` atomically resets old rows before provider preflight. An
  integration fixture covers failed-batch resume/fresh-run paths. Full backend
  suite: **814 passed**.
- Isolated HTTP/SSE agent regression `e2e-agent-foundation-r15.json` passed
  **7/7 turns** with `cx/gpt-5.6-sol`: 14 persisted history messages, sticky
  ASPIRE focus, valid table/figure evidence, and no transport/tool/model
  fallback. The figure URL was verified as HTTP 200 with PNG bytes.
- The regression first exposed a real portability bug: persisted absolute
  figure paths from the source workspace returned 404 in an isolated clone.
  `RagService.resolve_artifact_path()` now remaps legacy paths by the current
  approved artifact root while rejecting paths outside that root. Three
  resolver tests cover current, legacy and traversal cases.
- Evidence validation now normalizes mathematical Unicode metric headers (for
  example `CCC𝑉`/`CCC 𝐴`) and preserves their dimension identity. This removed
  a false quantitative fallback on a valid ASPIRE table without adding any
  paper-specific mapping.
- Public SPIQA visual sampling is now deterministic, stratum-balanced and
  gold-blind, with bounded question reuse per selected PDF and regression
  coverage for order independence and document caps.
- Full backend suite after the fixes: **807 passed**, 5 existing dependency
  deprecation warnings; `py_compile`, `compileall` and `git diff --check`
  passed. Feature flags remain off and no production index/database changed.
- A 20-case SPIQA visual probe was started in an isolated root. Three rows
  completed before the fourth PDF's CPU ColSmol page indexing was interrupted;
  the `.partial` checkpoint is retained and has no aggregate/promotion score.
- The resumed 20-case SPIQA probe completed all rows: text recall `0.50`,
  visual recall `0.90`, and equal-RRF T+V recall `0.95` (visual gain `+0.45`).
  This is still **not** Gate-E promotion: the corrected gate now requires both
  MMLong native/mixed/image-only PDF profiles and SPIQA coverage. The report
  is explicitly `promotion.passed=false` in
  `data/retrieval_eval/public/results/visual-gate-spiqa-r20-reuse.json`; CPU
  ColSmol visual flag remains off.
- Public evaluator resumption now resolves legacy source-path rows by raw
  content SHA-256 (with source-alias fallback), so moving a copied source does
  not create a false `document_missing` or duplicate projection.
- MMLongBench-Doc V2 visual probe r2 completed six documents (two each of
  `native_text`, `mixed_text_image` and `image_only`) with ColSmol CPU. Text
  recall was `0.75`, visual recall `1.00`, and equal-RRF T+V recall `1.00`;
  the report is still a six-document diagnostic, not a promotion result.
- A strict, LLM-free report merger now combines disjoint public-suite runs only
  when schema, model and embedding fingerprints match. The merged 26-case
  diagnostic (`visual-gate-arbitrary-format-r1.json`) covers both MMLong and
  SPIQA plus all three PDF profiles; it is eligible for an isolated visual
  trial under the current checks, but production visual and shadow gates stay
  closed pending answer-quality/RC validation.
- The public visual evaluator now reports metric scope explicitly: MMLong text
  uses page identity, while flattened SPIQA Markdown text uses canonical
  artifact identity (and exposes an unscored `T_page` diagnostic instead of
  inventing page-1 provenance). Recomputed r5 values are MMLong `T=0.75`,
  SPIQA `T=0.5167`, visual `V=0.9231` and fused `T+V=0.9936` across 26 rows.
  The report also records generic miss counts (`text_no_required_candidate=9`,
  `text_partial_evidence=4`, `visual_partial_evidence=1`,
  `fused_incomplete_evidence=1`) instead of attributing failures to a
  hardcoded paper/case rule.
- The public Sol answer-quality evaluator now writes an atomic, fingerprinted
  `.partial` checkpoint after every case/mode. `--resume` reuses only
  successful rows from the identical selection/model/context configuration;
  failed rows are retried and a changed configuration fails closed. This is
  infrastructure for the pending stratified Gate-E run, not a score.
- Its MMLong selector now rotates through native-text, mixed-text/image and
  image-only PDF profiles, while capping question reuse per document; SPIQA
  selection likewise uses deterministic document/stratum caps. This prevents
  a small answer-quality smoke from being dominated by one easy paper.
- The answer-quality runner preflights local EmbeddingGemma before any Sol
  generation when the Aya arm is selected and emits a structured
  `provider_unavailable` report on provider failure; it never falls back to a
  different embedding or answer model.
- An MMLong v2 second-profile probe was interrupted after one new image-only
  row while the old evaluator process still used path-only lookup; its
  checkpoint is non-promotable. A future resume with the new hash resolver can
  reuse the already committed visual manifest instead of repeating old work.

## Latest verification — 2026-08-28

- Isolated Gate-C rerun `retrieval-v2.2-live-r15.json` completed with the
  approved local `embeddinggemma:300m` provider: A0/A1/B-full are
  `Candidate/Evidence/Document = 0.97/0.91/1.00`; B-anchor is
  `0.98/0.93/1.00` but diagnostic-only. B-full activated `0/60`, so the
  recommendation remains `keep_dual_query_disabled_no_treatment_activation`.
  Retrieval made zero 9router/model calls and did not touch production.
- Isolated Gate-D lifecycle matrix `lifecycle-matrix-r15.json` passed all
  supported PDF/DOCX/MD/TXT retrieval and revision invariants. The deliberate
  corrupt PDF is contained as `partial`; it is not counted as a supported
  format failure.
- A 20-case visual Gate-E attempt was intentionally stopped after the first
  image-only 53-page PDF required about 8.5 minutes of CPU ColSmol indexing.
  Its checkpoint remains `.partial` for audit and no aggregate score was
  emitted. This is a measured resource limitation of the optional visual
  branch, not a Docling/OCR or text-RAG failure; the visual flag stays off.
- A document-reuse Gate-E r2 probe then completed seven rows across one
  image-only and one mixed-text profile (the image-only PDF was encoded once
  and reused for three questions) before the selected native PDF preparation
  became too slow for this foreground run. Its `.partial` checkpoint is kept;
  no aggregate or promotion score is inferred from it.
- Public evaluator now shares the live agent's bounded active-revision page
  projection for document-wide questions; 40-page documents are enumerated,
  documents beyond 64 pages remain explicitly bounded/partial.
- Full backend suite: **793 passed**; desktop `npm run build` passed. No
  production index/database or feature flag changed.
- Public retrieval-only checkpoints now retry failed cases and replace the
  previous per-case row on resume, so transient parser/provider failures cannot
  silently shrink the evaluated population or inflate aggregate scores.
- Both public retrieval runners now preflight the configured local embedding
  provider before cold ingestion. When Ollama is offline they emit a structured
  `provider_unavailable` report immediately and never produce a partial score.
- Isolated SPIQA text/table diagnostic (5 cases, no OCR) completed through the
  durable coordinator: phrase recall `0.60`, canonical artifact recall `0.70`,
  modality-aware support recall `0.80`; table subset artifact recall was `0.50`.
  This is diagnostic only (small sample and retrieval P50 about `1.25 s`), not
  a Gate-C or arbitrary-document promotion result.
- Unchanged retries now resume staged revisions left by a failed local
  projection instead of silently skipping them. Public evaluation preparation
  runs through the isolated durable folder coordinator, and revision-scoped
  vector cache checks cannot reuse an older active revision. LanceDB physical
  row counts are validated after writes when the table schema supports it.
- Gate-C CLI now defaults to the audited, orthogonally balanced
  `evals/retrieval_v2/dev-v2.2.jsonl`; the confounded v2.1 suite remains
  diagnostic-only and is rejected for promotion.
- The earlier provider-outage attempt remains fail-closed and is retained only
  as an operational diagnostic; it is superseded by the completed r15 run
  above.

## Gate-C decision-report correction — 2026-08-28

- The isolated ColSmol worker now treats `device=auto` conservatively: CPU is
  selected unless `COLSMOL_ALLOW_MPS=1` is explicitly set. An opt-in MPS child
  abort (`SIGABRT`/exit 134) is retried once on the same model on CPU, with the
  downgrade reason exposed in worker telemetry. This is a device-safety fix,
  not a model fallback or a visual promotion.
- Visual worker contract verification: **27 targeted tests passed**; the
  production visual feature remains disabled. Full backend suite after this
  change: **789 passed**.
- A one-case isolated SPIQA smoke with `--device auto` completed successfully
  after the fix (`T/V/T+V` page/artifact recall all `1.0`). Worker telemetry
  reports `effective_device=cpu`, visual index 4 entries / ~45 MB. This is a
  crash-safety smoke only; the Gate-E promotion checks correctly remain false
  because one case cannot establish visual gain.
- ColSmol page indexing now batches newly seen pages per bounded worker
  request instead of dispatching one model call per page. Per-page identity,
  atomic embedding files and resumable manifests are unchanged; this targets
  the measured 291-second/53-page CPU bottleneck without changing the model.
- Page aggregation now carries the source document's expected page count and
  marks a fully observed frontier as `partial` when it cannot cover the whole
  document. This prevents a top-k page sample from being presented as a full
  document census.
- Batch-path verification on an isolated four-artifact SPIQA case completed
  successfully (`4/4` indexed, visual recall `1.0`, worker effective device
  CPU). The batch run still took about `33.4 s`; batching removes avoidable
  dispatch overhead but does not make this 256M CPU visual model cheap enough
  to claim a full Gate-E pass.
- Active documents up to the bounded 64-page aggregation frontier now receive
  an exhaustive text-page projection before optional visual fusion; longer
  documents retain ranked frontier sampling and remain explicitly partial.
  This prevents avoidable page-census misses without widening normal fact
  retrieval. Full backend suite: **789 passed**.

- Fixed the evaluator recommendation contract: an audited report is not
  automatically promotion-eligible merely because qrels and the arm metrics
  pass. A treatment must have actually activated and must show the required
  paired gain (or the explicitly permitted latency/call improvement).
- Isolated Gate-C rerun `retrieval-v2.2-live-r14-recommendation.json` used the
  revision-aware 21-paper clone and local `embeddinggemma:300m`; it made zero
  model/9router calls and no production writes. A0/A1/B-full measured
  Candidate/Evidence/Document `0.97/0.91/1.00`; B-anchor was
  `0.98/0.93/1.00` and remains diagnostic.
- B-full activated **0/60** query pairs and had zero candidate gain over A0.
  The report now says `keep_dual_query_disabled_no_treatment_activation`,
  with `promotion_eligible=false` and
  `dual_query_promotion_eligible=false`; `DUAL_QUERY_RRF_ENABLED` remains
  false. This corrects the misleading recommendation in r13 without changing
  retrieval behavior.
- Verification after this evaluator-only fix: **777 backend tests passed**;
  Graphify was synchronized. Gate-C retrieval is green for this audited
  corpus, but it does not by itself close arbitrary-format or visual Gate-E.
- The isolated visual evaluator now exposes an explicit `--disable-pdf-ocr`
  diagnostic mode and runs synchronous Docling/RapidOCR preparation off the
  event loop. This separates ColSmol visual recall from cold OCR cost without
  changing the production/default OCR setting. A 25-case attempt was stopped
  after one pathological public scan exceeded 11 minutes before checkpoint;
  it produced no score and must not be treated as a failed benchmark result.
- Verification after this preparation-path change: **778 backend tests passed**;
  no production index/database or feature flag changed.

## Coverage/visual aggregation hardening (2026-08-28)

- Shared page-coverage merge is now used by both the live agent and the public
  evaluator. Same-document/page provenance is joined before context capping;
  new visual pages are marked as coverage evidence instead of being dropped by
  the ordinary per-document chunk limit.
- Coverage budgets are adaptive but bounded: up to 32 page candidates, 20k
  projected characters and 16 directly attached images. Ordinary fact/table
  requests keep the compact path. The visual flag remains off by default.
- When visual page retrieval is explicitly enabled, the approved 9router
  multimodal client can receive bounded local page data URLs. A generic
  page-level map/reduce contract verifies batches in parallel, rejects foreign
  or incomplete page IDs, and exposes an explicit incomplete status rather
  than turning partial observations into a full-document claim.
- Verification: **774 backend tests passed**, `git diff --check`, py_compile and
  Graphify sync. The public MMLong V2 visual smoke improved page recall from
  `0.385` to `0.692` and the answer from `4` to `8` after widening the
  frontier, but the reference is `13`; this remains a diagnostic failure, not
  a Gate-E promotion result. No production index/DB or feature flag changed.

## Generic page-coverage retrieval pass (2026-08-27)

- Added a conservative VI/EN `DocumentCoveragePolicy` for document-wide
  count/list/aggregation questions. It is opt-in by query shape, does not use
  qrels or paper-specific mappings, and leaves ordinary fact/table retrieval
  budgets unchanged.
- Coverage retrieval now searches a deeper active-revision FTS frontier and
  collapses child chunks to one canonical page unit. Context composition keeps
  bounded multi-page coverage (24 pages / 14k chars by default), preserves page
  provenance, and tells the answer contract to deduplicate by file+page and
  abstain when the page frontier is incomplete. Compact prior retrieval cache
  entries are not reused for these requests.
- Optional ColSmol page hits can be joined with OCR/native page evidence by
  document/page when the visual feature flag is explicitly enabled. Visual
  failure remains derived-channel diagnostics; local canonical text retrieval
  continues and no model fallback is introduced.
- Unit/contract verification for this pass: targeted retrieval/visual/agent
  tests pass; the full suite is still required before RC promotion. The
  MMLong V2 image-only triage showed the prior text/OCR frontier could only
  recover 12 pages, while an isolated CPU ColSmol run recovered 11/13 gold
  pages in top-24. This is diagnostic evidence, not a production score; the
  visual feature remains off and 9router was offline for the attempted Sol
  replay.

## Public visual diagnostic update (2026-08-26)

- Added support for the corrected MMLongBench-Doc-V2 metadata release. The
  isolated profile smoke completed 3/3 cases (native, mixed text/image and
  image-only) with page recall `1.00` for text, ColSmol visual and T+V; V
  Hit@1 was `1.00` on this three-case sample. This is diagnostic only, not a
  promotion score or an answer-quality benchmark.
- Visual reports now include separate orchestrator RSS and worker telemetry
  (PID, committed page count, index bytes and lazy model-load state), so the
  ColSmol child process is not hidden by parent-only memory measurements.
- Verification after this change: full backend suite **752 passed** and visual
  contract tests are included; desktop build and compile/diff checks pass.
  Graphify synchronized at **3,798 nodes / 11,114 edges / 145 communities**.
- Isolated Gate-D lifecycle matrix now passes across native/mixed/table-image
  PDF, DOCX, Markdown and TXT: all six valid sources project and retrieve
  correctly; the intentionally corrupt PDF is contained as a partial job.
  Duplicate bytes, unchanged rescan, content revision and rename invariants
  all hold. This is a format/lifecycle gate, not an arbitrary-PDF quality
  claim.
- Authorized Sol agent-path smoke completed 4/4 through the real production
  retrieval/context/generation path (2 historical MMLong V1 + 2 SPIQA).
  MMLong page recall was `1.00` but answer accuracy was `0.50`; SPIQA
  diagnostic token-F1 was `0.737`. The miss occurred despite retrieved pages
  being correct, so it is an answer/context/visual-reasoning limitation—not
  evidence to tune retrieval by one question. New runs now default to corrected
  MMLong V2. Report: `data/retrieval_eval/public/results/agent-smoke-v2.json`.

## RAG Foundation folder-corpus checkpoint (2026-08-25)

- Implemented additive content identity and revision lifecycle: raw-file
  SHA-256 blobs, source aliases, stable logical documents, revision rows,
  stage receipts, revision-aware chunks/artifacts and a versioned FTS
  projection. Legacy rows remain readable; active revisions are filtered in
  retrieval. No production database or index was migrated during this
  checkpoint.
- Added durable `DocumentProcessingService` jobs with lease/heartbeat,
  retryable failures, startup recovery and bounded worker execution. The
  folder path is discover/hash → parse/readiness → local projections →
  validation → activation; optional VLM/cards/LightRAG/visual work is not on
  the local-search critical path. API status endpoints are available under
  `/files/processing-jobs`, `/files/process-folder` and
  `/files/documents/{document_id}/status`; the Files UI remains intentionally
  unchanged.
- Parser coverage now preserves ordered DOCX paragraphs/tables/embedded
  images and triggers selective OCR for sparse PDF pages with provenance.
  A disabled `VisualRetrievalProvider` contract is present for isolated
  experiments; no ColPali/ColQwen model is enabled or promoted.
- Verification checkpoint: **727 backend tests passed** and modified modules
  compile. A real Ollama-backed projection smoke was correctly blocked because
  `embeddinggemma:300m` was offline; no fallback model was used and no
  production index was changed. Dual-query remains disabled.
- Atomicity hardening now keeps the active revision and its projections during
  a staged re-index. Revision-scoped canonical/vector identities are promoted
  only by `activate_revision()` after local projection; forced same-identity
  rebuilds replace rows in place to preserve existing citations.
- Still pending before claiming the foundation/RC complete: Ollama-backed
  isolated lifecycle smoke, audited Gate-C retrieval rerun, stratified
  arbitrary-format public evaluation, visual A/B measurement, and Release
  Candidate/shadow/rollback gates.

## Retrieval gate / ingestion consistency update (2026-08-24)

- Added an explicit, local-only legacy repair at
  `scripts/backfill_document_cards.py`. A document with canonical SQLite text
  but no `document_card` was correctly skipped by LanceDB; the repair builds a
  deterministic card from stored filename/text and vectorizes *only* repaired
  document IDs. It is idempotent, does not reparse files or call an LLM.
- Repaired the one observed legacy gap (`demo.md`): card + one text vector now
  exist. The direct retrieval dev gate improved from A0 Candidate R@30 `0.82`
  to `0.84` before scope-faithful scoring.
- Evaluation Gate v2.1 now applies Aya's current-turn catalog resolver to raw
  queries containing a unique filename/paper identity. This is not qrel
  injection: a conflicting catalog target fails closed. On the audited 60-case
  suite A0 becomes Candidate R@30 `0.87`, Evidence R@10 `0.72`, Document R@5
  `0.95`; `demo.md` is `1.0/1.0/1.0`. Table/figure are `1.0`, but text and
  multi-hop evidence still miss the acceptance gate. A0/A1/B-full remain tied;
  dual-query stays disabled.
- Important interpretation: direct evaluator multi-paper coverage `0.50` does
  **not** emulate Aya's production compare policy, which decomposes each
  explicitly scoped paper into its own retrieval branch then reserves context
  evidence per paper. Do not change production ranking solely to chase that
  direct-service metric; test the production compare path separately.
- Public multimodal smoke exposed and fixed two generic SPIQA adapter gaps:
  image artifacts named as tables now materialize in `document_tables` with an
  image reference, and a repeat adapter sync preserves VLM-derived visual
  context when the image SHA-256 is unchanged. It invalidates derived context
  when image bytes change. The isolated table case now retrieves Table 1 and
  returns all seven languages; SPIQA three-case diagnostic F1 rose `0.509` →
  `0.820` (gold `0.822`). This is a diagnostic smoke, not an official SPIQA
  score or a general production claim.
- Verification: **715 backend tests passed**. Graphify was refreshed after the
  changes: **3,393 nodes / 9,900 edges / 120 communities**.

## Evaluation Gate v2 implementation (2026-08-23)

- Added `scripts/evaluate_dual_query_retrieval.py`: fail-closed source/content
  fingerprint qrel resolver, A0/A1/B retrieval-only arms, deterministic order
  randomization, embedding-call accounting, modality/document/evidence metrics,
  promotion checks and sealed shadow opt-in. The runner never creates an LLM
  client or calls Sol.
- Added `scripts/build_retrieval_eval_v2.py` and `evals/retrieval_v2/`:
  visible `dev-v2` has 60 balanced cases (6 categories × 10, 30 VI/30 EN),
  and `shadow-v2` has 24 MMLongBench-Doc questions with private page qrels.
  Dev qrel preflight resolves **60/60** against the current SQLite index.
- Corrected the experimental agent wiring so the dual branch receives raw
  `request.task` as `original` and the enriched standalone retrieval query as
  `rewrite`; the production flag remains `DUAL_QUERY_RRF_ENABLED=false`.
- Backend verification after this work: **697 passed**. Live retrieval metrics
  are pending because local Ollama (`127.0.0.1:11434`) was offline; no model
  fallback was attempted and no production index was modified.

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

## PLAN.md implementation checkpoint (2026-08-23)

- Added typed modality-aware evidence contracts in `backend/app/rag/evidence.py`:
  text/table/figure/page tagged candidates, score components, query origins,
  provenance, facet coverage and readiness diagnostics. Existing dict context
  remains compatible; the contract is now available for future tools.
- Added deterministic `DocumentReadinessReport` in
  `backend/app/rag/readiness.py`. Native/OCR/visual-only/unresolved pages are
  classified without claiming an image-only page is searchable. Indexing stores
  the report in document metadata and `/rag/documents/{id}` exposes it as
  `readiness`.
- Added safe original + standalone-query fusion under
  `DUAL_QUERY_RRF_ENABLED=false` by default. The experimental branch runs card
  plus global FTS candidate-document union, bounded dense retrieval, query-aware
  RRF (`k=60`), one optional rerank, and an `EvidenceBundle` audit payload.
  Rewrites that drop required entities are not fused.
- Retrieval cache schema is now `v23`, so old context projections cannot be
  reused across the new evidence/fusion diagnostics.
- Public three-mode evaluator now distinguishes `full_extracted_document` from
  `windowed_long_context`; token budget defaults to 96k input with a 16k output
  reserve and reports token/visual coverage instead of silently calling a
  head+tail truncation “full”.
- Verification after this checkpoint: backend **690 passed**; desktop
  `tsc && vite build` passed. No production index, LightRAG graph, or model
  setting was changed; dual-query remains opt-in pending retrieval A/B.

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
| Numeric evidence | Chặn cross-paper/mixed-doc; parser chịu Docling/Markdown, slash/underscore metrics, hierarchical/multi-metric/transposed VAD cells, Pearson-r và owner scope; validation ledger nạp canonical table theo focus ngay cả khi cache mất card metadata; ambiguity vẫn fail-closed |
| RAG latency | Exact canonical table và một main-result table có caption semantic rõ render deterministic; grounded planner không gọi LLM; non-quantitative answer stream theo validated Markdown block |
| Desktop history | Persist/restore sources cùng `table_id`, `table_index`, `figure_type`; không tự suy Figure N |
| Codex code context | Graphify AST graph + project skill/MCP + Obsidian vault; graph-first, source-file second |

## Regression-v1 RC2 + deep audit (2026-08-23)

- Isolated 21-paper clone, 21/21 evidence cards; answer/router/model đều pin
  `cx/gpt-5.6-sol`, không model fallback.
- RC2 chạy đủ conversational regression-v1 **60/60 pass** qua HTTP/SSE:
  55 `file_qa`, 5 `chat`, 55 `search_local_docs`, 1 visual retrieval, zero
  transport/tool/model fallback. Median total **13.379 s**, median client-first
  delta **5.264 s**, max total **29.095 s**.
- Deep audit không dừng ở top-level pass: RC2 tìm 3 guard interventions ở
  h02/h03/h10. Các root cause được phân loại bằng canonical SQLite evidence và
  local opt-in raw trace, không tune theo paper ID:
  qualifier serialization; cache/card validation-boundary; transposed V/A/D
  matrices; `V(r)/A(r)/D(r)` Pearson headers; bullet owner/condition scope.
- Affected replay cuối: h02/h03 sạch trong 11-turn replay; h10 **4/4** và
  `hidden=[]` trong replay cuối. Không chạy full-60 lần thứ ba chỉ để lấy một
  file aggregate mới sau các deterministic/generalized fixes; RC2 report và
  affected replay reports phải được đọc cùng nhau.
- Validator còn hỗ trợ composite context cell, PDF soft-wrap, byte size khác
  parameter count, model tags như `qwen3.5:4b`, proposed-component ownership và
  long-table query-aware rows. Conflicting conditions vẫn fail-closed.
- Evaluator có `--debug-trace` và persist `run_id`; raw trace vẫn cần cả server
  opt-in lẫn loopback request.
- Verify code cuối: backend **680/680 passed**, desktop production build passed.
- Báo cáo local chính:
  `agent-heldout60-sol-rc2-20260823.json`,
  `agent-deepfix10-debug-sol-20260823.json`,
  `agent-deepfix12-h10-debug-sol-20260823.json` trong isolated regression dir.
- Bộ 60 đã được mở/tune và chỉ là regression-v1; không còn là blind held-out hay
  bằng chứng chất lượng production tuyệt đối.

## Sol stability check while coding with Luna (2026-08-23)

- Giữ vai trò tách biệt: code/verification có thể dùng Luna; answer, router,
  LightRAG và evaluation vẫn pin `cx/gpt-5.6-sol` qua 9router.
- Full dev-20 pass độc lập trên isolated 21-paper clone: **20/20**, 19
  `file_qa` + 1 `chat`, zero transport/tool/model fallback; median total
  **14.984 s**, median client-first delta **5.915 s**, max **25.940 s**.
- Một pass trước đó đạt 17/20 vì hai Sol stream interruption; affected
  MSF-SER/Mamba conversation replay đạt **4/4** ngay sau đó. Đây được ghi là
  provider/generation variability, không gộp thành score 20/20 giả.
- Nested audit của full rerun có một guarded section fallback trong câu
  compare (bốn CCC baseline bị owner ambiguity); các số đó bị loại khỏi output,
  không có unsupported numeric claim. Replay cuối cùng `hidden=[]`.
- Isolated backend đã dừng sau test; production DB/index không thay đổi. Report
  local: `agent-dev20-sol-luna-code-20260823-rerun.json` và
  `agent-dev05-sol-replay-luna-code-20260823.json`.

## Graphify / Obsidian cho Codex

- Graphify `0.9.37` runtime (`0.9.27` project skill) đã update local sau
  Evaluation Gate v2: **3.309 nodes, 9.633 edges, 128
  communities**; code AST
  update không gọi model/9router.
- Codex repo skill nằm ở canonical `.agents/skills/graphify/`; `.codex/config.toml`
  đăng ký Graphify MCP bằng absolute interpreter path và đã pass handshake/tool call.
- `AGENTS.md` bắt buộc graph-first với budget mặc định 1.200 token; chỉ đọc source
  đầy đủ khi verify/debug/edit/security hoặc graph thiếu/stale/ambiguous.
- Obsidian source-code vault ở `obsidian-vault/`: **3.105 generated notes** +
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

## Retrieval Evaluation Gate v2.1 correction — 2026-08-24

- Báo cáo v2 A0/A1/B trước đó bị đánh dấu `invalid_for_promotion`: gold caption
  lọt vào enriched rewrite, typed bundle làm mất table/figure identity, và
  final rerank chưa được đo như một stage riêng. Không tune production theo
  các metric đó.
- v2.1 dùng canonical identity chung cho fusion/bundle/evaluator, candidate
  top-30 + production rerank đúng một lần thành final top-10, và failure
  taxonomy/timing per-stage. Flag `DUAL_QUERY_RRF_ENABLED` vẫn `false`.
- `dev-v2.1.proposed.jsonl` 60 case chỉ là proposal structural, bắt buộc review
  semantic qrel/language/provenance trước khi publish thành suite audited.
  `shadow-v2` đã deprecated; chưa tạo/mở `shadow-v3`.
- Audit r3 bổ sung caption-backed Table/Figure labels (không dùng ingestion
  `artifact_index` làm số hiển thị), structural document coverage cho đủ 21
  tài liệu, `preflight` resolver và review packet portable. Preflight mới:
  **60/60 qrel resolve**, nhưng status vẫn `review_required` /
  `promotion_eligible=false`: parser OCR vẫn có thể chọn heading table/reference
  cho các qrel design, nên không được tự động chuyển proposal thành benchmark.
- `publish-reviewed` giờ từ chối chính generated proposal và yêu cầu audit
  approval, reviewer/timestamp, language/qrel attestation. Review packet:
  `evals/retrieval_v2/dev-v2.1.review-packet.jsonl`.
- Semantic audit đã publish `dev-v2.1.jsonl` với fingerprint
  `dev-v2.1-frozen-rewrite-manual-audit-r1`; 60/60 portable qrel resolve.
  Live retrieval-only result (local EmbeddingGemma, zero model/rewrite call):
  A0/A1/B-full đều Candidate R@30 `0.82`, Evidence R@10 `0.70`, Document
  R@5 `0.93`, multi-paper coverage `0.50`; table/figure evidence `1.00` nhưng
  text `0.50`. Vì vậy không pass/promotion và flag vẫn false.
- B-full activation `0/60`: full frozen rewrite đã preserve đủ raw anchors nên
  generic raw branch bị loại đúng policy, không được diễn giải là RRF thất bại.
  B-anchor activation `38/60` nhưng Evidence R@10 chỉ `0.64`, xác nhận full
  contextual rewrite tốt hơn stripped anchor. Cần raw-signal diagnostic slice
  trước khi kết luận về true dual-query RRF.
- Embedding rerank overlay bị dừng sau vài phút vì local Ollama phải re-embed
  candidate text; không đủ latency budget và không đổi production setting.
- Verification sau audit/report diagnostics: **712 backend tests passed**.
- Local draft smoke chỉ kiểm tra contract, không phải quality score: A0/A1/
  B-full `Candidate R@30=0.84/0.85/0.85`, nhưng proposal chưa audited nên
  không promotion. B-full có một embedding/case vì mọi raw follow-up trong
  proposal đều bị generic-signal guard loại hợp lệ.
- Verification sau audit hardening: **711 backend tests passed**. Không đổi
  embedding (`embeddinggemma:300m`), index, LightRAG hay answer prompt.
## RAG Foundation correctness pass — 2026-08-25

- Đã khóa folder processing theo lifecycle `discover → staged canonical →
  FTS/Lance validation → atomic activation`; job retry/lease/heartbeat và
  idempotency theo snapshot fingerprint được lưu bền trong SQLite. Parse hoặc
  provider lỗi không thể thay thế revision đang active.
- FTS reader dùng `rag_chunks_fts_v2`; Lance projection mới có cột vật lý
  `revision_id` và đẩy active-revision filter vào query. Cards, raw artifact,
  LightRAG provenance và figure enrichment đều từ chối revision staged/retired.
- `POST /files/process-folder` là đường canonical; các endpoint index cũ giữ
  tương thích nhưng đã deprecated và chỉ tạo staged revision. Enrichment VLM,
  evidence-card, LightRAG được enqueue riêng sau local-ready, có privacy gate
  `metadata.enrichment_approved`, không chặn local search và không fallback model.
- Bổ sung DOCX ordered table/image, Markdown local-image approved-root guard,
  selective OCR provenance/readiness (`ready/partial/failed/visual_only/
  unresolved`) và utility `scripts/migrate_lancedb_revision.py` để dựng index
  revision-isolated trong thư mục mới trước khi switch.
- Verification: backend **734 passed**, desktop `tsc && vite build` pass. Chưa
  chạy public/shadow Gate C, chưa bật dual-query, chưa mutate production index.
- Graphify đã đồng bộ sau correctness pass: **3.590 nodes / 11.469 edges**
  (no-cluster AST refresh; semantic document labels không bị gọi lại).

## Isolated lifecycle and Gate-C rerun — 2026-08-25

- Ollama đã online với `embeddinggemma:300m` 768d. Isolated folder smoke
  tạo revision staged, vectorize bằng Ollama, validate và activate thành công;
  local search state `ready`, enrichment `not_requested` khi privacy gate tắt.
- Production SQLite snapshot vẫn là schema cũ, nên evaluator không được phép
  chạy trực tiếp. Đã backup cô lập, chạy identity migration và dựng Lance
  revision-isolated projection trong `/private/tmp`; production DB/Lance không
  bị ghi.
- Sau khi adapter giữ physical Lance `revision_id` trong metadata contract,
  audited Gate-C rerun cho A0/A1/B-full/B-anchor ghi tại
  `data/retrieval_eval/internal/retrieval-v2.1-correctness-pass-isolated-r2.json`:
  A0/A1/B-full Candidate R@30 `0.87`, Evidence R@10 `0.72`, Document R@5
  `0.95`; B-anchor `0.83/0.67/0.95`. Table và figure evidence `1.00`, text
  `0.53`; warm P50/P95 khoảng `103/190 ms`. A1 không tăng, B-full không có
  raw-signal activation, B-anchor kém hơn. Gate chưa pass; dual-query vẫn off.
- Do retrieval gate chưa pass, chưa chạy Sol agent E2E hoặc shadow; không dùng
  kết quả này để tuyên bố production quality. Remaining failures tập trung ở
  text FTS/vector/final-top-k và production-faithful multi-paper decomposition.

## Gate-C benchmark validity correction — 2026-08-25

- v2.1 is now explicitly diagnostic-only: all VI cases were contextual and
  all EN cases standalone, while several categories repeated one raw-query
  template. The evaluator rejects such a suite for promotion even when qrels
  resolve.
- Generated `evals/retrieval_v2/dev-v2.2.proposed.jsonl` with orthogonal
  language/context cells (15 each), varied natural-language question forms,
  unchanged portable evidence locators and no qrel-derived rewrite facts.
  Preflight resolves **60/60**; independent semantic audit is still required.
- Added `scripts/evaluate_agent_policy_retrieval.py` to measure the real
  deterministic per-document comparison and facet-parallel multi-hop policy;
  isolated runtime completed on the migrated index with no LLM calls. On the
  draft (not promotion-valid) v2.2 proposal it measured Candidate R@30 `0.87`,
  Evidence R@10 `0.78`, Document R@5 `1.00`, multi-paper coverage `1.00`,
  modality recall text/table/figure `0.733/0.700/1.000`, warm P50/P95
  `126/203 ms`, and zero scope leaks. This is a policy diagnostic, not a
  production-quality claim: v2.2 still needs independent semantic audit and
  the remaining misses are mainly text FTS/final-cutoff/fusion losses.

- Added deterministic facet query expansion, deeper first-stage candidate
  frontier, coverage-aware final selection, explicit Table/Figure number
  reservation, and section-aware structural priors. On the corrected isolated
  draft v2.2 run, A0 reached Candidate R@30 `0.96`, Evidence R@10 `0.91`,
  Document R@5 `1.00`, text/table/figure `0.85/1.00/1.00`, multi-paper `1.00`,
  and P50/P95 `123/259 ms`; B-full was equivalent and B-anchor was not
  promoted. The report is still `draft_contract_smoke_invalid_for_promotion`
  until semantic audit attestation exists.
- Added `scripts/semantic_audit_retrieval_eval_v22.py`, a separate fail-closed
  reviewer that can inspect compact question/context/evidence packets through
  the explicitly selected Sol route without mutating the suite. It has not
  transmitted paper excerpts or produced an attestation yet; static audit and
  the 736-test suite pass, but promotion is still blocked pending explicit
  privacy approval and review.
- Added a no-LLM public retrieval-only smoke at
  `scripts/evaluate_public_retrieval_only.py`: one image-only MMLongBench PDF
  retrieved its annotated page (`1.00` page recall) after isolated indexing;
  warm reuse avoided a second full parse. Four SPIQA cases ingested without
  failure. Fixed the public adapter so official table/figure rows carry the
  active `revision_id`; otherwise revision-scoped Lance projections silently
  excluded them. Table-image answer recall remains intentionally partial when
  VLM or visual retrieval is disabled. Added a raw-byte unchanged-rescan fast
  path to avoid repeating Docling/OCR work; the changed/new-file identity pass
  also disables OCR so a cold PDF does not traverse RapidOCR twice.
- Added and ran `scripts/evaluate_format_lifecycle_smoke.py` on generated local
  TXT/MD/DOCX fixtures: all reached `ready`, all revision-scoped Lance
  projections validated, Markdown/DOCX table and image counts were preserved,
  and text retrieval passed. Text inside an embedded image remains explicitly
  outside text recall until VLM/visual retrieval is enabled.
- The format smoke also includes a one-page image-only PDF: selective OCR
  recovered its split identifier after a conservative token-boundary repair;
  Lance projection and retrieval passed while readiness stayed `partial`, as
  intended for a sparse OCR page.

## Gate-C semantic-review correction — 2026-08-25

- Sol review of the first v2.2 proposal found **43 accepted, 16 rejected, 1
  uncertain** cases. This was treated as a benchmark/evidence curation
  failure, not as a production retrieval result: several table qrels had only
  captions and some text qrels selected title/chrome chunks.
- Added a source-backed evidence refresh utility that reselects substantive
  task/architecture spans and attaches canonical table markdown/figure
  captions without changing any query or rewrite. The refreshed proposal has
  70/70 evidence groups refreshed with zero source-resolution failures and
  passes the structural 60-case audit.
- After explicit approval to transmit longer canonical excerpts, the final Sol
  review accepted **60/60 (0 rejected, 0 uncertain)**. The suite was published
  through a separate model-attestation path (reviewer is recorded as
  `cx/gpt-5.6-sol (independent semantic audit)`, not a human), and isolated
  qrel preflight resolves **60/60**. This is now the retrieval-evaluation
  suite, not a production quality claim.
- The next A0/A1/B-full/B-anchor run is pending service availability: the
  attempted isolated run failed closed because Ollama EmbeddingGemma and
  9router were offline. No fallback or production mutation occurred;
  `DUAL_QUERY_RRF_ENABLED` remains false.

## Gate-C retrieval rerun — 2026-08-25

The earlier service failure was a Codex sandbox localhost-permission issue.
Ollama was listening and, when checked with local-network permission,
`embeddinggemma:300m` returned 768-dimensional vectors. The isolated
retrieval-only run then completed without 9router calls or production writes:

- A0: Candidate `0.96`, Evidence `0.87`, Document `1.00`, multi-paper `1.00`,
  P50/P95 `128/266 ms`.
- A1: identical retrieval metrics to A0.
- B-full: identical metrics to A0; all raw branches were pruned by the generic
  activation policy (`60` embedding calls).
- B-anchor: Candidate `0.99`, Evidence `0.88`, Document `1.00`, multi-paper
  `1.00`, P50/P95 `181/281 ms`, with `109` embedding calls.

This does not pass promotion: text modality recall remains `0.7833` for A0/A1/
B-full (< `0.80`), and B-full adds no recall over A0. B-anchor remains a
diagnostic arm, not a production rewrite. Dual-query stays disabled.

An authenticated Sol HTTP/SSE seven-turn regression was then run against the
same isolated backend. 9router/Ollama health and all seven transport runs
passed, but the audit correctly failed on answer/focus assertions: ASPIRE
metric follow-up fell back, the WhiSER switch retrieved unrelated papers, the
isolated artifact root had no figure bytes (404), and the first focus event did
not carry the expected IDs despite active state being committed. This is a
diagnostic failure, not a promotion claim; the isolated backend was stopped and
production remains untouched.

The isolated E2E fixture was then aligned with the production directory layout
and the cache contract was hardened. `RETRIEVAL_CACHE_VERSION` is now `v24` and
the key includes raw request obligations (artifact modality, exact table/figure
number and inventory intent), preventing semantically different follow-ups from
sharing a rewritten-query cache entry. The corrected Sol HTTP/SSE run passed
**7/7**: focus switching, Acc/F1/CCC claim validation and figure HTTP 200 all
passed with zero errors/fallbacks. Backend tests are **737 passed**; this does
not enable dual-query or claim Gate-C promotion.

## Gate-C selector/revision compatibility pass — 2026-08-25

- Fixed legacy Lance initial-revision filtering; staged/retired revisions
  remain fail-closed.
- Unified final/context facet inference for older unlabelled text projections;
  table headers such as `Model` are not treated as architecture.
- Isolated Gate-C rerun (no 9router calls, no production writes): A0/A1/B-full
  Candidate `0.96`, Evidence `0.87`, Document `1.00`; table/figure `1.00`,
  text `0.7833`, multi-paper `1.00`, P50/P95 about `131/340 ms`. B-anchor is
  diagnostic at Candidate `0.99`, Evidence `0.89`, text `0.8167`.
- Gate C remains blocked on A0 text modality and three generic FTS/final-
  cutoff misses; dual-query stays false. Full suite: **740 passed**.

## Gate-C lexical/selector pass — 2026-08-26

- Added bounded FTS query variants for late filename/identifier/tail terms;
  semantic retrieval remains one EmbeddingGemma query per case and the index
  was not rebuilt.
- Final selection now preserves a second evidence opportunity per required
  document when the context budget permits, without qrel- or paper-specific
  rules.
- On an isolated migrated SQLite/LanceDB clone, the audited 60-case run now
  measures A0 Candidate/Evidence/Document `0.97/0.91/1.00`, text/table/figure
  `0.85/1.00/1.00`, multi-paper `1.00`, P50/P95 `139/338 ms`. FTS misses are
  gone; six remaining misses are final-cutoff diagnostics.
- A1 and B-full are identical to A0; B-full adds no value and remains off.
  `DUAL_QUERY_RRF_ENABLED=false` is unchanged. No Jina model, re-index,
  9router call or production mutation was used.
- Backend verification: **742 passed**. Graphify refreshed to **3,655 nodes /
  10,698 edges / 142 communities**.
- Sol HTTP/SSE agent regression on the same isolated clone passed **7/7** after
  rewriting clone-only absolute artifact paths; ASPIRE/WhiSER focus, table
  metrics, figure HTTP 200 and evidence validation all passed with zero errors
  or fallback. The isolated backend was stopped; production remains untouched.
## Gate-D format/lifecycle and public evaluator pass — 2026-08-26

- Added `scripts/evaluate_format_lifecycle_matrix.py`, which drives the
  durable folder coordinator in an isolated root across native PDF, mixed
  PDF, image-only/table-image PDF, DOCX, Markdown and TXT plus a corrupt PDF.
  Local projection/retrieval succeeded for all six valid fixtures; the corrupt
  source produced a partial scan without removing active revisions.
- Duplicate bytes, unchanged rescan, content change and rename/move invariants
  all passed. Content change retained the logical document and created a new
  revision; duplicate bytes did not create a second canonical document.
  Report: `data/retrieval_eval/internal/format-lifecycle-matrix-v2.json`.
- Fixed Docling page reconciliation so materially richer native fallback text
  wins over an empty/tiny Docling page export before selective OCR is enabled.
  This is a generic parser-performance fix, not a benchmark-specific rule.
- Public retrieval runner now caches preparation by document and writes a
  resumable `.partial` checkpoint. A four-case SPIQA retrieval-only sample
  completed without model calls: evidence recall `0.625` (figure `1.0`, table
  `0.25`), exposing a real limitation for table-image artifacts whose image
  text is not yet in canonical searchable content.
- Added a provenance-preserving local image-OCR stage for approved Markdown/
  DOCX images and public table-image adapters. OCR text is derived metadata,
  cached by image SHA-256, included in figure/table vector search, and never
  replaces the original image or caption. SPIQA v4 still reports raw evidence
  phrase recall `0.625`, but the previously missed “7 Indian languages” case
  now resolves its referred table and has post-retrieval answer-token coverage
  `1.0`; the source annotation is an explicit marker rather than the answer
  span. This is why both retrieval and annotation quality are reported.
- A one-document MMLong image-only PDF was intentionally stopped after the
  cold OCR path exceeded the bounded smoke window; no result was claimed. This
  demonstrates that arbitrary image-only PDF coverage still needs a durable
  page-OCR/visual-index preparation stage before a large public run.
- After adding raw-PDF/page OCR caching, the same 34-page image-only MMLong
  document resumed and completed: cold index `184,487 ms`, warm retrieval
  `873 ms`, page recall `1.0`. This is valid one-document evidence only; it is
  not a corpus benchmark and confirms why OCR belongs in a background durable
  stage rather than the chat request path.
- The cache is keyed by source SHA-256 and writes atomically per completed
  page, so an interrupted scan resumes without trusting stale text from a
  different file revision.
- Backend suite: **743 passed**. The professional RAG goal is **not complete**:
  Gate C is green for the audited 21-paper suite, while Gate D remains partial
  and visual Gate E/shadow RC are still open.
- Latest verification after image-OCR integration — 2026-08-26: **744
  backend tests passed** and desktop production build passed. Graphify was
  refreshed after code changes. The lifecycle matrix v3 still reports all
  valid-format retrievals and revision invariants green.
- Expanded cached SPIQA retrieval-only diagnostic to **8 cases** (3 figure,
  5 table): raw evidence phrase recall `0.6927` (figure `0.8889`, table
  `0.5750`), answer-token coverage `0.5028`, warm retrieval P50/P95 not a
  production latency claim (`~1.44s`/`~1.88s` in this isolated adapter).
  All 8 cases completed with zero failures. The report is
  `data/retrieval_eval/public/results/public-retrieval-only-spiqa-v6.json`;
  it remains diagnostic, not a Gate-C promotion result.
- Added reproducible MMLong profile selection and explicit per-profile report
  fields (`native_text`, `mixed_text_image`, `image_only`) so input-format
  coverage cannot be confused with question modality. The cached stratified
  smoke selected one document per profile and completed **3/3** with page
  recall `1.0`; profile-level answer-token diagnostics were `1.0/1.0/0.0`
  (native/mixed/image-only). MMLong's answer is often a derived calculation,
  so this token metric is not an answer-accuracy claim; the image-only case
  still demonstrates page retrieval, while chart-cell extraction/visual
  reasoning remains open. Report:
  `data/retrieval_eval/public/results/public-retrieval-only-mmlong-stratified-v1.json`.
- Public profile sampling is still a diagnostic smoke, not a broad benchmark:
  only three MMLong documents were run and no shadow/RC promotion was made.
- Visual Gate-E dependency isolation and resumed benchmark — 2026-08-26:
  ColSmol-256M is pinned and runs in `backend/.venv-visual` through an NDJSON
  worker, while the backend remains on Transformers 4.x. The interrupted
  MMLong build resumed from atomic page/index checkpoints and completed 3/3
  profile cases (native, mixed and image-only): page recall `1.00` for T, V
  and T+V; V Hit@1 `0.667`, Hit@3 `1.00`; warm V P50/P95 `136/140 ms`.
  Cold page indexing took roughly 143–212 s per document and is reported
  separately. SPIQA worker replay completed 4/4 with V and T+V recall `1.00`
  and warm V P50 about `79 ms`. These are diagnostics, not promotion scores;
  no production index/database, model setting or 9router state changed.
  Reports: `data/retrieval_eval/public/results/visual-gate-mmlong-worker-v1.json`
  and `visual-gate-spiqa-worker-v1.json`.
- The visual evaluator now performs a local EmbeddingGemma preflight before
  parsing pending documents, retries failed checkpoint rows on `--resume`, and
  replaces duplicate case rows. With Ollama offline it fails in about 10 s at
  the provider boundary instead of spending minutes in OCR/Docling; no score
  is emitted from that blocked run.
- Public MMLong evaluation now supports the corrected **MMLongBench-Doc-V2**
  release (1,071 questions over 134 documents) while retaining the v1 loader
  for historical report compatibility. New visual-gate runs default to V2;
  the V2 repository is public metadata only and reuses the downloaded
  upstream PDFs by filename. V1/V2 scores are separate and not comparable.
- Final verification after evaluator metadata/test hardening: **751 backend
  tests passed**, `npm run build`, `py_compile` and `git diff --check` passed.
  Graphify is synchronized at **3,794 nodes / 11,107 edges / 144
  communities**. No production index/database or 9router state was changed.
- Expanded the profile-balanced MMLong smoke to **6 documents** (2 native,
  2 mixed, 2 image-only), all completed without failures. Positive-page recall
  was `1.0` on 4 answerable cases; 2 cases are explicitly unanswerable and are
  excluded from that denominator. Profile-level warm retrieval P50 was about
  `1.16s` mixed, `1.33s` image-only and `2.16s` native in this isolated public
  adapter. This is still not a full public benchmark or visual-answer gate.
  Report:
  `data/retrieval_eval/public/results/public-retrieval-only-mmlong-stratified-v2.json`.
- The latest full backend verification remains **751 passed** after adding
  explicit unanswerable-case accounting; Graphify now has **3,691 nodes /
  10,795 edges / 147 communities**.
- Graphify was synchronized again after the visual-worker/evaluator changes:
  **3,794 nodes / 11,107 edges / 144 communities**.

## Latest foundation/evaluation correctness pass — 2026-08-27

- The retrieval evaluator now mirrors production compare decomposition: each
  required document is searched in its own scoped branch, candidates are
  rank-interleaved with canonical identity de-duplication, and final selection
  still runs once over the shared top-30 pool.
- The migrated isolated `dev-v2.2` run
  (`retrieval-v2.2-live-r11-decomposed.json`) reports A0/A1/B-full
  Candidate/Evidence/Document `0.97/0.91/1.00`, text/table/figure
  `0.85/1.00/1.00`, multi-paper `1.00`, P50/P95 about `140/354 ms`, and zero
  unresolved qrels, scope leaks or model calls. B-full is non-inferior but adds
  no recall value; dual-query remains off.
- The evaluator CLI now fails closed on a legacy non-revision-aware database
  instead of scoring stale projections or emitting an opaque SQL traceback.
  Only the isolated clone was migrated for this run; production was untouched.
- Document status no longer reports an old provider/model stage receipt as
  current `ready`; stale receipts remain visible for audit and force the
  current stage back to `pending` until its active fingerprint completes.
- Gate-E reports an explicit promotion decision requiring a minimum
  profile-balanced sample, measurable T+V gain, no per-case text regression,
  complete rows and a healthy worker. A one-case or four-case smoke can never
  be mistaken for a visual promotion result.
- A broader isolated Gate-E run was completed over one MMLongBench-Doc-V2
  case per `native_text`, `mixed_text_image` and `image_only` profile plus four
  SPIQA cases using ColSmol CPU. It is diagnostic only; no production visual
  flag, index or database is changed.

## Public arbitrary-format answer-quality diagnostic — 2026-08-27

- The approved `cx/gpt-5.6-sol` endpoint was reachable and the bounded
  three-mode run completed **21/21 generations** over 3 corrected MMLong V2
  cases and 4 SPIQA cases. No fallback model was used and the isolated
  production index remained untouched.
- Gold-context is an LLM ceiling diagnostic: MMLong official accuracy was
  `0.667`; bounded full extracted-document context scored `0.000` because the
  endpoint accepts text/images rather than native PDF objects.
- Aya's text/table/figure pipeline scored MMLong accuracy `0.333` with mean
  evidence-page recall `0.419`; the SPIQA diagnostic token-F1 was `0.677`
  versus `0.687` with gold evidence. These numbers are not promotion claims,
  but they expose the remaining gap: multi-page chart aggregation and visual
  page context are not yet consistently passed to answer generation.
- Aya cold preparation dominated latency (`229.6 s` median MMLong,
  `26.5 s` median SPIQA) because isolated public PDFs exercised local Docling
  plus selective OCR. Warm retrieval itself stayed about `0.8–1.4 s`; cold
  preparation and warm query latency must remain separate in future reports.
- Report: `data/retrieval_eval/public/results/public-three-mode-v2-bounded.json`.
  Keep `VISUAL_RETRIEVAL_ENABLED=false` and `DUAL_QUERY_RRF_ENABLED=false`;
  the next generic work is visual-page context fusion and a larger stratified
  answer-quality sample, not paper-specific query patches.
- Added a diagnostic-only `--visual-pages` path to the isolated public
  answer runner. It uses the revision-scoped ColSmol worker and never changes
  production flags. On MMLong chart aggregation case `0035`, visual page
  context raised page recall from `0.419` to `0.615` and changed the answer
  from `7` to `8`, but still missed the reference `13`; this is useful
  evidence that page retrieval alone is insufficient for reliable multi-page
  counting. A map/reduce or per-page aggregation policy is still required;
  no case-specific patch was made.
- Local page OCR now shares one lazy RapidOCR engine across image and PDF
  paths, while transient construction failures remain retryable. Backend
  verification after this change is **767 passed**.

## Gate-E broad diagnostic result — 2026-08-27

- Completed the isolated V2 run: **7/7 successful** cases, 300
  revision-scoped visual entries, and a healthy ColSmol worker across native,
  mixed, image-only, figure and table strata.
- Text recall was `0.9286`; visual recall `1.0000`; late-fused recall
  `1.0000` (`+0.0714` over text), with no per-case recall regression. Visual
  warm P50/P95 was `106.7/227.2 ms`; text warm P50/P95 was `1656.7/3221.4 ms`.
- Gate-E promotion is **false** by design: only 7 positive cases were run,
  while the contract requires at least 20 positives and at least two cases per
  profile. ColSmol therefore remains feature-flagged off; this is visual-value
  evidence, not a production-readiness claim.
- The worker now batches eight pages per NDJSON request; the prior unbounded
  request could stall on long PDFs. The final run resumed atomically from its
  manifest without data loss.
- A bounded Sol answer-quality run was attempted but failed closed because
  9router was unreachable; no fallback model was used.
- The local verification set is green after this pass: **766 backend tests**,
  lifecycle matrix and visual-worker contracts, `py_compile`, `git diff
  --check`, desktop production build and Graphify synchronization all pass.

## Visual context correctness checkpoint — 2026-09-01

- Added revision-scoped same-page native-text companions for selected
  figure/table/visual-page hits in `RagService`. The companion is bounded to
  two active-revision text chunks and 1,200 characters, preserving the primary
  artifact identity rather than consuming another retrieval slot. It is now
  used by both the production agent path and the isolated public evaluator.
- Extended the shared visual-intent policy to format-neutral terms such as
  `picture`, `photo`, `screenshot` and `slide`; public MMLong questions that do
  not declare an evidence-source label no longer silently become text-only.
- Verification: targeted retrieval/public tests **52 passed**; full backend
  suite **838 passed**; Graphify synchronized. Production visual and
  dual-query flags remain off.
- Corrected SPIQA isolated visual run `gate-e-spiqa-isolation-r2.json` is
  complete (8 cases, distinct document scopes, visual source/image rates
  1.0). Aya diagnostic token-F1 `0.6054` vs bounded full `0.6790`; table
  stratum remains weak (`0.2108` vs `0.3580`), so Gate-E is still open.
- A cached-index MMLong image-only replay `gate-e-mmlong-page-companion-r2.json`
  reduced cold preparation from ~250 s to ~3.9 s and warm retrieval stayed
  ~4.5 s, but the unanswerable visual case still produced an unsupported
  numeric answer. This is an answer/abstention limitation, not evidence that
  ColSmol or one smoke case is production-ready.

## Visual page-attachment hardening — 2026-09-02

- Explicit `page`/`slide`/`trang` hints now constrain multimodal image
  attachments in both the live agent and isolated public evaluator. Retrieval
  telemetry remains unchanged; only unrelated pages are removed from the
  bounded image budget. If a hint cannot resolve, the original ranked set is
  preserved for diagnosis.
- Visual sufficiency context now carries the same generic page constraint and
  tells the answer model not to transfer visual/numeric claims from another
  page. Public evaluation prompts use the same abstention rule when visual
  evidence is insufficient; no paper or benchmark case is hardcoded.
- Targeted visual/public tests: **147 passed**; full backend suite: **847
  passed**; Graphify synchronized. A cached MMLong replay was attempted but
  failed closed because local 9router was unreachable; no fallback model or
  production mutation occurred.
- Gate-E remains open: exact-page retrieval is now observable and safer, but
  answer quality on image-only/unanswerable and table-heavy public cases still
  needs a reachable Sol run over a larger stratified sample.
- Added a shared distinction between visual metadata and pixel-required
  questions. Caption/inventory requests may use a canonical `figure_id`, while
  appearance/"what does the image show" requests require a rendered image or
  page artifact; otherwise the answer path fails closed. Full backend suite is
  now **847 passed** and Graphify is synchronized.
- Visual multimodal prompts now include a page-aware attachment map in the
  same order as the data URLs. This removes ambiguity when multiple retrieved
  pages are sent to Sol without exposing local absolute paths.
- Canonical `figure`/`image` artifacts with safe local image paths are now
  eligible for the same bounded multimodal attachment path as visual pages;
  table/text retrieval behavior is unchanged.

## Gate-E arbitrary-format measurement — 2026-09-04 (r47/r49)

- Added explicit case-manifest support to the no-LLM public retrieval runner;
  retrieval-only and answer-quality/visual reports now use the same ordered
  frontier instead of independently sampled cases.
- Found and fixed a generic context-composition loss: exact technical anchors
  (acronyms, identifiers and numbers) already in the candidate pool could be
  evicted by long semantic chunks at the prompt budget boundary. Promotion is
  bounded by discriminative anchor frequency and does not use paper names,
  answers or qrels. Replay of MMLong `0307` changed page recall `0 -> 1` and
  answer-token coverage `0.6 -> 1.0`; accounting `0990` stayed page-recall `1`.
- Warm retrieval-only frontier: **20/20 successful**, page recall `0.90` on
  10 answerable MMLong page cases, retrieval-support recall `0.6875`, P50/Pmax
  `151.8/684.3 ms`. SPIQA artifact recall remains low without visual evidence;
  these are diagnostics, not Gate-C metrics.
- Re-ran ColSmol T/V/T+V on the same 20-case frontier with actual production
  reranker configuration (`rerank_enabled=false`). Retrieval recall over 18
  positive cases was T `0.5000`, V `0.7778`, T+V `0.9444`, with no per-case
  T+V regression. Visual query P50/P95 was `102.6/212.7 ms`; index size
  `413 MB`, worker RSS `363 MB`, CPU fallback. Promotion remains **false**:
  only 18 positives and no paired Sol answer-quality contract.
- Full backend suite after these changes: **885 passed**; desktop build and
  Graphify synchronization pass. Both production visual and dual-query flags
  remain false.

## Gate-E answer-quality replay and table-intent parity — 2026-09-04

- Resumed Sol replay `gate-e-answer-quality-r37-postfix-20case.json` completed
  **20/20 cases / 60/60 generations**, model `cx/gpt-5.6-sol`, with no failed
  rows or fallback. This remains a bounded diagnostic, not a promotion gate.
- Overall diagnostic score: Aya `0.4238`, bounded full extracted document
  `0.5312`, gold evidence `0.5719`; Aya delta vs full `-0.1074`. MMLong
  delta vs full `-0.0049` (Aya `0.4118` vs full `0.4167`); SPIQA delta vs
  full `-0.2611` (Aya `0.4418` vs full `0.7029`).
- The report shows a generic failure shape: natural queries about metrics,
  performance, scores or results did not always reserve canonical table
  evidence when they omitted the literal word `table`. The shared detector
  and public evaluator now use the same language-agnostic contract; no
  benchmark answer or paper name is used.
- Added regression tests for natural evaluation wording. Full backend suite is
  now **887 passed** (5 warnings). Production `DUAL_QUERY_RRF_ENABLED` and
  `VISUAL_RETRIEVAL_ENABLED` remain false; reports before this parity fix are
  historical and are not directly comparable.
- Marker matching is token-boundary aware, so generic terms such as `result`,
  `metric` and `value` do not spuriously activate table routing inside words
  such as `resulting` or `valuation`.

## Gate-E retrieval correctness pass — 2026-09-04

- Gate C remains green on the audited 21-paper corpus: A0 Candidate R@30
  `0.97`, Evidence R@10 `0.91`, Document R@5 `1.00`, multi-paper coverage
  `1.00`, warm retrieval P50/P95 `145.6/362.9 ms`; the dual-query flag stays
  off because treatment activation is not yet justified.
- Added a bounded obligation-query layer for explicit ratios, comparisons and
  anaphoric multi-hop questions. Deterministic row-label aliases are kept
  separate from query-derived anchors; they can widen local search but cannot
  certify claims or inject qrel/answer values. Phrase-aware reservation now
  preserves contiguous labels such as `product development` and `total assets`.
- The public evaluator now reports cached active projections honestly and keeps
  production rerank configuration in its fingerprint. Full backend verification
  is **885 passed** and desktop production build passes.
- Gate E remains open. The last valid 20-case Sol report is overall Aya
  diagnostic score `0.5872` (MMLong `0.4951`, SPIQA `0.7253`), with page recall
  mean `0.90`; the visual T+V smoke reached `0.8889` retrieval recall but was
  non-promotable (18 positive cases, answer-quality/profile gates failed).
- A fresh public replay was not claimed on 2026-09-04 because both local
  Ollama and 9router were unreachable at preflight; no fallback model and no
  production mutation were used. Next measurement is a warm, cached replay of
  the same hard multi-evidence cases followed by a new document-balanced set.
- After the services became reachable, isolated replay `r36` of the hard
  Activision ratio case selected pages `47` and `69` (both complementary
  operands) and returned the reference `5.03%`: official score `1.0`, page
  recall `1.0`, warm index `27.2 ms`, retrieval `3.76 s`, first token `2.80 s`.
  This is a one-case causal smoke, not a Gate-E promotion result.

## Gate-E measurement integrity follow-up — 2026-09-04

- Visual Gate-E now records complete T+V wall-clock latency (`text + visual +
  fusion`) and the isolated fusion stage. Merged reports aggregate these
  fields instead of silently emitting null fused latency. This is telemetry
  only; ranking, context, flags and production indexes are unchanged.
- Targeted public-evaluator verification: **48 passed**. The previous full
  backend run was **887 passed** before this telemetry-only change; rerun the
  full suite before RC.
- At the time of this check both local providers were unreachable (`Ollama
  :11434` and `9router :20128`), so the post-table-intent paired Sol replay is
  intentionally pending and no new quality number is claimed.
- [x] Corrected the artifact-only visual contract: a Markdown/adapter source
  with an explicit table/figure image locator now requires the pixel channel;
  native PDF/DOCX structured tables remain text-first. This prevents a visual
  table from being silently evaluated with only flattened Markdown.
- [x] Added regression coverage for artifact-only table images; targeted
  public-evaluator tests remain **48 passed**. A provider-backed answer run is
  still required to measure its quality impact.
- [x] Full backend suite after the visual-contract and latency telemetry
  changes: **887 passed, 5 warnings**; `git diff --check` is clean.

## 2026-09-10 — visual answer-quality checkpoint and embedding transport

- Added bounded Ollama embedding batches (default `16`, configurable via
  `EMBEDDING_BATCH_SIZE`); model, vectors and ranking semantics are unchanged.
  Full backend verification: **891 passed, 5 warnings**.
- Isolated visual-enabled Sol replay `gate-e-answer-quality-r50-visual-completed10.json`
  completed **10 cases / 30 generations**, zero failures. MMLong official
  accuracy: gold `0.70`, bounded full `0.50`, Aya `0.50`; Aya page recall
  `0.875`; Aya median total/first-token latency `7026/4170 ms`.
- A separate 20-case visual replay was stopped after a clean checkpoint at
  **30 successful rows / 10 cases** because a 198-page PDF required prolonged
  ColSmol encoding on M1. The checkpoint is retained for audit and excluded
  from aggregates or promotion claims.
- Gate-E remains open; visual and dual-query flags remain **false** and no
  production index/database was modified.
- Visual enrichment jobs now persist per-batch progress (`processed_pages`,
  `remaining_pages`, indexed/reused/failed counts) under the current lease;
  callback failures are telemetry-only and cannot invalidate committed pages.
  Full backend suite after this change: **891 passed, 5 warnings**.
- Added bounded transient retries for Ollama embedding transport failures,
  configurable with `EMBEDDING_TRANSIENT_RETRIES`; retries never select a
  fallback provider/model. The isolated format lifecycle matrix `r55` passed
  all retrieval checks and identity invariants across native/mixed/table-image
  PDF, DOCX, Markdown, TXT and a deliberately broken PDF (partial isolation).
  The matrix runner now warms Ollama before ingestion and honors durable retry
  backoff instead of mistaking a pending retry for a failed job.
- Visual-enabled SPIQA Test-C replay `r56` completed **8 cases / 24
  generations**, zero failures: Aya diagnostic token-F1 `0.6947` and ROUGE-L
  `0.6791` versus full extracted `0.7094` / `0.6726`. Visual context was
  available and attached on all 8 cases; median retrieval was `319 ms`, while
  cold visual indexing was `32.5 s`. This confirms visual provenance works but
  also keeps the latency cost explicit.

## 2026-09-11 — conversational metric-table follow-up fix

- Fixed a general follow-up failure where a metric-only benchmark question
  could depend on Sol sampling and fall back, or select an ablation table.
  Focused result requests now retain a bounded canonical main-table inventory;
  deterministic selection requires the table to expose every explicitly named
  metric and ranks caption semantics (comparison/performance/benchmark) while
  excluding ablation/statistics/training tables. No paper/table number is
  hardcoded.
- Added regression coverage for `benchmark Acc, F1 and CCC` with competing
  performance and ablation tables. Targeted tests: **98 passed**; full backend
  suite: **892 passed, 5 warnings**.
- Isolated HTTP/SSE replay `r63` completed **7/7 turns**, zero transport/model/
  tool fallback, valid evidence on every RAG turn, figure image `200 image/png`,
  and both benchmark follow-ups selected the canonical comparison table with
  checked Acc/F1/CCC claims.
- Production indexes/databases were not changed; dual-query and visual flags
  remain `false`. Gate-E visual quality and arbitrary-format promotion remain
  open.

## 2026-09-11 — Gate-E positive frontier paired replay

- Added a post-hoc, contract-checked merge utility for disjoint answer-quality
  reports; it makes no model calls and does not mutate an index.
- Paired a 20-positive-case frontier with the visual retrieval report:
  T+V recall `0.95`, T recall `0.55`, no per-case T+V regression, all rows and
  worker health valid. Profiles include image-only, mixed text/image, native
  text, table and figure cases; `figure+table` still has only one case.
- Sol answer-quality is complete for all `20/20` cases (`60/60` generations,
  zero failures). Aya diagnostic score is `0.5749` versus full extracted
  `0.5462` (delta `+0.0287`), but four per-case regressions remain, so the
  answer-quality safety check is false. The report is diagnostic, not a
  promotion claim.
- New image-only MMLong case `0845` is a useful causal result: Aya scored
  `1.0` while full extracted text scored `0.0`, but its cold 13-page visual
  index took about `248.9 s` (total about `549.6 s`). This cost must be treated
  as a background/readiness issue, not hidden in query latency.
- New SPIQA figure case `1703.05260:0001` retrieved a visual page but answered
  `Not answerable` (`0.0` vs full `0.3077`), adding one explicit visual
  failure for diagnosis rather than tuning to the case.
- Gate-E remains **not promotable**: profile minimum and answer-quality safety
  gates are false. `VISUAL_RETRIEVAL_ENABLED=false` and
  `DUAL_QUERY_RRF_ENABLED=false`; production indexes/databases unchanged.
- Current-code replay of the four reported answer regressions was attempted
  after this merge, but provider preflight found both Ollama `:11434` and
  9router `:20128` unreachable. It failed closed with no fallback; no new
  answer-quality number was recorded. Full backend verification remains
  **892 passed, 5 warnings** and the desktop production build passes.
- Added merge-frontier regression tests for both visual and answer-quality
  report utilities; the complete backend suite now passes **894 tests** (5
  warnings). Desktop build remains green.
- Prepared a 21-case profile-balanced visual frontier by adding a second
  `figure+table` case from the existing public replay. Retrieval-only checks
  are `T+V 0.9524` vs `T 0.5476`, no fused regressions, and all six observed
  profiles now meet the minimum count of two. This is still non-promotable
  until the added case has paired Sol answer-quality evidence.
- Added evaluator-code fingerprints to public answer checkpoints/reports and
  post-hoc merges. Resume/merge now fails closed when evaluator code differs,
  preventing stale answer rows from being mixed with a newer retrieval path.
  Full verification after this guard: **894 passed, 5 warnings**.
- Applied the same evaluator fingerprint contract to visual-gate checkpoints
  and merged reports; stale visual rows cannot be resumed or merged into a
  newer evaluator implementation. Targeted evaluator tests remain green
  (`64 passed`).
- Broadened both fingerprints to include all evaluator/adapter files plus the
  backend RAG, retrieval-store, service, SQLite and config dependency trees.
  At that checkpoint the fingerprints were answer `f7bd76eeb9a9ed46` and
  visual `7802433872a508ed`; current fingerprints are answer
  `b1aeb268d28ef0dc` and visual `21fee20d4ad34d5c`. Any backend retrieval
  change invalidates prior rows.
- Marked the pre-fingerprint merged reports `r68`, `r70` and `r73` as
  `invalid_for_promotion` in `evals/retrieval_v2/invalidated_reports.json`.
  Their numbers remain historical audit evidence only; the next provider-backed
  run must regenerate both retrieval and answer rows with the current code.
## 2026-09-11 — current provider-backed context projection replay

- Confirmed local Ollama `embeddinggemma:300m` (768d) and 9router
  `cx/gpt-5.6-sol` reachable; no fallback model was used.
- Added generic structured-evidence promotion in hybrid retrieval: a relevant
  table/figure already present in the candidate frontier can no longer be
  hidden by repeated title/intro chunks for metric/result queries.
- Re-asserted obligation coverage after final projection so ratio/comparison
  queries retain one candidate per complementary operand.
- Deduplicated visual/page projections by canonical `(document_id, page)` at
  both context ordering and final emission; removed boilerplate words from
  exact-match promotion so ordinary question wording does not demote a top
  semantic evidence chunk.
- Fresh Sol Aya-only replays `r76`–`r78` completed 5/5 cases each, zero
  failures, with warm SPIQA retrieval around `0.19–0.29 s`. Table and
  figure+table answers improved, but MMLong ratio `0990` still misses the
  denominator in final context and SPIQA figure `1703.05260:0001` remains
  unanswered. These are current diagnostic failures, not promotion evidence.
- Latest full backend verification: **900 passed, 5 warnings**. Both
  `VISUAL_RETRIEVAL_ENABLED` and `DUAL_QUERY_RRF_ENABLED` remain `false`; no
  production index/database was changed.
- Paired current-code replay `r80` completed **5/5 cases / 15 generations**
  with zero failures using Sol. Aya score was `0.4361` versus full extracted
  `0.3970` (delta `+0.0390`); MMLong ratio `0990` passed `1.0` with both
  evidence pages present. SPIQA diagnostic token-F1 was `0.2951`; one figure
  case remains slightly below its full-context counterpart, so this is a
  positive diagnostic replay, not a Gate-E promotion.
- Audited Gate-C retrieval replay `retrieval-v2.2-current-r7` completed all
  60 cases with no model calls or scope leaks. A0/A1/B-full were identical at
  Candidate R@30 `0.97`, Evidence R@10 `0.91`, Document R@5 `1.00`; B-anchor
  reached `0.98`/`0.93` but is diagnostic only. Warm A0 latency was `141 ms`
  P50 / `360 ms` P95. Since B-full adds no recall and A1 adds no value,
  dual-query remains correctly disabled; this is a passing baseline, not a
  promotion of the treatment.

## 2026-09-11 — current visual Gate-E retrieval frontier

- Added an explicit `--stage-version` option to the isolated visual evaluator.
  It defaults to `foundation-v3`; `foundation-v2` is only for reusing a
  matching isolated projection and is never a production setting.
- Current-code visual retrieval `r81` completed **21/21** cases, all rows
  `ok`, with `no_llm_calls=true`, using `embeddinggemma:300m` (768d) and
  ColSmol CPU. The frontier covers native, mixed, image-only, table, figure
  and figure+table profiles from MMLongBench and SPIQA.
- Aggregate retrieval: T `0.4286`, V `0.8095`, T+V `0.9524`; fused gain
  `+0.5238` with no per-case text regression. Warm median retrieval was
  T/V/T+V `147.8/128.6/268.6 ms`; fused P95 `961.0 ms`. Visual worker health
  was valid (943 entries, about 284 MB, CPU).
- The run remains **non-promotable** because the required paired Sol
  answer-quality frontier (>=20 cases) is not attached. Both
  `VISUAL_RETRIEVAL_ENABLED` and `DUAL_QUERY_RRF_ENABLED` remain `false`.

## 2026-09-11 — foundation-v3 arbitrary-format lifecycle matrix

- Ran `format-lifecycle-matrix-r56` in a disposable runtime through the same
  durable folder coordinator used by production. The fixture contains TXT,
  Markdown (heading/table/code/local image), DOCX (ordered paragraph/table/
  image), native PDF, table-image PDF, mixed native/image PDF and one broken
  PDF.
- All six valid sources reached revision-scoped canonical/FTS/Lance
  projections; the intentionally broken source made the folder job `partial`
  without affecting its siblings. Rescan kept revision IDs stable, duplicate
  bytes created no logical duplicate, content change created a new revision,
  rename retained the logical document, and source aliases were recorded.
- Retrieval targets passed **6/6** (Markdown table, DOCX table, native PDF,
  mixed image, table-image and renamed TXT). The report is
  `data/retrieval_eval/internal/format-lifecycle-matrix-r56.json`; it makes no
  9router calls and does not touch production data.

- Current public Sol regression `gate-e-answer-quality-r82-current-5case.json`
  completed **15/15 generations** (5 cases, zero failures) with the new
  evaluator fingerprint. MMLong `0990` remains `1.0`; SPIQA diagnostic Aya
  score is `0.2530` versus full `0.2702` (delta `-0.0172`). This is useful
  regression evidence but too small and slightly negative for promotion.

## 2026-09-11 — evaluator bounded-time execution

- Added explicit retrieval/isolated-prepare and 9router generation timeouts to
  `evaluate_public_multimodal_three_mode.py`. A slow public PDF or provider
  now records a failed row with `failed_stage` and continues/checkpoints,
  instead of holding the whole frontier indefinitely.
- Timeout values are part of the checkpoint fingerprint, so `--resume` cannot
  mix rows produced under different budgets.
- Targeted evaluator tests pass `17/17`; full backend suite passes `904/904`.
- One real public Sol smoke (`mmlongbench-doc-v2:0409`, gold/full) completed
  `2/2` generations with the timeout contract recorded. It is a connectivity
  smoke only, not a quality or promotion result.
- The first fresh visual `foundation-v3` attempt recorded MMLong `0990` as a
  bounded prepare timeout at 180s; after raising the budget and fixing worker
  cancellation, the same isolated `r85` frontier completed `5/5` with no
  timeout. T/V/T+V recall was `0.20/0.90/1.00` (P50 `125/107/231 ms`).
  Promotion correctly stayed false because this is only five cases, has no
  paired answer-quality evidence and is not a full PDF-profile gate.
- Hardened `ColSmolWorkerClient.close()` so cancellation cannot wait on the
  in-flight worker lock or respawn a killed process. A dedicated cancelled
  request regression passes, and no evaluator/worker process remains orphaned
  after interruption.
- The attempted fresh 21-case `r86` frontier was intentionally stopped while
  a long MMLong prepare remained in progress; its partial/interrupted report is
  invalid for scoring. Visual evaluator prepare now uses the same non-blocking
  timeout boundary as visual worker calls and aborts the frontier safely.
- Final verification after these changes: backend **904/904** and desktop
  production build pass; backend `:7777`, Vite `:5173`, 9router and Ollama
  health all report reachable. No feature flag was promoted.

## 2026-09-12 — bounded visual Gate-E frontier (r87)

- Reused only the isolated `foundation-v3` text/visual runtime and ran the
  current visual evaluator on the next public MMLongBench frontier. The run
  completed 4/5 selected cases; `mmlongbench-doc-v2:0569` hit the 900-second
  visual-index timeout. The resulting report is therefore **partial and
  non-promotable**, not a 21-case benchmark.
- On the four successful positive cases, T/V/T+V recall was `0.75/0.625/1.00`
  and T+V fused gain was `+0.25`. Warm T/V/T+V P50/P95 was approximately
  `554/1172`, `236/285` and `813/1425 ms`, respectively. The report records
  one timeout, unhealthy worker termination, no LLM calls and no production
  mutation at `data/retrieval_eval/public/results/gate-e-visual-r87-foundation-v3-21case-reuse5.json`.
- This confirms a useful diagnostic signal (late T+V fusion recovered the
  required page on these cases) but also confirms that ColSmol preparation is
  still too expensive for a broad foreground frontier on this M1 profile.
  `VISUAL_RETRIEVAL_ENABLED` remains `false`; no visual or dual-query flag was
  promoted. Next work is a pipeline-level prepare/index cost fix or bounded
  resumable batch, followed by the required paired Sol answer-quality run.
- Corrected the evaluator's timeout provenance: page rendering is reported
  under the prepare budget, while visual indexing/search is reported under the
  visual budget. The targeted evaluator suite is now **52/52** after this
  contract fix.

## 2026-09-13 — visual frontier r88 interrupted safely

- Retried the r87 frontier with the same isolated `foundation-v3` runtime so
  committed visual page embeddings could be reused. The run progressed through
  seven cases (`0569` completed after its prior partial index), then the host
  slept while `0315` was still in a long parser/OCR preparation stage.
- After several hours with no checkpoint progress, the parent was interrupted
  safely. The evaluator emitted
  `gate-e-visual-r88-foundation-v3-21case-reuse5.interrupted.json`; no aggregate
  is valid and no evaluator/worker process remains orphaned. Existing isolated
  page batches remain resumable.
- This is an environment/runtime-duration diagnostic, not a quality score.
  Production data and both visual/dual-query flags remain unchanged/off.

## 2026-09-13 — visual Gate-E retrieval r88 completed

- Resumed the isolated `foundation-v3` runtime to completion: **21/21** public
  cases, zero failed rows, zero LLM calls and no production mutation. The
  manifest covers MMLongBench native/mixed/image-only PDFs plus SPIQA
  figure/table/figure+table profiles.
- Aggregate T/V/T+V recall is `0.4762/0.8571/1.0000`; fused gain is `+0.5238`
  with no text regressions. Warm P50/P95 is `194/673 ms` (T), `184/231 ms`
  (V) and `367/873 ms` (T+V). Visual index size is about `362 MB`; ColSmol
  ran on CPU with 768-dimensional EmbeddingGemma as the text comparator.
- All retrieval checks pass, but promotion remains **false** because paired
  Sol answer quality was not yet attached. Visual and dual-query flags stay
  disabled. Full report:
  `data/retrieval_eval/public/results/gate-e-visual-r88-foundation-v3-21case-reuse5.json`.

## 2026-09-13 — paired public answer quality r90

- Ran the same 21-case manifest through gold evidence, bounded full extracted
  context and Aya production retrieval, with visual page attachments and
  `cx/gpt-5.6-sol` via 9router: **63/63 generations**, zero failures.
- MMLong (11 cases): Aya official accuracy `0.7219`, full extracted `0.2727`,
  gold `0.5455`; Aya evidence-page recall `0.9091`. SPIQA (10 cases,
  diagnostic): Aya token-F1 `0.6148` / ROUGE-L `0.5982`, versus full
  `0.6389` / `0.6083`. These are paired diagnostics, not a universal quality
  claim; per-case regressions remain visible in the report.
- Added `--text-runtime-root` so answer evaluation can reuse canonical text
  projections while keeping visual indexes in a separate isolated root. No
  production state changed; flags remain off. Full report:
  `data/retrieval_eval/public/results/gate-e-answer-quality-r90-foundation-v3-21case.json`.
- Full backend verification after this wiring change remains **905 passed, 5
  warnings**; all four local services continue to return HTTP 200.

## 2026-09-13 — active-revision artifact context projection

- Added a generic `RagService.expand_with_page_artifacts()` projection. Visual
  or page hits now receive bounded active-revision table/figure caption,
  markdown/OCR and provenance context before composition. This is shared by
  the live agent and public evaluator; it does not consult qrels or answers.
- Added retirement-isolation and canonical artifact-context tests. Full
  backend verification after the change is **908 passed** (5 dependency
  deprecation warnings); Graphify was synchronized.
- The r94 single-case replay predates this context projection and remains a
  diagnostic low-score result; no new aggregate or promotion decision is
  inferred until a paired frontier is rerun.
- A current-code r95 replay of SPIQA `1704.08960` (3/3 Sol generations)
  improved Aya diagnostic token-F1 from `0.1600` (r94) to `0.4444`, slightly
  above full extracted context `0.4242`. The improvement is evidence that the
  generic projection supplies missing OCR/markdown artifact context; one case
  is not a Gate-E promotion result.

## 2026-09-13 — current-code paired frontier r96 and failure replay r97

- Re-ran the complete isolated 21-case frontier with the page-artifact
  projection: **63/63 Sol generations**, zero failed rows, no production
  mutation. MMLong Aya official accuracy is `0.6310` (11 cases); SPIQA
  diagnostic token-F1 is `0.4378` (10 cases); overall Aya score is `0.5390`
  versus full extracted `0.4402`. This remains a diagnostic benchmark and
  does not pass the visual answer-quality promotion gate.
- The three-case replay of current r96 failure classes completed **9/9**.
  Two previously failing cases recovered to full-context parity, while one
  metrics-summary case remained slightly below full (`0.2308` vs `0.2609`).
  This separates context/retrieval misses from model-output variance; no
  paper-specific rule was added.
- Production visual and dual-query flags remain disabled. Backend/frontend/
  9router/Ollama services are live for UI testing; Graphify is synchronized.

## 2026-09-13 — r98 artifact-row replay and format lifecycle smoke

- Replayed the three r96 failure cases against the current canonical
  page-artifact projection (including first-class table/figure evidence rows):
  **9/9 Sol generations**, zero failed rows. Two cases reached full-context
  parity; the metrics-summary case remained a small diagnostic regression
  (`0.2500` vs `0.2899` token-F1). This is not a promotion result.
- The isolated format lifecycle smoke completed **4/4** local fixtures
  (TXT, Markdown with table/image, DOCX with table/image, mixed/image PDF):
  all projections and expected retrieval contracts passed. The image-only PDF
  is intentionally reported `partial`, not `ready`; its negative retrieval
  expectation is preserved. No 9router calls or production mutation.
- Backend verification after the artifact-row change is **908 passed** with
  five dependency deprecation warnings. Production visual and dual-query flags
  remain disabled; public full-frontier aggregate remains r96 until a complete
  rerun is justified.

## 2026-09-13 — durable canonical progress checkpoint

- `IndexingService.index_folder()` now emits best-effort per-file canonical
  progress to the owning durable job. The coordinator persists current path,
  file type and scan/index/skip/failure counters under the active lease, and
  preserves those counters when the job reaches `complete` or `partial`.
- Lifecycle service tests cover the callback contract; full backend
  verification is **909 passed** with five dependency deprecation warnings.
  Graphify is synchronized at **4,296 nodes / 12,401 edges / 158 communities**.
- Desktop production build remains green (`tsc && vite build`).

## 2026-09-13 — public agent HTTP/SSE scope regression

- Public isolated runtime exposed a real generality bug: filenames normalized
  arXiv IDs as `1907_04152`, while users commonly asked for `1907.04152`.
  Catalog resolution therefore returned no scope and the router rejected local
  retrieval as `unscoped_local_retrieval_rejected`.
- Added separator-insensitive, corpus-ownership-checked identifier aliases
  (no paper-specific mapping). After the fix, the public HTTP/SSE pair passed
  **2/2** with `file_qa`, retrieval evidence on both turns and one stable
  conversation ID; the follow-up retained paper focus.
- Full backend verification after the alias fix is **910 passed** with five
  dependency deprecation warnings. Graphify is synchronized at **4,297 nodes /
  12,404 edges / 149 communities**. The isolated server was stopped;
  production flags and indexes were unchanged.

## 2026-09-13 — stratified public agent regression

- Ran public isolated HTTP/SSE strata for table, figure/page, text, negative
  and cross-document comparison. The first five-case batch had four immediate
  contract passes; the comparison turn produced an empty generation despite
  correct two-document scope and retrieval. A same-input rerun completed with
  `file_qa`, both documents covered, validated evidence and a full comparison
  answer. Treat the initial empty generation as a transient generation
  incident, not as a hidden pass; the repaired run is **5/5 contract-pass after
  rerun**, with the transient retained in the audit notes.

## 2026-09-13 — Gate C retrieval and agent-policy evaluation

- The original `dev-v2.1` was rejected as promotion-unsafe because its
  language/context matrix is confounded. Balanced audited
  `dev-v2.2.model-attested` resolves **60/60** qrels and was evaluated
  retrieval-only with zero model calls.
- A0 passes Candidate/Evidence/Document R@30/R@10/R@5 **0.97/0.91/1.00**;
  modality text/figure/table **0.85/1.00/1.00**, multi-paper **1.00**, scope
  leaks **0**, warm P50/P95 **146/366 ms**.
- Agent policy/decomposition passes **0.98/0.88/1.00**, modality minimum
  **0.833**, multi-paper **1.00**, scope leaks **0**, P50/P95 **151/378 ms**.
- A1 and B-full are identical to A0; B-anchor is slower and diagnostic.
  Keep `DUAL_QUERY_RRF_ENABLED=false`.

## 2026-09-13 — current Gate-E retrieval-only replay

- Replayed the fixed 20-case public visual manifest against the current
  revision-scoped text and ColSmol indexes with no 9router calls. All 20 rows
  completed and 18 were positive/scored.
- Text/visual/T+V evidence recall was **0.500/0.833/1.000**; T+V had no text
  regression in this run. Warm T+V P50/P95 was **370/1,026 ms**.
- The runner correctly kept promotion false because the contract requires at
  least 20 positive cases and a paired Sol answer-quality report. This result
  is diagnostic evidence that visual late fusion helps, not a production
  promotion decision.
- Report: `data/retrieval_eval/public/results/gate-e-visual-current-r21.json`.

## 2026-09-13 — artifact context projection audit

- Added a generic, bounded query-relevant slice of parser-provided
  `artifact_context` to active table/figure evidence rows. This preserves
  surrounding metric/definition prose without copying an entire section into
  the prompt; no benchmark answer or qrel is consulted.
- Targeted context/evaluator tests passed **96/96** (27 RAG + 69 visual/eval).
  Full backend verification remains **910 passed** with five dependency
  deprecation warnings.
- A 3-case Sol replay of the previous SPIQA regressions completed **9/9** with
  no failures, but did not show a reliable aggregate improvement; Gate-E stays
  closed rather than being tuned to those cases.
- Graphify was refreshed to **4,299 nodes / 12,407 edges / 164 communities**.

## 2026-09-13 — arbitrary-format lifecycle matrix

- Ran the isolated durable folder coordinator against generated TXT, Markdown,
  DOCX, native PDF, mixed PDF, table-image PDF and an intentionally corrupt PDF
  using local `embeddinggemma:300m`; no 9router calls or production writes.
- The scan correctly ended `partial` only because of the corrupt source while
  six valid files reached local projections and all six retrieval checks hit.
- Rescan reused the job/revision, duplicate bytes created no logical document,
  content change retained the logical document and created a new revision, and
  rename retained identity. All lifecycle invariants passed.
- Report: `data/retrieval_eval/internal/format-lifecycle-matrix-current.json`.
  This closes the bounded lifecycle smoke, not arbitrary-format Gate D/E.

## 2026-09-16 — current-service and lifecycle revalidation

- Restarted the workspace backend/frontend so the running UI uses the current
  source tree: FastAPI `:7777` and Vite `:5173` both return HTTP 200.
- Ollama is reachable with `embeddinggemma:300m`; 9router is reachable and
  advertises `cx/gpt-5.6-sol`. Completion authorization remains unconfirmed
  (the last direct probe returned an invalidated OAuth token), so no Sol
  answer/E2E claim is made and no fallback model is used.
- Re-ran the isolated lifecycle matrix against generated TXT/MD/DOCX/native
  PDF/mixed PDF/table-image PDF plus one corrupt PDF. Six valid files projected
  and retrieved successfully; the scan was `partial` only for the corrupt
  source. Rescan, duplicate-byte deduplication, content-change revision,
  rename identity, active-revision and source-alias invariants all passed.
- Report: `data/retrieval_eval/internal/format-lifecycle-matrix-current.json`.
  This is fresh lifecycle evidence only; Gate C/E and Sol HTTP/SSE remain open.

## 2026-09-16 — visual active-revision pre-filter

- Fixed the unscoped visual retrieval path: it now resolves every active
  `(document_id, revision_id)` pair and passes that allow-list to the visual
  provider before ranking. Retired/staged visual vectors can no longer occupy
  top-k slots and disappear only after ranking.
- Added a regression fixture covering unscoped multi-document visual search;
  targeted visual tests pass **18/18**, and the full backend suite passes
  **928/928** (five dependency deprecation warnings).
- Restarted the backend so the UI uses this code. Dual-query and visual
  production flags remain off; no production index was rebuilt.

## 2026-09-16 — isolated Lance revision migration validation

- Added fail-closed destination validation to
  `scripts/migrate_lancedb_revision.py`: every copied canonical Lance table
  must expose physical `revision_id`, every row must belong to an active
  SQLite revision, and vector dimensions must agree.
- Ran the migration read-only from the legacy production projection into
  `/private/tmp/aya-lancedb-revision-rc-current-v2`: 21 active documents,
  0 invalid rows, physical revision columns in all four tables, and 768d
  vectors throughout. Counts were cards `21`, text `1055`, tables `60`,
  figures `59`; the lower figure count is preserved rather than fabricated.
- Read-only retrieval smoke on the isolated projection returned the canonical
  ViSEC Table 2, ASPIRE dataset section and MSF-SER result section. Production
  LanceDB remains legacy and is not switched until RC/shadow gates pass.
- Migration contract tests plus full backend suite: **931 passed**.

## 2026-09-16 — Gate C on isolated revision-scoped Lance projection

- Re-ran the audited 60-case Gate C suite (`dev-v2.2`) against the freshly
  migrated isolated Lance projection, with the production SQLite authority and
  local `embeddinggemma:300m`; compare mode made zero LLM calls.
- A0 passed Candidate R@30 **0.97**, Evidence R@10 **0.90**, Document R@5
  **1.00**, modality minimum **0.833**, multi-paper **1.00**, zero forbidden
  scope leaks, and warm total latency P50/P95 **142/299 ms**.
- A1 was unchanged from A0. B-full was also unchanged because its generic
  activation correctly pruned equivalent/non-informative raw variants. The
  diagnostic B-anchor reached Evidence R@10 **0.92** but cost P50/P95
  **216/405 ms** and is not promoted.
- Report: `data/retrieval_eval/internal/retrieval-v2.2-isolated-lance.json`.
  This proves the validated isolated projection preserves Gate C retrieval;
  production LanceDB remains unswitched until shadow and Sol E2E pass.

## 2026-09-16 — stratified public retrieval expansion checkpoint

- Started a 30-case native/mixed/image-only MMLongBench + SPIQA retrieval-only
  frontier in an isolated runtime with local `embeddinggemma:300m`.
- The first image-only case indexed and retrieved successfully, but cold
  public-PDF indexing measured roughly four minutes per new document. The run
  was interrupted after its durable checkpoint to avoid an hours-long
  foreground job; its partial report is explicitly `score_valid=false` and
  must be resumed to completion before any aggregate claim.
- No 9router calls, production writes, or promotion decisions came from this
  checkpoint.

## 2026-09-16 — evidence-card candidate revision isolation

- Fixed `PaperEvidenceService.candidate_sources()` so card generation reads
  only chunks/tables/figures from the document's active revision (with the
  explicit legacy-null compatibility case). Staged and retired artifacts can
  no longer enter a new card candidate pool.
- Added a staged-revision regression fixture; the full backend suite remains
  **931 passed**. Production data and feature flags were unchanged.

## 2026-09-16 — retrieval cache fingerprint revision isolation

- Updated `AgentRunStore.index_fingerprint()` to include the active revision
  identity and to hash only active-revision figures, tables and LightRAG
  provenance. Staged/retired derived rows no longer invalidate or contaminate
  retrieval-cache identity.
- Added regression coverage for staged-artifact exclusion and revision-switch
  invalidation; full backend suite remains **931 passed**.

## 2026-09-16 — LightRAG provenance active-revision isolation

- Tightened the LightRAG parent resolver: a provenance row with a null or old
  revision is rejected once the document has a different active revision.
  Legacy-null mappings remain accepted only for legacy-null documents.
- Added a stale-mapping regression fixture; full backend suite now passes
  **932 tests**. No production graph/index was rewritten.

## 2026-09-21 — live services and arbitrary-format retrieval checkpoint

- Verified FastAPI `:7777` and Vite `:5173` return HTTP 200; 9router exposes
  the approved Sol/Terra/Luna catalog; Ollama exposes `embeddinggemma:300m`.
- A real SSE request reached backend routing and graph completion, then failed
  at 9router with `401 token_revoked`. No fallback model was used; Sol E2E
  remains unverified until the token is refreshed.
- Completed an isolated 12-case public retrieval-only sample across
  native-text, mixed-text/image, image-only and SPIQA inputs: **12/12 rows**,
  zero production writes and zero LLM calls. Page/evidence/artifact recall was
  `0.6667/0.6667/0.6667`, support recall `0.8611`, and warm retrieval P50/max
  `115/220 ms`. This is diagnostic and below promotion gate.
- A larger cold visual attempt was stopped while an image-only PDF was still
  preparing; its partial checkpoint is invalid and excluded from scores.

## 2026-09-21 — resumable visual page rendering

- `render_pdf_page_records()` now atomically checkpoints the revision-scoped
  manifest after every rendered page with `status=partial` and
  `completed_pages/page_count`; the final manifest is `complete` only after
  all pages commit. Existing page pixels remain reusable after restart.
- Visual contract tests pass **18/18** and the full backend suite passes
  **932/932** (five dependency deprecation warnings). This changes recovery
  only; model, ranking, production flags and canonical evidence are unchanged.

## 2026-09-21 — generic multi-artifact visual attachment fix

- Fixed the visual context projection that kept only the first composed page
  even when the visual retriever had additional ranked pages. Enumerated/list
  queries now fill a bounded two-to-four-page visual frontier; singular visual
  asks remain one page and explicit page hints remain authoritative.
- This is query-shape driven and does not inspect gold answers, paper names or
  benchmark IDs. Targeted agent/public evaluator tests pass **135/135** and
  the full backend suite remains **932/932**.
- It is not promoted yet: the Sol paired answer-quality replay must be rerun
  after valid 9router authorization to measure the actual gain.

## 2026-09-21 — visual benchmark configuration audit

- Replayed the cached 21-case visual retrieval frontier with OCR disabled as a
  diagnostic control: T/V/T+V recall `0.5238/0.8571/0.9286`, fused P50/P95
  `232/929 ms`, with one text-regression case. This is **not comparable** to
  the earlier OCR-enabled `1.0000` fused result and is not a promotion score.
- Fixed the visual evaluator contract to persist `pdf_ocr_enabled`, render
  scale, device, local-files-only and stage version in both checkpoint and
  report. Resume now rejects a mismatched configuration instead of silently
  mixing projections.
- Public evaluator tests pass **71/71**; production flags and indexes remain
  unchanged.
- Visual-report merge/answer attachment now requires the persisted
  configuration contract and rejects missing/mismatched OCR or evaluation
  settings. Full backend suite after this guard: **932/932**.

## 2026-09-21 — live service restart and lifecycle r2 confirmation

- FastAPI `:7777` and Vite `:5173` are serving again for UI testing. Ollama is
  reachable with `embeddinggemma:300m`; the 9router process is reachable and
  lists the approved models.
- A real completion authorization check still reports `401 token_revoked` from
  9router. This is a gateway credential state, not a local service outage;
  no fallback model or alternate provider was used.
- Fresh isolated lifecycle report
  `data/retrieval_eval/internal/format-lifecycle-matrix-r2.json` passes all
  six valid-format retrieval checks (`retrieval_all_hit=true`) and all
  revision/dedup/rename/alias invariants. The intentionally corrupt PDF keeps
  the scan `partial`; this is the expected readiness behavior.
- The report uses local `embeddinggemma:300m`, makes no 9router calls and does
  not mutate production indexes. Gate-E Sol answer-quality and shadow claims
  remain open until the gateway token is refreshed.

## 2026-09-21 — current visual frontier replay (retrieval-only)

- Replayed all **21/21** cases from the pinned visual frontier with the current
  evaluator, OCR enabled, `foundation-v3`, local-only ColSmol and the cached
  isolated runtime. No 9router calls or production writes occurred.
- Current page/artifact-key recall: **T 0.5238, V 0.8571, T+V 0.9286**;
  T+V warm P50/P95 was **226.780/936.900 ms**. Worker health was valid
  (`844` indexed pages, CPU fallback, 260 MB visual index).
- The report is
  `data/retrieval_eval/public/results/gate-e-visual-frontier-v2-replay-20260921.json`.
  It remains diagnostic: two fused incomplete-evidence rows and annotation
  page/key alignment issues are visible; no visual flag promotion is claimed.
## 2026-09-21 — printed-page provenance anchor

- Added a conservative `ParsedPage.page_label` alias. It is populated only
  when standalone numeric labels form a sequential run across physical pages;
  physical `page_number` remains canonical for identity, citations and image
  paths.
- The alias is propagated through token/character chunking, chunk metadata,
  FTS lexical content and Lance embedding context. This addresses generic PDF
  layouts where a user-facing printed page differs from the PDF's physical
  index, without paper-specific offsets.
- Bumped the chunk/parser fingerprints so future isolated re-indexes produce a
  new revision rather than silently mixing old projections. Production index
  was not rebuilt or switched.
- Full backend suite: **934/934**. Isolated TXT/MD/DOCX/PDF smoke: all local
  projections and expected retrieval contracts passed; the image-only scan is
  explicitly `partial` and not falsely labelled searchable.
## 2026-09-21 — page-label lifecycle revalidation

- Full isolated matrix after the parser/chunker fingerprint change passed
  `retrieval_all_hit=true` with all dedup/rescan/content-change/rename/alias
  invariants true. Report:
  `data/retrieval_eval/internal/format-lifecycle-matrix-page-label-r3.json`.
- Native/mixed/table-image PDF, DOCX, Markdown and TXT projections remained
  valid; the intentionally corrupt PDF stayed `partial`.

## 2026-09-21 — printed-page visual replay

- Replayed MMLongBench case `mmlongbench-doc-v2:0315` after adding generic
  printed-page aliases. Text retrieval resolved the required physical page 14
  (the question says printed page 13) with page recall `1.0`; it was rank 15,
  so this is a provenance recovery, not a top-k promotion result.
- Visual-only returned physical page 13 and missed the required page; late
  `T+V` fusion retained page 14 at rank 16 and recovered it only in the
  unbounded page set. This confirms page-label handling helps text retrieval
  but does not yet solve visual ranking/fusion generically.
- One-case latency was T/V/T+V `1258/157/1415 ms`; all 54 pages indexed with
  local ColSmol and zero 9router calls. The replay is diagnostic only.

## 2026-09-21 — visual frontier after printed-page resolver

- Replayed the pinned **21-case** visual frontier in the isolated runtime after
  resolving explicit printed page labels per active document/revision. All
  21 rows completed, with no 9router calls and no production writes.
- Aggregate page/artifact recall is now **T 0.6667, V 0.9286, T+V 1.0000**;
  T+V P50/P95 is **226.717/959.474 ms**. There are no T+V regressions and
  the previous page-label regression is gone.
- Visual gate remains deliberately non-promoted: answer-quality pairs are not
  attached yet. Remaining text misses are classified as 3 empty rankings,
  6 no-required-candidate rows and 2 partial-evidence rows; visual has 1
  no-required-candidate and 1 partial row. ColSmol worker is healthy on CPU
  with 844 indexed pages and 259.97 MB index size.

## 2026-09-21 — final local lifecycle/build recheck

- Fresh isolated `format-lifecycle-matrix-final-r1.json` again passed all
  retrieval checks for native/mixed/table-image PDF, DOCX, Markdown and TXT;
  `retrieval_all_hit=true` and every dedup/rescan/content-change/rename/alias
  invariant is true. The intentionally corrupt PDF remains `partial`.
- Desktop production build passed (`tsc && vite build`). No production index,
  database or feature flag was changed.

## 2026-09-21 — visual revision-isolation regression

- Added a regression proving printed-page metadata from a retired revision is
  never used to resolve explicit visual page hints; resolution stays bound to
  the document's active revision.
- Visual contract tests are **20/20** and the full backend suite is now
  **936/936**. Graphify was synchronized after the test/code update.

## 2026-09-22 — 9router Sol recheck and comparison validation

- 9router completion authorization is valid again: `/v1/models` exposes the
  approved routes, a direct `cx/gpt-5.6-sol` completion returned `OK`, and the
  live backend health reported `gateway_reachable=true`, Sol as router/vision
  model, and Ollama reachable for embeddings. Older `token_revoked` notes are
  historical, not the current state.
- Paired public answer-quality run completed **21/21** with Sol. Aya score is
  **0.6251** versus **0.5331** full-extracted baseline (`+0.0919` paired
  delta). The attached visual gate remains non-promotable because individual
  SPIQA table/figure+table rows regress; no production flag changed.
- Seven-turn HTTP/SSE E2E passed **7/7**: sticky focus, table/figure evidence,
  image HTTP 200 and all run statuses were valid with no transport/tool/model
  fallback.
- Twenty-case conversational dev run completed **18/20**. The two failures
  were multi-paper comparison turns that retrieved the right documents but
  failed closed after slow Sol generation/answer validation; no transport
  error or OAuth failure occurred.
- Hardened evidence identity generically: validator now derives conservative
  title initialisms and distinctive filename fragments, and recognizes an
  identity-bound insufficiency row inside a comparison table. This fixes the
  LPMN/Wav2Small acronym replay and is covered by new generic tests; the
  MSF-SER “hai bài đó” group still needs a follow-up generation/coverage fix.
- Backend suite after this validator change: **940/940**; desktop production
  build and `git diff --check` also pass. Graphify was synchronized to the
  updated code graph.
- Follow-up sanitizer recheck: the MSF-SER comparison group passed **4/4**
  after unsafe table cells were removed while retaining document-bound
  abstentions. A subsequent full 20-case run passed **19/20**; the one
  ASPIRE/ViSEC failure was reproduced as a transient Sol generation/coverage
  miss, then a debug replay of that same group passed **4/4**. Treat 19/20 as
  the conservative suite result, not as a 20/20 promotion claim.

## 2026-09-22 — current 9router health and image-backed table path

- [x] Rechecked the live gateway: backend `/health` reports 9router reachable,
  `completion_status=ok`, `completion_authorized=true`, and
  `cx/gpt-5.6-sol` selected for answer/router/vision. A real SSE request through
  `/agent/run/stream` emitted `run.started` and streamed through the Sol route;
  no fallback model was used. Ollama remains only the local embedding provider.
- [x] Fixed the generic canonical image-backed table path: table image paths are
  promoted from metadata, lexical ranking weights caption/OCR above shared
  document artifact inventories, and same-page expansion preserves the
  selected canonical artifact before sibling artifacts.
- [x] Visual attachment and evaluator guards now accept image-backed canonical
  tables while keeping native text-only tables non-visual. An isolated SPIQA
  replay selected the correct Table 3 and attached its image; answer quality on
  that case remains diagnostic only, so Gate E is still not promoted.
- [x] Backend suite is **944/944**; desktop production build and `git diff
  --check` pass. Graphify was synchronized to the updated code graph.
- [ ] Public answer-quality regressions and the conservative 19/20
  conversational result still block visual/production promotion; keep visual
  and dual-query flags off and do not switch production projections.

## 2026-09-22 — bounded optional visual benchmark requests

- [x] Added a separate visual-index timeout to the isolated public evaluator;
  a slow ColSmol worker is aborted and recorded as a visual failure instead of
  blocking the whole answer-quality run indefinitely. The client remains
  reusable for later documents; shutdown remains terminal.
- [x] Timeout, retryability and visual contract tests are covered. Backend
  suite is now **946/946**; `git diff --check` and Graphify sync pass.
- [ ] A live recheck later in this session found the 9router listener timing
  out (and backend/Ollama not responding), so no new Sol benchmark rows are
  valid. Resume Gate E only after `/v1/models`, backend `/health` and Ollama
  embedding preflight all respond; never fall back to another model.

## 2026-09-22 — Gate C audited v2.2 rerun

- [x] With Ollama `embeddinggemma:300m` back online, retrieval-only Gate C ran
  against the audited `dev-v2.2` suite: **60/60 cases**, qrel preflight
  passed, zero model calls, and production corpus/index remained untouched.
- [x] A0 passed the retrieval gate: Candidate R@30 **0.97**, Evidence R@10
  **0.90**, Document R@5 **1.00**, modality floors text **0.8333** / figure
  **1.0** / table **1.0**, multi-paper **1.0**, warm P50/P95 **121/343 ms**,
  zero scope leaks.
- [x] A1 was identical to A0; B-full was also identical (candidate gain
  **0.00**). B-anchor reached Evidence R@10 **0.92** but raised P50/P95 to
  **213/426 ms** and used 129 embedding calls. The evaluator therefore keeps
  `DUAL_QUERY_RRF_ENABLED=false`; this is a baseline Gate-C pass, not a
  dual-query promotion.
- [ ] Gate E remains blocked by the unresponsive 9router listener; no Sol
  answer-quality or shadow result is claimed from this rerun.

- [x] Isolated arbitrary-format lifecycle matrix rerun with local Ollama:
  native/mixed/table-image PDF, DOCX table+image, Markdown+local image and
  TXT all projected and retrieved expected tokens. The intentionally corrupt
  PDF kept the folder job `partial` without deleting healthy documents;
  unchanged rescan, duplicate bytes, content change/new revision and rename
  invariants all passed. `retrieval_all_hit=true` and no production state was
  touched.

- [x] Hardened the ColSmol subprocess boundary: worker operations now call
  synchronous provider transactions inside the child, avoiding executor
  shutdown hangs when a parent aborts a timed-out visual request. Visual
  retrieval smoke completed **3/3** (V recall **1.0**, T+V recall **1.0**).
  A 21-case frontier started with the current runtime cache; first case passed
  and the next large PDF was honestly recorded as `visual_index` timeout, so
  the aggregate remains non-promotable rather than treating partial indexing
  as success. Full backend suite is now **947/947**.

## 2026-09-22 — canonical table-image modality pass

- [x] Normalized adapter rows whose physical storage is `document_figures` but
  whose metadata declares `asset_type=table`. The shared evidence identity now
  exposes a table-shaped contract (including a stable table alias and image
  provenance) across Lance payloads, page expansion, context reservations and
  typed `EvidenceBundle` conversion.
- [x] Relation/importance questions now reserve a bounded two-page visual
  frontier when the retriever has returned multiple relevant pages. The answer
  contract asks for row-to-column preservation and explicit baseline/without-X
  conditions, without paper-specific aliases or answer text.
- [x] Cached Sol replay of the three public regression cases completed **3/3**
  with no fallback. The image-backed external-data table is now typed as a
  canonical table and its answer improved from token-F1 **0.2857** to **0.50**;
  this remains diagnostic, not a Gate-E pass. The importance case receives two
  relevant images, but its lexical benchmark score is still not promotion
  evidence.
- [x] Backend suite is now **945/945**; desktop build and Graphify sync pass.
