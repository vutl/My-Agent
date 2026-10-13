# 04 — Already done (updated 2026-10-10)

## Misspelled names, real titles, formulas, post-verify — 2026-10-14 (Claude)

- Sticky focus survives dataset/metric names; entities switch papers only when
  they resolve to exactly one other document.
- Real catalog titles (`app/catalog/titles.py`, backfill script, applied to
  production with backup); fuzzy + description name resolution; router may
  return `library_document`.
- Wrong table numbers yield to the described table, with a correction note.
- KaTeX formulas; Unicode math letters normalized for search; formula facet.
- `ANSWER_VERIFY_MODE=post`: stream first, one Sol call fixes only flagged
  lines. Live: 6 legacy false flags all kept, 0 errors.
- Quota retry (same model) for 9router round-robin accounts.

## Canonical paper rows, paragraph scopes and completeness diagnostics — 2026-10-10

- Canonical document-ID binding for declared Paper cells preserves model,
  metric and condition constraints. Composite owner/dataset slash delimiter;
  trace compaction before draft-string budget clipping. Generic paired controls.
- Sentence-level document/model binding and sanitizer paragraph containment;
  preserve valid identity cells when unrelated prose names the same model.
- Frozen Sol full-20 remains **19/20**. New affected replay **4/4 rubric** but
  sanitized partial answer is flagged by diagnostic v2, **not quality acceptance**.
  Visible 172 probes remain 8 false accepts/8 false rejects; no retrieval gain.
- Final combined full suite **1,355 passed**, 7 existing warnings. Desktop/diff
  pass; Graphify **5,154 nodes / 15,919 edges**, vault AST synchronized.
  No production mutation; details in [07](07-NUMERIC-GROUNDING-REVIEW.md).

## Bounded diagnostics and local-owner scope — 2026-10-10 (Codex)

- Trace compaction retains draft/failure diagnostics under the redacted byte
  cap; typed paper/model row owners override stale headings only with unique
  canonical provenance. Counted duplicate metric occurrences, packed threshold
  eligibility, OCR decimal offsets and numbered-file heading scope repaired.
- Generic adversarial paired tests; **1,344 backend pass**, desktop build,
  graph/vault synchronized. Visible numeric suite unchanged, not acceptance.
- Sol full-20 **19/20**, failed comparison retained. Subsequent explicit
  packed-cell metric precedence fix passes paired controls; affected replay
  **3/4**. Subsequent separate Paper/Model ownership passes paired controls;
  no further Sol calls, not yet a clean final-source full gate.
  No production mutation or model fallback; details in [07](07-NUMERIC-GROUNDING-REVIEW.md).

## Explicit prose attribution — 2026-10-10 (Codex)

- Source-derived named-subject and counted-noun constraints reject borrowed
  counts, wrong measurement nouns and swapped respectively ordering. Added
  renamed-entity, omitted-descriptor and Vietnamese correct/wrong controls.
- Visible replay **77/85 wrong rejected, 79/87 correct accepted**, three gains
  without regressions; backend **1,334 pass**, desktop build and graph/vault sync.
- Sol HTTP/SSE group **3/4**, retained failed comparison/answer fallback and
  its **42.500 s** latency. Not full RAG acceptance; see [07](07-NUMERIC-GROUNDING-REVIEW.md).

## Row/condition binding and output cleanup — 2026-10-09 (Codex)

- Source/answer numeric columns bind measurements in the same row; explicit
  condition scope stays per clause. Week xfail resolved; red/green adversarial
  tests cover swapped conditions in prose and answer tables.
- General table ambiguity, bullet-owner inheritance and local literal
  inclusion/exclusion; visible regression **74/85 wrong rejected, 79/87 correct
  accepted**, **68/85 paired pass**, ten gains/no regressions versus prior run.
- Sanitizer removes orphan Markdown headers after all body rows are suppressed,
  preserving partial safe rows. Facts remain disabled/no cached-fact gain.
- Backend **1,330 pass**, desktop build; final Sol full run **19/20** (one
  provider disconnect), one bounded affected replay **4/4** with one answer
  retry. Failed run retained; not a blanket full-suite/semantic acceptance.
- Graph/vault synchronized (AST only); no production DB/index mutation.
- [Results and limits](07-NUMERIC-GROUNDING-REVIEW.md).

## Numeric review continuation — 2026-10-09 (Codex)

- Occurrence-scoped numeric checks, literal fact source anchors and durable
  source/model/policy-bound batch jobs. Conservative facts remain disabled.
- General composite benchmark/cross-source scope fixes; frozen Sol affected
  replay **4/4**, backend **1,311 pass / 1 xfail**; graph/vault synchronized.
- Portable 172-probe visible regression + zero-LLM replay runner. Remaining
  **20 false accepts / 9 false rejects** are not hidden or claimed as acceptance.
- [Detailed scope and remaining work](07-NUMERIC-GROUNDING-REVIEW.md).

## Numeric grounding layer — 2026-10-09 (Claude)

- Metric-agnostic number grounding + table-cell context checks, AND-ed into
  the claim validator; 21 fresh cross-domain tests (+1 documented xfail).
- Probe-driven diagnosis: false accepts 22 → 1 on the first blind set.

## P1 context selection + catalog-derived names — 2026-10-08 (Claude)

- VI/generic table roles, capped main-result and reference expansion,
  paged-format-only same-page logic, direct-table marker leak fixed.
- All hardcoded paper aliases replaced by catalog/keyword-derived vocabulary.
- 1,223 tests; local-60 identical, public-36 at baseline; dev-20 12/12 live
  before provider quota.

## P0 fail-closed validator + detector repairs — 2026-10-08 (Claude)

- Validator: same-document section-owner scope for ownerless fallback, label
  value owners, unified negation, qualifier ambiguity, LaTeX metrics. Four
  HEAD-regressing false supports and four pre-existing ones now rejected.
- Detectors: architecture facet, enumerated/result-table intent, visual and
  pixel intent narrowed; reference parser Vietnamese/Roman/sub-panel fixes.
- Test command fixed (pythonpath); **1,210 backend pass**.

## Main-results gathering and evaluator gold-scope removal — 2026-10-08 KST

- Generic main/ablation/statistics purpose, explicit metric/artifact precedence
  and analytical evidence gathering independent of direct rendering. Red/green
  fixtures, **1,168 backend pass**, desktop build and graph/vault sync.
- Public helper frontier **0.375 -> 0.40278**, one win/no regressions; still
  below acceptance and does not measure actual answer pixels or semantics.
- Full Sol dev-20 **20/20 rubric**, main comparison now retrieves Table 3 rather
  than only an ablation. SHA-bound manual review keeps a remaining misleading
  mixed-paper table heading visible; no blanket semantic acceptance claim.
- Removed qrel document IDs from final selection after cross-process ranking
  drift exposed unordered gold scope. Fresh runtime-only local 60 retains
  **0.97/0.90/1.00**, P50/P95 **124.7/341.8 ms**. Old affected reports invalidated;
  B-full still inactive/no benefit, production unchanged and feature flags off.

## Facet-role reservation and schema-owned identity — 2026-10-08 KST

- Typed dataset/architecture artifact reservations and contextual-task noun
  filtering improve the same public 36-case helper frontier **0.1944 -> 0.375**;
  seven wins, no regressions, zero LLM calls/catalog mutation. Not answer quality.
- Packed summary cells retain model/protocol under arbitrary header names;
  namespaced metric headers still parse as numeric columns. Document fragments
  from incidental dataset text cannot override a structural paper section;
  declared owner cells/full catalog names still can. Adversarial fixture was
  red before the scope repair; prior false-pass draft now rejected offline.
- Local audited 60-case gate maintains **0.97/0.90/1.00** recall, P50/P95
  **130/370 ms**; B-full has no activation/gain, so stays off. Final backend
  **1,157 pass**, 191 focused guard pass, desktop build and graph/vault sync.
- Full Sol dev-20 **20/20 rubric** revealed a hidden wrong-section table during
  manual audit; final-source affected replay **4/4**, ownership repaired in the
  observed answer. Missing MSF-SER main-results source and broader semantic
  acceptance remain open. Preserved all failed/partial/audited reports.

## Artifact purpose and measurement ownership — 2026-10-08 KST

- Fixed own-artifact purpose classification, raw facet obligations and sibling
  injection provenance/anchor indexing; retained dataset tables as datasets.
- Fixed repeated metric dimension offsets, composite owner/split schemas,
  model-family versus experiment scope, union of per-document source owners,
  and descriptive-label/reference context binding. Adversarial fixtures retain
  rejection of wrong models, splits, values, loss negation and hidden assertions.
- Latest focused **185 tests pass**. Previous full **1,137 pass** predates final
  fixes; offline canonical-table replay validates **34/34** old-draft claims.
  Do not conflate that with a fresh E2E or general semantic pass.
- Recorded local recall **0.97/0.90/1.00**, full dev-20's rubric-only pass and
  public helper regression **0.2222 -> 0.1944**. Failed reports remain intact;
  no dual/visual promotion, index migration or production writes.
- AST graph synchronized; isolated backend resumed after verified refusal.

## Facet-directed expansion and schema-owned projection — 2026-10-08 KST

- Removed blanket architecture expansion for explicit dataset, loss, limitations
  and contribution asks; kept mixed-facet anchors and bounded, proven-gap retry.
- Projected long-table rows with their own schema and dataset/group labels,
  including later repeated headers and separate Markdown blocks. Numeric guard
  remains strict on wrong metric labels; no paper/qrel-specific mapping.
- Full **1,129 backend tests pass**, desktop build and AST sync pass. Audited
  local retrieval rerun retains A0 **0.97/0.90/1.00** recall, not an increase.
- Completed full Sol SSE dev-20, **20/20 rubric**, median **9.21 s / 2.99 s**
  total/visible; immutable source/index fingerprints checked. Dataset source
  repair and dimensional-result preservation observed in actual answers.
- Preserved manual semantic failures and unchanged public helper score 0.2222;
  no broad quality/production acceptance inferred from green unit/rubric checks.


## Approved pixels, implicit inventory and live gateway — 2026-10-08 KST

- Canonical answer-image permission is separate from visual retrieval activation;
  active ownership, stored policy, immutable-path/digest checks, actual format
  verification and post-read authorization prevent stale/spoofed attachments.
  Labels match accepted bytes; debug trace stores source/revision/image digests.
- Generic dataset/numeric asks no longer inject whole focused table inventories
  or remove companion figures. Canonical reader results retain revision IDs.
- Transport errors retain exception type even when HTTP exception text is empty;
  genuine gateway hang was verified and same-version router restarted safely.
  Final native Sol check passes (first delta 1.405 s); backend/frontend reachable.
- Regression fixtures cover permission spoofing/revocation, old revisions, path
  substitution, changed/corrupt image bytes, ordering, missing DB, activation
  during read, page manifest authority and pixels-vs-UI-path guard behavior.
  Full **1,118 backend tests pass**, desktop build pass, AST graph updated.
- Preserved negative evidence: public helper artifact score 0.2222, agent smoke
  3/4 on rubric only, one numeric-guard fallback and a semantically unsupported
  dataset answer. These are NOT broad quality wins or acceptance/promotion.


## Revision-owned enrichment context and native frontier audit — 2026-10-06

- Added failing synthetic active/staged/retired/legacy fixtures before fixing
  VLM context ownership, card isolation, page image selection, completion,
  stale revision cache tags and activation-during-provider-call behavior.
  Developed in an isolated code copy while public evaluation was live; applied
  the tested source patch only after that frozen run exited successfully.
- Same-clone real context audit repaired 121 missing section contexts and 40
  missing nearby-table contexts across 136 active figures, with no data writes.
- Completed native public-60 and preserved weak final artifact/page scores.
  Added a fail-closed boundary audit with 12 tests; it does not modify scoring,
  exclude implicit asks, blame an embedding without evidence or call an LLM.
- Actual Sol VLM 2-asset public smoke verifies revision-tagged persistence and
  unchanged pixels/active revision. Two-turn HTTP/SSE smoke after backend reload
  preserves focus and streams; not promoted into a full semantic acceptance.
- Full backend **1,092 passed**, focused **75 passed**, desktop build passed,
  Graphify AST synchronized. Production/model/index defaults unchanged; native
  public final-context losses and wider acceptance requirements remain open.

## Public image input fidelity — 2026-10-06

- Corrected SPIQA public adapter input from 224px distorted thumbnails to
  checksum-pinned original images, for all authored artifacts and all modes.
  No silent lower-resolution fallback; added missing-source/native-pixel tests.
- Source/revision fingerprints preserve changed image bytes. Answer telemetry
  records actual dimensions and report contract declares the native policy.
  Historical thumbnail runs are retained separately; no ranking win claimed.
- Audited all 48 assets of six frozen public probe papers; paired real OCR
  inspection confirms native glyphs recover far better than thumbnail scaling.
- **101 focused tests / 1,060 full backend tests passed**; public source full
  checksum/ZIP validation passed; AST graph synchronized. Same 12-case Sol
  three-mode diagnostic completed 36/36, but aggregate quality did not improve.
- Probe audit exposed a second dedup path dropping alias policy rebuilds.
  Fixed both logical-ID lookups and added active-asset fail-closed protection
  before disk writes. Fresh/existing alias tests and real-Lance crash/retry
  verify retained active evidence. Final **113 focused / 1,065 full tests pass**,
  then the extra real-Lance alias variant passes separately. Public-60 native
  retrieval is running in another fresh clone; no acceptance claim.
- Preserved the native Sol probe's six old-revision mutations in a separate
  lifecycle audit, not silently presenting its quality result as valid RC data.
  Restarted only the owned isolated backend to load these fixes; production
  untouched, 9router/Ollama left running, feature flags still off.

## Condition identity and negative fixtures — 2026-10-06

- Added synthetic schema/owner/config tests, including wrong encoder/loss,
  negation, multi-document role collisions, ambiguous vectors, loss-only tables,
  missing context, duration-vs-metric numbers and uppercase subscripts.
- Canonical source schema, not numeric lookup or paper-specific rules, binds
  vector conditions and explicit owners. Missing context does not invent a
  condition; unknown models are still rejected and parameter claims still guarded.
- Audited local retrieval rerun: A0 **0.97 candidate / 0.90 evidence / 1.00
  document recall**, **119/348 ms** P50/P95. Four arms measured, no LLM calls.
  Treatment adds no candidate gain; dual-query remains disabled.
- First SSE replay **5/5 rubric** exposed two empty tables in independent
  structural audit, so no answer-completeness claim. Preserved failed content;
  final four-turn replay is **4/4**, no empty tables and twelve supported claims
  in the comparison, one attempt and no fallback. Still not semantic acceptance:
  CRAB's final conclusion is missing and evidence coverage is ablation-heavy.
- Final backend **1,058 passed**, desktop build passed; AST graph synchronized.

## Wide-table and metric semantics — 2026-10-06

- Repaired model-in-header ownership without breaking legacy row owners and
  repeated condition groups; added cross-owner/dataset substitution tests.
- Trace-based diagnosis fixed redundant intrinsic weighting conditions for
  WA/UA/WAR/UAR, while keeping the canonical metrics distinct. Not an evidence
  substitution, new model, re-index or paper mapping.
- Actual saved evidence + table now validates twelve UA/WA claims. Sol SSE
  replay shows the full three-row results table, thirteen supported claims,
  no block suppression/fallback. Earlier 4/6 family replay remains a failure.
- Full suite **1,030 passed**, desktop build passed, Graphify synchronized;
  service recovery uses the same isolated clone and leaves production unchanged.

## Context identity and metric-schema repair — 2026-10-05

- Context now deduplicates canonical artifacts, not entire pages: distinct
  figures/crops on the same page survive; aliases and duplicate pages collapse.
  SPIQA pixel selection prioritizes selected canonical source attachments,
  without consulting gold filenames. Frozen-code helper omits secrets and
  production SQLite paths; eval roots remain explicit isolated roots.
- Completed frozen public-60 paired retrieval and 12-case/36-generation
  before/after answer probes. Preserved regressions and diagnostic limitations;
  these results do not justify feature promotion or arbitrary-format readiness.
- Root-cause trace and canonical rows reproduced WA=fold-count false claims
  and unsupported valid values from transposed caption-only metric tables.
  Fixed explicit-schema precedence and owner/condition-preserving transpose
  extraction, with changed-owner/value/fold rejection tests (not paper rules).
- **1,028 backend tests passed**; Graphify AST synchronized. Six affected agent
  turns are replaying on the isolated backend before completeness can be claimed.

## Authored attachment revision/projection correctness — 2026-10-05

- Confirmed active canonical rows and physical Lance artifact projections
  diverged on nine SPIQA sources; repaired the pre-activation materialization
  order, attachment revision fingerprint, revision-scoped IDs/image snapshots
  and physical identity validation. No question-specific ranking/scorer tweak.
- Added real isolated-Lance identity-corruption and staged-adapter crash tests;
  previous active revision and same-byte VLM descriptions remain preserved.
  Focused suite **105 passed**.
- Public retrieval replay completed **21/21**, candidate artifact recall
  **1.00**, evidence phrase recall **0.8738**, query-scoped artifact support
  **1.00**, final-code P50 **135.12 ms**. Final artifact recall **0.50** remains diagnostic;
  no unseen/general-production quality claim.
- Hardened public checkpoint resume with shared atomic writes, evaluator/
  config/source/annotation checksums and all authored attachment bytes.
- Added the same physical-identity guard to manual vector activation; legacy
  empty/null revision compatibility retained, and duplicate physical rows fail
  validation even if their ID sets are identical. Desktop build passed.
- Final full suite **996 passed**, desktop build and AST graph update completed.
  Earlier regression-60
  interruption recorded as incomplete, not a pass.

## Fresh baseline and report eligibility checks — 2026-10-04

- Shared fail-closed visual report eligibility across merge, metric migration
  and answer attachment; enforce scored/declaration frontier identity and keep
  posthoc subsets diagnostic. No paper-specific logic or model changes.
- Full suite **991 passed**, latest pairing guard **25 focused tests passed**,
  desktop build passed. Graphify AST updated (JSON zero-node warnings retained).
- Current-code audited Gate C A0 passed **0.97/0.90/1.00** recall and
  **240/877 ms** P50/P95; Sol dev-20 **20/20**, separate figure/focus/history
  E2E **7/7** with real HTTP image bytes. All through isolated 21-paper clone.
- Public production-OCR frontier and visible regression-60 are ongoing, not
  completed tests. No shadow opening, production mutation or feature promotion.

## Materialization policy and checkpoint correctness — 2026-10-04

- Added OCR/config-sensitive revision and folder-job identity, plus immutable
  active-policy checks before isolated projection reuse.
- Added atomic visual checkpoint writes, resume-contract rejection and
  incomplete/OCR-disabled promotion guards to the main evaluator.
- Verified **982 backend tests**, all isolated lifecycle invariants and **6/6**
  fixture retrieval hits. These are correctness checks, not proof of broad
  public-document quality.
- Authenticated Sol completion and model listing through 9router both HTTP 200.


## Visual backpressure safeguards — 2026-09-24

- Added active-revision direct resolution for explicit page/slide hints before
  semantic visual search; this is a general scope fast path, not a
  paper-specific rule.
- Bounded exact ColSmol score batches to 16 pages and capped render/worker
  batches at 2M/4M pixels to avoid pathological page dimensions or giant image
  batches exhausting the local machine during optional background indexing.
- Full backend suite after the safeguards: **975/975**. These changes do not
  produce a valid visual-retrieval score; the feature remains disabled pending
  a completed stratified T/V/T+V gate.

## Current visual-provider smoke boundary — 2026-09-24

- Fresh ColSmol evaluation was fail-closed when the first MMLong visual index
  exceeded the 180-second budget: **0/1 completed**, no recall score emitted.
  Report:
  ``data/retrieval_eval/public/results/visual-gate-current-smoke-20260924.json``.
- The failure is recorded as a cold visual-index/provider latency boundary;
  it does not affect text/table/page retrieval or the 9router Sol path. Visual
  retrieval remains off and is not promoted from cached historical reports.
- A resume checkpoint later finished the first image-only case with T/V/T+V
  recall **1.0/1.0/1.0** in **165 s** of visual indexing, then was interrupted
  before the suite completed. The partial result is not a promotion score.

## Durable local/enrichment worker separation — 2026-09-24

- Split the folder-processing scheduler into bounded local and enrichment
  lanes (one worker each). Optional ColSmol/VLM/cards/graph work no longer
  queues ahead of canonical parse → FTS → LanceDB activation for new folders.
- SQLite leases and idempotency remain the concurrency authority. Full backend
  suite passed **972/972**; desktop production build passed. The isolated
  format lifecycle matrix also retained all dedup/revision/rename invariants
  and all six expected retrieval hits. Visual retrieval remains disabled
  pending Gate E.

## Arbitrary-format lifecycle smoke — 2026-09-24

- Isolated generated PDF/DOCX/MD/TXT fixtures passed projection and retrieval
  contracts **4/4**. DOCX/Markdown table+image provenance was retained; the
  image-only scan PDF was correctly reported ``partial`` rather than falsely
  marked text-ready. Report:
  ``data/retrieval_eval/internal/format-lifecycle-smoke-20260924.json``.

## Audited Gate C dev-v2.2 — 2026-09-24

- Completed A0/A1/B-full/B-anchor on 60 audited cases (50 positive, 10
  negative), with zero unresolved qrels, scope leaks or LLM calls.
- A0 passed candidate/evidence/document **0.97/0.90/1.00**, modality minimum
  **0.8333**, multi-paper **1.0**, and warm P50/P95 **122.697/356.881 ms**.
- A1/B-full added no gain. B-anchor's evidence gain to **0.91** cost
  P50/P95 **203.213/414.213 ms**, so promotion was denied and the dual-query
  flag remains false. Report:
  ``data/retrieval_eval/internal/gate-c-dev-v2.2-20260924.json``.

## Current-code Sol HTTP/SSE regression — 2026-09-24

- Isolated backend `:7792` replay passed **7/7** turns with Sol: focus was
  preserved, table evidence validated, and the Figure 1 attachment returned
  HTTP 200 with bytes. Report:
  ``data/retrieval_eval/internal/e2e-agent-conversation-v6-20260924.json``.
- The clone was stopped after completion; no production mutation or fallback
  model was used.

## Shadow-v3 sealed preflight — 2026-09-24

- Structural validation passes for 24 questions, 23 positive qrels, one
  abstention and six profile/modality buckets. No shadow run was opened;
  fingerprints are retained for the eventual RC-only evaluation.

## Candidate/final context replay — 2026-09-24

- A generic metric-projection heuristic was tested and rejected objectively:
  it lowered final evidence phrase recall from **0.8595** to **0.8024**. The
  heuristic was removed rather than tuned to public cases.
- Post-rollback report:
  ``data/retrieval_eval/public/results/public-retrieval-artifact-v11-reverted-context-20260924.json``.
  On 21/21 cases, candidate page/evidence/artifact recall was
  **1.000/0.9833/1.000**, final evidence phrase recall **0.8595**, query-scoped
  artifact support **1.000**, support recall **0.9500**, and retrieval latency
  P50/max **109.988/540.858 ms**. Gate C/E remains open because this is a
  diagnostic public slice and final-context completeness is below the target.
- Targeted regression suite after rollback: **123 passed**. 9router Sol and
  local Ollama EmbeddingGemma preflight both passed; no model fallback or
  production mutation occurred.

## Canonical artifact ranking correction and public v5 frontier — 2026-09-24

- Structured evidence promotion now scores canonical artifact fields rather
  than inherited whole-page context, deduplicates table/figure image aliases,
  and preserves canonical artifact identity through the normalized result
  shape. The logic is generic and does not map any paper to a fixed figure or
  table number.
- Public isolated frontier: **21/21** successful; page recall **1.0**,
  candidate page recall **1.0**, evidence recall **0.8595**, final-context
  artifact recall **0.50**, candidate artifact recall **1.0**, support recall
  **0.95**, retrieval P50 **111.43 ms**, max **563.76 ms**.
- The candidate/final split shows canonical artifact generation and candidate
  retrieval are not the current bottleneck; context modality projection and
  answer support remain to be validated. Full Gate C/E is still open. Backend
  suite is **970/970**.

## Authenticated Sol conversation regression — 2026-09-24

- Live HTTP/SSE regression completed **7/7** turns with
  ``cx/gpt-5.6-sol`` through 9router and no fallback or transport errors.
- The run preserved paper focus across a casual interlude and cross-document
  switch, returned validated table evidence, and returned a real figure
  attachment (HTTP 200). This is an agent-wiring smoke, not a retrieval-gate
  pass.
- Report:
  ``data/retrieval_eval/internal/e2e-agent-conversation-v4-20260924.json``.

## Post-artifact-correction E2E replay — 2026-09-24

- Current-code isolated backend replay passed **7/7** HTTP/SSE turns using
  ``cx/gpt-5.6-sol``. No run/transport/model fallback occurred; focus,
  validated table evidence and figure delivery (HTTP 200) all passed.
- Report:
  ``data/retrieval_eval/internal/e2e-agent-conversation-v5-20260924.json``.

## Revision-scoped page provenance and fresh frontier — 2026-09-24

- Added canonical page rows for trusted printed labels on textless pages;
  they never enter FTS or embeddings. Projection validators and Lance builds
  now ignore these rows when counting/indexing text chunks.
- Fresh public frontier: **21/21** successful, page recall **1.0**, retrieval
  P50 **127.97 ms**, max **553.70 ms**; figure/table evidence remains below
  the promotion gate (0.625/0.7778). Report:
  ``data/retrieval_eval/public/results/public-retrieval-page-row-v4-21-20260924.json``.
- Local arbitrary-format smoke (PDF/DOCX/MD/TXT) passed all projection and
  contract checks; scan PDF is explicitly ``partial`` rather than falsely
  marked text-ready. Full backend suite: **969/969**.

## Ingestion-time page-label OCR boundary — 2026-09-24

- Moved bounded printed-page footer OCR into PDF ingestion. Labels are cached
  by immutable source hash and attached to the staged revision before FTS/
  Lance activation; retrieval consumes persisted metadata only.
- Removed the synchronous query-time OCR fallback from ``RagService``. Missing
  labels on old revisions therefore use the conservative physical-page
  fallback rather than blocking a query or hiding a multi-second OCR cost.
- The old on-demand replay remains useful evidence for mapping correctness,
  but its cold latency is not a production retrieval measurement. A fresh
  isolated re-index must produce the next authoritative 21-case report.
- Image-only pages with a trusted printed label now receive a lightweight
  revision-scoped ``chunk_type=page`` provenance row. It is excluded from FTS
  and Lance embeddings, but lets explicit page hints resolve the correct
  physical page. Isolated 0315 replay now passes page recall **1.0** with
  retrieval **119 ms** after ingestion.

## Sparse printed-label inference — 2026-09-24

- Added generic interpolation for Arabic page labels observed at sampled OCR
  pages, allowing image-only intermediate pages to resolve printed references
  without a document-specific page offset or full-footer OCR pass.
- Parser fingerprints were bumped and regression tests cover the physical
  page 14 / printed page 13 pattern. Targeted tests passed (63); full backend
  suite passed **967/967**. The public frontier still needs an isolated
  re-index before its metrics can be updated.

## On-demand printed-label replay — 2026-09-24

- Added a bounded cached OCR fallback for explicit page queries whose active
  revision lacks a printed label. MMLong case 0315 now reaches physical page
  14, with candidate/retrieved/final page recall **1.0**. Retrieval is
  **166 ms** after cache/engine warmup (cold first call **9.30 s**); repeated
  Sol answers varied between ``2`` and ``3``, so answer variance remains a
  separate gate from retrieval.

## Explicit page candidates for image-only pages — 2026-09-24

- Added revision-scoped canonical page candidates for user-supplied
  ``page/slide`` hints, including pages with empty OCR/text content. The
  normal hybrid result schema now carries ``explicit_page_hint`` evidence;
  no gold annotation is consulted.
- Targeted public rerun improved MMLong candidate evidence recall to
  ``0.9091`` and made the explicit slides 12/14 case fully recoverable. The
  prior page miss was a generic printed-label mapping miss in case 0315;
  source audit confirms the public qrel is coherent (printed 13 = physical
  14), and the bounded on-demand resolver now recovers it.
- Full backend suite after the final code change: **965 passed**, 5 dependency
  deprecation warnings. Production index and feature flags were not changed.

## Corrected candidate-frontier evaluator and rerun — 2026-09-24

- Candidate page recall now uses the first-stage ``candidate_results`` rather
  than the final context projection; retrieved text and final multimodal
  evidence are separate fields.
- Cached-index public rerun completed 21/21 with zero failures:
  MMLong candidate evidence recall ``0.8182``, final evidence recall
  ``0.8636`` and SPIQA diagnostic token-F1 ``0.6281``. The candidate gate is
  still open; no feature flag or production index was changed.
- Report:
  ``data/retrieval_eval/public/results/gate-e-postfix-metrics-21-20260924.json``.

## Authoritative post-fix public frontier — 2026-09-23

- Completed the full isolated 21-case MMLongBench-Doc/SPIQA frontier with
  zero failed rows and authenticated ``cx/gpt-5.6-sol`` generation.
- The evaluator now records candidate text-page recall separately from final
  evidence-page recall including only the attachments actually selected by
  retrieval. MMLong reached candidate ``0.7727`` and final ``0.8636``;
  SPIQA diagnostic token-F1 was ``0.6535``. The retrieval gate remains open
  because candidate recall and visual source coverage are insufficient.
- No production index, dual-query flag or visual-retrieval flag was promoted.
  Report: ``data/retrieval_eval/public/results/gate-e-postfix-21-20260923.json``.

## Local visual OCR and conditional multi-hop verification — 2026-09-23

- Confirmed authenticated ``cx/gpt-5.6-sol`` completion through the live
  9router gateway; no router fallback is enabled.
- Added generic local OCR for sparse image-only visual artifacts and
  deterministic independent obligations for conditional multi-hop queries.
- Fresh isolated MMLong case 0413 now returns ``45%`` with official score
  ``1.0`` and evidence-page recall ``1.0``; no production index was changed.
- Full backend verification after the patch: **964 passed**; the complete
  public shadow still requires a post-change rerun before any gate claim.
- Isolated folder lifecycle recheck passed all six supported-format retrieval
  probes and identity/revision invariants; corrupt PDF stayed explicitly
  ``partial``.  No production projection was changed.
- Fresh backend HTTP/SSE Sol smoke streamed multiple answer deltas for an
  ASPIRE dataset query, kept the active paper focus and passed evidence
  validation without fallback.
- Post-OCR public frontier completed 21/21 with zero failed rows: MMLong
  official accuracy 0.9091 and SPIQA diagnostic token-F1 0.6352.  The
  evidence-page recall gate remains open at 0.7273, so no feature promotion
  was made.
- Generic page-superlative coverage was validated on MMLong 0537 end-to-end;
  official score and candidate/final page recall were all 1.0.
- Generic comparison obligation reservation and creator/view aliases were
  validated on MMLong 0494: answer ``5%``, score/page recall 1.0, retrieval
  reduced to ~301 ms without a planner model call.

## Printed-page provenance and gateway verification — 2026-09-23

- Confirmed live 9router ``cx/gpt-5.6-sol`` health and authorization; local
  Ollama remains the embedding provider only.
- Added shared printed-label parsing for standalone labels and facing-page
  ranges, learned page stride validation, and propagated labels through FTS,
  vector, page-coverage and multimodal attachment paths.
- Fresh isolated Sol replay of three explicit-page MMLong cases completed
  3/3; Page 14 on the facing-page PDF correctly resolved to physical page 9
  and the affected answer matched its reference. Targeted tests: **120 pass**.
- Full 24-case public frontier and arbitrary-format gates remain open; flags
  ``DUAL_QUERY_RRF_ENABLED`` and ``VISUAL_RETRIEVAL_ENABLED`` stay disabled.

## Folder lifecycle deferred activation — 2026-09-16

- Propagated the folder coordinator's `defer_activation` contract explicitly
  into `IndexingService._index_file`; staged revisions cannot become active
  during parsing and are activated only after local projection validation.
- Direct file indexing behavior remains compatible. Identity/processing
  regressions: **22 passed**; no production projection was switched.

## Arbitrary-format lifecycle revalidation — 2026-09-21

- Host-side isolated matrix passed native/mixed/table-image PDF, DOCX,
  Markdown and TXT retrieval plus dedup/rescan/change/rename invariants;
  corrupt PDF is reported as `partial`.
- Restored two accidentally deleted tracked MMLong PDFs so the public loader
  remains dataset-faithful; the three previously failing loader/sampling tests
  pass without changing benchmark semantics.
- Full backend suite: **932 passed**; desktop production build passed;
  backend/frontend local health: HTTP 200.

## Explicit artifact routing/context preservation — 2026-09-16

- Numbered/labelled artifact queries now recover the owning active document
  before vector routing when cards do not identify it; generic artifact
  requests remain scoped to the normal card/FTS frontier.
- Context composition now ranks the exact requested `Figure N`/`Table N`
  ahead of visually similar candidates already in the retrieved frontier.
- Added metadata label/index terms for caption-poor canonical artifacts and
  regression tests for routing plus context reservation.
- Isolated MMLong Figure 89 replay now selects page 118 (rather than page 82)
  without paper-specific logic. Full backend suite: **923 passed**.

## Gold-independent evaluator and lifecycle revalidation — 2026-09-16

- Public visual routing no longer uses gold artifact/source labels for boost,
  reservation or pixel guard; text-backed and image-only behavior is tested.
- Full backend suite: **914 passed**. Isolated PDF/DOCX/MD/TXT lifecycle
  matrix passed retrieval and identity invariants; corrupt input stayed
  `partial`. No production index or feature flag changed.

## Revision-aware Lance migration audit — 2026-09-16

- Ran the migration utility against the backend's real `data/sqlite/app.db`
  and production Lance source, writing only to a disposable `/private/tmp`
  target. The target has physical `revision_id` columns on all four vector
  tables and all 1,195 rows resolve to the corresponding active SQLite
  revision (zero mismatches).
- Corrected `rag_status.py` so a legacy production Lance projection is shown
  as migration-pending rather than conflated with a legacy SQLite schema.
- Added a real-Lance regression asserting revision columns and persisted card
  counts in isolated projections; vector/lifecycle targeted suite: **25
  passed**. Production Lance was not mutated or promoted.
- Replayed retrieval-only Gate C using the isolated revision-aware projection;
  its A0/A1/B-full and B-anchor recall matched the current baseline, with no
  unresolved qrels or scope leaks. Sol E2E and production switch remain open.
- Hardened the migration utility to reject a non-empty Lance source paired
  with a DB lacking active revisions; the wrong-DB regression and the
  authoritative-active-revision path both pass (**3 migration tests**).
- Full backend verification after this change: **918 passed** (5 dependency
  deprecation warnings).
- Removed the utility's implicit `init_db()` side effect; a read-only SQLite
  authority connection is covered by a regression test (**4 migration tests**).

## Public format diagnostic — 2026-09-16

- A no-LLM, no-OCR stratified MMLong/SPIQA sample completed 9/9 through the
  isolated production coordinator. The report is retained as a limitation
  diagnostic (`page_recall=0.333`, `evidence_recall=0.639`), not a gate pass.
- ColSmol local-only indexing was stopped after a measured high-CPU first case;
  no production visual flag or index was changed.

## Active artifact lexical recovery — 2026-09-14

- Added a bounded `RagService` fallback that searches canonical table/figure
  metadata only within already selected active documents. Results use the same
  `table:<id>`/`figure:<id>` identity and RRF payload as other channels.
- Added revision-isolation coverage for active versus retired figure rows and
  fixed the table UNION projection to respect the actual SQLite schema.
- Full backend verification: **912 passed**. This is a correctness recovery
  path; it does not change production embedding/indexes or promote visual or
  dual-query retrieval.
- Replayed Gate-C `dev-v2.2` with the fallback gated to explicit structured
  queries: A0 Candidate/Evidence/Document `0.97/0.91/1.00`, warm P50/P95
  `147/368 ms`; no regression versus the audited baseline.
- Re-ran the isolated folder lifecycle matrix across native/mixed/table-image
  PDF, DOCX, Markdown and TXT. All expected local hits and lifecycle
  invariants passed; corrupt input remains honestly `partial`.
- Replayed the paired 21-case public visual frontier with 63 Sol generations;
  all completed without fallback. Aya scored `0.7115` overall versus full
  context `0.4342`, but SPIQA remained slightly below full context and three
  per-case regressions kept Gate E closed.

## Generic visual-modality intent coverage — 2026-09-13

- Expanded visual intent/pixel policy for generic multimodal terminology
  (`visual grounding`, `pixel`, `multimodal`, `image-text`, `vision`) without
  paper- or benchmark-specific mappings; caption/list requests remain
  answerable from metadata.
- Regression and full backend verification: **911 passed**.

## Gate-C v2.2 replay — 2026-09-13

- Audited 60-case retrieval-only replay completed without LLM calls or
  production writes. A0 met the retrieval/latency thresholds (Candidate
  0.97, Evidence 0.91, Document 1.00, P50/P95 148/361 ms).
- B-full showed no delta over A0; B-anchor's small recall gain did not justify
  its extra latency/embedding work. `DUAL_QUERY_RRF_ENABLED` stays false.

## Shadow-v3 sealing — 2026-09-13

- Created a deterministic 24-case public shadow manifest spanning MMLong
  native/mixed/image-only and SPIQA figure/table strata, with 24 distinct
  documents excluded from the visible Gate-E frontier. It is answer-free and
  remains unopened pending RC freeze.
- Generated the corresponding ignored private qrel file with page/artifact
  locators and a checksum recorded in the manifest; no answers are tracked.
- Added and ran a fail-closed validator for the manifest/questions/private
  qrels; it confirms the 24-case matrix and sealed `opened=false` state.

## Multi-page visual context reservation — 2026-09-03

- Added one shared evidence-derived reservation contract for visual page
  context. Exact multi-page/slide requests preserve all retrieved hinted pages
  in the live agent and evaluator instead of reserving a single page.
- Added helper/composer regression tests; full backend verification: **873
  passed**. A Sol isolated smoke for MMLong `0409` attached pages 12 and 14
  and recovered page recall to `1.0`.
- This fixes context presentation after retrieval; it does not promote
  ColSmol or claim arbitrary-format production readiness.

## Gate-E Sol replay — 2026-09-03

- Replayed the exact 20-case public frontier (60 Sol generations) after the
  reservation fix; all rows completed with no fallback. MMLong Aya `0.4951`
  vs full `0.5000`; SPIQA diagnostic token-F1 `0.7253` vs full `0.6919`.
- Strict visual/answer join remains non-promotable (`T+V=0.8889`, 18 positive
  rows, answer/profile checks open). No production visual or dual-query flag
  was changed.

## Gate-E exact paired frontier — 2026-09-03

- Added strict case-manifest resolution and a no-LLM join utility so visual
  retrieval and answer-quality reports can only be combined when their ordered
  case IDs match exactly. The 20-case run is stored as
  `gate-e-combined-r21.json`; it correctly keeps promotion false because two
  SPIQA table answers regress and only 18 rows are positive.
- Added `--case-ids-file` and `--text-runtime-root` to the visual runner. This
  prevents a false “paired” run caused by different runtime layouts or two
  independent samplers, and permits reuse of an existing isolated text index.

## Stack status observability — 2026-09-03

- Hardened `scripts/rag_status.py` to show the 9router/LLM and Ollama/
  EmbeddingGemma split, current feature flags, parser readiness, and
  revision-aware projection status. Legacy production SQLite is reported as
  migration-pending until an isolated RC is approved.

## Gate-E cache-aware visual smoke and status contract — 2026-09-03

- Re-ran the isolated 7-case visual comparator from the existing ColSmol
  cache: `T=0.9286`, `V=1.0000`, `T+V=1.0000`; no 9router calls or production
  writes. This remains a diagnostic smoke (<20 cases), not a promotion.
- Document processing status now exposes parser `readiness_state` and its
  readiness payload independently from local FTS/Lance state. Full backend
  suite: **864 passed**; desktop production build passed.

## Visual readiness fail-closed contract — 2026-09-03

- Image-only/readiness-required turns now share the evaluator's visual
  sufficiency guard with explicit chart/figure questions. Missing page pixels
  cannot be replaced by OCR/text for a visual claim.
- Regression and full backend verification: **862 passed**; desktop build
  passed; production visual and dual-query flags remain off.

## Revision-owned paper evidence cards — 2026-09-03

- Added revision-scoped card/facet/item projections and an idempotent legacy
  backfill. Active card lookup is revision-filtered; the historical unsuffixed
  tables remain only as a compatibility mirror.
- Verified active-revision switching and full backend suite **863 passed**.

## Image-only modality activation — 2026-09-02

- The isolated multimodal evaluator now derives visual-channel necessity from
  the canonical projection: a native PDF with zero text chunks is treated as
  visual-required even for a fact/table-shaped question. This prevents an
  image-only document from reaching Sol without its pixels.
- The rule is projection/readiness based and answer-blind; it does not map a
  paper, page, caption or benchmark case. Mixed/native text documents retain
  the explicit visual-intent gate so optional visual hits cannot displace
  exact text.
- Cached ColSmol + `cx/gpt-5.6-sol` replay of MMLong `0845` returned `5`
  (official score `1.0`) with one visual page source and four page images.
  This remains isolated diagnostic evidence; `VISUAL_RETRIEVAL_ENABLED` is
  still false in production pending stratified Gate-E.

## Visual attachment scope — 2026-09-02

- Non-coverage visual answers now prefer ranked visual pages and deduplicate
  alternate rendered filenames by page identity. Singular requests are capped
  to one page; comparison/all-page requests retain a bounded multi-page set.
- The mixed-scan MMLong visual replay moved from four/duplicate attachments to
  one selected page image. This is an evidence-presentation fix, not a
  benchmark/case mapping, and the visual feature remains isolated.
- The live agent now consults active-revision readiness when visual retrieval
  is enabled, activating page retrieval for visual-only PDFs even on
  fact-shaped questions. Figure-asset expansion remains explicit-intent gated.

## Exact-value sibling context — 2026-09-02

- Added bounded, query-shape-driven same-page sibling selection for exact
  value/date/metric/identifier questions. It keeps one lexical companion and
  the strongest numeric-bearing sibling under the active revision, fixing the
  isolated mixed-scan date case without paper/case-specific mappings.
- Verification: Aya score on the isolated MMLong date case changed `0.0 → 1.0`;
  backend suite **858 passed**; production flags/indexes unchanged.

## Multimodal context reservation — 2026-08-31

- Added a generic visual-page reservation to context composition so selected
  page images cannot be evicted by ordinary text chunks before multimodal
  generation. Added regression coverage; full backend suite: **834 passed**.
  The visual flag remains off and no production projection changed.
- Applied the same reservation to the agent's hybrid and fallback retrieval
  paths; plain-text turns still do not reserve over-retrieved visual pages.
- Added per-case/aggregate telemetry for expected visual evidence, selected
  visual sources, and attached image bytes to distinguish retrieval completion
  from actual multimodal prompt coverage.
- Optional visual retrieval now covers explicit figure/diagram questions even
  when document-wide coverage mode is not active; it remains isolated and
  feature-gated.

## Visual evidence sufficiency wiring — 2026-08-30

- Canonical receipt fingerprint parity, graceful shutdown of cached visual
  workers, and truthful partial derived-job status/retry are covered by
  regression tests; DOCX parser-version drift is also covered; full backend
  suite: **830 passed**.

- Added a generic visual sufficiency signal to both normal and fallback agent
  generation. OCR/table excerpts no longer count as proof of chart, diagram or
  page appearance when no canonical visual artifact was retrieved. A bounded
  post-generation guard now replaces unsupported visual claims with a
  localized abstention before they reach the UI.
- Added visual/no-visual regression tests and corrected visual-page counts in
  tool telemetry; full backend suite is **828 passed**.
  `DUAL_QUERY_RRF_ENABLED` and `VISUAL_RETRIEVAL_ENABLED` remain disabled.

## Agent E2E and artifact portability — 2026-08-29

- [x] Re-ran Gate-C on the revision-aware isolated 21-paper clone after the
  evaluator contract fixes. Report `retrieval-v2.2-live-r16.json` records
  A0/A1/B-full `0.97/0.91/1.00`, B-anchor `0.98/0.93/1.00`, zero treatment
  activations and warm B-full P50/P95 about `139/309 ms`; dual-query remains
  off.
- [x] Ran a bounded public Sol three-mode smoke with one MMLong and one SPIQA
  case (6 generations, isolated runtime). The report is paired and complete,
  but explicitly diagnostic: Aya equals full context on SPIQA and misses the
  image-only unanswerable MMLong abstention. It does not close Gate-E.
- [x] Added an explicit `--disable-pdf-ocr` switch for isolated visual
  diagnostics. The switch is included in the resume fingerprint and report;
  production/default OCR behavior is unchanged.
- [x] Paired OCR-on/OCR-off Sol smoke confirms the distinction between
  ingestion cost and visual-answer coverage: image-only preparation fell from
  roughly 248 s to 28 s with OCR disabled, but the visual question still needs
  the optional page-image branch. No feature flag was changed.

- [x] Isolated HTTP/SSE regression passed **7/7 turns** with Sol, preserving
  14 history messages, paper focus, table/figure evidence and zero fallback.
  Report: `data/retrieval_eval/internal/e2e-agent-foundation-r15.json`.
- [x] Figure image serving now remaps legacy absolute paths into the active
  approved artifact root and rejects outside-root paths; the previously
  failing isolated figure URL returns HTTP 200. Added three resolver tests.
- [x] Evidence validator accepts Unicode math-form table headers such as
  `CCC𝑉` and `CCC 𝐴` with the correct valence/arousal identity; no
  paper-specific hardcoding was introduced.
- [x] Public three-mode answer-quality reports now include a paired,
  case-complete Aya-vs-full-vs-gold diagnostic with suite/stratum deltas;
  missing modes remain visible as partial and never trigger promotion.
  Final reports and per-row checkpoints are both atomically committed.
- [x] The same diagnostic reports deltas by source/document profile, keeping
  native, mixed, image-only and extracted-adapter cases distinguishable.
- [x] Gate-C semantic qrel review has a suite/model/prompt-fingerprinted,
  atomic batch checkpoint; failed batches retry on resume and provider outages
  remain explicitly non-promotable; stale failure entries are cleared after a
  successful retry. `--no-resume` resets stale rows atomically, and an
  integration fixture covers failed-batch resume/fresh-run paths.
- [x] SPIQA visual sampling is deterministic, stratum-balanced and gold-blind;
  it supports bounded question reuse per selected PDF and has regression
  coverage for order independence and document caps.
- [x] Full backend suite after this pass: **814 passed**; flags and production
  data remain unchanged.
- [x] A 20-case SPIQA visual probe was safely checkpointed and interrupted
  after 3/20 rows when CPU ColSmol indexing of the next PDF became too slow;
  no incomplete aggregate was reported or used for promotion.
- [x] The checkpoint was resumed to 20/20 successful SPIQA rows. T recall was
  `0.50`, V recall `0.90`, and equal-RRF T+V recall `0.95`; the corrected
  arbitrary-format gate correctly rejected promotion because MMLong native,
  mixed and image-only PDF profiles were absent. Report:
  `data/retrieval_eval/public/results/visual-gate-spiqa-r20-reuse.json`.
- [x] Public evaluator resume resolves legacy documents by raw content SHA-256
  before path lookup, preserving isolated revision/index reuse after a runtime
  path move.
- [x] Completed the six-document MMLong V2 visual probe across native, mixed and
  image-only PDF profiles. ColSmol remained isolated on CPU; T/V/T+V recall was
  `0.75/1.00/1.00`. Report:
  `data/retrieval_eval/public/results/visual-gate-mmlong-v2-r2.json`.
- [x] Added `merge_visual_gate_reports.py`, which fail-closes on incompatible
  fingerprints, duplicate case IDs or unproven LLM calls. Its 26-case merged
  SPIQA+MMLong diagnostic is recorded at
  `data/retrieval_eval/public/results/visual-gate-arbitrary-format-r1.json`.
- [x] Corrected public visual metric scope without changing rankings: native
  MMLong text is page-scored; flattened SPIQA text is artifact-scored and its
  page diagnostic is explicitly unscored. The recomputed 26-case aggregation
  is at `data/retrieval_eval/public/results/visual-gate-arbitrary-format-r5.json`;
  the report includes suite/profile aggregates and a generic failure taxonomy.
- [x] The MMLong v2 second-profile probe preserves its interrupted checkpoint
  and does not publish a partial aggregate; the path-only failure occurred in
  the pre-fix process and is covered by the new content-hash resolver test.

## Durable retry and isolated benchmark lifecycle — 2026-08-28

- [x] Live Gate-C r15 completed on the revision-aware isolated 21-paper clone:
  A0/A1/B-full `0.97/0.91/1.00`, B-anchor `0.98/0.93/1.00`; B-full had no
  activated query pair, so dual-query remains disabled. Report:
  `data/retrieval_eval/internal/retrieval-v2.2-live-r15.json`.
- [x] Gate-D lifecycle matrix r15 passed valid native/mixed/table-image PDF,
  DOCX, Markdown and TXT cases plus unchanged/duplicate/change/rename safety;
  the intentionally corrupt PDF is correctly isolated as partial.
- [x] Gate-E resource probe is recorded honestly: a 53-page image-only public
  PDF took about 8.5 minutes to encode with CPU ColSmol. The 20-case attempt
  was stopped before scoring, leaving a resumable checkpoint; no visual
  promotion or feature-flag change was made.
- [x] Unchanged folder rescans now return an existing staged revision when a
  prior parse succeeded but local projection was interrupted; retry can resume
  instead of silently skipping the document.
- [x] Public multimodal preparation now copies sources into its isolated root
  and runs the same durable staged → Lance/FTS validation → activation path as
  folder processing. Revision-scoped vector cache checks cannot reuse an old
  active revision for changed content.
- [x] LanceDB physical revision counts are checked after writes when the
  current table schema supports them; activation validation compares persisted
  card/text/table/figure counts and remains conservative for legacy tables.
- [x] Added exact-job durable execution (`run_job_now`) for isolated probes;
  full backend suite: **792 passed**, desktop build and Graphify update passed.
- [x] Public visual Gate-E runner can reuse a deterministic set of questions
  from each selected PDF (`--mmlong-cases-per-document`) while recording the
  selection; this reduces repeated cold indexing without gold/evidence-based
  sampling.
- [x] Public text/visual preparation resolves the document from the durable
  coordinator's validated `local_projection` receipt before falling back to a
  normalized source path; a completed job can no longer be misreported as
  `document_missing` because of path normalization.
- [x] Public retrieval-only checkpoints are fail-safe on resume: failed cases
  are retried and per-case rows are replaced instead of duplicated.
- [x] Public retrieval and visual-gate runners preflight Ollama/EmbeddingGemma
  before expensive ingestion and fail closed with a structured provider report
  when the local embedding service is unavailable.
- [x] Added regression coverage for the local-provider preflight contract;
  full backend suite is **793 passed**.
- [x] Re-ran a five-case isolated SPIQA text/table diagnostic with the corrected
  modality-aware report: phrase `0.60`, artifact `0.70`, support `0.80`.
  The sample remains non-promotable and does not alter production flags.

## Public/live coverage parity — 2026-08-28

- [x] Public visual evaluator and live agent use the same bounded active-page
  projection for document-wide questions; longer documents remain partial.
- [x] Backend suite: **791 passed**; desktop production build passed. Runtime
  providers were offline, so no retrieval or answer-quality score was emitted.
- [x] Gate-C evaluator default now points to audited `dev-v2.2`; v2.1 remains
  explicitly diagnostic-only.

## ColSmol worker device safety — 2026-08-28

- [x] `device=auto` is CPU-safe by default; MPS requires explicit
  `COLSMOL_ALLOW_MPS=1`. If an opted-in worker aborts before responding, the
  client retries once on CPU with the same pinned model and records telemetry.
- [x] Added regression coverage for device resolution and crash/retry; 27
  visual/public evaluator tests pass; full backend suite is **789 passed**.
  A one-case auto-device SPIQA smoke completed with worker effective device
  CPU and no production flag/index change. This did not promote visual search.
- [x] ColSmol indexing batches new page images within each bounded worker
  request, preserving per-page manifests and atomic files while reducing
  avoidable per-page model dispatch overhead.
- [x] Isolated four-artifact batch smoke indexed `4/4` and retrieved the gold
  artifact (visual recall `1.0`); the measured `33.4 s` remains diagnostic,
  not a promotion result.
- [x] Page map/reduce now receives active-document page counts and exposes
  `frontier_complete`; incomplete document coverage cannot be rendered as a
  complete census. Active documents up to the bounded 64-page aggregation
  frontier receive an exhaustive projection, while longer documents retain
  ranked sampling. Full backend suite after this contract pass is **789**.

## Gate-C decision-report correction — 2026-08-28

- [x] Corrected recommendation logic so a no-op B arm (zero raw/rewrite
  activations) cannot be reported as a successful dual-query promotion.
- [x] Re-ran the isolated revision-aware 60-case suite. A0/A1/B-full are
  `0.97/0.91/1.00` candidate/evidence/document; B-full activation is `0/60`,
  so the durable decision is `keep_dual_query_disabled_no_treatment_activation`.
- [x] Added regression tests for no-op treatment, candidate-union attribution
  and active dual-query promotion. Backend suite is **777 passed**; no
  production index/database or model setting changed.
- [x] Added an explicit visual-only evaluator OCR switch and moved synchronous
  public preparation off the event loop. The normal folder pipeline still
  enables OCR; a cold pathological scan was stopped without emitting a score.
  Backend suite after this change is **778 passed**.

## Coverage/visual aggregation hardening (2026-08-28)

- [x] Live agent and public evaluator share the same revision-aware page
  coverage merge; visual-only pages survive context limits and same-page OCR/
  visual provenance is joined deterministically.
- [x] Added bounded multimodal 9router message support for explicitly enabled
  visual page evidence, with local image-size limits and no change to the
  default text-only path.
- [x] Added generic page map/reduce contracts with strict page-ID validation,
  partial-batch reporting and deterministic reduction; no paper-specific or
  benchmark-specific answer rule was added.
- [x] Widened coverage diagnostics and re-ran public MMLong V2 case 0035:
  evidence-page recall `0.385 → 0.692`, answer `4 → 8` (reference `13`). The
  remaining miss is an aggregation/visual-census limitation, so Gate E and
  production visual promotion remain closed.
- [x] Backend suite: **774 passed**; Graphify synchronized after code changes.

## Generic page/document coverage (2026-08-27)

- [x] Added structural VI/EN coverage-intent detection with bounded page
  budgets; no fixed rewrite word-count or paper-specific rule.
- [x] Added active-revision FTS page projection, page deduplication and
  coverage-aware prompt/context composition. Ordinary fact/table paths keep
  their existing compact limits.
- [x] Added optional, feature-gated ColSmol page-provider wiring that joins
  visual and OCR/native evidence by document/page without weakening the local
  fallback or revision provenance.
- [x] Added regression tests for coverage classification, cache bypass,
  multi-page context, page identity and visual/OCR join semantics.

## RAG Foundation folder corpus (2026-08-25)

- [x] Added additive content-blob/source-alias/document-revision/stage-receipt
  schema and idempotent legacy backfill; raw SHA-256 detects byte changes,
  while extracted hashes remain integrity metadata.
- [x] Added revision-aware indexing, active-revision filtering, FTS v2
  projection, duplicate-byte aliasing, rename/move continuity and full-scan
  tombstone reconciliation.
- [x] Added durable folder processing jobs with leases, heartbeats, retries,
  recovery and explicit partial/failed stage receipts. Local search work is
  separate from optional enrichment and LightRAG.
- [x] Added ordered DOCX table/image extraction, sparse-page OCR provenance,
  readiness states and a disabled visual retrieval provider contract.
- [x] Added lifecycle/parser/coordinator/visual contract tests; current
  backend checkpoint is **725 passed**. Production indexing was not mutated.
- [x] Hardened atomic activation: staged revisions retain the prior active
  pointer, revision-scoped artifacts avoid primary-key collisions, and
  `activate_revision()` retires the prior revision only after projection.

## Document intelligence gate foundation (2026-08-23)

- [x] Added typed `EvidenceBundle`/tagged evidence candidates for text, table,
  figure and page units, preserving query origin, lexical/vector/RRF/rerank
  components, parent/provenance links and facet coverage without changing the
  historical context dict contract.
- [x] Added deterministic page/document readiness diagnostics with native text,
  OCR text, visual-only and unresolved states. Indexing persists the report;
  API document detail exposes it without removing `metadata_json` compatibility.
- [x] Added safe query-variant policy and experimental original + standalone
  rewrite RRF. Candidate-document union uses card-selected documents plus up to
  four FTS-discovered owners; dense retrieval remains bounded and rerank (if
  enabled) runs once after fusion. Default flag is off pending A/B.
- [x] Retrieval cache schema bumped to `v23` for the new projection contract.
- [x] Public multimodal evaluator now uses an explicit token-bounded
  `windowed_long_context` label when extracted text exceeds budget (96k input,
  16k output reserve by default); full mode no longer silently claims complete
  coverage after head/tail truncation.
- [x] New contracts/fusion/readiness tests plus all legacy tests: **690 passed**;
  desktop production build passed.

## Conversational agent / evidence validation (2026-08-23)

- [x] Chạy RC2 đủ 60 conversational turns trên isolated 21-paper clone bằng
  `cx/gpt-5.6-sol`: 60/60 pass, zero transport/tool/model fallback; median total
  13.379 s, median client-first delta 5.264 s.
- [x] Audit nested validation metadata thay vì chỉ tin top-level score; sửa và
  replay mọi affected family h02/h03/h10 đến khi affected reports `hidden=[]`.
- [x] Canonical validation tables theo focused document hoạt động cả card path,
  last-sources và retrieval cache thiếu card diagnostics; validation-only data
  không đi vào model prompt và không tăng token 9router.
- [x] Validator hiểu transposed row-metric V/A/D matrices, `V(r)/A(r)/D(r)` và
  Pearson-r; composite/atomic qualifier serialization vẫn phân biệt Test-1 với
  Test-2 và các dimension xung đột.
- [x] Progressive paper contract cấm trộn numeric row của nhiều paper trong một
  paper block; per-paper output vẫn validate theo canonical document ownership.
- [x] Conversation evaluator thêm opt-in `--debug-trace` và `run_id` để nối mỗi
  case với raw local prompt/draft trace mà không bật trace mặc định.
- [x] Verification code cuối: backend 680/680; desktop `tsc && vite build` pass.
- [x] Sol stability check sau khi chuyển model viết code sang Luna: full dev-20
  pass **20/20** trên clone cô lập, answer/router/LightRAG/vision đều pin Sol,
  zero fallback; affected MSF-SER/Mamba replay **4/4**. Nested guard audit vẫn
  giữ fail-closed và replay cuối `hidden=[]`.

## Codex developer context

- [x] Graphify runtime `0.9.37` + MCP extra cài bằng `uv tool`; graph sau PLAN
  foundation update có 3.221 nodes, 9.307 edges, 130 communities, zero model calls
- [x] Project skill cài ở `.agents/skills/graphify`; project MCP ở
  `.codex/config.toml`; `codex mcp list`, MCP initialize/list_tools,
  `graph_stats` và `query_graph` đều pass
- [x] `AGENTS.md` áp dụng graph-first budget 1.200 token và chỉ mở source đầy đủ
  khi cần verify/debug/edit/security hoặc graph không đủ tin cậy
- [x] Source-code Obsidian export: 3.105 generated notes + `graph.canvas`; có HTML graph,
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
# Evaluation Gate v2 — 2026-08-23

- Implemented the retrieval-only A0/A1/B harness with frozen rewrites, source/content-fingerprint qrels, canonical text/table/figure/page locators, fail-closed preflight, scope checks and promotion metrics.
- Curated `evals/retrieval_v2/dev-v2.jsonl` (60 cases) and sealed MMLongBench-derived `shadow-v2` questions (24 cases); private shadow qrels remain under ignored `data/retrieval_eval/private/`.
- Corrected dual-query agent wiring to preserve raw user wording while retaining enriched standalone context; feature flag remains disabled by default.
- Added evaluator/fixture tests; backend suite is green at 697 tests.

# Evaluation Gate v2.1 contract repair — 2026-08-24

- [x] Invalidate v2 promotion report with durable three-cause record; retain it
  for audit only rather than treating its scores as a baseline.
- [x] Add canonical evidence identity shared by typed bundles, RRF fusion and
  evaluator; table/figure IDs, captions and indices survive round-trip.
- [x] Implement two-stage A0/A1/B-full/B-anchor runner with equal external
  candidate depth, single production rerank stage, per-stage timing/call counts,
  ranking diagnostics and failure taxonomy.
- [x] Add generic raw-branch activation from exact/new retrieval signal, not a
  rewrite word-count rule; keep production dual-query flag off.
- [x] Add `propose` / `validate` / `publish-reviewed` eval workflow,
  `dev-v2.1` structural proposal, audit/provenance/rewrite-fingerprint guards,
  qrel alternatives, and deprecate shadow-v2 pending genuinely curated shadow-v3.
- [x] Verify focused/new tests and full backend suite: **711 passed**. Draft
  local embedding smoke completed with no 9router/model call or index mutation.
- [x] Audit hardening r3: numbered table/figure proposal questions use the
  caption's human-facing label rather than ingestion order; structural
  selection preserves two-qrel coverage for all 21 documents without a
  filename-specific exception.
- [x] Add read-only `review-packet`, `preflight` (60/60 portable locator
  resolution) and named audit approval/timestamp guards. The proposal remains
  explicitly non-audited; this is a guard against false benchmark promotion,
  not a retrieval quality result.
- [x] Complete Codex local semantic audit, apply five portable text-locator
  corrections, and publish audited `dev-v2.1.jsonl`: 60/60 qrel resolve with
  `dev-v2.1-frozen-rewrite-manual-audit-r1`.
- [x] Run the first valid local-only A0/A1/B-full/B-anchor measurement. A0,
  A1 and B-full are identical at Candidate R@30 0.82 / Evidence R@10 0.70;
  B-full has zero raw-branch activations, while B-anchor activates 38/60 but
  regresses to 0.64. No arm passes promotion and no runtime flag changed.
- [x] Correct evaluator diagnostics: a final top-k truncation is now reported
  as `final_top_k_cutoff` when reranking is disabled, rather than falsely as a
  reranker failure; report also exposes query-variant activation/reasons.
- [x] Verification after the audit/run diagnostics: **712 backend tests passed**.

# Legacy-card repair and scope-faithful retrieval measurement — 2026-08-24

- [x] Add deterministic, idempotent backfill for documents with canonical text
  but no `document_card`, plus explicit CLI that re-vectorizes only repaired
  IDs through local EmbeddingGemma. It does not call an LLM or rebuild the
  corpus. Added regression coverage for card backfill → LanceDB indexing.
- [x] Repair observed legacy `demo.md` gap in the local corpus (one card + one
  text vector); this is a generic migration/readiness repair, not a document
  exception.
- [x] Make retrieval evaluation apply production's catalog current-turn scope
  resolver from raw query only; conflict with the qrel target fails closed.
  Audited dev-v2.1 rerun: A0/A1/B-full `0.87` Candidate R@30, `0.72` Evidence
  R@10 and `0.95` Document R@5; B-anchor remains worse. `DUAL_QUERY_RRF_ENABLED`
  remains false.
- [x] Final verification: **715 backend tests passed**; Graphify update:
  **3,391 nodes / 9,897 edges / 126 communities**.

# Public multimodal adapter persistence — 2026-08-24

- [x] Make SPIQA's generic artifact adapter classify table artifacts from their
  source metadata, retain an authoritative table image through LanceDB, and
  keep `documents.table_count` truthful.
- [x] Preserve VLM-derived visual summaries/provenance across an idempotent
  adapter synchronization when image SHA-256 is unchanged; discard them when
  bytes change. This prevents a public-eval refresh from silently making a
  previously searchable visual/table opaque again.
- [x] Isolated public SPIQA three-case smoke after the fix: Aya diagnostic
  token-F1 `0.820` vs gold-context `0.822`; the formerly missed table case
  retrieves the selected table image and returns all seven languages. No
  production corpus or held-out set was modified.
- [x] Verification: **715 backend tests passed**; Graphify update:
  **3,393 nodes / 9,900 edges / 120 communities**.
# RAG Foundation correctness pass — 2026-08-25

- [x] Durable folder lifecycle now stages canonical rows, validates FTS/Lance
  counts and active-revision identity, then activates atomically; retryable
  provider failures preserve the previous active revision.
- [x] Added snapshot idempotency, leases, heartbeats, startup recovery and
  separate derived-enrichment jobs. VLM/evidence-card/LightRAG work is
  privacy-gated and cannot block local search.
- [x] Enforced revision isolation in FTS v2, Lance metadata/filters, cards,
  canonical artifact materialization, figure enrichment and LightRAG provenance.
- [x] Added Markdown approved-root image handling, DOCX ordered table/image
  extraction, readiness states and isolated Lance revision migration utility.
- [x] Verification: 727 backend tests and desktop production build pass.

## Gate-C v2.2 benchmark validity — 2026-08-25

- [x] Added fail-closed suite-quality diagnostics for language/context
  confounding and repeated question templates.
- [x] Generated `dev-v2.2.proposed.jsonl` and review packet with 60 portable
  cases, orthogonal language/context coverage and 60/60 qrel preflight.
- [x] Verification: **729 backend tests passed**; v2.2 remains review-required
  and is not promotion evidence yet.
- [x] Added the no-LLM production-policy retrieval evaluator for per-document
  comparison and facet-parallel multi-hop. On an isolated migrated index and
  the draft v2.2 suite it measured Candidate R@30 `0.87`, Evidence R@10
  `0.78`, Document R@5 `1.00`, multi-paper coverage `1.00`, modality recall
  text/table/figure `0.733/0.700/1.000`, warm P50/P95 `126/203 ms`, and zero
  scope leaks. This remains diagnostic because the suite is review-required.
  Verification at that point stood at **731 backend tests passed**; the final
  selector/evaluator pass below raises the suite to **734**.
- [x] Generalized Gate-C A0 retrieval with structural facet expansion, deeper
  candidate frontier, coverage-aware final selection, and exact artifact-number
  reservation. Corrected isolated draft-v2.2 A0 reached Candidate R@30 `0.96`,
  Evidence R@10 `0.91`, Document R@5 `1.00`, modality text/table/figure
  `0.85/1.00/1.00`, multi-paper `1.00`, P50/P95 `123/259 ms`; paired B-full
  was non-inferior but unnecessary. This remains draft-suite evidence until
  independent semantic audit.
- [x] Verification after the selector/evaluator changes: **734 backend tests**;
  Graphify **3.590 nodes / 11.469 edges**.
- [x] Later correctness/public-format pass: **736 backend tests**, public
  artifact revision identity fixed, unchanged-rescan and single-OCR identity
  pass added, and isolated TXT/MD/DOCX/image-only-PDF smoke passed its local
  projection/retrieval contract. Latest Graphify refresh: **3.622 nodes /
  11.585 edges**.

## Gate-C evidence curation audit — 2026-08-25

- [x] Ran independent Sol semantic review on the v2.2 proposal. It correctly
  rejected weak qrels (caption-only tables and title/chrome text spans) instead
  of allowing a false promotion.
- [x] Added canonical evidence refresh for role-matched text, table markdown
  and figure captions. The proposal remains review-required; no rewrite or
  production retrieval path was changed. Backend remains **736 passed**.
- [x] Static matrix audit remains **60/60**. With explicit approval for longer
  canonical excerpts, Sol accepted **60/60** cases. The separate semantic
  attestation command published `evals/retrieval_v2/dev-v2.2.jsonl` and an
  isolated preflight resolved 60/60 qrels; reviewer provenance explicitly says
  independent model, not human.
- [x] Gate-C paired retrieval rerun completed after granting the evaluator
  localhost access (the earlier failure was Codex sandbox permission, not an
  Ollama outage). A0/A1/B-full measured Candidate/Evidence/Document
  `0.96/0.87/1.00`, multi-paper `1.00`, with A0 P50/P95 `128/266 ms`.
  B-anchor measured `0.99/0.88/1.00` at `181/281 ms`; it remains diagnostic.
  Text modality recall for A0/B-full is `0.7833`, so Gate C is not promoted
  and `DUAL_QUERY_RRF_ENABLED` remains false. The run made no production writes
  and no 9router calls.
- [x] Verified 9router with an authenticated Sol completion and ran the
  seven-turn HTTP/SSE audit against an isolated backend. All transport/model
  runs completed, but the audit remained correctly red: an ASPIRE metric
  follow-up fell back, a WhiSER switch selected unrelated papers, and the
  isolated copy lacked figure bytes for the visual endpoint. This is retained
  as a diagnostic failure; no production state was changed.
- [x] Corrected the isolated root layout and hardened retrieval cache keys with
  raw artifact obligations (`RETRIEVAL_CACHE_VERSION=v24`). The subsequent Sol
  seven-turn HTTP/SSE audit passed **7/7**, including table metric validation,
  ASPIRE/WhiSER focus switching, and figure HTTP 200. Backend verification is
  **737 passed**; dual-query remains off.

## Gate-C selector/revision compatibility — 2026-08-25

- [x] Preserved legacy Lance vectors during the initial revision migration
  while rejecting missing revision metadata for later active revisions.
- [x] Unified context/evaluator facet inference and added regression coverage
  for unlabelled legacy text and false `Model`→architecture classification.
- [x] Isolated Gate-C rerun: A0/A1/B-full `0.96/0.87/1.00`
  (candidate/evidence/document), exact table/figure `1.00`, text `0.7833`,
  P50/P95 about `131/340 ms`; B-anchor remains diagnostic. Gate C and
  dual-query promotion remain closed. **740 backend tests pass.**

### Gate-C lexical/selector pass — 2026-08-26

- Added bounded lexical recovery for late filename/identifier/tail terms and
  generic per-document final-selection coverage; no embedding/index change.
- Isolated audited 60-case result: A0 Candidate/Evidence/Document
  `0.97/0.91/1.00`, text/table/figure `0.85/1.00/1.00`, P50/P95 `139/338 ms`,
  zero FTS misses and six final-cutoff diagnostics.
- A1/B-full remain non-inferior but add no value; dual-query stays disabled.
- **742 backend tests pass**; Graphify refreshed to 3,655 nodes / 10,698 edges.
- Sol HTTP/SSE regression on the same isolated clone passed **7/7** after
  clone-only artifact-path repair; no transport/model/tool fallback occurred.
## Gate-D format/lifecycle and public evaluator — 2026-08-26

- [x] Revalidated the generated format smoke under schema
  `format-lifecycle-smoke-v2`: native/OCR-searchable TXT/MD/DOCX require a
  real expected text hit, while the sparse image-only PDF remains explicitly
  `partial`/non-searchable with a valid revision-scoped projection. Its raw
  text miss is retained as diagnostics rather than misreported as `ready`.
  Report: `data/retrieval_eval/internal/format-lifecycle-smoke-r17.json`.
- [x] Visual Gate-E promotion now requires an explicit complete paired
  answer-quality contract (`promotion_safe=true`); retrieval-only visual
  recall can no longer appear as a passing promotion decision. A cold 6-case
  Sol/ColSmol attempt was interrupted after 2/18 rows and published no score.
- [x] Verification after the contract hardening: **860 backend tests pass**;
  no production index or feature flag changed.

- [x] Added the isolated `evaluate_format_lifecycle_matrix.py` gate for
  native/mixed/image-only/table-image PDF, DOCX, Markdown and TXT, including
  corrupt-source isolation and duplicate/rescan/change/rename checks.
- [x] Validated six valid fixtures through the durable folder coordinator,
  local FTS/Lance projection and retrieval; the corrupt PDF remains partial
  without losing active revisions.
- [x] Added per-document preparation caching and resumable checkpoints to the
  public retrieval-only runner.
- [x] Fixed Docling/native page text reconciliation before selective OCR.
- [x] Backend suite: **743 passed**. Public SPIQA four-case smoke completed,
  but table-image evidence recall is only `0.25` on its two table cases; this
  is recorded as a limitation, not a promotion.
- [x] Added cached local OCR metadata for approved Markdown/DOCX image
  artifacts and public table-image adapters; image identity/caption/provenance
  remain canonical and unchanged. SPIQA v4 reports separate answer-token
  coverage so annotation-marker spans are not mislabelled as retrieval misses.
- [x] Final verification after this change: **744 backend tests pass** and
  desktop production build passes.
- [x] Added reproducible MMLong document-profile sampling and per-profile
  reporting. A cached smoke covering one native-text, one mixed-text/image and
  one image-only PDF completed **3/3 page recall** with zero failures. This is
  diagnostic format coverage only; visual retrieval and broad public/RC gates
  remain open. Report:
  `data/retrieval_eval/public/results/public-retrieval-only-mmlong-stratified-v1.json`.
- [x] Final verification after report/test hardening: **745 backend tests
  pass**, desktop production build passes, and Graphify is synchronized at
  **3,691 nodes / 10,795 edges / 147 communities**.
- [x] Expanded the MMLong profile smoke to six document-balanced cases (2
  native, 2 mixed, 2 image-only): 4 answerable cases had page recall `1.0`,
  while 2 official `Not answerable` cases were kept outside that denominator.
  Zero cases failed. Report:
  `data/retrieval_eval/public/results/public-retrieval-only-mmlong-stratified-v2.json`.

- [x] Added dependency-isolated ColSmol-256M visual retrieval through a
  resumable NDJSON worker (`backend/.venv-visual`), preserving the backend's
  Transformers 4.x/Docling runtime. The resumed 3-profile MMLong run and
  4-case SPIQA replay completed without failures; cold indexing and warm
  retrieval are measured separately. This is an isolated diagnostic only:
  the feature flag remains off and no production path is promoted. Backend
  suite after isolation: **751 passed**.
- [x] Added explicit MMLongBench-Doc versioning and V2 support. New visual
  evaluations default to corrected V2 metadata, while v1 remains available
  for historical compatibility; reports cannot mix the two releases.
- [x] Completed a 3-profile V2 visual smoke (native, mixed text/image and
  image-only) in the isolated worker: 3/3 successful, page recall `1.00` for
  T, V and T+V. Added worker PID/index/model-load telemetry to avoid confusing
  the child model footprint with orchestrator RSS. This remains diagnostic;
  no visual feature flag was promoted. Full backend verification is **752
  passed** and the desktop build passes.
- [x] Re-ran the arbitrary-format lifecycle matrix through the durable folder
  coordinator. Native/mixed/table-image PDF, DOCX, Markdown and TXT all
  projected and retrieved; duplicate bytes, unchanged scans, revision changes
  and renames preserved identity, and a corrupt PDF was isolated as partial.
- [x] Ran an authorized 4-case Sol agent smoke through Aya's actual retrieval
  and generation path (2 historical MMLong V1 + 2 SPIQA). All calls completed
  without fallback; page retrieval was correct, while answer-level scores
  remain diagnostic (`0.50` MMLong accuracy, `0.737` SPIQA token-F1). New
  runner invocations default to corrected MMLong V2.

## 2026-08-27 correctness hardening

- [x] Evaluation Gate-C now models production compare decomposition rather than
  a single global search; the isolated audited v2.2 run reached
  `0.97/0.91/1.00` candidate/evidence/document recall with `1.00` multi-paper
  coverage and no scope/model-call violations.
- [x] Evaluation CLI rejects non-revision-aware databases; stale stage receipts
  no longer make a document appear ready under a changed provider/model
  fingerprint.
- [x] Gate-E visual reports now include an explicit, non-promoting decision
  contract. Small ColSmol/MMLong/SPIQA smokes remain diagnostics until sample,
  profile, gain and non-regression requirements are met.
- [x] Isolated ColSmol worker requests are now page-batched (default 8 pages)
  with aggregated counters and per-batch durable manifest commits. A large
  PDF no longer has to cross the NDJSON pipe as one unbounded request; an
  interrupted run can resume from completed batches.
- [x] Final verification after the batching/correctness pass: **766 backend
  tests passed**, visual contract **9 passed**, lifecycle contract **13
  passed**, and desktop production build passed. Gate-C r12 keeps dual-query
  disabled because A1/B-full are non-improving over A0.
- [x] Completed the broad isolated V2 Gate-E diagnostic: 7/7 cases succeeded
  across native/mixed/image-only MMLong and figure/table SPIQA strata; T recall
  `0.9286`, V `1.0000`, T+V `1.0000`. Promotion correctly stayed off because
  the sample is below the minimum 20 positives/two-per-profile contract.
- [x] Added page-batched NDJSON transport for ColSmol worker indexing so long
  PDFs checkpoint and resume without an unbounded pipe request.
- [x] Completed the approved Sol three-mode arbitrary-format diagnostic on
  corrected MMLong V2 and SPIQA (21/21 generations, no fallback). Gold/full/Aya
  answer quality and cold-preparation limits remain separated; no feature flag
  was promoted.
- [x] Added an isolated `--visual-pages` answer-quality diagnostic that can
  fuse ColSmol page hits into Aya context without touching production. The
  first chart-count replay improved page recall but did not solve multi-page
  arithmetic, so visual promotion remains off.
- [x] Reused a process-local RapidOCR engine across PDF/image OCR and kept
  transient OCR initialization failures retryable; backend suite reached
  **767 passed**.
# 2026-08-28 — Gate-E worker device safety

- ColSmol `auto` is now CPU-safe by default; MPS requires explicit
  `COLSMOL_ALLOW_MPS=1`. If an opted-in worker aborts before responding, the
  client retries once on CPU with the same pinned model and records telemetry.
- Added regression coverage for device resolution and the crash/retry path;
  26 visual/public evaluator tests pass. No production flag or index changed.

# 2026-09-02 — Exact visual page contract

- [x] Wired explicit page/slide/trang hints through image attachment in the
  production agent and isolated public evaluator. The filter is generic and
  leaves retrieval rankings/telemetry untouched.
- [x] Added a shared visual prompt contract: inspect the requested page first,
  do not transfer numbers from another page, and abstain when the supplied
  visual cannot support the claim.
- [x] Verification: targeted visual/public tests **147 passed**, full backend
  suite **847 passed**, Graphify synchronized. A Sol replay failed closed only
  because 9router was unavailable; no fallback was used.
- [x] Separated caption/figure metadata from pixel-required visual questions in
  the shared policy. Appearance claims now require a rendered image/page;
  metadata remains valid for caption/inventory questions.
- [x] Added page-aware attachment labels to the production multimodal prompt;
  labels and image URLs share one bounded candidate projection.
- [x] Included safe canonical figure/image artifacts in that bounded
  projection, not only visual-page records; exact page hints still apply.

# 2026-08-30 — Public answer-quality checkpointing

- The three-mode public Sol evaluator now checkpoints each case/mode
  atomically and fingerprints model, selection and context budgets. Resume
  reuses only successful rows with an identical fingerprint and retries failed
  rows; no live model run was claimed while 9router was unreachable.

# 2026-09-01 — Same-page evidence context

- [x] Added active-revision same-page native-text companions for selected
  figure/table/visual-page evidence in both production and isolated evaluator
  paths.
- [x] Expanded visual-intent detection to picture/photo/screenshot/slide.
- [x] Full backend verification: **838 passed**; targeted public/retrieval
  verification: **52 passed**; Graphify synchronized.
- [ ] Gate-E remains open: the corrected 8-case SPIQA visual smoke scored
  Aya `0.6054` token-F1 vs bounded full `0.6790`, and an image-only MMLong
  replay still exposed an unsupported numeric visual answer.

# 2026-09-04 — Gate-E evaluator and context correctness

- [x] Added ordered case-manifest support to retrieval-only evaluation so
  public retrieval and answer-quality reports share the same frontier.
- [x] Added discriminative exact-anchor preservation in context projection;
  late acronym/number/identifier evidence is retained without paper-specific
  mappings. MMLong `0307` replay improved page recall `0 -> 1` and token
  coverage `0.6 -> 1.0`; `0990` remained page-recall `1`.
- [x] Corrected visual evaluator T arm to use production reranker settings;
  the prior forced-rerank report is historical and not used for promotion.
- [x] Retrieval-only 20-case replay: 20/20 successful, MMLong page recall
  `0.90`, P50/Pmax `151.8/684.3 ms`.
- [x] Production-parity ColSmol replay: 18 positive cases, T/V/T+V recall
  `0.5000/0.7778/0.9444`, no T+V per-case regressions; visual flag remains
  off because minimum positive and Sol answer-quality gates are unmet.

# 2026-09-04 — Gate-E latency telemetry

- [x] Added explicit T+V and fusion-stage latency to the visual evaluator and
  report merger. The change only closes an observability gap; it does not
  promote visual retrieval or mutate production state.
- [x] Targeted public-evaluator tests: **48 passed**. A full-suite rerun is
  still required before an RC claim.
- [x] Artifact-only Markdown with an explicit table/figure image locator now
  requires visual context; native PDF/DOCX tables remain text-first. This
  closes a generic evaluator/provenance gap without changing production flags.
# 2026-09-02 — generic visual content-vs-metadata intent

- Added relation-aware pixel requirement detection for content requested
  inside/from/on visual artifacts, while preserving metadata-only caption and
  inventory queries. Added English/Vietnamese regression tests; no
  paper-specific or benchmark-specific mapping.
- Added explicit page labels to public multimodal attachment text so the
  answer model can bind rendered `page-N` images to the retrieved page.
- Made the durable folder-processing API discover all supported file formats
  by default, while leaving the deprecated legacy endpoint compatible.
- Re-ran the corrected MMLong image-only smoke through Sol with ColSmol. The
  former visual hallucination now abstains correctly (`1.0` official score);
  a warm repeat confirms the result and records reusable-index latency.
# 2026-09-02 — exact-value sibling context

- Added bounded, query-shape-driven same-page sibling selection for exact
  value/date/metric/identifier questions. It preserves the lexical companion
  and the strongest numeric-bearing sibling under the active revision, fixing
  a mixed-scan public QA miss without case-specific mappings.
- Verification: isolated MMLong date case Aya score `0.0 -> 1.0`; backend
  suite **853 passed**; production flags/indexes unchanged.

# 2026-09-04 — multi-evidence retrieval correctness

- [x] Added bounded deterministic operand aliases for explicit ratio,
  comparison and multi-hop queries; aliases remain retrieval-only and are not
  treated as evidence.
- [x] Preserved contiguous semantic phrases during obligation reservation,
  preventing exact labels such as `product development` and `total assets`
  from being diluted into unrelated token matches.
- [x] Corrected public evaluator cache telemetry so reused active projections
  are reported as cached and included in the production configuration
  fingerprint.
- [x] Full backend suite: **885 passed**; desktop production build and
  Graphify synchronization pass. Gate C metrics remain green; Gate E is still
  non-promotable pending Ollama/9router and a larger stratified replay.
- [x] Warm isolated Sol replay `r36` of MMLong case `0990` now selects both
  complementary pages (`47`, `69`) and scores `1.0` (`5.03%`), up from `0.0`
  before phrase-aware numeric reservation. This remains a causal smoke only.

# 2026-09-04 — Gate-E replay and natural table-intent parity

- [x] Completed the resumed Sol answer-quality replay: **20/20 cases,
  60/60 generations**, zero failed rows/fallbacks. Diagnostic overall Aya
  score `0.4238` vs full extracted `0.5312` and gold `0.5719`; not a
  production promotion claim.
- [x] Made table reservation production-shaped for prose queries about
  metrics, performance, scores and results, even when `table` is absent.
  Added regression coverage and kept the detector generic.
- [x] Verification: **887 backend tests passed**; both dual-query and visual
  flags remain disabled pending a larger unseen, profile-balanced gate.
- [x] Generic table-intent markers use token boundaries to avoid false
  positives in unrelated prose.

# 2026-09-10 — visual checkpoint and embedding batching

- [x] Ollama embedding requests use bounded batches (default 16), preserving
  order and query behavior; full backend suite: **891 passed, 5 warnings**.
- [x] Completed isolated visual-enabled Sol answer-quality report on 10
  document-balanced MMLong cases: 30/30 generations, zero failures, Aya
  accuracy `0.50`, page recall `0.875`.
- [x] Preserved the 20-case replay checkpoint (30 rows / 10 cases) after
  stopping a 198-page cold visual index; production state unchanged.
- [x] Added durable per-batch visual progress updates tied to the job lease,
  preserving resumability and partial readiness for long documents. Full
  backend suite after the change: **891 passed, 5 warnings**.
- [x] Added bounded, configurable Ollama transient retries without provider
  fallback. Isolated format lifecycle matrix `r55` verifies PDF/DOCX/MD/TXT,
  table-image and mixed-scan provenance, duplicate/rescan/change/rename
  invariants, and retrieval-all-hit.
- [x] Lifecycle evaluator preflights the local model and waits for persisted
  retry schedules, so provider warm-up/backoff is not misreported as a parser
  or revision failure.
- [x] Completed SPIQA Test-C visual-enabled Sol replay `r56`: 8/8 cases,
  24/24 generations, zero failures; Aya token-F1 `0.6947`, ROUGE-L `0.6791`,
  all cases carried visual context and image attachments.

# 2026-09-11 — metric-aware canonical table selection

- [x] Follow-up questions that name result metrics (for example Acc/F1/CCC)
  now use the canonical table path even when the user omits the word
  “table”. The selector filters out tables missing a requested metric and
  prefers generic main-result captions; no paper-specific mapping was added.
- [x] Targeted tests: **98 passed**; full backend suite: **892 passed**.
- [x] Isolated Sol HTTP/SSE replay `r63`: **7/7 turns**, all evidence valid,
  benchmark follow-ups selected the canonical comparison table, and the
  figure image endpoint returned `200 image/png`.

# 2026-09-11 — Gate-E paired positive frontier

- [x] Added generic post-hoc merge utilities for disjoint answer-quality and
  visual reports with include/exclude case selection, strict contract checks,
  and no model/index side effects.
- [x] Paired a profile-balanced 20-positive-case frontier: visual T+V recall
  `0.95`, T recall `0.55`, no fused text regressions, `60/60` Sol generations
  complete, and zero failed rows.
- [x] Recorded honest answer-quality limits: Aya diagnostic `0.5749` vs full
  extracted `0.5462`, but four per-case regressions keep the safety gate false.
  Image-only case `0845` shows visual rescue (`1.0` vs `0.0`) at a cold-index
  cost of roughly `248.9 s`; SPIQA figure case `1703.05260:0001` still
  abstains despite visual retrieval.
- [x] Kept both visual and dual-query feature flags off; no production
  projection/index/database was changed.
- [x] Added tests for disjoint frontier selection/exclusion in both report
  mergers; full backend suite: **894 passed, 5 warnings**.
- [x] Prepared the profile-balanced 21-case retrieval frontier (`r73`):
  `T+V 0.9524`, `T 0.5476`, no fused regressions, and every visual profile
  appears at least twice. It remains diagnostic until its added case is
  paired with Sol answer quality.
- [x] Added evaluator fingerprints to answer-quality checkpoints/reports and
  merge invariants; stale-code resume/merge now fails closed. Full backend
  suite after the guard: **894 passed, 5 warnings**.
- [x] Extended stale-code protection to visual-gate checkpoints and report
  merges; targeted evaluator suite: **64 passed**.
- [x] Fingerprints now cover the backend retrieval/parser dependency tree, not
  only runner files; current answer/visual fingerprints are recorded in the
  replay manifest.

# 2026-09-12 — bounded visual frontier r87

- [x] Ran the current-code isolated visual evaluator with resumable checkpoints
  on the next public MMLongBench frontier. Four cases completed and one case
  (`0569`) was correctly recorded as a 900-second visual-index timeout.
- [x] Reported the run as partial/non-promotable. The four-case diagnostic
  T/V/T+V recall was `0.75/0.625/1.00`; no LLM calls or production mutations.
- [x] Verified backend, frontend, 9router and Ollama health after the run;
  feature flags remain off.
- [x] Corrected stage-specific timeout attribution (`render_pages`/`prepare`
  versus `visual_index`/`visual_search`) and added a regression test; the
  targeted public evaluator suite is **52/52**.

# 2026-09-13 — safe interruption of visual frontier r88

- [x] Resumed the isolated frontier with existing visual cache; seven cases
  checkpointed before a host-sleep stall at `0315`.
- [x] Ctrl-C produced an explicit interrupted/score-invalid report and cleaned
  up the ColSmol worker; no production state changed.

# 2026-09-13 — visual r88 and paired answer-quality r90

# 2026-09-13 — service/build verification and r94

- [x] Backend, frontend, 9router and Ollama are live and health-checked.
- [x] Real SSE chat smoke completed through `cx/gpt-5.6-sol`.
- [x] Desktop production build passed and backend suite passed **906/906**.
- [x] r94 SPIQA artifact-projection smoke completed 3/3 rows; its low Aya
  score is retained as diagnostic evidence only, not a promotion result.
- [x] Added generic active-revision page-to-artifact context projection for
  visual/page hits; retirement isolation is covered by tests and the full
  backend suite is **908/908**.
- [x] r95 current-code SPIQA smoke confirms the projection effect (Aya
  token-F1 `0.4444` vs full `0.4242`); no promotion inferred from one case.
- [x] r96 current-code paired 21-case frontier completed 63/63 generations
  with zero failures; aggregate and limitations are recorded in handoff.
- [x] r97 replayed the three observed failure classes (9/9) without
  paper-specific tuning; two recovered to parity and one remained a small
  metrics-summary regression.

# 2026-09-13 — r98 artifact evidence rows and format smoke

- [x] Added and exercised first-class active-revision table/figure evidence
  rows for page hits; r98 replay completed 9/9 generations with zero failures.
- [x] Isolated lifecycle smoke passed for TXT, Markdown, DOCX and mixed/image
  PDF (4/4 expected projection/retrieval contracts); image-only PDF remains
  explicitly `partial`.
- [x] Full backend suite after this change: **908 passed, 5 warnings**.

# 2026-09-13 — durable canonical progress checkpoint

- [x] Added lease-guarded per-file progress callbacks from folder indexing;
  completed/partial jobs retain counters and last path for restart auditing.
- [x] Full backend suite after the checkpoint change: **909 passed, 5
  warnings**; Graphify synchronized.
- [x] Desktop production build after the checkpoint change passes.

# 2026-09-13 — public HTTP/SSE scope regression

- [x] Added generic separator-insensitive arXiv identifier aliases with
  corpus-wide ambiguity protection.
- [x] Public isolated HTTP/SSE explicit-paper + contextual follow-up passed
  2/2 with retrieval events and stable sticky focus.
- [x] Full backend suite after this fix: **910 passed, 5 warnings**.

# 2026-09-13 — stratified public agent regression

- [x] Exercised public HTTP/SSE agent strata (table, figure/page, text,
  negative, cross-document); the transient empty cross-document generation
  was rerun and then passed with two-document evidence coverage.
- [x] Preserved the transient as an audit limitation rather than hiding it in
  the pass count.

# 2026-09-13 — Gate C retrieval and agent policy

- [x] Balanced audited Gate C suite passed preflight 60/60 and A0 thresholds:
  Candidate/Evidence/Document `0.97/0.91/1.00`, modality minimum `0.85`,
  multi-paper `1.00`, zero scope leak.
- [x] Agent policy/decomposition passed `0.98/0.88/1.00`, modality minimum
  `0.833`, multi-paper `1.00`, zero scope leak and P50/P95 `151/378 ms`.
- [x] Confirmed A1/B-full add no measured gain; dual-query remains off.

- [x] Completed isolated visual Gate-E r88 at **21/21** cases with T/V/T+V
  recall `0.4762/0.8571/1.0000`, no failed rows, no LLM calls and no text
  regressions; promotion correctly stayed false pending answer quality.
- [x] Completed paired r90 answer quality on the same manifest: **63/63** Sol
  generations, zero failures, with visual attachments and production-style
  retrieval. MMLong Aya accuracy `0.7219`; SPIQA token-F1 `0.6148` (diagnostic).
- [x] Added explicit separate canonical text/visual runtime roots to prevent
  answer evaluation from rebuilding a second text projection unnecessarily.
- [x] Full backend suite after the change: **905 passed, 5 warnings**; service
  health checks for backend, frontend, 9router and Ollama all returned 200.
## 2026-09-13 — lifecycle matrix verification

- Completed the isolated format/lifecycle matrix for PDF/DOCX/MD/TXT with
  corrupt-input isolation, duplicate/rescan/change/rename checks and local
  vector retrieval. All required invariants and six expected retrieval hits
  passed; production state was untouched.

## 2026-09-13 — visual retrieval replay

- Replayed the current 20-case public visual manifest with isolated local
  indexes. T+V reached full evidence recall on the 18 scored positives with no
  text regression; promotion stayed disabled because answer-quality pairing and
  the minimum-positive contract are still unmet.

## 2026-09-13 — current visual frontier and context audit

- Completed the 21-positive visual retrieval replay and paired 63-generation
  Sol answer run. Retrieval T+V recall was `1.0`; answer quality was complete
  and aggregate-positive but still non-promotable due three SPIQA diagnostic
  per-case regressions. Added bounded artifact-context projection and retained
  the flags off pending a stronger, unseen release gate.
# 04 — Done

## 2026-09-14

- Made public visual evaluation routing gold-independent: artifact/source
  annotations no longer activate visual boost, page reservation or pixel guard.
  Text-backed adapters can answer ordinary fact/table questions without an
  annotation-induced abstention; image-only projections remain fail-closed.
- Full backend suite after the change: **914 passed**. A new Sol replay was
  intentionally halted on 9router HTTP 429 and is diagnostic-only.

## 2026-09-16

- Restarted backend/frontend for local UI testing; the 9router gateway is
  reachable and advertises the pinned Sol model, but completion requests are
  currently rejected with `401 token_revoked` until its OAuth session is
  re-authorized.
- Revalidated Gate C retrieval-only with Ollama and reran the full suite:
  **914/914 passed**. Gold-independent visual routing is covered; no
  production flags or indexes were changed.

## 2026-09-14

- Added a generic open-ended metric/evaluation text-prose reservation in
  `RagService`; it preserves explicit table/figure evidence and is covered by
  a quality regression test.
- Replayed Gate C after the change. A0 stayed at Candidate/Evidence/Document
  `0.97/0.91/1.00`; B-full was unchanged and B-anchor reached Evidence `0.92`
  at higher latency. Full backend suite: **914 passed**.

## 2026-09-13

- Expanded visual intent/pixel policy for generic multimodal terminology
  (`visual grounding`, `pixel`, `multimodal`, `image-text`, `vision`) without
  adding paper- or benchmark-specific mappings; caption/list requests remain
  answerable from metadata.
- Regression and full backend verification: **911 passed**.
# 2026-09-16 — generic list/table retrieval correctness

- Added shared enumerated-evidence intent and final-cutoff structured artifact
  promotion. Generic list questions now preserve a relevant canonical table
  when it is already in the production candidate frontier.
- Fixed modality-local vector ranking so sparse table/figure projections are
  not penalized by the larger text projection during hybrid fusion.
- Verified an isolated SPIQA list query retrieves the expected table/context;
  no benchmark gold labels were used for routing.
- Backend: **927 passed**; desktop build passed.

# 2026-09-16 — visual revision isolation

- Unscoped visual queries now pre-filter the provider by all active document
  revisions before top-k ranking; post-ranking stale filtering was removed as
  a correctness dependency.
- Added a multi-document regression test; visual contract tests **18/18** and
  full backend suite **928/928** pass. Production visual flag remains off.

# 2026-09-16 — Lance migration fail-closed validation

- The isolated Lance migration now validates physical revision columns,
  active-document ownership and vector-dimension consistency before atomic
  staging rename. A read-only migration of the current legacy projection
  passed with **0 invalid rows** across 21 active documents.
- Added migration validator fixtures; full backend suite now passes
  **931/931**. Production LanceDB was not mutated.

# 2026-09-16 — isolated Gate C replay

- Audited `dev-v2.2` replayed against the validated revision-scoped Lance
  projection: A0 Candidate/Evidence/Document **0.97/0.90/1.00**, modality
  minimum **0.833**, multi-paper **1.00**, zero scope leaks, P50/P95
  **142/299 ms**.
- A1 and B-full add no gain; B-anchor is diagnostic-only (Evidence **0.92**,
  P50/P95 **216/405 ms**). Dual-query remains disabled.

# 2026-09-16 — evidence-card candidate revision isolation

- `PaperEvidenceService.candidate_sources()` now applies the same active
  revision contract as retrieval and card materialization. Staged/retired
  chunks, tables and figures cannot be selected for a new evidence card.
- Added a regression fixture and re-ran the full backend suite: **931/931
  passed**. No production projection or feature flag was changed.

# 2026-09-16 — retrieval-cache revision isolation

- Retrieval cache fingerprints now include `active_revision_id` and exclude
  staged/retired artifact and LightRAG provenance rows. A revision switch
  invalidates the cache even when raw content/timestamps are unchanged.
- Added regression coverage; full backend suite remains **931/931 passed**.

# 2026-09-16 — LightRAG provenance active-revision isolation

- Parent provenance resolution now requires an exact active revision match;
  null-revision legacy mappings are rejected for revisioned documents.
- Added stale-provenance coverage; backend suite: **932/932 passed**.

# 2026-09-16 — 21-case public retrieval replay

- Replayed the pinned 21-case public frontier after the generic artifact
  cutoff fix in an isolated runtime: 21/21 completed, no fallback or
  production mutation.
- Diagnostic aggregate: page recall `0.8182`, evidence phrase recall `0.7333`,
  artifact recall `0.3000`, support recall `0.7500`, warm P50 `182 ms` and
  maximum `1,187 ms`.
- Kept this as a diagnostic only; visual-channel and Sol answer-quality gates
  remain open.

# 2026-09-21 — live services and public-format checkpoint

- [x] Verified backend `:7777`, frontend `:5173`, 9router model catalog and
  Ollama embedding service on the host.
- [x] A real SSE agent request reached backend routing/graph, then was rejected
  by 9router with `401 token_revoked`; no fallback was used and Sol E2E remains
  open.
- [x] Completed an isolated 12-case public retrieval-only sample across
  native/mixed/image-only MMLongBench profiles and SPIQA: 12/12 rows, P50/max
  `115/220 ms`, page/evidence/artifact recall `0.6667/0.6667/0.6667`.
  This is diagnostic and below promotion gate.
- [x] Kept the cold visual attempt as an invalid partial checkpoint; no partial
  score was used for promotion.

# 2026-09-21 — resumable visual preparation

- [x] Visual PDF rendering now writes an atomic revision-scoped manifest after
  each page (`partial` checkpoint) and marks it `complete` only after the full
  document finishes. Existing page images remain reusable after restart.
- [x] Visual contract tests: **18/18**; full backend suite: **932/932**.

# 2026-09-21 — generic visual attachment projection

- [x] List/count evidence queries now retain additional ranked visual pages
  after the first composed page (bounded two-to-four frontier); singular visual
  requests and explicit page hints keep their previous limits.
- [x] Added query-shape regression tests: targeted agent/evaluator tests
  **135/135**, full backend suite **932/932**. No feature flag or production
  index was changed.

# 2026-09-21 — visual evaluator configuration contract

- [x] Visual gate checkpoints/reports now record OCR, render scale, device,
  local-files-only and stage-version settings; resume fails closed on drift.
- [x] Cached no-OCR 21-case replay is explicitly diagnostic (`T/V/T+V`
  `0.5238/0.8571/0.9286`) and is not merged with the earlier OCR-enabled run.
- [x] Public evaluator tests: **71/71**.
- [x] Merge/attach now fail closed when visual reports lack or disagree on the
  persisted OCR/evaluation configuration; backend suite remains **932/932**.

# 2026-09-21 — lifecycle r2 and service restart

- [x] Restarted the local backend and frontend; `:7777` and `:5173` serve the
  current source tree for UI testing.
- [x] Fresh isolated `format-lifecycle-matrix-r2.json` completed all six valid
  format retrieval checks and all revision/dedup/rename/alias invariants.
- [x] The intentionally corrupt PDF remains `partial`; no invalid input was
  labelled ready. The matrix used local EmbeddingGemma only and made no
  production or 9router writes.
- [x] 9router catalog and Ollama are reachable; completion remains blocked by
  its revoked OAuth token, so no Sol E2E claim is made.

# 2026-09-21 — visual frontier replay

- [x] Replayed the pinned 21-case visual frontier with the current evaluator
  and cached isolated page/visual runtime: T/V/T+V recall **0.5238/0.8571/
  0.9286**, T+V P50/P95 **226.780/936.900 ms**.
- [x] The evaluator persisted OCR/render/device/stage configuration and the
  run made zero model calls outside local EmbeddingGemma/ColSmol.
- [x] Kept promotion closed because this is retrieval-only and still exposes
  two generic fused-incomplete cases; the required Sol answer-quality replay
  cannot run with the revoked gateway token.
# 2026-09-21 — printed-page provenance

- [x] Added conservative printed-page aliases to the parser/chunker/index
  contract while preserving physical page identity and citations.
- [x] FTS/Lance receive the alias as a retrieval anchor; parser/chunker
  fingerprints force a clean isolated revision instead of mixing projections.
- [x] Backend suite **934/934** and isolated four-format lifecycle smoke pass;
  image-only input remains explicitly `partial` when text recovery is absent.
# 2026-09-21 — page-label lifecycle revalidation

- [x] Full isolated matrix after the parser/chunker fingerprint change passed
  retrieval and all revision/dedup/rename/alias invariants.
- [x] Report: `format-lifecycle-matrix-page-label-r3.json`; corrupt PDF remains
  explicitly `partial`.

# 2026-09-21 — printed-page visual replay

- [x] Replayed one previously failing MMLongBench page-label case after the
  generic parser/chunker provenance change. Text retrieval recovered the
  physical target page (recall `1.0`), while visual-only still ranked the
  neighboring physical page; `T+V` recovered the target only at rank 16.
- [x] Kept the result diagnostic: local ColSmol indexed all 54 pages with
  zero 9router calls, and visual promotion remains closed.

# 2026-09-21 — visual frontier after printed-page resolver

- [x] Replayed all 21 pinned visual cases with the active-revision printed-page
  resolver. T/V/T+V recall is **0.6667/0.9286/1.0000**, with no fused text
  regressions; T+V P50/P95 is **226.717/959.474 ms**.
- [x] Kept promotion closed because answer-quality evidence is still absent.
  The run used local EmbeddingGemma/ColSmol only, zero 9router calls and zero
  production writes; worker health and all 21 rows are valid.

# 2026-09-21 — final local lifecycle/build recheck

- [x] Fresh isolated lifecycle matrix passed native/mixed/table-image PDF,
  DOCX, Markdown and TXT retrieval plus all revision/dedup/rename/alias
  invariants; corrupt PDF remained `partial`.
- [x] Desktop production build passed (`tsc && vite build`).

# 2026-09-21 — visual revision-isolation regression

- [x] Added and passed a regression for retired printed-page metadata leaking
  into visual page resolution; active revision remains authoritative.
- [x] Visual contract tests **20/20**; full backend suite **936/936**.
# 2026-10-05 — durable SSE evaluation and service recovery

- Verified authenticated 9router/Sol completion; started isolated backend :7790
  and frontend :5173 without production schema/index mutation.
- Added safe runtime clone command and atomic per-turn agent checkpoints with
  code/index/image/case contracts, HTTP clone binding, persisted-answer/history
  verification and fail-closed interrupted-run resume. Follow-up case selection
  now includes actual preceding turns instead of fabricating context.
- Real one-turn SSE plus resume passed with original timings retained and no
  second generation. Full backend 1,010 tests passed; Graphify AST synchronized.
- Dev-20 completed 20/20 on existing rubric; added separate structural answer
  audit (16 focused tests) which exposes one empty table/dangling conclusion.
  Retained the report without relabeling rubric success as semantic acceptance.
- Started frozen public 60-case/48-document visible frontier, excluding prior
  dev and shadow families. No claim of completed public/agent acceptance yet.
