# 05 — TODO và vấn đề tồn đọng (2026-10-10)

## RAG v1 RC1 — 2026-10-22

- [x] Regression on the final SHA, acceptance content-judged, rollback drill
  (`12-RELEASE-RAG-V1.md`).
- [ ] Production rollout of `3219b2bc` on `data/` — needs the user's approval.
- [ ] Maintenance backlog (not blockers): see `12-RELEASE-RAG-V1.md` §Maintenance.
- [ ] Next phase: OpenClaw-style tools on the router → evidence → answer →
  verify pipeline.

## After sealed v3 — 2026-10-21

- [ ] Re-measure v3 (dev) after `072984ec`; then a fresh sealed v4 written by
  an independent model for the next out-of-sample number.
- [ ] Remaining v3 failures not yet addressed: v3-09 "which of my files come
  from EIA" (library inventory by publisher), v3-04/74 single facts missed in
  long native PDFs, v3-59 image-only slide content, v3-56/70 required details.
- [x] Catalog titles (generic, legacy-encoded, section names) — `ae92f69f`.
- [ ] Catalog: wrong card language (English LIGO papers tagged vi),
  `_phrase_document_count` tokenizes with [A-Za-z0-9] so Vietnamese words with
  diacritics break.
- [x] Startup backfill no longer activates an empty placeholder for an
  interrupted ingest — `0f267f61`.
- [ ] 9router listens on *:20128 (all interfaces) — bind to localhost.

## Sealed evaluations — 2026-10-16

- [ ] Sealed v2 50.0% (gate 85%). Rerun of failing conversations after the
  generic fixes pending; scanned tables (OCR diacritics "Quàng Ninh", split
  columns) and FOMC Table 1 retrieval are the next suspects.

- [ ] Scanned PDFs keep the filename as title (OCR headers are garbled), so
  description matches favour titled documents (sealed v1 s22). Derive a
  title from OCR text robustly or use the router's view of content.
- [ ] Hybrid fusion vs pure vector: doc hit@1 0.850 vs 0.917 on sealed v1
  (card routing / channel mix suspected; FTS pruning and FTS weight were
  measured and rejected).
- [x] Re-login 9router account the uclab account (401) — done 2026-10-16.
- [x] v1 comparisons s33/s34/s40/s41 — correct on rerun7 (2026-10-16).

## Claude continuation — 2026-10-14

- [ ] `revision_projection_identity_mismatch` on re-index of some documents is
  pre-existing (old Lance layout unreadable), benign for answers; rebuild the
  projection for those documents when convenient.
- [ ] Post-verify misses some wrong prose numbers when nothing flags them
  (13/60 v3 wrong answers unflagged: period / negation / ambiguous /
  borrowed). Sol judges them correctly when asked (23/24), so the gap is the
  flagging step, not the verifier. Widen flagging only if it keeps correct
  answers' extra calls low.
- [ ] Per-document context cap stays on purpose (removing it: public-36
  0.653 → 0.514).
- [x] Terra content judge: `scripts/judge_aya_acceptance.py` (use it; the
  structural checker passes answers that miss facts).
- [ ] Judge is strict on sub-points (caption wording, "three parallel
  streams"); review its fails by hand before fixing.
- [ ] dev-20 in post mode; the sealed multi-format set (plan step 3) is still
  not built.

## Latest completeness continuation — 2026-10-10

- [x] Declared Paper/model provenance and composite model/dataset slash cells,
  sentence-level paper binding, draft compaction before string clipping.
- [x] Sanitizer recognizes unsafe sentence inside a labelled paragraph without
  clearing valid table identity cells; captured prior drafts pass after removal.
- [x] Completeness v2 flags partial/answer fallback even when internal rubric
  passes. This is a review trigger, never proof that every retrieved table is
  answerable. Latest **4/4 affected rubric** has **1 flagged partial answer**.
- [ ] Fix dotted condition/metric cells and remaining owner/scope ambiguity from
  complete source-bound traces, with synthetic paired controls; do not treat
  `slice.metric` as model names or accept global matching numbers. Latest
  sanitized output loses correct later results; not an E2E quality gate.
- [ ] Stop using dev-20 validator success as semantic completeness. Add external
  task-facet/source scoring; retain the failed full **19/20** report and the
  partial **4/4 rubric** report separately, not a merged claimed full pass.
- [ ] Public frontier context drop-reason telemetry and fixed 36-case failure
  attribution are still next retrieval work; helper recall **0.40278** remains.
- [ ] Independent multi-format/visual gates and 8 numeric false accepts/8 false
  rejects remain open. No promotion, facts/dual-query off, production unchanged.

## Latest numeric continuation — 2026-10-10

- [x] Preserve bounded full drafts/failures when trace exceeds 64 KB, without
  weakening redaction/cap; verified live and via security tests.
- [x] Typed paper/model local ownership, duplicate metric occurrences, OCR
  decimal offsets and numbered-document heading scope repaired.
- [ ] Unlabeled cross-source caveat values remain ambiguous; do not replace
  source attribution with global value matching. Full frozen dev-20 **19/20**,
  final comparison fallback; packed-cell precedence replay **3/4**; subsequent
  separate Paper/Model ownership repair has no live coverage yet. Fix abbreviated
  paper identities/composite owner scope offline before spending more Sol calls.
- [ ] Trace compaction currently activates only after hard-cap fallback;
  pre-fallback string clipping still loses middle draft sections. Compact
  successful proofs before competing draft strings lose essential diagnosis.
- [ ] Public 36-case artifact-selection diagnostic remains **0.40278**;
  pipeline-level failure attribution, independent arbitrary-format answers
  and visual acceptance remain open, separate from numeric guard fixes.

- [ ] Latest visible replay still has **8 false accepts / 8 false rejects**:
  year/condition attribution, inherited comparison predicates, qualitative
  negation, omitted population and located arithmetic. Explicit source-subject
  binding repaired three other failures, not arbitrary prose understanding.
- [ ] Latest frozen Sol dev05 is **3/4**, last comparison uses answer fallback
  after two attempts (**42.500 s**). Resolve metric/owner scoping and condition
  handling from captured drafts, without weakening wrong-claim controls.
- [x] Trace of the failed comparison exceeded 64-KB cap and retained only a
  truncation marker. Tool-call attempt validations survive, but not full drafts.
  Compact validation milestones now retain full drafts/failures; old failed
  reports remain preserved, not replaced by a randomly successful rerun.
- [ ] Clean full frozen dev-20 and independently reviewed balanced cross-domain
  evaluation remain pending. Facts/dual-query still off; no retrieval gain claim.

## Codex continuation — open, 2026-10-09

- [ ] Numeric/prose semantic binding still fails **11/85 wrong claims** and
  rejects **8/87 correct controls** on latest visible 172-case replay. Cached facts
  bring no gain under the stronger contract; keep `FACT_GROUNDING_ENABLED=false`.
- [ ] Independent ≥120 balanced/annotated suite, budgeted new fact extraction,
  flag-off/on comparison and clean full frozen dev-20. Latest full run has
  **19/20 + 1 upstream disconnect**; bounded replay **4/4**, one answer retry
  and 38.475 s worst turn. Not full acceptance or a latency improvement claim.
- [ ] General unit conversions/arithmetic, explicit not-applicable assessments,
  and automatic fact-worker startup/coordinator scheduling remain incomplete.
- [x] Old dev05-t4 0.514 rejection traced to caveat owner inheritance and
  composite benchmark binding; general fix and final frozen replay pass.
- See [07-NUMERIC-GROUNDING-REVIEW.md](07-NUMERIC-GROUNDING-REVIEW.md) before
  continuing; do not use the invalid 10-turn run as a promotion gate.

## Claude — open after numeric grounding (2026-10-09)

- [ ] Report the second blind probe set honestly (pending).
- [ ] Old metric validator false rejects (~19/43 correct claims on probes:
  owners in columns, "Model – Dataset" cells, "Ours"/"full model" aliases,
  Vietnamese tables). Correct answers then retry or fall back → unnatural.
  Fix inside the metric validator; do NOT let grounding override it.
- [x] Numeric condition columns bind peer row values; Week/tuần alias and
  per-clause scope verified, including answer tables (2026-10-09). Broader
  prose conditions and semantic unit conversions remain open.
- [ ] Pure-prose evidence has no row/column structure: borrowed values across
  entities in prose are only caught by the metric validator.
- [ ] Live false-reject rate of grounding: dev05-t4 correction flagged 0.514;
  raise debug trace size or log grounding mentions to verify.
- [ ] Detector gaps from probes: result-table intent misses finance/medical
  asks; reference parser misses "Tab. 2", "Exhibit", "Biểu đồ/Sơ đồ/Đồ thị",
  "Table A.2", lower-case roman; years after "and" become table numbers;
  table role misses "hazard ratios"/"baseline characteristics".

## Claude review — open items after P0 (2026-10-08)

P1 context selection (do next, no live E2E claims before it):
- [ ] Drop-reason telemetry in `compose_retrieval_context`; diff the 20
  zero-recall public cases one by one instead of global tuning.
- [x] Reference expansion gated by relevance order + budget (2026-10-08).
- [x] Main-result inventory capped at 2/document, ranked by query.
- [x] `table_result_role` VI captions, result-before-statistics, metric header.
- [x] Same-page logic only for paged formats; facets ignore sibling text.
      (Promotion/render of anchor `expanded_content` still includes siblings
      for PDFs — acceptable as prompt context, never rendered directly.)
- [x] Hardcoded paper aliases replaced by catalog-derived vocabulary.
- [~] Generic metric lexicon added for table roles; SER-specific lists remain
      in paper_facets/rag_service query expansion.
- [x] Cache key includes visual/rerank config (enrichment fingerprint still
      not included).
- [ ] UX: numeric answers buffered for validation (first text 7–14 s). Design
      streaming with incremental claim checks or a visible "checking" state.
- [ ] Provider: 9router Sol hit `429 usage limit` twice in one evening;
      a 20-turn suite cannot finish in one window. Decide (user) whether evals
      may use terra/luna or must wait for the window.

Validator leftovers:
- [ ] False negative: `ORION CCC V/A/D: 0.74/0.63/0.42` (correct) rejected —
  slash vectors get one qualifier. Causes retry, not fabrication.
- [ ] Split `evidence_validator.py` (5.7k lines) into table/prose parsing,
  answer binding and matching; then move toward structured (owner, condition,
  metric, value) records extracted at ingest.
- [ ] Generated title-fragment aliases can mark a paper "covered" from a
  generic phrase ("speech emotion recognition").

P2 evaluation credibility:
- [ ] Current suites are tuning/regression only: dev-v2.2 re-run per change,
  qrels refreshed from own chunks, "independent audit" by Sol; Document R@5 is
  guaranteed by construction; dev-20 rubric has empty answer_must_contain and
  grades with the system validator. Build a sealed multi-format corpus with
  human questions/answers, independent judge, paired tests + CIs.
- [ ] Metric bugs: nDCG can exceed 1 (no per-group dedup); three-mode means
  are unpaired on failures; negative cases have empty forbidden_documents.

P3 structure: `api/agent.py` 7.5k lines (visual/map-reduce orchestration in
API layer, no total timeout/call cap), `rag_service.py` 5k lines.

## Latest actions — 2026-10-08 KST main-result retrieval and honest final scope

- [x] Analytical comparisons gather canonical main results without enabling
  direct-table rendering. Own purpose/metric/artifact precedence preserved.
  Observed MSF-SER main IEMOCAP Table 3 is now in selected evidence and answer.
- [x] **1,168 backend pass**, 130 selection/context pass, 29 evaluator pass,
  desktop build; full Sol dev-20 **20/20 rubric**, zero fallback. Structural
  audit clean; manual audit still finds a mixed-paper table under a single-paper
  ASPIRE/IEMOCAP heading. Do not equate these checks with full semantic quality.
- [x] Remove gold qrel document scope from final-stage selection. Runtime-only
  60-case gate retains **0.97/0.90/1.00**, P50/P95 **124.7/341.8 ms**. Old affected
  reports are invalid for promotion. B-full has no activation/gain; flag off.
- [ ] Public helper **0.40278**, twenty zero-artifact cases remain. Traverse
  authored parent/reference -> canonical artifact relationships for implicit
  asks, rather than requiring users to say table/figure. Score only after
  selection; avoid artifact/Qrel filenames as query-specific routing rules.
- [ ] Mixed-paper enclosing labels and post-suppression semantic completeness;
  wider actual-image/DOCX/MD/TXT/mixed-PDF answers, visual A/B, frozen RC and unseen
  shadow-v3 remain unaccepted. Production index and embeddings remain unchanged.
- Backend :7790 **21260 / PID 24012**, frontend :5173; cache v31. Graph
  **4,789 / 13,854 / 176**, Obsidian **4,895 notes**. No live evaluator/test jobs.

## Latest actions — 2026-10-08 KST typed facet/owner completion checks

- [x] Own-facet dataset/architecture artifact preference and contextual-task
  role filtering: public 36 selected-artifact recall **0.375**, seven paired
  improvements, zero regressions/LLM calls/catalog mutation. Local 60 maintains
  A0 recall **0.97/0.90/1.00**, P50/P95 **130/370 ms**; dual stays off.
- [x] Packed summaries retain owner/split regardless of header wording; typed
  owner cells, not incidental title fragments, may reassign document identity.
  Final **1,157 backend pass**, 191 focused guard pass, desktop build pass.
- [x] Full dev-20 completed on the pre-final-scope source. Manual audit found a
  hidden wrong-section table; final-source four-turn Sol replay completed with
  corrected table ownership and no fallback. Keep both, do not blindly restart.
- [ ] Main-results selection still retrieves only an ablation table for one
  comparison document. Repair artifact-purpose/parent-reference selection and
  assess per-document result completeness, rather than making the guard accept
  numeric history absent from selected evidence. No fixed paper -> table map.
- [ ] Public helper is still **0.375**, 21/36 cases have zero selected-artifact
  recall. Diagnose relation/implicit artifact selection across the whole fixed
  frontier. Actual images/answers, format strata and unseen quality are not
  proved by this raw-helper metric or green unit/rubric results.
- [ ] Broader approved image/format semantic tests, visual A/B, frozen RC,
  unseen shadow-v3 and rollback remain unaccepted. No production migration.
- Backend :7790 **22252 / PID 19811**, same isolated clone; frontend :5173.
  Cache v30; dual/visual/LightRAG off; gateway not changed this interval.
  Graph **4,779/13,824/180**, Obsidian **4,885 notes**, existing warnings remain.

## Latest actions — 2026-10-08 KST purpose/provenance/owner binding

- [x] Own-artifact purpose and sibling payload/anchor contamination repaired;
  JAIT/KST fusion evidence appeared in the affected-family replay. Do not
  extrapolate that observation to unseen documents or semantic acceptance.
- [x] Numeric owner/schema/reference repairs pass 185 focused tests; old failed
  comparison draft checks **34/34** against all active canonical tables.
- [ ] Run full suite and fresh HTTP/SSE on the latest source, including actual
  selected evidence and post-suppression completeness. Earlier 20/20 and 4/4
  rubric runs still contained hidden failures and predate the final fix.
- [ ] Public helper selected-artifact recall is **0.1944**, below the preceding
  **0.2222**; inspect explicit facet reservation and implicit artifact selection
  before any promotion. No gold labels injected into runtime query/context.
- [ ] Broader approved actual-image agent quality, stratified arbitrary-format
  evaluation, visual A/B, frozen RC/unseen shadow and rollback gates remain open.
- Local audited retrieval thresholds met without B-full benefit; keep dual and
  visual defaults off. Backend :7790 now **12305 / PID 3163** in the same clone;
  production unchanged. Latest graph **4,762 / 13,781 / 180**, warnings remain.

## Latest actions — 2026-10-08 KST full dev-20 after facet/schema fixes

- [x] Scoped explicit facet expansion and schema/condition-owned table projection
  repaired generically. Full backend **1,129 passed**, desktop build, AST sync.
- [x] Full Sol dev-20 session **18758** is terminal, **20/20 existing rubric**;
  zero transport/tool/model/claim-guard fallback, one correction, one suppressed
  block. Median total/visible **9.21/2.99 s**. Index and source match frozen run.
- [x] Local audited 60-case/four-arm retrieval gate rerun retains A0 thresholds;
  no distinct B-full activation/gain, so dual remains off. This run preceded
  final schema projection, and frozen rewrites do not test runtime rewriting.
- [x] Same public 36 raw-query helper cases complete, **0.2222** artifact/potential
  recall unchanged, no LLM/catalog mutation. Does not test runtime rewrite or
  actual model image inputs. Both completed runs must not be restarted blindly.
- [ ] Semantic hidden failures remain: JAIT/KST comparison omits fusion evidence
  and turns missing excerpt into a wrong high-level characterization. Active
  canonical pages contain the fusion descriptions. MSF-SER/Mamba comparison
  inserts dataset statistics under MSF-SER results and lacks main result rows.
  See report-bound manual audit; a clean structural audit does not cover these.
- [ ] Strengthen whole-document facet/relationship coverage and artifact-purpose
  selection before final prompt; validate semantic coverage after suppression.
  No paper-name → table/figure mapping and no per-case prompt exemptions.
- [ ] Numeric referenced values in a cross-metric caution still get interpreted
  as asserted scores; repair with antecedent/provenance-bound adversarial tests.
- Backend :7790 **13268** same isolated 21-paper root; frontend :5173 and router
  remain up. Cache v27, dual/visual false, production unchanged. Public visual,
  arbitrary-format semantic, unseen shadow and full RC requirements remain open.


## Latest actions — 2026-10-08 KST actual answer pixels and trace findings

- [x] Final native Sol health pass; same-version recovery after verified gateway
  hang. Backend :7790 (40208/PID 15274) and frontend :5173 (23843) HTTP 200.
  No provider/model fallback or production data changes.
- [x] Canonical approval/active-revision image forwarding independent of visual
  search flag, digest-only trace telemetry, actual-pixel visual guard, generic
  inventory correction; full backend **1,118 passed** and AST graph synchronized.
- [x] Completed 36-case public helper parity diagnostic and four-turn Sol SSE;
  preserved old timeout report. Do not rerun completed handles or claim the
  weak public score (0.2222) and 3/4 rubric as semantic acceptance.
- [ ] Dataset/facet query resolves to architecture/introduction expansion and
  misses evaluation dataset evidence. Agent then introduces unsupported names
  while numeric-only guard passes. Repair query/evidence obligation coverage
  and semantic evaluation rather than adding a paper-name/dataset whitelist.
- [ ] Numeric guard incorrectly infers metric ownership for an explicit warning
  against comparing differently defined values. Distinguish cited/reference
  numbers from asserted metric assignments; keep true wrong-owner claims rejected.
  Initial comparison dimensional/split failures must remain visible too.
- [ ] Public final selection remains weak; evaluate actual approved native pixels
  with fixed non-gold selection and semantic scoring. One supplied image proves
  wiring, not visual answer accuracy. Broader format/visual/RC gates remain open.
- [ ] Audit canonical page/caption identity in image labels and cropped-provider
  page provenance before visual promotion. Do not blanket-enable approval/visual
  flags to bypass canonical ownership checks.
- Dual-query/visual defaults remain false; no production migration/cleanup or
  shadow-v3 acceptance. Next code changes need affected-family fixtures + wider
  Sol replay and refreshed retrieval fingerprints.


## Latest actions — 2026-10-06 revision-context fix and native public results

- [x] Native public-60 session **57284** is terminal, **60/60**, zero failures
  and LLM calls. Do not poll/restart this completed run. A source-image/revision
  audit validates all 48 active documents and 255 official image digests.
- [x] FigureEnrichService default context revision bug fixed generically;
  staged/retired cards cannot leak, page context follows the selected image,
  completion cannot mark another revision after concurrent activation.
  Same-clone section/table context gaps **121/40 -> 0/0** across 136 figures.
- [x] Full backend **1,092 passed**, **75** focused tests, desktop build passed;
  two actual public Sol VLM assets enriched and two HTTP/SSE agent dev turns
  pass. These small live checks do not establish broad answer quality.
- [ ] Public final selection remains weak: candidate artifact **0.9306** but
  final-context artifact-name **0.5833**, any-gold-page hit **0.7143**. The new
  audit observes **13 implicit-query cases** losing correct candidate artifacts
  at final context, plus three candidate-artifact and seven candidate-page misses.
  Fix query/content-based artifact selection and source-to-pixel projection;
  do not exempt implicit asks or insert gold artifact names into rewrites.
- [ ] Evaluate provenance-bound enrichment consumption (especially image tables)
  and actual supplied-pixel coverage on a fixed public frontier. Keep optional
  VLM work isolated/policy-gated; all assets of selected documents must be
  selected independently of questions/qrels, not only known gold images.
- [ ] Wider semantic-completeness/agent regression, visual/public format quality,
  frozen RC, unseen shadow-v3 and rollback acceptance are still outstanding.
- Current backend **27739/PID 46224** :7790 health 200, same durable 21-paper
  root; old **15074/PID 37848** is terminal. Frontend :5173 health 200. User
  9router/Ollama untouched; dual-query and visual flags off; goal remains active.

## Latest actions — 2026-10-06 native public image fidelity

- [x] Public SPIQA reader now uses pinned native archive and fails closed;
  all 48 authored assets of six probe documents were historically 224x224.
  Same engine reads native table glyphs; this corrects inputs, not ranking.
- [x] Full backend **1,060 passed**, focused public **101 passed**, public full
  download/ZIP validation pass and AST graph synchronized.
- [x] Same frozen 12-case/36-generation Sol probe session **34901** completed,
  output `public-answer-probe-native-20261006.json` under public/results.
  aggregate answer quality did not improve. Preserved separate lifecycle audit:
  alias dedup had changed six documents' image bytes under old active revisions.
  This report is input/answer diagnostic only, invalid for lifecycle acceptance.
- [x] Alias policy changes retain logical identity and stage a new revision;
  active artifact changes fail before image writes. Full **1,065 passed**, 113
  focused passed, additional real-Lance alias crash variant passes. Both fresh
  and already-known alias cases preserve active canonical content on failure.
- [ ] Finish broader native public retrieval session **57284** (confirmed live,
  not stopped). New clone `public-native-lifecycle-20261006`, output
  `public-foundation-dev-v2-native-20261006.json`. Latest **3/60**, zero failures;
  next paper has 15 table images, canonical-artifact receipt pending and job
  heartbeat advancing. Re-poll this handle; do not restart on observation timeout.
  continue diagnosing implicit image selection and final context losses without
  using gold labels/answers for query creation or ranking.
- [ ] Audit default FigureEnrichService context helpers: when revision_id is
  absent, active figures may receive NULL-revision chunks/tables and document
  cards may join without revision filtering. Verify with staged/retired fixtures
  before changing code; do not edit dependencies during the frozen live probe.
- Public/visual, semantic completeness and unseen RC requirements remain open.
  Backend :7790 session **15074/PID 37848** and frontend :5173 health pass;
  old backend 63710/PID 27221 stopped. 9router Sol pass, no fallback.

## Latest actions — 2026-10-06 condition-schema repair

- [x] Schema/owner/loss/negation repair with 177 focused tests. Audited local
  Gate C rerun meets A0 thresholds; no treatment activation/gain justifies dual.
- [x] Full backend session **58180**: 1,058 passed. Final four-turn SSE session
  **62411**: 4/4 rubric, no empty tables, twelve supported comparison claims,
  one attempt, no fallback; both sessions terminal. Desktop build pass.
  Output: `agent-condition-binding-final-replay-20261006.json` under internal.
- [ ] Semantic completeness still fails: final comparison conclusion covers
  GBAN only; retrieved CRAB results emphasize ablation, not main benchmark.
  Audit artifact-purpose selection and post-suppression task coverage. Do not
  equate a clean empty-table audit or two named documents with complete comparison.
- [ ] First five-turn replay completed 5/5 rubric but comparison/negative artifact
  outputs still contain empty tables. Do not hide the failures or widen numeric
  matching to accept ambiguous flattened loss rows. Explicit missing-artifact
  requests must not substitute other tables or speculate about another number.
- [ ] Finish semantic completeness/affected-family replay and wider dev-20/60
  agent regression. Public page/implicit-image gates still fail; no shadow-v3,
  frozen RC or production rollout yet.
- Active backend :7790 **63710/PID 27221**, same durable isolated clone;
  previous 68063/PID 24271 and 65772/PID 17180 backends are terminal. Frontend
  :5173 remains live; verified 9router does not need restarting.

## Current next actions — 2026-10-06

- [x] Six-turn family session 2086 is terminal: **4/6**, not acceptance.
  Keep h03-t4 coverage failure and h15-t1 negative/empty-table failure visible.
- [x] Fix wide metric-header ownership and redundant intrinsic weighting;
  real h06-t1 SSE replay (82611, terminal) contains all three table rows,
  twelve UA/WA values and thirteen supported claims with zero suppression.
- [x] Full backend **1,030 passed**, desktop build pass, Graphify synchronized.
- [ ] Fix generic owner/config binding: loss names such as MLCS are treated
  as model owners; models named in Condition cells can be ignored; blank
  rowspan owners in flattened source tables lose provenance. Tests must reject
  swapped encoder/loss/value combinations, not merely remove bad qualifiers.
- [ ] Explicitly missing artifact requests must not speculate on another table
  number. Audit negative-case rubric separately; preserve the failing answer
  and don't count a phrase-regex pass as semantic completeness.
- [ ] Complete local Gate C rerun and wider HTTP/SSE regression after remaining
  binding repairs; public page/implicit-image quality and shadow/RC gates remain
  incomplete. No production mutation or feature promotion.
- Current services: isolated backend session **65772** PID **17180**, frontend
  **85711**, 9router authenticated completion verified **1.97 s**. Earlier
  backend sessions 49899/74579 are terminal; don't poll stale process handles.

## Current next actions — 2026-10-05

- [x] Public-60 sessions 17730/53617 and visible regression-60 session 69334
  are terminal, not live. Public answer probes 76625/66038 are also complete.
- [x] Fix same-page canonical artifact dedup and selected-attachment priority;
  before/after public results are mixed, not a promotion win (see current status).
- [x] Repair explicit-row metric vs caption inference and transposed caption
  matrices with synthetic wrong-owner/fold-average rejection fixtures;
  full backend **1,028 passed**.
- [ ] Finish six-turn affected-family replay session 2086 and inspect actual
  answer rows/prose against canonical evidence, not only existing rubric.
  Backend session 49899, PID 3287; previous 97795/3427 backends are terminal.
- [ ] Public implicit evidence coverage remains a real miss: candidate artifact
  presence does not ensure final pixels. Current no-planner diagnostic receives
  no images for a language inventory and a wrong table for a numerical query.
  Verify production planner path separately; do not infer production quality
  from a bypassed planner, or rewrite gold labels into queries.
- [ ] Public page/final-artifact gates still fail; after current context fixes
  rerun audited local Gate C before freeze/shadow/production migration.

- [x] Fix authored-attachment lifecycle/projection mismatch before activation;
  same 21-case public frontier restored candidate artifact recall **1.00** and
  phrase recall **0.8738** without ranking/scorer tuning or LLM calls.
- [x] Public/visual checkpoint contract rejects stale code/options/source and
  authored-image bytes; atomic checkpoints preserve the previous write on failure.
- [x] Final full backend suite **996 passed** and desktop build passed after
  physical identity checks; AST graph synchronized.
- [ ] Classify remaining final-artifact projection losses, expand frozen public
  strata and run multimodal answer quality with actual supplied image coverage.
  Five SPIQA questions have gold artifacts present in candidates but omitted
  from final images; none explicitly requests a table/figure. That does NOT
  prove the images are unnecessary. Test implicit numerical/visual evidence
  needs and paired answer completeness; do not exclude these failures or change
  the scorer to make the headline artifact score green.
- [x] Add durable per-turn checkpoints to agent eval and move isolated clones
  into ignored workspace runtime roots before rerunning visible regression-60.
  The old temporary clone/session is gone; no final 60-case report exists.
- [x] Sol service reverified: authenticated models/completion HTTP 200, latest
  1.43 s. Old refused probes are superseded; do not ask the user to start it.
- [x] Dev-20 session 65974 completed 20/20 on current rubric; median total/first
  visible 11.83/4.15 s. Independent structural audit flags one incomplete
  answer (empty result table/dangling conclusion); do not claim full quality.
- [ ] Complete current public-60 session 17730 and regression-60 session 69334 without
  changing their code/index contracts; results remain partial until terminal.
  Durable clone root: `data/retrieval_eval/internal/runtime/e2e-foundation-20261005`.
  Backend :7790 session 97795; frontend :5173 session 89379.
- [ ] Audit semantic completeness of stored answers, including guard-suppressed
  comparisons. New structural audit is a diagnostic only, not entailment or a
  replacement semantic rubric. Retain the dev05-t4 failing content for replay;
  fix the general suppression/composition family, not its paper names.
- [ ] After frozen baseline completes, fix same-page distinct-figure loss in
  context composition. Synthetic two-figure input produces only one source
  despite min_figures=2; page-level dedup must not equate separate artifacts.
  Also audit selected-artifact attachment priority versus page catalog fill.
- [ ] Keep dual-query/visual disabled; unseen shadow-v3, frozen RC, broader
  arbitrary-format quality and rollback acceptance remain incomplete.

## Earlier acceptance runs — 2026-10-04 (terminal, not live)

- [x] Shared visual report eligibility and scored-answer frontier validation;
  **991 full-suite tests**, then **25 focused pairing/report tests** after the
  final guard; desktop build and Graphify AST update completed.
- [x] Current isolated A0 Gate C passed; **20/20** Sol dev regression and **7/7**
  focus/history/image E2E passed. Do not generalize these visible-case scores
  into unseen/public quality claims.
- [x] Public frontier session **85914**, PID **9788** is terminal; output
  `data/retrieval_eval/public/results/public-retrieval-policy-21-20261004.json`.
  **21/21** completed, but exposed revision projection losses. Superseded by
  the fingerprinted corrected replay in the current section.
- [ ] Visible regression-60 session **76762** was interrupted; output
  `data/retrieval_eval/internal/agent-regression60-policy-20261004.json`.
  is absent. Backend clone session **61948**, PID **10835**, root
  `/private/tmp/aya-e2e-policy-20261004` no longer exists. Do not poll stale handles.
- [ ] Visual session **16313** is terminal (exit 130): intentionally interrupted
  because it used pre-fix code and disabled OCR. Its one completed case/cache
  is diagnostic only; do not resume scores under the new evaluator fingerprint.
- [ ] After the public frontier completes, audit failures/format coverage and
  compare current Sol answer quality before RC freeze or shadow-v3 opening.

## Current remaining validation — 2026-10-04

- [x] Authenticated Sol completion through 9router succeeds; do not mistake
  sandbox localhost denial for provider outage.
- [x] OCR/materialization configuration changes create distinct staged
  revisions/jobs without replacing the old active revision prematurely.
- [x] Propagate complete-frontier and production-text-control guards through
  visual report merge/recompute/answer-quality postprocessors; main evaluator
  guards alone must not be treated as sufficient for every promotion path.
- [ ] Complete a frozen-code public visual run with production OCR baseline.
  The current OCR-disabled run is diagnostic only, regardless of its score.
- [ ] Public arbitrary-format quality, frozen RC and unseen shadow acceptance
  remain outstanding; 982 unit tests and generated lifecycle fixtures do not
  close those gates.


## 2026-09-24 — visual gate remains open after backpressure hardening

- [x] Added generic resource safeguards: explicit page/slide resolution before
  visual semantic search, exact ColSmol scoring batches of 16, and bounded
  2M-pixel rendering / 4M-pixel worker request batches. Backend suite:
  **975/975**.
- [ ] Rerun a completed stratified 20+ case T/V/T+V evaluation from a frozen
  current-code runtime. Existing interrupted reports cannot be aggregated or
  used to promote the visual provider.
- [ ] If CPU-only ColSmol cannot complete that bounded background evaluation,
  record the provider as non-promotable and keep the flag off; compare a
  compatible production visual backend in an isolated A/B rather than
  weakening text/table retrieval or tuning individual cases.

## 2026-09-24 — visual smoke fail-closed

- [x] Ran a fresh ColSmol smoke against the current isolated projection. The
  worker loaded but the first visual index exceeded 180 seconds; the runner
  aborted with **0/1 completed** and no fabricated recall. Report:
  ``data/retrieval_eval/public/results/visual-gate-current-smoke-20260924.json``.
- [ ] Visual provider needs a bounded warm-index/background policy and a fresh
  stratified 20+ case run before any T/V/T+V promotion. Keep
  ``VISUAL_RETRIEVAL_ENABLED`` off; historical cached-index scores are not
  authoritative for the current code.
- [x] Resume checkpoint verified one image-only case at T/V/T+V recall
  **1.0/1.0/1.0**, but required **165 s** cold visual indexing and was stopped
  before the remaining cases; partial metrics are not a gate result.
- [x] Split durable processing into one local lane and one enrichment lane, so
  a cold visual/VLM job cannot starve a new folder's local parse/FTS/LanceDB
  activation. This fixes scheduler isolation but does not solve ColSmol cold
  latency or promote the visual provider. Full backend suite: **972/972**;
  the isolated lifecycle matrix retained all invariants and six expected
  retrieval hits.

## 2026-09-24 — arbitrary-format smoke checkpoint

- [x] Isolated PDF/DOCX/MD/TXT lifecycle smoke passed **4/4** projection and
  retrieval contracts. DOCX/Markdown table/image rows were preserved; the
  image-only scan PDF remained correctly ``partial``. Report:
  ``data/retrieval_eval/internal/format-lifecycle-smoke-20260924.json``.
- [ ] Expand this smoke to a stratified public corpus and crash/restart
  lifecycle run before declaring arbitrary-format Gate D complete.

## 2026-09-24 — audited Gate C result

- [x] Ran audited ``dev-v2.2`` (60 cases) through A0/A1/B-full/B-anchor with
  zero unresolved qrels, scope leaks or model calls. A0 passed
  candidate/evidence/document **0.97/0.90/1.00**, modality minimum **0.8333**,
  multi-paper **1.0**, and P50/P95 **122.697/356.881 ms**.
- [x] Confirmed A1 and B-full add no recall; B-anchor's +0.01 evidence gain
  costs P50/P95 **203.213/414.213 ms** and fails the treatment-gain rule.
- [ ] Keep ``DUAL_QUERY_RRF_ENABLED=false``. Next work is current-RC agent
  HTTP/SSE regression plus final-context and visual stratified validation, not
  more dual-query tuning.

## 2026-09-24 — current-code agent regression

- [x] Current-code isolated Sol HTTP/SSE replay passed **7/7** with no errors or
  fallback; focus, validated table evidence and Figure 1 image delivery (HTTP
  200) all passed. Report:
  ``data/retrieval_eval/internal/e2e-agent-conversation-v6-20260924.json``.
- [ ] Gate E still needs a visual provider that can finish a cold index within
  a bounded background budget; do not promote ColSmol from historical cached
  runs.

## 2026-09-24 — shadow-v3 preflight

- [x] Shadow-v3 structural validation passes (24 questions, 23 positive qrels,
  one abstention, six profile/modality buckets); it remains sealed and was not
  opened.
- [ ] Open shadow-v3 only after visual/public Gate E and RC fingerprints are
  frozen; preflight validity is not a retrieval-quality result.

## 2026-09-24 — post-rollback retrieval checkpoint

- [x] Replayed the isolated 21-case public frontier after testing a generic
  metric context heuristic; the heuristic regressed phrase recall and was
  removed. Post-rollback report:
  ``data/retrieval_eval/public/results/public-retrieval-artifact-v11-reverted-context-20260924.json``.
- [x] Current diagnostic values: candidate page/evidence/artifact recall
  **1.000/0.9833/1.000**, final evidence phrase recall **0.8595**,
  query-scoped artifact support **1.000**, support recall **0.9500**, P50/max
  **109.988/540.858 ms**, 21/21 successful.
- [x] Verified 9router ``cx/gpt-5.6-sol`` and Ollama ``embeddinggemma:300m``
  independently; the former is the answer/VLM provider and the latter is the
  local embedding provider. No fallback was used.
- [ ] Do not promote from this slice: classify final-context losses and run a
  larger stratified public/held-out suite before changing retrieval behavior.

## 2026-09-24 — current retrieval gate after artifact correction

- [x] Fixed generic artifact-selection pollution from inherited page context
  and deduplicated table/figure image aliases before ranking.
- [x] Reran isolated 21-case public frontier: page/candidate-page recall
  **1.0**, evidence recall **0.8595**, final-context artifact recall **0.50**,
  candidate artifact recall **1.0**, support recall **0.95**, P50 **111.43 ms**,
  max **563.76 ms**; 21/21 rows succeeded.
- [x] Full backend suite passes **970/970**.
- [x] Added explicit first-stage candidate page/artifact recall@30 fields and
  separated candidate misses from final-context projection misses.
- [ ] Classify remaining final-context artifact/evidence misses by
  qrel/parser/document routing/FTS/vector/fusion/rerank/composer rather than
  tuning individual cases.
- [ ] Close Gate C/E only after candidate @30, every modality, multi-paper
  coverage, P95 and scope/model-call checks are measured together.
- [x] Reran authenticated Sol HTTP/SSE regression against the corrected code;
  it passed **7/7** with table/figure
  provenance and focus checks; report is
  ``data/retrieval_eval/internal/e2e-agent-conversation-v5-20260924.json``.
- [ ] Keep ``DUAL_QUERY_RRF_ENABLED`` and ``VISUAL_RETRIEVAL_ENABLED``
  disabled until the retrieval gate and isolated visual A/B pass.
- [ ] Evaluate visual T/V/T+V only in the isolated visual gate; do not promote
  ColSmol/ColPali or open ``shadow-v3`` before the retrieval RC is frozen.

## 2026-09-24 — agent smoke complete; retrieval gate remains open

- [x] Authenticated Sol HTTP/SSE regression completed **7/7** turns with no
  fallback, transport error, or invalid evidence result.
- [x] Figure attachment and table evidence were exercised against the live
  backend; this does not substitute for the public retrieval gate.
- [ ] Improve canonical figure/table evidence generation and selection:
  current public artifact recall is **0.30** and evidence recall **0.7167**.
- [ ] Re-run stratified visual A/B only after artifact identity parity is
  fixed; keep ``VISUAL_RETRIEVAL_ENABLED`` and dual-query disabled.

## 2026-09-24 — current gate state after page provenance v4

- [x] Fresh isolated MMLong/SPIQA frontier completed **21/21** with page
  recall **1.0** and retrieval P50 **127.97 ms** under v4 lifecycle.
- [ ] Gate C/E remains open: evidence phrase recall **0.7167**, figure
  evidence **0.625**, table evidence **0.7778**, artifact recall **0.30**.
  Diagnose canonical artifact generation/selection before changing ranking.
- [ ] Run the next stratified visual retrieval A/B only after artifact
  identity parity is measured; do not promote ColSmol/ColPali by name alone.
- [ ] Run authenticated Sol HTTP/SSE agent regression against the fresh v4
  revision after retrieval fixes; keep dual-query and visual flags off.

## 2026-09-24 — fresh ingestion-label rerun required

- [x] Moved sparse printed-page OCR to ingestion/cache and removed query-time
  OCR from ``RagService``. This preserves the generic mapping fix without
  putting OCR work on the interactive retrieval latency path.
- [x] Persisted trusted labels for textless pages as revision-scoped page
  provenance rows (excluded from FTS/Lance); fixed activation counts to count
  text chunks only. Isolated MMLong 0315 now resolves printed 13 -> physical
  14 with **1.0** page recall and **119 ms** retrieval.
- [ ] Re-index the isolated public frontier with the new parser fingerprint
  and page-label OCR budget, then rerun the 21-case retrieval-only report.
  The earlier page-label report is superseded for latency because it used
  query-time OCR; do not use its P95/max for Gate E or promotion.
- [ ] Compare candidate/retrieved/final page evidence and SPIQA support after
  the fresh run. Keep visual and dual-query flags disabled until the full
  stratified gate is measured.

## 2026-09-24 — explicit page candidate verification

- [x] Added active-revision page candidates for explicit page/slide hints,
  including image-only pages with no searchable text; added regression test.
- [x] Cached-index 21-case Sol rerun completed with zero failures. MMLong
  candidate evidence recall improved to ``0.9091``; final evidence recall is
  ``0.8636`` and SPIQA diagnostic token-F1 is ``0.6443``.
- [x] Full backend suite after the page-candidate fix: **965 passed** with 5
  dependency deprecation warnings.
- [x] Audited MMLong case 0315: printed page 13 is physical PDF page 14 and
  the public qrel ``[14]`` is coherent. Runtime selected physical page 13
  because the image-only page had no extracted printed label. The remaining
  issue was a generic page-label resolver, not a qrel conflict. Sparse Arabic
  label interpolation is now implemented and parser fingerprints were bumped;
  see
  ``data/retrieval_eval/internal/mmlong-0315-provenance-audit.json``.
- [ ] Re-index the isolated public frontier with the new parser fingerprint
  and rerun 0315/full stratified cases. Candidate recall remains below
  ``0.95`` until that measurement is complete; keep visual and dual-query
  flags disabled.
- [ ] The 0315 page-label fallback is ``166 ms`` after cache/engine warmup but
  ``9.30 s`` on the first OCR call. Measure cold/warm proportions over the
  full frontier and avoid promoting the fallback if P95 or answer variance
  regresses the gate.

## 2026-09-24 — corrected candidate-frontier measurement

- [x] Fixed evaluator semantics: candidate recall is now scored from the
  actual first-stage candidate frontier; retrieved-text and final attachment
  recall remain separate.
- [x] Cached-index 21-case rerun completed with zero failures. MMLong
  candidate evidence recall is ``0.8182`` and final evidence recall
  ``0.8636``; SPIQA diagnostic token-F1 is ``0.6281``.
- [ ] Candidate recall still fails the ``0.95`` gate. Before changing ranking,
  audit page-label/qrel semantics (notably printed page 13 vs physical page
  14 in case 0315) and image-only candidate provenance in case 0409.
- [ ] Keep answer-generation normalization separate from retrieval: several
  rows have complete evidence but Sol produces a different numeric/list form.
- [ ] Do not open shadow-v3 or promote visual/dual-query flags until these
  provenance and candidate-frontier failures are resolved and rerun.

## 2026-09-23 — post-fix public frontier audit

- [x] Completed the authoritative post-fix 21-case public frontier (11
  MMLongBench-Doc + 10 SPIQA) with zero failed rows and Sol-only generation.
- [x] Persisted separate candidate text-page and final selected-evidence page
  recall fields. MMLong candidate recall is ``0.7727`` and final evidence
  recall is ``0.8636``; SPIQA diagnostic token-F1 is ``0.6535``.
- [ ] Candidate recall still fails the production gate (``0.95``), and visual
  retrieval source rate is zero while ``VISUAL_RETRIEVAL_ENABLED=false``.
  Do not promote based on final attachment recall alone.
- [ ] Remaining public rows must be classified into candidate/document/FTS/
  vector/artifact/attachment versus answer-normalization failures. In
  particular, do not “fix” retrieval for rows whose evidence is present but
  the answer scorer rejects list or numeric formatting.
- [ ] Keep shadow-v3 sealed until the retrieval and agent RC gates are frozen;
  the completed report above is diagnostic evidence, not a release claim.

## 2026-09-23 — post-OCR/conditional-multihop verification

- [x] Confirmed authenticated 9router completion with ``cx/gpt-5.6-sol``;
  the previous 401 was caused by a missing manual Authorization header, not
  an offline gateway.  Ollama is still used only for local EmbeddingGemma.
- [x] Added generic bounded local OCR for image-only visual artifacts lacking
  page text/caption, plus independent conditional multi-hop obligations.
  Fresh case 0413 now returns ``45%`` with official score ``1.0`` and page
  evidence recall ``1.0``.
- [x] Re-ran verification after the patch: **964 backend tests passed**;
  compile and diff checks are clean.
- [x] Re-ran the isolated arbitrary-format lifecycle matrix with live Ollama:
  six PDF/DOCX/MD/TXT retrieval probes hit; dedup, unchanged-rescan,
  revision-on-change and rename/move invariants pass.  The intentionally
  corrupt PDF remains classified ``partial``.  See
  ``data/retrieval_eval/internal/format-lifecycle-matrix-postocr-20260923.json``.
- [x] Fresh authenticated HTTP/SSE smoke through the backend completed the
  ``file_qa`` ASPIRE query with streamed ``message.delta`` events, validated
  evidence and no fallback/transport error.
- [x] Evaluator now records candidate text-page recall separately from final
  multimodal evidence-page recall (retrieved pages union the actually selected
  image attachments).  It never uses gold pages to select those attachments.
- [ ] Rerun the 21-case frontier once after this metric-schema change so the
  report contains the new final-page-recall field; the post-hoc ``0.8182`` is
  diagnostic only and is not a replacement report.
- [x] Completed the post-OCR public frontier: **21/21** rows, zero failures;
  MMLong official accuracy ``0.9091`` (11 cases), SPIQA diagnostic token-F1
  ``0.6352`` (10 cases).  MMLong evidence-page recall is ``0.7273`` and
  therefore still fails the ``0.85`` gate; median local retrieval is ~117 ms
  but PDF preparation is ~157 s median.  Keep visual retrieval and dual-query
  disabled.  Report:
  ``data/retrieval_eval/public/results/gate-e-postocr-21-20260923.json``.
- [ ] Close the remaining page-evidence gap without changing the scorer to
  count a gold-derived attachment: make requested page/artifact records enter
  the canonical retrieved evidence set, and separately report candidate text
  recall versus final multimodal evidence recall.  Current MMLong lower rows
  include correct image attachments that the text-only page metric excludes.
- [x] Added a generic page-superlative document-coverage policy (slowest,
  fastest, highest/lowest, and Vietnamese equivalents) with regression tests;
  no paper/case mapping is used.
- [x] Isolated Sol replay of MMLong 0537 passed with
  ``coverage_reason=page_superlative`` and official score ``1.0``; candidate
  and final page recall were both ``1.0``.
- [x] Comparison 0494 replay now keeps both operands and answers ``5%`` with
  score/page recall ``1.0``; deterministic creator/view aliases remove the
  unnecessary planner round-trip (retrieval ~301 ms vs ~6.33 s).  Ratio plans
  still retain the Sol planner path.
- [ ] Rerun the complete 24-case public shadow after the OCR change.  The
  isolated 0413 result is a regression proof, not a full arbitrary-format
  promotion gate.

## 2026-09-23 — printed-page provenance follow-up

- [x] 9router health is currently green for ``cx/gpt-5.6-sol``; Ollama is
  separately used only for local EmbeddingGemma vectors. The earlier offline
  note was stale and must not be used to explain current benchmark results.
- [x] Facing-page labels (``14 15``/``14-15``) now resolve to physical pages
  through a shared parser helper and survive FTS, vector, coverage and visual
  attachment projections. A fresh isolated replay fixed MMLong case 0336
  (answer 3 → 1, attachment physical page 9).
- [ ] Rerun the full 24-case public frontier after this provenance fix and
  remeasure all strata. The three-case replay is only a regression check;
  it does not satisfy the arbitrary-format gate.

## 2026-09-23 — latest verification checkpoint

- [x] Re-ran the isolated PDF/DOCX/MD/TXT lifecycle matrix with Ollama
  EmbeddingGemma: all expected retrievals and revision/dedup/rename/content
  invariants pass; corrupt input remains explicitly partial.
- [x] Pinned public dataset validation and sealed shadow-v3 structural
  validation pass. Shadow-v3 is still unopened until the RC fingerprints are
  frozen.
- [x] The previous backend/SSE smoke through 9router Sol passed with zero
  fallback or transport error. A fresh check in the current shell currently
  sees 20128/11434 unavailable, so restart those sidecars before any new
  model-backed benchmark; no model fallback is allowed.
- [ ] Production Lance still reports the legacy projection; the physically
  revision-scoped migration is validated only in the isolated runtime. Do not
  switch production or open shadow-v3 until the RC/rollback record is frozen.
- [x] Large-PDF ingestion now has a bounded fallback: sources over 120 pages
  or 40 MB avoid Docling's full picture/table conversion and retain honest
  native-text/image/page provenance. A 170-page public manual parses in about
  2.15 s instead of about 114 s; table extraction remains partial unless a
  later enrichment stage is explicitly run.
- [ ] The first shadow-v3 run was intentionally stopped at a 5/24 isolated
  checkpoint after a large PDF exceeded practical parser wait time. The
  checkpoint is retained, but it is not a valid score. A configurable bounded
  OCR page budget is now in place; rerun the sealed shadow only after the
  sidecars are back and the new parser policy is verified.
- [x] The next bounded shadow reached 23/24 but exposed a real generic
  ``document_figures.id`` collision from repeated caption labels. Figure IDs
  now include the revision-scoped parser ordinal; isolated replay of the
  failing 77-page source succeeds. The old partial report is invalid and must
  be rerun, not scored.
- [x] Fresh rerun is now complete at 24/24 with zero failed rows, but the
  arbitrary-format quality gate still fails (MMLong official accuracy
  ``0.3258``, evidence-page recall ``0.6818``; SPIQA diagnostic token-F1
  ``0.5016``). Keep visual and dual-query flags off.
- [ ] Improve retrieval/answer quality by failure stratum (especially
  page/table/visual evidence) using held-out documents; do not tune prompts or
  locators to individual shadow cases. Visual retrieval remains an isolated
  experiment because its source rate is currently zero.
- [x] Generic observable-visual routing is covered by query tests and a fresh
  24-case Sol replay; MMLong improved to ``0.4056`` without changing the
  index/embedding. Keep the feature flag off because the gate still fails.
- [x] Agent HTTP/SSE dev-20 remains green after the routing change (20/20,
  zero fallback/transport errors). Latency remains variable (median total
  about 16.7 s, max first delta about 50.6 s) and should be optimized after
  retrieval/visual acceptance, not hidden by a fallback model.

## 2026-09-23 — answer completeness and ColSmol frontier result

- [x] Generic list/free-form answer instructions now preserve atomic entities
  and supported numeric ranges; four-case Sol replay improved SPIQA
  external-source diagnostic F1 to ``0.70``.
- [x] Attempted the existing 21-case visual retrieval frontier without LLM
  calls. ColSmol timed out on case 2; the report is correctly non-promotable
  (1 valid case, worker unhealthy). Keep ``VISUAL_RETRIEVAL_ENABLED=false``.
- [ ] The strict MMLong list scorer still penalizes semantically correct
  grouping/splitting (e.g. slide-stage answers); this is an evaluator
  limitation to document, not a paper-specific answer patch.
- [ ] Gate E remains open. Need a stable multi-case visual provider run and a
  broader answer-quality audit before any RC/shadow promotion. Do not turn on
  visual or dual-query flags based on the one-case/diagnostic frontier.

## 2026-09-23 — Gate E current paired report

- [x] Replayed 21 public cases (63 Sol generations) with the new generic
  answer contract: Aya ``0.734988`` vs full ``0.488157`` and gold ``0.724743``;
  MMLong official ``9/11``; no transport or model fallback. Report:
  ``gate-e-answer-quality-current-21-20260923-prompt.json``.
- [x] The explicit-slide and mixed-scan cases now attach query-selected page
  pixels even when ColSmol times out; no qrel/page-number hardcoding was used.
- [ ] Gate E is not a production acceptance gate yet: strict benchmark
  formatting/annotation misses remain, visual provider cold-index timeouts
  remain, and SPIQA metrics are diagnostic. Keep both visual and dual-query
  flags off until a stable visual frontier and RC/shadow checks pass.

## 2026-09-23 — Lance migration remains staged

- [x] Isolated migration from current Lance to physical revision-scoped tables
  validated all 21 active documents (21 cards, 1055 text, 60 tables, 59
  figures; zero stale/unattributed rows). Isolated status and dimensions pass.
- [x] Desktop production build passed; backend suite is 953/953.
- [ ] Do not switch the isolated Lance path into production yet. Gate E and
  shadow-v3 are still open; the production legacy projection is a known,
  explicitly reported blocker rather than silently claimed as fixed.

## 2026-09-23 — visual scope follow-up after live Sol replay

- [x] Plural visual wording now activates the shared visual intent detector;
  explicit page/slide hints are attached even when optional ColSmol retrieval
  is unavailable. This is a generic query/projection contract, not a case
  mapping.
- [x] Isolated 0409/0315 replay through 9router Sol: requested pages were
  attached; 0315 returned ``2`` (official score 1.0). 0409 received pages
  12/14 and produced the semantically correct grouped stage list, although
  the strict benchmark list scorer still marks it 0.0.
- [x] Targeted tests: 78/78. Full backend suite should be rerun before the
  next RC fingerprint.
- [ ] Gate E remains open. The full 21-case paired report is diagnostic, not
  a promotion decision; remaining work is generic answer/evidence quality
  (not paper-specific patching), plus a cold visual-provider latency policy.
  Keep visual and dual-query flags off and do not open shadow-v3 yet.

## 2026-09-23 — visual readiness contract follow-up

- [x] Propagated active document readiness into the isolated evaluator. Partial
  OCR scans with no native pages now enter visual recovery even when OCR has
  produced chunks; ready native documents remain text/table-first.
- [x] Selected image-backed figure/table evidence and readiness-selected page
  renders now satisfy the pixel guard. Generic MMLong replays: `0569` returns
  `14%`; `0845` returns `5` with page recall `1.0` and no ColSmol dependency.
- [x] Tests: **952/952** backend; Graphify synchronized; no production index
  or feature flag changed.
- [ ] Gate E is still not promotable: the full paired 21-case public report
  must be rerun with the current evaluator, and SPIQA free-form completeness
  plus ColSmol cold-index latency remain diagnostic. Keep visual and
  dual-query flags off until a complete paired run and RC/shadow checks pass.

## 2026-09-23 — clean Sol conversational regression

- [x] Re-ran the complete conversational dev-20 through HTTP/SSE with live
  `cx/gpt-5.6-sol`: **20/20 passed**, `transport_error=null`, zero fallback,
  routes `19 file_qa + 1 chat`. Raw report:
  `data/retrieval_eval/internal/agent-conversational-dev-sol-current-20260923.json`.
- [x] The prior `19/20` result was a transient upstream stream failure and is
  superseded for the current code by this clean rerun; keep the old report for
  historical audit rather than deleting it.
- [ ] Gate E remains open: public visual answer-quality rows still regress on
  the bounded diagnostic suite. Keep `VISUAL_RETRIEVAL_ENABLED` and
  `DUAL_QUERY_RRF_ENABLED` off; do not open shadow-v3 before RC fingerprints
  and public/agent release gates are frozen.

## 2026-09-23 — current Sol/agent gate and remaining release blockers

- [x] Rechecked 9router completion authorization through backend/SSE with
  `cx/gpt-5.6-sol`; approved model catalog, completion and streaming all pass.
- [x] Added generic identity-bound multi-document comparison prompting,
  deterministic bounded temperature for strict comparisons, and safe partial
  evidence preservation with document-bound insufficiency rows.
- [x] Full backend verification after the change: **949/949**; Graphify is
  synchronized.
- [x] ASPIRE/ViSEC and LPMN/Wav2Small comparison groups replayed **12/12**;
  retrieval covered every required document and no fallback was used.
- [ ] Full conversational dev-20 is not yet a clean aggregate: latest run is
  **19/20** because one `dev05-t1` stream ended with `upstream connection
  lost`; the isolated dev05 replay is **4/4**. Repeat one full run only when a
  stable gateway window is available; do not count a transport failure as an
  answer-quality miss.
- [ ] Gate E remains open: the complete 21-case public Sol report at
  `gate-e-answer-quality-current-21-20260922.json` is diagnostic and
  non-promotable due to per-case visual answer regressions. Its overall Aya
  diagnostic score is `0.5914` versus full extracted `0.4416`; SPIQA scores
  are diagnostic rather than official. Keep visual and dual-query flags off.
- [ ] Do not open shadow-v3 or switch production Lance/projections until the
  Release Candidate fingerprints are frozen and the public visual/agent gates
  pass. No production index or database was mutated in this pass.

## 2026-09-16 — folder lifecycle activation contract

- [x] Folder ingestion now passes `defer_activation` explicitly to the
  per-file indexer; activation remains coordinator-owned after FTS/Lance
  validation.
- [ ] Keep the production Lance migration pending until isolated projection,
  rollback and shadow fingerprints are frozen.

## 2026-09-21 — lifecycle/public corpus verification

- [x] Host-side isolated PDF/DOCX/MD/TXT lifecycle matrix passes all expected
  local retrieval and revision invariants; corrupt input remains `partial`.
- [x] Restored two tracked public MMLong PDFs removed from the local clone;
  loader/sampling tests now pass without silently dropping cases.
- [ ] Gate-E still requires a valid 9router completion authorization for the
  paired Sol answer-quality run; visual and dual-query flags remain off.

## 2026-09-16 — stratified public retrieval checkpoint

- [ ] Resume `public-retrieval-only-r30.json.partial` when a long isolated
  indexing run is acceptable; the interrupted checkpoint is not a score.
- [x] Keep the partial run out of promotion reports: one image-only case
  completed, but the 30-case aggregate remains `score_valid=false`.

## 2026-09-16 — evidence-card revision isolation

- [x] Card candidate generation now filters every canonical source by the
  document's active revision; staged/retired rows are excluded and covered by
  regression tests.
- [ ] Re-run card enrichment only after valid 9router authorization and an
  approved isolated corpus policy; this fix does not create cards by itself.

## 2026-09-16 — retrieval-cache revision isolation

- [x] Cache fingerprints now bind to the active revision and ignore staged or
  retired derived rows; revision-switch invalidation is regression-tested.

## 2026-09-16 — LightRAG provenance isolation

- [x] Parent resolver rejects null/retired provenance for revisioned
  documents; exact active-revision matching is covered by regression tests.

## 2026-09-16 — post-cutoff artifact preservation

- [x] Generic enumerated/list questions reserve a canonical table at the
  final visible cutoff when that artifact is already retrieved.
- [x] Text/table/figure vector channels are ranked independently before RRF;
  regression tests cover sparse artifact channels.
- [ ] Re-run the larger stratified public retrieval suite to measure aggregate
  impact; the single SPIQA replay is diagnostic, not a Gate-E promotion.
- [ ] Sol answer-quality replay still waits for a valid 9router completion
  authorization; health reachability alone is not proof of OAuth validity.

## 2026-09-16 — public 21-case diagnostic

- [x] Current retrieval-only frontier completed 21/21 in isolation after the
  generic list/table fix.
- [ ] Gate-E remains closed: artifact recall is `0.30` on this mixed frontier;
  run paired visual retrieval and answer-quality evaluation before promotion.
- [ ] 9router Sol completion probe still fails with `401 invalidated oauth
  token`; reauthorize before any paper-content E2E run.

## 2026-09-16 — numbered artifact routing fix

- [x] Explicit `Figure N`/`Table N` queries recover the owning active
  document before vector routing, and context composition reserves that exact
  label instead of the highest-scoring similar artifact.
- [x] Isolated MMLong Figure 89 replay selects page 118 through the generic
  path; no paper/case mapping or production mutation was used.
- [x] Full backend suite: **923/923**.
- [ ] Re-run the complete public stratified suite after OAuth/local runtime
  preconditions are available; the single-case replay is diagnostic only.

## 2026-09-16 — public replay after routing fix

- [x] Cached 9-case no-LLM replay completed 9/9; page recall rose to `1.0`
  (`0.6667` before the fix), with retrieval P50/max `145/236 ms`.
- [ ] Evidence/artifact recall is still `0.6389`/`0.3333` on this mixed
  sample; inspect modality/annotation alignment and run a larger stratified
  set before any Gate-E conclusion.

## 2026-09-16 — 9router token checkpoint

- [x] Gateway process was reachable and advertised the approved Sol/Terra/Luna
  model IDs.
- [ ] Refresh/reauthorize the 9router OAuth token: the 3-case Sol smoke failed
  closed with `401 token_revoked` (HTTP 503 wrapper), so it produced no valid
  answer-quality aggregate. A retry after `/v1/models` became reachable
  reproduced the same error.
- [ ] After token refresh, rerun the same smoke before attempting the full
  public 21-case frontier; do not treat gateway `/v1/models` reachability as
  proof that completions are authorized.
- [x] Health/UI now distinguish catalog reachability from completion
  authorization; backend was restarted with the new contract and the full
  suite after the change is **921/921**.

## 2026-09-16 — OCR versus no-OCR public diagnostic

- [x] Same 9-case public sample with selective OCR completed 9/9 in isolation:
  page recall `0.6667` vs `0.3333` no-OCR; evidence/artifact recall unchanged
  (`0.6389`/`0.3333`), with P50/Pmax `181/264 ms`.
- [ ] Keep OCR selective and provenance-labelled (`native`/`ocr`/`visual`);
  native-text and table artifact misses remain open for general retrieval
  work. This sample is diagnostic and does not promote visual retrieval.

## 2026-09-14 — evaluator routing fairness

- [x] Removed gold artifact/source labels from public visual routing and
  pixel-guard decisions; added text-backed versus image-only regression tests.
- [x] Full backend suite: **914/914**.
- [ ] Re-run the 21-case paired Sol frontier after the 9router quota resets.
  The current attempt stopped at HTTP 429 and must not be interpreted as a
  score or promotion result.
- [ ] Keep Gate E, `VISUAL_RETRIEVAL_ENABLED` and `DUAL_QUERY_RRF_ENABLED`
  disabled until the complete replay and unseen shadow validation pass.

## 2026-09-16 — current runtime checkpoint

- [x] Restarted backend/frontend; frontend serves locally and backend health is
  degraded because the reachable 9router gateway rejects completion with
  `401 token_revoked`.
- [x] Gate C retrieval-only revalidation remains green for A0 thresholds;
  B-full adds no gain and B-anchor is slower.
- [x] Full backend suite after evaluator changes: **914/914**.
- [x] Lifecycle matrix replay passed all expected retrieval hits and
  duplicate/rescan/change/rename invariants; corrupt PDF remained `partial`.
- [ ] Resume the fair-routing 21-case Sol answer replay only after 9router is
  reachable; the current checkpoint has no valid answer-quality aggregate.

## 2026-09-16 — revision-aware Lance migration checkpoint

- [x] Read-only migration audit against `data/sqlite/app.db` produced a
  revision-aware isolated Lance projection: 21 active documents and 1,195
  vectors mapped with zero identity mismatches.
- [x] `rag_status.py` distinguishes SQLite's physical revision schema from
  Lance's legacy production projection; the latter remains migration-pending.
- [x] Isolated real-Lance revision assertions: **25 targeted tests passed**.
- [ ] Do not switch production Lance until the isolated projection is paired
  with retrieval/HTTP smoke and an approved rollback; no production mutation
  has occurred.
- [x] Gate-C retrieval-only parity against the isolated revision-aware Lance
  projection matched the current baseline (`A0/B-full=0.97/0.91/1.00`,
  `B-anchor=0.97/0.92/1.00`); this does not replace Sol E2E/shadow gates.
- [x] Migration utility now fails closed on a non-empty Lance source with no
  authoritative active revisions, avoiding silent wrong-DB audits.
- [x] Full backend suite after migration hardening: **918/918**.
- [x] Migration reads SQLite in `mode=ro` and fails on missing authority; it
  cannot silently initialize or mutate the production DB.

## 2026-09-16 — arbitrary-format public diagnostic

- [x] No-LLM/no-OCR stratified sample completed 9/9; its low page/table recall
  is recorded as a visual-channel limitation, not treated as a parser crash.
- [x] ColSmol local-only trial was stopped after measured high CPU cost and a
  1-case partial checkpoint; visual indexing remains background/optional.
- [ ] Run paired visual + Sol answer quality only after 9router is reachable;
  do not infer visual promotion from the 9-case text diagnostic.

## 2026-09-14 — metric-prose frontier replay

- [x] Added and tested a generic reservation for open-ended metric/evaluation
  prose; explicit table/figure requests retain artifact priority.
- [x] Replayed Gate C: A0/A1/B-full remain `0.97/0.91/1.00`; B-anchor is
  `0.97/0.92/1.00` but slower. Backend suite is **914/914**.
- [ ] Keep dual-query disabled: B-full has no recall gain and B-anchor's small
  gain does not justify its additional embedding/latency cost.

## 2026-09-14 — artifact recovery follow-up

- [x] Canonical table/figure lexical recovery is revision-scoped and covered
  by an active-versus-retired regression test.
- [ ] Re-run the paired public Gate-E frontier after this recovery path; do
  not attribute any score change until the same case manifest and answer gate
  complete.
- [ ] Keep `DUAL_QUERY_RRF_ENABLED=false` and
  `VISUAL_RETRIEVAL_ENABLED=false` until the existing Gate-C/E and unseen
  shadow requirements pass.
- [x] Replayed `dev-v2.2` after query-shape gating; A0 stayed at
  Candidate/Evidence/Document `0.97/0.91/1.00` with warm P50/P95 `147/368 ms`.
- [x] Re-ran the isolated PDF/DOCX/MD/TXT lifecycle matrix; all expected hits
  and revision/dedup/rename invariants passed, with corrupt PDF classified
  `partial`.
- [x] Replayed the paired Gate-E frontier after artifact gating; 63/63 Sol
  generations completed and the strict join remains non-promotable because
  SPIQA answer quality is slightly below full context with three per-case
  regressions.

## 2026-09-13 — generic visual query policy

- [x] Natural visual-modality terms that omit concrete artifact nouns are now
  recognized generically and require pixels when they ask about visual
  content/grounding; metadata-only inventory remains text-safe.
- [ ] Re-run the paired unseen Gate-E frontier to measure whether activation
  improves visual page selection; do not infer a promotion from intent tests.

## 2026-09-13 — Gate-C v2.2 result

- [x] Audited 60-case A0/A1/B-full/B-anchor replay completed. A0 already
  clears the retrieval and warm-latency thresholds; B-full is non-inferior but
  not better, while B-anchor's small gain costs latency.
- [ ] Keep dual-query off. Remaining retrieval work is targeted at the five
  A0 final-cutoff/fusion misses and at unseen shadow validation, not broad
  RRF tuning.

## 2026-09-13 — shadow-v3

- [x] Sealed a 24-case, 24-document answer-free manifest across six public
  format/modality buckets; no answer/qrel content is tracked and `opened=false`.
- [x] Generated ignored private qrels (24 rows, 23 positive/1 abstention) with
  source/content fingerprints for later preflight.
- [x] Manifest/qrel checksum and matrix validator passes without opening the
  shadow suite.
- [ ] Do not open or score it until the Release Candidate parser/index/runtime
  fingerprints are frozen; then run the one-time retrieval + Sol E2E gate.

## 2026-09-13 — lifecycle matrix checkpoint

- [x] Isolated durable folder matrix passed for native/mixed/table-image PDF,
  DOCX, Markdown and TXT, including duplicate bytes, unchanged rescan,
  content revision, rename identity and corrupt-source partial isolation.
- [ ] This bounded fixture matrix does not close arbitrary-format Gate D or
  visual Gate E; the stratified public corpus and paired answer-quality gate
  remain required.

## 2026-09-13 — current visual retrieval replay

- [x] Current-code 20-case visual retrieval replay completed all rows. T/V/T+V
  evidence recall is `0.500/0.833/1.000`, with no fused text regression.
- [ ] Gate E remains closed: only 18 rows are positive, answer quality is not
  attached, and T+V P95 is `1,026 ms`; run a paired Sol frontier with the
  contract's minimum positive count before any flag decision.

## 2026-09-13 — current frontier and context audit

- [x] Re-ran the complete 21-positive visual retrieval frontier: T/V/T+V
  evidence recall `0.4762/0.8571/1.0000`; all rows and required profiles pass,
  but answer-quality pairing is still required for promotion.
- [x] Paired Sol answer frontier completed **21/21 cases, 63/63 generations**;
  overall Aya-vs-full delta is positive, while three SPIQA per-case diagnostic
  regressions remain visible. No fallback or production write occurred.
- [x] Added bounded artifact-context projection and replayed its three failure
  classes (9/9 complete). The change is retained for correctness/context, not
  claimed as a score fix.
- [ ] Keep both visual and dual-query flags disabled; Gate-E still requires a
  policy-approved answer-quality contract with no per-case regression and then
  unseen shadow/RC validation.

## Gate-E after multi-page reservation fix — 2026-09-03

- [x] Explicit multi-page visual evidence is now reserved generically in all
  live composition paths and the public evaluator; no paper/case mapping was
  added. Backend suite: **873 passed**.
- [x] One-case Sol smoke confirms MMLong `0409` keeps both requested pages and
  two image attachments (`evidence_page_recall=1.0`). The strict official list
  score remains a separate answer-format diagnostic.
- [ ] Re-run the complete paired 20-case Gate-E frontier after this fix. The
  previous complete report remains diagnostic: T+V recall `0.8889`, visual
  quality not promotable, and answer-quality regressions remain.
- [ ] Keep `VISUAL_RETRIEVAL_ENABLED=false` until a stratified 20+ positive
  visual run and paired Sol answer-quality report pass the no-regression gate.

## Gate-E replay result — 2026-09-03

- [x] Exact 20-case/60-generation Sol replay completed after multi-page
  reservation; no failed rows or model fallback. Report:
  `gate-e-answer-quality-r25-page-reservation-20case.json`.
- [x] Strict join generated:
  `gate-e-combined-r25-page-reservation.json`.
- [ ] Gate-E is still open: MMLong Aya `0.4951` vs full `0.5000`, T+V visual
  recall `0.8889` on 18 positive rows, and the suite/profile minimum is not
  met. Next work must address generic native-text/table/multi-page retrieval
  and answer-quality no-regression, not tune the 0409 case.

## Gate-E paired 20-case run — 2026-09-03

- [x] Joined the exact 20-case answer-quality and visual retrieval frontiers
  with a strict manifest; no LLM or production write occurs during the join.
  Report: `data/retrieval_eval/public/results/gate-e-combined-r21.json`.
- [ ] Gate-E remains closed: visual T+V recall is `0.8333`, only 18 rows are
  positive, and two SPIQA table answer rows regress. Fix the generic table/
  artifact retrieval path and rerun the same manifest before considering any
  visual promotion.
- [ ] Build unseen `shadow-v3` only after the corrected retrieval/answer gate;
  do not reuse this dev frontier as a release claim.

## Stack status observability — 2026-09-03

- [x] `scripts/rag_status.py` now makes provider/model split, flags,
  readiness, and legacy-versus-active revision counts explicit. Production
  SQLite currently remains legacy by policy; no migration was performed.

## Latest verification — 2026-09-03 (Gate-E cache-aware visual smoke)

- [x] Re-ran the bounded visual retrieval comparator from the existing
  isolated cache: 7/7 cases completed (`T=0.9286`, `V=1.0000`, `T+V=1.0000`)
  across MMLong native/mixed/image-only PDFs and SPIQA figure/table cases.
  This used local Ollama EmbeddingGemma + the isolated ColSmol worker only;
  no 9router calls or production writes occurred.
- [ ] Gate-E is not closed by this smoke: the minimum 20 positive cases and
  paired Sol answer-quality report are still required. 9router is currently
  unreachable, so the pending 20-case answer run remains provider-unavailable.
- [x] Document status now reports parser `readiness_state`/details separately
  from local FTS/Lance readiness, so `partial` or `visual_only` documents are
  visible to clients even when their local projection is healthy.
- [x] Full backend suite after the status-contract change: **864 passed**;
  desktop production build passed.

## Latest verification — 2026-09-03 (visual readiness guard)

- [x] Image-only/readiness-required evaluator turns now fail closed when page
  pixels are absent, even if the question is phrased as a normal fact query;
  mixed-document explicit visual intent remains protected. Added regression;
  backend suite **862 passed**, desktop build passed.
- [ ] Re-run the larger post-fix Gate-E answer-quality sample after 9router is
  reachable. The attempted 20-case run was provider-unavailable before any
  row and is not a score.
- [x] Re-ran the four-format Gate-D smoke through local EmbeddingGemma:
  text formats are `ready`, the image-only PDF is `partial` with a valid
  projection and an explicit non-searchable contract; no raw-hit inflation.
- [x] Paper evidence cards now have revision-owned card/facet/item projections;
  active lookup cannot read a retired/staged card. Legacy tables remain a
  compatibility mirror. Revision-switch regression and backend suite
  **863 passed**.

## Latest verification — 2026-09-02

- [x] Non-coverage visual attachments are now scoped to ranked visual pages
  with stable page-number deduplication. Singular visual turns receive one
  page by default; comparison/all-page turns keep a bounded set. This fixes a
  generic adjacent-page contamination path without changing retrieval rank.
- [x] Live agent visual routing now consults active-revision readiness and can
  activate page retrieval for visual-only documents even on fact-shaped
  questions when the visual feature flag is enabled; production remains off
  pending Gate-E.
- [x] Isolated answer-quality path now infers a required visual channel for
  native image-only PDFs (`text_chunks == 0`) even when the question has no
  explicit visual wording. This fixes the generic modality-selection gap
  without using gold/qrel content; cached ColSmol + Sol replay of MMLong
  `0845` answered `5` with official score `1.0`.
- [ ] Run a larger post-fix Gate-E sample with cache-aware document reuse.
  The interrupted three-profile retrieval smoke is non-scoring because its
  sampler selected documents not present in the selected runtime cache.
- [x] Exact-value/date/metric questions now receive bounded same-page sibling
  context when the first text hit is only a heading or lacks numeric signal.
  The projection is query-shape driven, active-revision scoped and tested on a
  mixed-scan public form; the isolated Aya answer changed from `0.0` to `1.0`.
- [x] Visual pixel contract now recognises relational requests such as
  “values in the picture/chart” and Vietnamese equivalents; metadata-only
  caption/list requests remain eligible for canonical figure metadata.
- [x] Public multimodal prompts expose explicit page provenance for rendered
  page attachments; this is generic and does not alter retrieval scoring.
- [x] Durable folder processing defaults to PDF/DOCX/MD/TXT; legacy indexing
  endpoint defaults remain unchanged for compatibility.
- [x] Post-fix MMLong image-only smoke now passes the unanswerable case with
  canonical visual source and page attachment; warm ColSmol reuse is fast.
- [ ] Run a larger stratified post-fix Gate-E sample (MMLong profiles plus
  SPIQA table/figure) before any visual promotion; one passing case is not a
  production-quality claim.
- [ ] Re-run stratified Gate-E answer quality with the approved 9router model;
  the current endpoint is unavailable, so no post-fix score is valid.

## Latest verification — 2026-08-31

- [x] Fixed visual-page eviction in multimodal context composition. A
  successful ColSmol page hit now has an explicit reservation and is not
  hidden behind ordinary text/per-document caps; a regression test verifies
  the image-bearing page reaches the composed source list. Full backend suite:
  **834 passed**; visual retrieval remains isolated and feature-off.
- [x] Made that reservation modality-aware so visual over-retrieval cannot
  displace exact text in plain-text cases; added a benchmark-contract test for
  the distinction.
- [x] Wired the reservation into both production agent composition paths;
  visual retrieval remains disabled until Gate-E answer quality passes.
- [x] Added explicit visual-context and image-attachment telemetry to the
  public evaluator; the next rerun must inspect these fields before judging
  answer quality.
- [x] Extended the optional visual retrieval hook to ordinary visual-intent
  questions as well as page-aggregation queries; feature flag remains off.
- [ ] Gate E is still open. Re-run a stratified unseen answer-quality sample
  after this context fix, then inspect paired deltas before considering any
  visual promotion. Do not infer a pass from the current r18 report alone.

## Latest verification — 2026-08-30 (continued)

- [x] Completed the corrected public Gate-E r18 diagnostic (20 paired cases,
  60 generations, 12 MMLong + 8 SPIQA; all visual indexes complete). Aya
  reached MMLong official accuracy `0.5507` and SPIQA diagnostic token-F1
  `0.4863`; overall Aya-vs-full delta was `-0.0005`. The result is not a
  promotion: image-only/unanswerable and SPIQA strata still show quality
  gaps, and `VISUAL_RETRIEVAL_ENABLED` remains false.
- [x] Fixed SPIQA's schema-drift failure (`asset_type` is metadata-backed in
  the canonical DB), made visual-failed Aya rows retryable, and added SIGSEGV
  (`-11`) to the isolated worker's auto-device CPU fallback. The 8 previously
  failed SPIQA Aya rows were retried successfully. Full backend suite after
  these fixes: **830 passed**.
- [x] Reduced isolated ColSmol query overhead without changing scores:
  candidate embeddings are scored in one processor call with bounded internal
  tiling, and worker CPU parallelism is configurable through
  `COLSMOL_TORCH_THREADS`. The long r18 public run was resumed from its
  checkpoint with this optimization; Gate-E remains non-promotable pending
  completion and answer-quality aggregation.
- [x] Fixed canonical receipt/materialization fingerprint parity and drained
  the agent's process-backed visual provider cache on shutdown. Derived
  visual jobs now preserve `partial` and retry provider failures instead of
  claiming completion; DOCX parser-version drift is covered; full backend
  suite is **828 passed**.
- [x] Fixed visual-sufficiency context wiring so primary and local fallback
  generation receive the same canonical-artifact guard. This is a generic
  prompt safety contract, not a paper/case rule.
- [x] Added visual evidence-shape tests and corrected visual-page telemetry;
  full backend suite is **828 passed**. The no-canonical-visual path now also
  fails closed after generation, so OCR/table text cannot leak as visual proof.
- [ ] Gate E remains open: visual answer quality still needs a larger,
  stratified sample and unseen shadow/RC validation; no visual flag is
  promoted from the small Sol smoke.

## Latest E2E verification — 2026-08-29

- [x] Gate-C retrieval-only was rerun against the correct revision-aware
  isolated clone (`retrieval-v2.2-live-r16.json`): baseline A0/A1/B-full
  remains `0.97/0.91/1.00`, with no B-full activation; keep dual-query off.
- [x] Public Sol three-mode smoke infrastructure is live and checkpointed;
  the 2-case r16 result is intentionally non-promotable. It surfaced a real
  arbitrary-format gap: image-only/unanswerable MMLong missed abstention after
  a long OCR/index phase, while SPIQA extracted-text retrieval matched full
  context. Next Gate-E work must be stratified and answer-quality-aware.
- [x] OCR-on/OCR-off visual diagnostics are now separated by a configuration
  fingerprint; do not resume a checkpoint created with the other setting.
- [x] The paired r16/r17 smoke measured the trade-off explicitly: OCR is a
  local ingestion cost, while image-only visual QA remains incomplete when
  ColSmol/page retrieval is disabled. Gate-E still requires a larger
  stratified visual + answer-quality sample.

- [x] Public three-mode answer-quality output now includes a paired,
  case-complete Aya-vs-full-vs-gold diagnostic with suite/stratum summaries.
  Missing or failed modes remain visible as partial; no result is treated as a
  promotion decision. Final reports are atomically written; full backend suite
  after this change is **814 passed**.
- [x] Visual retrieval-only reports explicitly distinguish an isolated trial
  from production eligibility; a visual score alone cannot flip the feature
  flag.
- [x] Gate-C v2.2 proposal passes the deterministic 60-case static audit with
  no structural or scope issues; semantic review is still intentionally
  provider-gated and the proposal is not publishable yet.
- [x] Semantic review runner now resumes successful batches from an atomic
  fingerprinted checkpoint and records provider-unavailable status without
  fallback. A fresh `--no-resume` run clears stale rows before preflight; the
  60-case review still awaits a reachable Sol endpoint.
- [x] Isolated Sol HTTP/SSE regression passed 7/7 turns with 14 persisted
  messages, valid evidence and HTTP 200 figure bytes. The first run exposed
  and the subsequent run fixed legacy absolute artifact paths in isolated
  roots; resolver tests cover safe remapping and rejection outside the root.
- [x] Unicode mathematical metric headers are normalized before evidence
  validation, preventing a valid `CCC𝑉` column from causing a quantitative
  fallback. Full backend suite is **807 passed**.
- [x] SPIQA visual case selection now uses a deterministic, answer-blind
  document frontier with a per-document reuse cap. The 20-case probe produced
  a resumable 3/20 checkpoint before a slow CPU page-indexing step was
  interrupted; it remains non-promotable and has no aggregate score.
- [x] Resuming that checkpoint completed 20/20 SPIQA rows (`T=0.50`,
  `V=0.90`, `T+V=0.95`). The promotion contract was hardened after this run:
  SPIQA-only strata cannot pass arbitrary-format Gate E without MMLong
  native/mixed/image-only PDF coverage; the report at
  `data/retrieval_eval/public/results/visual-gate-spiqa-r20-reuse.json` is now
  explicitly false.
- [x] Public evaluator resume resolves legacy documents by raw content hash and
  source alias before path lookup, preventing false missing-document failures
  and duplicate isolated projections.
- [x] MMLong V2 visual probe r2 completed six documents across all three PDF
  profiles (`T=0.75`, `V=1.00`, `T+V=1.00`). The result is still small and
  diagnostic; it does not by itself establish arbitrary-format quality.
- [x] Independent visual reports can now be merged only under a strict common
  contract. The 26-case SPIQA+MMLong aggregation passes the current isolated
  visual checks, but no production flag was changed and no answer-quality or
  shadow-v3 gate has passed.
- [x] Visual reports now separate text metric scope from visual artifact scope;
  the corrected public aggregate is `MMLong T=0.75`, `SPIQA T=0.5167`,
  `V=0.9231`, `T+V=0.9936`. The generic taxonomy records nine text
  no-candidate cases and leaves the text retrieval/arbitrary-Markdown quality
  issue open.
- [x] The MMLong v2 second-profile probe remains checkpointed/non-promotable;
  one new image-only row completed before the old path-only process was
  interrupted. Resume it with the new content-hash resolver when running the
  full arbitrary-format Gate E sample.
- [ ] Gate E public visual promotion and shadow-v3 remain open; ColSmol stays
  isolated/disabled until a sufficiently large stratified sample and answer
  quality gate are measured.
- [x] Three-mode public answer-quality runner now has atomic checkpoint/resume
  with model/context/selection fingerprinting; use it for the next Sol run so
  long ColSmol/9router jobs can resume without mixing configurations.
- [x] The same runner now samples MMLong by PDF readability profile and bounds
  per-document reuse, so the pending answer-quality run is stratified rather
  than accidentally native-text-heavy.
- [x] Answer-quality runs fail fast on unavailable local EmbeddingGemma before
  spending 9router calls for an Aya arm, and record the provider failure
  explicitly without fallback.

## Durable retry / evaluator parity — 2026-08-28

- [x] Completed live Gate-C r15 with the revision-aware isolated index:
  A0/A1/B-full `0.97/0.91/1.00`, B-anchor `0.98/0.93/1.00`; B-full activation
  `0/60`, therefore `DUAL_QUERY_RRF_ENABLED` remains false. Report:
  `data/retrieval_eval/internal/retrieval-v2.2-live-r15.json`.
- [x] Completed Gate-D lifecycle matrix r15 for native/mixed/table-image PDF,
  DOCX, MD and TXT with revision/dedup/rename checks; deliberate corrupt input
  is reported partial rather than hidden.
- [x] Measured the optional Gate-E cost on public data: CPU ColSmol took about
  8.5 minutes for one 53-page image-only PDF. The 20-case attempt was stopped
  before aggregation and left a checkpoint; this is a resource diagnostic, not
  a visual score or a reason to enable the flag.
- [x] A document-reuse r2 probe completed seven checkpoint rows across
  image-only and mixed profiles, reusing each document index for multiple
  questions, before native-PDF preparation became too slow. The partial file
  remains auditable and is not aggregated or used for promotion.
- [x] Added document-reuse sampling to the visual runner. It keeps profile and
  case IDs deterministic and gold-independent, so a future larger diagnostic
  does not encode duplicate PDFs unnecessarily; it still cannot promote an
  incomplete visual frontier.
- [x] Fixed public evaluator job-to-document resolution to consume the
  coordinator's validated projection receipt first; path lookup is now only a
  legacy fallback and is covered by the public evaluator contract tests.
- [x] Public retrieval-only resume no longer treats failed rows as completed;
  retries replace the old row and preserve one-row-per-case accounting.
- [x] Public benchmark runners now fail fast on an unavailable local
  EmbeddingGemma provider, avoiding misleading per-case failures after cold
  parsing and preserving the no-fallback contract.
- [x] Preflight response shape is covered by a regression test; full backend
  suite is **793 passed**.
- [x] SPIQA five-case diagnostic now separates OCR phrase quality from
  canonical artifact selection (`0.60` phrase / `0.70` artifact / `0.80`
  support); broader stratified public and visual samples are still required.
- [x] A staged revision left by an Ollama/Lance outage is returned by an
  unchanged rescan and can be projected/activated on retry.
- [x] Public multimodal evaluation uses an isolated durable folder coordinator
  and revision-keyed vector cache; it no longer bypasses the production
  staged-to-activation contract.
- [x] Physical Lance revision counts are checked after projection when the
  current table schema supports them; legacy tables remain fail-safe during
  migration.
- [x] The provider-outage verification was superseded by the completed r15
  Gate-C run; unit tests remain correctness evidence, not retrieval-quality
  evidence.

## Current gate interpretation — 2026-08-28

- [x] Revalidated the generated format smoke with a readiness-aware contract:
  native/OCR-searchable TXT/MD/DOCX require an actual text hit; an image-only
  sparse PDF may pass only as `partial`/`visual_only`/`unresolved` with a valid
  projection. The raw miss remains visible in the report and is never labelled
  `ready`. Report: `data/retrieval_eval/internal/format-lifecycle-smoke-r17.json`.
- Gate D is consequently a lifecycle/readiness gate, not a claim that OCR can
  recover arbitrary pixels. Gate E still requires a larger visual and paired
  answer-quality evaluation before any visual flag or arbitrary-format claim.
- [x] Hardened Gate-E promotion semantics: retrieval-only T/V/T+V reports now
  remain non-promotable until a complete paired answer-quality report provides
  an explicit `promotion_safe=true`; no score is inferred from a visual smoke.

- [x] Fixed a false-positive evaluator recommendation. `promotion_eligible`
  now reflects the measured comparison decision instead of being unconditionally
  true for an audited suite. The latest r14 report records zero B-full
  activations and keeps dual-query disabled.
- Gate C retrieval on the audited local 21-paper clone is green for A0/A1/B-full
  (`0.97/0.91/1.00`), but this is not a claim that arbitrary PDFs or visual
  reasoning are production-ready.
- Gate D is a format/lifecycle gate, not a new retrieval algorithm: it checks
  revision safety, dedup/rescan/rename behavior and PDF/DOCX/MD/TXT readiness.
  The isolated matrix passes valid fixtures and contains corrupt input as
  partial.
- Gate E is a visual-value/answer-quality gate. It remains open because the
  widened MMLong smoke improved page recall and the page-census answer to
  `8/13`, but the sample is too small and the aggregation answer is still
  incomplete. No ColSmol/visual flag is promoted.
- [x] Visual worker `auto` no longer assumes MPS is safe. It defaults to CPU;
  explicit MPS is opt-in and an abort before NDJSON response retries once on
  CPU with the same model. This removes a worker-crash ambiguity but does not
  close Gate E.
- [x] New ColSmol pages are encoded in bounded batches; this removes the
  one-page-at-a-time dispatch bottleneck while keeping the visual index
  revision-scoped and resumable.
- [x] Batch path passed an isolated four-artifact smoke, but Gate E still needs
  a larger stratified sample and answer-quality aggregation; no visual flag is
  promoted based on the smoke.
- [x] Aggregation completeness is now source-aware: expected active-document
  page counts are propagated into coverage evidence and map/reduce diagnostics;
  partial frontiers remain explicitly incomplete.
- [x] Active documents up to the bounded 64-page aggregation frontier now use
  an exhaustive text-page projection before optional visual fusion; longer
  documents remain bounded, ranked and explicitly partial.
- [x] Live page map/reduce now derives the same bounded frontier (32 by
  default, up to 64 when indexed page counts justify it), so a medium document
  is not truncated back to 32 pages after projection. Backend suite: **789
  passed**.
- [x] Gate-C CLI no longer defaults to the confounded v2.1 diagnostic suite;
  audited cross-balanced v2.2 is now the safe default.
- [x] Re-ran audited `dev-v2.2` Gate-C with Ollama online for the whole
  process; see r15 report above.
- [ ] Run the required stratified public visual/map-reduce answer sample (at
  least 20 positive cases, multiple native/mixed/image-only and chart/table
  strata), then perform Sol E2E and unseen shadow/RC checks. Do not tune to
  the single 0035 census case.

## RAG Foundation follow-up (2026-08-25)

## Current arbitrary-format/visual follow-up (2026-08-27)

- [x] Generic page-coverage policy and bounded page projection are implemented;
  normal fact/table queries remain on the compact path.
- [x] Shared visual/text page merge now runs before context composition; visual
  page data can reach the approved 9router multimodal request only when the
  visual feature flag is enabled.
- [x] Added generic bounded page map/reduce contracts with strict page identity
  checks and partial-batch diagnostics. It is opt-in for visual coverage and
  does not alter canonical evidence or ranking.
- [ ] Re-run the corrected MMLong V2/SPIQA retrieval and Sol agent smoke after
  9router and Ollama are online. The latest isolated visual replay was allowed
  to run, but the answer-quality result still failed the page-census case;
  no fallback was used.
- [ ] Finish visual Gate E with a stratified sample large enough to separate
  native/mixed/image-only PDFs and chart/table aggregation. The isolated
  ColSmol CPU triage recovered 11/13 pages on one chart case; the widened
  public smoke reached 0.692 evidence-page recall and answer 8/13. Neither is
  a promotion gate result or justification for enabling the feature.
- [ ] Run a public stratified map/reduce answer-quality sample (at least 20
  positive cases) and compare one-shot versus verified page aggregation by
  query shape, document length and visual/text availability.
- [ ] Validate production-enabled visual indexing/retrieval against active
  revision manifests before any staged flag promotion; keep the default off.
- [x] Prevent a cold public scan from silently becoming an answer/retrieval
  result: the visual runner now supports `--disable-pdf-ocr` for visual-only
  diagnostics and records no report when an interrupted preparation has no
  completed checkpoint. The default production OCR path remains enabled.

- [x] Packet A additive identity/revision schema, compatibility backfill,
  source aliases, active-revision filtering and FTS v2 scaffolding.
- [x] Packet B durable folder coordinator, job status API, leases/retries and
  startup recovery; Files UI intentionally remains out of scope.
- [x] Packet D DOCX ordered tables/images and selective PDF OCR provenance.
- [x] Packet E visual retrieval interface is isolated and feature-off; no
  visual model has been promoted.
- [x] Run the isolated local projection path with Ollama
  `embeddinggemma:300m` online where required by the public evaluator; the
  visual worker remains dependency-isolated and does not alter production.
- [ ] Finish Packet C audit/publish of `dev-v2.1`, rerun A0/A1/B-full/B-anchor
  with production-faithful two-stage scoring, and keep dual-query off until
  every gate passes.
- [ ] Run stratified arbitrary-format evaluation (native/mixed/image-only PDF,
  table-image, chart/diagram, DOCX, MD/TXT), then visual A/B and Packet F RC
  freeze/shadow/rollback checks.

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
4. Conversational 60-case v1 đã mở và tune, nên chỉ còn là regression suite,
   không phải held-out. Nếu cần blind quality estimate, phải dựng v2 từ unseen
   documents/public samples, split theo document/family trước khi xem label.
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

## Evaluation Gate v2 — updated 2026-08-23

- [x] Implemented retrieval-only A0/A1/B runner with frozen rewrite inputs,
  source/content-fingerprint qrels, canonical locator preflight, modality and
  latency metrics, embedding-call accounting and fail-closed promotion checks.
- [x] Curated visible `dev-v2` (60 balanced cases) and sealed MMLongBench-derived
  `shadow-v2` (24 questions + private qrels); dev preflight resolves 60/60.
- [x] Corrected feature-flag query wiring to raw user query + enriched standalone
  query; `DUAL_QUERY_RRF_ENABLED` remains false pending real retrieval scores.
- [ ] Run the live dev-v2 A0/A1/B retrieval benchmark after Ollama
  `embeddinggemma:300m` is available; current attempt was blocked by the local
  embedding service being offline. Do not substitute a model or fallback.
- [ ] After retrieval gate passes, run Sol HTTP/SSE agent regression and open
  shadow-v2 once for the Release Candidate.

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
- [x] Đã rerun đủ regression-v1 60-turn đúng `cx/gpt-5.6-sol`: RC2 60/60,
  zero fallback/transport. Deep audit tìm 3 guard intervention; affected 11-turn
  và h10 4-turn replay trên code mới đều sạch cho các family liên quan. Không
  gọi v1 là blind benchmark.
- [ ] Nếu cần blind release score về sau, tạo shadow/held-out v2 từ unseen
  PDFs/public samples sau khi regression-v1 ổn; không tái gọi v1 là held-out.
- [ ] Sau release gate, rollout production theo staged backup/health/rollback;
  authorization hiện tại chỉ cho isolated clone, không tự cho phép persist sang
  production DB.
- [ ] Card freshness phụ thuộc content/parser/schema/prompt fingerprints; corpus
  re-index tạo pending rebuild nhưng không tự gửi provider. Monitor pending,
  stale, failed counts trong `/health` và evidence-card status API.

## Sau RC2/deep audit — 2026-08-23

### PLAN.md gate implementation checkpoint

- [x] Typed EvidenceBundle and modality-tagged candidates are available without
  changing the existing context/validator boundary.
- [x] DocumentReadinessReport is persisted per indexed document and distinguishes
  native/OCR/visual-only/unresolved page states.
- [x] Original + safe standalone rewrite RRF and card/FTS candidate union are
  implemented behind `DUAL_QUERY_RRF_ENABLED=false`; query drift guard rejects a
  rewrite that drops required entities. No production flag promotion yet.
- [x] Public three-mode baseline reports token budget/overflow explicitly and
  labels truncated extracted context `windowed_long_context`.
- [ ] Run retrieval-only dev-v2/shadow-v2 A/B with qrels before enabling dual
  RRF. Compare candidate recall, required-document coverage, evidence recall,
  answer validation and local latency separately; do not infer quality from the
  existing regression-v1 60-turn pass.
- [ ] Add per-document/page/modality diversity assertions to EvidenceBundle
  assembly and evaluate optional cross-encoder only after Candidate Recall@30
  reaches the plan gate.

- [ ] Dựng blind v2 từ unseen PDFs/public samples trước khi tuyên bố chất lượng
  production; regression-v1 hiện hữu không còn khách quan cho release score.
- [ ] Mở rộng MMLongBench-Doc/SPIQA qua chính production ingestion/retrieval path
  theo stratified dev slice; smoke hiện có chứng minh OCR + persisted VLM visual
  context cho một table-image miss, nhưng chưa đủ đo arbitrary-PDF quality.
- [ ] Giảm latency generation: RC2 median first visible delta 5.264 s và total
  13.379 s; chat thuần vẫn trả thêm provider-router latency. Không đánh đổi
  quantitative fail-closed guard để lấy TTFT giả.
- [ ] Agent hiện mới expose `search_local_docs` và `retrieve_visual_assets` trong
  regression contract. Chỉ thêm web/file/shell/MCP/sub-agent tools sau khi RAG,
  authorization, provenance và eval gate của tool tương ứng rõ ràng.
- [ ] Cân nhắc lưu canonical numeric ledger lúc ingest thay vì quét mọi table
  focus tại validation khi corpus/document lớn hơn; cách hiện tại không tốn model
  token nhưng CPU/RAM cần benchmark trên PDF nhiều bảng.
- [x] Sol stability check khi Codex dùng Luna để code: dev-20 full pass 20/20 và
  targeted affected replay 4/4 trên isolated clone; không promote kết quả này
  thành blind/production score, vẫn cần theo dõi provider stream variability.

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
- [ ] Tối ưu/background cold page OCR và visual/table enrichment, kèm UI indexing
  progress. Public smoke xác nhận visual context giúp trả đúng table-image
  answer, nhưng cold OCR/enrichment không được block first chat response; đo
  riêng parse/OCR/VLM/index stage trước khi đặt latency target.
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

## Retrieval Evaluation Gate v2.1 follow-up — 2026-08-24

- [ ] Human-review `evals/retrieval_v2/dev-v2.1.proposed.jsonl`: verify real
  question→span/table/figure answer relation, language label, follow-up state,
  rewrite provenance and zero gold leakage; mark `language_reviewed` /
  `qrels_reviewed` then publish audited `dev-v2.1.jsonl`. Start from the
  generated `dev-v2.1.review-packet.jsonl`, never from runtime UUIDs. Prioritize
  OCR-sensitive design qrels (notably table/reference-like headings) instead
  of treating preflight 60/60 as semantic approval.
- [x] Run audited A0/A1/B-full/B-anchor with frozen rewrite fingerprint. It
  failed the retrieval gate (A0/A1/B-full Evidence R@10 `0.70`, B-anchor
  `0.64`), so Sol dev-20/regression-v1 and any promotion remain blocked.
- [ ] Build a semantically audited raw-signal diagnostic slice before judging
  RRF: B-full activated 0/60 because the frozen full rewrite retained every raw
  anchor. Include naturally conversational exact acronyms/identifiers whose
  standalone rewrite legitimately normalizes them, and report activation rate.
- [ ] Improve the retrieval baseline before another promotion run. After
  card-repair + production-faithful current-turn catalog scope, audited A0 is
  Candidate R@30 `0.87`, Evidence R@10 `0.72`, Document R@5 `0.95`; table and
  figure are `1.00`, while text/multi-hop remain weak. Diagnose FTS/vector miss
  and final top-k cutoff without disguising it as a reranker failure.
- [ ] Add a separate production-compare retrieval/E2E report. The direct
  RagService evaluator's `0.50` multi-paper coverage bypasses the real
  per-document decomposition + context reservation policy, so it is a useful
  primitive diagnostic but not a valid score for Aya compare behavior.
- [ ] Keep `scripts/backfill_document_cards.py` as the operational repair for
  old snapshots; add a readiness dashboard/count and isolated migration smoke
  when testing new file formats. New normal ingestion already creates cards.
- [ ] If reranking is revisited, benchmark an index-vector/cross-encoder design
  with bounded candidates first. The current embedding reranker re-embeds
  candidate text and was stopped after minutes on local Ollama, so it is not a
  viable default under the existing latency gate.
- [ ] Build sealed `shadow-v3` from real public MMLongBench-Doc/SPIQA
  question-evidence pairs after isolated production ingestion. Do not open or
  reuse synthetic/cyclic `shadow-v2`.

## Gate-C v2.2 semantic review update (2026-08-25)

- [x] Ran the independent Sol semantic audit through the approved local
  9router route on the 60-case proposal. The first report was deliberately
  non-promotable: `43 accept / 16 reject / 1 uncertain`; rejects exposed
  missing table markdown and weak/title-only text qrels rather than a
  retrieval failure.
- [x] Added `scripts/refresh_retrieval_eval_evidence.py`. It reselects
  substantive text spans by facet and attaches canonical table markdown or
  figure captions from the local SQLite projection. It never edits rewrites,
  never marks a suite audited, and fails closed when a source identity cannot
  be refreshed. The refreshed proposal currently has `50 text + 10 table +
  10 figure` groups with zero refresh failures; the two synthetic short-note
  cases remain valid compact evidence.
- [x] Broadened table questions to ask for the table's reported information,
  not an unsupported performance/metric facet. Static audit remains **60/60**.
- [x] After explicit approval to transmit longer canonical excerpts (up to
  ~3.6k text/~5k table chars), Sol accepted **60/60, 0 reject, 0 uncertain**
  on `dev-v2.2.canonical-proposed.r5.jsonl`. A separate
  `apply-semantic-audit` command records this as an `independent_model`
  attestation (not a human review) and publishes `evals/retrieval_v2/dev-v2.2.jsonl`;
  isolated preflight resolves **60/60**.
- [ ] Rerun A0/A1/B-full/B-anchor on the model-attested suite. The first
  attempt was correctly fail-closed because both local Ollama EmbeddingGemma
  and 9router were offline; no fallback model was used and no production index
  was touched.
# RAG Foundation correctness pass — remaining gates (2026-08-25)

## Gate-C v2.2 benchmark validity (2026-08-25)

- [x] Added structural suite-quality diagnostics. An audited qrel suite is
  rejected for promotion when language/context is confounded or one template
  dominates a category.
- [x] Generated `evals/retrieval_v2/dev-v2.2.proposed.jsonl`: 60 cases,
  15 each across VI-contextual, VI-standalone, EN-contextual and
  EN-standalone; every category covers all four cells and raw-query template
  dominance is bounded. Portable qrel preflight resolves **60/60**.
- [ ] Independent semantic audit still required. v2.2 remains a proposal and
  must not be reported as a production benchmark until questions/qrels are
  reviewed separately.
- [x] Added `scripts/evaluate_agent_policy_retrieval.py`: a no-LLM diagnostic
  that measures per-document comparison branches and facet-parallel multi-hop
  with deterministic round-robin evidence quotas. Isolated runtime completed
  on the migrated index: Candidate R@30 `0.87`, Evidence R@10 `0.78`, Document
  R@5 `1.00`, multi-paper `1.00`, modality text/table/figure
  `0.733/0.700/1.000`, P50/P95 `126/203 ms`, zero scope leaks. These are
  draft-suite diagnostics only; semantic audit and A0 improvement remain open.
- [x] Added generic structural facet expansion, deeper candidate frontier,
  section-aware priors, coverage-aware final selector, and exact Table/Figure
  number matching. Corrected isolated draft-v2.2 A0 now measures Candidate
  R@30 `0.96`, Evidence R@10 `0.91`, Document R@5 `1.00`, text/table/figure
  `0.85/1.00/1.00`, multi-paper `1.00`, P50/P95 `123/259 ms`. Dual-query is
  still off; report is not promotion evidence until semantic audit.
- [x] Added `scripts/audit_retrieval_eval_v22.py`; static audit passes 60/60
  structure/focus/artifact checks, but it deliberately leaves semantic review
  required and never changes curation status.
- [x] Added `scripts/semantic_audit_retrieval_eval_v22.py` for an independent,
  fail-closed Sol review of question/context/evidence alignment. It records
  suite and prompt fingerprints and never publishes or overwrites a suite.
  The review has not been run yet because sending paper excerpts for this new
  semantic-review purpose needs an explicit privacy approval; no excerpt was
  transmitted and promotion remains blocked.
- [x] Ran a no-LLM public retrieval smoke through the isolated production
  coordinator (`scripts/evaluate_public_retrieval_only.py`). One image-only
  MMLongBench PDF had page recall `1.00` after its first index; warm reuse was
  ~32 ms for indexing and ~2.1 s retrieval. Four SPIQA cases all ingested
  successfully; the diagnostic evidence phrase recall was `0.50` overall
  (`0.833` figure, `0.167` table), exposing the known table-image limitation
  when no VLM/visual retrieval is enabled rather than a parser crash.
- [x] Fixed a real revision-isolation bug in the SPIQA public artifact adapter:
  table/figure rows were inserted without `revision_id` and therefore could be
  excluded by active-revision Lance queries. Adapter rows now carry the active
  revision and regression coverage asserts it. Also added an unchanged-rescan
  fast path so raw-byte-identical PDFs are not parsed/OCR'd again. The identity
  pass for changed/new files now disables page OCR and leaves OCR to the single
  canonical artifact pass, removing a second cold OCR traversal.
- [x] Added and ran `scripts/evaluate_format_lifecycle_smoke.py` on generated
  local TXT/MD/DOCX fixtures. All three reached `ready`, produced valid
  revision-scoped Lance projections, preserved Markdown table/image and DOCX
  table/image counts, and passed text retrieval. Image text itself remains a
  separate visual/VLM capability rather than being misreported as text recall.
- [x] Extended the same smoke with a one-page image-only PDF. Selective OCR
  recovered a split identifier after a conservative OCR token-boundary repair;
  projection and retrieval passed while readiness correctly remained `partial`
  because the recovered page is too sparse to claim fully reliable text.

- [x] Run an isolated real-Ollama Lance migration/activation smoke; the new
  `scripts/migrate_lancedb_revision.py` only builds a separate projection and
  must not be pointed at production until counts/recall are audited.
- [x] Run audited `dev-v2.1` Gate C and classify baseline misses before any
  dual-query decision. The run is valid but fails promotion: A0/A1/B-full
  Candidate `0.87`, Evidence `0.72`, Document `0.95`; B-anchor Candidate
  `0.83`, Evidence `0.67`, Document `0.95`. `DUAL_QUERY_RRF_ENABLED` remains
  false.
- [ ] Enable enrichment jobs only for explicitly approved folders, then measure
  9router VLM/cards/LightRAG retry behavior and provenance on a small isolated
  corpus; provider failures must leave local-ready active.
- [ ] Build stratified PDF/DOCX/MD/TXT public evaluation and visual T/V/T+V A/B;
  no ColVision promotion from smoke cases.
- [ ] Freeze RC fingerprints, backup/rollback and unseen `shadow-v3` only after
  the isolated lifecycle and retrieval gates pass.
- [x] Run isolated EmbeddingGemma lifecycle smoke and revision-scoped Lance
  migration. Smoke passed; direct production evaluation was correctly blocked
  by the old schema and was rerun only on the isolated copy. The run also
  exposed and fixed Lance row metadata propagation for physical `revision_id`.
- [ ] Fix/generalize text retrieval misses and production-faithful multi-paper
  decomposition, then rerun Gate C. Current audited isolated score is
  Candidate `0.87`, Evidence `0.72`, Document `0.95`; no promotion yet.

### Gate-C rerun after local-network permission correction — 2026-08-25

- The previous `Ollama offline` result was caused by the Codex shell sandbox
  refusing localhost connections, not by the Ollama app being stopped. An
  outside-sandbox health check confirmed `embeddinggemma:300m` is available
  and returns 768-dimensional vectors.
- Re-ran the isolated, retrieval-only evaluator against the model-attested
  `dev-v2.2` suite with no 9router/model calls and no production mutation.
  A0/A1/B-full all measured Candidate `0.96`, Evidence `0.87`, Document `1.00`,
  multi-paper `1.00`, P50/P95 `128/266 ms` (A0). B-anchor measured Candidate
  `0.99`, Evidence `0.88`, Document `1.00`, multi-paper `1.00`, P50/P95
  `181/281 ms`.
- The retrieval gate is still not promoted: A0/B-full text modality recall is
  `0.7833` (< `0.80`), B-full provides no delta over A0 because the generic
  query-activation policy pruned all raw branches, and B-anchor is diagnostic
  rather than the production rewrite. `DUAL_QUERY_RRF_ENABLED` remains false.
- The remaining actionable work is generic text-recall/final-cutoff analysis,
  followed by a real production-faithful rerank-stage measurement and Sol
  HTTP/SSE regression; do not tune individual paper/caption cases.

### Sol HTTP/SSE E2E after Gate-C retrieval rerun — 2026-08-25

- 9router health and an authenticated `cx/gpt-5.6-sol` completion both passed.
  A seven-turn isolated HTTP/SSE conversation completed all turns with no
  transport, model or tool fallback, but the regression exit was non-zero.
- The run exposed real follow-up risks: turn 3 fell back on the requested
  ASPIRE metrics, turn 4 selected unrelated documents instead of WhiSER, and
  the isolated artifact root lacked figure bytes needed by the visual endpoint
  (404). The focus event contract also omitted expected turn-1 focus IDs even
  though active state was set.
- This is diagnostic only: the isolated copy has no LightRAG graph and no
  copied artifact images. Fix the fixture/provenance setup, then address the
  generic document-routing/follow-up retrieval miss. No production data was
  changed; the isolated backend was stopped after the run.

### Corrected isolated E2E and cache contract — 2026-08-25

- The diagnostic fixture was corrected to mirror the production layout:
  `APP_DATA_DIR/sqlite/app.db`, LanceDB and artifacts now share one isolated
  root, and absolute image paths are rewritten only inside that clone.
- The remaining failure was a real cache-contract bug. Retrieval cache keys
  now include raw-turn obligations (table/figure intent, exact artifact
  numbers, inventory and visual flags), with cache version `v24`; a rewritten
  query can no longer reuse an incompatible Table/Figure result.
- After the fix, the seven-turn Sol HTTP/SSE E2E passed **7/7**, including
  Acc/F1/CCC validation, ASPIRE ↔ WhiSER focus switching, figure bytes (HTTP
  200), and zero errors/fallbacks. Backend suite is **737 passed**. Dual-query
  remains disabled; this is isolated regression evidence, not a Gate-C
  promotion by itself.

### Gate-C improvement plan (after policy diagnostic)

1. **Audit before tuning.** Independently review every v2.2 question, required
   evidence group, alternative locator, language/context label and negative
   case. Publish only after semantic review; preflight alone is not enough.
2. **Make decomposition evidence-driven.** For multi-hop, derive required
   facets from the question plus conversation/working state, and fail closed
   when the facet classifier is uncertain. Run bounded facet branches in
   parallel, reserve evidence slots per facet/document, and record branch
   coverage rather than silently falling back to one generic query.
3. **Repair A0 text retrieval generically.** For each miss, distinguish FTS
   tokenization/normalization, document routing, vector miss, candidate fusion,
   parent expansion and final cutoff. Add fixtures for numbers, acronyms,
   hyphenation, two-column extraction and multilingual text; do not add
   paper-specific aliases or qrel-derived terms.
4. **Separate candidate recall from final selection.** Keep the same top-30
   candidate pool, then test deterministic diversity/coverage selection before
   considering a reranker. A reranker may be enabled only after measuring its
   latency and exact table/figure regression on the isolated index.
5. **Re-run paired arms.** On audited v2.2, run A0/A1/B-full/B-anchor and the
   production-policy evaluator with fixed seed, warm-up, stage timings and
   zero-LLM enforcement. Promote no feature based on the draft report.
6. **Agent gate and unseen data.** Only if retrieval passes all Gate-C metrics,
   run Sol HTTP/SSE answer regression, then stratified unseen PDF/DOCX/MD/TXT
   smoke and finally one sealed shadow-v3 run. Keep dual-query off and rollback
   available until every gate passes.

### Latest correctness result — 2026-08-25

- Fixed the migration-window defect where backfilled
  `rev:<document>:initial` documents caused old Lance rows without a revision
  field to disappear. The compatibility rule is restricted to the initial
  revision; later missing/stale revisions are rejected.
- Context and evaluator now share conservative facet inference for legacy text
  projections, with no paper-specific rules.
- Fresh isolated run: A0/A1/B-full Candidate `0.96`, Evidence `0.87`,
  Document `1.00`; table/figure `1.00`, text `0.7833`, P50/P95 about
  `131/340 ms`. B-anchor Candidate `0.99`, Evidence `0.89`, text `0.8167`.
  Gate C is not passed; dual-query remains off.
- Remaining work is a generic audit of three FTS/final-cutoff text misses and
  production-faithful rerank measurement. Do not tune individual paper/qrel
  cases. Verification: **740 backend tests**.

### Latest Gate-C result — 2026-08-26

- [x] Added bounded FTS variants and generic two-source-per-document final
  selection; no Jina, re-index or production mutation.
- [x] Isolated audited run: A0 Candidate/Evidence/Document `0.97/0.91/1.00`,
  modality text/table/figure `0.85/1.00/1.00`, multi-paper `1.00`, P50/P95
  `139/338 ms`; FTS misses cleared, six final-cutoff diagnostics remain.
- [x] A1/B-full produce the same metrics as A0; keep
  `DUAL_QUERY_RRF_ENABLED=false`.
- [x] Backend suite **742 passed**; Graphify refreshed.
- [x] Sol HTTP/SSE regression against the same isolated clone passed **7/7**
  after clone-only absolute artifact-path repair; no transport/model/tool
  fallback occurred. The folder-corpus RAG milestone can now be frozen.
- [ ] Later, outside this milestone: arbitrary-file expansion (new formats,
  visual retrieval and upload UI). Do not start Jina/ColPali or a new embedding
  benchmark for the current goal.
## Current open work after Gate-D matrix — 2026-08-26

- [x] Corrected MMLongBench-Doc-V2 metadata is supported and a profile-balanced
  isolated smoke completed 3/3 cases (native, mixed text/image, image-only).
  T/V/T+V page recall was `1.00`; the sample is diagnostic only. Reports now
  record child visual-worker telemetry separately from orchestrator RSS. The
  post-telemetry backend verification is **752 passed**.
- [x] Re-ran the isolated Gate-D lifecycle matrix: six valid PDF/DOCX/MD/TXT
  fixtures passed projection and retrieval, while the corrupt PDF remained
  contained in a `partial` folder job. Duplicate/rescan/change/rename checks
  passed; no production path was touched.
- [x] Sol agent-path smoke (2 historical MMLong V1 + 2 SPIQA) completed 4/4
  with no fallback. Page recall stayed `1.00` on MMLong, but answer accuracy
  was `0.50`; SPIQA diagnostic token-F1 was `0.737`. New runs default to
  corrected MMLong V2. This keeps the unresolved issue explicit: visual/context
  answer quality still needs a broader, judge-faithful gate.

- Gate C is complete only for the audited 21-paper suite. Do not call the
  whole RAG production-ready yet.
- Gate D needs a durable preparation path for image-only/public PDFs. A
  34-page scan spent beyond the bounded smoke window in full local OCR; the
  run was stopped and no score was claimed. Page OCR must be resumable and
  separately measured from warm retrieval.
- Page OCR is now resumable by source SHA-256/page; a 34-page MMLong scan
  completed at 184.5 s cold and 0.87 s warm retrieval. The next step is not
  to hide that cost, but to run it as a durable background stage and report
  cold-index versus warm-query latency separately.
- SPIQA table-image smoke exposed the next generic coverage issue: the
  official image artifact is retained and retrievable as an artifact, but its
  cell text is not searchable without a canonical OCR/visual extraction
  stage. Do not fix this by copying gold captions/evidence into queries.
- Local OCR is now attached as derived metadata for approved image artifacts
  and table-image adapters. Keep raw evidence-phrase recall separate from
  answer-token coverage because some SPIQA “highlighted evidence” values are
  annotation markers (for example `FLOAT SELECTED`) rather than content.
  The public evaluator must be audited before using that suite as a promotion
  gate.
- Gate E visual retrieval provider and T/V/T+V measurement are still open;
  current feature flag remains off. Shadow-v3, RC freeze, rollback drill and
  broad Sol E2E are also still open.
- Keep `DUAL_QUERY_RRF_ENABLED=false` and EmbeddingGemma/Ollama unchanged
  until the above gates are measured on unseen formats.
- Image OCR stage is implemented and tested, but the public suite is still a
  diagnostic sample (4 SPIQA cases + 1 MMLong document), not a broad promotion
  gate. Run grouped, stratified public cases and report cold preparation,
  warm retrieval, modality and annotation-quality metrics separately.
- Public SPIQA diagnostic is now 8 cases (3 figure/5 table), but its raw
  evidence metrics remain below a promotion threshold and answer-token
  coverage is not a substitute for full answer quality. Expand by document
  strata and add MMLong native/mixed/image-only groups before any RC claim.
- The first MMLong format-stratified smoke now covers one native-text, one
  mixed-text/image and one image-only PDF (3/3 page recall). The report keeps
  `document_profile` and per-profile latency/coverage so future samples are
  auditable. The image-only question requires arithmetic over chart values;
  `answer_token_coverage=0` is therefore not a retrieval miss by itself.
  Remaining work is a larger document-balanced sample and a real visual/page
  retrieval A/B, not a paper-specific query patch.
- The follow-up smoke covers two documents per profile (six total), with 4/4
  answerable page-recall hits and two explicit unanswerable cases excluded
  from the positive denominator. Cold preparation remains expensive for
  image-only PDFs; warm retrieval is measured separately. This improves
  confidence in format handling but does not close visual Gate E or constitute
  a full MMLong/SPIQA benchmark.
- ColSmol Gate-E worker isolation is verified: the main backend keeps its
  Transformers 4.x/Docling environment, while `backend/.venv-visual` owns the
  Transformers 5.x/colpali-engine stack. An interrupted page/index build
  resumes atomically. The 3-document MMLong and 4-case SPIQA runs remain
  diagnostics only; visual retrieval is not enabled in the default hybrid path.
- MMLongBench-Doc-V2 metadata is now cloned and supported by the evaluator;
  its corrected annotations should be used for new public answer-quality runs.
  Existing V1 reports are historical and not numerically comparable to V2.
  Source PDFs are resolved from the upstream local document cache, so missing
  files fail closed before scoring.
- The public visual runner now fails fast on an unavailable EmbeddingGemma
  provider and treats failed checkpoint rows as retryable, preventing an
  interrupted/outage run from being silently counted or from reparsing scans.

## Follow-up correctness work — 2026-08-27

- [x] Make the Gate-C runner production-shaped for comparisons: scoped
  per-document branches, stable rank interleaving and canonical identity
  de-duplication before the shared final top-k selector.
- [x] Add a revision-schema preflight so a legacy evaluation DB cannot be
  mistaken for an isolated active-revision index.
- [x] Mark provider/model-mismatched stage receipts as stale in document
  status; an old complete receipt no longer implies current readiness.
- [x] Add a visual Gate-E promotion contract with minimum sample/profile
  coverage, T+V gain, per-case non-regression and worker-health checks.
- [x] Completed the broad V2 visual run (three MMLong profiles plus four
  SPIQA cases). Its result remains diagnostic because the explicit Gate-E
  sample/profile checks did not pass; do not enable the visual feature by
  benchmark name alone.
- [x] Ran a bounded Sol answer-quality sample over the same public strata after
  visual retrieval measurement. Gold, full-document and Aya pipeline modes are
  separated; token overlap remains diagnostic only.
- [x] Retried the bounded Sol sample after the approved local 9router endpoint
  became reachable; it completed 21/21 generations without fallback. Remaining
  work is generic visual-page context fusion and a larger stratified gate.
- [x] Added and exercised the isolated visual-page answer diagnostic. It
  improved one MMLong chart case but remained below answer correctness, so the
  remaining problem is generic multi-page aggregation, not a missing page
  lookup or a paper-specific alias.
- [x] Added a bounded, provenance-preserving map/reduce path for
  document-wide page aggregation (map page/batch evidence, reduce
  counts/lists). A single final Sol prompt over 20+ page images is not
  reliable enough for chart arithmetic; the map/reduce path remains opt-in and
  must still be measured on a larger stratified sample before enabling it by
  default.
- [x] Completed the resumed V2 visual run after page-batched worker hardening;
  report `data/retrieval_eval/public/results/visual-gate-v2-broad.json` has
  7/7 successful rows and `promotion.passed=false` solely because the sample
  is below the minimum 20 positives/two-per-profile contract. A visual smoke
  is not a claim that arbitrary chart arithmetic or image-only QA is solved.

## 2026-09-01 — Gate-E follow-up

- [x] Same-page active-revision text companions now accompany selected visual
  and table/page artifacts without changing canonical identity or retrieval
  slots.
- [x] Visual intent recognizes picture/photo/screenshot/slide wording, so a
  public question cannot bypass visual sufficiency merely because metadata
  omits `Figure`/`Image`.
- [ ] Gate-E is still not promotable. Corrected SPIQA visual smoke (8 cases)
  remains below bounded full-context answer quality, especially table cases.
  The MMLong image-only unanswerable replay still needs a general abstention /
  unsupported-claim policy rather than a case-specific rule.
- [ ] Do not enable `VISUAL_RETRIEVAL_ENABLED` until a >=20-positive,
  profile-balanced T/V/T+V retrieval and Sol answer-quality run passes with no
  exact text/table regressions. Keep `DUAL_QUERY_RRF_ENABLED=false`.

## 2026-09-02 — Remaining Gate-E issue

- [x] Exact page/slide hints now constrain only multimodal attachments in both
  live and isolated paths; unresolved hints fall back to the ranked pages so
  diagnostics remain honest.
- [x] Added generic cross-page numeric-transfer and visual-abstention guidance
  to the evidence sufficiency context and public evaluator prompt.
- [ ] Gate-E is still open. The remaining failures are answer-level visual
  grounding (especially image-only/unanswerable and table-heavy cases), not a
  missing ColSmol page lookup. Re-run the cached stratified Sol sample when
  9router is reachable before deciding whether map/reduce or stricter visual
  claim validation should be enabled.

## 2026-09-04 — current blockers and next measurements

- [x] Gate C is green on the audited 21-paper retrieval suite; no need to
  change the embedding model or enable dual-query solely from that report.
- [x] Added generic, bounded multi-evidence operand expansion and phrase-aware
  reservation. This targets the class of failures where both complementary
  facts are in the candidate frontier but one is discarded before generation.
- [ ] Gate-E still blocks the arbitrary-format completion claim. The current
  public answer-quality report is only 20 cases and remains diagnostic; visual
  T+V is 18 positive cases and below promotion minimums.
- [ ] Re-run warm cached MMLong/SPIQA with Ollama and 9router reachable,
  starting with hard numeric multi-page cases, then expand to a
  document-balanced profile set. Record retrieval and answer deltas without
  changing qrels or production flags.
- [x] The first post-fix causal replay (`r36`, MMLong 0990) is green: both
  numerator/denominator pages were selected and Sol returned `5.03%`.
  Generalization is still unproven until the paired multi-case run.
- [ ] If numeric multi-page retrieval passes, evaluate generic page-level
  map/reduce and visual T/V/T+V on unseen documents. Keep ColSmol isolated and
  off by default until profile-balanced gates pass.
- [x] Visual sufficiency now distinguishes pixel-required appearance questions
  from caption/inventory questions; a figure ID alone cannot prove what an
  image depicts. Full verification reached **847 backend tests**.
- [x] Production multimodal requests now carry a page-aware attachment map;
  this is transport provenance only and does not change retrieval ranking.
- [x] Canonical figure/image rows with safe local paths can now reach the
  multimodal model under the same bounded, page-aware contract.

## 2026-09-04 — measured Gate-E follow-up

- [x] Added a shared case-manifest frontier to retrieval-only evaluation;
  paired retrieval and answer-quality runs can no longer silently sample
  different cases.
- [x] Fixed generic prompt-budget loss for exact technical anchors. A late
  already-retrieved acronym/identifier/value is promoted only when its terms
  are discriminative across the candidate set; no benchmark answer or paper
  mapping is consulted.
- [x] Warm 20-case retrieval-only report `gate-e-retrieval-r47-final-20case`:
  20/20 successful, MMLong page recall `0.90`, retrieval-support recall
  `0.6875`, P50/Pmax `151.8/684.3 ms`.
- [x] Production-parity visual report `gate-e-visual-r49-production-parity`:
  18 positive cases, T/V/T+V recall `0.5000/0.7778/0.9444`, no per-case
  fused regression; visual index `413 MB`, worker RSS `363 MB`, query
  P50/P95 `102.6/212.7 ms`.
- [ ] Gate-E remains open: visual promotion requires >=20 positives, paired
  Sol answer-quality evidence, and a larger unseen/document-balanced suite.
  Keep `VISUAL_RETRIEVAL_ENABLED=false` and `DUAL_QUERY_RRF_ENABLED=false`.

## 2026-09-04 — answer-quality and intent-parity follow-up

- [x] Completed the resumed 20-case Sol replay (`r37-postfix`): 20/20 cases,
  60/60 generations, zero failed rows/fallbacks. Its score is diagnostic only:
  Aya `0.4238`, full extracted `0.5312`, gold `0.5719`.
- [x] Unified natural-language result/table intent across the live agent and
  public evaluator. Queries mentioning metrics, performance, scores or
  results now reserve canonical table evidence without requiring the literal
  word `table`.
- [x] Marker matching is token-boundary aware; generic intent cues do not
  trigger on unrelated words such as `resulting` or `valuation`.
- [ ] Re-run the paired retrieval/answer frontier after this parity change
  when Ollama and 9router are reachable. Do not compare the new run directly
  with the pre-parity report; keep the same case manifest and runtime root.
- [ ] Gate-E is still open: SPIQA table/figure answer quality, image-only
  abstention and >=20-positive profile-balanced unseen coverage remain
  unresolved. Keep visual and dual-query flags off.

## Gate-E completion plan (research-informed, no case-specific tuning)

1. **Measurement integrity:** re-run the exact manifest after the natural
   table-intent parity change, then publish paired retrieval and Sol answer
   reports. Never compare reports with different context/rerank contracts.
2. **Canonical artifact recovery:** keep text/table/figure/page identities
   unified through candidate generation, context projection and validation;
   add generic table/figure presence and final-stage recall checks before any
   prompt change.
3. **Multi-evidence reasoning:** evaluate generic sub-question decomposition
   and coverage for ratios, comparisons and cross-page/table questions. Track
   operand coverage and rerank loss separately; do not add paper aliases.
4. **Visual channel:** keep ColSmol/ColPali-style page retrieval isolated and
   late-fused. Require >=20 positives, at least two examples per PDF profile,
   paired Sol answer quality, no exact text/table regression and measured
   CPU/RAM/latency before enabling the flag. ColPali/VisRAG/M3DocRAG motivate
   the channel, but model names are not acceptance evidence.
5. **Arbitrary formats:** ingest a document-balanced set of native PDF,
   mixed-scan, image-only, table-image, chart/diagram, DOCX and MD/TXT through
   the same folder coordinator. Report readiness/provenance and separate
   parser, artifact, retrieval and generation failures.

Completion requires the existing Gate-C thresholds, the visual/answer gates,
and an unseen shadow suite. Until then this remains an advanced local RAG
prototype with explicit measured limits, not a production-ready claim.

## 2026-09-04 — observability follow-up

- [x] Visual evaluator and report merger now expose T+V wall-clock and fusion
  stage latency; previously these fields were null even though T and V were
  measured. This is a general measurement-contract fix, not retrieval tuning.
- [ ] Re-run the same paired public frontier after Ollama and 9router are
  reachable, then rerun the full backend suite before any RC decision.
- [x] Artifact-only table/figure locators now activate pixel evidence in the
  isolated answer evaluator; native structured tables are not forced through
  visual retrieval. Measure the change on the paired Sol frontier before
  treating it as a quality improvement.

# 2026-09-10 — remaining Gate-E work

- [x] Full backend suite after embedding batching/progress telemetry: **891
  passed, 5 warnings**.
- [x] Visual-enabled Sol smoke completed on 10 MMLong cases (30 generations,
  zero failures; Aya accuracy `0.50`, page recall `0.875`). Diagnostic only.
- [ ] Run visual indexing as a bounded background job with per-document
  progress, resumable page batches and `partial` readiness; local text/table
  search must remain usable while visual pages are pending.
- [x] Batch progress is now persisted by the durable visual job; remaining
  work is to expose/verify the end-to-end background policy on a long PDF.
- [x] Isolated format lifecycle matrix `r55` passed all required identity and
  retrieval invariants across PDF/DOCX/MD/TXT fixtures; broken PDF remains
  explicitly partial. Matrix runs must use the approved network context for
  Ollama localhost.
- [ ] Repeat Gate-E on a larger profile-balanced set with no exact text/table
  regression, >=20 positive cases, paired Sol quality and CPU/RAM/latency.
- [x] SPIQA visual diagnostic `r56` completed 8/8 cases with zero failures;
  keep its `32.5 s` median cold visual-index cost visible in any RC report.
- [ ] Keep `VISUAL_RETRIEVAL_ENABLED=false` and
  `DUAL_QUERY_RRF_ENABLED=false` until unseen shadow and agent E2E gates pass.

# 2026-09-11 — remaining work after conversational fix

- [x] Resolved the reproducible metric-follow-up/table-selection failure in a
  general way; do not revert to prompt-only or paper-specific fixes.
- [ ] Re-run the paired public Gate-E answer-quality suite with the current
  canonical-table behavior and report retrieval vs generation separately.
- [ ] Complete the larger profile-balanced visual replay (>=20 positive
  cases), including cold-index cost and no exact text/table regression.
- [ ] Keep both visual and dual-query flags disabled until the unseen shadow
  and agent E2E release gates pass.

# 2026-09-11 — Gate-E paired frontier results

- [x] Completed a contract-matched 20-positive-case visual/answer frontier;
  T+V retrieval recall is `0.95` with no fused text regression.
- [x] Completed paired Sol answer quality (`20/20` cases, `60/60` rows,
  zero failures). Aggregate Aya-vs-full is positive, but four per-case
  regressions (`0990`, `1905.12260:0000`, `1906.05474:0000`, and
  `1703.05260:0001`) fail the safety gate.
- [ ] Add at least one more `figure+table` positive case before treating the
  profile-balanced visual gate as representative; the current count is `1`.
- [x] Prepared a non-tuning manifest for the additional public
  `spiqa-c:1704.08960:0000` figure+table case; it still needs a provider-backed
  Sol replay before it can join the paired frontier.
- [ ] Make cold visual indexing a resumable background/readiness stage and
  expose its cost; case `0845` required about `248.9 s` index time.
- [ ] Diagnose the remaining table/figure answer regressions by failure class
  (retrieval, context projection, visual sufficiency, generation), without
  adding case-specific mappings or prompt hacks.
- [ ] Run unseen shadow-v3 and final HTTP/SSE agent regression after the above;
  keep `VISUAL_RETRIEVAL_ENABLED=false` and `DUAL_QUERY_RRF_ENABLED=false`.
- [ ] Replay the four answer-quality regression IDs with current code once
  Ollama and 9router pass preflight; the attempted replay failed closed while
  both providers were unreachable, so it produced no valid score.
- [x] Evaluation-tooling regression tests cover visual/answer frontier
  selection and exclusion; full backend suite is **894 passed**.
- [x] Answer-quality checkpoints and merges now carry an evaluator-code
  fingerprint, preventing stale reports from entering a current frontier.
- [x] Visual-gate checkpoints and merges carry the same fingerprint contract;
  old visual rows are discarded on resume and incompatible reports fail closed.
- [x] Marked pre-fingerprint frontier reports `r68`/`r70`/`r73` invalid for
  promotion; regenerate them under the current evaluator before any gate
  decision.
- [x] Expanded evaluator fingerprints to backend RAG/retrieval dependencies;
  replay must be regenerated whenever those modules change.
# 2026-09-11 — current replay diagnosis and generic fixes

- [x] Confirmed both local providers are reachable and ran fresh Sol
  Aya-only public replays in an isolated runtime; no fallback was used.
- [x] Prevented structured table/figure candidates from being displaced by
  repeated top-of-document text when the query carries generic metric/result
  intent.
- [x] Preserved complementary evidence obligations through the final bounded
  projection and deduplicated visual/page rows by canonical page identity.
- [x] Added regression coverage; full backend suite is **900 passed, 5
  warnings**.
- [x] Generic value-bearing operand projection fixed the MMLong ratio failure
  (`0990`) without page mapping; current paired replay returns `5.03%` and
  evidence page recall `1.0`.
- [ ] SPIQA figure `1703.05260:0001` now retains its agreement chunk and is
  near full-context quality, but broader visual-generation variance remains
  to be measured on the expanded frontier.
- [ ] Re-run paired gold/full/Aya frontier after these changes, then rerun
  unseen shadow-v3 and HTTP/SSE agent regression. Keep both feature flags off
  until all gates pass.
- [x] Paired replay `r80` is current-code and fingerprinted: `5/5` cases,
  `15/15` generations, zero failures; Aya-vs-full delta `+0.0390`. The result
  validates the generic context fixes but is too small for promotion.
- [ ] Expand the paired frontier across unseen PDF/DOCX/MD/TXT and visual
  strata; retain the remaining SPIQA figure regression as a visible failure.
- [x] Re-ran audited Gate-C `dev-v2.2` retrieval-only (`r7`): A0/A1/B-full
  Candidate R@30 `0.97`, Evidence R@10 `0.91`, Document R@5 `1.00`, warm
  A0 P50/P95 `141/360 ms`, zero scope leaks/model calls. B-anchor is
  diagnostic at `0.98/0.93`; B-full is non-inferior but not a `+0.02`
  improvement, so the dual-query flag stays off.

# 2026-09-11 — current visual frontier

- [x] Added explicit isolated evaluator stage-contract selection; default is
  `foundation-v3`, and old `foundation-v2` reuse is opt-in and report-visible.
- [x] Completed current-code visual retrieval `r81` on 21 balanced public
  cases (MMLongBench + SPIQA), all successful and zero LLM calls. T/V/T+V
  recall is `0.4286/0.8095/0.9524`; fused P50/P95 is `268.6/961.0 ms`.
- [ ] Attach a paired Sol answer-quality run for the same 21-case manifest;
  retrieval-only success cannot satisfy the visual promotion gate.
- [ ] Run fresh `foundation-v3` arbitrary-format ingestion on the public
  stratified set (native/mixed/image-only PDF, DOCX, MD/TXT) and expose
  resumable visual readiness; `r81` reused old isolated projections for
  ranking diagnostics.
- [ ] Keep visual and dual-query flags off until paired quality, unseen
  shadow-v3 and HTTP/SSE agent regression pass.
- [ ] Public 16-case answer replay `r84` was stopped fail-closed at 17/48
  rows while a subsequent cold foundation-v3 ingestion was running; no
  aggregate is valid. Add an explicit per-stage/case timeout or resumable
  lease boundary before retrying large Sol frontiers, so a slow parser/job
  cannot hold the evaluator indefinitely.
- [ ] The local-paper seven-turn HTTP/SSE E2E was not started in this run:
  the execution safety gate requires explicit per-run authorization before
  exporting private corpus excerpts to 9router. Chat-only SSE smoke remains
  verified; do not treat it as RAG-agent E2E.

# 2026-09-11 — foundation-v3 lifecycle evidence

- [x] `format-lifecycle-matrix-r56` passed the folder coordinator contract:
  valid PDF/DOCX/MD/TXT/table-image/mixed fixtures projected and retrieved
  correctly; broken PDF is honestly `partial`.
- [x] Verified stable rescan, duplicate-byte deduplication, content-change
  revisioning, rename identity and source aliases in the same isolated run.
- [ ] Run a fresh foundation-v3 public stratified ingestion set (not reused
  foundation-v2 projections), then pair it with Sol answer-quality and agent
  HTTP/SSE evidence.
- [ ] Create/open `shadow-v3` only after the fresh RC freeze; keep visual and
  dual-query flags disabled until its gates pass.
- [x] Re-ran current public Sol answer-quality smoke (`r82`, 5 cases / 15
  generations, zero failures); it remains diagnostic because Aya-vs-full is
  `-0.0137` and the paired frontier is below the 20-case gate.
- [ ] Expand that answer-quality run to the full 21-case visual manifest only
  after deciding the cold-index cost is acceptable; do not merge the 5-case
  result into a promotion report.

# 2026-09-11 — evaluator reliability

- [x] Added bounded retrieval/prepare and generation timeouts to the public
  multimodal evaluator, with stage-attributed failed rows and checkpoint
  fingerprints that include both budgets.
- [x] Verified timeout-aware evaluator tests (`17/17`) and full backend suite
  (`904/904`); one real public gold/full Sol smoke completed (`2/2`).
- [ ] Re-run the remaining public frontier with an explicit budget and review
  timeout rows as ingestion/provider failures; do not convert timeout rows into
  quality scores or silently skip them.
- [x] Fresh `foundation-v3` visual smoke `r85` now completes `5/5` under an
  explicit budget; the earlier 180s timeout is retained as a failure diagnostic
  and is not merged into quality scores.
- [x] Worker cancellation now terminates the isolated ColSmol child without
  waiting on an in-flight request lock; timeout cleanup is bounded and tested.
- [x] Visual evaluator `prepare` now uses a non-blocking timeout boundary and
  marks an interrupted 21-case frontier invalid instead of presenting partial
  rows as a score.
- [ ] Investigate the generic image-only MMLong prepare timeout (parser/vector
  cost profile) before expanding the visual frontier; fix the pipeline class,
  not the individual case.
- [x] Desktop production build passes and backend/frontend/9router/Ollama were
  health-checked after the evaluator runs for UI testing.

# 2026-09-12 — visual frontier r87

- [x] Ran the timeout-aware visual evaluator on the next isolated public
  frontier. Four cases completed; one 78-page-class case (`0569`) hit the
  900-second visual-index budget and was recorded as a failed row.
- [x] Kept the report fail-closed: no partial aggregate is treated as Gate-E
  promotion evidence. The diagnostic four-case T+V recall was `1.00` versus
  T `0.75`, but fused P50/P95 was about `813/1425 ms` and the worker was
  terminated after timeout.
- [ ] Fix the generic visual prepare/index cost profile (or add a resumable
  bounded batch policy) before retrying the full public frontier; do not tune
  around `0569`.
- [x] Fixed stage-specific timeout reporting so a visual-index timeout cannot
  be mislabeled as the parse/prepare budget; evaluator contract test added.
- [ ] Pair the same completed manifest with Sol answer-quality and then run
  HTTP/SSE agent regression. Keep visual and dual-query flags off until all
  retrieval, answer-quality and shadow gates pass.

# 2026-09-13 — r88 interrupted after host sleep

- [x] Resumed the isolated visual frontier and reused committed page embeddings;
  seven cases reached checkpoints, including successful completion of `0569`.
- [x] Stopped the run safely after host sleep left `0315` without progress for
  several hours. The interrupted report is explicitly invalid for scoring and
  no worker process was left behind.
- [ ] Add/verify a durable per-document preparation lease/checkpoint so a long
  parser/OCR stage can resume independently after host sleep, then retry the
  remaining public frontier. Do not interpret the interrupted run as a visual
  retrieval regression.

# 2026-09-13 — r88 complete and r90 answer quality

# 2026-09-13 — current-code verification and r94 diagnostic

- [x] Verified backend/frontend/9router/Ollama are running; `/chat/stream`
  emitted real SSE deltas through `cx/gpt-5.6-sol`.
- [x] Desktop production build passes; full backend suite is **906 passed**.
- [x] Completed the isolated r94 SPIQA artifact-projection replay (3/3
  generations). It attached three retrieval-selected images but Aya remained
  below full-context quality (`0.1600` vs `0.4242` diagnostic token-F1), so no
  promotion or aggregate merge was made.
- [ ] Re-run the complete paired 21-case answer-quality frontier only if a
  current-code aggregate is needed; r92 remains the last authoritative paired
  aggregate and all visual/dual-query flags stay off.
- [x] Added active-revision page-to-artifact context projection for visual
  hits, with generic table/figure OCR/caption enrichment and retirement tests;
  full suite is now **908 passed**.
- [x] Current-code r95 replay of SPIQA `1704.08960` improved Aya token-F1
  `0.1600 → 0.4444` and exceeded full extracted context `0.4242`; retained as
  a one-case diagnostic only.
- [x] Current-code paired r96 frontier completed **21/21 cases, 63/63
  generations**, zero failures: MMLong Aya `0.6310`, SPIQA diagnostic
  token-F1 `0.4378`, overall Aya `0.5390`; visual promotion remains closed.
- [x] r97 replay of three r96 failures completed 9/9; two recovered and one
  remained slightly below full, confirming both context and generation-variance
  components without adding case-specific logic.
- [ ] Run permitted agent HTTP/SSE regression on a public/isolated manifest;
  do not use the private 21-paper corpus without a new explicit upload gate.

# 2026-09-13 — r98 and format lifecycle smoke

- [x] Replayed r96 failure classes with canonical first-class artifact rows:
  9/9 generations, zero failures; two cases recovered to full-context parity,
  one remained a small metrics-summary variance.
- [x] Ran isolated PDF/DOCX/MD/TXT lifecycle smoke: 4/4 projection and
  expected-retrieval contracts passed; image-only PDF correctly remains
  `partial`.
- [ ] Add durable per-document preparation leases/checkpoints so long OCR or
  visual preparation resumes after host sleep instead of restarting.
- [ ] Run public/isolated HTTP-SSE agent regression and a full current-code
  paired frontier before any promotion decision; keep private corpus gated.

# 2026-09-13 — canonical progress checkpoint

- [x] Persist per-file canonical scan/index/skip/failure counters and current
  path under the durable job lease; terminal jobs retain the last progress
  snapshot. Full suite: **909 passed**.
- [ ] Add independent parser/OCR batch checkpoints if profiling shows a
  single document itself needs resumable preparation; current file-level
  restart is safe and idempotent.

# 2026-09-16 — service/lifecycle verification checkpoint

- [x] Restarted current backend/frontend and verified UI/backend HTTP 200;
  Ollama and the 9router model catalog are reachable (completion auth is
  separately rejected with `401 token_revoked`).
- [x] Fresh isolated arbitrary-format lifecycle matrix passed all invariants
  and six valid-file retrieval contracts; corrupt input remains explicitly
  `partial`.
- [ ] Re-authorize 9router before Sol HTTP/SSE agent regression; gateway
  reachability alone is not completion authorization.
- [x] Enforced active-revision allow-list before unscoped visual provider
  ranking; added regression coverage and reran full backend tests (928 pass).
- [x] Added fail-closed Lance migration validation and verified an isolated
  21-document migration (0 invalid rows, physical revision columns, 768d).
- [ ] After RC/shadow gates and explicit approval, switch production LanceDB
  to the validated projection; until then legacy status is intentionally
  reported as migration-pending.
- [x] Replayed Gate C on the isolated migrated Lance projection: A0 passes all
  retrieval/latency thresholds; A1/B-full are non-inferior but add no gain.
- [ ] Run Sol HTTP/SSE agent regression and unseen shadow after valid 9router
  OAuth completion before any production LanceDB path switch.

# 2026-09-13 — public HTTP/SSE scope regression

- [x] Fixed generic arXiv identifier separator resolution (`1907.04152` ↔
  `1907_04152`) with corpus-wide ownership checks; public isolated sticky-focus
  smoke passed 2/2 through `file_qa`.
- [x] Full backend suite after the fix: **910 passed**.
- [ ] Expand HTTP/SSE regression to a stratified public manifest (table,
  figure/page, unanswerable and multi-document) before any promotion decision.

# 2026-09-13 — stratified public agent regression

- [x] Public isolated HTTP/SSE strata exercised table, figure/page, text,
  negative and cross-document turns. Four passed in the first batch; the
  cross-document turn had one empty-generation transient and passed on exact
  rerun with both documents covered. Keep transient telemetry visible.
- [ ] Add bounded generation retry/empty-stream classification to the
  regression runner; never convert an empty stream into a pass silently.

# 2026-09-13 — Gate C retrieval and agent policy

- [x] Rejected confounded `dev-v2.1`; evaluated balanced audited
  `dev-v2.2.model-attested` (60/60 qrels resolved, zero model calls).
- [x] A0 passed `0.97/0.91/1.00` Candidate/Evidence/Document, modality minimum
  `0.85`, multi-paper `1.00`, zero scope leak, P50/P95 `146/366 ms`.
- [x] Agent policy passed `0.98/0.88/1.00`, modality minimum `0.833`,
  multi-paper `1.00`, zero scope leak, P50/P95 `151/378 ms`.
- [x] A1/B-full add no gain; B-anchor is slower diagnostic. Dual-query remains
  disabled.
- [ ] Reduce remaining generic final-top-k/fusion losses before answer-quality
  or visual promotion; do not tune individual cases.

- [x] Completed the full 21-case isolated visual retrieval frontier after
  resume: T/V/T+V recall `0.4762/0.8571/1.0000`, fused P50/P95
  `367/873 ms`, zero failed rows and zero model calls. Retrieval checks pass;
  promotion is still false until answer quality is attached.
- [x] Completed paired Sol answer-quality on the same 21-case manifest:
  63/63 generations, zero failures. MMLong Aya accuracy `0.7219` and page
  recall `0.9091`; SPIQA Aya token-F1 `0.6148` versus full `0.6389` (diagnostic).
- [x] Added separate text-runtime-root wiring to reuse canonical projections
  without conflating visual index state; targeted evaluator tests remain green.
- [ ] Review r90 per-case answer regressions and run the required HTTP/SSE
  agent regression before any visual promotion decision.
- [ ] Add durable per-document preparation leases/checkpoints to reduce cold
  OCR/parser cost after host sleep; do not tune around one PDF.
- [x] Re-ran the full backend suite after the evaluator runtime-root change:
  **905 passed, 5 warnings**; backend/frontend/9router/Ollama health all 200.

# 2026-09-21 — remaining blockers and public-format evidence

- [ ] Refresh/re-authorize the 9router OAuth token. Live SSE reaches backend
  routing and graph, but generation returns `401 token_revoked`; gateway
  reachability is not completion authorization.
- [x] Public isolated retrieval-only sample completed 12/12 across multiple
  document profiles. Diagnostic aggregate is page/evidence/artifact recall
  `0.6667/0.6667/0.6667`, support recall `0.8611`, P50/max `115/220 ms`.
- [ ] Improve arbitrary-format artifact/page recall generically (parser,
  artifact projection and visual/page routing), then rerun a larger stratified
  sample. Do not tune to individual missed case IDs.
- [ ] After token refresh, run Sol HTTP/SSE regression and only then open the
  unseen shadow/RC gate. Production dual-query, visual flag and Lance path
  switch remain disabled.
- [x] Visual PDF rendering now checkpoints each committed page atomically;
  crash/sleep recovery can reuse the completed prefix. Visual tests 18/18 and
  backend suite 932/932 pass.
- [x] Fixed generic list/count visual attachment loss: additional ranked visual
  pages are retained within a bounded frontier instead of being discarded by
  text composition. Sol answer-quality replay is still required before any
  promotion claim.
- [x] Visual benchmark now records OCR/render/device configuration and rejects
  mismatched resume checkpoints. The no-OCR replay is diagnostic only and must
  not be compared with OCR-enabled reports.
- [x] Visual report merge/answer pairing now rejects legacy reports without the
  same configuration contract; full backend suite 932/932 remains green.
- [ ] Refresh/reauthorize the reachable 9router token before repeating any
  Sol answer-quality or HTTP/SSE regression; `/v1/models` reachability alone
  is insufficient and the current completion error is `token_revoked`.
- [x] Revalidated the local lifecycle matrix in
  `format-lifecycle-matrix-r2.json`; six valid-format retrieval checks and all
  lifecycle invariants pass, with corrupt input explicitly `partial`.
- [x] Replayed the current 21-case visual frontier in isolation with OCR and
  cached ColSmol runtime; T/V/T+V recall is `0.5238/0.8571/0.9286` and fused
  P50/P95 is `226.780/936.900 ms`.
- [ ] Investigate the remaining generic visual fusion/annotation alignment
  failures on a broader unseen sample; do not patch by case ID or paper name.

## 2026-09-22 — current Sol and compare blockers

- [x] Revalidated 9router completion authorization with direct Sol completion
  and live `/health`; the previous `token_revoked` checkpoint is stale.
- [x] Paired public Gate-E answer-quality run completed 21/21 with Sol;
  visual promotion remains off because per-case answer regressions remain.
- [x] Seven-turn HTTP/SSE agent E2E passed 7/7 with table/figure/image checks.
- [x] Added generic title-initialism, filename-fragment and comparison-table
  insufficiency identity coverage tests; the isolated LPMN/Wav2Small replay
  now passes.
- [ ] Fix the remaining contextual multi-paper follow-up (`MSF-SER` ->
  “hai bài đó”): retrieval covers both sources, but Sol's long comparison
  generation still sometimes fails answer-document coverage and falls back.
  Diagnose/replay the production prompt contract; do not relax validation or
  add paper-specific aliases.
- [ ] Re-run the full 20-case conversational suite after the compare fix; keep
  shadow-v3, visual promotion and production projection switch gated until
  this regression and the existing public answer-quality regressions are
  resolved.
- [x] Comparison sanitizer now removes only unsafe numeric table cells,
  preserves source identity and appends a document-bound number-free
  abstention. The MSF-SER follow-up group passes 4/4; the conservative full
  suite result is 19/20 with one transient ASPIRE/ViSEC generation miss.
# 2026-09-21 — printed-page provenance

- [x] Preserve printed-page aliases as non-canonical provenance in parser,
  chunk, FTS and Lance embedding context; physical page IDs remain stable.
- [ ] Re-index the isolated public visual frontier with the new page-label
  fingerprint and measure whether page-label queries recover physical pages;
  do not mutate production Lance until RC/shadow gates pass.
- [x] Re-ran the full lifecycle matrix after page-label fingerprinting;
  `retrieval_all_hit=true` and all lifecycle invariants pass in
  `format-lifecycle-matrix-page-label-r3.json`.
- [x] One-case visual replay confirms printed-page aliases recover the text
  target, but visual-only/fused ranking remains below top-10; keep visual
  promotion off and evaluate a broader unseen sample before changing fusion.
- [x] Broader 21-case replay after the per-document printed-page resolver
  removed the prior fused regression: T/V/T+V recall **0.6667/0.9286/1.0000**
  and no T+V regressions. Remaining blocker is paired Sol answer-quality,
  not visual retrieval recall; keep the flag off until that contract is run.
- [x] Final isolated lifecycle matrix and desktop build recheck passed; no
  production projection was switched.
- [x] Visual page-label resolution now has an explicit retired-revision
  regression; full backend suite is **936/936**.

## 2026-09-22 — current 9router and image-backed table follow-up

- [x] Live 9router recheck passes through the real backend: Sol is reachable,
  authorized and streaming; Ollama is used only for local embeddings.
- [x] Generic image-backed table handling is fixed across retrieval, page
  expansion, production visual attachment and the public evaluator. Shared
  artifact inventories are now weak context, not a replacement for a matching
  caption/OCR; selected canonical tables remain first during same-page
  expansion.
- [x] Full backend suite: **944/944**. Desktop build and Graphify sync pass.
- [ ] Gate E remains open: one isolated Table 3 replay now proves retrieval and
  image attachment, but its answer score is still low; broader public per-case
  regressions and the conservative 19/20 agent suite require further generic
  answer-quality work. Do not tune a single paper or enable visual/dual-query
  production flags.

## 2026-09-22 — table-image modality and visual relation contract

- [x] Image-backed tables stored by an adapter in the figures projection are
  normalized to `table` for identity, context facet reservation, vector
  payloads, visual attachment and typed evidence round-trips.
- [x] Effect/importance queries can retain two relevant visual pages, and the
  shared answer contract preserves table row/column relationships and names
  explicit baseline/without conditions.
- [x] Three-case cached Sol replay: **3/3** complete, zero fallback; external
  data table token-F1 improved **0.2857 -> 0.50** after canonical table
  normalization. This is a diagnostic improvement, not release promotion.
- [x] Full backend suite: **945/945**; desktop build and Graphify sync pass.
- [ ] Re-run the full paired 21-case answer-quality Gate E after the modality
  pass. Keep visual/dual-query flags off until per-case regressions and the
  conservative 19/20 conversational suite are resolved generically.

## 2026-09-22 — benchmark service boundary

- [x] Public Gate E now has an explicit `--visual-timeout-seconds` budget and
  aborts only the optional visual worker on timeout; this prevents image-only
  documents from making the entire evaluator appear hung.
- [ ] The current environment check observed a stale/listening-but-unresponsive
  9router port, with backend and Ollama also unavailable. Treat the run as
  provider-unavailable, not as a retrieval or answer-quality result. Recheck
  the three health contracts before resuming the public benchmark.

## 2026-09-22 — audited Gate C result

- [x] `dev-v2.2` retrieval-only rerun completed on the current local index with
  EmbeddingGemma: A0 passed all Gate-C thresholds (**0.97/0.90/1.00** recall
  for candidate/evidence/document; P50/P95 **121/343 ms**). A1 added nothing.
- [x] B-full produced no candidate gain; B-anchor improved evidence recall only
  slightly while adding latency and embedding calls. Keep dual-query disabled;
  do not tune the baseline or promote RRF from this result.
- [ ] 9router remains the only missing provider for the next step: paired Sol
  answer-quality Gate E and shadow-v3. Backend and Ollama are now serving.
- [x] The isolated arbitrary-format matrix was rerun with Ollama and passed:
  PDF/DOCX/MD/TXT retrieval, image/table projections, dedup/revision/rename
  invariants and corrupt-source partial isolation all held. This validates the
  local foundation, not visual answer quality.
- [x] ColSmol worker shutdown is now a hard subprocess boundary (no nested
  executor thread in the worker). Three-case visual smoke passed **3/3**;
  larger frontier documents can still be `visual_index` partial/timeouts and
  remain excluded from promotion. Backend suite: **947/947**.
