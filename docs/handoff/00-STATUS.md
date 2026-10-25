# 00 — Status snapshot (updated 2026-10-25)

## 2026-10-25 — P1.1 task foundation

- Added persistent tasks/events, guarded state transitions, plans/steps, budgets,
  usage and atomic leases, plus `/tasks` inspection/create/cancel/pause/resume.
- Additive migration links `tool_calls.task_id`; deleting a conversation unlinks
  its tasks while preserving tasks and events. SSE contract is documented only.
- Runner, planning and UI remain in P1.2/P1.3/P1.7.
- Validation: required backend suite **1753 passed, 3 skipped, 0 failed**;
  Graphify AST update and whitespace checks passed.
## 2026-10-25 — P1.5 capability awareness

- Tools declare capability groups; cached Vietnamese/English summaries feed
  chat and agent system prompts. Chat includes the always-present library;
  the agent sheet follows the enabled tools. RAG prompts remain unchanged.
- Router policy distinguishes casual action requests from questions about
  Aya's abilities; tool allowlists still gate the agent route.
- Review follow-up: the greeting shortcut accepts only complete social
  greetings; greeting-prefixed requests reach the router and keep allowlist gates.
- Further review: all personal-question and banter shortcuts now require a
  complete social message. Mixed vi/en social/action requests reach the router;
  fake-LLM regressions cover enabled tools and the disallowed-tool downgrade.
- Removed the conflicting blanket chat instruction against mentioning tools
  for capability/action questions. Head-agent live isolated-stack check pending.
- Required backend suite after further review: **1725 passed, 3 skipped, 0 failed**; Graphify AST
  update and diff whitespace check passed.

## 2026-10-25 — Local voice backend (Plan A)

- Optional `services/tts` CPU sidecar: VieNeu presets/approved Vietnamese clones,
  lazy VOICEVOX Japanese styles, central character credits and clone policy.
- Backend `/speech/*`, offline sidecar supervision, speech cleanup, cached
  Japanese translation through the configured router; chat/vision models unchanged.
- Follow-up audit: relative TTS paths survive the child's cwd change while
  preserving the venv interpreter symlink; fence cleanup handles incomplete
  openings, nested fences and inline-code prose. Offline flags are import-tested.
- Further audit (2026-10-25): silence fences inside lists/quotes, match inline
  backtick runs exactly, preserve nested Markdown link labels and remove URL
  targets with balanced/escaped parentheses. Code-only Japanese requests return
  204 before translation or synthesis; offline regressions added.
- Validation after further audit: backend **1626 passed, 3 skipped** (host denies nested Seatbelt);
  sidecar **41 passed**, zero failures. Required combined command exited 0.
- Setup and API contract: `services/tts/README.md`. Online model setup and real
  audio listening checks remain for the head agent; desktop integration is Plan B.

# RAG v1 RC1 accepted — 2026-10-22 (Claude, after Codex review)

- Release record: `12-RELEASE-RAG-V1.md`. Release SHA `3219b2bc`, rollback
  baseline `7fcc5bd9`; rollback drill on a copy of `data/` passed (backup,
  release, code-only rollback, restore + rollback).
- Final regression (`e7fec5ce`): acceptance 60/60 structural and **58/60
  content-correct** (2 partial, 0 fabricated); vague-ref 46/47.
- Codex's corrections recorded: sealed audits are samples (28–29 turns/set);
  acceptance 60/60 alone is structural, now also content-judged.
- Drill found that importing `app.main` ran the title refresh on the production
  DB (stamped a metadata marker, changed no title) — fixed in `3219b2bc`.
- Remaining issues moved to the maintenance backlog. **Released to production
  2026-10-22 20:28** (user approved): backup `data/backups/pre-rc1-20261022-202756`,
  health ok, 21/21 documents, 0 titles changed, smoke conversation correct.

# RAG RC — 2026-10-22 (Claude): gate met on two independent sealed sets

| Sealed set (one run each, questions by an independent agent) | Correct | Correct+partial | Fabricated | False abstention | Unanswerable declined |
|---|---|---|---|---|---|
| v3 (2026-10-19 code) | 71/87 = 81.6% (CI 72–88) | 88.5% | 1 | 13.8% | — |
| v4 (`1cb91b08`) | 81/95 = 85.3% (CI 77–91) | 94.7% | 0 | 4.2% | 11/12 |
| v5 (`fd814dbd`) | 90/104 = 86.5% (CI 79–92) | 94.2% | 0 | 3.8% | 12/13 |

- v4+v5 pooled: 171/199 = 85.9%. Gate (≥85%, 0 fabricated, ≤10% false
  abstention) met on both. Independent audits (28–29 turns each, raw pages):
  no false "correct", no missed fabrication; the judge is strict (2–5
  partials per set only miss points the question did not ask).
- Regression on `d1bb7747`: acceptance 60/60, vague-ref variants 45/47 (96%),
  0 fabricated, 0 false abstentions.
- After v5 (`8e65ba16`, `c6fc3884`): "the other one" with 2+ other documents always
  asks; another version of a listed document counts as unlisted; titles drop
  HTML. v5-11, v5-42, v5-29 t3, v5-23 correct on trace. 1510 tests.
- Open (v5, dev now): within-document retrieval misses (v5-08 visitor counts,
  v5-15 slide 16, v5-17 scanned appendix, v5-33 second rule), shared-number
  names ("Nghị định 85" = 85/2017 and 85/2024; two READMEs) answered for one
  only. Next phase per product goal: OpenClaw-style tools on the same
  router → evidence → answer → verify pipeline.

## 2026-10-22 (Claude) — v4 failures fixed; regressions green

- `c6458b1f`: unresolved document names ("bài MAE") and shared-name sets
  ("USDA" → two WASDE reports while the user means a 1905 bulletin) go through
  the router; "which of my files / reports / văn bản…" is library-wide; unit
  conversions stay chat. Traced live: v4-28, v4-30, v4-42, v4-21 correct; v3-76
  control unchanged. 1501 tests.
- Regression on `c6458b1f`: acceptance 60/60; **vague-ref variants 44/47
  (94%), 0 fabricated, 0 false abstentions** (was 41/47). Open: var-07 t2
  ("bài giảng kia" → wrong lecture), neg-02 t4 (3-way ambiguity should ask).
- `d1bb7747`: a vague reference that could mean several discussed documents
  makes Aya ask which one (router `ambiguous_referents`); a turn that switches
  documents tells the answer model which one it is about (var-07, neg-02 fixed
  on trace).
- `ae92f69f`: catalog titles reject TCVN3/VNI metadata, blank-date and gazette
  headers, chapter/section names; "THÔNG TƯ" titles take their subject line;
  existing libraries are re-titled once at startup (`title_refresh`, no re-parse).
- `0f267f61`: a restart no longer gives an interrupted ingest an empty active
  placeholder revision. 1507 tests.
- UI smoke test (Playwright, :5173 → :7790): answer streams with [S#] chips
  and figure cards, history and trace panel work.
- Sealed v5 being built by an independent agent for the final number.

## 2026-10-21 (Claude) — sealed v4: gate met out of sample

- **Sealed v4, one run on `1cb91b08` (20 new files, 8 Vietnamese, 3 scanned;
  95 turns written by an independent agent): 81/95 = 85.3% correct (CI 77–91),
  94.7% correct+partial, 0 fabricated, 4 false abstentions (4.2%), 11/12
  unanswerable turns abstained.** Gate (≥85%, 0, ≤10%) met on the point
  estimate; the crashed turn (empty gateway completion) counts as incorrect.
- Independent audit (28 turns, raw pages): no false "correct", every sampled
  number found in the documents; the judge is strict — 5 of 13 non-correct
  turns lose only must_mention points the question did not ask (audited
  estimate ≈ 90%). Official number stays 85.3%.
- Remaining failures: v4-24 (TT 75 annex count) and v4-28 (MAE Figure 7)
  false abstentions; v4-30 router said the library holds only WASDE (Bulletin 59
  missed); v4-42 t3 off-topic conversion refused (library-topic rule too
  eager); v4-21 library inventory missed two files; v4-16/45 omitted items.
- `0c5acfbb` chat() retries empty completions (v4-13 crash). v4 is a dev set now.
- Codex quota: vutlpro Plus limit hit 2026-10-21 (reset 21:29); uclab Team
  workspace is out of credits (owner must refill).

## 2026-10-21 (Claude) — sealed v3: first out-of-sample number

- **Sealed v3, one run (19 docs: native/scanned PDF, slides, DOCX, MD, TXT; 87
  turns, en/vi): 71/87 = 81.6% correct (CI 72–88), 88.5% correct+partial,
  1 fabricated number, 12 false abstentions (13.8%).** Gate (85% / 0 / ≤10%)
  not met. Judge Sonnet; head spot-check by an independent agent on 28 turns
  (32%) against the raw documents agreed 27–28/28. Scanned PDFs 9/9, DOCX
  10/15. Files: `data/retrieval_eval/sealed/v3/judged-aya-20261019.jsonl`.
- Failure causes (ledger rows 15–20): loose catalog name matches decided the
  scope without the router (a title "THÔNG TƯ" captured every "thông tư N";
  "the English article" stopped at the Vietnamese one); empty tables on scanned
  pages; "cho tôi xem" read as an image request; an unlisted decree answered
  from memory; ingest hung on a vision call and retries hit leftover rows.
- Fixed generically: `0de489a6`, `78810da1`, `072984ec`. v3 is a dev set
  from now on.
- **v3 dev rerun (same 87 turns, after the fixes): 79/87 = 90.8% (CI 83–95),
  95.4% correct+partial, 4 false abstentions, 1 fabricated** (v3-24: an
  unlisted paper answered from memory). 13 turns up, 5 down (v3-02 t4 is model
  variance, correct on trace; v3-38/73 generation/retrieval variance; v3-24 and
  v3-76 t3 routing) → `b16fabeb`: unlisted-document and library-topic questions
  search the library; both correct on trace, chat control unchanged. 1498 tests.
- Regression (`b16fabeb`): acceptance 60/60 turns; vague-ref variants 41/47
  (was 42; 0 fabricated). Their persistent failures (var-01 t2/t3 "bài kia",
  var-04 t3 "the first paper") were deterministic shortcuts keeping the active
  document → `d53e122a`: a vague reference in a document conversation always
  reaches the router; all 11 traced turns of var-01/04/09 correct.
- Sealed v4 built by an independent agent (20 files, 95 turns, colliding
  generic titles); run once on `d53e122a` (`sealed/v4/run_sealed_v4.sh`, :7794).

## 2026-10-19 (Claude) — RAG rounds on GPT-6 (dev estimates; v2 is no longer sealed)

- Aya answers on cx/gpt-6.1-sol, router cx/gpt-6-luna (user's choice); UI and
  eval runners no longer force gpt-5.6-sol.
- Six measured rounds (ledger `11-FAILURE-LEDGER.md`): vague-reference
  variants on a held-out library 34 → 40 → 42/47 (89%); sealed-v2 merged dev
  estimate 50% (sealed) → 81.5% → **83.7% (CI 75–90), 0 fabricated, 5.4% false
  abstention**; acceptance 20/20 throughout.
- Generic fixes: model router resolves vague references; named 2–3 document
  sets and one-sided comparisons cover every document; router picks beat weak
  rewrite entities; layout-preserving + Apple Vision OCR for scans; per-claim
  [S#] citations with UI chips and a cited-number check; question-focused
  sections led by per-document hybrid search for each router sub-question;
  whole-document re-answers on typographic apostrophes and with companion
  documents; crash fixes (Decimal overflow, StopIteration).
- Offline end-to-end stream tests (`tests/test_agent_stream_e2e.py`); 1489 tests.
- Next: sealed v3 (independent corpus and questions, one run) for an
  out-of-sample number.

## 2026-10-18 (Claude) — v2 after fixes (dev estimate) and agent harness

- Rerun of the 40 failing conversations + 10 controls on the fixed code,
  judged by gpt-6-astra through the new harness (`judged-rerun1-20261017.jsonl`),
  merged with the 31 untouched turns: **60/92 = 65.2% (CI 55–74%)**,
  82.6% correct+partial (sealed baseline 50.0% / 70.7%); 13 false abstentions.
  23 turns moved up to correct. Controls 8 C / 2 P (astra is stricter than the
  Claude judges on minor omissions; head spot-check: no real regression).
- **2 numeric errors (gate is 0):** v2-07 attributes the World Bank's 8.3% H1
  growth to ADB (8.2%) and the post-verifier did not catch it; v2-63 t3 reads a
  scanned SEP table whose OCR flattened rows and columns ("3.8 | 3.4 | 3.1 |
  3.6 …"), so year/edition cannot be recovered. Next: structured table OCR for
  scans; verifier must check attribution across documents.
- Agent harness (`tools/harness`, plan 10): Codex accounts vutlpro + uclab
  (browser login; Team workspace blocks device auth), shared skills in
  `.agents/skills`, context7/playwright MCP.

## 2026-10-16 (Claude) — sealed v2 (independent corpus, 18 docs, 92 turns)

- Run once with cx/gpt-5.6-sol on code `5d882208`, judged by two
  independent Claude sub-agents (`sealed/v2/judged-aya-sol-20261016.jsonl`):
  **46/92 correct = 50.0% (CI 40–60%)**, 70.7% correct+partial, **0
  fabricated numbers**, 28 false abstentions (30%). Unanswerable S10 7/8.
  Weakest: cross-document C1 3/17, C2 6/16, multi-turn S7 5/14, scanned
  PDFs 0/7, multi-document 7/26. en 22/47, vi 24/45.
- Generic causes found and fixed (`91d37dd4` … `7ed0bd98`): guessed focus
  bypassed its retrieval check (compared with this turn's own state); title
  heads ("ReToolSQL:") and lower-case file acronyms ("fomc") were not names;
  generic filename phrases pinned the wrong edition (period conflict now
  defers to the router; editions resolved by named period; 2–3 matching
  documents read together instead of a canned clarification); long reports
  (>160k chars) never reached whole-document context (now question-focused
  sections); visual words ("slide", "national figure") triggered a canned
  refusal; language detection (" bro" in "broken", accented place names);
  "bài kia" / "the other paper". The set is regression-only from now on.

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
- **Plan A (2026-10-16):** router query_translation as a retrieval variant
  (s21 Vietnamese→English scan fixed), short-title aliases (s62 fixed),
  compared documents read whole from the first answer (`ea933885`), OCR
  budget 64 pages (`a7b4eaac`, found while ingesting v2: a 33-page scan had
  every third page read). Rerun4 of open v1 cases: 6 C / 2 P / 3 I, 0
  fabricated. Known open: untitled scans lose description matches (s22,
  CPI scan vs labour release); comparability wording (s34).
- **9router:** both cx accounts work again (re-login 2026-10-16 15:03).
- **Comparisons fixed (2026-10-16, `59755a66`, `9cf11db8`, `7b6c053f`):**
  v1 S5 cases s33/s34/s40/s41 all correct on rerun7 (were 1 P / 3 I).
  Generic causes: the post-verifier ranked only the first 12 sources, so
  whole documents (appended last) never reached it and correct lines were
  deleted; years drove the whole-document windows; "RAG" title acronym was
  blocked as an app word; English questions had no explicit answer language.
- **After fixes (dev estimate, set no longer sealed):** rerun3 of the 29
  failing/control conversations judged 25 C / 8 P / 6 I, 0 fabricated;
  merged with the 50 untouched (originally correct) turns: **84.3% correct
  (CI 75–90%), 93.3% correct+partial, 0 fabricated, 6.7% false
  abstention** (sealed baseline 75.3% / 85.4% / 1 / 13.5%). Remaining:
  cross-document comparisons S5 4/9 (s33, s34, s40, s41), Vietnamese
  question → English scanned doc (s21), CPI scan vs labour release (s22),
  sticky focus on a new document (s62). Acceptance c03–c08 20/20.
- **Embedding bake-off** (sub-agent C, `sealed/v1/embed_bakeoff/report.md`,
  retrieval-only, hybrid top-10, 60 sealed turns): embeddinggemma:300m
  hit@1 0.850 / hit@5 0.950 (vi 0.806, en 0.897), index 145 s, query 42 ms;
  qwen3-embedding:0.6b 0.850 / 0.950; bge-m3 0.800 / 0.933; qwen3-embedding:4b
  0.850 / 0.983 at 21× index time and 9× query latency. **Keep embeddinggemma.**
  Pure vector beats hybrid on doc hit@1 (0.917 vs 0.850); measured and
  rejected: FTS IDF term pruning (vi hit@1 0.806 → 0.774) and lower FTS RRF
  weight (no gain). The gap likely comes from card routing / channel mix,
  not the embedding model. Open: vi question → en scanned doc (s21) misses
  for every model (cross-lingual).
- Reruns of the 29 failing/control conversations: rerun1 21 C / 7 P / 11 I
  (figures, s09, s62 fixed; controls kept); a "switch to top-ranked doc"
  rule hurt s22/s51 (dense Q2 report dominates rankings) and a guessed-focus
  check dropped focus on vague follow-ups (acceptance G4 12 → 8) — both
  fixed (`00dd9d65`, `11fc722d`); c03–c08 back to 20/20.
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

---
Older entries (before 2026-10-13): `docs/archive/handoff-history/00-STATUS-before-2026-10-13.md`.
