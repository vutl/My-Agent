# 11 — Failure ledger (generalisation status per mechanism)

One row per failure *mechanism* (not per case). A row is **verified** only
when its variant suite — written by a different model, on a library the fix
was not debugged on, with negative controls — passes on a live run, and the
regression suites do not drop. Generate a suite with
`python tools/harness/variants.py` (plan 10); record the result here.

Status: `open` (known, unfixed) · `fixed` (code + unit tests, not yet verified
live) · `verified` (variant suite passed) · `regressed`.

| # | Mechanism | Example (where seen) | Fix | Variant suite | Status |
|---|---|---|---|---|---|
| 1 | Vague document references (nó, bài đấy, bài kia, cái trước, cả hai, the other one) resolved by hand rules | v2-64: "Còn bài kia thì sao?" stayed on DualSQL | `e066150c` model router resolves from conversation_documents; rules are the fallback | `sealed/v1/variants/vague-ref.jsonl` (47 turns, 5 negative): 2026-10-18 34/47 → after `2543e8c3` 40/47 (85%), 0 fabricated, 5 false abstentions | fixed, 85% on variants |
| 2 | Guessed focus skipped its retrieval check (compared with this turn's own state) | v2-30: ReToolSQL question focused a decree via two shared syllables | `91d37dd4` | — | fixed |
| 3 | Document names not recognised: title head before ":", lower-case file acronyms, title acronyms that are app words | v2-30 ReToolSQL, v2-42 FOMC, s33 RAG | `91d37dd4`, `faeb7c50`, `59755a66` | — | fixed |
| 4 | Generic phrase pins the wrong edition/period of a report series | v2-20 "NSO Q3 report" → August report | `91d37dd4` period conflict, `faeb7c50` period disambiguation | — | fixed |
| 5 | Name matching 2–3 documents returns a canned clarification | v2-09, v2-02 "Tên sql khớp nhiều tài liệu" | `faeb7c50`, `c43f2361` small sets read together | — | fixed |
| 6 | Long documents (>160k chars) never reach whole-document context → false "not in the excerpts" | v2-05/06/07 World Bank, ADB | `faeb7c50` question-focused sections | — | fixed |
| 7 | Missing-information phrasings not detected → no whole-document re-answer | v2-63 "do not expose" | `faeb7c50` | — | fixed |
| 8 | Visual words ("slide", "national figure") trigger a canned visual refusal | v2-16/17/24 | `5de8cc37` depiction questions only | — | fixed |
| 9 | Answer language wrong (substring " bro" in "broken", accented names) | v2-11/25/26 | `7b6c053f`, `0905dce5` | — | fixed |
| 10 | Post-verifier deletes correct lines (sources cut before whole documents; years/model names as claims; removals of sourced lines) | s41 ResNet-18 row deleted, s40 rewritten | `59755a66`, `9cf11db8` | — | fixed |
| 11 | Number attributed to the wrong document survives verification (cross-chunk confabulation) | v2-07 ADB 8.2 vs World Bank 8.3 | `d32d3752` labelled windows + attribution rule | — | fixed |
| 12 | Scanned tables flattened by OCR (rows/columns lost) | v2-63 t3 SEP fed funds 3.8/3.4 | `36f6e7fb` layout-preserving OCR; needs re-ingest of scans | — | fixed (re-ingest pending) |
| 13 | OCR diacritics on Vietnamese scans ("Quàng Ninh", "Đà Nāng") | v2-44, v2-68 | Apple Vision page OCR on macOS (RapidOCR fallback), rows rebuilt from boxes | — | fixed (re-ingest pending) |
| 14 | Numbers in answers carry no per-claim citation (verifier must guess the source) | design gap (ReClaim / citation gate) | `5c961fb2` [S#] citations + cited-number check, `f6bf7959` UI chips persisted | — | fixed |
| 15 | A loose catalog match (title word, file-name phrase, description) decides scope without the router | v3-72 "Thông tư 83" → TT 12 titled "THÔNG TƯ"; v3-10/16 English vs Vietnamese dengue article; v3-19 | `072984ec` only file-name/identifier matches skip the router; its named documents replace a loose match | — | fixed |
| 16 | Tables on scanned pages stored empty (Docling finds the region, no cells without a text layer) | v3: 53/53 scanned tables empty; v3-31, v3-62 t4 TT 01 criteria | `78810da1` OCR of the table region, rows kept | — | fixed |
| 17 | "Show me" without a visual noun read as an image request → canned visual refusal | v3-66 "Cho tôi xem danh mục bệnh…" | `072984ec` only "cho tôi xem hình/ảnh/biểu đồ…" | — | fixed |
| 18 | Question about a document not in the library answered from memory as if from it (fabricated figures) | v3-36 NĐ 81/2021 tuition caps | `072984ec` router `unlisted_document`; chat prompt says it is missing, no figures as the document's | — | fixed |
| 19 | Ingest hangs on a vision call that never answers (read timeout resets per byte) | v3 ingest during a network drop | `0de489a6` wall-clock deadline + one retry, figure kept without summary | — | fixed |
| 21 | A vague reference ("bài kia", "the first paper") caught by a deterministic shortcut (structured request, resume) that keeps the active document | vague-ref var-01 t2/t3, var-04 t3, var-09 t4 (persistent) | `d53e122a` vague reference in a document conversation always reaches the router | vague-ref: 11/11 traced turns of var-01/04/09 | fixed |
| 20 | Retry after an interrupted ingest fails on chunk primary key (same revision id, leftover rows) | v3: 4 DOCX `UNIQUE constraint failed: chunks.id` | `0de489a6` leftover rows of a non-active revision cleared first | — | fixed |

## Runs

- 2026-10-18 (gpt-6.1-sol answers, gpt-6-luna router): acceptance regression
  20/20 structural; vague-ref variants 34/47 correct; v2 rerun2 (32 failing
  conversations + 8 controls, 50 turns) 26 C / 13 P / 11 I, 0 fabricated, 17
  false abstentions — 18 of 24 failures are retrieval (one side of a
  multi-document question "missing"); 3 turns crashed on Decimal overflow
  from OCR digit runs (fixed `ab7beba3`); comparison turns resolved by the
  catalog never read documents whole (fixed `e60df9a7`).
- 2026-10-18 round 2 (`2543e8c3`): variants 40/47; v2 subset 26 C / 12 P / 12 I
  (unchanged); new crash 'async generator raised StopIteration' from empty
  number readings (fixed `79786c4b`). Traces: whole-document re-answers never
  fired on English answers (typographic apostrophe in "don’t"), and the
  router read a slide deck instead of its report (fixed `fe291733`).
- 2026-10-19 round 3 (`fe291733`): variants 42/47 (89%), v2 subset 26/51.
- 2026-10-19 round 4 (`b04e6461`, judge Sonnet): v2 subset 34/51 (8 up, 0 down),
  0 fabricated, 7 false abstentions; acceptance 20/20. Merged v2 dev estimate
  75/92 = 81.5% (CI 72–88), 95.7% correct+partial, false abstention 7.6%.
  Then `c3aa4d81` router list beats rewrite entities (v2-07 now correct in trace).
- 2026-10-19 round 5 (`c3aa4d81`): v2 subset 33/50, merged 80.4%.
- 2026-10-19 round 6 (`94fc83d8`, per-document hybrid priority passages +
  sub-questions): v2 subset 36/51 (3 up, 0 down); **merged v2 dev estimate
  77/92 = 83.7% (CI 75–90), 94.6% correct+partial, 0 fabricated, 5 false
  abstentions (5.4%)**; acceptance 20/20. Remaining: World Bank May/October
  edition choice varies between runs (v2-05/06/58), DualSQL Table 7 (v2-03),
  library-wide 'which other documents mention X' (v2-59), false-premise
  correction detail (v2-72); ~8 partials ask for details the question did not.
- 2026-10-21 sealed v3 (one run, `81a4dec1`+): 71/87 = 81.6% (CI 72–88),
  1 fabricated, 12 false abstentions; rows 15–20 from its failures.
- 2026-10-21 v3 dev rerun (`072984ec`): 79/87 = 90.8% (CI 83–95), 1
  fabricated (v3-24), 4 false abstentions; 13 up / 5 down → `b16fabeb`.
- 2026-10-21 regression (`b16fabeb`): acceptance 60/60; vague-ref 41/47, 0
  fabricated → `d53e122a` (row 21).
