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
| 1 | Vague document references (nó, bài đấy, bài kia, cái trước, cả hai, the other one) resolved by hand rules | v2-64: "Còn bài kia thì sao?" stayed on DualSQL | `e066150c` model router resolves from conversation_documents; rules are the fallback | `sealed/v1/variants/vague-ref.jsonl` (47 turns, 5 negative): 2026-10-18 34 C / 1 P / 12 I, 0 fabricated, 11 false abstentions | fixed, not verified |
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

## Runs

- 2026-10-18 (gpt-6.1-sol answers, gpt-6-luna router): acceptance regression
  20/20 structural; vague-ref variants 34/47 correct; v2 rerun2 (32 failing
  conversations + 8 controls, 50 turns) 26 C / 13 P / 11 I, 0 fabricated, 17
  false abstentions — 18 of 24 failures are retrieval (one side of a
  multi-document question "missing"); 3 turns crashed on Decimal overflow
  from OCR digit runs (fixed `ab7beba3`); comparison turns resolved by the
  catalog never read documents whole (fixed `e60df9a7`).
