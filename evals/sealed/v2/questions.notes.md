# Sealed eval v2 — question set notes

Author: independent question author (subagent), 2026-10-16. Written only from the 18 files in `raw/`
and `manifest.json`; v1 material, backend/apps/evals/scripts and Aya's API were not consulted.
The two scanned PDFs were read by rendering pages to PNG and inspecting them visually.

## Files

- `questions.jsonl` — 92 turns in 81 conversations (78 single-turn + 3 multi-turn conversations of 4, 5 and 5 turns).
- Schema as specified. Extra conventions:
  - `format: "multi"` + `expected_documents` for every turn whose answer needs more than one file
    (`expected_document` is then `null`). The S6 library turns list every file that should be named.
  - The single off-topic turn (S7, 3rd turn of the first multi-turn conversation) has `format: null`,
    `expected_document: null`, `off_topic: true`, `expect_abstain: false`.

## Counts

| group | n | | language | n | | format | n |
|---|---|---|---|---|---|---|---|
| C1 cross-document comparison | 17 | | en | 47 | | pdf_native | 39 |
| C2 cross-language | 16 | | vi | 45 | | multi | 26 |
| S1 single-document fact | 8 | | | | | docx | 9 |
| S2 table number + computation/ranking | 8 | | | | | pdf_scanned | 7 |
| S3 figure/caption | 4 | | | | | slides_pdf | 4 |
| S4 formula | 3 (+2 formula turns inside S7) | | | | | md | 3 |
| S6 library-level | 5 | | | | | txt | 3 |
| S7 multi-turn | 14 (3 conversations) | | | | | null (off-topic) | 1 |
| S8 messy input | 5 | | | | | | |
| S9 false premise | 4 | | | | | | |
| S10 unanswerable (expect_abstain) | 8 | | | | | | |

`must_attach`: table 19, figure 4, formula 5, null 64. Every one of the 18 documents is the expected
document of at least two turns.

## Grading guidance

- Number formats: Vietnamese answers may use decimal commas and dot thousands separators (7,8% / 5.310.000);
  English answers the reverse. Treat 4,45% and 4.45% as equal. Units/rounding that keep the same value
  (e.g., 2,856.4 thousand = about 2.86 million) are fine.
- `must_mention` items are short, independently checkable facts; paraphrase is acceptable. Parenthetical
  labels such as "(English Wikipedia)" identify which source the number must be attributed to.
- `must_not_mention` lists wrong claims. A refutation of the claim (e.g., "it was not cut") is not a violation.
- S9 false premises: the answer must correct the premise, not answer it as posed.
- S10: `expect_abstain: true` means the answer must say the information is not in the documents. Giving the
  closest available fact (e.g., "only through 2028; 2028 = 7.3%") is good, inventing the missing value is a fail.
- C2 cross-language: the question language differs from the document language; the answer should be in the
  question's language and carry the document's numbers correctly.
- Off-topic turn: a brief general answer or a polite redirect are both acceptable; failure = citing a library
  document as a source for it. The next turn tests that the conversation returns to the survey document.
- Multi-turn conversations: later turns are deliberately vague ("And core?", "Còn bài kia thì sao?",
  "those two meetings", "nó"). The expected document per turn is set explicitly. The Fed conversation moves
  September → June (scan) → back to September; the text-to-SQL conversation moves DualSQL → ReToolSQL →
  both. Do not penalise an answer that also names the other paper in the last text-to-SQL turn — it is a comparison.

## Evidence conventions

- `pN` = 1-based PDF page index (the page counter of the file), not the printed folio. For the yearbook scan
  and the World Bank reports the printed folio is given in parentheses where it differs.
- docx files have no pages; evidence names the section heading or table caption (Biểu/Table N).
- The NOAA txt is cited by the issue date of the discussion block.

## Computed / derived values (verified arithmetic, not printed in the documents)

- C1 FDI turn: September-only disbursed FDI ≈ 21.07 − 17.25 = 3.82 bn USD (difference of two cumulative figures).
- C1 Bắc Ninh: 3,619,433 / 4,718.6 km² ≈ 767/km² (explains the Vietnamese-Wikipedia density; the English 1,132 is inconsistent).
- S2: Region I − IV wage gaps (1,610,000 VND/month; 7,700 VND/hour); DualSQL oracle gains (+5.4 EX, +3.5 REX —
  5.4 is also printed); ReToolSQL RL-gain gap 3.33 − 1.95 = 1.38; Fed 2027 GDP median vs longer run 2.4 − 2.0 = 0.4;
  RONI JAS 2026 − JAS 2015 = 0.2; ADB − WB gaps 0.4 / 0.3 pp.
- Fed June-scan values: medians read from the scanned Table 1 (all legible at 2.5x render); they match the
  "June projection" rows printed in the September release.

## Low-confidence labels

1. C1 "Đà Nẵng population" — the *reason* for the difference (average-year population vs another population
   count/date) is an inference; neither source explains it. Grade mainly on the two numbers.
2. C1 "ADB vs NOAA on El Niño" — ADB's slide 5 reproduces a NOAA-sourced RONI outlook chart; an answer that
   reads approximate values off that chart is not wrong. The key point is that ADB treats El Niño as an
   economic risk (0.44 pp / 0.91 pp scenario).
3. C1 "H1 2026 growth 8.2% (ADB) vs 8.3% (World Bank)" — both printed; the NSO Q3 report gives only Q1 8.15% and
   Q2 8.81%, no H1 figure. Do not require an explanation of the gap.
4. C1 "trade balance / FDI across the two NSO reports" — preliminary monthly data get revised (e.g., 20.46 − 19.42
   = 1.04 ≠ the 1.27 bn September surplus printed); the computed 3.82 bn September FDI is an approximation.
5. S6 "other documents mentioning minimum wage" — exhaustive text search over all native/docx/md/txt files and a
   visual read of all 33 yearbook pages. The yearbook mentions "mức sống tối thiểu" (minimum living standard,
   1,829.1 thousand VND), a different concept; mentioning it as such is not an error.
6. S6 "how many Vietnamese documents" — 6 per the manifest. The yearbook excerpt has bilingual table headings;
   accept an answer that flags it as bilingual as long as the count is 6.
7. S6 "which documents forecast Viet Nam 2026 growth" — the NSO Q3 report only quotes the 7.5–8.2% range from
   other organisations; omitting it is acceptable if the four must_mention values are present.
8. S8 "Table 4 of ReToolSQL" — the leaderboard is Table 3; credit the correct rank/EX even if the answer does not
   explicitly correct the table number (the table-number must_mention is a soft item).
9. S7 off-topic turn — what counts as acceptable is policy-dependent (general answer vs redirect).
10. C1 "challenging questions" — ReToolSQL labels its tiers Simple/Moderate/Challenging, DualSQL Easy/Moderate/Challenging;
    both use BIRD's annotation but different models and protocols.

## Verified-absent facts used by S10

- BIRD hidden-test score for DualSQL: not in the paper (only BIRD-dev and Spider).
- Fed funds median for 2030: SEP covers 2026–2029 + longer run; "2030" does not occur in the September SEP.
- 2027 minimum-wage increase: not in any document.
- World Bank 2029 forecast for Viet Nam: both WB reports stop at 2028.
- ReToolSQL training duration/GPU-hours: hardware and epochs only; no "hours"/"days" in the paper.
- Province-level unemployment (Hà Nội): only national rates in the NSO Q3 report; no province rates elsewhere.
- Niño-3.4 on 20 October 2026: latest material is the 13 October slides / 8 October discussion.
- ADB inflation forecast for 2028: ADB tables stop at 2027; "2028" does not occur in either ADB file.
