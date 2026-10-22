# Sealed eval v3: question set notes

Written by an independent question author (subagent) on 2026-10-19. The only sources were the 19 files in `raw/`, `manifest.json` and `CORPUS_NOTES.md`. The author did not consult the v1/v2 question files, backend/apps/scripts/tools/evals, or Aya itself. The schema conventions were taken from `../v2/questions.notes.md`.

Every fact in a reference answer was checked against the document text: PyMuPDF for native PDFs, python-docx for DOCX, and plain text for MD/TXT. The three scanned PDFs were rendered to PNG and read visually. The pages used were STEO-Sep pp3–6 and p31, COE-2023 pp8–12 and pp31/38/39, and TT 01/2024 pp1, 4–11 and 20–22. Slide 2 and slide 5 of the EIA deck are image-only, and both were read visually.

## Files

- `questions.jsonl`: 87 turns in 76 conversations. That is 72 single-turn conversations plus 4 multi-turn conversations of 4, 3, 4 and 4 turns (v3-02, v3-22, v3-62, v3-76).
- The schema is as specified. Extra conventions:
  - Any turn that needs more than one file has `format: "multi"`, the files in `expected_documents`, and `expected_document: null`. The S6 turns list every file that should be named.
  - The one off-topic turn (v3-02 turn 3) has `format: null`, both expected-document fields set to null, `off_topic: true` and `expect_abstain: false`.
  - `must_attach` is set only where the user explicitly asks to see something. That gives `figure` once (v3 COE-2024 Figure 12) and `table` twice (the NĐ 238 vocational tuition table and the TT 12 prenatal list). Every other turn has null.
- Conversation order was shuffled with a fixed seed, so the groups are interleaved.

## Counts

| group | en | vi | total |
|---|---|---|---|
| S1 single-document fact | 5 | 4 | 9 |
| S2 table number + computation/ranking | 6 | 4 | 10 |
| S3 figure/caption | 3 | 2 | 5 |
| S4 formula | 2 | 2 | 4 |
| C1 cross-document (pairs 0, 2, 3, 4, 5) | 7 | 4 | 11 |
| C2 cross-language (incl. dengue pair 6) | 5 | 5 | 10 |
| S6 library-level | 2 | 2 | 4 |
| S7 multi-turn (4 conversations) | 7 | 8 | 15 |
| S8 messy input | 2 | 4 | 6 |
| S9 false premise | 2 | 2 | 4 |
| S10 unanswerable (`expect_abstain: true`) | 4 | 5 | 9 |
| **total** | **45** | **42** | **87** |

Formats: pdf_native 33, multi 19, docx 15, pdf_scanned 9, md 5, txt 3, slides_pdf 2, null 1. Every one of the 19 files is the expected document, or one of the expected documents, of at least 3 turns. The scanned files appear as follows:

- STEO Sep: 8 turns (2 single, plus 6 multi or S7).
- TT 01/2024: 6 turns.
- COE 2023: 5 turns.

Several of these turns ask for numbers that sit only in scanned tables or figures: STEO-Sep Table 1 (shut-ins), COE-2023 Figures 12 and 18, and the TT 01 threshold and coefficient tables.

C2 counts any turn where the question language differs from the language of the document. Some turns in S2, S3, S7, S8, S9 and S10 are also cross-language (for example, Vietnamese questions on OpenVLA and COE-2023), but they are counted under their main group.

## Grading guidance

- Number formats: Vietnamese answers may use decimal commas and dot thousands separators (3.500; 71,3%), and English answers the reverse. Values in the reference are given in the style of the question language, and either style counts as correct. Units that keep the same value are fine, for example 3.500 nghìn đồng = 3,5 triệu đồng.
- `must_mention` items are short, independently checkable facts, and paraphrase is acceptable. Labels in parentheses, such as "(4 năm)" or "Sep:", say which source or category the number belongs to.
- `must_not_mention` lists wrong claims. An answer that refutes the claim does not violate it.
- S9: the answer must correct the false premise. Answering the question as posed fails.
- S10: the answer must say that the requested information is not in the documents. Giving the closest available fact is good, for example "RT-1-X got 33.3%; RT-1 itself is not evaluated", or the Wikipedia catalog masses for GW170817 attributed to Wikipedia. Inventing the missing value fails.
- Off-topic turn (v3-02 t3): either a short general answer (1 Mpc ≈ 3.26 million light-years) or a polite redirect is acceptable. Citing a library document as the source fails.
- Multi-turn vague references, resolved:
  - v3-02: "bản năm trước" = COE 2023. "bài đấy" in t4 = COE 2023, the last document discussed before the detour. The 3,542 institutions figure exists only in the 2023 report; the 2024 report gives 3,633 for 2022–23 on a different institution set.
  - v3-22: "the 7B model that was benchmarked against it" = OpenVLA. "the first one" = Octo.
  - v3-62: "văn bản cũ đó" = TT 01/2024. "hai thông tư lúc nãy" = TT 01/2024 and TT 83/2026.
  - v3-76: "And in California?" = CAISO in the October STEO. "the edition before that one" = the September STEO (scan).

## Evidence conventions

- `pN` is the 1-based PDF page index, not the printed folio. For reference, the printed folios relate to the indices like this:
  - NĐ 238: Công báo page = index + 2.
  - TT 01/2024: Công báo page = index + 35.
  - COE 2024: printed page = index − 8.
  - COE 2023: printed page = index − 9.
  - STEO: printed page = index − 1.
- DOCX files have no pages. Evidence names the article (Điều/khoản), the section heading, or the table position in document order. For TT 83, "Table 6" is the 33×4 threshold table. For TT 12, Phụ lục I is the 74-row table and Phụ lục II A is the 4×3 table.
- Some single-document turns cite several sections of the same file.

## Computed values (verified arithmetic; not printed in the documents)

- STEO-Oct Table 1: Saudi share of September shut-ins = 1,700 / 4,770 ≈ 35.6%.
- STEO Sep vs Oct: end-October gas storage 3,969 − 3,850 = 119 Bcf.
- OpenVLA Table 2: int4 saves 16.8 − 7.0 = 9.8 GB (≈58%). LIBERO-Goal gap 84.6 − 79.2 = 5.4.
- Octo Table I: Berkeley Coke lead 100 − 20 = 80 points.
- NĐ 238: doctoral ceiling for Dược học/y học in 2026-27 = 3.500 × 2,5 = 8.750 nghìn đồng/tháng. The master's figure, 3.500 × 1,5 = 5.250, is mentioned but not required.
- COE 2024 Figure 19: net-price gap $24,200 − $9,700 = $14,500.
- COE 2023 Figure 18: 4-year for-profit 612 − 278 = 334 fewer institutions.
- TT 12 Phụ lục I group sizes were counted from the STT ranges: Nhóm I 1–7 (7), Nhóm II 8–38 (31), Nhóm III 39–70 (32), for a total of 70.
- OpenVLA 7B vs Octo-Base 93M (≈75×) is mentioned in no reference. The v3-22 t3 return turn asks only for Octo's 72%.

## Ambiguities and low-confidence labels

1. **STEO-Oct "Notable forecast changes" table:** the Brent row is mislabelled "dollars per gallon" in the source. All references use $/b, as the narrative does. Do not penalise an answer that notes the mislabel.
2. **Shut-in windows differ between editions:** STEO-Sep's first column is "Mar–June-26 average" and STEO-Oct's is "Mar–Jul-26 average". The C1 shut-in question compares only the August column, which both tables report.
3. **August STEO (S10):** the September STEO says its 2H26 Brent (~$90/b) is "$8/b higher than in last month's STEO" and plots the August path. An answer that infers "about $82/b for 2H26 in August" is a reasonable inference, not a fabrication. The annual 2026 August value is still absent.
4. **COE institution counts:** the 2023 report counts institutions "with first-year undergraduates" (3,542 in 2021-22). The 2024 report notes that it uses a slightly different institution set (3,633 in 2022-23). The v3-02 t4 turn targets the 2023 figure only.
5. **COE ACGR footnotes:** the imputed states differ between the two reports (Illinois and Texas in 2023; New Mexico and Oklahoma in 2024). They are not required.
6. **Dengue vintages:**
   - The Vietnamese article's infobox also says "390 million per year" (copied from English Wikipedia), while its body cites WHO's 50–100 million cases a year. The C2 dengue-pair reference requires the body figure and 390 million (EN, 2013). An answer that also mentions the VI infobox value is fine.
   - The EN article is internally inconsistent about Dengvaxia's availability year (2016 in the lead, 2015 in the Vaccine section). No question uses that fact.
7. **GW150914 chirp mass:** the paper's ≈30 M☉ is the detector-frame value from f and ḟ. The Wikipedia table gives 28.6 M☉, which is source-frame and from a later catalog. The S4 question asks for the paper's value only.
8. **GW151226 localization:** the paper gives ~1,400 deg² (online, within 3 min) and 850 deg² (refined). Wikipedia gives 1,033 deg². Either paper number may be cited, but the refined 850 should appear.
9. **TT 83 vs TT 01 graduation criterion 5.3:** the thresholds are identical (≥ 60% / ≥ 40%). The change is in the time window (≤ 2 years late vs ≤ 1.5× the standard duration). An answer saying "no change" without noting the window is incomplete.
10. **TT 01 vs TT 83 4.1:** TT 01 requires 0–30% and TT 83 requires ≥ 0. The TT 01 threshold table prints this as "≥ 0 và ≤ 30%".
11. **TT 01 field coefficients:** the TT 01 field-coefficient table (scan pp21–22) is identical to TT 83 Table 7. No question asks about a change there.
12. **README vs paper training time:** the README gives 8 h (Octo-S) and 14 h (Octo-B) on a TPUv4-128 pod. The paper gives only the ViT-B run (300k steps, 14 h). The S1 README question is graded on the README.
13. **The S6 "Octo" question:** the OpenVLA paper counts as a document about Octo, because it compares against Octo, reuses Octo's data-mixture weights, and fine-tunes Octo as a baseline. It is listed as required.

## Verified-absent facts used by S10

- The EIA slide deck has no forecast values: no Brent, price or production numbers anywhere in its 13 slides.
- NĐ 81/2021 and NĐ 97/2023 appear only as replaced decrees in NĐ 238 Điều 29. No 2023-24 tuition ceilings exist in any file.
- The August 2026 STEO is not in the library. No annual August Brent figure is printed in the September or October editions.
- RT-1 (as opposed to RT-1-X) results: OpenVLA and Octo evaluate RT-1-X only. The RT-1 paper is absent.
- 2026 assessment results under TT 83: TT 83 gives only procedures and deadlines (31/5, 30/6, first data year 2026).
- The GW170817 discovery paper is absent. GW170817 exists only as a Wikipedia table row.
- There is no Condition of Education 2025. COE 2024's latest undergraduate data is fall 2022.
- Vietnam-specific dengue case counts for 2023 appear in neither dengue article. The EN article mentions Vietnam only as the 2013 host of an Anti-Dengue Day event.
- Screening costs are absent from TT 12/2026, which gives only a principle that costs should be affordable or covered by the budget or insurance (Điều 3.3.b).
