# Sealed eval v1: question notes

Author: independent question author (subagent), 2026-10-15. Source of truth is `questions.jsonl` (89 turns, 75 conversations). I wrote it only from the 19 files in `raw/` and from `manifest.json`. I did not read or call Aya code, endpoints, or 9router.

## Layout

| Group | Conversations | Turns | What it tests |
| --- | --- | --- | --- |
| S1 single-doc facts | s01–s12 | 12 | definitions / methods / findings, one document each |
| S2 table numbers | s13–s24 | 12 | exact cell values, sums (Q2 Biểu 7, WEN bank table), differences (FinBERT, LitM Table 1, BEA estimates), rankings (CheXNet Table 2, Census Table 2, CPI footnote) |
| S3 figures/captions | s25–s29 | 5 | only what the caption or surrounding text states |
| S4 formulas | s30–s32 | 3 | ResNet Eq. 1–2, RAG-Sequence/RAG-Token, CheXNet weighted BCE |
| S5 cross-document | s33–s41 | 9 | related pairs from manifest. 5 of them are "comparison not valid / sources disagree" cases (RAG EM vs LitM accuracy, FinBERT vs ClimateBERT F1, full-year vs 6-month exports, BEA SAAR vs VN y/y, EN vs VI Wikipedia 2024) |
| S6 library | s42–s46 | 5 | count of VI docs, healthcare docs, US-government releases, papers NOT mentioning ImageNet, formats/count |
| S7 multi-turn | s47 (vi, CPI scan, 5), s48 (en, ResNet, 5), s49 (vi, Labour Code, 5), s50 (en, Census scan, 3) | 18 | vague follow-ups ("cái đó", "that one", "bảng kia"/"that table"), one off-topic turn in s47–s49, then a return to the earlier document |
| S8 messy input | s51–s57 | 7 | no diacritics, typos in paper/tool names, wrong table numbers (CheXNet "Table 5", Q4 "Biểu 9"), FiQA asked with the wrong metric |
| S9 false premise | s58–s63 | 6 | the user states a wrong number or claim, and the answer must correct it |
| S10 unanswerable | s64–s75 | 12 | `expect_abstain: true`. The info is absent from the whole corpus. `must_not_mention` lists plausible invented values |

Language split: 45 vi / 44 en. Three vi messages have no diacritics (s51, s72, s73). Several vi questions ask about English papers.
Format split (`format` field): pdf_native 35, pdf_scanned 15, docx 10, md 11, multi 14, slides_pdf 3, txt 1. Every one of the 19 files is an expected document at least once. The scanned PDFs get the most weight: the CPI scan is expected in 8 turns and the Census scan in 6. That count includes S2/S7 table reads from the scanned Census Tables 1–3 and the CPI footnote.

## Conventions / interpretation choices

- **Multi-document turns:** `format: "multi"`, `expected_document: null`, plus `expected_documents`. S6 and S5 use this.
- **Off-topic turns** (s47 t3, s48 t4, s49 t4): `expected_document: null`, `off_topic: true`, `expect_abstain: false`. A general-knowledge answer or a polite "not in your documents" are both acceptable. `must_not_mention` holds terms from the conversation's document, so it catches a model that drags the paper into the off-topic answer. In these turns `format` is copied from the conversation's main document. The schema has no "none" value.
- **Abstentions (S10):** `expected_document` is set when the user names or clearly targets one document, so retrieval can still be checked. A correct answer says the info is absent and may cite the nearest real fact given in `must_mention` (for example, the release date of the next report).
- **Numbers:** Vietnamese items use the source's own formatting (decimal comma, dot thousands: `4,52%`, `166.896 triệu USD`). Judges should accept equivalent formatting (`4.52%`) and reasonable rounding (166,9 tỷ USD).
- **must_mention:** items are short facts to be judged on meaning, not exact strings. For example, "not higher than Radiologist 4" can be conveyed by giving both numbers.

## How the facts were checked

1. **Native PDF, DOCX, MD, TXT files:** I extracted the text with pymupdf. For DOCX I walked the body XML in reading order, keeping tables in place. I then read the relevant sections myself and took every number in a reference answer from the extracted text or table cells. Page numbers in `evidence` are PDF page indices. For RFC 8259 the evidence gives section numbers. For DOCX it gives section headings and Biểu/Table numbers.
2. **Scanned PDFs:** I rendered all 7 Census pages and all 6 CPI pages to PNG (110 dpi) and read every page. I re-rendered every value used in an answer at 150–200 dpi as zoomed crops and re-read it: Census Tables 1–3, the CPI p.3 y/y paragraph, the gold and USD paragraphs, core inflation, and footnote 1.
3. **Computed values:** I computed these in Python from the verified cells:
   - Q2 Biểu 7 sum = 166.896 (62,6%, which matches the text)
   - Q4 Biểu 7 sum = 333.372 (70,2%, which matches the text; used only as a must_not_mention distractor)
   - WEN bank capital = US$7.74 bn / 180,254.3 bn VND
   - LitM oracle minus closed-book gaps (MPT +50.4 largest)
   - FinBERT − ULMFit = 0.04
   - ResNet-34 plain − residual = 3.51
   - Q2/2026 − Q2/2025 GDP = 0.23 pp
   - Census gasoline adjusted 62,303 / 51,506 = +21.0%, which matches Table 2
4. **Absence checks for S10 and S6-"NOT":** I ran case-insensitive regex searches over the extracted text of all 17 text-bearing files. I covered the 2 scanned files by reading every page. Searches run:
   - `lương tối thiểu vùng`, `minimum wage`, `4.960`, `4.680`
   - `nghỉ hưu`, `tuổi hưu`, `retire`
   - `carbon`, `CO2` (FinBERT has none)
   - `Gemini`, `CheXpert`, `GRDP`
   - `Hà Nội` (in the labour release it appears only in the dateline)
   - `parameters` (ResNet gives no ImageNet-model parameter count)
   - `imagenet` (among the papers, hits only in ResNet and CheXNet)

   Note: the system `grep` in this environment (ugrep alias) silently returned nothing for some PDF text files. All absence checks were redone with a Python regex search.

## Low-confidence or judgement-dependent labels

- **s46 (formats/count):** 12 PDF / 4 DOCX / 2 MD / 1 TXT follows `manifest.json`. A model that reports "8 PDFs + 2 scans + 2 slide decks" is also correct. Judge the total of 19 and the breakdown leniently.
- **s45 (papers not mentioning ImageNet):** this assumes "research papers" means the six arXiv papers. The check covered the full text, including references. If Aya counts the MIT slides as papers, Lecture 13 and Lecture 24 do mention ImageNet, so they would correctly be excluded anyway.
- **s37 (GDP Q2/2026 vs Q2/2025):** the 8,16% for Q2/2025 comes from the Q4/2025 report. The Q2/2026 report does not restate Q2/2025, so a later revision is possible in the real world, but within the corpus 8,16% is the only value.
- **s40 (EN vs VI Wikipedia 2024):** the two tables genuinely disagree (7.1 / 476.4 vs 6,0 / 470). The VI Q4/2025 NSO report gives 2024 nominal GDP as 476 tỷ USD and does not give 2024 real growth. A model citing the NSO 476 is acceptable extra context.
- **s50 t3 (median CV):** department stores = 0.0 is odd but is what the scan prints (verified at 200 dpi). Building materials = 3.7.
- **s50 t2:** "categories that fell from July" refers to the first column of Census Table 2. The July (preliminary) columns contain other negatives, which do not count here.
- **s22 (CPI fuel footnote):** the August value prints as "15,8%" (one decimal), as shown in the scan.
- **s48 t3:** "the 152-layer one" is ambiguous between Table 3 (10-crop, 5.71 top-5) and Table 4 (single-model, 4.49). The question says "single-model", so 4.49 is required. Also mentioning 5.71 is fine.
- **s33 (RAG vs LitM on more documents):** the reference says "partly agree" because the setups differ. Accept any answer that conveys both findings correctly.
- **s71 (Gemini in LitM):** LitM also evaluates GPT-4 (subset) and Llama-2 (appendix). Do not penalise an answer that lists those as the extra models.
- **S10 "nearest fact" hints** (next-release dates, 11.3 B FLOPs, K80 GPU): these are optional context and appear in `must_mention` only where they help distinguish a grounded refusal from a generic one.
