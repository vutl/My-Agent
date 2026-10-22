# Sealed eval v5: question set notes

Written by an independent question author (subagent) on 2026-10-22. Inputs: the 20 files in `raw/`, `manifest.json`, `CORPUS_NOTES.md`, and the v4 notes/schema (format and process only). The author did not read v1-v4 questions or answers, did not read Aya's code, scripts or docs, and did not run Aya. Reference answers were written from the document text (PyMuPDF text for native PDFs and for the native originals of the two rasterized scans, python-docx for DOCX, plain text for MD/TXT). The one naturally scanned file (the Government programme under Nghị quyết 263) has no text layer, so it was read visually from rendered pages.

## Files

- `questions.jsonl`: 104 turns in 46 conversations (7 single-turn, 22 of 2 turns, 16 of 3 turns, 1 of 5 turns). Conversation order was shuffled with a fixed seed (20261022).
- Schema as in v4: `conversation, turn, message, language, group, format, expected_document, expected_documents, reference_answer, evidence, must_mention, must_not_mention, expect_abstain, must_attach` (+ `off_topic: true` on the three general-knowledge detour turns).
  - A turn that needs more than one file has `format: "multi"`, the files in `expected_documents`, and `expected_document: null`. Library-level (S6) turns list every file that should be named.
  - The detours (v5-22 t2 mph to km/h, v5-39 t2 m3 to litres, v5-42 t2 "grep") and the S10 library question with no matching file (v5-01 t2) have `format: null` and no expected document; detours have `expect_abstain: false`. `format` is also null for "S10 with nearest document" turns that name no single file.
  - `must_attach` is `figure` once (v5-15 t2, FHWA slide 16, when the user asks to see the slide).
  - `group` is assigned per turn: a turn keeps its intrinsic group when it is self-contained, and is labelled S7 when it depends on dialogue context (vague reference, follow-up, detour, return). Several multi-turn conversations deliberately chain independent questions on one or two documents (so group labels vary inside a conversation) to test sticky focus and document switching.
- Evidence conventions: `pN` is the 1-based PDF page index (for the three papers, NHTSA notes and the 2022 scan it equals the printed number except where noted in CORPUS_NOTES); slides are `slide N`; DOCX/MD/TXT cite sections or tables. Wikipedia tables are numbered in document order.

## Counts

| group | en | vi | total |
|---|---|---|---|
| S1 single-document fact | 11 | 6 | 17 |
| S2 table number + computation | 6 | 4 | 10 |
| S3 figure/caption/slide image | 4 | 1 | 5 |
| S4 formula | 3 | 0 | 3 |
| C1 cross-document (editions, paper/README, EN/VI) | 5 | 2 | 7 |
| C2 cross-language (question and document language differ; main point is the crossing) | 5 | 5 | 10 |
| S6 library-level | 2 | 2 | 4 |
| S7 multi-turn dependent (follow-up, detour, return) | 12 | 10 | 22 |
| S8 messy input (typos, no diacritics, ambiguous reference) | 3 | 4 | 7 |
| S9 false premise | 4 | 2 | 6 |
| S10 unanswerable (`expect_abstain: true`) | 7 | 6 | 13 |
| **total** | **62** | **42** | **104** |

13 of 104 turns (12.5%) expect an abstention. Formats (of the expected document): pdf_native 39, multi 18, pdf_scanned 12, docx 10, slides_pdf 9, md 7, txt 5, null 4. Turns whose question language differs from the language of every expected document: 17 (C2 plus some S2, S7, S10 and C1 turns, for example Vietnamese questions on the English NHTSA, FHWA, WeNet and BESSIE files, English questions on the Vietnamese laws, tables and programme).

Documents as the expected document of at least 3 turns each: NHTSA 2023 12, FHWA slides 12, VBHN 50 (NĐ 80/2014) 11, WeNet paper 10, VBHN 55 (road-order law) 9, NHTSA 2022 (scan) 7, Nghị quyết 263 programme (scan) 7, MatGL paper 6, MatGL README 5, ripgrep guide 5, FB 43 5, SS comparison docx 5, BESSIE 4, FB 1227 4, WeNet README 4, CT 4, CP (scan) 4, VI Hạ Long 4, EN water 4, EN Hạ Long 3. The three scanned PDFs appear in 17 turns, several of which need numbers that exist only in scanned tables (NHTSA 2022 Tables 4/6/9, Annex I rows, the 13-row target table of the programme).

## Grading guidance

- Number formats: Vietnamese references use decimal commas and dots as thousands separators (7.314; 2,5 triệu); English ones use commas for thousands. Either style counts.
- `must_mention` items are short checkable facts; paraphrase is acceptable. `must_not_mention` lists wrong claims; an answer that refutes the claim does not violate it.
- S9: the answer must correct the false premise (v5-04 t3 Kaldi, v5-06 t2 septic tanks, v5-19 t2 MEGNet, v5-36 t2 alcohol limit, v5-40 and v5-45 stopping drainage service / NHTSA "increase"). Answering as posed fails.
- S10: the answer must say the requested information is not in the library; giving the closest available fact is good (WeNet 2.0 is only cited by the README; Nghị định 88/2007 is only named in two files; the draft circular has indicator names but no values; the programme has no budget figure; no 2024 NHTSA count; Japan is not on slide 16; the ripgrep guide only uses a 0.7.1 archive as an example). Inventing the value fails.
- S6: list-type answers need all `expected_documents` named; extra files are tolerated only where noted below.
- Off-topic detours: a short general answer or a polite redirect is fine; citing a library document as the source of the general fact fails.

## Vague references, resolved

- v5-33: "the other bulletin" = FB 1227 (after FB 43); "the first one" in t3 = FB 43.
- v5-14: "the earlier NHTSA report" = the 2022 overview (scanned); "the two" in t3 = the 2022 overview and the 2023 summary.
- v5-35 (5 turns): "the second one" (t4, Vietnamese "bài thứ hai") = the MatGL paper, the second document named; "the first one" (t5) = the WeNet paper.
- v5-15: "slide đó" = FHWA slide 16 (adopters).
- v5-22: "that speed" = 42 mph (slide 36); "the slide" in t3 = slide 36.
- v5-26: "Nó" = the Government action programme under NQ 263 (t1 asked about it).
- v5-30: "Nó" in t3 = the draft decree replacing Nghị định 85/2017.
- v5-39: "nghị định đó" = Nghị định 80/2014 (VBHN 50).
- v5-21: "Nghị định 88/2007 đó" = the decree named in t1 (EN article), and "Nghị định 80/2014 trong văn bản hợp nhất của tôi" = VBHN 50.
- v5-43: "phụ lục đó" = Phụ lục I of the draft circular on cultural-industry indicators (the scanned annex).
- v5-08: "Còn bài tiếng Việt thì sao?" = the Vietnamese Hạ Long article.
- v5-42: "that file" / "the config file" = ripgrep's configuration file.
- v5-31 t2: "the README" = the MatGL README and "the paper" = the MatGL paper (named in t1).
- Deliberately ambiguous: v5-29 t1 ("the README": two READMEs, WeNet and MatGL) and v5-27 ("Nghị định số 85": 85/2017 and 85/2024). A good answer names both or asks which.
- Library-level S6/S10: v5-01 (WeNet files; nuclear power), v5-02 (Vietnamese legal texts incl. the comparison docx; US traffic-death data), v5-34 t2 (sewage/wastewater).

## Known ambiguities and caveats

- NHTSA 2022 vs 2023: the 2023 summary restates 2022 as final (fatalities 42,721; speeding 12,157; alcohol-impaired 13,458) while the 2022 overview note (ARF) gives 42,514 / 12,151 / 13,524. The questions are built on this difference (v5-14, v5-46); the 2023 Table 1 crash total for 2022 (5,930,697) also differs from the 2022 note (5,930,496), so the computed difference in v5-20 uses the 2023 table only.
- FHWA slide 8 values (2020 = 38,824, 2023* = 40,990 preliminary) differ from NHTSA (39,007 and 40,901).
- FHWA slide 16 shows percentages with down arrows under "Changes from 2000 to 2019"; the unit (reduction in road deaths) is implied by the deck, not written on the slide.
- v5-20 t1 compares the 2014 and 2021 rates from Table 2: the text says the 10-year range was 1.08 to 1.38, so the 0.30 gap is computed.
- WeNet paper vs README (v5-04 t3, v5-29 t2): the paper says WeNet has no dependency on Kaldi; the README only acknowledges borrowing Kaldi code for WFST decoding. The README is the current main branch (2026) and no longer describes the paper's model.
- Hạ Long: the EN article says both "some 1,600 islands" and "1,969 islets"; the VI article gives the core zone as 335 km2 in the lead and 434 km2 later. No question depends on those.
- Nghị quyết 263: the stamped red seal overprints the title on p1 of the scan; the number and date (07/9/2026) are printed in the "Kèm theo" line. The 1-page resolution text (Điều 1-3, effect "kể từ ngày ký") is not in the library, hence v5-44 t2 expects an abstention.
- VBHN 50 and the draft circular have blank number/date fields in the text layer ("Số: /VBHN-BXD"; "Số: /2026/TT-BVHTTDL"), so the VBHN number is only visible in the file name.
- The comparison docx merges cells: the old/new/explanation columns repeat text in the second table; row labels are in the first column of the second table (Table 2 of the file, Table 1 is the letterhead).
