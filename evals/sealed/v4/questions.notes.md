# Sealed eval v4: question set notes

Written by an independent question author (subagent) on 2026-10-21. The only inputs were the 20 files in `raw/`, `manifest.json`, `CORPUS_NOTES.md`, and the v3 notes/schema (format and process only). The author did not read v1-v3 questions or answers, did not read Aya's code, scripts or docs, and did not run Aya.

Every fact in a reference answer was checked against the document text: PyMuPDF for native PDFs and for the native originals of the scanned PDFs, python-docx for DOCX, plain text for MD/TXT. The three scanned PDFs were also rendered and read visually: all five pages of TT 27/BYT, WASDE-Sep pp11 and 14, and NTSB synopsis p3 (its p1 was checked against the native text). MAE Figures 5 and 9 were checked on rendered pages. Self-check: for about 80 of the 95 turns the key numbers and phrases were re-checked by a scripted string search on the cited page (PDFs; the native originals for scanned files) or section (DOCX/MD/TXT), and 11 cited pages were re-opened as rendered images (all 5 pages of TT 27/BYT, WASDE-Sep pp11 and 14, NTSB synopsis p3, MAE pp4 and 7, the stamped headers of TT 28 and TT 27/BNNMT). The check found no wrong reference; the fixes made were small (the page of the YOLOv10 T4 latency statement, the A-24-9 description, the Figure 5 wording).

## Files

- `questions.jsonl`: 95 turns in 45 conversations. 25 single-turn conversations, 3 of 2 turns (v4-08, v4-19, v4-22), 6 of 3 turns, 9 of 4 turns and 2 of 5 turns (v4-12, v4-35). Conversation order was shuffled with a fixed seed.
- Schema as in v3: `conversation, turn, message, language, group, format, expected_document, expected_documents, reference_answer, evidence, must_mention, must_not_mention, expect_abstain, must_attach` (+ `off_topic: true` on the three general-knowledge detour turns).
  - A turn that needs more than one file has `format: "multi"`, the files in `expected_documents`, and `expected_document: null`. Library-level (S6) turns list every file that should be named.
  - The three detour turns (v4-42 t3 bushels per ton, v4-44 t3 capital of Australia, v4-35 t4 kcal to kJ) and the one library-level S10 turn (v4-15, no matching file) have `format: null` and no expected document. Detours have `expect_abstain: false`.
  - `must_attach` is `figure` once (v4-16 t3, MAE Figure 9) and `table` twice (v4-32 t1, WASDE wheat-by-class table; v4-27 t2, the vaccine disease table of the draft circular). They appear only where the user asks to see the item.
  - `group` is assigned per turn. In multi-turn conversations a turn keeps its intrinsic group (S1, S2, C1, S10 ...) when it is self-contained or compares documents, and is labelled S7 when it depends on dialogue context (vague reference, "the previous edition", detour, return after a detour). That is why S7 (21 turns) is larger than in v3; no conversation consists only of S7 turns.

## Counts

| group | en | vi | total |
|---|---|---|---|
| S1 single-document fact | 4 | 5 | 9 |
| S2 table number + computation/ranking | 6 | 4 | 10 |
| S3 figure/caption/slide image | 3 | 2 | 5 |
| S4 formula | 3 | 1 | 4 |
| C1 cross-document (editions, README/paper, NTSB vs Wikipedia, circulars) | 8 | 2 | 10 |
| C2 cross-language | 4 | 5 | 9 |
| S6 library-level | 2 | 2 | 4 |
| S7 multi-turn dependent (follow-up, detour, return) | 10 | 11 | 21 |
| S8 messy input | 3 | 3 | 6 |
| S9 false premise | 3 | 2 | 5 |
| S10 unanswerable (`expect_abstain: true`) | 7 | 5 | 12 |
| **total** | **53** | **42** | **95** |

12 of 95 turns (12.6%) expect an abstention. Formats (of the expected document): pdf_native 39, multi 14, docx 10, md 9, pdf_scanned 8, txt 6, slides_pdf 5, null 4. Turns whose question language differs from the language of (any) expected document: 21; the C2 group counts the ones whose main point is the crossing, the others sit in S2, S7, S8, S9, S10 and C1 (for example the Vietnamese questions on the English WASDE or on YOLOv10).

Every file is the expected document of at least 3 turns: WASDE-Oct 10, YOLOv10 paper 10, WASDE-Sep (scanned) 7, MAE 7, TT 75 7, VBHN 16 7, NTSB synopsis (scanned) 6, NTSB slides 6, vaccination draft 6, TT 27/BNNMT 6, FB 447 5, TT 28 5, Lúa mì 5, Sốt rét 5, Alaska Wikipedia 4, TT 27/BYT (scanned) 4, FB 59 3, Malaria 3, Wheat 3, YOLOv10 README 3. The three scanned PDFs appear in 17 turns; several need numbers that only exist in scanned tables (Sep WASDE wheat-by-class and rice tables, TT 27/BYT table rows).

## Grading guidance

- Number formats: Vietnamese references use decimal commas (4,56 tỷ; 35,9%), English ones decimal points; either style counts. Units that keep the same value are fine (3,5 triệu = 3.500 nghìn).
- `must_mention` items are short, independently checkable facts; paraphrase is acceptable. Parenthesised labels say which source or period a number belongs to.
- `must_not_mention` lists wrong claims. An answer that refutes the claim does not violate it.
- S9: the answer must correct the false premise (v4-01 knowlesi, v4-20 t3 altimeter pressure, v4-22 t1 corn "cut", v4-30 t4 Langstroth size, v4-31 t4 "50 withdrawn"). Answering the question as posed fails.
- S10: the answer must say the requested information is not in the library. Giving the closest available fact is good (for example T4 latency for the YOLOv10 FPS question, the 2005 table for the Vietnamese wheat question, or the English article's 2024 figure attributed to the English article). Inventing the missing value fails.
- S6: list-type answers need all `expected_documents` named; extra files are tolerated only where noted below.
- Off-topic detours: either a short general answer or a polite redirect is fine; citing a library document as the source of the general fact fails.
- Multi-turn vague references, resolved:
  - v4-42: "the previous edition" = September 2026 WASDE (scanned); "those two editions" = Sep and Oct WASDE.
  - v4-32: "the previous edition" = September WASDE; "from then to October" = September to October.
  - v4-22: "September's edition" = the scanned September WASDE.
  - v4-12: "bản tiếng Anh" = the English Wikipedia "Wheat"; "that English one" in t3 = the same; "bài tiếng Việt" in t4 = "Lúa mì" (2005 table).
  - v4-44: "bài tiếng Anh" = English Malaria article; "bài tiếng Việt" (t4) = "Sốt rét", the first article.
  - v4-13: "Thông tư số 27/2026" is deliberately ambiguous (two circulars); "Cái của Bộ Y tế" = TT 27/BYT; "Còn cái kia" = TT 27/BNNMT; "Thông tư của Bộ Y tế đó" = TT 27/BYT.
  - v4-31: "the later circular that amends it" = TT 28/2026; "altogether" = TT 75 + TT 28 trade names.
  - v4-24: "the other one that has a transition rule" = benfuracarb (after carbosulfan).
  - v4-07: "the other NTSB board meeting" = the Alaska 1282 synopsis (scanned); "the Wikipedia article" = Alaska 1282.
  - v4-19: "the NTSB board meeting synopsis" = the Alaska 1282 synopsis.
  - v4-20: "the one for the helicopter" = slide 9 (UH-60L); "the slides" = the Potomac board-meeting slides.
  - v4-23: "the GitHub README" = the YOLOv10 README; "the other paper" = MAE.
  - v4-38: "the downsampling part" = YOLOv10 section 3.2; "then" in t3 = YOLOv10 Table 1.
  - v4-16, v4-08: "there" / "they" = the MAE paper.
  - v4-30: "Bản 1911" = FB 447; "bản nào thay thế bản nào" = FB 59 vs FB 447. v4-05: "the older one" = FB 59.
  - v4-35: "thông tư đó" = the consolidated hospital-nutrition circular; v4-27 t3 "Trong số đó" = the 14 diseases of Điều 3; v4-45: "Còn giống lúa" = same BNNMT circular.
  - Detours and returns: v4-42 t3-t4, v4-44 t3-t4, v4-35 t4-t5.

## Evidence conventions

- `pN` is the 1-based PDF page index. For WASDE, the NTSB synopsis, MAE and YOLOv10 the printed page numbers equal the PDF index (e.g. "WASDE - 676 - 11" is PDF page 11), so there is no offset to worry about.
- Slides are cited as "slide N" (= PDF page N).
- DOCX/MD/TXT have no pages. Evidence names the section, article (Điều/khoản), table or paragraph; Wikipedia tables are numbered in document order (Wheat Table 5 = the 2024 production table). For the vaccination draft, "bảng đầu tiên" in Điều 3 is Table 2 of the file (Table 1 is the letterhead).
- TXT: sections of the Farmers' Bulletins are cited by their headings.

## Computed values (verified arithmetic; not printed in the documents)

- WASDE-Oct wheat class: HRW 333 / 740 = 45.0% (Sep: 319 / 717 = 44.5%).
- Wheat 2024: (140.1 + 113.3) / 798.5 = 31.7%. 2005 (VI): (96 + 72 + 57) / 626 = 35.9%.
- TT 28 additions: 105+114+34+4+1+2 = 260 active ingredients; 136+149+53+7+1+2 = 348 trade names. TT 75 banned: 23+6+1+1 = 31. Herbicide trade names 996 + 53 = 1,049 (v4-31 t3).
- Alaska timeline: 5:26:46 pm - 5:12:33 pm = 14 min 13 s.
- MAE: 85.9 - 85.2 = 0.7; 81.0 - 73.5 = 7.5.
- Rice ending stocks: 40.4 - 40.0 = 0.4. Wheat ending stocks 740 - 717 = 23.
- General-knowledge detours: 1 metric ton wheat is about 36.7 bushels; Canberra; 1 kcal = 4.184 kJ.

## Ambiguities and low-confidence labels

1. **Injuries, NTSB vs Wikipedia (v4-07 t3)**: the synopsis says 8 minor injuries (1 flight attendant + 7 passengers) and Wikipedia says 3. Both are quoted; the question asks whether they agree, so the answer must report the difference. The synopsis is "subject to editing"; the final report (not in the library) may differ.
2. **Altitude (v4-19 t2)**: both sources give 14,830 ft for the depressurization. Wikipedia's lead says "about 16,000 feet", which is only the approximate climb altitude. Either remark is acceptable if 14,830 is given.
3. **Two TT 27/2026 (v4-13)**: the first turn expects both circulars. An answer that picks only one (for example the BNNMT one because it is longer) is incomplete. In t3 "cái kia" is only unambiguous after t2.
4. **v4-31 t3 (1,049)**: TT 28 withdraws 5 records (category not given), so the sum is "before withdrawals", as the question says. Adding active-ingredient counts is not required (overlaps are possible).
5. **S6 wheat list (v4-21)**: four files are required. The BNNMT circular (TT 27/BNNMT) lists wheat only as an HS-code line in a table of feed raw materials; naming it is tolerated but not required.
6. **WASDE wheat-by-class "Sep" and "Oct" rows**: in the Oct file the table has two ending-stock rows (Oct, then Sep); in the Sep file they are Sep then Aug. The v4-32 questions use the Oct row of the Oct file and the Sep row of the Sep file.
7. **WASDE 2025/26 estimates differ between editions** (for example production 1,985 vs 1,989 million bu). No question depends on these.
8. **YOLOv10 AP**: Table 1 prints "53.2 / 53.4†" (NMS-free / original one-to-many). v4-23 t1 accepts 53.2 as the main AP (53.4 is mentioned in the reference). The README prints only 53.2.
9. **MAE figure values** (Figures 5, 7, 9) are read from data labels printed on the plots and from the text (73.5 -> 81.0 is also stated in section 4.3). v4-16 t2 treats "about 84.9 / 73.5 at 75%" as the key facts; the 10% value for linear probing (54.6) is quoted in the text as well.
10. **Sốt rét infobox vs body (v4-44 t1)**: the Vietnamese infobox says 296 million (2015) and the lead says about 515 million per year. The reference requires the lead's figure and mentions the infobox; an answer giving only the infobox numbers is partially correct.
11. **Vaccine disease table (Điều 3)**: python-docx shows duplicated rows for merged cells; the document has 14 numbered diseases. The 14 count is the numbered rows (TT 1-14).
12. **Draft status**: the vaccination text is "Dự thảo 1"; answers may say it is a draft. TT 75 and TT 28 leave the effective date blank in the text layer, and no question asks for it.
13. **FB 59 vs FB 447 prices**: both give prices in cents per pound for different situations (wholesale vs direct). The reference lists the figure for each stated condition; v4-30 t1 asks only for the 1905 values.
14. **v4-38 t4 and v4-23**: Aya may reasonably answer "the paper reports latency on a T4, not FPS" for the RTX question and give 1.84 ms; that is the expected abstention answer, not a failure.

## Verified-absent facts used by S10

- No 2027/28 column or text in WASDE-Oct (v4-06); no ViT-G, RTX 4090 or FPS anywhere (v4-08, v4-38).
- The Potomac slides give no total count of recommendations (they name five route-design recommendations and the two urgent ones).
- FB 397 appears only as a superseded bulletin; TT 08/2011/TT-BYT appears only as a replaced circular (VBHN 16 Điều 15), QĐ 2879/QĐ-BYT only as a reference.
- The English malaria article has no Vietnam case count; neither wheat article gives Vietnam's production; the Vietnamese wheat article has no 2024 figure.
- TT 27/BYT has no penalty provisions; the vaccination draft gives no compensation amounts.
- No file about electric-vehicle batteries (v4-15).
