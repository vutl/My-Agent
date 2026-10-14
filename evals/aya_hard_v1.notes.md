# aya_hard_v1 — adversarial E2E suite (notes)

91 turns in 19 conversations (48 vi, 43 en). The file is `aya_hard_v1.jsonl`. It uses the same schema as `aya_acceptance_v1.jsonl` with these additions: `must_not_mention`, `expect_abstain`, `evidence`, and an optional `expected_documents` list. When a turn spans several papers, `expected_document` is `null` and `expected_documents` lists the papers involved.

## Categories (turn counts)

| Group | Turns | What it tests |
|---|---|---|
| H1 false premise | 17 | User states a wrong number or claim. Aya must correct it with the paper's value. |
| H2 unanswerable | 7 | GPU, seed, epochs or code that the paper never states, or a dataset it never uses. `expect_abstain: true`. |
| H3 cross-paper | 8 | Exact numbers from 2–3 papers, including comparisons that are not valid (metric, version, class-set or protocol mismatch). |
| H4 multi-hop / arithmetic | 14 | Differences, per-fold winners, best-K per metric, gains between rows. |
| H5 long conversations | 8 | h04 (ViSEC ↔ CMDM), h07 (GBAN ↔ MSF-SER), h12 (KST ↔ Wav2Small). These use vague references ("bài trước CMDM", "cái bài kia", "the second table you showed") and include an off-topic turn. |
| H6 messy input | 8 | No diacritics, typos (aspier, crabb, wav2smal, IEMOCPA, "whisper corpus"), wrong table or figure numbers with a correct description. |
| H7 figure/table/formula detail | 17 | Values inside tables and figures, 4 formula requests (GGF, MSF-SER emphasis score, CMDM MMD, Robust-AV loss). |
| H8 ambiguity / negation | 7 | "baseline WavLM", "STSER", "Whisper", "late fusion" each match several papers. Also "paper nào KHÔNG dùng IEMOCAP" and "MSP-Podcast papers that are audio-only". |
| H9 library/meta | 5 | Paper count, MSP-Podcast versions, LSSED users, papers that introduce a dataset, GPUs stated. |

## How labels were verified

- Source: `data/retrieval_eval/internal/runtime/e2e-foundation-20261005/sqlite/app.db`. I used only chunks where `revision_id = documents.active_revision_id`, plus `document_tables` and `document_figures` for the same documents. All 15 emotion papers were dumped and the body of each was read; reference sections were excluded from absence checks.
- Every number appears in the table markdown, in a chunk, or in both. Where the DB markdown is garbled, I checked the raw chunk text. Computed values (differences, sums, means) are written out in `evidence`.
- H2 absences were confirmed with case-insensitive keyword scans of every active chunk of that document. Examples:
  - ASPIRE: GPU / NVIDIA / RTX / A100 / hours / MELD → 0 hits.
  - CRAB: seed / runs / repeated → 0.
  - GBAN: epoch / batch / github / available → 0.
  - KST: only one "epoch" mention.
  - CMDM: GPU → 0.
- H9 lists come from a per-document keyword scan, followed by reading the context to separate the paper's own experiments from related work and references.

## Deliberate traps that come from inconsistencies inside the papers

These are "hard" because the paper's own prose disagrees with its tables. The labels follow the table, or ask Aya to point out the mismatch:

- **ViSEC, Thai WA gain:** the table gives 87.73 − 77.07 = **10.66**, but the text says 12.30% (h04 t6).
- **Robust AV, dataset:** the abstract and introduction claim a noise-augmented AFEW, but the experiments use **Aff-Wild2 with no injected noise** (h15 t1).
- **Robust AV, inference time:** Table 3 says 9.18 ms / 108.93 FPS, but the text says 27.4 ms / 36.5 FPS (h15 t2).
- **Robust AV, valence gain over Leader-Follower:** the table gives **0.235**, but the text says 0.220 (h11 t5).
- **CMDM Table 4:** the DB markdown mislabels the A+L baseline rows as "A+V". The chunk text (chunks 30–31) shows the correct grouping (h17 t1).

## Low-confidence or strict labels (review before trusting a failure)

1. **h09 t1 ("15 papers"):** this depends on the runtime library matching this DB snapshot. The snapshot has 21 documents: 15 emotion papers plus two 9router PDFs, three PROJECT_PLAN `.md` files and `demo.md`. The Dimensional-emotion paper is text-only but is counted as emotion recognition.
2. **h04 t6, h11 t5, h15 t1–t2 (strict):** `must_mention` requires Aya to flag the paper's internal mismatch. An answer that only repeats the prose number fails these turns by design.
3. **h05 t2 (Crab uses 8 MSP-Podcast categories):** this is inferred from CRAB TABLE I, which lists 8 MSP classes. The prose never says "eight classes" explicitly.
4. **h08 t5 (negation, IEMOCAP):** the Dimensional-emotion paper makes a one-line zero-shot IEMOCAP remark (chunk 44), so it is deliberately left out of both `must_mention` and `must_not_mention`. JAIT is labeled as using IEMOCAP.
5. **h15 t4 (Robust AV loss):** the formula extraction reads "Ltotal =1−ρa+ρv 2". The label assumes it means 1 − (ρa+ρv)/2.
6. **h18 t3:** the LPMN vs ASPIRE comparison is labeled "reference only". Both use IEMOCAP 4-class with session-wise 5-fold, but the pipelines differ, so a judge may accept a softer framing.
7. **Artifacts with no extracted caption:** CRAB tables (h02 t2) and the Wav2Small Table I (h12 t2) have `caption = None` in `document_tables`, so these turns have no `expected_artifact_caption`. The CMDM MELD table's caption is mis-extracted as the "Figure 5" text, so h04 t8 does not require an attachment.
8. **h01 t8 (ASPIRE confusion counts 100 / 51):** these come from the prose and the extracted matrix text, not from pixels.

## Facts avoided because the source is garbled or self-contradictory

- Fear Test2 F1 = 0.062 in the multi-label paper's Table 3 (it is inconsistent with its P/R).
- JAIT weighted F1 77.9 exceeds both weighted P (77.3) and R (77.0).
- LPMN's MER2023 MSE headline (0.9953) does not match the selected K'=60 row (0.9968).
- The Dimensional ablation "+.302 / +.027" deltas do not match its table.
- CRAB's "~0.08 Macro-F1 over best non-ensemble" claim does not match its table.
- CRAB says MELD "Sadness 3%", but its Table I gives 0.07.
- Any Wav2Small per-model CCC that is visible only in the Fig. 2 / Fig. 4 pixels.
