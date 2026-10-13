# aya_acceptance_v1 notes

## Papers used (filenames as in documents table)
- ASPIRE.pdf: primary (G1 task/dataset/numeric, G2 figures 1, 3, 5, table 1, formulas, G4 follow-ups)
- CRAB.pdf: G1, G2 figure 1, table, MPCL formula (eq. 5/7), G3 return; also the 2-paper comparison (c02 t4)
- KST.pdf: G1, G2 figures 2/3, table 2, G3 return
- ICASSP_2024___ViSEC.pdf: G1, G2 figure 1/2, table 2 (Vietnamese, Chinese, Thai results), G3 return
- CMDM.pdf: G1, G2 figure 2/7, table 4, G3 return
- GBAN.pdf, WhiSER, wav2small, etc. not used.

## Formula requests
- All 3 formula turns (c02 t3, c05 t2, c12 t2) are answerable from equations present in the text: ASPIRE total loss (L_total), ASPIRE attention Softmax(QK^T/sqrt(d_h))V, CRAB MPCL loss (eq. 5 and 7).
- No impossible formula requests. Note the DB text for the ASPIRE attention equation is garbled (glyph-split), so the grader should accept the formula in any equivalent notation.

## Ambiguities and caveats
- CRAB results table (c05 t3, t4, t5): the DB table has blank column headers. I assigned the first WAR/UAR pair to IEMOCAP and the second to MELD, inferred from the results paragraph (MemoCMT WAR 0.6490 is described as the MELD baseline result). Treat these as inferred column mapping.
- ViSEC table 2 (c07 t2): headers are UA(%) and WA(%) in the markdown, with blank dataset-label rows; the values used come from the Vietnamese, Chinese and Thai rows.
- c02 t4 is a 2-paper comparison; expected_document is set to ASPIRE.pdf (the primary paper). Both papers are named in must_mention.
- G3 off-topic turns (c04 t2, c06 t4, c08 t4, c10 t2, c11 t4) have must_mention that describe required behaviour, not paper facts, since there is no paper text for weather, code, recipes.
- G4 referents, all resolved to the immediately preceding figure or table: c01 t3 (Fig 1), c02 t2 (Table 1), c03 t3 (Fig 2), c04 t5 (Fig 1), c05 t5 (CRAB table), c06 t3 (ViSEC Fig 1), c07 t3 (ViSEC table), c08 t3 (CMDM Table 4), c09 t3 (CMDM Fig 2), c09 t5 (CMDM Fig 7), c11 t3 (ASPIRE Fig 3), c12 t5 (ASPIRE Fig 5). No ambiguous referents found.
- Caption-only facts (e.g. CMDM Fig 2, ASPIRE Fig 3) are short; the grader should check them against the caption text.
- c02 t3 must_mention includes "CCC instead of MSE", taken from the ASPIRE text ("optimize ... CCC rather than standard MSE").
