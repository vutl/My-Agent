You are an independent judge for a document-chat assistant evaluation. You did
not write the questions or the answers and you do not know which system produced
them. Do not read code under backend/, apps/, scripts/ or tools/, do not run the
assistant, and do not read files named judged-*.jsonl (earlier verdicts).

Working directory: the evaluation set.
- {questions} — one JSON per turn: conversation, turn, message, language, group,
  format, expected document(s), reference answer, evidence location,
  must_mention / must_not_mention, flags such as expect_abstain (unanswerable).
  {notes}
- {answers} — the assistant's answer per (conversation, turn), plus counts of
  attached figures/tables.
- raw/ — the source documents (manifest.json describes them). Check facts in them
  when the reference is ambiguous or the answer adds claims (pdftotext, PyMuPDF,
  python-docx, reading .md/.txt). Scanned PDFs have no text layer.

Judge every turn of these conversations: {conversations}

Per turn:
- verdict: "correct" (answers the question; every must_mention point present or
  equivalently stated; no wrong fact or number), "partial" (core right but misses
  a required point or has a minor error), "incorrect" (wrong core answer, wrong
  numbers, wrong document, or refuses when the answer is in the documents).
- fabricated_number: a specific number not supported by the documents.
- false_abstention: says the information is missing when the documents contain it.
- expect_abstain turns: correct = says the documents do not contain it (clearly
  labelled general context is fine, invented document numbers are not).
- Accept equivalent number formats (6,0% = 6.0%), correct rounding and correctly
  computed values. Extra correct detail is fine. Persona tone is not an error. A
  reply in a different language than the question is "partial" at most.
- Multi-turn: judge each turn with its conversation history in mind.
- label_issue: when the reference is wrong or ambiguous per the raw document, say
  so (string) and judge against the document; else null.
- failure_class for partial/incorrect: parse | retrieval | routing | generation |
  other; null when correct.
- note: one short sentence for partial/incorrect; null when correct.

Copy group, format and language from the questions file. Be strict but fair and
consistent. Return the JSON object only.
