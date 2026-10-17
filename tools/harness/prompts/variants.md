You write evaluation cases that test whether a fix to a document-chat assistant
(Aya) generalises. You did not write the fix. Do not read code under backend/,
apps/, scripts/ or tools/.

The failure that was fixed:
{failure}

Mechanism (what kind of decision went wrong):
{mechanism}

Library to write against (working directory: its raw/ folder holds the
documents; read them to write reference answers):
{library}

Write {n} conversations that exercise the SAME mechanism with different
surface forms, and {negatives} negative controls where the tempting behaviour
of the fix would be WRONG (the reference must say what the right behaviour is).
Vary across cases:
- documents (use many different documents of this library, not one pair);
- language: Vietnamese, English, Vietnamese without diacritics, typos;
- wording of the reference (e.g. for pronouns: nó, bài đấy, bài kia, cái trước,
  paper đó, the other one, it, the first paper, cái thứ hai);
- conversation shape: 2–5 turns, the reference at different turns, a detour
  and a return, 2 vs 3 documents.
Every turn must be answerable from the library (or be a deliberate
unanswerable control) and carry a short reference answer with the evidence
location. Do not copy the failing case's documents or wording.

Return JSON with "cases": one object per TURN with fields conversation (string,
e.g. "var-01"), turn (1-based), message, language ("vi"|"en"), group
("variant"|"negative"), format ("multi" or the expected document's format),
expected_document (filename or null), expected_documents (list or null),
reference_answer, evidence, must_mention (list), must_not_mention (list),
expect_abstain (bool), purpose (one sentence: what this turn checks).
