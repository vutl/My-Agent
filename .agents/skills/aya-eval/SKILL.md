---
name: aya-eval
description: Run and grade Aya's evaluation suites (acceptance, hard, codex-adapted, sealed v1/v2 multi-format sets), judge answers independently, compute accuracy with intervals, and classify failures. Use when measuring a change, rerunning failed turns, judging saved answers, or reporting Aya's quality.
---

# Aya evaluation

## Suites

| Suite | Files | Backend |
|---|---|---|
| Acceptance regression (c03–c08, 20 turns) | `data/retrieval_eval/sealed/v1/accept-regress.jsonl` | :7790 |
| Hard / codex-adapted | `evals/aya_hard_v1.jsonl`, `evals/codex_adapted/*.jsonl` | :7790 |
| Sealed v1 (19 docs, 89 turns) | `data/retrieval_eval/sealed/v1/questions.jsonl` | :7791 |
| Sealed v2 (18 docs, 92 turns) | `data/retrieval_eval/sealed/v2/questions.jsonl` | :7792 |

Sealed sets were authored independently; once their failures have been
inspected they are regression-only — report later runs as dev estimates.

## Run

```bash
backend/.venv/bin/python scripts/evaluate_aya_acceptance.py \
  --dataset <questions.jsonl> --base-url http://127.0.0.1:<port> --output <run.json>
```

- Structural checks only (focus, attachments, fallback). The content judge is the score.
- 9router quota runs out after ~50 Sol turns with long prompts: use
  `data/retrieval_eval/sealed/v2/run_subset.sh <subset.jsonl> <out.json>` (batches
  of 8 conversations, waits for quota, re-runs aborted batches).
- Restart the backend on the latest commit before a run (aya-runtime skill).

## Judge

Independent of Aya's answer model (gpt-5.6-sol):

```bash
python tools/harness/judge.py --set-dir <set dir> --answers <answers.jsonl> --out judged-<tag>.jsonl
```

`answers.jsonl` rows: conversation, turn, message, answer, attached_figures,
attached_tables (build from the run JSON `rows`). Verdicts: correct / partial
/ incorrect, fabricated_number, false_abstention, label_issue, failure_class
(parse, retrieval, routing, generation, other). The head agent spot-checks at
least 20% of verdicts against the raw documents before reporting.

## Generalisation

A fix is general only when a variant suite passes: `python tools/harness/variants.py --name <mechanism> --library-dir <a library the fix was not debugged on> --failure ... --mechanism ...` writes same-mechanism variants and negative controls by another model. Track every mechanism in `docs/handoff/11-FAILURE-LEDGER.md`.

## Report

- Accuracy with a Wilson 95% interval; correct+partial; fabricated numbers;
  false abstentions; per group, format and language.
- Gates: correct ≥ 85%, 0 fabricated numbers, false abstention ≤ 10%.
- Classify every failure, reproduce one per class with a debug trace, and fix
  the mechanism generally (never one paper, figure or question).
- Record results in `docs/handoff/00-STATUS.md`, `04-DONE.md`, `05-TODO-AND-ISSUES.md`.
