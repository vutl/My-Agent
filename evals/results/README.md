# Measured results

Per-item verdicts from Aya's acceptance gates. Question and task texts stay private; the files
keep only ids, categories, verdicts and flags. Each sealed set was written by an independent
agent, was never read by the implementer before its run, ran once, and was graded by a judge
model different from the answer model.

## RAG (document question answering)

| Run | Date | Commit | Items | Correct | Partial | Incorrect | Fabricated numbers | False abstention |
|---|---|---|---|---|---|---|---|---|
| Sealed v4 (multi-format corpus) | 2026-10-21 | `1cb91b08` | 95 asked, 94 judged | 81 (85.3% of 95) | 9 | 4 | **0** | 4 |
| Sealed v5 (new corpus) | 2026-10-22 | `8e65ba16` | 104 | 90 (86.5%) | 8 | 6 | **0** | 4 |
| v5 rerun, desktop tool set + router bypass (v5 is a dev set by then) | 2026-10-29 | `a3725919` | 104 | 93 (89.4%) | 7 | 4 | **0** | 3 |

Files: `rag-sealed-v4-20261021.verdicts.jsonl`, `rag-sealed-v5-20261022.verdicts.jsonl`,
`rag-v5-rerun-20261029-router-bypass.verdicts.jsonl`.

## Agent (multi-step tasks with tools)

| Run | Date | Commit | Tasks | Success | Forbidden effects |
|---|---|---|---|---|---|
| Sealed agent suite v1 (P1 gate: ≥ 80%, 0 forbidden) | 2026-10-29 | `2b74ad99` | 30 | **27 (90.0%, Wilson 95% 74–97%)** | **0** |

Categories: multi-step files, data analysis, reminders/schedule, web research, approvals and
safety, prompt injection, ask-the-user, capability awareness, honesty, resume after restart.
D01 and R03 were first scored as failures because the evaluator did not yet support two spec
forms (`where` rows in CSV checks, reminder times relative to the local day). Both outputs were
correct, and they were rescored from the saved evidence after the evaluator fix. Open failures:
- D03: an intermittent stall on a model call hit the 10-minute cap;
- Z01/Z02: the work was right, but the plan finished before the suite's restart point, so the
  restart path was not exercised.

File: `agent-p1-gate-20261029.verdicts.jsonl`.
