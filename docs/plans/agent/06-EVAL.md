# 06 — Evaluation

The RAG work proved the method: sealed sets written before running Aya,
independent judges, intervals, and a failure ledger. The agent gets the same.

## Agent task suite

- `evals/agent/v1/tasks.jsonl` (sealed): 30 tasks for P1, 60 for v2 (P2 tools).
  Each: goal text (Vi/En), setup fixture (workspace files, fake mailbox, fake
  calendar, local web pages), allowed tools, success criteria (checkable:
  files exist with content, event created with fields, reply drafted not sent),
  forbidden effects.
- Runs on the isolated stack with fake connectors (no real mail/messages), real
  model calls; one run per task, reruns only for infrastructure errors.
- Judge: deterministic checks first; a model judge (judge_ag / astra, never the
  answer model) for report quality; 20% manual spot check.

## Metrics

Task success; partial credit by criteria; unrequested side effects (must be 0);
approvals asked when required / not asked when not needed; steps and tool calls;
wall time; model calls; honest final report (claims match events).

## Suites around it

- Injection suite (04-SAFETY) — 0 escapes.
- RAG regression: acceptance + vague-ref variants unchanged from v1.
- Voice/avatar: start time and first-audio latency measured on every rollout.

## Reporting

`evals/agent/v*/report.md` with per-task verdicts, failure classes (planning,
tool choice, tool error, safety, reporting), and the fixes they led to in
`docs/handoff/11-FAILURE-LEDGER.md`.
