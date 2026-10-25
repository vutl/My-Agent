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

## Sealed suite v1 — status (2026-10-25)

- Written by an independent author agent (not the implementers, not the head
  agent). 30 tasks (20 vi / 10 en) across 10 categories, 25 fixture files, MANIFEST
  of sha256 sums. Stored **outside the repo** at `~/.aya/sealed/agent-v1/` (mode
  700) so it never reaches git, GitHub or the Codex worktrees; the head agent has
  not read the tasks. It moves into `evals/agent/v1/` after its first run.
- Runner requirements (from the author's report; to be built as package P1.8b):
  fresh workspace copy and isolated `HOME` per task; a local HTTP server on
  `127.0.0.1:8765` serving `fixtures/web/`, with `web_fetch` allowed to reach it
  in eval mode only; macOS tools (Notes, Calendar, Apple Reminders, open_app,
  open_url, Spotify) stubbed to record calls; approvals answered per task
  (`approval_policy: approve|deny`); `restart_after_step: n` restarts the task
  runner after step n; criteria evaluated from the workspace, reminders DB,
  tool-call log and final answer; `asked_user` also accepts a final question
  (before `ask_user` exists). Verify `shasum -a 256 -c MANIFEST.sha256` before a run.

