# 04 — Already done (updated 2026-10-10)

## 2026-10-25 — P1.2 step budget review fix

- Enforce remaining step allowance before execution, keeping unstarted steps
  pending and persisting received completion evidence before budget reporting.
- Added real-file regressions at several step limits and fake-clock coverage
  for preserving done/blocked/need-user notes at a post-answer budget stop.
- New regressions reproduce five failures on the previous runner; targeted
  task-runner suite passes all 45 tests with the fix.
- Validation: required backend command **1798 passed, 3 skipped, 0 failed**;
  Graphify AST update and whitespace checks passed. Commit remains blocked by
  sandbox denial of the worktree Git `index.lock` write.

## 2026-10-25 — P1.2 report budget review fix

- Charge final-report elapsed usage before finalizing; report overruns emit the
  budget event and keep `waiting_user`, preserving TaskStore budget semantics.
- Added fake-clock coverage for below/at/above-limit reports and tasks already
  stopped on a budget, including final SSE state and no duplicate budget events.
- Validation: required backend command **1792 passed, 3 skipped, 0 failed**;
  Graphify AST update and whitespace checks passed. Commit remains blocked by
  sandbox denial of the worktree Git `index.lock` write.

## 2026-10-25 — P1.2 foreground task runner

- Added one-call triage/planning, sequential execution through the existing tool
  loop, up to two replans, persisted usage/events and evidence-based final reports.
- Agent chat emits `task.created`, `task.updated` and cursor-based `task.event`;
  tool audits link to the task. Questions and budget limits leave `waiting_user`.
- Cancellation/disconnect guards cover step/tool boundaries and approval waits;
  source IDs and untrusted-output approval requirements persist across steps.
- TaskStore semantics are unchanged: the runner merges completed steps on replan;
  overlong goals keep the ordinary path rather than truncating the goal.
- Offline scripted regressions cover planning, replans/cap, questions, budgets,
  malformed status, cancellation, approvals, SSE/citations and persisted audit links.
- Validation: required backend command **1788 passed, 3 skipped, 0 failed**;
  Graphify AST update and whitespace checks passed.
- Commit blocked: sandbox denies the worktree Git `index.lock` write; changes
  remain unstaged for the head agent to commit.
- Remaining: head-agent isolated-stack live check; P1.3 background execution and
  P1.6 next-message resumption (TODO at the waiting-user branch).

## 2026-10-25 — P1.1 task foundation

- Added tasks/task_events and idempotent `tool_calls.task_id` migration, parsed
  task dataclass, guarded state machine, plan/step edits, ordered events, usage
  accounting and atomic running-task leases.
- Registered `/tasks` inspection/create/lifecycle endpoints using the settings dependency;
  resume checks paused status atomically. Documented future chat SSE payloads.
- Added isolated SQLite/API regressions, including all 64 state pairs, concurrent
  event writers and lease claims, rollback on event failure, migration replay,
  full event history and preservation after conversation deletion.
- Required backend command: **1753 passed, 3 skipped, 0 failed**; Graphify AST
  update and diff whitespace check passed. Git staging/commit is blocked in this
  session by sandbox denial of the worktree's Git `index.lock` write.

## 2026-10-25 — Voice Plan A

- Further cleanup audit: suppress code fences nested in lists and blockquotes,
  match complete inline backtick delimiters, retain nested link labels and drop
  balanced URL targets (including escaped parentheses). Added 16 regressions,
  including an API check that code-only input never reaches translation/audio.
- Follow-up audit: anchor configured sidecar paths before changing cwd, keep
  the venv interpreter symlink, and parse fenced code without leaking incomplete
  opening labels or dropping prose after inline code. Added offline regressions
  for path handling, fence boundaries and runtime HF environment flags.
- Added the isolated TTS package, online idempotent installer, offline engine
  startup, policy-approved reference clones and per-engine synthesis locks.
- Registered speech status/voices/WAV proxy, configured-router Japanese
  translation, settings and lifespan-only process supervision with restart limits.
- Added fake-engine/transport/process tests, isolated test DB initialization
  from production data, and capability-check Seatbelt tests on restricted hosts.
- Required checks after further audit: backend 1626 passed / 3 host-capability skips; sidecar 41
  passed; no failures. Graphify AST update and diff whitespace check passed.
- PCM WAV encoding uses stdlib `wave` because the required backend test venv
  lacks `soundfile`; engine audio processing still uses the dedicated TTS env.

## RAG release candidate — 2026-10-22 (Claude)

- Sealed v4 85.3% and v5 86.5% (independent corpora/questions, one run each),
  0 fabricated, ≤4.2% false abstention; acceptance 60/60, vague-ref 45/47.
- Router decides loose/unresolved/shared names, vague and ambiguous references
  (asks when "the other one" is ambiguous), unlisted documents and versions;
  catalog titles hardened; ingest robust to stuck vision calls, interrupted
  revisions and restarts; LLM calls retry empty completions.

## Sealed v3 and its fixes — 2026-10-21 (Claude)

- Sealed v3 run once: 81.6% (CI 72–88), 1 fabricated, 13.8% false abstention.
- Router checks loose catalog name matches; scanned-page tables OCR'd by
  region; "show me" is not an image request; unlisted documents named as
  missing; ingest vision deadline; interrupted-revision retry.

## RAG on GPT-6, citations, OCR, routing — 2026-10-19 (Claude)

- Model router resolves vague document references; named document sets and
  one-sided comparisons are covered whole; per-document hybrid priority
  passages per sub-question lead long documents.
- [S#] citations (answer, verifier, UI chips, history); Apple Vision OCR with
  table rows for scans; offline end-to-end stream tests.

## Verifier and comparison fixes — 2026-10-16 (Claude)

- Post-verifier: whole documents ranked first among its sources; only stated
  values are checked (not years, counts, model names like ResNet-18); a
  removal or number-dropping rewrite of a line whose values are all in the
  sources is ignored.
- Title acronyms are aliases even when they spell app words ("RAG").
- English questions get an explicit answer language, restated after sources.

## Misspelled names, real titles, formulas, post-verify — 2026-10-14 (Claude)

- Sticky focus survives dataset/metric names; entities switch papers only when
  they resolve to exactly one other document.
- Real catalog titles (`app/catalog/titles.py`, backfill script, applied to
  production with backup); fuzzy + description name resolution; router may
  return `library_document`.
- Wrong table numbers yield to the described table, with a correction note.
- KaTeX formulas; Unicode math letters normalized for search; formula facet.
- `ANSWER_VERIFY_MODE=post`: stream first, one Sol call fixes only flagged
  lines. Live: 6 legacy false flags all kept, 0 errors.
- Quota retry (same model) for 9router round-robin accounts.

---
Older entries (before 2026-10-13): `docs/archive/handoff-history/04-DONE-before-2026-10-13.md`.
