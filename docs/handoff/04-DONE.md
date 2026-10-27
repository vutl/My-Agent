# 04 — Already done (updated 2026-10-10)

## 2026-10-27 — P1.9 stream the agent route

- Review fixes: adapt pending tool calls to the desktop's started-card contract,
  replacing placeholder arguments under the same call ID; reconcile authoritative
  final answers with partial streams using message.reset when needed.
- Empty final reports retry through the shared streaming client's bounded policy;
  leading whitespace stays hidden and visible text prevents retries. Added desktop
  consumer, attached-report failure, empty-then-success and exhausted-retry regressions.
- Added OpenAI-compatible tool SSE assembly (text, call starts, argument deltas,
  final ToolTurn), shared whole-turn parsing and retries only before public output.
- Forward live answer deltas/pending tool cards through the loop and API; retain
  narration and compute citations on the complete answer without replaying text.
- Stream final task reports while keeping step control messages nonstreaming;
  reconcile task fan-out ordering and missed deltas for attached chats.
- Offline regressions cover progressive delivery, tool-card ordering, narration,
  repaired arguments, empty output, malformed streams, retries, private step JSON,
  report persistence, late attachment and overflowing event queues.
- Validation after review fixes: required backend suite **2064 passed, 3 skipped, 0 failed**;
  Graphify AST refresh and whitespace checks passed.
- Review-fix commit blocked: sandbox denied creation of
  `.git/worktrees/p1-9-streaming/index.lock`; the seven changed files remain unstaged.

## 2026-10-27 — P1.4 quick triage, validation, rate limits and worker follow-ups

- Added router-first planning triage, safe leading-object JSON repair and schema
  validation, shared monotonic rate windows for web/code tools, and repaired-call receipts.
- Added event-driven task wakeups on start and every resume, filtered internal
  execution notes from public events/inspection, and bounded retry evidence per step.
- Persist completion before channel dispatch and deduplicate each notification/channel
  durably. Offline tests cover router selection/fallback/deadline, argument failures,
  rate window boundaries, sync-thread wakeups, retry evidence, private history and
  crashes after external sends.
- Validation: required backend command **2006 passed, 3 skipped, 0 failed**;
  worker/task API checks **153 passed**; Graphify AST refresh and whitespace checks passed.
- Commit blocked by sandbox denial of `.git/worktrees/p1-4-speed/index.lock`;
  changes remain unstaged for the head agent to commit.

## 2026-10-25 — P1.6b interaction primitives on the task runner

- Re-applied the P1.6 primitives on the runner: `TaskStore.user_interaction` (atomic
  ask/answer), `services/task_interactions.py` (resume triage), runner `ask.user` handling,
  artifact handles on `tool_result` events, `notify` truthfulness. Tests:
  `test_assistant_core.py`, `test_task_interactions_stream.py` (resume through the worker,
  restart, triage fallback, new request, macOS failure).

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
## 2026-10-25 — P1.5 capability sheet and routing

- Grouped all first-party tools, with registry-derived cached vi/en summaries,
  label fallback for unknown groups and approval guidance. The description-only
  registry needs no library searcher; chat adds the app's library summary.
- Chat system messages and agent tool loops share capability descriptions;
  file_qa/research system and document prompts do not receive the sheet.
- Strengthened action-versus-ability routing without keyword dispatch; added
  registry, language, cache, allowlist, prompt-isolation and offline SSE tests.
- Review follow-up: restrict greeting shortcuts to complete social greetings;
  fake-LLM regressions cover prefixed actions with tools allowed/disallowed,
  plus standalone greetings that still skip the model. The public-corpus guard
  verifies actual model calls instead of assuming a fixed shortcut count.
- Further review (2026-10-25): require complete-message matches for every
  personal-question/banter shortcut; ambiguous open-ended questions reach the
  router. Added 38 vi/en regressions for social-prefixed actions, tool allowlist
  gates and complete social messages that still bypass the model.
- Required backend command after further review: **1725 passed, 3 host-capability skips**,
  zero failures. Graphify AST update and diff whitespace check passed.

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
