# 04 — Already done (updated 2026-11-13)

## 2026-11-13 — R1 search navigation and fallback review

- Preserve provider-returned URLs for navigation/citations while using canonical
  keys to deduplicate and rank provider agreement.
- Base fallback and search success on usable URL hits, retaining citation-free
  Codex summaries separately with the existing unverified label.
- Added original-URL fetch regressions for `www`/trailing slashes and repeated
  query parameter order, plus citation-free text/refusal responses with successful,
  empty, failing and unconfigured DuckDuckGo fallback. Source tests: **75 passed**.
- Required full suite: **2343 passed, 3 skipped, 4 failed + 2 errors**; all six
  failures/errors are existing eval-runner loopback binds denied by the sandbox.
  Graphify and whitespace checks passed. Git staging cannot create the worktree
  `index.lock` (`Operation not permitted`), so the changes remain uncommitted.

## 2026-11-13 — R1 model-facing research output

- Fixed the shared tool-loop formatter rather than individual research handlers:
  citation snippets supplement substantive content instead of replacing it.
  Content over 16,000 characters keeps its complete artifact and bounded preview.
- Offline integration tests inspect actual model messages for two papers'
  authors/years/venues/citation counts/IDs/PDF URLs, two Wikipedia extracts longer
  than their snippets, and a Codex summary with its unverified label; each runs
  with streaming and whole-turn clients. Updated generic artifact regressions.
- Focused source/tool-loop/assistant-core suite: **145 passed, 0 failed**.
  Required full suite: **2329 passed, 3 skipped, 4 failed + 2 errors**, all six
  from existing eval-runner loopback binds denied by the sandbox. Graphify AST
  refresh and whitespace checks passed. Commit blocked by sandbox denial of
  Git's worktree `index.lock`; changes remain unstaged.

## 2026-11-13 — R1 review fixes

- Fixed truncated PDF parsing with bounded streaming (30 MB), explicit oversize
  errors and a generated PDF whose requested page follows 4.3 MB of data.
- Keep successful short PDF page ranges instead of replacing them with Jina's
  whole-document text. Preserve title/markers and existing HTML reader fallback.
- Prefer paper identifier aliases over titles; reject conflicting title matches
  and avoid ambiguous title-only bridges between separate identities.
- Added both research tools to the read-only handoff allowlist; expanded the
  streaming/whole-turn regression for each tool and both routing handoffs.
- Focused source/router-bypass/tool-loop suite: **205 passed, 0 failed**;
  Graphify AST refresh and whitespace checks passed.
- Required full suite: **2322 passed, 3 skipped, 4 failed + 2 errors**, all six
  from sandbox-denied loopback binds in existing eval-runner tests. Git staging
  is denied at `.git/worktrees/r1-search-sources/index.lock`; no commit created.

## 2026-11-13 — R1 web and research sources

- Added Codex/Tavily/Serper/Exa/Brave/Jina/SearXNG/DDG adapters; first-two concurrent
  searches, sequential supplementation, DDG last resort, URL agreement ranking,
  persistent per-provider caps (Serper lifetime) and auth/quota cooldowns.
- Added PDF paging and guarded Jina reader fallback, academic metadata merging
  across DOI/arXiv/title aliases, Wikipedia summaries, registry labels/rate limits,
  provider status API, configuration docs and offline HTTP/PDF regressions.
- Search-model selection follows the approved gateway model policy; credentials
  remain in request headers and never appear in provider status or error output.
- Focused source/capability/model validation: **93 passed**. Required full suite:
  **2305 passed, 3 skipped, 4 failed + 2 errors** (sandbox-denied loopback binds).
  Graphify AST refresh and whitespace checks passed; Git staging is denied at
  `.git/worktrees/r1-search-sources/index.lock`, so no commit could be created.

## 2026-10-28 — P1.10 optional router bypass

- Validation review: defer handoff closure until handler entry. Unknown tools,
  malformed JSON and schema rejections retain both handoffs; handlers that start
  still prevent routing replay even if they fail. Expanded temporary-DB HTTP
  regressions cover both clients and handoffs, narration reset, unchanged original
  routing/planning and retained schemas. Focused checks: **128 passed, 0 failed**.
- Full backend suite after validation review: **2244 passed, 3 skipped, 4 failed
  + 2 errors**; all six are sandbox-denied eval-runner loopback binds. Graphify
  AST refresh and whitespace checks passed. Git staging remains blocked at
  the worktree `index.lock`; validation-review changes remain uncommitted.
- Latency review: extracted the pure request/state/scope precheck and run it
  before gate catalog/content lookups, preserving signal precedence. A temporary-DB
  HTTP regression checks focused results turns with the flag on/off: no gate
  catalog read or whole-library content scan, and retrieval retains paper focus.
  Bypass suite: **82 passed**; Graphify AST refresh and whitespace checks passed.
- Review: resolve handoffs before executing tools; remove handoff schemas once
  work starts and reject late control calls inside the existing loop, without
  tool cards or execution charges. Completed actions, trust and budgets stay intact.
- Review: normalize both catalog/title tokens and task tokens before splitting
  words, preserving acronym case. Added NFC/NFD title, filename and input cases.
- Added temporary-DB reminder → late plan/library handoff regressions for streaming
  and whole-turn clients; verify one side effect, retained receipt and step limit,
  no reset, planner or router replay.
- Added deterministic signals for conversation/scope documents, local/vague/visual
  references, resume/comparison, named terms/IDs, folded distinctive title/filename
  tokens, request mode/collection and client/tool capability. Catalog errors retain routing.
- Added bypass-only handoff schemas and loop interception before any mixed-batch
  execution, with narration reset and original-message routing/planning continuation.
  Planned tasks keep their existing capabilities and persisted tool allowlist.
- Added gateway fallback before the first public delta and persisted bypass/signal/
  handoff/first-delta telemetry. Default flag false; model settings unchanged.
- Fake-client tests cover signal isolation, title acronyms/diacritics/stopwords,
  streaming/whole-turn clients, unchanged document/social routes, routing-event
  equivalence after library handoff, planning, mixed batches, gateway and catalog failures.
- Required full backend command after latency review: **2230 passed, 3 skipped, 4 failed + 2 errors**,
  all six from existing eval-runner socket binds denied by the sandbox. Final focused
  bypass checks: **82 passed, 0 failed**.
  Graphify AST refresh and whitespace checks passed. Live acceptance belongs to the head agent.
- Requested commit blocked: sandbox denies creation of
  `.git/worktrees/p1-10-router-bypass/index.lock`; implementation, tests and notes remain unstaged.

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
## 2026-10-27 — P1.8b blocking review fixes

- Enforce/persist per-task assistant tool scope and fail out-of-list calls;
  reset discarded streamed answers; merge durable results/DB output payloads
  into judge evidence and deduplicate final persisted step completions.
- Separate goal submission time from the startup-inclusive deadline, and use
  recursive type-aware JSON equality with CSV coercion confined to CSV checks.
- Added synthetic regressions for every finding in `test_agent_eval_runner.py`;
  focused runner/task/worker checks: 146 passed (6 socket tests excluded).
- Full backend run with existing virtualenv dependencies: 2050 passed, 3 skipped;
  4 failures + 2 errors are denied loopback binds. Graphify update passed;
  commit blocked by sandbox denial of the worktree Git `index.lock`.

## 2026-10-27 — P1.8b agent eval runner (suite sealed outside the repo)

- Backend eval mode (env only, off by default; refuses the production data dir, warns at startup):
  `AYA_EVAL_WEB_ALLOW=host:port` opens `web_fetch` to exactly that fixture server, and
  `AYA_EVAL_STUB_MACOS=1` makes Notes/Calendar/Apple Reminders/open_app/open_url/Spotify/notify record
  to `$APP_DATA_DIR/eval_calls.jsonl` (`app/core/eval_mode.py`).
- `scripts/run_agent_eval.py --suite DIR --out DIR [--only ids] [--model M]`: MANIFEST check, fixture web
  server on :8765, per-task HOME/data/backend, approvals per `approval_policy`, `restart_after_step`,
  wall-clock cap, `results.jsonl` + `report.md` (Wilson 95%). Evaluators in `scripts/agent_eval_lib.py`.
- `scripts/judge_agent_eval.py --run DIR`: `judge_rubric` criteria via the `judge_ag` harness role.
- Tests: `backend/tests/test_agent_eval_runner.py` (synthetic fixtures in `backend/tests/fixtures/agent_eval/`).
  Not yet run end to end against a live backend and the real suite.
- Review fixes: steps counted by distinct `step_id`; the suite's `max_steps` is sent as
  `AgentRunRequest.max_steps` (task budget / tool-loop allowance) and overruns fail the task; tool calls
  merged by (tool, args) identity; `network_host` is judged from the eval-mode audit
  (`$APP_DATA_DIR/eval_network.jsonl`, every redirect hop); one absolute deadline (startup, stream, restart,
  polling) with a watchdog; the judge sees outcomes and changed-file contents and rejects malformed replies.

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
