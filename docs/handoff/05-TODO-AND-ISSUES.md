# 05 — TODO và vấn đề tồn đọng (2026-11-14)

## 2026-11-14 — R2 lease, numeric expressions and redirect identity

- [x] Reject stale-owner results/cleanup after awaited operations, validate whole
  numeric expressions, and reconcile final source URLs without false domain coverage.
- [x] Add model/source/report ownership, numeric-fragment, redirect merge,
  alias/checkpoint migration, concurrent read and resume regressions.
- [x] Offline research/store/scheduler/source checks: **217 passed, 0 failed**;
  refresh Graphify AST and pass whitespace checks.
- [ ] Re-run full validation where fixture sockets are permitted: the required
  command on final code returned **2485 passed, 3 skipped, 4 failed + 2 errors**;
  all six failures/errors are existing eval-runner loopback binds denied here.
- [ ] Commit the corrections: sandbox denies worktree `index.lock` creation
  (`Operation not permitted`); approval policy provides no escalation.

## 2026-11-14 — R2 deadline, evidence associations and graceful stops

- [x] Bound source operations by wall time, preserve shared-quote associations,
  interrupt quota waits on finish and drain current steps at page/call limits.
- [x] Add migration/concurrency/resume and runner/scheduler regressions:
  offline research/source checks **174 passed, 0 failed**.
- [x] Refresh Graphify AST and pass whitespace checks.
- [ ] Re-run the full backend suite where fixture sockets are permitted:
  required command returned **2442 passed, 3 skipped, 4 failed + 2 errors**;
  all six failures/errors are existing eval-runner loopback binds denied here.
- [ ] Commit review fixes: sandbox denies the worktree's `index.lock`
  (`Operation not permitted`), with no escalation available in this session.

## 2026-11-14 — R2 hosted search and worker failure isolation

- [x] Route hosted research search through scheduling/accounting/quota backoff;
  charge logical calls once and preserve evidence after individual step failures.
- [x] Offline research/store/scheduler/source checks: **161 passed, 0 failed**;
  update environment comments/code map and refresh Graphify AST.
- [ ] Re-run full validation where fixture sockets are permitted: required
  command returned **2429 passed, 3 skipped, 4 failed + 2 errors**, all six
  failures/errors from existing eval-runner loopback binds denied by the sandbox.
- [ ] Commit these corrections: sandbox denies creation of the worktree's
  `index.lock` (`Operation not permitted`), with no approval escalation available.

## 2026-11-14 — R2 review corrections

- [x] Fix real HTTP 429 backoff, library passage merging/checkpoints, optional
  library failure isolation/empty-library skip, and research HTML/PDF truncation.
- [x] Add offline regressions: **147 passed, 0 failed**; refresh Graphify AST.
- [ ] Re-run full suite where fixture sockets are permitted: **2415 passed,
  3 skipped, 4 failed + 2 errors**, all six are sandbox-denied loopback binds.
- [ ] Commit review fixes; staging cannot create this worktree's `index.lock`
  (`Operation not permitted`), and session policy disallows escalation.

## 2026-11-14 — R2 research follow-up

- [x] Implement research storage/orchestration/API, read-only source cores, priority
  scheduling and deterministic R2 reports; focused compatibility: **294 passed**.
- [ ] R3: replace `write_report(run_id, store, deps)` with synthesis/verification.
- [ ] R4: wire desktop research controls and the chat/handoff entry point.
- [ ] Re-run the full backend suite on a host that permits fixture sockets:
  required command returned **2404 passed, 3 skipped, 4 failed + 2 errors**;
  all six failures/errors are existing eval-runner loopback binds denied by the sandbox.
- [x] Refresh Graphify AST and pass whitespace checks.
- [ ] Commit R2: `git add` cannot create `.git/worktrees/r2-research-orchestrator/index.lock`
  (`Operation not permitted`); session approval policy does not permit escalation.

## 2026-11-13 — R1 search navigation and fallback review

- [x] Preserve original source URLs and trigger fallback based on usable URL hits;
  source regressions: **75 passed, 0 failed**. Update map and refresh Graphify.
- [ ] Re-run full backend validation where loopback binds are permitted:
  required command returned **2343 passed, 3 skipped, 4 failed + 2 errors**;
  all six failures/errors are sandbox-denied existing eval-runner binds.
- [ ] Commit the fixes: staging cannot create the worktree `index.lock`
  (`Operation not permitted`).

## 2026-11-13 — R1 model-facing output validation

- [x] Preserve substantive tool content with citation IDs generically; cover
  paper metadata/PDFs, unverified Codex summaries and full Wikipedia extracts
  through actual streaming/whole-turn model messages. Focused: **145 passed**.
- [x] Refresh Graphify AST and check diff whitespace.
- [ ] Re-run full backend validation where loopback binds are permitted:
  required command returned **2329 passed, 3 skipped, 4 failed + 2 errors**;
  all six failures/errors are sandbox-denied existing eval-runner binds.
- [ ] Commit the formatter/tests/docs: staging cannot create
  `.git/worktrees/r1-search-sources/index.lock` (`Operation not permitted`).

## 2026-11-13 — R1 review validation

- [x] Fix complete-PDF download limits, paged-PDF/Jina replacement, conflicting
  paper identifiers and research-tool handoffs with generic regressions.
- [x] Focused source/router-bypass/tool-loop suite: **205 passed, 0 failed**;
  update code map and refresh Graphify AST.
- [ ] Re-run the full backend suite where loopback binds are permitted. Required
  command: **2322 passed, 3 skipped, 4 failed + 2 errors**; all six failures/errors
  are existing eval-runner tests denied by the sandbox's socket policy.
- [ ] Commit the review fixes: `git add` cannot create
  `.git/worktrees/r1-search-sources/index.lock` (`Operation not permitted`).

## 2026-11-13 — R1 source validation

- [x] Implement the approved R1 source layer, update capability expectations,
  environment documentation and code map; refresh Graphify AST.
- [ ] Re-run the complete backend suite on a host that permits loopback socket
  binding: the existing eval-runner integration tests raise `PermissionError`
  under this sandbox. Required full run: **2305 passed, 3 skipped, 4 failed +
  2 errors**. Focused source/capability/model checks: **93 passed**.
- [ ] Commit R1 changes: `git add` was denied creation of
  `.git/worktrees/r1-search-sources/index.lock`; changes remain unstaged.

## 2026-10-28 — P1.10

- [x] Validation review: unknown/invalid calls retain handoffs until real handler
  entry; both handoffs tested after unknown, malformed and schema-invalid calls
  with streaming/whole-turn clients. Focused checks: **128 passed, 0 failed**.
- [x] Latency review: short-circuit request/state/scope signals before gate
  catalog/content lookups; focused document HTTP regressions cover flag on/off.
  Bypass suite: **82 passed**. No model/default-flag changes.
- [x] Review: prevent late handoffs from replaying completed actions; keep the
  same loop's receipts, trust and budget. Normalize before title/filename/input
  tokenization; NFC/NFD equivalence and temporary-DB reminder regressions added.
- [x] Pure conservative gate, bypass-only handoffs, original-request continuation,
  first-delta/gate/handoff telemetry and fallback before public gateway output.
- [x] Flag defaults off; deterministic social shortcuts and document routing stay.
- [x] Offline fake-client regression coverage; previous combined bypass/tool-loop/task/worker
  checks: **211 passed, 0 failed**. Latest bypass checks: **82 passed, 0 failed**;
  Graphify AST refresh and whitespace checks passed.
- [ ] Full suite needs loopback-bind permission: the required command returned
  **2244 passed, 3 skipped, 4 failed + 2 errors** after validation review; all six are existing eval-runner
  tests blocked by this sandbox's socket policy.
- [ ] Commit the P1.10 changes: staging is blocked by sandbox denial of
  `.git/worktrees/p1-10-router-bypass/index.lock`; changes remain unstaged.
- [ ] Head agent, after merge: G1 acceptance/vague-ref counts and gate-only document
  replay with zero bypass misses; independent G2 routing probe; G3 RAG v5 rerun
  together with P1 agent gate; warm first text ≤ 4 s and unchanged document latency.
  Keep `AYA_ROUTER_BYPASS` off until all acceptance gates pass.

## 2026-10-27 — P1.9

- [x] Review: desktop-supported early cards, authoritative failure-answer resets,
  and bounded empty-report retries with consumer/attached-chat regressions.
- [x] Stream agent text and pending tool-call cards, retaining narration and full answers.
- [x] Keep task step answers private; stream final reports and reconcile live/durable events.
- [x] Add streaming/retry/empty-output and attached-chat regressions; refresh Graphify AST.
- [x] Required backend suite after review fixes **2064 passed, 3 skipped, 0 failed**.
- [ ] Commit P1.9 review fixes: sandbox denied creation of
  `.git/worktrees/p1-9-streaming/index.lock`; seven changed files remain unstaged.
- [ ] Head agent: warm-backend “Mấy giờ rồi cậu?” first text ≤ 4.5 s.

## 2026-10-27 — P1.4

- [x] Router-first planning triage, tool argument repair/schema validation and process-wide web/code rate limits.
- [x] Wake workers on start/resume; filter internal execution history; bound retry tool events to the step's last 20.
- [x] Persist completion before external dispatch; durable per-notification/channel deduplication.
- [x] Required backend suite **2006 passed, 3 skipped, 0 failed**; Graphify AST refresh and whitespace checks passed.
- [ ] Commit P1.4 changes: sandbox denied creation of `.git/worktrees/p1-4-speed/index.lock`;
  implementation and tests remain unstaged in this worktree.
- [ ] Head agent: measure live “Mấy giờ rồi cậu?” first-text latency before/after (target ≤ 5 s).
## 2026-10-27 — P1.8b review validation

- [x] Fix all five blocking runner findings with synthetic regressions; refresh Graphify.
- [ ] Full backend validation needs installed dependencies and permission to bind
  loopback sockets. The exact required command fails collection on missing
  FastAPI/other packages; existing virtualenv dependencies allow the tests to run,
  yielding 2050 passed, 3 skipped, 4 failures + 2 errors from denied socket binds.
  Sealed suite remains untouched.
- [ ] Commit the review fixes: requested staging failed because the sandbox
  denies creation of `.git/worktrees/p1-8b-runner/index.lock`.

## 2026-10-25 — P1.6b

- [x] P1.6 primitives ported onto the task runner; waiting tasks resume through the worker.
- [ ] Run the full backend suite and `graphify update .` (neither could be run in the
  authoring session: command execution was denied).

## 2026-10-25 — Agent tasks

- [x] P1.1: persistent task storage, lifecycle API, event contract and isolated tests.
- [x] 2026-10-25 P1.2: foreground planning/execution/replanning runner, persisted
  events/usage, chat SSE, budget/question stops and cancellation through approvals.
- [x] 2026-10-25 P1.2 review: charge final-report time and stop completion when
  it exceeds the budget; fake-clock regressions include the exact-limit boundary.
- [x] 2026-10-25 P1.2 review: check step allowance before starting work and
  preserve received results before budget exits; regressions verify file writes
  and persisted progress at and beyond the step allowance.
- [ ] P1.2 head-agent acceptance: isolated-stack 3-step file task plus simple-time
  question (no task). Offline regressions cover these paths with fake clients.
- [ ] P1.3: background task execution/restart recovery.
- [x] P1.6: resume waiting-user tasks on the next message (done in P1.6b, see above).
- [ ] P1.7: task UI and user controls.
## 2026-10-25 — P1.5 capability awareness

- [x] Generate capability sheets from registered tools and add them to chat
  and agent prompts; preserve RAG prompts and tool allowlist behavior.
- [x] Teach the router that casual requests to act need agent tools, while
  questions about Aya's abilities stay chat; add deterministic regressions.
- [x] Review follow-up: greeting-prefixed actions reach the router; test agent
  routing when allowed, chat downgrade when disallowed and social-only shortcuts.
- [x] Further review (2026-10-25): restrict every social shortcut to complete,
  unambiguous messages; cover vi/en social-prefixed actions with assistant tools
  enabled/disallowed, acknowledgements and standalone social messages.
- [ ] Head agent: check the isolated stack live — “cậu làm được gì?” lists
  real capabilities; “mở nhạc đi” routes to agent.
- [ ] Commit review follow-ups on `harness/p1-5-capabilities`: `git add` was
  denied creation of `.git/worktrees/p1-5-capabilities/index.lock` by the
  session sandbox again on 2026-10-25. Changes remain unstaged; validation: **1725 passed,
  3 skipped, 0 failed**, Graphify AST refresh and diff whitespace check passed.

## 2026-10-25 — Voice follow-up

- [x] Plan A: local speech sidecar, backend speech API, setup docs and offline tests.
- [x] 2026-10-25 audit: fix relative sidecar paths and streamed fence cleanup;
  cover venv symlinks and offline import flags without models or network.
- [x] 2026-10-25 further audit: nested list/quote code fences stay silent;
  preserve nested Markdown link labels and strip their complete targets, with
  offline cleaner and Japanese-request 204 regressions.
- [ ] Head agent: run the online TTS installer and listen to real Vietnamese,
  Vietnamese anime-clone and Japanese output (including pitch/speed controls).
- [ ] Plan B: desktop voice controls, playback/lip sync, visible character credits.
- Seatbelt integration tests require a host that permits `sandbox-exec`;
  nested sandbox denial is now reported as a capability skip.

## Assistant phase — 2026-10-22 (see 13-ASSISTANT-TOOLS-PLAN.md)

- [ ] Codebase cleanup: split backend/app/api/agent.py (harness run
  split-agent), ChatPage/types (harness run frontend-chat), group scripts/,
  retire the legacy evidence_validator path, refresh CODEBASE_MAP.md.
- [x] 2026-10-27 P1.9: stream tool-call deltas so tool cards appear while the model writes.
- [x] 2026-10-27 P1.4: safely repair complete JSON objects followed by gateway junk;
  reject unrepairable/schema-invalid arguments before executing. Gateway translation
  itself is unchanged.
- [ ] Loopback token for the backend + `aya doctor`.
- [ ] Phase 3: reminders/heartbeat, Telegram (owner only), daily-brief skill.
- [ ] Phase 4: skills (research, paper-digest, file-organizer, data-analyst),
  TTS voice with lip sync from audio, VRM/own Live2D model for Aya.
- [ ] Deploy the assistant + Live2D build to production (:7777/:5173).

## RAG v1 RC1 — 2026-10-22

- [x] Regression on the final SHA, acceptance content-judged, rollback drill
  (`12-RELEASE-RAG-V1.md`).
- [x] Production rollout of `3219b2bc` on `data/` — 2026-10-22 20:28 (backup `data/backups/pre-rc1-20261022-202756`).
- [ ] Maintenance backlog (not blockers): see `12-RELEASE-RAG-V1.md` §Maintenance.
- [ ] Next phase: OpenClaw-style tools on the router → evidence → answer →
  verify pipeline.

## After sealed v3 — 2026-10-21

- [ ] Re-measure v3 (dev) after `072984ec`; then a fresh sealed v4 written by
  an independent model for the next out-of-sample number.
- [ ] Remaining v3 failures not yet addressed: v3-09 "which of my files come
  from EIA" (library inventory by publisher), v3-04/74 single facts missed in
  long native PDFs, v3-59 image-only slide content, v3-56/70 required details.
- [x] Catalog titles (generic, legacy-encoded, section names) — `ae92f69f`.
- [ ] Catalog: wrong card language (English LIGO papers tagged vi),
  `_phrase_document_count` tokenizes with [A-Za-z0-9] so Vietnamese words with
  diacritics break.
- [x] Startup backfill no longer activates an empty placeholder for an
  interrupted ingest — `0f267f61`.
- [ ] 9router listens on *:20128 (all interfaces) — bind to localhost.

## Sealed evaluations — 2026-10-16

- [ ] Sealed v2 50.0% (gate 85%). Rerun of failing conversations after the
  generic fixes pending; scanned tables (OCR diacritics "Quàng Ninh", split
  columns) and FOMC Table 1 retrieval are the next suspects.

- [ ] Scanned PDFs keep the filename as title (OCR headers are garbled), so
  description matches favour titled documents (sealed v1 s22). Derive a
  title from OCR text robustly or use the router's view of content.
- [ ] Hybrid fusion vs pure vector: doc hit@1 0.850 vs 0.917 on sealed v1
  (card routing / channel mix suspected; FTS pruning and FTS weight were
  measured and rejected).
- [x] Re-login 9router account the uclab account (401) — done 2026-10-16.
- [x] v1 comparisons s33/s34/s40/s41 — correct on rerun7 (2026-10-16).

---
Older entries (before 2026-10-13): `docs/archive/handoff-history/05-TODO-AND-ISSUES-before-2026-10-13.md`.

- 2026-10-25 — Backend hang: all threads blocked in SQLite `connect`/`close` (SQLite 3.51.0 unix-VFS deadlock, fixed in 3.51.2). Anaconda's sqlite upgraded to 3.53.4; frontend polls now time out so a hung backend cannot exhaust the browser's connections. If it recurs: `kill -USR1 <pid>` + `sample <pid> 2`, check `sqlite3.sqlite_version`.

- 2026-10-25 — Agent core follow-ups (from P1.3/P1.7 reviews, minor):
  - fixed 2026-10-27 P1.4: task start/resume wakes the worker immediately;
  - fixed 2026-10-27 P1.4: internal `execution` notes stay out of public streams/inspection;
  - fixed 2026-10-27 P1.4: retry prompts carry only the step's last 20 tool events;
  - resuming after an approval rebuilds only one synthetic tool turn (multi-call turns lose siblings);
  - fixed 2026-10-27 P1.4: completion marker precedes external delivery; notification/channel claims prevent re-sends;
  - the event hub drops the oldest queued event when full (an approval card can be lost);
  - UI: latest note may show a finished step's note; a resumed task from an earlier turn can get two cards;
    the ask card is not restored on reload; the stage lists at most 4 active tasks; no frontend test runner.
  - fixed 2026-10-27 P1.4: two-stage triage (configured Luna router first, Sol planning only when needed); live latency check pending.

- 2026-10-27 — Agent-route latency after P1.9 streaming: "Mấy giờ rồi cậu?" first text 4.9–6.9 s (was 9 s;
  target ≤ 4.5 s). Remaining cost: router ~3 s (long prompt on cx/gpt-6-luna) + Sol first token ~2 s + 9router
  overhead per call (~1.6–2.3 s). Next levers: shorter router prompt for turns without library context,
  or start the agent turn speculatively while the router decides.
- P1 acceptance gate (sealed agent suite v1, ~300 Sol calls) not run yet: waiting for the user to pick a
  time when Aya is not needed (a full run can exhaust both Codex accounts shared with 9router).


## 2026-10-29 — after P1.10 gates

- D03-style stall: an agent turn sometimes never gets its second Sol reply (reproduced 2/4 via the
  backend, 0/2 direct). Cause unproven (9router had uclab at `401 token_revoked`). Repro + HTTP
  logging: `data/retrieval_eval/agent-eval/debug-d03/` (`site/sitecustomize.py` logs every httpx
  call). Data-frame deadline added to agent streams; recheck after uclab re-login.
- v5-32 t2: with assistant tools, a "document not in the library" follow-up went to the agent route
  and was answered from the web instead of "not in the library" (router choice, not the bypass).
- resume_after_restart tasks plan fewer steps than the suite's restart point; durable resume is
  not exercised by the bypass path unless Sol calls `plan_task`.
- Japanese voice: per-segment translation costs ~3 s (router model through 9router).
- ~~9router uclab connection `401 token_revoked`~~: re-logged in by the user 2026-11-08.
- Web search sources configured 2026-11-13 (keys in backend/.env, never committed): Tavily, Serper,
  Jina, OpenAlex, Decodo; local SearXNG on :8888 (`~/.aya/searxng`, started by `aya-up.sh`); Codex
  hosted search works only through `/v1/responses` (chat/completions + web_search crashed 9router).
  Decodo (search + scrape fallback) to be added after R1 merges.
- Harness: `*_c5` roles on the user's "Claude 500" reseller (dev agents only, never Aya).
