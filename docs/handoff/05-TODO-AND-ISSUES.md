# 05 — TODO và vấn đề tồn đọng (2026-10-10)

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
- [ ] Stream tool-call deltas so tool cards appear while the model writes.
- [ ] Junk appended to function arguments by the model/gateway
  (`…)"} regretting? ma`) — investigate 9router argument translation.
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
  - task start waits for the worker's next 2 s tick (wake the worker on start);
  - `watch_task` replays the `execution` note (up to 40k chars of history) to every client — filter it;
  - each step prompt carries all tool events of the task (bound it to the retried step);
  - resuming after an approval rebuilds only one synthetic tool turn (multi-call turns lose siblings);
  - a crash between deliverers and the `completion_delivered` note re-sends macOS/Telegram notices;
  - the event hub drops the oldest queued event when full (an approval card can be lost);
  - UI: latest note may show a finished step's note; a resumed task from an earlier turn can get two cards;
    the ask card is not restored on reload; the stage lists at most 4 active tasks; no frontend test runner.
  - Agent route latency: plan triage runs on Sol for every agent turn ("mấy giờ rồi" 9 s) → P1.4 two-stage triage (Luna first).

