# 00 — Roadmap: from tool-using chat to an agent that does everything

## What "does everything" means for Aya

Given a goal in plain Vietnamese or English, Aya:

1. understands it and asks only when something essential is missing;
2. plans the steps and shows the plan;
3. carries them out with tools — local (files, code, apps, screen, browser) and
   connected (mail, calendar, messages, web, MCP servers);
4. keeps going when the goal takes minutes or hours (in the background, across
   restarts), and reports back where the user is (app, notification, Telegram);
5. asks for approval before anything risky, and can undo what she did;
6. remembers what matters (preferences, ongoing projects, how the user likes
   things done) and gets better at recurring jobs;
7. stays Aya: in character, with voice and face, and honest about what she did
   and did not do.

## Where we are (2026-10-25)

| Area | Today | Gap |
|---|---|---|
| Loop | One turn = model ↔ tools, up to a step cap; approvals; untrusted-output wrapping | No task object, no plan, no resume after restart, no background run, no budgets |
| Tools | 23: web, files, sandboxed code, reminders, Apple Reminders/Calendar/Notes, open app/url, Spotify, library search | No browser control, mail, messages, contacts, screen, clipboard, documents (docx/pdf/xlsx), system settings, MCP |
| Skills | Planned in 13-ASSISTANT-TOOLS-PLAN, none loaded | No skill format/loader, no scheduled workflows, no triggers |
| Awareness | Asked "what can you do?", Aya lists writing/coding help and none of her tools (seen 2026-10-25) | Capabilities must come from the registry, in every route |
| Memory | L0–L3 conversation/long-term memory | No `remember`/`recall` tools, no task/project memory, no learned procedures |
| Channels | App, macOS notifications, Telegram (owner only, needs token) | Voice input; spoken notifications |
| Safety | Permission levels, approvals, Seatbelt, secret dirs, owner-only Telegram | No audit UI, no undo log beyond Trash, no loopback token, no budgets/rate limits |
| Eval | RAG sealed sets; tool unit tests | No agent task suite, no injection suite at agent level |

## Phases and gates

Each phase is a set of work packages (07-EXECUTION.md). A phase is done when its
gate passes on the isolated stack; then it is rolled out with backup + drill.

**P1 — Agent core.** Task model, plan/execute/replan, background runs that
survive restart, capability awareness, `ask_user`, `remember`/`recall`, budgets,
tool-output summarisation, streaming tool-call deltas.
Gate: agent suite v1 (06-EVAL) ≥ 80% task success on 30 multi-step tasks; 0
unapproved side effects; RAG gates unchanged; a task interrupted by a backend
restart resumes and finishes.

**P2 — Reach.** Browser automation (Playwright), screen + vision, clipboard,
documents (docx/xlsx/pdf/pptx), Apple Mail + Contacts + Messages, system controls,
weather/maps, MCP client.
Gate: per-tool tests + agent suite v2 (60 tasks incl. P2 tools) ≥ 80%; injection
suite (web page, email, file) 0 escapes.

**P3 — Skills and workflows.** Skill format + loader + first-party skills
(research, daily brief, paper digest, inbox triage, file organizer, data analyst,
meeting prep), scheduled workflows and triggers (time, folder, mail, calendar).
Gate: each skill's acceptance task passes 5/5 runs; scheduled daily brief runs 3
days in a row without help.

**P4 — Presence.** Emotion-aware avatar and voice, interruptions, voice input
(push-to-talk, then wake word), spoken notifications, faster voice start.
Gate: user listening/acceptance session; voice start ≤ 3 s after backend start;
first audio ≤ 1.5 s after the first sentence is written.

**P5 — Hardening.** Audit UI, undo log, loopback token + `aya doctor`, budgets
UI, packaging (Tauri app with sidecars), backups.
Gate: `aya doctor` green on a clean machine; security review of the diff since P1.

P1 comes first because every later tool and skill needs tasks, plans,
background runs and awareness. P2 and P4 can run in parallel with P3 once P1's
task API is stable.
