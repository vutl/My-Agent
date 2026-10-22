# 13 — Assistant phase: tools, skills and Aya's anime presence

Decided with the user on 2026-10-22: OpenClaw is the inspiration (one local
gateway that owns sessions, tools, a scheduler and channels); the tool and
skill choices below are ours. Aya keeps her anime voice everywhere and gets
an on-screen anime character.

## Principles (lessons from OpenClaw's incidents)

- First-party tools only; no skill marketplace. User skills are plain
  `SKILL.md` folders, shown in full and approved before they are enabled.
- Backend listens on 127.0.0.1 with a token (a web page can reach localhost).
- Tool output (web pages, files, messages) is untrusted data: wrapped and
  labelled, never followed as instructions; after untrusted content enters a
  turn, side-effecting tools need approval.
- Every tool has a permission level: `auto` or `ask` (UI shows the exact
  action; Approve / Deny). Every call is logged (`tool_calls` table).
- Document questions keep the measured RAG pipeline (v1, 86% sealed); the
  library is one tool among others for the agent loop.

## Architecture

- `app/llm/openai_client.py`: native function calling (`tools`,
  `tool_choice`, `tool_calls`; 9router passes them to cx/gpt-6.1-sol —
  verified 2026-10-22).
- `app/tools/`: registry (name, JSON schema, permission, async handler) and
  the tool modules; `app/agents/tool_loop.py`: model ↔ tools loop (max steps,
  approvals via asyncio futures + `POST /agent/runs/{run}/tool-calls/{id}/decision`).
- Router gets a new route `agent` (act / fresh or external information);
  `chat` and the RAG routes are unchanged.
- SSE: `tool.call.started`, `tool.call.approval_required`,
  `tool.call.completed`, `avatar.state`; UI renders tool cards with
  Approve/Deny in the message, and the avatar reacts.

## Tools

| Group | Tools | Implementation | Permission |
|---|---|---|---|
| Library | `search_library`, `add_to_library` | existing RAG search / ingest | auto |
| Web | `web_search`, `web_fetch` | Tavily (if `TAVILY_API_KEY`), else DuckDuckGo (`ddgs`); httpx + trafilatura; URL citations `[W#]` | auto |
| Files | `list_dir`, `read_file`, `search_files`, `write_file`, `edit_file`, `move_path`, `delete_path` | Python + ripgrep; workspace `~/Aya` | auto inside the workspace; ask outside / for delete |
| Code | `run_python`, `run_shell` | deny-default Seatbelt profile, no network by default, cwd pinned, timeout, output cap | allow-listed commands auto; else ask |
| Time | `create_reminder`, `list_reminders`, `cancel_reminder`, recurring tasks, heartbeat | SQLite table + lifespan worker (lease pattern) | auto |
| Channels | Telegram in/out, macOS notifications | aiogram long polling, owner chat id only; Tauri notification | messaging others: ask |
| Memory | `remember`, `recall` | existing long-term memory | auto |
| Later | browser automation, mail, calendar | Playwright, connectors | ask |

## Skills (first-party, `skills/<name>/SKILL.md`)

research (search → read 3–5 sources → cited brief), daily-brief (morning
reminders + followed topics to Telegram), paper-digest (summary and questions
for a newly added document), file-organizer (tidy a folder, approval per
move), data-analyst (CSV/Excel → Python → chart + summary).

## Anime presence

- Voice: one persona prompt shared by chat, RAG and the tool loop; tool steps
  narrated in character ("để tớ tìm thử nha~").
- Character: `Avatar` interface (`setState(idle|thinking|tool|talking|happy|
  confused)`, `setMouth(level)`). Step 1: built-in SVG character with
  expressions and mouth animation. Step 2: VRM (three-vrm, MIT) with a model
  the user owns (VRoid Studio) — lip-sync and expressions; Live2D only with a
  confirmed licence.

## Phases and gates

1. Tool loop + approvals + web tools + avatar MVP. Gate: tool suite (search,
   fetch, multi-step, approval deny/approve, injection probe) passes; RAG
   regression (acceptance content, vague-ref) unchanged.
2. Files + code sandbox. Gate: sandbox escape/network/path tests; approval on
   every destructive op.
3. Reminders, heartbeat, Telegram, daily-brief. Gate: reminders survive
   restart; only the owner can talk to the bot.
4. VRM avatar, voice (TTS), remaining skills, `aya doctor`.

## Progress — 2026-10-22

- Phase 1 done (`799c9a00`, `2151c920`): tool loop with approvals, web
  search/fetch with [W#] citations, library tool, tool cards in the UI.
  Live: Apple news, python.org version, SJC gold price (cited table).
- Phase 2 done (`e530e8ce`, `ce07ca8c`): file tools (secrets never readable,
  workspace writes direct, outside asks, trash reversible) and Seatbelt
  sandbox for Python/shell (tests: escape, secrets, .env, network, timeout).
  Live: sales CSV + chart inline; delete → approval card → Trash.
- Avatar: Live2D Mao (`52ba6339`) — expressions, gaze, lip sync; SVG fallback.
- Not yet in production (:7777 runs RAG RC1); test stack :7790/:5174.

Open:
- The model/gateway sometimes appends junk to function arguments
  (`…)"} regretting? ma`), so the first run_python fails and Aya retries;
  investigate 9router's argument translation.
- Long silent first steps (big code in one tool call): stream tool-call
  deltas so the card appears while the model is still writing.
- Backend has no loopback token yet (plan principle); add with `aya doctor`.
- Phase 3: reminders/heartbeat (SQLite + lifespan worker), Telegram
  (aiogram, owner-only), daily-brief skill. Phase 4: skills, TTS voice.
