# 02 — Tools

Every tool: name, JSON schema, permission (`auto` / `ask` / `ask_after_untrusted`),
`side_effects`, timeout, output cap, handler in `backend/app/tools/<group>.py`,
registration in `build_registry`, tests with fakes (no network, no real apps).
Output from outside (web, mail, files, screen, MCP) is wrapped as untrusted.

## Existing (2026-10-25)

web_search, web_fetch · list_dir, read_file, search_files, write_file, edit_file,
move_path, trash_path · run_python, run_shell (Seatbelt) · create/list/cancel_reminder ·
apple_reminders_add/list/complete, calendar_events, calendar_add_event, notes_create,
open_app, open_url · spotify · search_library.

## To add, by priority

### P1 (with the agent core)

| Tool | What | Implementation | Permission |
|---|---|---|---|
| `ask_user` | structured question to the user | task waits for the answer | auto |
| `remember` / `recall` | long-term memory with provenance | existing L3 store | auto |
| `notify` | in-app/macOS/Telegram notification | existing deliverers | auto |
| `schedule_task` | run an agent goal later or on a schedule (cron-like) | reminder worker extended to tasks | ask (shows schedule) |
| `spawn_subtask` | parallel child task | task runner | auto (budgeted) |
| `read_artifact` | read a stored large tool output | artifact store | auto |

### P2 (reach)

| Tool | What | Implementation | Permission |
|---|---|---|---|
| `browser_open/read/click/type/screenshot` | drive a real browser for sites without APIs | Playwright (Chromium, dedicated profile in `~/Aya/.browser`), accessibility snapshot as text | read auto; click/type ask after untrusted; submit/purchase/login always ask |
| `screen_look` | "what's on my screen?" | `screencapture` + VISION_MODEL description; needs Screen Recording permission | ask (first use per session) |
| `clipboard_get/set` | read/write clipboard | `pbpaste`/`pbcopy` | get ask, set auto |
| `make_document` | create docx/xlsx/pptx/pdf/markdown from content | python-docx, openpyxl, python-pptx, reportlab inside the sandbox; files in workspace | auto (workspace) |
| `read_document` | read docx/xlsx/pptx/pdf outside the library | existing parsers | auto inside workspace, ask outside |
| `mail_search/read/draft/send` | Apple Mail | JXA (like Reminders); send = draft + approval | read auto, draft auto, send always ask |
| `contacts_find` | Contacts lookup | JXA | auto |
| `messages_send` | iMessage/SMS | JXA/AppleScript | always ask (shows recipient + text) |
| `system_control` | volume, brightness, dark mode, Do Not Disturb, battery/wifi info, sleep display | osascript / `shortcuts run` | info auto; changes auto (reversible) |
| `shortcuts_run` | run a macOS Shortcut by name | `shortcuts run` | ask |
| `weather`, `geo` | forecast, place lookup | Open-Meteo, Nominatim (no keys) | auto |
| `github` | issues/PRs/repos | `gh` CLI | read auto, write ask |
| `mcp_*` | tools of configured MCP servers | MCP client (stdio/http), `~/Aya/mcp.json`, each server approved once, tools namespaced `mcp.<server>.<tool>`, output untrusted | per server: read auto / write ask |

### P4 (presence)

| Tool | What | Implementation |
|---|---|---|
| `speak` | say something now (announcement, reminder) | TTS sidecar |
| `listen` | voice input (push-to-talk) | local STT (Zipformer/sherpa-onnx like the user's RAG-TTS, or Whisper via MLX) |

## Order inside P2

browser → documents → screen/clipboard → mail/contacts → system/shortcuts →
weather/geo → github → MCP client → messages. Rationale: browser and documents
unlock the most tasks; MCP multiplies tools but needs the safety work of 04.

## Per-tool acceptance

Unit tests with fakes; one live smoke per tool on the isolated stack; a task in
the agent suite that needs it; for side-effecting tools a test that the approval
card shows the exact action and that deny leaves no trace.
