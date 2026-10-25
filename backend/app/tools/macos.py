"""macOS integrations: notifications, Reminders, Calendar, Notes, opening apps and links.

Apple apps are driven through JavaScript for Automation (``osascript -l
JavaScript``) with arguments passed as JSON (never spliced into the script).
The first call to each app makes macOS ask the user for permission; a refusal
comes back as a plain message for the model. Reads run directly; writes are
side effects (approval after untrusted content entered the turn).
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
from typing import Any
from urllib.parse import urlparse

from app.tools.registry import Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

_TIMEOUT = 45
_NOT_AUTHORIZED = ("-1743", "Not authorized", "not allowed assistive", "-10004")


class AppleScriptError(RuntimeError):
    pass


async def run_jxa(script: str, payload: dict[str, Any]) -> Any:
    process = await asyncio.create_subprocess_exec(
        "/usr/bin/osascript", "-l", "JavaScript", "-e", script, json.dumps(payload, ensure_ascii=False),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=_TIMEOUT)
    except asyncio.TimeoutError:
        process.kill()
        raise AppleScriptError("the app did not answer in time") from None
    if process.returncode != 0:
        message = stderr.decode("utf-8", errors="replace").strip()
        if any(marker in message for marker in _NOT_AUTHORIZED):
            raise AppleScriptError(
                "macOS has not allowed Aya to use this app yet: open System Settings → Privacy & Security → "
                "Automation (and Reminders/Calendars) and allow it, then try again."
            )
        raise AppleScriptError(message[-300:] or "osascript failed")
    text = stdout.decode("utf-8", errors="replace").strip()
    return json.loads(text) if text else None


def notify(title: str, body: str) -> None:
    """A native macOS notification (works while the backend runs, UI open or not)."""
    if sys.platform != "darwin":
        return
    script = 'on run argv\ndisplay notification (item 2 of argv) with title (item 1 of argv) sound name "Glass"\nend run'
    result = subprocess.run(["/usr/bin/osascript", "-e", script, title[:120], body[:240]], timeout=15,
                            check=False, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("macos_notify_failed code=%s error=%s", result.returncode, result.stderr.strip()[:200])


async def _jxa_tool(script: str, payload: dict[str, Any], summary: str) -> ToolResult:
    try:
        result = await run_jxa(script, payload)
    except AppleScriptError as exc:
        return ToolResult(content=str(exc), summary="Chưa được macOS cho phép" if "allowed" in str(exc) else
                          "Ứng dụng macOS báo lỗi", ok=False)
    return ToolResult(content=json.dumps(result, ensure_ascii=False)[:12_000], summary=summary, data={"result": result})


# --- Reminders -------------------------------------------------------------
_REMINDERS_ADD = """
function run(argv) {
  const a = JSON.parse(argv[0]);
  const app = Application('Reminders');
  const list = a.list ? app.lists.byName(a.list) : app.defaultList();
  const props = { name: a.title };
  if (a.notes) props.body = a.notes;
  if (a.due) props.dueDate = new Date(a.due);
  const reminder = app.Reminder(props);
  list.reminders.push(reminder);
  return JSON.stringify({ list: list.name() });
}"""

_REMINDERS_LIST = """
function run(argv) {
  const a = JSON.parse(argv[0]);
  const app = Application('Reminders');
  const out = [];
  for (const list of app.lists()) {
    const items = list.reminders.whose({ completed: false })();
    for (const item of items) {
      const due = item.dueDate();
      out.push({ title: item.name(), due: due ? due.toISOString() : null, list: list.name() });
      if (out.length >= a.limit) break;
    }
    if (out.length >= a.limit) break;
  }
  return JSON.stringify(out);
}"""

_REMINDERS_COMPLETE = """
function run(argv) {
  const a = JSON.parse(argv[0]);
  const app = Application('Reminders');
  for (const list of app.lists()) {
    const items = list.reminders.whose({ name: a.title, completed: false })();
    if (items.length) { items[0].completed = true; return JSON.stringify({ done: a.title, list: list.name() }); }
  }
  return JSON.stringify({ done: null });
}"""


async def apple_reminders_add(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    title = " ".join(str(arguments.get("title") or "").split())[:200]
    if not title:
        return ToolResult(content="Missing title.", summary="Thiếu tên lời nhắc", ok=False)
    payload = {"title": title, "due": arguments.get("due") or None, "notes": arguments.get("notes") or "",
               "list": arguments.get("list") or ""}
    return await _jxa_tool(_REMINDERS_ADD, payload, f"Thêm vào Apple Reminders: {title}")


async def apple_reminders_list(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    result = await _jxa_tool(_REMINDERS_LIST, {"limit": max(1, min(int(arguments.get("limit") or 30), 100))},
                             "Xem Apple Reminders")
    if result.ok:
        result.summary = f"Apple Reminders: {len(result.data.get('result') or [])} việc chưa xong"
    return result


async def apple_reminders_complete(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    title = str(arguments.get("title") or "").strip()
    result = await _jxa_tool(_REMINDERS_COMPLETE, {"title": title}, f"Đánh dấu xong: {title}")
    if result.ok and not (result.data.get("result") or {}).get("done"):
        return ToolResult(content="No open reminder with exactly that title.", summary="Không thấy lời nhắc đó",
                          ok=False)
    return result


# --- Calendar --------------------------------------------------------------
_CALENDAR_EVENTS = """
function run(argv) {
  const a = JSON.parse(argv[0]);
  const app = Application('Calendar');
  const start = new Date(new Date(a.start).getTime() - 1000);
  const end = new Date(a.end);
  const out = [];
  for (const calendar of app.calendars()) {
    const events = calendar.events.whose({ _and: [
      { startDate: { _greaterThan: start } }, { startDate: { _lessThan: end } } ] })();
    for (const event of events) {
      out.push({ title: event.summary(), start: event.startDate().toISOString(), end: event.endDate().toISOString(),
                 calendar: calendar.name(), location: event.location() || '' });
    }
  }
  out.sort((x, y) => (x.start < y.start ? -1 : 1));
  return JSON.stringify(out.slice(0, a.limit));
}"""

_CALENDAR_ADD = """
function run(argv) {
  const a = JSON.parse(argv[0]);
  const app = Application('Calendar');
  const calendar = a.calendar ? app.calendars.byName(a.calendar) : app.calendars.whose({ writable: true })()[0];
  const event = app.Event({ summary: a.title, startDate: new Date(a.start), endDate: new Date(a.end),
                            location: a.location || '', description: a.notes || '' });
  calendar.events.push(event);
  return JSON.stringify({ calendar: calendar.name() });
}"""


async def calendar_events(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    payload = {"start": arguments.get("start"), "end": arguments.get("end"),
               "limit": max(1, min(int(arguments.get("limit") or 50), 200))}
    if not payload["start"] or not payload["end"]:
        return ToolResult(content="start and end are required (ISO local date-times).", summary="Thiếu khoảng thời gian",
                          ok=False)
    result = await _jxa_tool(_CALENDAR_EVENTS, payload, "Xem lịch")
    if result.ok:
        result.summary = f"Lịch: {len(result.data.get('result') or [])} sự kiện"
    return result


async def calendar_add_event(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    title = " ".join(str(arguments.get("title") or "").split())[:200]
    if not title or not arguments.get("start") or not arguments.get("end"):
        return ToolResult(content="title, start and end are required.", summary="Thiếu thông tin sự kiện", ok=False)
    payload = {key: arguments.get(key) or "" for key in ("start", "end", "calendar", "location", "notes")}
    payload["title"] = title
    return await _jxa_tool(_CALENDAR_ADD, payload, f"Thêm vào Lịch: {title}")


# --- Notes -----------------------------------------------------------------
_NOTES_CREATE = """
function run(argv) {
  const a = JSON.parse(argv[0]);
  const app = Application('Notes');
  const note = app.Note({ body: a.html });
  app.defaultAccount().notes.push(note);
  return JSON.stringify({ created: a.title });
}"""


def _note_html(title: str, body: str) -> str:
    escape = lambda text: text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")  # noqa: E731
    lines = "".join(f"<div>{escape(line) or '<br>'}</div>" for line in body.splitlines())
    return f"<h1>{escape(title)}</h1>{lines}"


async def notes_create(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    title = " ".join(str(arguments.get("title") or "").split())[:150]
    body = str(arguments.get("body") or "")[:20_000]
    if not title:
        return ToolResult(content="Missing title.", summary="Thiếu tiêu đề", ok=False)
    return await _jxa_tool(_NOTES_CREATE, {"title": title, "html": _note_html(title, body)}, f"Tạo ghi chú: {title}")


# --- Apps and links ------------------------------------------------------------
async def open_app(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    name = str(arguments.get("name") or "").strip()
    if not name or "/" in name:
        return ToolResult(content="Give an application name, e.g. Spotify.", summary="Tên ứng dụng không hợp lệ", ok=False)
    process = await asyncio.create_subprocess_exec("/usr/bin/open", "-a", name,
                                                   stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    _, stderr = await process.communicate()
    if process.returncode != 0:
        return ToolResult(content=f"Could not open {name}: {stderr.decode()[-200:]}", summary=f"Không mở được {name}",
                          ok=False)
    return ToolResult(content=f"Opened {name}.", summary=f"Mở {name}")


async def open_url(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    url = str(arguments.get("url") or "").strip()
    if urlparse(url).scheme not in {"http", "https"}:
        return ToolResult(content="Only http(s) links are opened.", summary="Liên kết không hợp lệ", ok=False)
    process = await asyncio.create_subprocess_exec("/usr/bin/open", url)
    await process.communicate()
    return ToolResult(content=f"Opened {url} in the browser.", summary=f"Mở {urlparse(url).hostname}")


_LOCAL_TIME = "ISO local date-time, e.g. 2026-10-23T15:00"

MACOS_TOOLS = (
    Tool(group="apple", name="apple_reminders_add",
         description="Add a reminder to Apple Reminders (syncs to the user's iPhone via iCloud). Use when the user "
                     "asks to put something in Reminders / on their phone, or for to-dos without a fixed alert time.",
         parameters={"type": "object", "properties": {
             "title": {"type": "string"}, "due": {"type": "string", "description": _LOCAL_TIME},
             "notes": {"type": "string"}, "list": {"type": "string", "description": "Reminders list name (optional)"}},
             "required": ["title"]},
         handler=apple_reminders_add, label="Apple Reminders", side_effects=True),
    Tool(group="apple", name="apple_reminders_list", description="List open (not completed) reminders from Apple Reminders.",
         parameters={"type": "object", "properties": {"limit": {"type": "integer"}}},
         handler=apple_reminders_list, label="Xem Apple Reminders"),
    Tool(group="apple", name="apple_reminders_complete", description="Mark an open Apple Reminders item done (exact title).",
         parameters={"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]},
         handler=apple_reminders_complete, label="Hoàn thành lời nhắc", side_effects=True),
    Tool(group="apple", name="calendar_events", description="List Calendar events between two local times (all calendars).",
         parameters={"type": "object", "properties": {
             "start": {"type": "string", "description": _LOCAL_TIME}, "end": {"type": "string", "description": _LOCAL_TIME},
             "limit": {"type": "integer"}}, "required": ["start", "end"]},
         handler=calendar_events, label="Xem lịch"),
    Tool(group="apple", name="calendar_add_event", description="Add an event to Apple Calendar.",
         parameters={"type": "object", "properties": {
             "title": {"type": "string"}, "start": {"type": "string", "description": _LOCAL_TIME},
             "end": {"type": "string", "description": _LOCAL_TIME}, "calendar": {"type": "string"},
             "location": {"type": "string"}, "notes": {"type": "string"}}, "required": ["title", "start", "end"]},
         handler=calendar_add_event, label="Thêm sự kiện", side_effects=True),
    Tool(group="apple", name="notes_create", description="Create a note in Apple Notes (title + plain-text body).",
         parameters={"type": "object", "properties": {"title": {"type": "string"}, "body": {"type": "string"}},
                     "required": ["title", "body"]},
         handler=notes_create, label="Tạo ghi chú", side_effects=True),
    Tool(group="apps", name="open_app", description="Open a macOS application by name (e.g. Spotify, Safari).",
         parameters={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
         handler=open_app, label="Mở ứng dụng"),
    Tool(group="apps", name="open_url", description="Open an http(s) link in the user's default browser.",
         parameters={"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
         handler=open_url, label="Mở liên kết", side_effects=True),
)
