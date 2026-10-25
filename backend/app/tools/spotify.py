"""Spotify control on macOS: play a song by name, pause/resume, skip, volume.

The desktop app has no search over AppleScript, so a song name is resolved to
a track id through a web search restricted to open.spotify.com/track, then
played by URI. Commands pass values as osascript arguments, never spliced
into the script.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from app.tools.registry import Tool, ToolContext, ToolResult

_TRACK_RE = re.compile(r"open\.spotify\.com/(?:intl-[a-z]+/)?(track|album|playlist)/([A-Za-z0-9]{22})")
_STATE = 'tell application "Spotify" to return (name of current track) & "\\n" & (artist of current track) & "\\n" & (player state as string)'
_COMMANDS = {
    "pause": 'tell application "Spotify" to pause',
    "resume": 'tell application "Spotify" to play',
    "next": 'tell application "Spotify" to next track',
    "previous": 'tell application "Spotify" to previous track',
}


async def _osascript(script: str, *args: str) -> tuple[int, str]:
    process = await asyncio.create_subprocess_exec(
        "/usr/bin/osascript", "-e", script, *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=20)
    return process.returncode or 0, (stdout or stderr).decode("utf-8", errors="replace").strip()


async def _ensure_running() -> None:
    code, running = await _osascript('tell application "System Events" to (name of processes) contains "Spotify"')
    if running != "true":
        await (await asyncio.create_subprocess_exec("/usr/bin/open", "-a", "Spotify")).communicate()
        await asyncio.sleep(4)


def _search_spotify_uri(query: str) -> tuple[str, str] | None:
    from ddgs import DDGS  # type: ignore

    with DDGS() as client:
        results = list(client.text(f"site:open.spotify.com {query}", max_results=8))
    for item in results:
        match = _TRACK_RE.search(str(item.get("href") or ""))
        if match:
            title = str(item.get("title") or "").split("|")[0].strip()
            return f"spotify:{match.group(1)}:{match.group(2)}", title
    return None


async def _now_playing() -> str:
    code, text = await _osascript(_STATE)
    if code != 0:
        return ""
    name, _, rest = text.partition("\n")
    artist, _, state = rest.partition("\n")
    return f"{name} — {artist} ({state})"


async def spotify(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    action = str(arguments.get("action") or "").strip()
    await _ensure_running()
    if action == "play":
        query = " ".join(str(arguments.get("query") or "").split())[:200]
        if not query:
            return ToolResult(content="Give a song, artist, album or playlist to play.", summary="Thiếu tên bài", ok=False)
        found = await asyncio.to_thread(_search_spotify_uri, query)
        if found is None:
            return ToolResult(content=f"No Spotify result found for '{query}'.", summary=f"Không tìm thấy “{query}”", ok=False)
        uri, title = found
        code, error = await _osascript('on run argv\ntell application "Spotify" to play track (item 1 of argv)\nend run', uri)
        if code != 0:
            return ToolResult(content=f"Spotify refused to play: {error[-200:]}", summary="Spotify báo lỗi", ok=False)
        await asyncio.sleep(1.5)
        playing = await _now_playing()
        return ToolResult(content=f"Playing {playing or title} ({uri}).", summary=f"Đang phát {playing or title}")
    if action in _COMMANDS:
        code, error = await _osascript(_COMMANDS[action])
        if code != 0:
            return ToolResult(content=f"Spotify error: {error[-200:]}", summary="Spotify báo lỗi", ok=False)
        await asyncio.sleep(0.8)
        playing = await _now_playing()
        label = {"pause": "Tạm dừng", "resume": "Tiếp tục", "next": "Bài tiếp", "previous": "Bài trước"}[action]
        return ToolResult(content=f"{action} done. Now: {playing}", summary=f"{label} — {playing}")
    if action == "volume":
        level = max(0, min(int(arguments.get("volume") if arguments.get("volume") is not None else 50), 100))
        code, error = await _osascript('on run argv\ntell application "Spotify" to set sound volume to (item 1 of argv as integer)\nend run',
                                       str(level))
        return ToolResult(content=f"Volume {level}.", summary=f"Âm lượng {level}%", ok=code == 0)
    if action == "now_playing":
        playing = await _now_playing()
        return ToolResult(content=playing or "Nothing playing.", summary=f"Đang phát: {playing or 'không có'}")
    return ToolResult(content="action must be play, pause, resume, next, previous, volume or now_playing.",
                      summary="Lệnh Spotify không hợp lệ", ok=False)


SPOTIFY = Tool(
    group="music",
    name="spotify",
    description=(
        "Control the Spotify app on this Mac. action=play with query (song and/or artist, album or playlist name) "
        "finds and plays it; pause, resume, next, previous; volume (0–100); now_playing. For a vague request "
        "('bật bài gì đó vui vui'), pick a fitting well-known song yourself and play it."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["play", "pause", "resume", "next", "previous", "volume", "now_playing"]},
            "query": {"type": "string"},
            "volume": {"type": "integer", "minimum": 0, "maximum": 100},
        },
        "required": ["action"],
    },
    handler=spotify,
    label="Spotify",
)
