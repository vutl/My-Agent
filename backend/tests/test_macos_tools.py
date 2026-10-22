from __future__ import annotations

import asyncio
import json

import pytest

from app.tools import macos


class FakeProcess:
    def __init__(self, stdout: bytes = b"", stderr: bytes = b"", code: int = 0) -> None:
        self._out, self._err, self.returncode = stdout, stderr, code

    async def communicate(self):
        return self._out, self._err

    def kill(self) -> None:
        pass


@pytest.fixture
def calls(monkeypatch):
    seen: list[tuple] = []
    reply = {"process": FakeProcess(stdout=b'{"list": "Reminders"}')}

    async def fake_exec(*argv, **_kwargs):
        seen.append(argv)
        return reply["process"]

    monkeypatch.setattr(macos.asyncio, "create_subprocess_exec", fake_exec)
    return seen, reply


def test_arguments_travel_as_json_never_spliced_into_the_script(calls) -> None:
    seen, _ = calls
    title = 'Mua sữa"); Application("Finder").delete(…'
    result = asyncio.run(macos.apple_reminders_add({"title": title, "due": "2026-10-23T15:00"}, None))
    assert result.ok and "Apple Reminders" in result.summary
    argv = seen[0]
    assert argv[:3] == ("/usr/bin/osascript", "-l", "JavaScript")
    assert title not in argv[4]  # the script itself is constant
    assert json.loads(argv[5])["title"] == " ".join(title.split())


def test_permission_refusal_becomes_a_clear_message(calls) -> None:
    _, reply = calls
    reply["process"] = FakeProcess(stderr=b"execution error: Not authorized to send Apple events to Reminders. (-1743)",
                                   code=1)
    result = asyncio.run(macos.apple_reminders_list({}, None))
    assert not result.ok and "Privacy & Security" in result.content


def test_calendar_needs_a_range_and_notes_escape_html() -> None:
    assert not asyncio.run(macos.calendar_events({"start": "2026-10-23T00:00"}, None)).ok
    assert macos._note_html("A<b>", "x & y\n") == "<h1>A&lt;b&gt;</h1><div>x &amp; y</div>"


def test_only_web_links_and_plain_app_names_are_opened() -> None:
    assert not asyncio.run(macos.open_url({"url": "file:///etc/passwd"}, None)).ok
    assert not asyncio.run(macos.open_app({"name": "/bin/sh"}, None)).ok
    writes = {tool.name for tool in macos.MACOS_TOOLS if tool.side_effects}
    assert writes == {"apple_reminders_add", "apple_reminders_complete", "calendar_add_event", "notes_create", "open_url"}
