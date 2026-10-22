from __future__ import annotations

import asyncio

from app.tools import spotify as module


def test_play_resolves_a_song_and_passes_the_uri_as_an_argument(monkeypatch) -> None:
    scripts: list[tuple[str, tuple]] = []

    async def fake_osascript(script, *args):
        scripts.append((script, args))
        if "System Events" in script:
            return 0, "true"
        if "current track" in script:
            return 0, "Nơi Này Có Anh\nSơn Tùng M-TP\nplaying"
        return 0, ""

    monkeypatch.setattr(module, "_osascript", fake_osascript)
    monkeypatch.setattr(module, "_search_spotify_uri",
                        lambda query: ("spotify:track:5rSAUmbOg4giDrM29vUyHe", "Nơi Này Có Anh"))
    monkeypatch.setattr(module.asyncio, "sleep", lambda *_: asyncio.sleep(0))
    result = asyncio.run(module.spotify({"action": "play", "query": "Nơi này có anh"}, None))
    assert result.ok and "Nơi Này Có Anh — Sơn Tùng M-TP" in result.summary
    play = [item for item in scripts if "play track" in item[0]][0]
    assert play[1] == ("spotify:track:5rSAUmbOg4giDrM29vUyHe",) and "5rSAU" not in play[0]


def test_unknown_action_and_missing_query_are_refused(monkeypatch) -> None:
    async def fake_osascript(script, *args):
        return 0, "true"

    monkeypatch.setattr(module, "_osascript", fake_osascript)
    assert not asyncio.run(module.spotify({"action": "delete_library"}, None)).ok
    assert not asyncio.run(module.spotify({"action": "play"}, None)).ok
