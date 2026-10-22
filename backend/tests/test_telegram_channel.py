from __future__ import annotations

import asyncio

from app.channels.telegram import AyaTelegramBot, chunk_text, iter_sse, plain_answer


def test_only_the_owner_is_served() -> None:
    bot = AyaTelegramBot(token="t", owner_id=42, backend_url="http://x")
    assert bot.is_owner(42) and not bot.is_owner(7) and not bot.is_owner(None)
    assert not AyaTelegramBot(token="t", owner_id=0, backend_url="http://x").is_owner(0)


def test_sse_lines_become_events() -> None:
    lines = ['event: run.started', 'data: {"conversation_id": "c1"}', '', 'event: message.delta',
             'data: {"delta": "Chào"}', '', 'data: not json', '']

    async def source():
        for line in lines:
            yield line

    async def collect():
        return [item async for item in iter_sse(source())]

    assert asyncio.run(collect()) == [("run.started", {"conversation_id": "c1"}), ("message.delta", {"delta": "Chào"})]


def test_long_answers_are_split_and_web_sources_listed() -> None:
    text = "\n\n".join(["đoạn " + "x" * 3000 for _ in range(3)])
    parts = chunk_text(text)
    assert len(parts) == 3 and all(len(part) <= 3900 for part in parts)
    answer = plain_answer("Giá vàng 142,3 [W2].", [{"id": "W1", "url": "https://a"}, {"id": "W2", "title": "SJC", "url": "https://b"}])
    assert "https://b" in answer and "https://a" not in answer


def test_approval_buttons_use_short_keys() -> None:
    bot = AyaTelegramBot(token="t", owner_id=1, backend_url="http://x")
    key = bot.remember_approval("7f1c" * 9, "call_" + "a" * 32)
    assert len(f"y|{key}".encode()) <= 64 and bot._pending[key][1].startswith("call_")
