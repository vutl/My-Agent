"""Telegram channel: talk to Aya from the phone.

The bot is a client of Aya's own HTTP API, so a Telegram turn runs exactly
like a desktop turn (router, RAG, tools, approvals, history). Only the
owner's user id is served; every other update is ignored. Tool calls that
need approval arrive as a message with Cho phép / Từ chối buttons, answered
through the same decision endpoint the desktop uses. Long polling: no public
URL or webhook is needed.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
import json
import logging
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_MAX_MESSAGE = 3900  # Telegram limit is 4096 characters
_TOOLS = ["search_local_docs", "retrieve_visual_assets", "assistant_tools"]


@dataclass
class TurnOutcome:
    text: str = ""
    tools: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    conversation_id: str | None = None


async def iter_sse(lines: AsyncIterator[str]) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    event, data = "message", []
    async for line in lines:
        if not line:
            if data:
                try:
                    yield event, json.loads("\n".join(data))
                except ValueError:
                    pass
            event, data = "message", []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())


def chunk_text(text: str, limit: int = _MAX_MESSAGE) -> list[str]:
    """Split on paragraph boundaries under Telegram's message limit."""
    parts, current = [], ""
    for paragraph in text.split("\n\n"):
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            parts.append(current)
        while len(paragraph) > limit:
            parts.append(paragraph[:limit])
            paragraph = paragraph[limit:]
        current = paragraph
    if current:
        parts.append(current)
    return parts or [""]


def plain_answer(text: str, sources: list[dict[str, Any]]) -> str:
    """Telegram shows plain text here: keep [W#]/[S#] markers and list the
    web sources the answer cites at the end."""
    cited = {match for match in re.findall(r"\[(W\d+)\]", text)}
    links = [f"[{item['id']}] {item.get('title') or ''} {item.get('url')}".strip()
             for item in sources if item.get("id") in cited and item.get("url")]
    return text.strip() + ("\n\nNguồn:\n" + "\n".join(links) if links else "")


class AyaTelegramBot:
    def __init__(self, *, token: str, owner_id: int, backend_url: str) -> None:
        self.token = token
        self.owner_id = int(owner_id)
        self.backend_url = backend_url.rstrip("/")
        self.conversation_id: str | None = None  # one running thread per owner
        self._bot: Any = None
        # Telegram button data is limited to 64 bytes: buttons carry a short
        # key for the pending (run id, tool call id).
        self._pending: dict[str, tuple[str, str]] = {}

    def remember_approval(self, run_id: str, call_id: str) -> str:
        key = f"{len(self._pending) + 1:x}"
        self._pending[key] = (run_id, call_id)
        return key

    def is_owner(self, user_id: int | None) -> bool:
        return bool(self.owner_id) and user_id == self.owner_id

    async def run_turn(self, text: str, on_approval) -> TurnOutcome:
        outcome = TurnOutcome(conversation_id=self.conversation_id)
        body = {"task": text, "conversation_id": self.conversation_id, "allowed_tools": _TOOLS}
        async with httpx.AsyncClient(timeout=httpx.Timeout(900, connect=10)) as client:
            async with client.stream("POST", f"{self.backend_url}/agent/run/stream", json=body) as response:
                response.raise_for_status()
                async for event, data in iter_sse(response.aiter_lines()):
                    if event == "run.started":
                        outcome.conversation_id = self.conversation_id = data.get("conversation_id")
                    elif event == "message.delta":
                        outcome.text += str(data.get("delta") or "")
                    elif event == "message.reset":
                        outcome.text = ""
                    elif event == "tool.call.completed":
                        outcome.tools.append(str(data.get("summary") or data.get("label") or ""))
                    elif event == "tool.call.approval_required":
                        await on_approval(data)
                    elif event == "answer.citations":
                        outcome.sources = list(data.get("citations") or [])
                    elif event == "run.failed":
                        outcome.error = str(data.get("error") or "failed")
        return outcome

    async def decide(self, run_id: str, call_id: str, approved: bool) -> bool:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                f"{self.backend_url}/agent/runs/{run_id}/tool-calls/{call_id}/decision", json={"approved": approved},
            )
        return response.status_code == 200

    async def send_to_owner(self, text: str) -> None:
        if not self.owner_id:
            return
        if self._bot is None:
            # Explicit notifications also work without the polling worker.
            from aiogram import Bot

            async with Bot(token=self.token) as bot:
                for part in chunk_text(text):
                    await bot.send_message(self.owner_id, part)
            return
        for part in chunk_text(text):
            await self._bot.send_message(self.owner_id, part)

    async def run(self, stop: asyncio.Event) -> None:
        from aiogram import Bot, Dispatcher, F
        from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

        bot = Bot(self.token)
        self._bot = bot
        dispatcher = Dispatcher()

        @dispatcher.message(F.text)
        async def on_message(message: Message) -> None:
            if not self.is_owner(message.from_user.id if message.from_user else None):
                return  # silently ignore everyone but the owner
            if message.text.strip() in {"/new", "/moi"}:
                self.conversation_id = None
                await message.answer("Ừ cậu~ mình bắt đầu cuộc trò chuyện mới nha.")
                return

            async def on_approval(data: dict[str, Any]) -> None:
                arguments = data.get("arguments")
                detail = json.dumps(arguments, ensure_ascii=False, indent=1) if isinstance(arguments, dict) else str(arguments)
                key = self.remember_approval(str(data.get("run_id")), str(data.get("call_id")))
                keyboard = InlineKeyboardMarkup(inline_keyboard=[[
                    InlineKeyboardButton(text="Cho phép", callback_data=f"y|{key}"),
                    InlineKeyboardButton(text="Từ chối", callback_data=f"n|{key}"),
                ]])
                await message.answer(f"Aya muốn: {data.get('label')}\n{detail[:1500]}\nCậu cho phép không?",
                                     reply_markup=keyboard)

            await bot.send_chat_action(message.chat.id, "typing")
            try:
                outcome = await self.run_turn(message.text, on_approval)
            except Exception as exc:  # noqa: BLE001 - tell the owner instead of failing silently
                await message.answer(f"Ơ… Aya gặp lỗi khi gọi backend ({type(exc).__name__}).")
                return
            reply = plain_answer(outcome.text, outcome.sources) if outcome.text else f"Không hoàn tất: {outcome.error}"
            for part in chunk_text(reply):
                await message.answer(part)

        @dispatcher.callback_query()
        async def on_decision(query: CallbackQuery) -> None:
            if not self.is_owner(query.from_user.id if query.from_user else None):
                return
            verdict, _, key = str(query.data).partition("|")
            pending = self._pending.pop(key, None)
            if pending is None:
                await query.answer("Hết hạn rồi")
                return
            ok = await self.decide(pending[0], pending[1], verdict == "y")
            await query.answer("Đã gửi~" if ok else "Hết hạn rồi")
            if query.message:
                await query.message.edit_reply_markup(reply_markup=None)

        polling = asyncio.create_task(dispatcher.start_polling(bot, handle_signals=False))
        await stop.wait()
        await dispatcher.stop_polling()
        polling.cancel()
        await bot.session.close()
