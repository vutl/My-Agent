"""Interaction, explicit memory, notifications and artifact-reading tools."""

from __future__ import annotations

import asyncio
import hashlib
import re
import sys
from typing import Any
import unicodedata

from app.services.artifacts import ArtifactStore
from app.services.long_term_memory import (
    MemoryItemStore, _contains_sensitive_memory_value, _MEMORY_OPERATION_KEY_RE,
)
from app.services.reminders import ReminderStore
from app.services.tasks import TaskStore
from app.tools.registry import ApprovalChanged, Handler, Tool, ToolContext, ToolResult


def _text(arguments: dict[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _memory(context: ToolContext) -> MemoryItemStore:
    return MemoryItemStore(context.settings.sqlite_db_path)


def _scope(arguments: dict[str, Any], context: ToolContext) -> tuple[str, str | None]:
    scope = arguments.get("scope", "user")
    if scope not in {"user", "conversation"}:
        raise ValueError("scope must be user or conversation")
    return scope, context.conversation_id if scope == "conversation" else None


async def ask_user(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    question = _text(arguments, "question")
    options = arguments.get("options", [])
    reason = arguments.get("reason")
    if (not isinstance(options, list) or len(options) > 6
            or any(not isinstance(item, str) or not item.strip() for item in options)):
        raise ValueError("options must contain at most six non-empty strings")
    if reason is not None and not isinstance(reason, str):
        raise ValueError("reason must be a string")
    # The loop ends here; a task runner records the ask and waits for the answer.
    return ToolResult(content=question, summary=question,
                      data={"ask_user": {"question": question, "options": options, "reason": reason}})


async def remember(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    fact = _text(arguments, "fact")
    scope, conversation = _scope(arguments, context)
    kind = arguments.get("kind", "semantic")
    if kind not in {"semantic", "procedural"}:
        raise ValueError("kind must be semantic or procedural")
    key = arguments.get("key")
    if key is None:
        ascii_fact = unicodedata.normalize("NFKD", fact).encode("ascii", "ignore").decode().lower()
        slug = re.sub(r"[^a-z0-9]+", "-", ascii_fact).strip("-")[:100]
        key = (slug or "fact") + "-" + hashlib.sha256(fact.encode()).hexdigest()[:24]
    if not isinstance(key, str) or not _MEMORY_OPERATION_KEY_RE.fullmatch(key):
        raise ValueError("key must match [a-z0-9][a-z0-9_.:-]{0,127}")
    if _contains_sensitive_memory_value(key, fact):
        return ToolResult(content="Refusing to remember sensitive content (passwords, secrets or tokens).",
                          summary="Không lưu thông tin nhạy cảm", ok=False)
    item = _memory(context).upsert(
        scope=scope, kind=kind, memory_key=key, content=fact, conversation_id=conversation,
        source_conversation_id=context.conversation_id, metadata={"source": "tool"},
    )
    return ToolResult(content=f"Remembered [{item.scope}/{item.memory_key}]: {item.content}", summary="Đã ghi nhớ")


async def recall(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    query = _text(arguments, "query")
    limit = arguments.get("limit", 5)
    if type(limit) is not int or not 1 <= limit <= 50:
        raise ValueError("limit must be an integer between 1 and 50")
    items = _memory(context).relevant(query, conversation_id=context.conversation_id, limit=limit)
    content = "\n".join(f"[{item.scope}/{item.memory_key}] {item.content} (updated_at={item.updated_at})"
                        for item in items)
    return ToolResult(content=content or "No relevant memories.", summary=f"{len(items)} mục ghi nhớ",
                      untrusted=True)


def forget_preview(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    key = _text(arguments, "key")
    if "scope" not in arguments:
        raise ValueError("scope is required")
    scope, conversation = _scope(arguments, context)
    items = _memory(context).for_key(scope=scope, memory_key=key, conversation_id=conversation)
    return {"scope": scope, "key": key, "memories": [item.to_dict() for item in items]}


async def forget(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    preview = context.approved_preview
    scope, conversation = _scope(arguments, context)
    key = _text(arguments, "key")
    if preview is None or preview["scope"] != scope or preview["key"] != key:
        raise ApprovalChanged("Approval of the current memory versions is required.")
    if not _memory(context).forget_versions(
        scope=scope, memory_key=key, conversation_id=conversation,
        version_ids=[item["id"] for item in preview["memories"]],
    ):
        raise ApprovalChanged("Memory changed while approval was pending. Please approve the current versions.")
    return ToolResult(content=f"Forgot {len(preview['memories'])} memories for {preview['scope']}/{preview['key']}.",
                      summary="Đã quên mục ghi nhớ")


async def deliver_macos(title: str, text: str) -> None:
    from app.tools.macos import notify as macos_notify

    if not await asyncio.to_thread(macos_notify, title, text):
        raise RuntimeError("osascript did not deliver the notification")


async def deliver_telegram(context: ToolContext, title: str, text: str) -> None:
    from app.channels.telegram import AyaTelegramBot

    bot = AyaTelegramBot(token=context.settings.telegram_bot_token, owner_id=context.settings.telegram_owner_id,
                         backend_url=context.settings.telegram_backend_url)
    await bot.send_to_owner(f"{title}\n{text}")


async def notify(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    text = _text(arguments, "text")
    title = arguments.get("title", "Aya")
    channel = arguments.get("channel", "all")
    if channel not in {"app", "macos", "telegram", "all"}:
        raise ValueError("channel must be app, macos, telegram or all")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be a non-empty string")
    if context.task_id:
        allowed = TaskStore(context.settings.sqlite_db_path).reserve_notification(context.task_id)
    else:
        count = context.state.get("notifications", 0)
        allowed = count < 5
        if allowed:
            context.state["notifications"] = count + 1
    if not allowed:
        return ToolResult(content="Notification limit reached (5 per task/turn).", summary="Đã đạt giới hạn", ok=False)
    delivered, errors = [], []
    if channel in {"app", "all"}:
        try:
            ReminderStore(context.settings.sqlite_db_path).notify(
                kind="assistant", title=title, body=text, ref_id=context.task_id or context.run_id,
                conversation_id=context.conversation_id,
            )
            delivered.append("app")
        except Exception as exc:  # noqa: BLE001 - independent channels
            errors.append(f"app: {type(exc).__name__}")
    for name, enabled in (("macos", sys.platform == "darwin"),
                          ("telegram", bool(getattr(context.settings, "telegram_bot_token", "")
                                            and getattr(context.settings, "telegram_owner_id", "")))):
        if channel not in {name, "all"}:
            continue
        if not enabled:
            if channel == name:
                errors.append(f"{name}: not configured/available")
            continue
        try:
            if name == "macos":
                await deliver_macos(title, text)
            else:
                await deliver_telegram(context, title, text)
            delivered.append(name)
        except Exception as exc:  # noqa: BLE001 - independent channels
            errors.append(f"{name}: {type(exc).__name__}")
    content = f"Notification delivered to: {', '.join(delivered) or 'none'}."
    if errors:
        content += " Delivery errors: " + "; ".join(errors)
    return ToolResult(content=content, summary=content, ok=bool(delivered) and not errors)


async def read_artifact(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    content = ArtifactStore(context.workspace).read(
        _text(arguments, "handle"), arguments.get("offset", 0), arguments.get("length", 4000),
    )
    return ToolResult(content=content, summary=f"Đọc {len(content)} ký tự", untrusted=True)


def _tool(name: str, description: str, handler: Handler, properties: dict[str, Any],
          required: list[str], **kwargs: Any) -> Tool:
    return Tool(name=name, group="assistant", description=description, handler=handler,
                parameters={"type": "object", "properties": properties, "required": required},
                label=name, **kwargs)


_SCOPE = {"type": "string", "enum": ["user", "conversation"], "default": "user"}
ASSISTANT_TOOLS = (
    _tool("ask_user", "Ask the user a question and end this turn. Their answer continues the task.", ask_user,
          {"question": {"type": "string"}, "options": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
           "reason": {"type": "string"}}, ["question"]),
    _tool("remember", "Remember a user fact or procedure, refusing sensitive credentials.", remember,
          {"fact": {"type": "string"}, "scope": _SCOPE, "key": {"type": "string"},
           "kind": {"type": "string", "enum": ["semantic", "procedural"], "default": "semantic"}}, ["fact"],
          side_effects=True),
    _tool("recall", "Recall relevant user and current-conversation memories.", recall,
          {"query": {"type": "string"}, "limit": {"type": "integer", "default": 5}}, ["query"]),
    _tool("forget", "Forget an exact memory key in the specified scope, across its memory kinds. Requires approval.",
          forget, {"key": {"type": "string"}, "scope": _SCOPE}, ["key", "scope"],
          permission="ask", side_effects=True, approval_preview=forget_preview),
    _tool("notify", "Notify the user now (at most five per task/turn).", notify,
          {"text": {"type": "string"}, "title": {"type": "string"},
           "channel": {"type": "string", "enum": ["app", "macos", "telegram", "all"], "default": "all"}},
          ["text"], side_effects=True),
    _tool("read_artifact", "Read a saved tool output by handle; offsets and length are in characters.", read_artifact,
          {"handle": {"type": "string"}, "offset": {"type": "integer", "minimum": 0, "default": 0},
           "length": {"type": "integer", "minimum": 1, "maximum": 6000, "default": 4000}}, ["handle"]),
)
