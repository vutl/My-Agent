import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import aclosing
import re
import time
from typing import Any
from app.core.config import Settings
from app.core.events import sse_event
from app.services.rag_service import RagService
from .common import _elapsed_ms


async def _stream_assistant_tool_turn(
    *, service: Any, settings: Settings, rag: RagService, request: Any, run_id: str,
    conversation_id: str, model: str, previous_messages: list[Any], assistant_chunks: list[str],
    assistant_citations: list[dict[str, Any]], run_store: Any,
    is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    task_result: dict[str, Any] | None = None,
) -> AsyncIterator[str]:
    """The agent route: Aya's tool loop with live tool cards, approvals and
    URL citations; the answer is streamed in chunks once the loop settles."""
    from app.agents import tool_loop
    from app.agents.task_runner import plan_goal, start_task
    from app.services.task_deps import task_deps_from_settings
    from app.services.task_events import watch_task
    from app.services.task_worker import report_message_id
    from app.services.tasks import TaskStore
    from app.agents.graph import _answer_language

    task_deps = task_deps_from_settings(settings, conversation_id)
    task_deps.client, task_deps.model = service.client, model
    task_deps.configure({"collection_id": getattr(request, "collection_id", None)})
    registry = task_deps.registry_factory()
    context = task_deps.context_factory(run_id, conversation_id)
    history_messages = [
        {"role": message.role, "content": str(message.content or "")[:4000]}
        for message in previous_messages[-10:]
        if getattr(message, "role", "") in {"user", "assistant"} and str(message.content or "").strip()
    ]
    language = _answer_language(request.task)
    tool_names = registry.names()
    plan = await plan_goal(service.client, model, request.task, history_messages, [
        f"{registry.get(name).name}: {registry.get(name).description}" for name in tool_names
    ], language)
    loop_kwargs = dict(
        client=service.client, model=model, history=history_messages,
        registry=registry, tool_names=tool_names, context=context,
        language=language, approvals=tool_loop.APPROVALS,
    )
    # TaskStore bounds persisted goals to 2000 characters. Keep longer turns
    # intact on the existing loop rather than truncating the user's goal.
    task_id = None
    if plan["needs_plan"] and len(request.task.strip()) <= 2000:
        task_deps.language = language
        task_id = start_task(goal=request.task, plan=plan["steps"], deps=task_deps,
                             conversation_id=conversation_id, history=history_messages, run_id=run_id,
                             collection_id=getattr(request, "collection_id", None))
        if task_result is not None:
            task_result["task_id"] = task_id
        async def attached_events():
            yield "task.created", {"task": task_deps.store.get(task_id).to_dict()}
            async with aclosing(watch_task(task_deps.store, task_id, stop_when_waiting=True)) as watcher:
                async for event in watcher:
                    if is_disconnected is not None and await is_disconnected():
                        return
                    yield event
        events = attached_events()
    else:
        events = tool_loop.run_tool_loop(**loop_kwargs, task=request.task)
    started: dict[str, float] = {}
    async with aclosing(events):
        async for name, data in events:
            if name == "task.created":
                task_id = data["task"]["id"]
            if name == "answer":
                text = str(data.get("text") or "").strip()
                for index in range(0, len(text), 48):
                    chunk = text[index:index + 48]
                    assistant_chunks.append(chunk)
                    yield sse_event("message.delta", {"run_id": run_id, "conversation_id": conversation_id, "delta": chunk})
                used = {match for match in re.findall(r"\[(W\d+)\]", text)}
                assistant_citations.extend(
                    {"id": source["id"], "title": source["title"], "url": source["url"], "filename": source.get("path") or None}
                    for source in data.get("sources") or [] if source["id"] in used
                )
                yield sse_event("answer.citations", {"run_id": run_id, "conversation_id": conversation_id,
                                                     "citations": list(assistant_citations)})
                yield sse_event("avatar.state", {"run_id": run_id, "state": "happy"})
                continue
            if name == "tool.call.started":
                started[str(data.get("call_id"))] = time.perf_counter()
            if name == "tool.call.completed":
                if task_id is None:
                    try:
                        run_store.record_tool_call(
                            run_id=run_id, task_id=task_id, tool_name=str(data.get("name")), input_payload=data.get("arguments")
                            if isinstance(data.get("arguments"), dict) else {"raw": data.get("arguments")},
                            output_payload={"summary": data.get("summary"), "sources": data.get("sources")},
                            status="completed" if data.get("ok") else "failed",
                            approved=False if data.get("declined") else None,
                        )
                    except Exception:  # noqa: BLE001 - the audit log must not break the answer
                        pass
                data = {**data, "ms": _elapsed_ms(started.get(str(data.get("call_id")), time.perf_counter()))}
            yield sse_event(name, {"run_id": run_id, "conversation_id": conversation_id, **data})

    if task_id is not None and TaskStore(settings.sqlite_db_path).get(task_id).status == "cancelled":
        raise asyncio.CancelledError("Task cancelled")

    if task_id is not None and task_result is not None:
        if task_deps.store.get(task_id).status in {"done", "failed"}:
            task_result["message_id"] = report_message_id(task_id)
