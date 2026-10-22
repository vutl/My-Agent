from collections.abc import AsyncIterator, Callable
from pathlib import Path
import re
import time
from typing import Annotated, Any
from app.core.config import Settings, get_settings
from app.core.events import sse_event
from app.services.rag_service import (
    RagService,
    caption_identifies_figure,
    requested_visual_page_numbers,
    visual_page_reservation_count,
)
from . import deps
from .common import _elapsed_ms


async def _stream_assistant_tool_turn(
    *, service: Any, settings: Settings, rag: RagService, request: Any, run_id: str,
    conversation_id: str, model: str, previous_messages: list[Any], assistant_chunks: list[str],
    assistant_citations: list[dict[str, Any]], run_store: Any,
) -> AsyncIterator[str]:
    """The agent route: Aya's tool loop with live tool cards, approvals and
    URL citations; the answer is streamed in chunks once the loop settles."""
    from app import tools as tools_module
    from app.agents import tool_loop
    from app.agents.graph import _answer_language
    from app.tools import ToolContext

    store = deps._retrieval_store(settings)
    embeddings = deps._embedding_provider(settings)

    async def library_searcher(query: str, top_k: int) -> list[dict[str, Any]]:
        found = await rag.search_hybrid(
            query=query, top_k=top_k, collection_id=request.collection_id,
            retrieval_store=store, embeddings=embeddings, rerank=settings.rerank_enabled,
            rerank_mode=settings.rerank_mode, cross_encoder_model_path=settings.rerank_cross_encoder_path,
            rerank_max_candidates=settings.rerank_max_candidates,
        )
        return [item for item in found.get("results") or [] if not item.get("figure_id")]

    registry = tools_module.build_registry(library_searcher=library_searcher)
    workspace = Path(settings.assistant_workspace).expanduser()
    context = ToolContext(run_id=run_id, conversation_id=conversation_id, workspace=workspace, settings=settings)
    history_messages = [
        {"role": message.role, "content": str(message.content or "")[:4000]}
        for message in previous_messages[-10:]
        if getattr(message, "role", "") in {"user", "assistant"} and str(message.content or "").strip()
    ]
    started: dict[str, float] = {}
    async for name, data in tool_loop.run_tool_loop(
        client=service.client, model=model, task=request.task, history=history_messages,
        registry=registry, tool_names=registry.names(), context=context,
        language=_answer_language(request.task), approvals=tool_loop.APPROVALS,
    ):
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
            try:
                run_store.record_tool_call(
                    run_id=run_id, tool_name=str(data.get("name")), input_payload=data.get("arguments")
                    if isinstance(data.get("arguments"), dict) else {"raw": data.get("arguments")},
                    output_payload={"summary": data.get("summary"), "sources": data.get("sources")},
                    status="completed" if data.get("ok") else "failed",
                    approved=False if data.get("declined") else None,
                )
            except Exception:  # noqa: BLE001 - the audit log must not break the answer
                pass
            data = {**data, "ms": _elapsed_ms(started.get(str(data.get("call_id")), time.perf_counter()))}
        yield sse_event(name, {"run_id": run_id, "conversation_id": conversation_id, **data})

