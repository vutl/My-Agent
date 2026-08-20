from collections.abc import AsyncIterator
import time
import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.core.events import sse_event
from app.llm.ollama_client import OllamaError
from app.llm.openai_client import get_llm_client
from app.services.chat_history import ChatHistory
from app.services.chat_service import ChatService
from app.services.conversation_memory import ConversationMemoryStore, schedule_memory_fold
from app.services.conversation_runtime import ConversationRuntimeGate
from app.services.long_term_memory import HistoricalConversationSearch, MemoryItemStore

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str = Field(min_length=1)
    model: str | None = None
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    system_prompt: str | None = None


class ChatResponse(BaseModel):
    conversation_id: str | None
    model: str
    message: str


class ConversationResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class StoredMessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    model: str | None
    created_at: str
    sources: list[dict[str, Any]] = Field(default_factory=list)


def get_chat_service(settings: Annotated[Settings, Depends(get_settings)]) -> ChatService:
    client = get_llm_client(
        provider=settings.llm_provider,
        ollama_host=settings.ollama_host,
        openai_api_base=settings.openai_api_base,
        openai_api_key=settings.openai_api_key,
        timeout_seconds=settings.request_timeout_seconds,
    )
    return ChatService(client=client, default_model=settings.default_model)


def get_chat_history(settings: Annotated[Settings, Depends(get_settings)]) -> ChatHistory:
    return ChatHistory(settings.sqlite_db_path)


def get_conversation_memory_store(
    settings: Annotated[Settings, Depends(get_settings)],
) -> ConversationMemoryStore:
    return ConversationMemoryStore(settings.sqlite_db_path)


def get_historical_search(
    settings: Annotated[Settings, Depends(get_settings)],
) -> HistoricalConversationSearch:
    return HistoricalConversationSearch(settings.sqlite_db_path)


def get_long_term_memory(
    settings: Annotated[Settings, Depends(get_settings)],
) -> MemoryItemStore:
    return MemoryItemStore(settings.sqlite_db_path)


def get_runtime_gate(request: Request) -> ConversationRuntimeGate:
    return request.app.state.conversation_runtime_gate


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    service: Annotated[ChatService, Depends(get_chat_service)],
    history: Annotated[ChatHistory, Depends(get_chat_history)],
    memory_store: Annotated[
        ConversationMemoryStore,
        Depends(get_conversation_memory_store),
    ],
    historical_search: Annotated[
        HistoricalConversationSearch,
        Depends(get_historical_search),
    ],
    long_term_memory: Annotated[MemoryItemStore, Depends(get_long_term_memory)],
    runtime_gate: Annotated[ConversationRuntimeGate, Depends(get_runtime_gate)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ChatResponse:
    conversation_id = history.ensure_conversation(request.conversation_id, request.message)
    async with runtime_gate.turn(conversation_id):
        previous_messages = history.list_messages(conversation_id)
        memory_context = _chat_memory_context(
            conversation_id=conversation_id,
            query=request.message,
            previous_messages=previous_messages,
            memory_store=memory_store,
            historical_search=historical_search,
            long_term_memory=long_term_memory,
        )
        user_record = history.save_message(
            conversation_id=conversation_id,
            role="user",
            content=request.message,
            model=request.model or service.default_model,
        )

        try:
            result = await service.complete(
                message=request.message,
                model=request.model,
                temperature=request.temperature,
                system_prompt=request.system_prompt,
                recent_messages=previous_messages,
                conversation_context=memory_context,
            )
        except OllamaError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        assistant_record = history.save_message(
            conversation_id=conversation_id,
            role="assistant",
            content=result.message,
            model=result.model,
        )
        _schedule_chat_memory(
            settings=settings,
            memory_store=memory_store,
            conversation_id=conversation_id,
            user_text=request.message,
            assistant_text=result.message,
            user_message_id=user_record.id,
            assistant_message_id=assistant_record.id,
        )

    return ChatResponse(
        conversation_id=conversation_id,
        model=result.model,
        message=result.message,
    )


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    service: Annotated[ChatService, Depends(get_chat_service)],
    history: Annotated[ChatHistory, Depends(get_chat_history)],
    memory_store: Annotated[
        ConversationMemoryStore,
        Depends(get_conversation_memory_store),
    ],
    historical_search: Annotated[
        HistoricalConversationSearch,
        Depends(get_historical_search),
    ],
    long_term_memory: Annotated[MemoryItemStore, Depends(get_long_term_memory)],
    runtime_gate: Annotated[ConversationRuntimeGate, Depends(get_runtime_gate)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> StreamingResponse:
    conversation_id = history.ensure_conversation(request.conversation_id, request.message)

    async def unlocked_event_stream():
        model = request.model or service.default_model
        previous_messages = history.list_messages(conversation_id)
        memory_context = _chat_memory_context(
            conversation_id=conversation_id,
            query=request.message,
            previous_messages=previous_messages,
            memory_store=memory_store,
            historical_search=historical_search,
            long_term_memory=long_term_memory,
        )
        user_record = history.save_message(
            conversation_id=conversation_id,
            role="user",
            content=request.message,
            model=model,
        )
        yield sse_event(
            "message.started",
            {"conversation_id": conversation_id, "model": model},
        )

        assistant_chunks: list[str] = []
        try:
            async for delta in _buffer_text_stream(
                service.stream(
                    message=request.message,
                    model=request.model,
                    temperature=request.temperature,
                    system_prompt=request.system_prompt,
                    recent_messages=previous_messages,
                    conversation_context=memory_context,
                )
            ):
                if delta:
                    assistant_chunks.append(delta)
                    yield sse_event("message.delta", {"delta": delta})
        except OllamaError as exc:
            yield sse_event("message.failed", {"error": str(exc)})
            return

        assistant_message = "".join(assistant_chunks)
        if not assistant_message.strip():
            yield sse_event("message.failed", {"error": "LLM returned an empty answer"})
            return
        assistant_record = history.save_message(
            conversation_id=conversation_id,
            role="assistant",
            content=assistant_message,
            model=model,
        )
        _schedule_chat_memory(
            settings=settings,
            memory_store=memory_store,
            conversation_id=conversation_id,
            user_text=request.message,
            assistant_text=assistant_message,
            user_message_id=user_record.id,
            assistant_message_id=assistant_record.id,
        )
        yield sse_event("message.completed", {"conversation_id": conversation_id})

    async def event_stream():
        async with runtime_gate.turn(conversation_id):
            async for event in unlocked_event_stream():
                yield event

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _chat_memory_context(
    *,
    conversation_id: str,
    query: str,
    previous_messages: list,
    memory_store: ConversationMemoryStore,
    historical_search: HistoricalConversationSearch,
    long_term_memory: MemoryItemStore,
) -> str:
    memory = memory_store.get_memory(conversation_id)
    recent_ids = tuple(message.id for message in previous_messages[-12:])
    blocks = [
        long_term_memory.prompt_block(
            query,
            conversation_id=conversation_id,
            limit=10,
            max_chars=3600,
            min_confidence=0.5,
        ),
        memory.prompt_block(),
        historical_search.prompt_block_for_context(
            query,
            current_conversation_id=conversation_id,
            exclude_message_ids=recent_ids,
            limit=4,
            max_chars=4800,
        ),
    ]
    packed = "\n\n".join(block for block in blocks if block)
    if not packed:
        return ""
    return (
        "Persistent conversation context follows. Prefer newer completed turns "
        "over older summaries; do not invent missing details.\n\n"
        f"{packed}"
    )


def _schedule_chat_memory(
    *,
    settings: Settings,
    memory_store: ConversationMemoryStore,
    conversation_id: str,
    user_text: str,
    assistant_text: str,
    user_message_id: str,
    assistant_message_id: str,
) -> None:
    try:
        client = get_llm_client(
            provider="openai_compatible",
            ollama_host=settings.ollama_host,
            openai_api_base=settings.openai_api_base,
            openai_api_key=settings.openai_api_key,
            timeout_seconds=min(settings.request_timeout_seconds, 45.0),
        )
        schedule_memory_fold(
            store=memory_store,
            conversation_id=conversation_id,
            client=client,
            model=settings.default_model,
            user_text=user_text,
            assistant_text=assistant_text,
            user_message_id=user_message_id,
            assistant_message_id=assistant_message_id,
        )
    except Exception:
        # The answer is already durable in L0; a memory-index failure must not
        # retroactively turn it into a failed chat response.
        logger.exception("Could not enqueue direct-chat L2 memory")


async def _buffer_text_stream(
    chunks: AsyncIterator[str],
    *,
    min_chars: int = 64,
    max_wait_ms: int = 55,
) -> AsyncIterator[str]:
    buffer: list[str] = []
    buffered_chars = 0
    last_flush = time.perf_counter()

    async for chunk in chunks:
        if not chunk:
            continue
        buffer.append(chunk)
        buffered_chars += len(chunk)
        elapsed_ms = (time.perf_counter() - last_flush) * 1000
        if buffered_chars >= min_chars or elapsed_ms >= max_wait_ms:
            yield "".join(buffer)
            buffer.clear()
            buffered_chars = 0
            last_flush = time.perf_counter()

    if buffer:
        yield "".join(buffer)


@router.get("/conversations", response_model=list[ConversationResponse])
async def list_conversations(
    history: Annotated[ChatHistory, Depends(get_chat_history)],
) -> list[ConversationResponse]:
    return [ConversationResponse(**summary.__dict__) for summary in history.list_conversations()]


@router.get("/conversations/{conversation_id}/messages", response_model=list[StoredMessageResponse])
async def list_messages(
    conversation_id: str,
    history: Annotated[ChatHistory, Depends(get_chat_history)],
) -> list[StoredMessageResponse]:
    return [StoredMessageResponse(**message.__dict__) for message in history.list_messages(conversation_id)]
