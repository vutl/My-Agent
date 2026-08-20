import asyncio

from app.llm.ollama_client import ChatCompletion, StreamChunk
from app.services.chat_service import ChatService


class FakeClient:
    async def chat(self, *, model, messages, temperature):
        return ChatCompletion(model=model, message=f"{messages[-1]['content']}:{temperature}")

    async def stream_chat(self, *, model, messages, temperature):
        yield StreamChunk(model)
        yield StreamChunk(messages[-1]["content"])
        yield StreamChunk("", done=True, finish_reason="stop")


def test_complete_uses_default_model_when_request_model_is_missing() -> None:
    service = ChatService(client=FakeClient(), default_model="qwen3.5:4b")

    result = asyncio.run(
        service.complete(
            message="hello",
            model=None,
            temperature=0.3,
            system_prompt=None,
        )
    )

    assert result.model == "qwen3.5:4b"
    assert result.message == "hello:0.3"


def test_stream_uses_requested_model() -> None:
    service = ChatService(client=FakeClient(), default_model="qwen3.5:4b")

    async def collect_chunks() -> list[str]:
        return [
            chunk
            async for chunk in service.stream(
                message="hello",
                model="custom-model",
                temperature=0.2,
                system_prompt="be brief",
            )
        ]

    chunks = asyncio.run(collect_chunks())

    assert chunks == ["custom-model", "hello"]


def test_chat_service_packs_persistent_context_as_system_context() -> None:
    captured: dict = {}

    class CapturingClient:
        async def chat(self, *, model, messages, temperature):
            captured["messages"] = messages
            return ChatCompletion(model=model, message="ok")

    service = ChatService(client=CapturingClient(), default_model="cx/gpt-5.5")
    asyncio.run(
        service.complete(
            message="continue",
            model=None,
            temperature=0.2,
            system_prompt="You are Aya.",
            conversation_context="Stable summary plus pending turns.",
        )
    )

    assert captured["messages"][0] == {
        "role": "system",
        "content": "You are Aya.\n\nStable summary plus pending turns.",
    }
    assert captured["messages"][-1] == {"role": "user", "content": "continue"}
