from collections.abc import AsyncIterator
from dataclasses import dataclass

from app.llm.ollama_client import ChatCompletion, OllamaClient
from app.services.long_term_memory import sentence_safe_clip


RECENT_CHAT_CONTEXT_CHARS = 12_000


@dataclass(frozen=True)
class ChatService:
    client: OllamaClient
    default_model: str

    def _messages(
        self,
        message: str,
        system_prompt: str | None,
        recent_messages: list | None = None,
        conversation_context: str = "",
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        system_parts = [part.strip() for part in (system_prompt, conversation_context) if part and part.strip()]
        if system_parts:
            messages.append({"role": "system", "content": "\n\n".join(system_parts)})

        selected_reversed: list[dict[str, str]] = []
        used = 0
        for stored in reversed((recent_messages or [])[-12:]):
            role = getattr(stored, "role", "")
            content = getattr(stored, "content", "")
            if role in {"user", "assistant"} and content:
                remaining = RECENT_CHAT_CONTEXT_CHARS - used
                if remaining <= 0:
                    break
                clipped = sentence_safe_clip(content, remaining)
                if not clipped:
                    break
                selected_reversed.append({"role": role, "content": clipped})
                used += len(clipped)
        messages.extend(reversed(selected_reversed))
        messages.append({"role": "user", "content": message})
        return messages

    async def complete(
        self,
        *,
        message: str,
        model: str | None,
        temperature: float,
        system_prompt: str | None,
        recent_messages: list | None = None,
        conversation_context: str = "",
    ) -> ChatCompletion:
        selected_model = model or self.default_model
        return await self.client.chat(
            model=selected_model,
            messages=self._messages(
                message,
                system_prompt,
                recent_messages,
                conversation_context,
            ),
            temperature=temperature,
        )

    async def stream(
        self,
        *,
        message: str,
        model: str | None,
        temperature: float,
        system_prompt: str | None,
        recent_messages: list | None = None,
        conversation_context: str = "",
    ) -> AsyncIterator[str]:
        selected_model = model or self.default_model
        async for delta in self.client.stream_chat(
            model=selected_model,
            messages=self._messages(
                message,
                system_prompt,
                recent_messages,
                conversation_context,
            ),
            temperature=temperature,
        ):
            if delta.content:
                yield delta.content
