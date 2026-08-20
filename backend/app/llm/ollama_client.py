from collections.abc import AsyncIterator
from dataclasses import dataclass
import json
from typing import Any

import httpx


class OllamaError(RuntimeError):
    """Raised when the Ollama runtime is unavailable or returns an error."""


@dataclass(frozen=True)
class ChatCompletion:
    model: str
    message: str


@dataclass(frozen=True)
class StreamChunk:
    content: str
    done: bool = False
    finish_reason: str | None = None
    metadata: dict[str, Any] | None = None


class OllamaClient:
    def __init__(self, host: str, timeout_seconds: float) -> None:
        self.host = host.rstrip("/")
        self.timeout = httpx.Timeout(timeout_seconds)
        self.keep_alive = "30m"

    async def health(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.host}/api/tags")
                response.raise_for_status()
        except httpx.HTTPError as exc:
            return {
                "reachable": False,
                "host": self.host,
                "error": str(exc),
                "models": [],
            }

        payload = response.json()
        model_names = [model.get("name") for model in payload.get("models", [])]
        return {
            "reachable": True,
            "host": self.host,
            "models": [name for name in model_names if name],
        }

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        num_predict: int = 512,
    ) -> ChatCompletion:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": temperature, "num_predict": num_predict},
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.host}/api/chat", json=payload)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OllamaError(f"Ollama chat failed: {exc}") from exc

        data = response.json()
        if error := data.get("error"):
            raise OllamaError(str(error))

        message = data.get("message", {}).get("content", "")
        return ChatCompletion(model=data.get("model", model), message=message)

    async def stream_chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        num_predict: int = 768,
    ) -> AsyncIterator[StreamChunk]:
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "think": False,
            "keep_alive": self.keep_alive,
            "options": {"temperature": temperature, "num_predict": num_predict},
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", f"{self.host}/api/chat", json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        data = json.loads(line)
                        if error := data.get("error"):
                            raise OllamaError(str(error))
                        if data.get("done"):
                            yield StreamChunk(
                                content="",
                                done=True,
                                finish_reason=data.get("done_reason"),
                                metadata=data,
                            )
                            break
                        yield StreamChunk(content=data.get("message", {}).get("content", ""))
        except httpx.HTTPError as exc:
            raise OllamaError(f"Ollama stream failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise OllamaError(f"Ollama returned malformed stream data: {exc}") from exc
