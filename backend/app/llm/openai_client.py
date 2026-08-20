from collections.abc import AsyncIterator
from dataclasses import dataclass
import json
from typing import Any

import httpx

from app.llm.ollama_client import ChatCompletion, OllamaError, StreamChunk

_ROUTER_MODEL_NAMESPACES = frozenset({"cx", "cc", "cu", "ag", "gh", "xai"})


class OpenAICompatibleClient:
    """HTTP client for any OpenAI-compatible endpoint (9router, OpenRouter, etc.)."""

    def __init__(self, base_url: str, api_key: str, timeout_seconds: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = httpx.Timeout(timeout_seconds)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def health(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{self.base_url}/models",
                    headers=self._headers(),
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            return {
                "reachable": False,
                "host": self.base_url,
                "error": str(exc),
                "models": [],
            }

        payload = response.json()
        model_ids = [m.get("id") for m in payload.get("data", [])]
        return {
            "reachable": True,
            "host": self.base_url,
            "models": [mid for mid in model_ids if mid],
        }

    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        num_predict: int = 512,
        response_format: dict[str, Any] | None = None,
    ) -> ChatCompletion:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "max_tokens": num_predict,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self._headers(),
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise OllamaError(f"LLM request failed ({exc.response.status_code}): {exc.response.text}") from exc
        except httpx.HTTPError as exc:
            raise OllamaError(f"LLM request failed: {exc}") from exc

        try:
            data = response.json()
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise OllamaError("LLM request returned an invalid JSON envelope") from exc
        if not isinstance(data, dict):
            raise OllamaError("LLM request returned a non-object JSON envelope")
        if error := data.get("error"):
            raise OllamaError(str(error.get("message") if isinstance(error, dict) else error))

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise OllamaError("LLM request returned no completion choice")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise OllamaError("LLM request returned an invalid completion message")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise OllamaError("LLM request returned an empty completion")
        reported_model = data.get("model")
        _assert_reported_model(requested=model, reported=reported_model)
        returned_model = str(reported_model).strip() if reported_model else model
        return ChatCompletion(model=returned_model, message=content)

    async def stream_chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        num_predict: int = 768,
        response_format: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "max_tokens": num_predict,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=self._headers(),
                ) as response:
                    # Streaming responses do not buffer error bodies. Read a
                    # non-2xx response before raise_for_status so diagnostics
                    # expose the gateway error instead of httpx's
                    # ResponseNotRead exception.
                    if response.is_error:
                        await response.aread()
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        # Standard SSE may include comments/heartbeats plus
                        # event/id/retry metadata. Only ``data:`` frames (and
                        # raw-JSON lines emitted by a few compatible gateways)
                        # are completion payloads.
                        if line.startswith(":") or line.startswith(
                            ("event:", "id:", "retry:")
                        ):
                            continue
                        is_data_frame = line.startswith("data:")
                        if is_data_frame:
                            line = line[5:].strip()
                        elif not line.lstrip().startswith(("{", "[")):
                            continue
                        if line == "[DONE]":
                            yield StreamChunk(content="", done=True, finish_reason="stop")
                            return
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise OllamaError(
                                "LLM stream returned malformed JSON data"
                            ) from exc
                        if not isinstance(data, dict):
                            raise OllamaError(
                                "LLM stream returned a non-object JSON frame"
                            )

                        if error := data.get("error"):
                            msg = error.get("message") if isinstance(error, dict) else str(error)
                            raise OllamaError(str(msg))

                        _assert_reported_model(
                            requested=model,
                            reported=data.get("model"),
                        )

                        choices = data.get("choices")
                        if choices is None or choices == []:
                            # Usage-only / gateway metadata frames are valid.
                            continue
                        if (
                            not isinstance(choices, list)
                            or not isinstance(choices[0], dict)
                        ):
                            raise OllamaError(
                                "LLM stream returned an invalid completion choice"
                            )

                        choice = choices[0]
                        finish_reason: str | None = choice.get("finish_reason")
                        delta = choice.get("delta")
                        if delta is None:
                            delta = {}
                        if not isinstance(delta, dict):
                            raise OllamaError(
                                "LLM stream returned an invalid completion delta"
                            )
                        raw_content = delta.get("content")
                        if raw_content is not None and not isinstance(raw_content, str):
                            raise OllamaError(
                                "LLM stream returned non-text completion content"
                            )
                        content = raw_content or ""

                        if finish_reason:
                            yield StreamChunk(content=content, done=True, finish_reason=finish_reason)
                            return

                        if content:
                            yield StreamChunk(content=content)

                # Be tolerant of gateways that close a valid HTTP stream without
                # an explicit [DONE] or finish_reason frame.
                yield StreamChunk(content="", done=True, finish_reason="stream_closed")

        except httpx.HTTPStatusError as exc:
            raise OllamaError(f"LLM stream failed ({exc.response.status_code}): {exc.response.text}") from exc
        except httpx.HTTPError as exc:
            raise OllamaError(f"LLM stream failed: {exc}") from exc


def get_llm_client(
    *,
    provider: str,
    ollama_host: str,
    openai_api_base: str,
    openai_api_key: str,
    timeout_seconds: float,
) -> "OllamaClient | OpenAICompatibleClient":
    from app.llm.ollama_client import OllamaClient

    normalized_provider = (provider or "").strip().lower()
    if normalized_provider == "openai_compatible":
        return OpenAICompatibleClient(
            base_url=openai_api_base,
            api_key=openai_api_key,
            timeout_seconds=timeout_seconds,
        )
    if normalized_provider == "ollama":
        return OllamaClient(ollama_host, timeout_seconds)
    raise ValueError(
        f"Unsupported LLM provider {provider!r}; refusing to silently switch providers"
    )


def _assert_reported_model(*, requested: str, reported: object) -> None:
    """Reject substitutions while accepting 9router's de-namespaced model ID."""
    if reported is None:
        return
    actual = str(reported).strip()
    if not actual:
        raise OllamaError(
            f"LLM gateway returned an empty model identifier for requested {requested!r}"
        )
    if not _reported_model_matches(requested=requested, reported=actual):
        raise OllamaError(
            f"LLM gateway model mismatch: requested {requested!r}, reported {actual!r}"
        )


def _reported_model_matches(*, requested: str, reported: str) -> bool:
    """Match an exact ID or the same ID after 9router removes its route prefix."""
    selected = str(requested or "").strip()
    actual = str(reported or "").strip()
    if not selected or not actual:
        return False
    if actual == selected:
        return True
    namespace, separator, upstream_model = selected.partition("/")
    return bool(
        separator
        and namespace in _ROUTER_MODEL_NAMESPACES
        and upstream_model
        and actual == upstream_model
    )


@dataclass
class _ClientType:
    """Type alias used for annotations."""
