import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass, replace
import json
from typing import Any

import httpx

from app.llm.ollama_client import ChatCompletion, OllamaError, StreamChunk

_ROUTER_MODEL_NAMESPACES = frozenset({"cx", "cc", "cu", "ag", "gh", "xai"})

# ``/v1/models`` only proves that the gateway catalog is reachable.  Keep the
# last authenticated completion result separately so health/UI can distinguish
# a live catalog from a revoked provider credential without issuing a probe
# completion on every health poll.
_COMPLETION_HEALTH: dict[str, dict[str, Any]] = {}


def _record_completion_health(
    base_url: str,
    *,
    status: str,
    model: str | None = None,
    error: str | None = None,
) -> None:
    payload: dict[str, Any] = {"status": status}
    if model:
        payload["model"] = model
    if error:
        payload["error"] = error[:500]
    _COMPLETION_HEALTH[base_url.rstrip("/")] = payload


def _completion_health(base_url: str) -> dict[str, Any]:
    return dict(_COMPLETION_HEALTH.get(base_url.rstrip("/"), {"status": "unknown"}))


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON text as the model wrote it


@dataclass(frozen=True)
class ToolTurn:
    """One assistant step of a function-calling conversation: either text
    (the answer) or tool calls to run (content may then be empty)."""

    model: str
    content: str
    tool_calls: tuple[ToolCall, ...] = ()


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
            "completion": _completion_health(self.base_url),
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
        """One completion, retried on the SAME model when the gateway reports
        quota (the next upstream account usually answers) or returns an empty
        completion (a transient gateway glitch that otherwise fails the turn)."""
        attempts = 0
        while True:
            try:
                return await self._chat_once(
                    model=model, messages=messages, temperature=temperature,
                    num_predict=num_predict, response_format=response_format,
                )
            except OllamaError as exc:
                if attempts >= _QUOTA_RETRIES or not _is_retryable_error(exc):
                    raise
                attempts += 1
                await asyncio.sleep(_QUOTA_BACKOFF_SECONDS * attempts)

    async def chat_with_tools(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
        temperature: float = 0.2,
        num_predict: int = 1200,
        timeout_seconds: float = 90.0,
    ) -> ToolTurn:
        """One function-calling step, retried on quota like ``chat``. A reply
        with tool calls and no text is valid; one with neither is not. A step
        has no side effects, so a gateway that stalls (read timeout, dropped
        connection) is retried too, with a per-step deadline."""
        attempts = 0
        while True:
            try:
                return await self._tool_turn_once(
                    model=model, messages=messages, tools=tools, tool_choice=tool_choice,
                    temperature=temperature, num_predict=num_predict, timeout_seconds=timeout_seconds,
                )
            except OllamaError as exc:
                transient = any(marker in str(exc) for marker in ("Timeout", "RemoteProtocolError", "ConnectError"))
                if attempts >= _QUOTA_RETRIES or not (_is_retryable_error(exc) or transient):
                    raise
                attempts += 1
                await asyncio.sleep(_QUOTA_BACKOFF_SECONDS * attempts)

    async def chat_with_tools_stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
        temperature: float = 0.2,
        num_predict: int = 1200,
        timeout_seconds: float = 90.0,
    ) -> AsyncIterator[tuple[str, Any]]:
        """Stream one tool turn; retry only before any public delta escapes."""
        attempts, yielded = 0, False
        while True:
            try:
                async with aclosing(self._tool_turn_stream_once(
                    model=model, messages=messages, tools=tools, tool_choice=tool_choice,
                    temperature=temperature, num_predict=num_predict, timeout_seconds=timeout_seconds,
                )) as events:
                    async for event in events:
                        yielded = True
                        yield event
                return
            except OllamaError as exc:
                transient = any(marker in str(exc) for marker in ("Timeout", "RemoteProtocolError", "ConnectError"))
                if yielded or attempts >= _QUOTA_RETRIES or not (_is_retryable_error(exc) or transient):
                    raise
                attempts += 1
                await asyncio.sleep(_QUOTA_BACKOFF_SECONDS * attempts)

    async def _tool_turn_stream_once(
        self, *, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
        tool_choice: str, temperature: float, num_predict: int, timeout_seconds: float,
    ) -> AsyncIterator[tuple[str, Any]]:
        payload: dict[str, Any] = {
            "model": model, "messages": messages, "stream": True,
            "temperature": temperature, "max_tokens": num_predict,
        }
        if tools:
            payload.update(tools=tools, tool_choice=tool_choice)
        content, leading = [], []
        calls: dict[int, dict[str, Any]] = {}
        started: set[int] = set()
        reported_model = model
        loop = asyncio.get_running_loop()
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds, connect=10.0)) as client:
                async with client.stream(
                    "POST", f"{self.base_url}/chat/completions", json=payload, headers=self._headers(),
                ) as response:
                    if response.is_error:
                        await response.aread()
                    response.raise_for_status()
                    # The read timeout restarts on every byte, so heartbeats alone
                    # could hold a stalled turn open forever: bound the wait
                    # between real data frames as well.
                    last_frame = loop.time()
                    async for line in response.aiter_lines():
                        if loop.time() - last_frame > timeout_seconds:
                            raise OllamaError(f"Timeout: no stream data for {timeout_seconds:.0f}s")
                        if line.startswith("data:"):
                            line = line[5:].strip()
                        elif not line.lstrip().startswith(("{", "[")):
                            continue
                        if line == "[DONE]":
                            break
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except ValueError as exc:
                            raise OllamaError("LLM stream returned malformed JSON data") from exc
                        if not isinstance(data, dict):
                            raise OllamaError("LLM stream returned a non-object JSON frame")
                        last_frame = loop.time()
                        if error := data.get("error"):
                            raise OllamaError(str(error.get("message") if isinstance(error, dict) else error))
                        _assert_reported_model(requested=model, reported=data.get("model"))
                        reported_model = data.get("model") or reported_model
                        choices = data.get("choices")
                        if choices is None or choices == []:
                            continue
                        if not isinstance(choices, list) or not isinstance(choices[0], dict):
                            raise OllamaError("LLM stream returned an invalid completion choice")
                        choice = choices[0]
                        delta = choice.get("delta")
                        if delta is None:
                            delta = {}
                        if not isinstance(delta, dict):
                            raise OllamaError("LLM stream returned an invalid completion delta")
                        text = delta.get("content")
                        if text is not None and not isinstance(text, str):
                            raise OllamaError("LLM stream returned non-text completion content")
                        if text:
                            content.append(text)
                            # Keep whitespace-only empty attempts invisible and retryable.
                            if leading or len(content) == 1:
                                leading.append(text)
                                if "".join(leading).strip():
                                    yield "text", "".join(leading)
                                    leading.clear()
                            else:
                                yield "text", text
                        fragments = delta.get("tool_calls") or []
                        if not isinstance(fragments, list):
                            raise OllamaError("LLM stream returned invalid tool calls")
                        for fragment in fragments:
                            if not isinstance(fragment, dict) or type(fragment.get("index")) is not int:
                                raise OllamaError("LLM stream returned an invalid tool call index")
                            index = fragment["index"]
                            item = calls.setdefault(index, {"id": "", "function": {"name": "", "arguments": ""}})
                            function = fragment.get("function") or {}
                            if not isinstance(function, dict):
                                raise OllamaError("LLM stream returned an invalid tool function")
                            for key, target in (("id", item), ("name", item["function"]), ("arguments", item["function"])):
                                value = fragment.get(key) if key == "id" else function.get(key)
                                if value is not None and not isinstance(value, str):
                                    raise OllamaError("LLM stream returned non-text tool call data")
                                target[key] += value or ""
                            if index not in started and item["id"] and item["function"]["name"].strip():
                                started.add(index)
                                yield "tool_call_start", {"index": index, "id": item["id"],
                                                          "name": item["function"]["name"].strip()}
                            if function.get("arguments"):
                                yield "tool_call_args", {"index": index, "delta": function["arguments"]}
                        if choice.get("finish_reason"):
                            break
        except httpx.HTTPStatusError as exc:
            raise OllamaError(f"LLM stream failed ({exc.response.status_code}): {exc.response.text}") from exc
        except httpx.HTTPError as exc:
            raise OllamaError(f"LLM stream failed: {type(exc).__name__}: {exc}") from exc
        turn = self._parse_tool_turn(model, {
            "model": reported_model, "choices": [{"message": {
                "content": "".join(content), "tool_calls": [calls[i] for i in sorted(calls)],
            }}],
        })
        for index, call in zip((i for i in sorted(calls) if calls[i]["function"]["name"].strip()),
                               turn.tool_calls, strict=True):
            if index not in started:
                yield "tool_call_start", {"index": index, "id": call.id, "name": call.name}
        yield "turn", turn

    async def _tool_turn_once(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str,
        temperature: float,
        num_predict: int,
        timeout_seconds: float,
    ) -> ToolTurn:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "max_tokens": num_predict,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds, connect=10.0)) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=self._headers(),
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise OllamaError(f"LLM request failed ({exc.response.status_code}): {exc.response.text}") from exc
        except httpx.HTTPError as exc:
            raise OllamaError(f"LLM request failed: {type(exc).__name__}: {exc}") from exc
        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaError("LLM request returned an invalid JSON envelope") from exc
        return self._parse_tool_turn(model, data)

    def _parse_tool_turn(self, model: str, data: Any) -> ToolTurn:
        """Use the same envelope validation and call defaults for both paths."""
        if not isinstance(data, dict):
            raise OllamaError("LLM request returned a non-object JSON envelope")
        if error := data.get("error"):
            raise OllamaError(str(error.get("message") if isinstance(error, dict) else error))
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise OllamaError("LLM request returned no completion choice")
        message = choices[0].get("message") if isinstance(choices[0].get("message"), dict) else {}
        calls: list[ToolCall] = []
        for item in message.get("tool_calls") or []:
            function = (item or {}).get("function") or {}
            name = str(function.get("name") or "").strip()
            if name:
                calls.append(ToolCall(
                    id=str(item.get("id") or f"call_{len(calls)}"),
                    name=name,
                    arguments=str(function.get("arguments") or "{}"),
                ))
        content = message.get("content") if isinstance(message.get("content"), str) else ""
        if not calls and not content.strip():
            raise OllamaError("LLM request returned an empty completion")
        reported_model = data.get("model")
        _assert_reported_model(requested=model, reported=reported_model)
        _record_completion_health(self.base_url, status="ok", model=str(reported_model or model))
        return ToolTurn(model=str(reported_model or model), content=content, tool_calls=tuple(calls))

    async def _chat_once(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        num_predict: int,
        response_format: dict[str, Any] | None,
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
            _record_completion_health(
                self.base_url,
                status="failed",
                model=model,
                error=f"HTTP {exc.response.status_code}: {exc.response.text}",
            )
            raise OllamaError(f"LLM request failed ({exc.response.status_code}): {exc.response.text}") from exc
        except httpx.HTTPError as exc:
            detail = f"{type(exc).__name__}: {str(exc) or 'request failed without a response'}"
            _record_completion_health(
                self.base_url,
                status="failed",
                model=model,
                error=detail,
            )
            raise OllamaError(f"LLM request failed: {detail}") from exc

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
            _record_completion_health(
                self.base_url,
                status="failed",
                model=model,
                error="gateway returned an empty completion",
            )
            raise OllamaError("LLM request returned an empty completion")
        reported_model = data.get("model")
        _assert_reported_model(requested=model, reported=reported_model)
        returned_model = str(reported_model).strip() if reported_model else model
        _record_completion_health(
            self.base_url,
            status="ok",
            model=returned_model,
        )
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
        """Stream a completion, retrying quota/empty responses on the SAME model.

        The gateway round-robins across upstream accounts, so a 429 "usage
        limit" from one account is usually cleared by the next attempt. Only
        retryable errors are retried, only before visible content was yielded (a
        partial answer must never be replayed), and never with another model.
        """
        attempts = 0
        while True:
            yielded = False
            leading = ""
            try:
                async with aclosing(self._stream_chat_once(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    num_predict=num_predict,
                    response_format=response_format,
                )) as events:
                    async for chunk in events:
                        if not yielded:
                            leading += chunk.content
                            if not leading.strip():
                                continue
                            chunk = replace(chunk, content=leading)
                        yielded = True
                        yield chunk
                if not yielded:
                    _record_completion_health(self.base_url, status="failed", model=model,
                                              error="gateway returned an empty completion")
                    raise OllamaError("LLM request returned an empty completion")
                return
            except OllamaError as exc:
                if yielded or attempts >= _QUOTA_RETRIES or not _is_retryable_error(exc):
                    raise
                attempts += 1
                await asyncio.sleep(_QUOTA_BACKOFF_SECONDS * attempts)

    async def _stream_chat_once(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        num_predict: int,
        response_format: dict[str, Any] | None,
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
                            _record_completion_health(
                                self.base_url,
                                status="ok",
                                model=model,
                            )
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
                            _record_completion_health(
                                self.base_url,
                                status="ok",
                                model=model,
                            )
                            yield StreamChunk(content=content, done=True, finish_reason=finish_reason)
                            return

                        if content:
                            yield StreamChunk(content=content)

                # Be tolerant of gateways that close a valid HTTP stream without
                # an explicit [DONE] or finish_reason frame.
                _record_completion_health(
                    self.base_url,
                    status="ok",
                    model=model,
                )
                yield StreamChunk(content="", done=True, finish_reason="stream_closed")

        except httpx.HTTPStatusError as exc:
            _record_completion_health(
                self.base_url,
                status="failed",
                model=model,
                error=f"HTTP {exc.response.status_code}: {exc.response.text}",
            )
            raise OllamaError(f"LLM stream failed ({exc.response.status_code}): {exc.response.text}") from exc
        except httpx.HTTPError as exc:
            detail = f"{type(exc).__name__}: {str(exc) or 'request failed without a response'}"
            _record_completion_health(
                self.base_url,
                status="failed",
                model=model,
                error=detail,
            )
            raise OllamaError(f"LLM stream failed: {detail}") from exc


_QUOTA_RETRIES = 2
_QUOTA_BACKOFF_SECONDS = 2.0


def _is_quota_error(exc: BaseException) -> bool:
    """Gateways report upstream quota as 429 or as a 503 wrapping ``[429]``."""
    text = str(exc).casefold()
    return "usage limit" in text or "[429]" in text or " 429" in text or "rate limit" in text


def _is_retryable_error(exc: BaseException) -> bool:
    return _is_quota_error(exc) or "empty completion" in str(exc).casefold()


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
