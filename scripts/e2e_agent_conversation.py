#!/usr/bin/env python3
"""Repeatable end-to-end audit for paper focus, memory and visual sources.

The script intentionally drives the public HTTP/SSE API exactly like the desktop
client.  It creates a fresh conversation, runs seven turns, then verifies the
persisted history and each agent run.  It never reads SQLite directly, so a pass
means the API contract used by the UI also works.

Run with the backend, 9router and Ollama already available::

    python scripts/e2e_agent_conversation.py

The answer and router model are deliberately fixed to ``cx/gpt-5.6-sol``.  A JSON
report is always printed; assertion or transport failures return exit code 1.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
import json
from pathlib import PurePath
import re
import sys
import time
from typing import Any
import unicodedata
import urllib.error
import urllib.parse
import urllib.request


MODEL = "cx/gpt-5.6-sol"
REQUIRED_BENCHMARK_METRICS = frozenset({"accuracy", "f1", "ccc"})
QUESTIONS = (
    "Trong paper ASPIRE, mục tiêu và kiến trúc tổng thể của mô hình là gì?",
    "Ngoài lề một chút: bạn thích cà phê hay trà đá hơn? Trả lời ngắn thôi.",
    "Quay lại bài lúc nãy: benchmark Acc, F1 và CCC theo evidence nói gì?",
    "Chuyển sang paper WhiSER và giải thích kiến trúc cùng contribution chính.",
    "Quay lại bài trước: tóm tắt contribution chính của bài đó.",
    (
        "Vẫn ở paper này: cho mình figure hoặc sơ đồ kiến trúc phù hợp nhất, "
        "trả kèm hình và giải thích theo caption cùng ngữ cảnh trang; đừng lấy logo."
    ),
    (
        "Vẫn paper này: lấy đúng bảng benchmark có Acc, F1 và CCC; "
        "nếu evidence thiếu thì nói thiếu, không tự điền số."
    ),
)

FORBIDDEN_VISUAL_KINDS = {
    "branding",
    "decorative",
    "logo",
    "page",
    "publisher_mark",
}
QUANTITATIVE_METRIC_RE = re.compile(
    r"(?<![\w-])(?:accuracy|acc|f1(?:[- ]score)?|ccc|uar|wa|ua|wer|"
    r"precision|recall|auc|eer|trainable parameters?|parameter count|params)(?![\w-])",
    re.IGNORECASE,
)
QUANTITATIVE_VALUE_RE = re.compile(
    r"(?<![\w.-])[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)\s*(?:[kmb]|%)?(?![%\w-])",
    re.IGNORECASE,
)
QUANTITATIVE_REFERENCE_PREFIX_RE = re.compile(
    r"(?:figure|fig\.?|table|page|section|sec\.?|hình|bảng|trang)\s*$",
    re.IGNORECASE,
)
QUANTITATIVE_STRUCTURAL_SUFFIX_RE = re.compile(
    r"^\s*(?:(?:top|transformer|encoder|decoder|attention|hidden)\s+)*"
    r"(?:layers?|heads?|epochs?|folds?|stages?|blocks?|modules?)\b",
    re.IGNORECASE,
)


class E2ETransportError(RuntimeError):
    """Raised when an HTTP request or SSE payload is unusable."""


@dataclass(frozen=True)
class SSEEvent:
    event: str
    data: dict[str, Any]


@dataclass
class Turn:
    index: int
    question: str
    events: list[SSEEvent]
    answer: str
    elapsed_seconds: float

    def payloads(self, event_name: str) -> list[dict[str, Any]]:
        return [item.data for item in self.events if item.event == event_name]

    def first(self, event_name: str) -> dict[str, Any]:
        payloads = self.payloads(event_name)
        return payloads[0] if payloads else {}

    def last(self, event_name: str) -> dict[str, Any]:
        payloads = self.payloads(event_name)
        return payloads[-1] if payloads else {}

    @property
    def sources(self) -> list[dict[str, Any]]:
        # The fallback path can emit a retrieval after an initial chat answer.
        # The last non-empty retrieval is the one persisted on the assistant.
        for payload in reversed(self.payloads("retrieval.completed")):
            documents = payload.get("documents")
            if isinstance(documents, list):
                return [item for item in documents if isinstance(item, dict)]
        return []

    @property
    def conversation_id(self) -> str | None:
        value = self.first("run.started").get("conversation_id")
        return str(value) if value else None

    @property
    def run_id(self) -> str | None:
        value = self.first("run.started").get("run_id")
        return str(value) if value else None


@dataclass
class Audit:
    errors: list[str] = field(default_factory=list)

    def check(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)


def _decode_line(raw_line: bytes | str) -> str:
    if isinstance(raw_line, bytes):
        return raw_line.decode("utf-8")
    return raw_line


def parse_sse_lines(lines: Iterable[bytes | str]) -> Iterator[SSEEvent]:
    """Parse SSE according to field and blank-line boundaries.

    FastAPI currently sends one JSON ``data:`` line, but handling comments,
    CRLF and multiple data lines makes this useful if buffering changes later.
    """

    event_name = "message"
    data_lines: list[str] = []

    def emit() -> SSEEvent | None:
        nonlocal event_name, data_lines
        if not data_lines:
            event_name = "message"
            return None
        raw_data = "\n".join(data_lines)
        try:
            payload = json.loads(raw_data)
        except json.JSONDecodeError as exc:
            raise E2ETransportError(
                f"Malformed JSON in SSE event {event_name!r}: {raw_data[:500]}"
            ) from exc
        if not isinstance(payload, dict):
            raise E2ETransportError(
                f"SSE event {event_name!r} returned {type(payload).__name__}, expected object"
            )
        result = SSEEvent(event=event_name, data=payload)
        event_name = "message"
        data_lines = []
        return result

    for raw_line in lines:
        line = _decode_line(raw_line).rstrip("\r\n")
        if not line:
            event = emit()
            if event is not None:
                yield event
            continue
        if line.startswith(":"):
            continue
        field_name, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field_name == "event":
            event_name = value or "message"
        elif field_name == "data":
            data_lines.append(value)
        # id/retry and unknown extension fields do not affect this audit.

    event = emit()
    if event is not None:
        yield event


def _url(base_url: str, path: str) -> str:
    return urllib.parse.urljoin(f"{base_url.rstrip('/')}/", path.lstrip("/"))


def _http_error_message(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace").strip()
    except Exception:  # pragma: no cover - defensive diagnostics
        body = ""
    suffix = f": {body[:1200]}" if body else ""
    return f"HTTP {exc.code} {exc.reason}{suffix}"


def _request_json(base_url: str, path: str, *, timeout: float) -> object:
    request = urllib.request.Request(_url(base_url, path), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise E2ETransportError(f"GET {path} failed: {_http_error_message(exc)}") from exc
    except urllib.error.URLError as exc:
        raise E2ETransportError(f"GET {path} failed: {exc.reason}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise E2ETransportError(f"GET {path} returned malformed JSON: {raw[:500]}") from exc


def _request_image(base_url: str, path: str, *, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(_url(base_url, path), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get_content_type()
            prefix = response.read(64)
            status = response.status
    except urllib.error.HTTPError as exc:
        raise E2ETransportError(f"GET {path} failed: {_http_error_message(exc)}") from exc
    except urllib.error.URLError as exc:
        raise E2ETransportError(f"GET {path} failed: {exc.reason}") from exc
    return {
        "url": path,
        "status": status,
        "content_type": content_type,
        "has_bytes": bool(prefix),
    }


def _run_turn(
    base_url: str,
    *,
    index: int,
    question: str,
    conversation_id: str | None,
    timeout: float,
) -> Turn:
    payload = {
        "conversation_id": conversation_id,
        "task": question,
        "mode": "auto",
        "model": MODEL,
        "temperature": 0.2,
        "allowed_tools": ["search_local_docs", "retrieve_visual_assets"],
        "require_confirmation": True,
        "collection_id": None,
        "retrieval_mode": "auto",
        "agent_reasoning": "smart",
    }
    request = urllib.request.Request(
        _url(base_url, "/agent/run/stream"),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get_content_type()
            if content_type != "text/event-stream":
                body = response.read().decode("utf-8", errors="replace")
                raise E2ETransportError(
                    f"turn {index}: expected text/event-stream, got {content_type}: {body[:500]}"
                )
            events = list(parse_sse_lines(response))
    except urllib.error.HTTPError as exc:
        raise E2ETransportError(
            f"turn {index}: POST /agent/run/stream failed: {_http_error_message(exc)}"
        ) from exc
    except urllib.error.URLError as exc:
        raise E2ETransportError(f"turn {index}: agent stream failed: {exc.reason}") from exc

    answer = "".join(
        str(event.data.get("delta") or "")
        for event in events
        if event.event == "message.delta"
    )
    return Turn(
        index=index,
        question=question,
        events=events,
        answer=answer,
        elapsed_seconds=round(time.perf_counter() - started, 2),
    )


def _normalized_filename(value: object) -> str:
    text = str(value or "").strip()
    return PurePath(text).name.casefold() if text else ""


def _filenames(turn: Turn) -> set[str]:
    return {
        filename
        for source in turn.sources
        if (filename := _normalized_filename(source.get("filename")))
    }


def _document_ids_from_sources(turn: Turn) -> set[str]:
    return {
        str(source["document_id"])
        for source in turn.sources
        if source.get("document_id")
    }


def _focus_ids(turn: Turn) -> set[str]:
    return {
        str(item)
        for item in turn.first("query.rewritten").get("focus_document_ids") or []
        if item
    }


def _active_ids(turn: Turn) -> set[str]:
    return {
        str(item)
        for item in turn.last("agent.working_state.updated").get("active_document_ids") or []
        if item
    }


def _recent_thread_ids(turn: Turn) -> list[set[str]]:
    result: list[set[str]] = []
    for thread in turn.last("agent.working_state.updated").get("recent_document_threads") or []:
        if not isinstance(thread, dict):
            continue
        result.append({str(item) for item in thread.get("document_ids") or [] if item})
    return result


def _figure_sources_from(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [source for source in sources if source.get("figure_id")]


def _figure_sources(turn: Turn) -> list[dict[str, Any]]:
    return _figure_sources_from(turn.sources)


def _is_table_source(source: dict[str, Any]) -> bool:
    return bool(
        source.get("table_id")
        or str(source.get("artifact_type") or "").casefold() == "table"
        or str(source.get("chunk_type") or "").casefold() == "table"
        or str(source.get("chunk_id") or "").startswith("table:")
    )


def _table_sources_from(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [source for source in sources if _is_table_source(source)]


def _table_sources(turn: Turn) -> list[dict[str, Any]]:
    return _table_sources_from(turn.sources)


def _fold_text(value: object) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value or "")).casefold()
    normalized = "".join(character for character in decomposed if not unicodedata.combining(character))
    return " ".join(normalized.split())


def _looks_like_architecture_figure(source: dict[str, Any]) -> bool:
    searchable = _fold_text(
        " ".join(
            str(source.get(field) or "")
            for field in ("figure_type", "caption", "content", "citation_label")
        )
    )
    return any(
        marker in searchable
        for marker in ("architecture", "architectural", "kien truc", "model pipeline")
    )


def _table_contains_requested_metrics(source: dict[str, Any]) -> bool:
    text = _fold_text(
        f"{source.get('caption') or ''}\n{source.get('content') or ''}"
    )
    return all(marker in text for marker in ("acc", "f1", "ccc"))


def _source_identity(source: dict[str, Any]) -> str:
    # Stable visual identities are preserved by the bounded history projection
    # and are stronger than a backend-specific chunk id.
    for field_name in ("figure_id", "table_id", "chunk_id", "source_id"):
        if source.get(field_name):
            return f"{field_name}:{source[field_name]}"
    return "fallback:" + "|".join(
        str(source.get(field_name) or "")
        for field_name in ("document_id", "page_number", "caption", "chunk_type")
    )


def _retrieval_evidence_is_valid(turn: Turn) -> bool:
    payload = turn.last("retrieval.completed")
    validation = payload.get("evidence_validation") or {}
    return validation.get("valid") is True


def _answer_evidence_is_valid(turn: Turn) -> bool:
    return turn.last("answer.evidence.validated").get("valid") is True


def _answer_has_metric_values(answer: str) -> bool:
    for line in answer.splitlines():
        if not QUANTITATIVE_METRIC_RE.search(line):
            continue
        for match in QUANTITATIVE_VALUE_RE.finditer(line):
            prefix = line[: match.start()]
            suffix = line[match.end() :]
            if QUANTITATIVE_REFERENCE_PREFIX_RE.search(prefix):
                continue
            if QUANTITATIVE_STRUCTURAL_SUFFIX_RE.search(suffix):
                continue
            raw_number = re.match(r"[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)", match.group())
            if raw_number:
                normalized = raw_number.group().replace(",", ".")
                try:
                    numeric = float(normalized)
                except ValueError:
                    numeric = None
                if numeric is not None and numeric.is_integer() and 1900 <= numeric <= 2100:
                    continue
            return True
    return False


def _assert_metric_values_were_checked(
    audit: Audit,
    turn: Turn,
    *,
    required_metrics: frozenset[str] | set[str] | None = None,
) -> None:
    has_metric_values = _answer_has_metric_values(turn.answer)
    if required_metrics is None and not has_metric_values:
        return
    validation = turn.last("answer.evidence.validated")
    checked_claims = validation.get("checked_claims")
    audit.check(
        isinstance(checked_claims, list) and bool(checked_claims),
        (
            f"turn {turn.index}: answer contains metric values but "
            "answer.evidence.validated checked no claims"
        ),
    )
    checked_metrics = {
        str(claim.get("metric") or "").strip().casefold()
        for claim in checked_claims or []
        if isinstance(claim, dict)
    }
    if required_metrics is not None:
        missing = set(required_metrics) - checked_metrics
        audit.check(
            not missing,
            (
                f"turn {turn.index}: quantitative answer omitted unchecked requested "
                f"metrics; missing={sorted(missing)}, checked={sorted(checked_metrics)}"
            ),
        )


def _assert_sources_are_only(
    audit: Audit,
    turn: Turn,
    *,
    filename: str,
    document_ids: set[str] | None = None,
) -> None:
    expected_filename = filename.casefold()
    audit.check(bool(turn.sources), f"turn {turn.index}: no retrieved sources")
    audit.check(
        _filenames(turn) == {expected_filename},
        f"turn {turn.index}: expected only {filename}, got {sorted(_filenames(turn))}",
    )
    audit.check(
        all(source.get("document_id") for source in turn.sources),
        f"turn {turn.index}: a source has no document_id",
    )
    if document_ids is not None:
        audit.check(
            _document_ids_from_sources(turn) == document_ids,
            (
                f"turn {turn.index}: source document ids escaped focus; "
                f"expected={sorted(document_ids)}, got={sorted(_document_ids_from_sources(turn))}"
            ),
        )


def _validate_preflight(health: object, audit: Audit) -> dict[str, Any]:
    if not isinstance(health, dict):
        audit.check(False, "GET /health did not return an object")
        return {}
    gateway = health.get("gateway") or {}
    vision = health.get("vision") or {}
    ollama = health.get("ollama") or {}
    audit.check(gateway.get("reachable") is True, "9router gateway is not reachable")
    audit.check(ollama.get("reachable") is True, "Ollama embeddings runtime is not reachable")
    audit.check(health.get("default_model") == MODEL, f"default_model is not {MODEL}")
    audit.check(health.get("router_model") == MODEL, f"router_model is not {MODEL}")
    audit.check(vision.get("provider") == "9router", "vision provider is not 9router")
    audit.check(vision.get("model") == MODEL, f"vision model is not {MODEL}")
    return {
        "status": health.get("status"),
        "gateway_reachable": gateway.get("reachable"),
        "ollama_reachable": ollama.get("reachable"),
        "default_model": health.get("default_model"),
        "router_model": health.get("router_model"),
        "vision_provider": vision.get("provider"),
        "vision_model": vision.get("model"),
    }


def _validate_turns(turns: list[Turn], audit: Audit) -> tuple[set[str], set[str]]:
    for turn in turns:
        failed = turn.first("run.failed")
        audit.check(not failed, f"turn {turn.index}: run.failed: {failed.get('error')}")
        audit.check(bool(turn.first("run.completed")), f"turn {turn.index}: missing run.completed")
        audit.check(bool(turn.answer.strip()), f"turn {turn.index}: empty answer")
        audit.check(
            _fold_text("chưa có đủ evidence khớp để khẳng định các số liệu định lượng")
            not in _fold_text(turn.answer),
            f"turn {turn.index}: agent fell back despite the known ASPIRE/WhiSER evidence fixture",
        )
        audit.check(
            turn.first("run.started").get("model") == MODEL,
            f"turn {turn.index}: run did not use {MODEL}",
        )
        validation = turn.last("answer.evidence.validated")
        if validation:
            audit.check(
                validation.get("fallback_used") is False,
                f"turn {turn.index}: quantitative evidence guard used fallback",
            )

    conversation_ids = {turn.conversation_id for turn in turns if turn.conversation_id}
    audit.check(len(conversation_ids) == 1, "conversation_id changed across turns")

    aspire_ids = _document_ids_from_sources(turns[0])
    audit.check(bool(aspire_ids), "turn 1: ASPIRE document id is empty")
    _assert_sources_are_only(audit, turns[0], filename="ASPIRE.pdf", document_ids=aspire_ids)
    audit.check(_focus_ids(turns[0]) == aspire_ids, "turn 1: focus is not exactly ASPIRE")
    audit.check(_active_ids(turns[0]) == aspire_ids, "turn 1: L1 was not committed to ASPIRE")
    audit.check(_retrieval_evidence_is_valid(turns[0]), "turn 1: retrieval evidence invalid")
    audit.check(_answer_evidence_is_valid(turns[0]), "turn 1: answer evidence invalid")

    casual = turns[1]
    route_state = casual.first("agent.route.decided").get("working_state") or {}
    route_active_ids = {str(item) for item in route_state.get("active_document_ids") or [] if item}
    audit.check(
        casual.first("agent.route.decided").get("route") == "chat",
        "turn 2: casual detour was not routed as chat",
    )
    audit.check(not casual.payloads("retrieval.started"), "turn 2: casual detour started retrieval")
    audit.check(not casual.payloads("retrieval.completed"), "turn 2: casual detour retrieved docs")
    audit.check(
        not any(
            item.get("tool_name") == "search_local_docs"
            for item in casual.payloads("tool.started")
        ),
        "turn 2: casual detour started search_local_docs",
    )
    audit.check(route_active_ids == aspire_ids, "turn 2: casual detour lost ASPIRE L1")
    audit.check(
        not casual.payloads("agent.working_state.updated"),
        "turn 2: casual detour unexpectedly rewrote L1",
    )

    _assert_sources_are_only(audit, turns[2], filename="ASPIRE.pdf", document_ids=aspire_ids)
    audit.check(_focus_ids(turns[2]) == aspire_ids, "turn 3: resume focus differs from ASPIRE")
    audit.check(_active_ids(turns[2]) == aspire_ids, "turn 3: L1 no longer points to ASPIRE")
    audit.check(_retrieval_evidence_is_valid(turns[2]), "turn 3: retrieval evidence invalid")
    audit.check(_answer_evidence_is_valid(turns[2]), "turn 3: answer evidence invalid")
    _assert_metric_values_were_checked(
        audit,
        turns[2],
        required_metrics=REQUIRED_BENCHMARK_METRICS,
    )

    whiser_ids = _document_ids_from_sources(turns[3])
    audit.check(bool(whiser_ids), "turn 4: WhiSER document id is empty")
    audit.check(whiser_ids.isdisjoint(aspire_ids), "turn 4: WhiSER and ASPIRE ids overlap")
    _assert_sources_are_only(audit, turns[3], filename="WhiSER.pdf", document_ids=whiser_ids)
    audit.check(_focus_ids(turns[3]) == whiser_ids, "turn 4: focus is not exactly WhiSER")
    audit.check(_active_ids(turns[3]) == whiser_ids, "turn 4: L1 was not switched to WhiSER")
    audit.check(
        aspire_ids in _recent_thread_ids(turns[3]),
        "turn 4: ASPIRE was not retained as a resumable document thread",
    )
    audit.check(_retrieval_evidence_is_valid(turns[3]), "turn 4: retrieval evidence invalid")
    audit.check(_answer_evidence_is_valid(turns[3]), "turn 4: answer evidence invalid")

    _assert_sources_are_only(audit, turns[4], filename="ASPIRE.pdf", document_ids=aspire_ids)
    audit.check(_focus_ids(turns[4]) == aspire_ids, "turn 5: previous-paper focus is not ASPIRE")
    audit.check(_active_ids(turns[4]) == aspire_ids, "turn 5: L1 did not restore ASPIRE")
    audit.check(
        whiser_ids in _recent_thread_ids(turns[4]),
        "turn 5: WhiSER was not retained as a resumable document thread",
    )
    audit.check(_retrieval_evidence_is_valid(turns[4]), "turn 5: retrieval evidence invalid")
    audit.check(_answer_evidence_is_valid(turns[4]), "turn 5: answer evidence invalid")

    _assert_sources_are_only(audit, turns[5], filename="ASPIRE.pdf", document_ids=aspire_ids)
    audit.check(_focus_ids(turns[5]) == aspire_ids, "turn 6: figure query escaped ASPIRE focus")
    figures = _figure_sources(turns[5])
    audit.check(bool(figures), "turn 6: no figure source")
    audit.check(
        len(figures) == 1,
        f"turn 6: best-figure request returned {len(figures)} figures instead of exactly one",
    )
    audit.check(
        all(source.get("quality_status") == "accepted" for source in figures),
        "turn 6: a non-accepted visual leaked",
    )
    audit.check(
        all(source.get("is_content") is True for source in figures),
        "turn 6: a non-content visual leaked",
    )
    audit.check(
        all(source.get("is_complete") is True for source in figures),
        "turn 6: an incomplete visual leaked",
    )
    audit.check(
        all(_fold_text(source.get("asset_kind")) not in FORBIDDEN_VISUAL_KINDS for source in figures),
        "turn 6: a logo/branding/page fallback visual leaked",
    )
    audit.check(
        all(source.get("image_url") for source in figures),
        "turn 6: a figure has no image_url",
    )
    audit.check(
        bool(figures) and _looks_like_architecture_figure(figures[0]),
        "turn 6: top visual is not semantically an architecture figure",
    )
    audit.check(_retrieval_evidence_is_valid(turns[5]), "turn 6: retrieval evidence invalid")
    audit.check(_answer_evidence_is_valid(turns[5]), "turn 6: answer evidence invalid")

    _assert_sources_are_only(audit, turns[6], filename="ASPIRE.pdf", document_ids=aspire_ids)
    audit.check(_focus_ids(turns[6]) == aspire_ids, "turn 7: table query escaped ASPIRE focus")
    tables = _table_sources(turns[6])
    audit.check(bool(tables), "turn 7: no table source")
    audit.check(
        any(_table_contains_requested_metrics(source) for source in tables),
        "turn 7: no retrieved table contains Acc, F1 and CCC evidence together",
    )
    audit.check(_retrieval_evidence_is_valid(turns[6]), "turn 7: retrieval evidence invalid")
    audit.check(_answer_evidence_is_valid(turns[6]), "turn 7: answer evidence invalid")
    _assert_metric_values_were_checked(
        audit,
        turns[6],
        required_metrics=REQUIRED_BENCHMARK_METRICS,
    )
    return aspire_ids, whiser_ids


def _validate_history(
    history: object,
    turns: list[Turn],
    audit: Audit,
) -> list[dict[str, Any]]:
    if not isinstance(history, list):
        audit.check(False, "conversation history is not a list")
        return []
    messages = [item for item in history if isinstance(item, dict)]
    audit.check(len(messages) == len(QUESTIONS) * 2, f"history has {len(messages)} messages, expected 14")
    users = [message for message in messages if message.get("role") == "user"]
    assistants = [message for message in messages if message.get("role") == "assistant"]
    audit.check(len(users) == len(QUESTIONS), f"history has {len(users)} user messages, expected 7")
    audit.check(
        len(assistants) == len(QUESTIONS),
        f"history has {len(assistants)} assistant messages, expected 7",
    )

    for index, question in enumerate(QUESTIONS):
        if index < len(users):
            audit.check(users[index].get("content") == question, f"history user turn {index + 1} differs")
        if index >= len(assistants) or index >= len(turns):
            continue
        persisted = assistants[index]
        persisted_sources = [
            source
            for source in persisted.get("sources") or []
            if isinstance(source, dict)
        ]
        audit.check(
            persisted.get("content") == turns[index].answer,
            f"history assistant turn {index + 1} differs from streamed answer",
        )
        audit.check(
            persisted.get("model") == MODEL,
            f"history assistant turn {index + 1} did not persist model {MODEL}",
        )
        if index == 1:
            audit.check(not persisted_sources, "history casual turn unexpectedly persisted sources")
        else:
            expected_ids = Counter(_source_identity(source) for source in turns[index].sources)
            persisted_ids = Counter(_source_identity(source) for source in persisted_sources)
            audit.check(
                expected_ids == persisted_ids,
                (
                    f"history source identities differ at turn {index + 1}; "
                    f"missing={sorted((expected_ids - persisted_ids).elements())}, "
                    f"extra={sorted((persisted_ids - expected_ids).elements())}"
                ),
            )

    if len(assistants) >= 7:
        persisted_figures = _figure_sources_from(
            [source for source in assistants[5].get("sources") or [] if isinstance(source, dict)]
        )
        persisted_tables = _table_sources_from(
            [source for source in assistants[6].get("sources") or [] if isinstance(source, dict)]
        )
        audit.check(
            len(persisted_figures) == 1,
            f"history persisted {len(persisted_figures)} figures instead of exactly one",
        )
        audit.check(bool(persisted_tables), "history did not persist the table source")
        audit.check(
            all(source.get("image_url") for source in persisted_figures),
            "history figure attachment lost image_url",
        )
        audit.check(
            all(source.get("figure_type") for source in persisted_figures),
            "history figure attachment lost figure_type",
        )
        audit.check(
            any(
                source.get("table_id") and source.get("table_index") is not None
                for source in persisted_tables
            ),
            "history table source lost table_id/table_index",
        )
    return messages


def _validate_run_records(
    base_url: str,
    turns: list[Turn],
    audit: Audit,
    *,
    timeout: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for turn in turns:
        if not turn.run_id:
            audit.check(False, f"turn {turn.index}: missing run_id")
            continue
        payload = _request_json(base_url, f"/agent/runs/{turn.run_id}", timeout=timeout)
        if not isinstance(payload, dict):
            audit.check(False, f"turn {turn.index}: run detail is not an object")
            continue
        records.append(payload)
        audit.check(payload.get("status") == "completed", f"turn {turn.index}: run status={payload.get('status')}")
        audit.check(
            payload.get("conversation_id") == turn.conversation_id,
            f"turn {turn.index}: persisted run conversation differs",
        )
        audit.check(
            payload.get("final_answer") == turn.answer,
            f"turn {turn.index}: persisted final_answer differs from stream",
        )
    return records


def _turn_report(turn: Turn) -> dict[str, Any]:
    route = turn.first("agent.route.decided")
    rewrite = turn.first("query.rewritten")
    figures = _figure_sources(turn)
    tables = _table_sources(turn)
    failed = turn.first("run.failed")
    checked_claims = turn.last("answer.evidence.validated").get("checked_claims") or []
    checked_metrics = sorted(
        {
            str(claim.get("metric") or "").strip().casefold()
            for claim in checked_claims
            if isinstance(claim, dict) and claim.get("metric")
        }
    )
    return {
        "turn": turn.index,
        "seconds": turn.elapsed_seconds,
        "run_id": turn.run_id,
        "route": route.get("route"),
        "route_reason": route.get("reason"),
        "rewrite_reason": (rewrite.get("diagnostics") or {}).get("reason"),
        "focus_document_ids": sorted(_focus_ids(turn)),
        "active_document_ids": sorted(_active_ids(turn)),
        "filenames": sorted(_filenames(turn)),
        "source_count": len(turn.sources),
        "figure_ids": [source.get("figure_id") for source in figures],
        "figure_labels": [source.get("figure_label") for source in figures],
        "table_source_count": len(tables),
        "retrieval_evidence_valid": (
            _retrieval_evidence_is_valid(turn) if turn.payloads("retrieval.completed") else None
        ),
        "answer_evidence_valid": (
            _answer_evidence_is_valid(turn) if turn.payloads("answer.evidence.validated") else None
        ),
        "answer_checked_claim_count": len(checked_claims),
        "answer_checked_metrics": checked_metrics,
        "answer_chars": len(turn.answer),
        "answer_preview": turn.answer[:180].replace("\n", " "),
        "error": failed.get("error"),
    }


def _write_report(report: dict[str, Any], output_path: str | None) -> None:
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if output_path:
        # This is an audit output requested explicitly through the CLI, not a
        # repository mutation performed by normal test execution.
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit the seven-turn ASPIRE/WhiSER agent conversation over HTTP/SSE."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:7777")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", help="Optionally write the JSON report to this path")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    audit = Audit()
    turns: list[Turn] = []
    conversation_id: str | None = None
    health_summary: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []
    image_checks: list[dict[str, Any]] = []

    try:
        health_summary = _validate_preflight(
            _request_json(base_url, "/health", timeout=min(args.timeout, 15.0)),
            audit,
        )
        if audit.errors:
            raise E2ETransportError("preflight failed; agent turns were not started")

        for index, question in enumerate(QUESTIONS, start=1):
            turn = _run_turn(
                base_url,
                index=index,
                question=question,
                conversation_id=conversation_id,
                timeout=args.timeout,
            )
            turns.append(turn)
            started_conversation_id = turn.conversation_id
            if conversation_id is not None:
                audit.check(
                    started_conversation_id == conversation_id,
                    f"turn {index}: backend changed conversation_id",
                )
            conversation_id = started_conversation_id or conversation_id
            if turn.first("run.failed") or not turn.first("run.completed"):
                break

        if len(turns) != len(QUESTIONS):
            audit.check(False, f"only {len(turns)}/{len(QUESTIONS)} turns completed")
        else:
            _validate_turns(turns, audit)

        if conversation_id:
            history_payload = _request_json(
                base_url,
                f"/chat/conversations/{conversation_id}/messages",
                timeout=args.timeout,
            )
            history = _validate_history(history_payload, turns, audit)
            run_records = _validate_run_records(
                base_url,
                turns,
                audit,
                timeout=args.timeout,
            )

        if len(turns) >= 6:
            seen_urls: set[str] = set()
            for source in _figure_sources(turns[5]):
                image_url = str(source.get("image_url") or "")
                if not image_url or image_url in seen_urls:
                    continue
                seen_urls.add(image_url)
                check = _request_image(base_url, image_url, timeout=args.timeout)
                image_checks.append(check)
                audit.check(check["status"] == 200, f"figure image {image_url} did not return 200")
                audit.check(
                    str(check["content_type"]).startswith("image/"),
                    f"figure image {image_url} returned {check['content_type']}",
                )
                audit.check(check["has_bytes"] is True, f"figure image {image_url} was empty")
    except E2ETransportError as exc:
        audit.errors.append(str(exc))
    except (TimeoutError, OSError) as exc:
        audit.errors.append(f"transport failure: {exc}")

    report = {
        "ok": not audit.errors,
        "base_url": base_url,
        "model": MODEL,
        "health": health_summary,
        "conversation_id": conversation_id,
        "turns": [_turn_report(turn) for turn in turns],
        "history_messages": len(history),
        "run_statuses": [record.get("status") for record in run_records],
        "image_checks": image_checks,
        "errors": audit.errors,
    }
    _write_report(report, args.output)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
