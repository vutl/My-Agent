#!/usr/bin/env python3
"""Deterministic held-out conversational evaluator over Aya's public SSE API."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, Iterable, Iterator
import urllib.error
import urllib.parse
import urllib.request


# None: the backend uses the product default (DEFAULT_MODEL in backend/.env).
DEFAULT_MODEL = None
DEFAULT_ALLOWED_TOOLS = ("search_local_docs", "retrieve_visual_assets")
_SCORE_TOKEN_RE = re.compile(r"[A-Za-zÀ-ỹĐđ0-9]+", re.UNICODE)


@dataclass(frozen=True)
class Event:
    event: str
    data: dict[str, Any]


def parse_sse_lines(lines: Iterable[bytes | str]) -> Iterator[Event]:
    event_name = "message"
    data_lines: list[str] = []

    def emit() -> Event | None:
        nonlocal event_name, data_lines
        if not data_lines:
            event_name = "message"
            return None
        payload = json.loads("\n".join(data_lines))
        if not isinstance(payload, dict):
            raise RuntimeError(f"SSE event {event_name!r} returned non-object JSON")
        result = Event(event_name, payload)
        event_name = "message"
        data_lines = []
        return result

    for raw in lines:
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        line = line.rstrip("\r\n")
        if not line:
            event = emit()
            if event:
                yield event
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_name = value or "message"
        elif field == "data":
            data_lines.append(value)
    event = emit()
    if event:
        yield event


def _url(base_url: str, path: str) -> str:
    return urllib.parse.urljoin(f"{base_url.rstrip('/')}/", path.lstrip("/"))


def _run_turn(
    *,
    base_url: str,
    case: dict[str, Any],
    conversation_id: str | None,
    model: str,
    timeout: float,
    debug_trace: bool = False,
    on_started: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    payload = {
        "conversation_id": conversation_id,
        "task": case["message"],
        # Routing and reference-context generation are different evaluation
        # contracts.  Public suites may pin chat mode so an inline benchmark
        # passage cannot accidentally trigger Aya's private-document router.
        "mode": case.get("mode", "auto"),
        "model": model,
        "temperature": 0.2,
        # This evaluator deliberately measures Aya as deployed today.  It must
        # not award itself web/shell/MCP capabilities that the agent does not
        # expose yet.
        "allowed_tools": list(DEFAULT_ALLOWED_TOOLS),
        "require_confirmation": True,
        "collection_id": case.get("collection_id"),
        "retrieval_mode": "auto",
        "agent_reasoning": "smart",
        "debug_trace": debug_trace,
    }
    request = urllib.request.Request(
        _url(base_url, "/agent/run/stream"),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        events: list[Event] = []
        first_delta_ms: float | None = None
        for event in parse_sse_lines(response):
            events.append(event)
            if event.event == "run.started" and on_started is not None:
                on_started(event.data)
            if event.event == "message.delta" and first_delta_ms is None:
                first_delta_ms = round((time.perf_counter() - started) * 1000, 2)
    # "message.reset" discards an opening the server replaced after validation.
    answer_parts: list[str] = []
    for event in events:
        if event.event == "message.reset":
            answer_parts.clear()
        elif event.event == "message.delta":
            answer_parts.append(str(event.data.get("delta") or ""))
    answer = "".join(answer_parts)
    return {
        "events": events,
        "answer": answer,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "client_first_delta_ms": first_delta_ms,
    }


def _first(events: list[Event], name: str) -> dict[str, Any]:
    return next((event.data for event in events if event.event == name), {})


def _all(events: list[Event], name: str) -> list[dict[str, Any]]:
    return [event.data for event in events if event.event == name]


def _ordered_unique(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value))


def _has_abstention(answer: str) -> bool:
    normalized = " ".join(
        answer.lower()
        .replace("’", "'")
        .replace("‘", "'")
        .replace("`", "'")
        .split()
    )
    # Presentation markup must not change the semantic verdict. In particular,
    # ``không có **Figure 99**`` is the same direct non-existence statement as
    # its plain-text form.
    normalized = re.sub(r"[*_~]+", "", normalized)
    markers = (
        "chưa có đủ",
        "chưa đủ bằng chứng",
        "không có đủ",
        "không tìm thấy",
        "not enough evidence",
        "isn't enough evidence",
        "is not enough evidence",
        "insufficient information",
        "references are insufficient",
        "references do not provide",
        "do not have any information",
        "don't have any information",
        "no reference passage",
        "could not find",
    )
    if any(marker in normalized for marker in markers):
        return True
    return bool(
        # Direct non-existence is a valid abstention too.  This is common for
        # exact-artifact requests (``the paper has no Table 99`` / ``không có
        # Figure 99``) and is stronger than a generic uncertainty phrase.
        # Require a document/source subject and an evidence/artifact object so
        # ordinary negative factual answers are not scored as abstentions.
        re.search(
            r"\b(?:papers?|documents?|sources?|references?|excerpts?|"
            r"tables?|figures?|[a-z0-9_.-]+\.pdf)\b.{0,80}\b"
            r"(?:do|does|did|has|have|is|are)\s+(?:not|no)\b.{0,64}\b"
            r"(?:tables?|figures?|evidence|information|results?|details?)\b",
            normalized,
        )
        or re.search(
            r"\b(?:papers?|documents?|sources?|references?|excerpts?|"
            r"tables?|figures?|[a-z0-9_.-]+\.pdf)\b.{0,80}\b"
            r"(?:contains?|includes?|provides?|reports?|shows?)\s+no\b.{0,64}\b"
            r"(?:tables?|figures?|evidence|information|results?|details?)\b",
            normalized,
        )
        or re.search(
            r"\b(?:bài|paper|tài\s+liệu|nguồn|trích\s+đoạn|"
            r"[a-z0-9_.-]+\.pdf)\b.{0,80}\bkhông\s+có\b.{0,64}\b"
            r"(?:bảng|hình|figure|table|bằng\s+chứng|thông\s+tin|kết\s+quả|chi\s+tiết)\b",
            normalized,
        )
        or re.search(
            r"(?:^|\b)[a-z0-9_.-]{2,}(?:\s+(?:paper|bài))?\s+"
            r"không\s+có\s+(?:bảng|hình|figure|table)\b",
            normalized,
        )
        or re.search(
            r"(?:^|\b)[a-z0-9_.-]{2,}(?:\s+paper)?\s+"
            r"(?:does|do|has|have)\s+not\s+(?:have|contain|include)?\s*"
            r"(?:a\s+|an\s+|the\s+)?(?:figure|table)\b",
            normalized,
        )
        or
        re.search(
            r"\b(?:i\s+)?(?:can(?:not|'t)|could(?: not|n't)|unable\s+to)\s+"
            r"(?:identify|find|confirm|verify|locate|determine)\b",
            normalized,
        )
        or re.search(
            r"\b(?:was|were|is|are)\s+not\s+"
            r"(?:provided|reported|stated|specified|mentioned|available|found)\b",
            normalized,
        )
        or re.search(
            r"\b(?:sources?|references?|papers?|excerpts?|tables?|figures?|"
            r"provided\s+(?:sources?|references?|excerpts?))\b.{0,96}\b"
            r"(?:do|does|did)\s+not\s+"
            r"(?:provide|report|state|specify|mention|contain|include|show)\b",
            normalized,
        )
        or re.search(
            r"\bneither\b.{0,96}\b(?:source|reference|paper|excerpt)\b.{0,64}\b"
            r"(?:provides?|reports?|states?|specifies|mentions?|contains?|includes?|shows?)\b",
            normalized,
        )
        or re.search(r"\bno\s+meaningful\s+comparison\s+(?:is|was)\s+possible\b", normalized)
        or re.search(
            r"\b(?:không|chưa)\s+(?:được\s+)?"
            r"(?:nêu|cung\s+cấp|đề\s+cập|báo\s+cáo|xác\s+nhận|tìm\s+thấy)\b",
            normalized,
        )
        or re.search(r"\b(?:không|chưa)\s+thể\s+xác\s+nhận\b", normalized)
    )


def _expected_tools(case: dict[str, Any], expected_routes: list[str]) -> list[str]:
    configured = case.get("expected_tools")
    if configured is not None:
        return _ordered_unique(configured)
    if expected_routes and set(expected_routes) == {"chat"}:
        return []
    expected = ["search_local_docs"]
    artifacts = case.get("expected_artifacts") or {}
    if artifacts.get("figure_ids"):
        expected.append("retrieve_visual_assets")
    return expected


def _allowed_tools(
    case: dict[str, Any],
    expected_routes: list[str],
    required_tools: list[str],
) -> list[str]:
    configured = case.get("allowed_tools")
    if configured is not None:
        return _ordered_unique(configured)
    allowed = list(required_tools)
    if case.get("expected_tools_exact"):
        return allowed
    message = " ".join(str(case.get("message") or "").casefold().split())
    visual_ask = bool(
        re.search(r"(?<!\w)(?:figure|fig\.?|hình|hinh|visual|diagram)(?!\w)", message)
    )
    if expected_routes and set(expected_routes) != {"chat"} and visual_ask:
        allowed.append("retrieve_visual_assets")
    return _ordered_unique(allowed)


def _tool_contract_failure(
    actual: list[str],
    *,
    required: list[str],
    allowed: list[str],
) -> bool:
    actual_set = set(actual)
    return bool(set(required) - actual_set or actual_set - set(allowed))


def _token_f1(prediction: str, reference: str) -> float:
    from collections import Counter

    prediction_tokens = _SCORE_TOKEN_RE.findall(prediction.lower())
    reference_tokens = _SCORE_TOKEN_RE.findall(reference.lower())
    if not prediction_tokens or not reference_tokens:
        return 0.0
    overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def _evaluate_case(
    case: dict[str, Any],
    result: dict[str, Any],
    *,
    model: str,
    include_answer: bool = False,
) -> dict[str, Any]:
    events: list[Event] = result["events"]
    failures: list[str] = []
    started = _first(events, "run.started")
    route = _first(events, "agent.route.decided")
    rewrite = _first(events, "query.rewritten")
    retrieval = _first(events, "retrieval.completed")
    expected_ids = list(case.get("expected_document_ids") or [])
    focus_ids = _ordered_unique(rewrite.get("focus_document_ids") or [])
    retrieved_ids = _ordered_unique(
        source.get("document_id") for source in retrieval.get("documents") or []
    )
    actual_route = str(route.get("route") or "")
    configured_route = case.get("expected_route")
    expected_routes = (
        [str(value) for value in configured_route]
        if isinstance(configured_route, list)
        else [str(configured_route)] if configured_route else []
    )
    expected_tools = _expected_tools(case, expected_routes)
    allowed_tools = _allowed_tools(case, expected_routes, expected_tools)
    selected_tools = _ordered_unique(route.get("selected_tools") or [])
    started_tools = _ordered_unique(
        item.get("tool_name") for item in _all(events, "tool.started")
    )
    completed_tools = _ordered_unique(
        item.get("tool_name") for item in _all(events, "tool.completed")
    )
    unsupported_tools = sorted(
        set(selected_tools + started_tools + completed_tools).difference(
            DEFAULT_ALLOWED_TOOLS
        )
    )

    if _first(events, "run.failed"):
        failures.append(f"run_failed:{_first(events, 'run.failed').get('error')}")
    if not _first(events, "run.completed"):
        failures.append("run_not_completed")
    if str(started.get("model") or "") != model:
        failures.append(f"model_mismatch:{started.get('model')}")
    if expected_routes and actual_route not in expected_routes:
        failures.append(f"route:{actual_route}!in{expected_routes}")
    if _tool_contract_failure(
        selected_tools,
        required=expected_tools,
        allowed=allowed_tools,
    ):
        failures.append(
            f"selected_tools:{selected_tools}!required={expected_tools},allowed={allowed_tools}"
        )
    if _tool_contract_failure(
        started_tools,
        required=expected_tools,
        allowed=allowed_tools,
    ):
        failures.append(
            f"started_tools:{started_tools}!required={expected_tools},allowed={allowed_tools}"
        )
    if _tool_contract_failure(
        completed_tools,
        required=expected_tools,
        allowed=allowed_tools,
    ):
        failures.append(
            f"completed_tools:{completed_tools}!required={expected_tools},allowed={allowed_tools}"
        )
    if unsupported_tools:
        failures.append(f"unsupported_tools:{unsupported_tools}")
    fallback_events = _all(events, "tool.fallback.started")
    if fallback_events:
        failures.append(
            "tool_fallback:" + ",".join(
                _ordered_unique(item.get("tool_name") for item in fallback_events)
            )
        )
    if expected_ids and focus_ids != expected_ids:
        failures.append(f"focus:{focus_ids}!={expected_ids}")
    if expected_ids and any(value not in retrieved_ids for value in expected_ids):
        failures.append(f"retrieval_missing:{[value for value in expected_ids if value not in retrieved_ids]}")
    if actual_route == "chat" and retrieved_ids:
        failures.append(f"chat_retrieved_documents:{retrieved_ids}")
    forbidden = set(case.get("forbidden_document_ids") or [])
    leaked = forbidden.intersection(retrieved_ids)
    if leaked:
        failures.append(f"forbidden_documents:{sorted(leaked)}")

    expected_artifacts = case.get("expected_artifacts") or {}
    actual_tables = {
        str(source.get("table_id"))
        for source in retrieval.get("documents") or []
        if source.get("table_id")
    }
    actual_figures = {
        str(source.get("figure_id"))
        for source in retrieval.get("documents") or []
        if source.get("figure_id")
    }
    for table_id in expected_artifacts.get("table_ids") or []:
        if table_id not in actual_tables:
            failures.append(f"missing_table:{table_id}")
    for figure_id in expected_artifacts.get("figure_ids") or []:
        if figure_id not in actual_figures:
            failures.append(f"missing_figure:{figure_id}")

    card_coverage = _first(events, "evidence.card.coverage")
    expected_facets = case.get("expected_facets") or {}
    if expected_facets and card_coverage:
        by_document = {
            str(item.get("document_id")): item
            for item in card_coverage.get("documents") or []
        }
        for document_id, facets in expected_facets.items():
            observed = set((by_document.get(document_id) or {}).get("requested_facets") or [])
            missing = set(facets).difference(observed)
            if missing:
                failures.append(f"facet_contract:{document_id}:{sorted(missing)}")

    validation_events = _all(events, "answer.evidence.validated")
    if expected_ids and not validation_events:
        failures.append("answer_validation_missing")
    elif (
        expected_ids
        and not case.get("expected_abstention")
        and not bool(validation_events[-1].get("valid"))
    ):
        failures.append(f"answer_validation:{validation_events[-1].get('reason')}")

    retrieval_validation = retrieval.get("evidence_validation") or {}
    if expected_ids and not retrieval:
        failures.append("retrieval_event_missing")
    elif expected_ids and not case.get("expected_abstention") and not bool(
        retrieval_validation.get("valid")
    ):
        failures.append(
            f"retrieval_validation:{retrieval_validation.get('reason') or 'missing'}"
        )

    answer_lower = str(result["answer"]).lower()
    if case.get("expected_abstention"):
        if not _has_abstention(answer_lower):
            failures.append("expected_abstention_missing")
    for required in case.get("answer_must_contain") or []:
        if str(required).lower() not in answer_lower:
            failures.append(f"answer_missing:{required}")
    for forbidden_text in case.get("answer_must_not_contain") or []:
        if str(forbidden_text).lower() in answer_lower:
            failures.append(f"answer_forbidden:{forbidden_text}")
    if not str(result["answer"]).strip():
        failures.append("empty_answer")
    stream_chunks = len(_all(events, "message.delta"))
    if result["answer"] and stream_chunks == 0:
        failures.append("answer_not_streamed")
    if result["answer"] and not _first(events, "message.finished"):
        failures.append("message_not_finished")
    references = [str(item) for item in case.get("reference_answers") or [] if item]
    best_token_f1 = max(
        (_token_f1(result["answer"], reference) for reference in references),
        default=None,
    )
    minimum_token_f1 = case.get("minimum_token_f1")
    if minimum_token_f1 is not None and (
        best_token_f1 is None or best_token_f1 < float(minimum_token_f1)
    ):
        failures.append(
            f"token_f1:{0.0 if best_token_f1 is None else best_token_f1:.4f}"
            f"<{float(minimum_token_f1):.4f}"
        )

    timing = {
        str(item.get("stage")): item.get("ms")
        for item in _all(events, "timing")
        if item.get("stage")
    }
    evaluated = {
        "run_id": str(started.get("run_id") or ""),
        "case_id": case["case_id"],
        "conversation_group": case["conversation_group"],
        "turn_index": case["turn_index"],
        "ok": not failures,
        "failures": failures,
        "route": actual_route,
        "expected_tools": expected_tools,
        "allowed_tools": allowed_tools,
        "selected_tools": selected_tools,
        "started_tools": started_tools,
        "completed_tools": completed_tools,
        "focus_document_ids": focus_ids,
        "retrieved_document_ids": retrieved_ids,
        "table_ids": sorted(actual_tables),
        "figure_ids": sorted(actual_figures),
        "answer_chars": len(result["answer"]),
        "best_reference_token_f1": (
            round(best_token_f1, 6) if best_token_f1 is not None else None
        ),
        "elapsed_ms": result["elapsed_ms"],
        "client_first_delta_ms": result.get("client_first_delta_ms"),
        "timing": timing,
        "stream_chunks": stream_chunks,
        "retrieval_mode": retrieval.get("retrieval_mode"),
        "retrieval_retry_performed": bool(retrieval.get("retry_performed")),
        "retrieval_validation": retrieval_validation,
        "answer_validation": validation_events[-1] if validation_events else None,
        "evidence_card_coverage_observed": bool(card_coverage),
        "paper_sections_validated": len(_all(events, "answer.paper.validated")),
    }
    if include_answer:
        evaluated["answer"] = result["answer"]
    return evaluated


def _load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    seen_turns: set[tuple[str, int]] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path}:{line_number} contains invalid JSON: {exc.msg}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")

        required_fields = (
            "case_id",
            "conversation_group",
            "turn_index",
            "message",
            "expected_document_ids",
        )
        missing = [field for field in required_fields if field not in value]
        if missing:
            raise ValueError(
                f"{path}:{line_number} uses an unsupported conversation-eval schema; "
                f"missing required field(s): {', '.join(missing)}"
            )

        for field in ("case_id", "conversation_group", "message"):
            if not isinstance(value[field], str) or not value[field].strip():
                raise ValueError(
                    f"{path}:{line_number} field {field!r} must be a non-empty string"
                )
        if isinstance(value["turn_index"], bool):
            raise ValueError(
                f"{path}:{line_number} field 'turn_index' must be a positive integer"
            )
        try:
            turn_index = int(value["turn_index"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}:{line_number} field 'turn_index' must be a positive integer"
            ) from exc
        if turn_index < 1:
            raise ValueError(
                f"{path}:{line_number} field 'turn_index' must be a positive integer"
            )
        value["turn_index"] = turn_index

        list_fields = (
            "expected_document_ids",
            "forbidden_document_ids",
            "answer_must_contain",
            "answer_must_not_contain",
            "acceptable_evidence_groups",
            "expected_tools",
        )
        for field in list_fields:
            if field in value and not isinstance(value[field], list):
                raise ValueError(
                    f"{path}:{line_number} field {field!r} must be a JSON array"
                )
        for field in ("expected_facets", "expected_artifacts"):
            if field in value and not isinstance(value[field], dict):
                raise ValueError(
                    f"{path}:{line_number} field {field!r} must be a JSON object"
                )
        expected_route = value.get("expected_route")
        if expected_route is not None and not isinstance(expected_route, (str, list)):
            raise ValueError(
                f"{path}:{line_number} field 'expected_route' must be a string or array"
            )
        mode = value.get("mode", "auto")
        if mode not in {"auto", "chat", "file_qa", "research"}:
            raise ValueError(
                f"{path}:{line_number} field 'mode' must be one of "
                "auto, chat, file_qa, research"
            )

        case_id = value["case_id"].strip()
        turn_key = (value["conversation_group"].strip(), turn_index)
        if case_id in seen_case_ids:
            raise ValueError(f"{path}:{line_number} duplicates case_id {case_id!r}")
        if turn_key in seen_turns:
            raise ValueError(
                f"{path}:{line_number} duplicates conversation turn {turn_key!r}"
            )
        seen_case_ids.add(case_id)
        seen_turns.add(turn_key)
        cases.append(value)
    return cases


def _write_report(report: dict[str, Any], output: str | None) -> None:
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if output:
        from evaluation_checkpoint import write_checkpoint
        write_checkpoint(Path(output), report)


def _select_with_context(cases: list[dict[str, Any]], ids: set[str]) -> list[dict[str, Any]]:
    """Selecting a follow-up includes earlier turns, never an invented history."""
    missing = ids - {case["case_id"] for case in cases}
    if missing:
        raise ValueError("Unknown --case-id value(s): " + ", ".join(sorted(missing)))
    last_turn: dict[str, int] = {}
    for case in cases:
        if case["case_id"] in ids:
            group = str(case["conversation_group"])
            last_turn[group] = max(last_turn.get(group, 0), int(case["turn_index"]))
    return [case for case in cases if str(case["conversation_group"]) in last_turn
            and int(case["turn_index"]) <= last_turn[str(case["conversation_group"])]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="data/retrieval_eval/conversational-dev-v1.jsonl",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:7777")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--case-id",
        action="append",
        help="Run only the named case_id; repeat to select multiple cases.",
    )
    parser.add_argument("--output")
    parser.add_argument("--runtime-root", type=Path,
                        help="Isolated backend data root; enables verified per-turn checkpoints.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume only an identical checkpoint with intact persisted history.")
    parser.add_argument(
        "--include-answers",
        action="store_true",
        help="Include full generated answers in the local report for manual quality audit.",
    )
    parser.add_argument(
        "--debug-trace",
        action="store_true",
        help="Request a local raw prompt/draft trace (server gate must also be enabled).",
    )
    parser.add_argument(
        "--allow-heldout",
        action="store_true",
        help="Required when the dataset filename contains 'heldout'.",
    )
    args = parser.parse_args()
    if args.resume and (not args.output or args.runtime_root is None):
        parser.error("--resume requires --output and --runtime-root")
    if args.runtime_root is not None and not args.output:
        parser.error("--runtime-root requires --output for durable checkpoints")
    path = Path(args.dataset)
    if "heldout" in path.name.lower() and not args.allow_heldout:
        parser.error("Refusing to open frozen held-out labels without --allow-heldout")
    cases = _load_cases(path)
    if args.case_id:
        try:
            cases = _select_with_context(cases, set(args.case_id))
        except ValueError as exc:
            parser.error(str(exc))
    if args.max_cases is not None:
        cases = cases[: max(0, args.max_cases)]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[str(case["conversation_group"])].append(case)
    cases = [case for group in grouped.values()
             for case in sorted(group, key=lambda item: int(item["turn_index"]))]

    checkpoint = None
    if args.runtime_root is not None:
        from agent_evaluation_checkpoint import (
            AgentCheckpoint, digest, evaluator_fingerprint, runtime_fingerprint,
            verify_runtime_binding,
        )

        def get_json(api_path: str) -> Any:
            with urllib.request.urlopen(_url(args.base_url, api_path), timeout=args.timeout) as response:
                return json.load(response)

        project = Path(__file__).resolve().parents[1]
        verify_runtime_binding(args.runtime_root, project, get_json)
        contract = {
            "code": evaluator_fingerprint(project),
            "index": runtime_fingerprint(args.runtime_root),
            "runtime_root": str(args.runtime_root.resolve()),
            "cases": digest(cases), "model": args.model,
            "base_url": args.base_url.rstrip('/'), "timeout": args.timeout,
            "include_answers": args.include_answers, "debug_trace": args.debug_trace,
        }
        checkpoint = AgentCheckpoint(Path(args.output + ".partial"),
                                     contract=contract, cases=cases, resume=args.resume,
                                     get_json=get_json)
        checkpoint.save()

    results: list[dict[str, Any]] = checkpoint.rows if checkpoint else []
    transport_error: str | None = None
    try:
        for group, group_cases in grouped.items():
            conversation_id: str | None = (
                checkpoint.histories.get(group, {}).get("conversation_id") if checkpoint else None)
            for case in sorted(group_cases, key=lambda item: int(item["turn_index"])):
                if any(row["case_id"] == case["case_id"] for row in results):
                    continue
                if checkpoint:
                    checkpoint.begin(case, conversation_id)
                turn = _run_turn(
                    base_url=args.base_url,
                    case=case,
                    conversation_id=conversation_id,
                    model=args.model,
                    timeout=args.timeout,
                    debug_trace=args.debug_trace,
                    on_started=checkpoint.started if checkpoint else None,
                )
                started = _first(turn["events"], "run.started")
                actual_conversation = str(started.get("conversation_id") or "") or None
                if conversation_id and actual_conversation != conversation_id:
                    turn["events"].append(
                        Event("run.failed", {"error": "conversation_id_changed"})
                    )
                conversation_id = actual_conversation or conversation_id
                evaluated = _evaluate_case(
                        case,
                        turn,
                        model=args.model,
                        include_answer=args.include_answers,
                    )
                completed = bool(_first(turn["events"], "run.completed"))
                if checkpoint:
                    checkpoint.finish(evaluated, conversation_id=conversation_id,
                                      answer=turn["answer"], completed=completed)
                else:
                    results.append(evaluated)
                print(f"[{len(results)}/{len(cases)}] {case['case_id']}: "
                      f"{'pass' if evaluated['ok'] else evaluated['failures']}",
                      file=sys.stderr, flush=True)
                if not completed:
                    raise RuntimeError("Agent run failed; stopping without model fallback")
        if checkpoint:
            if runtime_fingerprint(args.runtime_root) != checkpoint.contract["index"]:
                raise RuntimeError("Canonical projection changed during agent evaluation")
            if evaluator_fingerprint(project) != checkpoint.contract["code"]:
                raise RuntimeError("Evaluator/backend code changed during agent evaluation")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, RuntimeError) as exc:
        transport_error = str(exc)
        if checkpoint:
            checkpoint.save(error=transport_error)

    failures = [item for item in results if not item["ok"]]
    elapsed = [float(item["elapsed_ms"]) for item in results]
    token_f1_values = [
        float(item["best_reference_token_f1"])
        for item in results
        if item.get("best_reference_token_f1") is not None
    ]
    first_token_values = [
        float(item["timing"]["first_token"])
        for item in results
        if item.get("timing", {}).get("first_token") is not None
    ]
    client_first_delta_values = [
        float(item["client_first_delta_ms"])
        for item in results
        if item.get("client_first_delta_ms") is not None
    ]

    def distribution(values: list[float]) -> dict[str, float | int | None]:
        ordered = sorted(values)
        return {
            "count": len(ordered),
            "min": ordered[0] if ordered else None,
            "median": ordered[len(ordered) // 2] if ordered else None,
            "mean": round(sum(ordered) / len(ordered), 6) if ordered else None,
            "max": ordered[-1] if ordered else None,
        }

    report = {
        "ok": transport_error is None and len(results) == len(cases) and not failures,
        "dataset": str(path),
        "model": args.model,
        "cases_requested": len(cases),
        "cases_completed": len(results),
        "passed": len(results) - len(failures),
        "failed": len(failures),
        "latency_ms": {
            "min": min(elapsed) if elapsed else None,
            "median": sorted(elapsed)[len(elapsed) // 2] if elapsed else None,
            "max": max(elapsed) if elapsed else None,
        },
        "first_token_ms": distribution(first_token_values),
        "client_first_delta_ms": distribution(client_first_delta_values),
        "reference_token_f1": {
            **distribution(token_f1_values),
            "diagnostic_only": True,
        },
        "routes": {
            route: sum(item["route"] == route for item in results)
            for route in sorted({item["route"] for item in results})
        },
        "retrieval_free_cases": sum(
            not item["retrieved_document_ids"] for item in results
        ),
        "tool_inventory": {
            "allowed": list(DEFAULT_ALLOWED_TOOLS),
            "selected_counts": {
                tool: sum(tool in item["selected_tools"] for item in results)
                for tool in DEFAULT_ALLOWED_TOOLS
            },
            "fallback_cases": sum(
                any(failure.startswith("tool_fallback:") for failure in item["failures"])
                for item in results
            ),
        },
        "evidence_cards": {
            "coverage_events": sum(
                bool(item["evidence_card_coverage_observed"]) for item in results
            ),
            "paper_sections_validated": sum(
                int(item["paper_sections_validated"]) for item in results
            ),
        },
        "transport_error": transport_error,
        "checkpoint_contract": checkpoint.contract if checkpoint else None,
        "resumed_cases": checkpoint.resumed_cases if checkpoint else 0,
        "suite_complete": len(results) == len(cases) and transport_error is None,
        "results": results,
    }
    _write_report(report, args.output)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
