#!/usr/bin/env python3
"""Acceptance evaluator for Aya's four user capabilities (docs/handoff/08).

Runs each conversation turn-by-turn over the public SSE API, reusing the
transport helpers of evaluate_agent_conversations, then checks STRUCTURE
without any model call:

- G1/G3/G4 focus: the answer's sources come from ``expected_document``.
- G2 attach: a figure event, a Markdown table, or a formula is present when
  ``must_attach`` asks for it.
- G3 off-topic: a side-question must not call search_local_docs.
- Fallback: the answer must not be the "not enough evidence" fallback when the
  turn is answerable.

Content judging (must_mention, naturalness, fabricated numbers) is a separate
step with a judge model; it is reported as "pending" when not run.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_agent_conversations import _run_turn, _first, _all  # noqa: E402

DEFAULT_MODEL = "cx/gpt-5.6-sol"
_TABLE_RE = re.compile(r"^\s*\|.*\|\s*$\n^\s*\|?\s*:?-{3,}", re.MULTILINE)
_FORMULA_RE = re.compile(r"\$\$.+?\$\$|\$[^$\n]+\$", re.DOTALL)
_FALLBACK_RE = re.compile(
    r"(chưa có đủ evidence|không đủ (thông tin|evidence|dữ liệu)|not enough (evidence|information)|"
    r"cannot (determine|find)|không thể xác định)",
    re.IGNORECASE,
)


def load_turns(path: Path) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            turn = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: invalid JSON ({exc})") from exc
        for key in ("conversation", "turn", "message", "group"):
            if key not in turn:
                raise ValueError(f"{path}:{number}: missing {key!r}")
        turns.append(turn)
    return turns


def _retrieved_artifacts(events: list[Any], key: str) -> set[str]:
    """Artifact ids the retrieval step selected (figure_id / table_id)."""
    documents = _first(events, "retrieval.completed").get("documents") or []
    return {str(item.get(key)) for item in documents if isinstance(item, dict) and item.get(key)}


def _attached(answer: str, events: list[Any], kind: str) -> bool:
    if kind == "figure":
        # The backend attaches figures from retrieval sources (see the figure
        # sources in retrieval.completed); a caption mention alone is not an attachment.
        return bool(_retrieved_artifacts(events, "figure_id"))
    if kind == "table":
        if _first(events, "direct_canonical_table") or _retrieved_artifacts(events, "table_id"):
            return True
        return bool(_TABLE_RE.search(answer))
    if kind == "formula":
        return bool(_FORMULA_RE.search(answer))
    return True


def _sources_document(events: list[Any], expected: str | None) -> bool | None:
    if expected is None:
        return None
    sources = _first(events, "retrieval.completed").get("documents") or []
    names = {str(item.get("filename") or "") for item in sources if isinstance(item, dict)}
    if not names:
        return False
    return any(expected in name for name in names)


def _searched(events: list[Any]) -> bool:
    return any(
        str(item.get("tool_name") or item.get("name") or "") == "search_local_docs"
        for item in _all(events, "tool.started")
    )


def check_turn(turn: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    events = result["events"]
    answer = result["answer"]
    group = turn["group"]
    checks: dict[str, bool | None] = {}

    checks["focus_document"] = _sources_document(events, turn.get("expected_document"))
    attach = turn.get("must_attach")
    checks["attach"] = _attached(answer, events, attach) if attach else None
    checks["off_topic_no_search"] = (not _searched(events)) if turn.get("off_topic") else None
    checks["no_fallback"] = (not _FALLBACK_RE.search(answer)) if not turn.get("off_topic") else None
    checks["answer_nonempty"] = bool(answer.strip())

    failed = [name for name, value in checks.items() if value is False]
    return {
        "conversation": turn["conversation"],
        "turn": turn["turn"],
        "group": group,
        "message": turn["message"],
        "checks": checks,
        "passed": not failed,
        "failed_checks": failed,
        "answer_chars": len(answer),
        "elapsed_ms": result["elapsed_ms"],
        "client_first_delta_ms": result.get("client_first_delta_ms"),
    }


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        by_group[row["group"]].append(row["passed"])
    return {
        group: {"passed": sum(values), "total": len(values),
                "rate": round(sum(values) / len(values), 3) if values else None}
        for group, values in sorted(by_group.items())
    }


def run(args: argparse.Namespace) -> int:
    turns = load_turns(Path(args.dataset))
    conversations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for turn in turns:
        conversations[turn["conversation"]].append(turn)

    rows: list[dict[str, Any]] = []
    for name in sorted(conversations):
        conversation_id: str | None = None
        for turn in sorted(conversations[name], key=lambda item: item["turn"]):
            case = {
                "case_id": f"{name}-t{turn['turn']}",
                "message": turn["message"],
                "expected_document_ids": [],
                "expected_route": "file_qa",
            }
            result = _run_turn(
                base_url=args.base_url, case=case, conversation_id=conversation_id,
                model=args.model, timeout=args.timeout,
            )
            conversation_id = conversation_id or _conversation_id(result["events"])
            row = check_turn(turn, result)
            rows.append(row)
            status = "pass" if row["passed"] else f"fail {row['failed_checks']}"
            print(f"[{name} t{turn['turn']} {turn['group']}] {status}", flush=True)

    report = {
        "dataset": str(args.dataset),
        "model": args.model,
        "summary": summarise(rows),
        "rows": rows,
        "content_judge": "pending",
    }
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


def _conversation_id(events: list[Any]) -> str | None:
    started = _first(events, "run.started")
    value = started.get("conversation_id")
    return str(value) if value else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="evals/aya_acceptance_v1.jsonl")
    parser.add_argument("--base-url", default="http://127.0.0.1:7790")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
