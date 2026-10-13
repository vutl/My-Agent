#!/usr/bin/env python3
"""Independent semantic review for a Gate-C retrieval suite.

This is deliberately separate from the deterministic proposal builder and the
retrieval runner.  It sends only the question/context and the proposed evidence
excerpts to the explicitly selected 9router model.  It never rewrites a suite
or promotes a draft; a human (or an explicit publish command) must make that
decision after inspecting the report.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.config import APPROVED_9ROUTER_MODELS, get_settings  # noqa: E402
from app.llm.openai_client import OpenAICompatibleClient  # noqa: E402


PROMPT_VERSION = "gate-c-v22-semantic-audit-v1"
CHECKPOINT_SCHEMA_VERSION = "retrieval-eval-v22-semantic-audit-checkpoint-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _audit_config_fingerprint(
    *,
    suite_sha256: str,
    model: str,
    expected_cases: int,
    batch_size: int,
) -> str:
    payload = {
        "suite_sha256": suite_sha256,
        "model": model,
        "expected_cases": int(expected_cases),
        "batch_size": int(batch_size),
        "prompt_version": PROMPT_VERSION,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def _load_checkpoint(
    path: Path,
    *,
    config_fingerprint: str,
    expected_case_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load only validated reviews; failed batches are always retried."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid semantic audit checkpoint: {path}") from exc
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("semantic audit checkpoint schema mismatch; use --no-resume")
    if payload.get("config_fingerprint") != config_fingerprint:
        raise ValueError("semantic audit checkpoint config mismatch; use --no-resume")
    reviews: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in payload.get("reviews") or []:
        if not isinstance(item, dict):
            continue
        case_id = str(item.get("case_id") or "")
        if not case_id or case_id not in expected_case_ids or case_id in seen:
            continue
        if item.get("status") not in {"accept", "reject", "uncertain"}:
            continue
        seen.add(case_id)
        reviews.append(item)
    failed: list[dict[str, Any]] = []
    for item in payload.get("failed_batches") or []:
        if not isinstance(item, dict):
            continue
        try:
            batch_index = int(item.get("batch_index"))
        except (TypeError, ValueError):
            continue
        failed.append({**item, "batch_index": batch_index})
    return reviews, failed


def _write_checkpoint(
    path: Path,
    *,
    config_fingerprint: str,
    suite_sha256: str,
    model: str,
    expected_cases: int,
    batch_size: int,
    reviews: list[dict[str, Any]],
    failed_batches: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "status": "partial",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_fingerprint": config_fingerprint,
        "suite_sha256": suite_sha256,
        "model": model,
        "expected_cases": expected_cases,
        "batch_size": batch_size,
        "reviews": reviews,
        "failed_batches": failed_batches,
        "reviewed_cases": len({str(item.get("case_id")) for item in reviews}),
    }
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _compact_text(value: object, limit: int = 900) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _audit_packet(row: dict[str, Any]) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    for group in row.get("evidence_groups") or []:
        document = group.get("document") or {}
        kind = group.get("kind")
        # Review the complete canonical artifact/span needed for the facet.
        # A fixed 1,200-character preview made long architecture excerpts and
        # tables look unsupported even when the stored qrel contained them.
        excerpt_limit = 5000 if kind == "table" else 3600 if kind == "text" else 1800
        groups.append(
            {
                "group_id": group.get("group_id"),
                "kind": kind,
                "filename": document.get("filename"),
                "page": group.get("page_number"),
                "section": _compact_text(group.get("section"), 220),
                "caption": _compact_text(group.get("caption"), 320),
                # The reviewer sees the candidate evidence, not hidden index
                # rows or the expected answer generated by the benchmark.
                "excerpt": _compact_text(group.get("quote"), excerpt_limit),
            }
        )
    return {
        "case_id": row.get("case_id"),
        "category": row.get("category"),
        "language": row.get("language"),
        "raw_query": row.get("raw_query"),
        "conversation_context": row.get("conversation_context") or [],
        "working_state": row.get("working_state") or {},
        "scope": row.get("scope"),
        "focus_documents": [item.get("filename") for item in row.get("focus_documents") or []],
        "required_entities": row.get("required_entities") or [],
        "expected_abstention": bool(row.get("expected_abstention")),
        "evidence_groups": groups,
    }


def _parse_json(message: str) -> dict[str, Any]:
    candidate = message.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
        if not match:
            raise ValueError("model did not return a JSON object")
        value = json.loads(match.group(0))
    if not isinstance(value, dict) or not isinstance(value.get("cases"), list):
        raise ValueError("model JSON must contain a cases list")
    return value


def _validate_batch_result(result: dict[str, Any], expected_ids: set[str]) -> None:
    seen: set[str] = set()
    for item in result["cases"]:
        if not isinstance(item, dict):
            raise ValueError("case review is not an object")
        case_id = str(item.get("case_id") or "")
        if case_id not in expected_ids or case_id in seen:
            raise ValueError(f"unexpected or duplicate case_id: {case_id!r}")
        seen.add(case_id)
        if item.get("status") not in {"accept", "reject", "uncertain"}:
            raise ValueError(f"invalid status for {case_id}")
        if not isinstance(item.get("evidence", []), list):
            raise ValueError(f"invalid evidence review for {case_id}")
    if seen != expected_ids:
        raise ValueError(f"batch omitted cases: {sorted(expected_ids - seen)}")


def _messages(packet: list[dict[str, Any]]) -> list[dict[str, str]]:
    system = (
        "You are an independent semantic auditor for a retrieval benchmark. "
        "Do not answer the questions and do not rewrite them. Judge only whether "
        "the proposed evidence groups genuinely support the requested information "
        "given the raw query, conversation context, and working state. Do not infer "
        "support from a filename alone. For a negative case, confirm that the "
        "question should be abstained from because the supplied corpus evidence is "
        "not sufficient. Mark uncertain instead of guessing. Return JSON only with "
        "this shape: {\"cases\":[{\"case_id\":string,\"status\":\"accept\"|\"reject\"|\"uncertain\","
        "\"evidence\":[{\"group_id\":string,\"status\":\"supported\"|\"unsupported\"|\"uncertain\","
        "\"reason\":string}],\"issues\":[string]}]}. Keep reasons concise."
    )
    user = (
        "Review every case below independently. Accept only when the evidence is "
        "sufficient for the exact facet requested (task, architecture, dataset, "
        "results, contribution, table/figure, or comparison) and the context makes "
        "references such as 'this paper' resolvable. Reject when a qrel is clearly "
        "irrelevant, a required document/facet is missing, or a negative case is "
        "answerable from the supplied evidence.\n\n"
        + json.dumps({"cases": packet}, ensure_ascii=False, separators=(",", ":"))
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


async def _review_batch(
    *,
    client: OpenAICompatibleClient,
    model: str,
    rows: list[dict[str, Any]],
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    packets = [_audit_packet(row) for row in rows]
    expected = {str(row["case_id"]) for row in rows}
    async with semaphore:
        completion = await client.chat(
            model=model,
            messages=_messages(packets),
            temperature=0.0,
            num_predict=2200,
            response_format={"type": "json_object"},
        )
    result = _parse_json(completion.message)
    _validate_batch_result(result, expected)
    return result


async def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = [json.loads(line) for line in args.suite.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != args.expected_cases:
        raise ValueError(f"expected {args.expected_cases} cases, found {len(rows)}")
    if args.model not in APPROVED_9ROUTER_MODELS:
        raise ValueError(f"refusing unapproved model: {args.model}")
    if any(row.get("curation_status") not in {"proposal", "review_required"} for row in rows):
        raise ValueError("semantic audit expects a proposal/review_required suite")

    suite_sha256 = _sha256(args.suite)
    batch_size = max(1, int(args.batch_size))
    config_fingerprint = _audit_config_fingerprint(
        suite_sha256=suite_sha256,
        model=args.model,
        expected_cases=args.expected_cases,
        batch_size=batch_size,
    )
    output = Path(args.output).expanduser().resolve()
    checkpoint_arg = getattr(args, "checkpoint", None)
    checkpoint = Path(checkpoint_arg).expanduser().resolve() if checkpoint_arg else output.with_suffix(output.suffix + ".partial")
    resume = bool(getattr(args, "resume", True))
    expected_case_ids = {str(row["case_id"]) for row in rows}
    reviews: list[dict[str, Any]] = []
    failed_batches: list[dict[str, Any]] = []
    resumed = False
    if not resume and checkpoint.is_file():
        # ``--no-resume`` must also invalidate the old checkpoint before any
        # provider preflight.  Otherwise a failed fresh run could leave old
        # rows behind for a later ``--resume`` invocation.
        _write_checkpoint(
            checkpoint,
            config_fingerprint=config_fingerprint,
            suite_sha256=suite_sha256,
            model=args.model,
            expected_cases=args.expected_cases,
            batch_size=batch_size,
            reviews=[],
            failed_batches=[],
        )
    if resume and checkpoint.is_file():
        reviews, failed_batches = _load_checkpoint(
            checkpoint,
            config_fingerprint=config_fingerprint,
            expected_case_ids=expected_case_ids,
        )
        resumed = bool(reviews or failed_batches)

    batches = [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]
    reviewed_ids = {str(item.get("case_id")) for item in reviews}
    batch_ids_by_index = {
        index: {str(row["case_id"]) for row in batch}
        for index, batch in enumerate(batches)
    }
    # A successful retry supersedes an older failure entry.  Remove stale
    # diagnostics before deciding completeness so a checkpoint cannot remain
    # permanently partial merely because its previous attempt failed.
    failed_batches = [
        item
        for item in failed_batches
        if not batch_ids_by_index.get(int(item.get("batch_index", -1)), set()).issubset(reviewed_ids)
    ]
    pending = [
        (index, batch)
        for index, batch in enumerate(batches)
        if not {str(row["case_id"]) for row in batch}.issubset(reviewed_ids)
    ]
    settings = get_settings()
    health: dict[str, Any] | None = None
    if pending:
        client = OpenAICompatibleClient(
            base_url=settings.openai_api_base,
            api_key=settings.openai_api_key,
            timeout_seconds=args.timeout,
        )
        health = await client.health()
        if not health.get("reachable") or args.model not in set(health.get("models") or []):
            raise RuntimeError(f"9router preflight failed for {args.model}: {health}")

        semaphore = asyncio.Semaphore(max(1, int(args.concurrency)))

        async def _run_one(index: int, batch: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]], dict[str, Any] | None, dict[str, Any] | None]:
            try:
                result = await _review_batch(
                    client=client,
                    model=args.model,
                    rows=batch,
                    semaphore=semaphore,
                )
                return index, batch, result, None
            except Exception as exc:
                return index, batch, None, {
                    "batch_index": index,
                    "case_ids": [str(row["case_id"]) for row in batch],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }

        tasks = [asyncio.create_task(_run_one(index, batch)) for index, batch in pending]
        for task in asyncio.as_completed(tasks):
            index, batch, result, error = await task
            failed_batches = [item for item in failed_batches if item.get("batch_index") != index]
            if error is not None:
                failed_batches.append(error)
            else:
                assert result is not None
                batch_ids = {str(row["case_id"]) for row in batch}
                reviews = [item for item in reviews if str(item.get("case_id")) not in batch_ids]
                reviews.extend(result["cases"])
            _write_checkpoint(
                checkpoint,
                config_fingerprint=config_fingerprint,
                suite_sha256=suite_sha256,
                model=args.model,
                expected_cases=args.expected_cases,
                batch_size=batch_size,
                reviews=reviews,
                failed_batches=failed_batches,
            )
    reviews.sort(key=lambda item: str(item["case_id"]))
    counts = {status: sum(item.get("status") == status for item in reviews) for status in ("accept", "reject", "uncertain")}
    reviewed_ids = {str(item.get("case_id")) for item in reviews}
    incomplete_case_ids = sorted(expected_case_ids - reviewed_ids)
    complete = not incomplete_case_ids and not failed_batches
    return {
        "schema_version": "retrieval-eval-v2.2-semantic-audit-v1",
        "prompt_version": PROMPT_VERSION,
        "suite": str(args.suite),
        "suite_sha256": suite_sha256,
        "model": args.model,
        "endpoint": settings.openai_api_base,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cases": len(rows),
        "reviewed_cases": len(reviewed_ids),
        "incomplete_case_ids": incomplete_case_ids,
        "failed_batches": sorted(failed_batches, key=lambda item: int(item.get("batch_index", 0))),
        "status": "complete" if complete else "partial",
        "complete": complete,
        "config_fingerprint": config_fingerprint,
        "checkpoint_path": str(checkpoint),
        "resumed": resumed,
        "health": health,
        "counts": counts,
        "promotion_eligible": complete and counts["reject"] == 0 and counts["uncertain"] == 0,
        "reviews": reviews,
        "note": "This report is an independent model review; it never publishes or mutates the suite.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=PROJECT_ROOT / "evals/retrieval_v2/dev-v2.2.proposed.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="checkpoint path; defaults to <output>.partial",
    )
    parser.add_argument("--model", default="cx/gpt-5.6-sol")
    parser.add_argument("--expected-cases", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--resume", action="store_true", dest="resume")
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    try:
        report = asyncio.run(run(args))
    except RuntimeError as exc:
        # Provider preflight failures are operationally expected.  Emit a
        # machine-readable, non-promotable report rather than a traceback that
        # could be mistaken for a semantic rejection.
        if "9router preflight failed" not in str(exc):
            raise
        output = args.output.expanduser().resolve()
        checkpoint = (
            args.checkpoint.expanduser().resolve()
            if args.checkpoint
            else output.with_suffix(output.suffix + ".partial")
        )
        report = {
            "schema_version": "retrieval-eval-v2.2-semantic-audit-v1",
            "status": "provider_unavailable",
            "complete": False,
            "promotion_eligible": False,
            "model": args.model,
            "suite": str(args.suite),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "checkpoint_path": str(checkpoint),
            "note": "No semantic review was recorded; retry after the approved 9router model is reachable.",
        }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(
        json.dumps(
            {key: report.get(key) for key in ("schema_version", "model", "cases", "counts", "promotion_eligible")},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report.get("promotion_eligible") else 2


if __name__ == "__main__":
    raise SystemExit(main())
