#!/usr/bin/env python3
"""Merge independent visual-retrieval gate reports under a strict contract.

The visual gate deliberately samples different public suites in separate
processes (and often separate isolated indexes).  This utility combines only
reports that were produced with the same evaluator/model contract; it never
re-runs retrieval, calls an LLM, or mutates an index.  A merged report is an
aggregation of disjoint, already-scored rows, not a replacement for the
per-suite reports.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "backend"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_visual_retrieval_gate import (  # noqa: E402
    _aggregate_by_key,
    _visual_failure_taxonomy,
    visual_report_promotion_gate,
)


CONTRACT_VERSION = "visual-arbitrary-format-v2"
METRIC_CONTRACT_VERSION = "visual-evidence-scope-v2"
SCHEMA_VERSION = "visual-retrieval-gate-v1"
ARMS = ("T", "V", "T+V")


class MergeContractError(ValueError):
    """Raised when reports cannot be safely combined."""


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 3)
    return round(
        ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower),
        3,
    )


def _validate_report(path: Path, report: Any) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise MergeContractError(f"{path}: report must be an object")
    if report.get("schema_version") != SCHEMA_VERSION:
        raise MergeContractError(f"{path}: incompatible schema_version")
    if report.get("promotion_contract_version") != CONTRACT_VERSION:
        raise MergeContractError(f"{path}: incompatible promotion contract")
    if report.get("metric_contract_version") != METRIC_CONTRACT_VERSION:
        raise MergeContractError(f"{path}: incompatible metric contract; recompute first")
    if not isinstance(report.get("evaluator_fingerprint"), str) or not report["evaluator_fingerprint"]:
        raise MergeContractError(f"{path}: missing evaluator_fingerprint; stale report cannot be merged")
    if report.get("no_llm_calls") is not True:
        raise MergeContractError(f"{path}: report is not proven LLM-free")
    rows = report.get("rows")
    if not isinstance(rows, list) or report.get("cases") != len(rows):
        raise MergeContractError(f"{path}: cases/rows mismatch")
    for row in rows:
        if not isinstance(row, dict) or not str(row.get("case_id") or ""):
            raise MergeContractError(f"{path}: row has no stable case_id")
        if row.get("status") == "ok":
            if not isinstance(row.get("metrics"), dict) or not isinstance(
                row.get("latency_ms"), dict
            ):
                raise MergeContractError(f"{path}: successful row lacks scoring fields")
    return report


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    positive = [
        row
        for row in rows
        if row.get("status") == "ok"
        and not row.get("unanswerable")
        and row.get("required_keys")
    ]
    aggregate: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        metrics = [row["metrics"][arm] for row in positive if arm in row["metrics"]]
        latency = [
            float(row["latency_ms"][arm])
            for row in positive
            if arm in row.get("latency_ms", {})
        ]
        aggregate[arm] = {
            "positive_cases": len(metrics),
            "recall": round(statistics.mean(item["recall"] for item in metrics), 4)
            if metrics
            else None,
            "hit_at_1": round(statistics.mean(item["hit_at_1"] for item in metrics), 4)
            if metrics
            else None,
            "hit_at_3": round(statistics.mean(item["hit_at_3"] for item in metrics), 4)
            if metrics
            else None,
            "mrr": round(statistics.mean(item["mrr"] for item in metrics), 4)
            if metrics
            else None,
            "ndcg": round(statistics.mean(item["ndcg"] for item in metrics), 4)
            if metrics
            else None,
            "p50_ms": _percentile(latency, 0.5),
            "p95_ms": _percentile(latency, 0.95),
        }
    return aggregate


def merge_reports(
    paths: list[Path],
    include_case_ids: set[str] | None = None,
    exclude_case_ids: set[str] | None = None,
) -> dict[str, Any]:
    if len(paths) < 2:
        raise MergeContractError("at least two reports are required")
    reports = [_validate_report(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]
    invariant_fields = (
        "schema_version",
        "promotion_contract_version",
        "model",
        "model_revision",
        "model_fingerprint",
        "evaluator_fingerprint",
        "embedding_model_T",
        "pdf_ocr_enabled",
        "evaluation_configuration",
    )
    reference = reports[0]
    for report, path in zip(reports[1:], paths[1:]):
        for field in invariant_fields:
            if report.get(field) != reference.get(field):
                raise MergeContractError(f"{path}: {field} differs from first report")
    included = include_case_ids
    excluded = exclude_case_ids or set()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for report, path in zip(reports, paths):
        for row in report["rows"]:
            case_id = str(row["case_id"])
            if included is not None and case_id not in included:
                continue
            if case_id in excluded:
                continue
            if case_id in seen:
                raise MergeContractError(f"{path}: duplicate case_id {case_id}")
            seen.add(case_id)
            rows.append(row)
    worker_health = [report.get("worker_health") for report in reports]
    original_frontier = [
        case_id
        for report in reports
        for case_id in ((report.get("selection") or {}).get("case_ids") or [])
    ]
    source_complete = all(
        visual_report_promotion_gate(report)["checks"]["complete_frontier"]
        and report.get("score_valid") is True
        and report.get("suite_complete") is True
        for report in reports
    )
    posthoc_selection = included is not None or bool(excluded) or any(
        report.get("posthoc_selection") for report in reports
    )
    merged = {
        "schema_version": SCHEMA_VERSION,
        "promotion_contract_version": CONTRACT_VERSION,
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        "merge_contract_version": "visual-report-merge-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "model": reference.get("model"),
        "model_revision": reference.get("model_revision"),
        "model_fingerprint": reference.get("model_fingerprint"),
        "evaluator_fingerprint": reference.get("evaluator_fingerprint"),
        "embedding_model_T": reference.get("embedding_model_T"),
        "pdf_ocr_enabled": reference.get("pdf_ocr_enabled"),
        "evaluation_configuration": reference.get("evaluation_configuration"),
        "no_llm_calls": True,
        "score_valid": source_complete and not posthoc_selection,
        "suite_complete": source_complete,
        "posthoc_selection": posthoc_selection,
        "selected_cases": len(original_frontier),
        "cases": len(rows),
        "successful": sum(row.get("status") == "ok" for row in rows),
        "failures_by_type": dict(
            Counter(
                str(row.get("error_type") or "unknown")
                for row in rows
                if row.get("status") != "ok"
            )
        ),
        "aggregate": _aggregate(rows),
        "aggregate_by_suite": _aggregate_by_key(rows, "suite"),
        "aggregate_by_profile": _aggregate_by_key(rows, "profile"),
        "failure_taxonomy": _visual_failure_taxonomy(rows),
        "visual_index_size_bytes": sum(
            int(report.get("visual_index_size_bytes") or 0) for report in reports
        ),
        "worker_health": {
            "ok": all(isinstance(item, dict) and item.get("ok") for item in worker_health),
            "reports": worker_health,
        },
        "runtime_roots": [str(report.get("runtime_root") or "") for report in reports],
        "source_reports": [str(path) for path in paths],
        "selection": {
            "suites": sorted(
                {
                    str(row.get("suite") or "")
                    for row in rows
                    if str(row.get("suite") or "")
                }
            ),
            "case_ids": [str(row["case_id"]) for row in rows],
        },
        "original_selection": {"case_ids": original_frontier},
        "rows": rows,
    }
    if not posthoc_selection:
        merged["selection"]["case_ids"] = original_frontier
    merged["promotion"] = visual_report_promotion_gate(merged)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--include-case-id",
        action="append",
        default=None,
        help="Restrict merge to these case IDs (repeatable).",
    )
    parser.add_argument(
        "--exclude-case-id",
        action="append",
        default=[],
        help="Exclude a case ID from source reports (repeatable).",
    )
    args = parser.parse_args()
    try:
        report = merge_reports(
            [path.expanduser().resolve() for path in args.reports],
            {str(item) for item in args.include_case_id} if args.include_case_id else None,
            {str(item) for item in args.exclude_case_id},
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError, MergeContractError) as exc:
        print(f"merge_failed: {exc}", file=sys.stderr)
        return 2
    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().resolve().write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "promotion": report["promotion"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
