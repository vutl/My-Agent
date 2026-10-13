#!/usr/bin/env python3
"""Merge disjoint, complete public answer-quality reports.

Answer-quality runs deliberately use case-specific config fingerprints so a
checkpoint cannot be resumed with a different frontier.  That fingerprint
must not prevent a *post-hoc* merge of disjoint reports when the semantic
evaluation contract is identical.  This utility validates that contract,
recomputes case/mode aggregates from rows, and preserves source provenance.
It never calls a model or changes an index.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime
import json
from pathlib import Path
import statistics
from typing import Any


class MergeContractError(ValueError):
    """Raised when answer-quality reports cannot be safely paired."""


MODES = ("gold_evidence", "full_extracted_document", "aya_pipeline")


def _load(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("status") != "complete":
        raise MergeContractError(f"{path}: report is not complete")
    if report.get("model") not in {
        "cx/gpt-5.6-sol",
        "cx/gpt-5.6-terra",
        "cx/gpt-5.6-luna",
    }:
        raise MergeContractError(f"{path}: unapproved answer model")
    if not isinstance(report.get("evaluator_fingerprint"), str) or not report["evaluator_fingerprint"]:
        raise MergeContractError(f"{path}: missing evaluator_fingerprint; stale report cannot be merged")
    quality = report.get("answer_quality")
    if not isinstance(quality, dict) or quality.get("status") != "complete":
        raise MergeContractError(f"{path}: answer_quality is not complete")
    return report


def _case_ids(report: dict[str, Any]) -> list[str]:
    selected = (report.get("selection") or {}).get("case_ids")
    if not isinstance(selected, list) or not selected:
        raise MergeContractError("report has no ordered case frontier")
    return [str(item) for item in selected]


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 3) if values else None


def _aggregate_case_results(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        mode_rows = [row for row in rows if row.get("mode") == mode and row.get("status") == "ok"]
        scores = [float((row.get("score") or {}).get("official_mmlong_score")) for row in mode_rows if (row.get("score") or {}).get("official_mmlong_score") is not None]
        first_tokens = [float((row.get("generation") or {}).get("first_token_ms")) for row in mode_rows if (row.get("generation") or {}).get("first_token_ms") is not None]
        totals = [float(row.get("total_ms")) for row in mode_rows if row.get("total_ms") is not None]
        recalls = [float(row.get("evidence_page_recall")) for row in mode_rows if row.get("evidence_page_recall") is not None]
        payload: dict[str, Any] = {
            "cases": len(mode_rows),
            "median_total_ms": _median(totals),
            "median_first_token_ms": _median(first_tokens),
            "official_accuracy": statistics.mean(scores) if scores else None,
            "mean_evidence_page_recall": statistics.mean(recalls) if recalls else None,
        }
        if mode == "aya_pipeline":
            index_values = [float((row.get("input") or {}).get("index_ms")) for row in mode_rows if (row.get("input") or {}).get("index_ms") is not None]
            retrieval_values = [float((row.get("input") or {}).get("retrieval_ms")) for row in mode_rows if (row.get("input") or {}).get("retrieval_ms") is not None]
            visual_rows = [row for row in mode_rows if (row.get("input") or {}).get("visual_context_expected")]
            payload.update(
                {
                    "median_index_ms": _median(index_values),
                    "median_retrieval_ms": _median(retrieval_values),
                    "visual_context_expected_cases": len(visual_rows),
                    "visual_context_source_rate": (
                        statistics.mean(bool((row.get("input") or {}).get("visual_context_source_count")) for row in visual_rows)
                        if visual_rows else None
                    ),
                    "visual_image_attachment_rate": (
                        statistics.mean(bool((row.get("input") or {}).get("visual_images_attached")) for row in visual_rows)
                        if visual_rows else None
                    ),
                }
            )
        result[mode] = payload
    return result


def _group_quality(per_case: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_case:
        groups[str(row.get(key) or "unknown")].append(row)
    return {
        name: _quality_summary(items)
        for name, items in sorted(groups.items())
    }


def _quality_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean_score(mode: str) -> float | None:
        values = [float((row.get("scores") or {}).get(mode)) for row in rows if (row.get("scores") or {}).get(mode) is not None]
        return round(statistics.mean(values), 6) if values else None

    aya = mean_score("aya_pipeline")
    full = mean_score("full_extracted_document")
    gold = mean_score("gold_evidence")
    return {
        "cases": len(rows),
        "aya_vs_full_delta": round(aya - full, 6) if aya is not None and full is not None else None,
        "aya_vs_gold_delta": round(aya - gold, 6) if aya is not None and gold is not None else None,
        "aya_score": aya,
        "full_score": full,
        "gold_score": gold,
    }


def merge(
    paths: list[Path],
    ordered_case_ids: list[str],
    excluded_case_ids: set[str] | None = None,
) -> dict[str, Any]:
    if len(paths) < 2:
        raise MergeContractError("at least two reports are required")
    reports = [_load(path) for path in paths]
    invariant_fields = (
        "model",
        "evaluator_fingerprint",
        "modes",
        "visual_pages_enabled",
        "pdf_ocr_enabled",
        "retrieval_configuration",
    )
    reference = reports[0]
    for report, path in zip(reports[1:], paths[1:]):
        for field in invariant_fields:
            if report.get(field) != reference.get(field):
                raise MergeContractError(f"{path}: {field} differs from first report")

    excluded = excluded_case_ids or set()
    per_case: list[dict[str, Any]] = []
    case_results: list[dict[str, Any]] = []
    seen_cases: set[str] = set()
    seen_rows: set[tuple[str, str]] = set()
    for path, report in zip(paths, reports):
        for row in report["answer_quality"].get("per_case") or []:
            case_id = str(row.get("case_id") or "")
            if case_id in excluded:
                continue
            if not case_id or case_id in seen_cases:
                raise MergeContractError(f"{path}: duplicate answer-quality case {case_id!r}")
            seen_cases.add(case_id)
            per_case.append(row)
        for row in report.get("case_results") or []:
            key = (str(row.get("case_id") or ""), str(row.get("mode") or ""))
            if key[0] in excluded:
                continue
            if key in seen_rows:
                raise MergeContractError(f"{path}: duplicate case/mode row {key}")
            seen_rows.add(key)
            case_results.append(row)

    if set(ordered_case_ids) != seen_cases or len(ordered_case_ids) != len(seen_cases):
        raise MergeContractError("ordered frontier does not exactly match merged cases")
    by_id = {str(row["case_id"]): row for row in per_case}
    ordered_per_case = [by_id[case_id] for case_id in ordered_case_ids]
    quality = {
        "status": "complete",
        "expected_cases": len(ordered_case_ids),
        "paired_cases": len(ordered_case_ids),
        "incomplete_cases": 0,
        "per_case": ordered_per_case,
        "overall": _quality_summary(ordered_per_case),
        "by_suite": _group_quality(ordered_per_case, "suite"),
        "by_stratum": _group_quality(ordered_per_case, "stratum"),
        "by_document_profile": _group_quality(ordered_per_case, "document_profile"),
        "promotion_decision": "not_evaluated",
        "source_reports": [str(path) for path in paths],
        "note": "Merged from disjoint complete paired reports; no model call was made during merge.",
    }
    return {
        "schema_version": 1,
        "ok": True,
        "status": "complete",
        "suite": reference.get("suite"),
        "model": reference.get("model"),
        "evaluator_fingerprint": reference.get("evaluator_fingerprint"),
        "modes": reference.get("modes"),
        "visual_pages_enabled": reference.get("visual_pages_enabled"),
        "pdf_ocr_enabled": reference.get("pdf_ocr_enabled"),
        "retrieval_configuration": reference.get("retrieval_configuration"),
        "case_count": len(ordered_case_ids),
        "generation_count": len(case_results),
        "failed_count": sum(row.get("status") != "ok" for row in case_results),
        "complete": True,
        "config_fingerprint": "posthoc-union:" + ":".join(sorted(report.get("config_fingerprint", "") for report in reports)),
        "selection": {"method": "explicit_ordered_union", "case_ids": ordered_case_ids},
        "aggregates": _aggregate_case_results(case_results),
        "answer_quality": quality,
        "case_results": case_results,
        "production_corpus_modified": False,
        "heldout_60_opened": False,
        "source_reports": [str(path) for path in paths],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--case-ids-file", required=True, type=Path)
    parser.add_argument(
        "--exclude-case-id",
        action="append",
        default=[],
        help="Exclude a case from source reports (repeatable for audited frontier subsets).",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        payload = json.loads(args.case_ids_file.read_text(encoding="utf-8"))
        ordered = payload.get("case_ids") if isinstance(payload, dict) else payload
        if not isinstance(ordered, list):
            raise MergeContractError("case IDs file must contain a list or case_ids")
        result = merge(
            [path.expanduser().resolve() for path in args.reports],
            [str(item) for item in ordered],
            {str(item) for item in args.exclude_case_id},
        )
        args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.expanduser().resolve().write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output), "cases": result["case_count"]}, indent=2))
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError, MergeContractError) as exc:
        print(f"merge_failed: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
