#!/usr/bin/env python3
"""Attach a paired answer-quality report to a visual retrieval gate report.

The retrieval and answer-quality runners intentionally have different output
schemas.  This utility joins them only when they describe the exact same
ordered case frontier and approved model contract.  It never re-runs
retrieval, calls an LLM, or changes a promotion flag outside the shared
fail-closed gate.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_visual_retrieval_gate import visual_report_promotion_gate  # noqa: E402


APPROVED_ANSWER_MODELS = {
    "cx/gpt-5.6-sol",
    "cx/gpt-5.6-terra",
    "cx/gpt-5.6-luna",
}


class PairingContractError(ValueError):
    """Raised when the two reports cannot be compared safely."""


def _case_ids_from_answer(report: dict[str, Any]) -> list[str]:
    selection = report.get("selection") or {}
    selected = selection.get("case_ids")
    if isinstance(selected, list) and selected:
        return [str(value) for value in selected]
    rows = report.get("case_results") or []
    return list(dict.fromkeys(str(row.get("case_id")) for row in rows if row.get("case_id")))


def _case_ids_from_visual(report: dict[str, Any]) -> list[str]:
    selection = report.get("selection") or {}
    selected = selection.get("case_ids")
    if isinstance(selected, list) and selected:
        return [str(value) for value in selected]
    return [str(row.get("case_id")) for row in report.get("rows") or [] if row.get("case_id")]


def _promotion_safe(answer_quality: dict[str, Any], *, minimum_pairs: int) -> dict[str, Any]:
    """Return an explicit answer-level safety attestation.

    Overall improvement alone is not enough for a visual promotion: a large
    gain on one stratum must not hide a regression on another.  We therefore
    require complete paired rows, a non-negative aggregate delta, and no
    per-case answer regression.  The result is an attestation consumed by
    ``visual_promotion_gate``; it never enables a production flag.
    """

    per_case = [row for row in answer_quality.get("per_case") or [] if isinstance(row, dict)]
    regressions = [
        str(row.get("case_id"))
        for row in per_case
        if row.get("status") == "ok"
        and float(row.get("delta_aya_vs_full") or 0.0) < 0.0
    ]
    overall = answer_quality.get("overall") or {}
    try:
        pairs = int(answer_quality.get("paired_cases") or 0)
    except (TypeError, ValueError):
        pairs = 0
    try:
        aggregate_delta = float(overall.get("aya_vs_full_delta"))
    except (TypeError, ValueError):
        aggregate_delta = float("-inf")
    checks = {
        "complete": str(answer_quality.get("status") or "").casefold() == "complete",
        "minimum_pairs": pairs >= minimum_pairs,
        "all_rows_scored": len(per_case) == pairs
        and all(row.get("status") == "ok" for row in per_case),
        "non_negative_aggregate_delta": aggregate_delta >= 0.0,
        "no_per_case_regression": not regressions,
    }
    return {
        **answer_quality,
        "promotion_safe": all(checks.values()),
        "promotion_checks": checks,
        "regression_case_ids": regressions,
    }


def attach(
    visual: dict[str, Any],
    answer: dict[str, Any],
    *,
    minimum_pairs: int = 20,
) -> dict[str, Any]:
    if visual.get("schema_version") != "visual-retrieval-gate-v1":
        raise PairingContractError("visual report schema is unsupported")
    if visual.get("no_llm_calls") is not True:
        raise PairingContractError("visual report is not proven LLM-free")
    if not isinstance(visual.get("pdf_ocr_enabled"), bool):
        raise PairingContractError("visual report lacks pdf_ocr_enabled configuration")
    evaluation_configuration = visual.get("evaluation_configuration")
    if not isinstance(evaluation_configuration, dict):
        raise PairingContractError("visual report lacks evaluation_configuration")
    quality = answer.get("answer_quality") if isinstance(answer.get("answer_quality"), dict) else answer
    if answer.get("status") != "complete" or quality.get("status") != "complete":
        raise PairingContractError("answer-quality report is incomplete")
    visual_ids = _case_ids_from_visual(visual)
    answer_ids = _case_ids_from_answer(answer)
    if not visual_ids or visual_ids != answer_ids:
        raise PairingContractError(
            "case frontier mismatch; retrieval and answer reports must have identical ordered IDs"
        )
    quality_rows = quality.get("per_case")
    quality_ids = [
        row.get("case_id") for row in quality_rows if isinstance(row, dict)
    ] if isinstance(quality_rows, list) else []
    if (
        len(quality_ids) != len(visual_ids)
        or any(not isinstance(case_id, str) for case_id in quality_ids)
        or len(set(quality_ids)) != len(quality_ids)
        or set(quality_ids) != set(visual_ids)
        or quality.get("paired_cases") != len(visual_ids)
    ):
        raise PairingContractError("scored answer rows do not match the declared case frontier")
    answer_model = str(answer.get("model") or "").strip()
    if answer_model not in APPROVED_ANSWER_MODELS:
        raise PairingContractError(f"answer report uses unapproved model: {answer_model!r}")
    answer_ocr = answer.get("pdf_ocr_enabled")
    if isinstance(answer_ocr, bool) and answer_ocr != visual["pdf_ocr_enabled"]:
        raise PairingContractError("answer/visual reports use different pdf_ocr_enabled settings")
    paired = _promotion_safe(quality, minimum_pairs=minimum_pairs)
    result = dict(visual)
    result["answer_quality"] = paired
    result["answer_model"] = answer_model
    result["pairing_contract"] = {
        "version": "visual-answer-quality-pair-v1",
        "case_ids_sha256": __import__("hashlib").sha256(
            "\n".join(visual_ids).encode("utf-8")
        ).hexdigest(),
        "cases": len(visual_ids),
        "exact_order_match": True,
        "llm_calls": False,
    }
    result["promotion"] = visual_report_promotion_gate(
        result,
        minimum_positive_cases=minimum_pairs,
    )
    result["attached_at"] = datetime.now(UTC).isoformat()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("visual_report", type=Path)
    parser.add_argument("answer_report", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-pairs", type=int, default=20)
    args = parser.parse_args()
    try:
        visual = json.loads(args.visual_report.read_text(encoding="utf-8"))
        answer = json.loads(args.answer_report.read_text(encoding="utf-8"))
        if args.minimum_pairs < 1:
            raise PairingContractError("minimum-pairs must be positive")
        result = attach(visual, answer, minimum_pairs=args.minimum_pairs)
    except (OSError, TypeError, ValueError, json.JSONDecodeError, PairingContractError) as exc:
        print(f"pair_failed: {exc}", file=sys.stderr)
        return 2
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "promotion": result["promotion"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
