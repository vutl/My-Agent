#!/usr/bin/env python3
"""Run a fail-closed, non-semantic audit over a Gate-C proposal.

This command checks annotation structure and provenance without pretending to
replace an independent human/LLM semantic review.  It never flips curation
status or writes an audited suite.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_dual_query_retrieval import (  # noqa: E402
    EvalCase,
    EvaluationContractError,
    POSITIVE_CATEGORIES,
    _validate_suite_shape,
    load_cases,
    suite_quality_diagnostics,
)


def _artifact_number(text: str, kind: str) -> int | None:
    marker = "table" if kind == "table" else "figure"
    localized = "bảng" if kind == "table" else "hình"
    match = re.search(rf"(?:{marker}|{localized})\s*(\d+)", text.casefold())
    return int(match.group(1)) if match else None


def _static_case_issues(case: EvalCase) -> list[str]:
    issues: list[str] = []
    if case.scope == "focused" and not case.focus_documents:
        issues.append("focused_scope_missing_catalog")
    if case.scope == "focused":
        context_text = " ".join(
            [
                *case.required_entities,
                json.dumps(case.working_state, ensure_ascii=False),
                " ".join(str(item.get("content") or "") for item in case.conversation_context),
            ]
        ).casefold()
        for item in case.focus_documents:
            filename = str(item.get("filename") or "").strip()
            if filename and filename.casefold() not in context_text:
                issues.append(f"focus_not_grounded:{filename}")

    if case.category == "negative_unanswerable":
        if case.evidence_groups or not case.expected_abstention:
            issues.append("negative_case_has_positive_evidence")
        return issues

    if not case.evidence_groups:
        issues.append("positive_case_missing_qrel")
        return issues
    documents = {str(locator.document.get("filename") or "") for locator in case.evidence_groups}
    if case.category == "cross_document_compare" and len(documents) < 2:
        issues.append("comparison_has_one_document")
    if case.category == "intra_document_multihop" and len(case.evidence_groups) < 2:
        issues.append("multihop_has_one_evidence_group")
    if case.category == "table_numeric":
        if any(locator.kind != "table" for locator in case.evidence_groups):
            issues.append("table_case_has_non_table_qrel")
        requested = _artifact_number(case.raw_query + " " + case.standalone_full, "table")
        if requested is not None:
            for locator in case.evidence_groups:
                caption_number = _artifact_number(locator.caption or "", "table")
                if caption_number is not None and caption_number != requested:
                    issues.append(f"table_label_mismatch:{requested}!={caption_number}")
    if case.category == "figure_page":
        if any(locator.kind not in {"figure", "page"} for locator in case.evidence_groups):
            issues.append("figure_case_has_non_visual_qrel")
    return issues


def audit(cases: list[EvalCase]) -> dict[str, Any]:
    _validate_suite_shape(cases, expected=60)
    quality = suite_quality_diagnostics(cases)
    issues = list(quality["issues"])
    for case in cases:
        for issue in _static_case_issues(case):
            issues.append(f"{case.case_id}:{issue}")
    return {
        "schema_version": "retrieval-eval-v2.2-static-audit-v1",
        "suite_cases": len(cases),
        "static_checks_passed": not issues,
        "issues": issues,
        "language_context_matrix": {
            f"{language}:{'contextual' if contextual else 'standalone'}": count
            for (language, contextual), count in quality["language_context_matrix"].items()
        },
        "semantic_review_required": True,
        "promotion_eligible": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=PROJECT_ROOT / "evals/retrieval_v2/dev-v2.2.proposed.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = load_cases(args.suite, require_audited=False)
    report = audit(cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["static_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
