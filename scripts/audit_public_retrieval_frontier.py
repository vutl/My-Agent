#!/usr/bin/env python3
"""Audit observed candidate/context boundaries without rerunning or tuning RAG.

This postprocessor never imports the retrieval pipeline, reads gold answers,
calls a model, or changes index state. A filename in context is not proof that
the corresponding image pixels reached the answer model.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from evaluation_checkpoint import write_checkpoint


def _hits(value: Any, name: str) -> dict[str, bool]:
    if value is None:
        return {}
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(hit, bool) for key, hit in value.items()
    ):
        raise ValueError(f"invalid {name} telemetry")
    return {key: hit for key, hit in value.items()
            if key != "annotation_present_but_name_unresolved"}


def audit_case(row: dict[str, Any]) -> dict[str, Any]:
    labels: list[str] = []
    losses: list[str] = []
    missing: list[str] = []
    if row.get("status") != "ok":
        labels.append("ingestion_or_retrieval_error")
    elif row.get("unanswerable"):
        labels.append("negative_requires_e2e_abstention_check")
    else:
        candidate = _hits(row.get("candidate_artifact_hits"), "candidate_artifact_hits")
        final = _hits(row.get("referred_artifact_hits"), "referred_artifact_hits")
        if candidate.keys() != final.keys():
            raise ValueError("candidate/final artifact locator mismatch")
        for name, found in candidate.items():
            if not found:
                missing.append(name)
            if found and not final[name]:
                losses.append(name)
            if not found and final[name]:
                labels.append("artifact_recovered_after_candidate_frontier")
        if missing:
            labels.append("candidate_artifact_miss")
        if losses:
            labels.append("final_context_artifact_loss")
        image_count = row.get("image_count")
        if image_count is not None and (type(image_count) is not int or image_count < 0):
            raise ValueError("invalid image_count telemetry")
        if final and any(final.values()) and image_count == 0:
            labels.append("artifact_name_present_without_attached_pixels")
        before, after = row.get("candidate_evidence_hits"), row.get("evidence_phrase_hits")
        if before is not None and after is not None:
            if (not isinstance(before, list) or not isinstance(after, list)
                    or len(before) != len(after)
                    or any(type(hit) is not bool for hit in [*before, *after])):
                raise ValueError("invalid paired evidence phrase telemetry")
            if any(found and not kept for found, kept in zip(before, after)):
                labels.append("final_context_phrase_loss")
            if any(not found and kept for found, kept in zip(before, after)):
                # Parent expansion may legitimately recover a wider span;
                # do not misdiagnose that as a serialization mismatch.
                labels.append("phrase_recovered_after_candidate_frontier")
        required = set(row.get("required_pages") or [])
        if required:
            candidate_pages = set(row.get("candidate_evidence_pages") or [])
            final_pages = set(row.get("retrieved_pages") or [])
            if required - candidate_pages:
                labels.append("candidate_page_miss")
            if (required & candidate_pages) - final_pages:
                labels.append("final_context_page_loss")
        if row.get("evidence_annotation_note"):
            labels.append("annotation_requires_review")
    return {
        "case_id": row["case_id"], "suite": row.get("suite"),
        "document_profile": row.get("document_profile"), "stratum": row.get("stratum"),
        "labels": sorted(set(labels)), "missing_candidate_artifacts": missing,
        "lost_final_artifacts": losses,
        "implicit_artifact_question": bool(
            (missing or losses) and not row.get("artifact_support_required_by_query")
        ),
        "pixel_identity_verified": False,
    }


def audit_report(report: dict[str, Any], *, allow_partial: bool = False) -> dict[str, Any]:
    selected = (report.get("selected_case_ids")
                or (report.get("checkpoint_contract") or {}).get("case_ids"))
    rows = report.get("rows")
    if not isinstance(selected, list) or not selected or not isinstance(rows, list):
        raise ValueError("missing frozen frontier or rows")
    if len(set(selected)) != len(selected):
        raise ValueError("duplicate frozen case IDs")
    if any(not isinstance(row, dict) or not row.get("case_id") for row in rows):
        raise ValueError("invalid case rows")
    seen = [row["case_id"] for row in rows]
    if len(set(seen)) != len(seen) or set(seen) - set(selected):
        raise ValueError("duplicate or unexpected case rows")
    complete = set(seen) == set(selected)
    if not complete and not allow_partial:
        raise ValueError("frontier incomplete; use --allow-partial for a diagnostic only")
    audited = [audit_case(row) for row in rows]
    counts = Counter(label for row in audited for label in row["labels"])
    return {
        "schema_version": "public-frontier-boundary-audit-v1",
        "created_at": datetime.now(UTC).isoformat(), "selected_cases": len(selected),
        "observed_cases": len(rows), "frontier_complete": complete,
        "source_evaluator_fingerprint": report.get("evaluator_fingerprint"),
        "boundary_counts": dict(sorted(counts.items())),
        "implicit_artifact_failure_cases": sum(row["implicit_artifact_question"] for row in audited),
        "no_model_calls": True, "promotion_eligible": False, "cases": audited,
        "limitations": [
            "Observed boundaries do not distinguish FTS/vector/rerank without channel telemetry.",
            "Artifact filename hits do not verify image identity, OCR accuracy or answer support.",
            "Phrase matches and any-page hits are diagnostics, not semantic completeness.",
            "Implicit questions remain counted; absence of table/figure keywords is not an exemption.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.report.resolve() == args.output.resolve():
        raise ValueError("refusing to replace the source evaluation report")
    source = args.report.read_bytes()
    result = audit_report(json.loads(source), allow_partial=args.allow_partial)
    result.update(source_report=str(args.report), source_report_sha256=hashlib.sha256(source).hexdigest())
    write_checkpoint(args.output, result)
    print(json.dumps({key: result[key] for key in (
        "selected_cases", "observed_cases", "frontier_complete", "boundary_counts",
        "implicit_artifact_failure_cases", "promotion_eligible",
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
