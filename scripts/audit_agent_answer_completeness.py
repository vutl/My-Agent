#!/usr/bin/env python3
"""Structural completeness diagnostics; not a semantic answer-quality grader."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from evaluation_checkpoint import write_checkpoint


def structural_findings(answer: str) -> list[dict]:
    lines = answer.splitlines()
    findings = []
    fence = None
    visible = []
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            if fence is None:
                fence = marker
            elif marker == fence:
                fence = None
            continue
        if fence is None and stripped:
            visible.append((number, stripped))
    if fence is not None:
        findings.append({"reason": "unclosed_code_fence"})
    if visible and re.match(r"^#{1,6}\s+", visible[-1][1]):
        findings.append({"reason": "dangling_final_heading", "line": visible[-1][0],
                         "excerpt": visible[-1][1][:180]})
    delimiter = re.compile(r"^\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?$")
    for index, (number, text) in enumerate(visible):
        if not delimiter.fullmatch(text):
            continue
        if index == 0 or "|" not in visible[index - 1][1]:
            continue
        if index + 1 == len(visible) or "|" not in visible[index + 1][1]:
            findings.append({"reason": "markdown_table_without_data_rows", "line": number,
                             "excerpt": visible[index - 1][1][:180]})
        for row_number, row in visible[index + 1:]:
            if "|" not in row:
                break
            if delimiter.fullmatch(row):
                break
            # Remove at most the Markdown outer delimiters: stripping all
            # trailing pipes would erase the empty cell we need to inspect.
            cells = row.strip().removeprefix("|").removesuffix("|").split("|")
            if len(cells) >= 2 and cells[0].strip() and not any(
                cell.strip() for cell in cells[1:]
            ):
                # A legitimate group header can have the same shape. Surface
                # it for review; do not turn this diagnostic into a grader.
                findings.append({"reason": "markdown_table_empty_data_row", "line": row_number,
                                 "excerpt": row[:180]})
    return findings


def guarded_answer_findings(row: dict) -> list[dict]:
    """Surface partial/fallback output even when the rubric calls it valid.

    Retrieved artifacts do not prove answerability. This is a review trigger,
    not a rule that forbids honest abstention or a semantic correctness score.
    """
    validation = row.get("answer_validation") or {}
    findings = []
    if validation.get("fallback_used"):
        findings.append({"reason": "answer_fallback_requires_review"})
    answer = row.get("answer") or ""
    insufficient = re.search(
        r"insufficient\s+evidence|(?:chưa|không)\s+có\s+đủ\s+"
        r"(?:evidence|bằng\s+chứng)|not\s+enough\s+evidence", answer, re.I,
    )
    if insufficient and (row.get("table_ids") or row.get("figure_ids")):
        findings.append({"reason": "partial_answer_with_retrieved_artifacts",
                         "note": "Check task coverage against sources; artifact presence is not answerability."})
    return findings


def audit(path: Path) -> dict:
    raw = path.read_bytes()
    report = json.loads(raw)
    rows = report.get("results", [])
    if not rows or any("answer" not in row for row in rows):
        raise ValueError("Audit requires non-empty results with --include-answers")
    findings = [{"case_id": row["case_id"], "run_id": row.get("run_id"),
                 "findings": structural_findings(row["answer"]) + guarded_answer_findings(row)} for row in rows]
    affected = [row for row in findings if row["findings"]]
    return {"schema_version": "agent-answer-completeness-diagnostic-v2",
            "source_report": str(path), "source_sha256": hashlib.sha256(raw).hexdigest(),
            "diagnostic_only": True, "semantic_quality_verified": False,
            "requires_review": bool(affected), "cases_audited": len(rows),
            "affected_cases": len(affected), "findings": affected,
            "limitations": ["No fact/source entailment scoring", "No task-facet coverage scoring",
                            "Findings require review; honest abstention and empty templates can be valid",
                            "A clean structural audit does not prove answer completeness"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.report)
    write_checkpoint(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 2 if report["requires_review"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
