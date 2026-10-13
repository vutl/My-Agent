#!/usr/bin/env python3
"""Recompute visual-gate metrics from an already-scored report.

This is intentionally a no-retrieval/no-LLM migration utility.  It repairs
reports written before ``visual-evidence-scope-v2`` was introduced: MMLong
text is page-scored, while flattened SPIQA Markdown text is artifact-scored.
Rankings and evidence annotations are never changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "backend"))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_visual_retrieval_gate import (  # noqa: E402
    _aggregate_by_key,
    _aggregate_metrics,
    _visual_failure_taxonomy,
    _ranking_metrics,
    visual_report_promotion_gate,
)


METRIC_CONTRACT_VERSION = "visual-evidence-scope-v2"


def recompute(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("schema_version") != "visual-retrieval-gate-v1":
        raise ValueError("incompatible schema_version")
    if report.get("promotion_contract_version") != "visual-arbitrary-format-v2":
        raise ValueError("incompatible promotion contract")
    if report.get("no_llm_calls") is not True:
        raise ValueError("report is not proven LLM-free")
    rows = report.get("rows")
    if not isinstance(rows, list) or report.get("cases") != len(rows):
        raise ValueError("cases/rows mismatch")
    migrated: list[dict[str, Any]] = []
    for original in rows:
        row = dict(original)
        if row.get("status") != "ok":
            migrated.append(row)
            continue
        rankings = dict(row.get("rankings") or {})
        required = {
            str(value)
            for value in (row.get("required_keys") or [])
            if str(value)
        }
        suite = str(row.get("suite") or "")
        scope = "page" if suite == "mmlongbench-doc" else "artifact"
        text_ranking = list(
            rankings.get("T_page" if scope == "page" else "T_artifact")
            or rankings.get("T")
            or []
        )
        artifact_ranking = list(rankings.get("T_artifact") or [])
        page_ranking = list(rankings.get("T_page") or [])
        text_required = required if scope == "artifact" else {
            str(value)
            for value in (row.get("text_page_required_keys") or [])
            if str(value)
        }
        if scope == "page" and not text_required:
            text_required = required
        row["text_metric_scope"] = scope
        row["text_required_keys"] = sorted(text_required)
        row.setdefault("text_page_required_keys", sorted(text_required))
        # Preserve the raw rankings byte-for-byte.  The migration changes only
        # which already-recorded ranking is used by each metric; it must not
        # masquerade as a new retrieval run.
        row["metrics"] = {
            **dict(row.get("metrics") or {}),
            "T": _ranking_metrics(text_ranking, text_required),
            "T_artifact": _ranking_metrics(artifact_ranking, required),
            "T_page": (
                _ranking_metrics(page_ranking, text_required)
                if scope == "page"
                else {
                    "scored": False,
                    "reason": "source_has_no_native_page_provenance",
                }
            ),
        }
        migrated.append(row)
    result = dict(report)
    result["metric_contract_version"] = METRIC_CONTRACT_VERSION
    result["rows"] = migrated
    result["aggregate"] = _aggregate_metrics(migrated)
    result["aggregate_by_suite"] = _aggregate_by_key(migrated, "suite")
    result["aggregate_by_profile"] = _aggregate_by_key(migrated, "profile")
    result["failure_taxonomy"] = _visual_failure_taxonomy(migrated)
    result["promotion"] = visual_report_promotion_gate(result)
    result["metric_migration"] = {
        "contract": METRIC_CONTRACT_VERSION,
        "rankings_unchanged": True,
        "llm_calls": False,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = recompute(json.loads(args.input.read_text(encoding="utf-8")))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"recompute_failed: {exc}", file=sys.stderr)
        return 2
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "cases": report["cases"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
