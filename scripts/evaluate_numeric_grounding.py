#!/usr/bin/env python3
"""Replay reviewed numeric claims without any LLM calls or database writes.

This is a regression measurement, not a blind retrieval/agent benchmark.
Optional cached facts are revalidated against each source by the guard.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.services.evidence_validator import validate_answer_claims  # noqa: E402


def evaluate(suite: dict, facts: dict | None = None) -> dict:
    if suite.get("split") != "regression-visible" or not suite.get("not_blind"):
        raise ValueError("Expected an explicitly visible regression suite")
    rows, ids = [], set()
    for case in suite["cases"]:
        case_id = case["case_id"]
        if case_id in ids or not isinstance(case["expected"], bool):
            raise ValueError("Duplicate case or non-boolean expectation")
        ids.add(case_id)
        language, layout, documents = suite["scenarios"][case["scenario"]]
        documents = [{**d, "facts": (facts or {}).get(d["document_id"], [])} for d in documents]
        start = time.perf_counter()
        result = validate_answer_claims(answer=case["answer"], documents=documents)
        rows.append({"case_id": case_id, "scenario": case["scenario"], "language": language,
            "category": case["category"], "evidence_layout": layout, "expected": case["expected"],
            "got": result.valid, "passed": result.valid == case["expected"],
            "latency_ms": round((time.perf_counter() - start) * 1000, 3), "validation": result.to_dict()})

    def summarize(selected):
        correct = [r for r in selected if r["expected"]]
        wrong = [r for r in selected if not r["expected"]]
        return {"cases": len(selected), "correct_accepted": sum(r["got"] for r in correct),
            "correct_total": len(correct), "wrong_rejected": sum(not r["got"] for r in wrong),
            "wrong_total": len(wrong), "false_accepts": sum(r["got"] for r in wrong),
            "false_rejects": sum(not r["got"] for r in correct)}

    slices = {}
    for field in ("language", "category", "evidence_layout"):
        grouped = defaultdict(list)
        for row in rows:
            grouped[row[field]].append(row)
        slices[field] = {key: summarize(values) for key, values in sorted(grouped.items())}
    paired = defaultdict(dict)
    for row in rows:
        if row["case_id"].endswith(("_w", "_c")):
            paired[row["case_id"][:-2]][row["case_id"][-1]] = row
    pairs = [v for v in paired.values() if "w" in v and "c" in v]
    return {"created_at": datetime.now(UTC).isoformat(), "suite_id": suite["suite_id"],
        "evaluation_type": "visible_numeric_regression", "valid_for_blind_promotion": False,
        "llm_calls": 0, "cached_facts": facts is not None,
        "summary": summarize(rows), "paired": {"total": len(pairs),
            "both_pass": sum(v["w"]["passed"] and v["c"]["passed"] for v in pairs)},
        "slices": slices, "cases": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=ROOT / "evals/numeric_grounding/regression-claude-v1.json")
    parser.add_argument("--cached-facts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite_bytes = args.suite.read_bytes()
    facts_bytes = args.cached_facts.read_bytes() if args.cached_facts else None
    report = evaluate(json.loads(suite_bytes), json.loads(facts_bytes) if facts_bytes else None)
    report["suite_sha256"] = hashlib.sha256(suite_bytes).hexdigest()
    report["facts_sha256"] = hashlib.sha256(facts_bytes).hexdigest() if facts_bytes else None
    report["code_fingerprints"] = {name: hashlib.sha256((ROOT / "backend/app/services" / name).read_bytes()).hexdigest()
        for name in ("numeric_grounding.py", "evidence_validator.py", "fact_extraction_service.py")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "paired": report["paired"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
