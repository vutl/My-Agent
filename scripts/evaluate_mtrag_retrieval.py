#!/usr/bin/env python3
"""Evaluate the isolated MTRAG FTS baseline with official qrels, without models."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import statistics
import sys
import time
from typing import Any

from mtrag_eval_lib import (
    DEFAULT_INDEX_PATH,
    DOMAINS,
    load_human_retrieval_cases,
    load_un_retrieval_cases,
    mean_metrics,
    read_index_manifest,
    retrieval_metrics,
    search_fts,
)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return round(ordered[index], 3)


def evaluate(
    *,
    index_path: Path,
    suite: str,
    query_mode: str,
    domains: list[str],
    top_k: int,
    max_queries: int | None,
    include_cases: bool,
) -> dict[str, Any]:
    manifest = read_index_manifest(index_path)
    missing_domains = set(domains) - set(manifest.get("domains") or [])
    if missing_domains:
        raise ValueError(f"Index does not contain domains: {sorted(missing_domains)}")
    if suite == "human":
        cases = load_human_retrieval_cases(query_mode=query_mode, domains=domains)
    else:
        cases = load_un_retrieval_cases(query_mode=query_mode, domains=domains)
    if max_queries is not None:
        cases = cases[: max(0, max_queries)]

    case_results: list[dict[str, Any]] = []
    with closing(sqlite3.connect(f"file:{index_path.resolve()}?mode=ro", uri=True)) as connection:
        for case in cases:
            started = time.perf_counter()
            hits = search_fts(
                connection,
                domain=case["domain"],
                query=case["query"],
                top_k=top_k,
            )
            latency_ms = (time.perf_counter() - started) * 1000
            ranked_ids = [hit["passage_id"] for hit in hits]
            metrics = retrieval_metrics(
                ranked_ids,
                set(case["relevant_passage_ids"]),
                cutoffs=(1, 3, 5, 10),
            )
            case_results.append(
                {
                    "query_id": case["query_id"],
                    "domain": case["domain"],
                    "metrics": metrics,
                    "latency_ms": round(latency_ms, 3),
                    "relevant_count": len(case["relevant_passage_ids"]),
                    "retrieved_passage_ids": ranked_ids,
                }
            )

    by_domain_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    for result in case_results:
        by_domain_rows[result["domain"]].append(result["metrics"])
    latencies = [float(result["latency_ms"]) for result in case_results]
    report: dict[str, Any] = {
        "schema_version": 1,
        "ok": bool(case_results),
        "suite": f"mtrag-{suite}",
        "query_mode": query_mode,
        "engine": "sqlite_fts5_bm25",
        "index": str(index_path),
        "index_manifest": manifest,
        "cases": len(case_results),
        "metrics": mean_metrics(result["metrics"] for result in case_results),
        "by_domain": {
            domain: {
                "cases": len(rows),
                "metrics": mean_metrics(rows),
            }
            for domain, rows in sorted(by_domain_rows.items())
        },
        "latency_ms": {
            "p50": round(statistics.median(latencies), 3) if latencies else None,
            "p95": _percentile(latencies, 0.95),
            "max": round(max(latencies), 3) if latencies else None,
        },
        "evaluation_scope": "official_full_corpus_lexical_baseline",
        "production_corpus_modified": False,
    }
    if include_cases:
        report["case_results"] = case_results
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--suite", choices=("human", "un"), default="human")
    parser.add_argument("--query-mode")
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--include-cases", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    query_mode = args.query_mode or ("rewrite" if args.suite == "human" else "questions")
    report = evaluate(
        index_path=args.index,
        suite=args.suite,
        query_mode=query_mode,
        domains=args.domains,
        top_k=max(10, args.top_k),
        max_queries=args.max_queries,
        include_cases=args.include_cases,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
