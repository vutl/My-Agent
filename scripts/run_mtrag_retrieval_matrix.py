#!/usr/bin/env python3
"""Run the complete no-model MTRAG retrieval matrix and save compact reports."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

from evaluate_mtrag_retrieval import evaluate
from mtrag_eval_lib import DEFAULT_INDEX_PATH, DOMAINS, PROJECT_ROOT


MATRIX = {
    "human": ("lastturn", "rewrite", "questions"),
    "un": ("lastturn", "questions", "conversation"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "retrieval_eval" / "public" / "results",
    )
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument("--max-queries", type=int)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse a matching complete report when present (disabled for --max-queries).",
    )
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    matrix_rows: list[dict] = []
    started_all = time.perf_counter()
    for suite, modes in MATRIX.items():
        for mode in modes:
            print(f"running mtrag-{suite}/{mode} ...", flush=True)
            started = time.perf_counter()
            output = args.results_dir / f"mtrag-{suite}-{mode}-fts-v1.json"
            if args.reuse_existing and args.max_queries is None and output.exists():
                report = json.loads(output.read_text(encoding="utf-8"))
                print(f"reused {output.name}", flush=True)
            else:
                report = evaluate(
                    index_path=args.index,
                    suite=suite,
                    query_mode=mode,
                    domains=args.domains,
                    top_k=10,
                    max_queries=args.max_queries,
                    include_cases=False,
                )
                output.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            elapsed = time.perf_counter() - started
            metrics = report["metrics"]
            matrix_rows.append(
                {
                    "suite": report["suite"],
                    "query_mode": mode,
                    "cases": report["cases"],
                    "hit@3": metrics["hit@3"],
                    "hit@10": metrics["hit@10"],
                    "mrr@10": metrics["mrr@10"],
                    "ndcg@10": metrics["ndcg@10"],
                    "recall@10": metrics["recall@10"],
                    "p50_ms": report["latency_ms"]["p50"],
                    "p95_ms": report["latency_ms"]["p95"],
                    "elapsed_seconds": round(elapsed, 3),
                    "report": str(output.relative_to(PROJECT_ROOT)),
                }
            )
            print(
                f"done {report['cases']} cases in {elapsed:.1f}s; "
                f"Hit@10={metrics['hit@10']:.4f}",
                flush=True,
            )

    summary = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "engine": "sqlite_fts5_bm25",
        "evaluation_scope": "official_full_corpus_lexical_baseline",
        "rows": matrix_rows,
        "elapsed_seconds": round(time.perf_counter() - started_all, 3),
        "production_corpus_modified": False,
    }
    summary_path = args.results_dir / "mtrag-retrieval-matrix-fts-v1.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
