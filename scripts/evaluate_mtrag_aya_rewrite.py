#!/usr/bin/env python3
"""Evaluate Aya's real QueryRewriteService on isolated MTRAG retrieval.

Public benchmark conversation turns may be sent to the configured 9router only
for this explicit evaluation. The selected model is pinned and provider/model
errors stop the run; there is no fallback. Retrieval remains read-only against
the isolated MTRAG FTS index.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import statistics
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import APPROVED_9ROUTER_MODELS, get_settings  # noqa: E402
from app.llm.openai_client import get_llm_client  # noqa: E402
from app.services.query_rewrite_service import QueryRewriteService  # noqa: E402
from mtrag_eval_lib import (  # noqa: E402
    DEFAULT_INDEX_PATH,
    DOMAINS,
    MTRAG_ROOT,
    PUBLIC_ROOT,
    load_human_retrieval_cases,
    mean_metrics,
    read_index_manifest,
    reciprocal_rank_fusion,
    retrieval_metrics,
    search_fts,
)


DEFAULT_OUTPUT = PUBLIC_ROOT / "results" / "mtrag-human-aya-rewrite-sol-12.json"
DEFAULT_CACHE = PUBLIC_ROOT / "results" / "mtrag-human-aya-rewrite-sol-cache-v1.jsonl"
_TURN_PREFIX_RE = re.compile(r"^\|user\|\s*:\s*", re.IGNORECASE)


def _rewrite_policy_fingerprint() -> str:
    source = BACKEND_ROOT / "app" / "services" / "query_rewrite_service.py"
    return hashlib.sha256(source.read_bytes()).hexdigest()


def _ensure_public_results_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    allowed = (PUBLIC_ROOT / "results").resolve()
    if resolved == allowed or allowed not in resolved.parents:
        raise ValueError(f"MTRAG rewrite output must be below {allowed}; got {resolved}")
    return resolved


def _read_generation_tasks() -> dict[str, dict[str, Any]]:
    path = MTRAG_ROOT / "mtrag-human" / "generation_tasks" / "reference.jsonl"
    tasks: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        tasks[str(item["task_id"])] = item
    return tasks


def conversation_for_rewrite(
    task: dict[str, Any],
    *,
    expected_latest_query: str,
) -> tuple[str, list[dict[str, str]]]:
    messages = list(task.get("input") or [])
    if not messages or messages[-1].get("speaker") != "user":
        raise ValueError(f"Task {task.get('task_id')} has no latest user message")
    latest = str(messages[-1].get("text") or "").strip()
    expected = _TURN_PREFIX_RE.sub("", expected_latest_query).strip()
    if " ".join(latest.split()).casefold() != " ".join(expected.split()).casefold():
        raise ValueError(
            f"Task {task.get('task_id')} latest query does not match retrieval row"
        )
    previous: list[dict[str, str]] = []
    for message in messages[:-1]:
        speaker = str(message.get("speaker") or "")
        content = str(message.get("text") or "").strip()
        if not content:
            continue
        previous.append(
            {
                "role": "assistant" if speaker == "agent" else "user",
                "content": content,
            }
        )
    return latest, previous


def _select_cases(cases: list[dict[str, Any]], *, per_domain: int) -> list[dict[str, Any]]:
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_domain[str(case["domain"])].append(case)
    selected: list[dict[str, Any]] = []
    for domain in DOMAINS:
        # Stable hash order avoids selecting only the earliest conversations
        # while remaining exactly reproducible across machines.
        ordered = sorted(
            by_domain[domain],
            key=lambda item: hashlib.sha256(
                str(item["query_id"]).encode("utf-8")
            ).hexdigest(),
        )
        selected.extend(ordered[:per_domain])
    return selected


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        rows[str(item["query_id"])] = item
    return rows


def _append_cache(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return round(ordered[index], 3)


async def evaluate(
    *,
    index_path: Path,
    cache_path: Path,
    per_domain: int,
    top_k: int,
    model: str,
) -> dict[str, Any]:
    if model not in APPROVED_9ROUTER_MODELS:
        raise ValueError(f"Refusing unapproved rewrite model: {model}")
    manifest = read_index_manifest(index_path)
    lastturn_cases = load_human_retrieval_cases(query_mode="lastturn", domains=DOMAINS)
    official_rewrite = {
        case["query_id"]: case["query"]
        for case in load_human_retrieval_cases(query_mode="rewrite", domains=DOMAINS)
    }
    cases = _select_cases(lastturn_cases, per_domain=per_domain)
    generation_tasks = _read_generation_tasks()
    cache_path = _ensure_public_results_path(cache_path)
    cached = _load_cache(cache_path)
    policy_fingerprint = _rewrite_policy_fingerprint()

    settings = get_settings()
    if settings.llm_provider != "openai_compatible":
        raise RuntimeError("Aya rewrite eval requires configured 9router/openai_compatible")
    client = get_llm_client(
        provider=settings.llm_provider,
        ollama_host=settings.ollama_host,
        openai_api_base=settings.openai_api_base,
        openai_api_key=settings.openai_api_key,
        timeout_seconds=settings.request_timeout_seconds,
    )
    service = QueryRewriteService(client=client, default_model=model)

    rewrites: dict[str, dict[str, Any]] = {}
    for case in cases:
        query_id = str(case["query_id"])
        cached_row = cached.get(query_id)
        if (
            cached_row
            and cached_row.get("model") == model
            and cached_row.get("rewrite_policy_fingerprint") == policy_fingerprint
        ):
            rewrites[query_id] = cached_row
            continue
        task = generation_tasks.get(query_id)
        if task is None:
            raise ValueError(f"Missing generation task for {query_id}")
        latest, previous = conversation_for_rewrite(
            task,
            expected_latest_query=case["query"],
        )
        started = time.perf_counter()
        result = await service.rewrite(
            query=latest,
            previous_messages=previous,
            model=model,
        )
        row = {
            "query_id": query_id,
            "model": model,
            "rewrite_policy_fingerprint": policy_fingerprint,
            "standalone_query": result.standalone_query,
            "rewrite_used": result.rewrite_used,
            "diagnostic_reason": result.diagnostics.get("reason"),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        _append_cache(cache_path, row)
        cached[query_id] = row
        rewrites[query_id] = row

    result_rows: list[dict[str, Any]] = []
    retrieval_latency: dict[str, list[float]] = defaultdict(list)
    with closing(sqlite3.connect(f"file:{index_path.resolve()}?mode=ro", uri=True)) as index:
        for case in cases:
            query_id = str(case["query_id"])
            queries = {
                "lastturn": case["query"],
                "aya_rewrite": rewrites[query_id]["standalone_query"],
                "official_rewrite": official_rewrite[query_id],
            }
            metrics: dict[str, dict[str, float]] = {}
            ranked_ids: dict[str, list[str]] = {}
            for name, query in queries.items():
                started = time.perf_counter()
                hits = search_fts(
                    index,
                    domain=case["domain"],
                    query=query,
                    top_k=top_k,
                )
                retrieval_latency[name].append((time.perf_counter() - started) * 1000)
                ranked_ids[name] = [hit["passage_id"] for hit in hits]
                metrics[name] = retrieval_metrics(
                    ranked_ids[name],
                    set(case["relevant_passage_ids"]),
                    cutoffs=(1, 3, 5, 10),
                )
            dual_ids = reciprocal_rank_fusion(
                [ranked_ids["lastturn"], ranked_ids["aya_rewrite"]]
            )
            metrics["aya_dual_rrf"] = retrieval_metrics(
                dual_ids,
                set(case["relevant_passage_ids"]),
                cutoffs=(1, 3, 5, 10),
            )
            result_rows.append(
                {
                    "query_id": query_id,
                    "domain": case["domain"],
                    "queries": queries,
                    "rewrite_used": rewrites[query_id]["rewrite_used"],
                    "diagnostic_reason": rewrites[query_id]["diagnostic_reason"],
                    "metrics": metrics,
                }
            )

    rewrite_latencies = [float(rewrites[case["query_id"]]["latency_ms"]) for case in cases]
    methods = ("lastturn", "aya_rewrite", "aya_dual_rrf", "official_rewrite")
    return {
        "schema_version": 1,
        "ok": True,
        "suite": "mtrag-human",
        "cases": len(cases),
        "cases_per_domain": per_domain,
        "model": model,
        "engine": "aya_query_rewrite_then_isolated_fts5_bm25",
        "index_manifest": manifest,
        "metrics": {
            method: mean_metrics(row["metrics"][method] for row in result_rows)
            for method in methods
        },
        "rewrite_policy": {
            "llm_rewrite_used": sum(row["rewrite_used"] for row in result_rows),
            "deterministic_or_direct": sum(not row["rewrite_used"] for row in result_rows),
            "reasons": {
                reason: sum(row["diagnostic_reason"] == reason for row in result_rows)
                for reason in sorted({str(row["diagnostic_reason"]) for row in result_rows})
            },
        },
        "latency_ms": {
            "rewrite_p50": round(statistics.median(rewrite_latencies), 3),
            "rewrite_p95": _percentile(rewrite_latencies, 0.95),
            "retrieval": {
                method: {
                    "p50": round(statistics.median(retrieval_latency[method]), 3),
                    "p95": _percentile(retrieval_latency[method], 0.95),
                }
                for method in ("lastturn", "aya_rewrite", "official_rewrite")
            },
        },
        "cache": str(cache_path),
        "rewrite_policy_fingerprint": policy_fingerprint,
        "selection": "stable_sha256_order_with_equal_per_domain_quota",
        "evaluation_scope": "bounded_public_conversation_rewrite_diagnostic",
        "case_results": result_rows,
        "production_corpus_modified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-domain", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--model", default="cx/gpt-5.6-sol")
    args = parser.parse_args()
    output_path = _ensure_public_results_path(args.output)
    report = asyncio.run(
        evaluate(
            index_path=args.index,
            cache_path=args.cache,
            per_domain=max(1, args.per_domain),
            top_k=max(10, args.top_k),
            model=args.model,
        )
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
