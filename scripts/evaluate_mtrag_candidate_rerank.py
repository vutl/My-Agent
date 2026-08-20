#!/usr/bin/env python3
"""Bounded, isolated MTRAG BM25-candidate rerank with local EmbeddingGemma.

This is deliberately a two-stage diagnostic, not a full dense-index score:
semantic ranking can only reorder passages that BM25 placed in ``candidate_k``.
The script never injects qrels into the candidate set and only writes its cache
below ``data/retrieval_eval/public/indexes``.
"""

from __future__ import annotations

import argparse
from array import array
import asyncio
from collections import defaultdict
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics
import sys
import time
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.rag.embeddings import OllamaEmbeddingProvider  # noqa: E402
from mtrag_eval_lib import (  # noqa: E402
    DEFAULT_INDEX_PATH,
    DOMAINS,
    PUBLIC_ROOT,
    ensure_isolated_index_path,
    fetch_passages,
    load_human_retrieval_cases,
    load_un_retrieval_cases,
    mean_metrics,
    read_index_manifest,
    retrieval_metrics,
    search_fts,
)


DEFAULT_CACHE_PATH = PUBLIC_ROOT / "indexes" / "mtrag-embeddinggemma-candidates-v1.sqlite"
QUERY_PREFIX = "task: search result | query: "
DOCUMENT_PREFIX = "title: none | text: "


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return round(ordered[index], 3)


def _distribution(values: list[float]) -> dict[str, float | None]:
    return {
        "p50": round(statistics.median(values), 3) if values else None,
        "p95": _percentile(values, 0.95),
        "max": round(max(values), 3) if values else None,
    }


def _vector_blob(vector: list[float]) -> bytes:
    return array("f", vector).tobytes()


def _blob_vector(blob: bytes) -> list[float]:
    values = array("f")
    values.frombytes(blob)
    return values.tolist()


def _dot(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"Embedding dimension mismatch: {len(left)} != {len(right)}")
    return sum(a * b for a, b in zip(left, right, strict=True))


def rank_candidate_ids(
    lexical_ids: list[str],
    *,
    query_vector: list[float],
    document_vectors: dict[str, list[float]],
    rrf_k: int = 60,
) -> tuple[list[str], list[str]]:
    lexical_rank = {
        passage_id: rank for rank, passage_id in enumerate(lexical_ids, 1)
    }
    semantic_ids = sorted(
        lexical_ids,
        key=lambda passage_id: (
            -_dot(query_vector, document_vectors[passage_id]),
            lexical_rank[passage_id],
        ),
    )
    semantic_rank = {passage_id: rank for rank, passage_id in enumerate(semantic_ids, 1)}
    hybrid_ids = sorted(
        lexical_ids,
        key=lambda passage_id: (
            -(
                1.0 / (rrf_k + lexical_rank[passage_id])
                + 1.0 / (rrf_k + semantic_rank[passage_id])
            ),
            lexical_rank[passage_id],
        ),
    )
    return semantic_ids, hybrid_ids


class CandidateEmbeddingCache:
    def __init__(self, path: Path, *, fingerprint: str) -> None:
        self.path = ensure_isolated_index_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS candidate_embeddings (
                fingerprint TEXT NOT NULL,
                passage_id TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                vector BLOB NOT NULL,
                PRIMARY KEY (fingerprint, passage_id)
            )
            """
        )
        self.fingerprint = fingerprint

    def close(self) -> None:
        self.connection.close()

    def get_many(self, passage_ids: Iterable[str]) -> dict[str, list[float]]:
        ordered_ids = list(dict.fromkeys(passage_ids))
        found: dict[str, list[float]] = {}
        for offset in range(0, len(ordered_ids), 800):
            batch = ordered_ids[offset : offset + 800]
            placeholders = ",".join("?" for _ in batch)
            rows = self.connection.execute(
                f"SELECT passage_id, dimensions, vector FROM candidate_embeddings "
                f"WHERE fingerprint = ? AND passage_id IN ({placeholders})",
                [self.fingerprint, *batch],
            ).fetchall()
            for passage_id, dimensions, blob in rows:
                vector = _blob_vector(blob)
                if len(vector) != int(dimensions):
                    raise ValueError(f"Corrupt cached embedding for {passage_id}")
                found[str(passage_id)] = vector
        return found

    def put_many(self, rows: Iterable[tuple[str, list[float]]]) -> None:
        self.connection.executemany(
            """
            INSERT OR REPLACE INTO candidate_embeddings(
                fingerprint, passage_id, dimensions, vector
            ) VALUES (?, ?, ?, ?)
            """,
            [
                (self.fingerprint, passage_id, len(vector), _vector_blob(vector))
                for passage_id, vector in rows
            ],
        )
        self.connection.commit()


def _select_cases(
    cases: list[dict[str, Any]],
    *,
    per_domain: int,
    full: bool,
) -> list[dict[str, Any]]:
    if full:
        return cases
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    for case in cases:
        domain = str(case["domain"])
        if counts[domain] >= per_domain:
            continue
        selected.append(case)
        counts[domain] += 1
    return selected


def _fingerprint(
    *,
    model: str,
    document_prefix: str,
    max_chars: int,
    source_sha256: dict[str, str],
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "document_prefix": document_prefix,
            "max_chars": max_chars,
            "source_sha256": source_sha256,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def evaluate(
    *,
    index_path: Path,
    cache_path: Path,
    suite: str,
    query_mode: str,
    domains: list[str],
    per_domain: int,
    full: bool,
    candidate_k: int,
    top_k: int,
    ollama_host: str,
    model: str,
    query_prefix: str,
    document_prefix: str,
    max_document_chars: int,
    batch_size: int,
) -> dict[str, Any]:
    manifest = read_index_manifest(index_path)
    cases = (
        load_human_retrieval_cases(query_mode=query_mode, domains=domains)
        if suite == "human"
        else load_un_retrieval_cases(query_mode=query_mode, domains=domains)
    )
    cases = _select_cases(cases, per_domain=per_domain, full=full)
    if not cases:
        raise ValueError("No MTRAG cases selected")

    lexical_ms: list[float] = []
    candidate_ids_by_query: dict[str, list[str]] = {}
    with closing(sqlite3.connect(f"file:{index_path.resolve()}?mode=ro", uri=True)) as index:
        for case in cases:
            started = time.perf_counter()
            hits = search_fts(
                index,
                domain=case["domain"],
                query=case["query"],
                top_k=candidate_k,
            )
            lexical_ms.append((time.perf_counter() - started) * 1000)
            candidate_ids_by_query[case["query_id"]] = [
                hit["passage_id"] for hit in hits
            ]
        all_candidate_ids = list(
            dict.fromkeys(
                passage_id
                for ids in candidate_ids_by_query.values()
                for passage_id in ids
            )
        )
        passages = fetch_passages(index, all_candidate_ids)

    missing_passages = set(all_candidate_ids) - set(passages)
    if missing_passages:
        raise ValueError(f"Index failed to resolve {len(missing_passages)} candidate passages")

    fingerprint = _fingerprint(
        model=model,
        document_prefix=document_prefix,
        max_chars=max_document_chars,
        source_sha256=dict(manifest.get("source_sha256") or {}),
    )
    cache = CandidateEmbeddingCache(cache_path, fingerprint=fingerprint)
    try:
        document_vectors = cache.get_many(all_candidate_ids)
        cached_count = len(document_vectors)
        missing_ids = [value for value in all_candidate_ids if value not in document_vectors]
        provider = OllamaEmbeddingProvider(
            host=ollama_host,
            model=model,
            timeout_seconds=180.0,
            query_prefix=query_prefix,
            document_prefix=document_prefix,
        )
        embed_documents_started = time.perf_counter()
        for offset in range(0, len(missing_ids), batch_size):
            batch_ids = missing_ids[offset : offset + batch_size]
            texts = [
                (f"{passages[passage_id]['title']}\n{passages[passage_id]['text']}")[
                    :max_document_chars
                ]
                for passage_id in batch_ids
            ]
            vectors = await provider.embed_texts(texts)
            rows = list(zip(batch_ids, vectors, strict=True))
            cache.put_many(rows)
            document_vectors.update(rows)
        document_embedding_ms = (time.perf_counter() - embed_documents_started) * 1000

        query_ms: list[float] = []
        query_vectors: dict[str, list[float]] = {}
        # Warm model startup outside per-query latency accounting.
        await provider.embed_query(str(cases[0]["query"]))
        for case in cases:
            started = time.perf_counter()
            query_vectors[case["query_id"]] = await provider.embed_query(case["query"])
            query_ms.append((time.perf_counter() - started) * 1000)
    finally:
        cache.close()

    result_rows: list[dict[str, Any]] = []
    for case in cases:
        lexical_ids = candidate_ids_by_query[case["query_id"]]
        semantic_ids, hybrid_ids = rank_candidate_ids(
            lexical_ids,
            query_vector=query_vectors[case["query_id"]],
            document_vectors=document_vectors,
        )
        relevant = set(case["relevant_passage_ids"])
        result_rows.append(
            {
                "query_id": case["query_id"],
                "domain": case["domain"],
                "candidate_recall": bool(relevant.intersection(lexical_ids)),
                "bm25": retrieval_metrics(lexical_ids, relevant, cutoffs=(1, 3, 5, 10)),
                "embedding_rerank": retrieval_metrics(
                    semantic_ids, relevant, cutoffs=(1, 3, 5, 10)
                ),
                "hybrid_rrf": retrieval_metrics(hybrid_ids, relevant, cutoffs=(1, 3, 5, 10)),
            }
        )

    metric_names = ("bm25", "embedding_rerank", "hybrid_rrf")
    return {
        "schema_version": 1,
        "ok": True,
        "suite": f"mtrag-{suite}",
        "query_mode": query_mode,
        "cases": len(cases),
        "domains": domains,
        "engine": "fts5_bm25_candidate_then_local_embeddinggemma",
        "candidate_k": candidate_k,
        "top_k": top_k,
        "model": model,
        "query_prefix": query_prefix,
        "document_prefix": document_prefix,
        "index_manifest": manifest,
        "metrics": {
            name: mean_metrics(row[name] for row in result_rows)
            for name in metric_names
        },
        "candidate_recall": round(
            sum(row["candidate_recall"] for row in result_rows) / len(result_rows),
            6,
        ),
        "cache": {
            "path": str(cache_path),
            "unique_candidates": len(all_candidate_ids),
            "hits": cached_count,
            "embedded": len(missing_ids),
            "fingerprint": fingerprint,
        },
        "latency_ms": {
            "lexical_query": _distribution(lexical_ms),
            "embedding_query": _distribution(query_ms),
            "document_embedding_total": round(document_embedding_ms, 3),
        },
        "evaluation_scope": (
            "full_bm25_candidate_rerank" if full else "bounded_stratified_candidate_rerank"
        ),
        "limitations": [
            "Not a full dense index; semantic retrieval cannot recover passages outside BM25 candidate_k.",
            "Reference qrels are used only for scoring and are never injected into candidates.",
            "Cold document embedding time is reported separately from warm query latency.",
        ],
        "production_corpus_modified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--suite", choices=("human", "un"), default="human")
    parser.add_argument("--query-mode")
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument("--per-domain", type=int, default=10)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="embeddinggemma:300m")
    parser.add_argument("--query-prefix", default=QUERY_PREFIX)
    parser.add_argument("--document-prefix", default=DOCUMENT_PREFIX)
    parser.add_argument("--max-document-chars", type=int, default=8_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    query_mode = args.query_mode or ("rewrite" if args.suite == "human" else "questions")
    report = asyncio.run(
        evaluate(
            index_path=args.index,
            cache_path=args.cache,
            suite=args.suite,
            query_mode=query_mode,
            domains=args.domains,
            per_domain=max(1, args.per_domain),
            full=args.full,
            candidate_k=max(args.top_k, args.candidate_k),
            top_k=max(10, args.top_k),
            ollama_host=args.ollama_host,
            model=args.model,
            query_prefix=args.query_prefix,
            document_prefix=args.document_prefix,
            max_document_chars=max(1_000, args.max_document_chars),
            batch_size=max(1, args.batch_size),
        )
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
