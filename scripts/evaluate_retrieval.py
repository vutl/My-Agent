#!/usr/bin/env python3
"""Evaluate hybrid retrieval quality against a labeled question set."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
import statistics
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings
from app.db.sqlite import connect
from app.rag.embeddings import (
    HashEmbeddingProvider,
    HuggingFaceEmbeddingProvider,
    OllamaEmbeddingProvider,
)
from app.retrieval_store.lancedb_store import LanceDBRetrievalStore, LanceDBUnavailable
from app.services.rag_service import RagService


def _load_questions(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Question file must be a JSON array")
    return payload


def _hit_at_k(results: list[dict], expected_document_id: str, k: int) -> bool:
    for item in results[:k]:
        if str(item.get("document_id") or "") == expected_document_id:
            return True
    return False


def _resolve_expected_document_id(db_path: Path, item: dict) -> tuple[str, str]:
    configured = str(item.get("expected_document_id") or "").strip()
    filename_hint = str(item.get("expected_filename_contains") or "").strip()
    with connect(db_path) as connection:
        if configured:
            exact = connection.execute(
                "SELECT id FROM documents WHERE id = ?",
                (configured,),
            ).fetchone()
            if exact is not None:
                return configured, "document_id"
        if filename_hint:
            rows = connection.execute(
                """
                SELECT id
                FROM documents
                WHERE lower(filename) LIKE ?
                ORDER BY indexed_at DESC
                """,
                (f"%{filename_hint.lower()}%",),
            ).fetchall()
            if len(rows) == 1:
                return str(rows[0]["id"]), "filename_hint"
            if len(rows) > 1:
                raise ValueError(
                    f"Ambiguous expected_filename_contains={filename_hint!r}"
                )
    raise ValueError(
        f"Could not resolve evaluation label for question={item.get('question')!r}"
    )


def _first_relevant_rank(results: list[dict], expected_document_id: str) -> int | None:
    for rank, item in enumerate(results, start=1):
        if str(item.get("document_id") or "") == expected_document_id:
            return rank
    return None


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[max(index, 0)], 2)


async def _evaluate(
  questions: list[dict],
  *,
  top_k: int,
  embeddings,
  embedding_model_name: str,
  rerank: bool,
  rerank_mode: str,
  cross_encoder_model_path: Path | None,
  lancedb_path: Path | None,
) -> dict:
    settings = get_settings()
    rag = RagService(settings.sqlite_db_path, artifact_root=settings.artifacts_path)
    store = LanceDBRetrievalStore(lancedb_path or settings.lancedb_path)

    hits = {3: 0, 5: 0, top_k: 0}
    reciprocal_ranks: list[float] = []
    ndcg_scores: list[float] = []
    latencies_ms: list[float] = []
    rows: list[dict] = []
    if questions:
        # Exclude one-time model loading from foreground query latency. Ollama
        # usually remains warm after indexing; in-process HF models do not.
        await embeddings.embed_query(str(questions[0]["question"]))
    for item in questions:
        question = str(item["question"])
        expected_id, label_source = _resolve_expected_document_id(
            settings.sqlite_db_path,
            item,
        )
        started = time.perf_counter()
        hybrid = await rag.search_hybrid(
            query=question,
            top_k=top_k,
            collection_id=None,
            retrieval_store=store,
            embeddings=embeddings,
            rerank=rerank,
            rerank_mode=rerank_mode,
            cross_encoder_model_path=cross_encoder_model_path,
            rerank_max_candidates=settings.rerank_max_candidates,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        latencies_ms.append(latency_ms)
        results = hybrid.get("results") or []
        relevant_rank = _first_relevant_rank(results, expected_id)
        reciprocal_rank = 1.0 / relevant_rank if relevant_rank is not None else 0.0
        ndcg = (
            1.0 / math.log2(relevant_rank + 1)
            if relevant_rank is not None and relevant_rank <= top_k
            else 0.0
        )
        reciprocal_ranks.append(reciprocal_rank)
        ndcg_scores.append(ndcg)
        row = {
            "question": question,
            "expected_document_id": expected_id,
            "label_source": label_source,
            "top_document_ids": [str(result.get("document_id") or "") for result in results[:top_k]],
            "hit@3": _hit_at_k(results, expected_id, 3),
            "hit@5": _hit_at_k(results, expected_id, 5),
            f"hit@{top_k}": _hit_at_k(results, expected_id, top_k),
            "relevant_rank": relevant_rank,
            "reciprocal_rank": round(reciprocal_rank, 4),
            f"ndcg@{top_k}": round(ndcg, 4),
            "latency_ms": round(latency_ms, 2),
        }
        rows.append(row)
        for key in hits:
            if row.get(f"hit@{key}"):
                hits[key] += 1

    total = len(questions) or 1
    return {
        "total": len(questions),
        "embedding_model": embedding_model_name,
        "rerank_enabled": rerank,
        "rerank_mode": rerank_mode if rerank else "disabled",
        "metrics": {
            **{f"hit@{k}": round(hits[k] / total, 3) for k in hits},
            f"mrr@{top_k}": round(statistics.fmean(reciprocal_ranks), 4)
            if reciprocal_ranks
            else 0.0,
            f"ndcg@{top_k}": round(statistics.fmean(ndcg_scores), 4)
            if ndcg_scores
            else 0.0,
            "latency_p50_ms": _percentile(latencies_ms, 0.50),
            "latency_p95_ms": _percentile(latencies_ms, 0.95),
            "latency_p99_ms": _percentile(latencies_ms, 0.99),
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local hybrid retrieval.")
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "data/retrieval_eval/questions.json",
    )
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument(
        "--rerank-mode",
        choices=("embedding", "cross_encoder"),
        default=None,
    )
    parser.add_argument("--cross-encoder-path", type=Path, default=None)
    parser.add_argument("--lancedb-path", type=Path, default=None)
    parser.add_argument("--embedding-provider", choices=("ollama", "huggingface", "hash"), default="ollama")
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--query-prefix", default=None)
    parser.add_argument("--document-prefix", default=None)
    parser.add_argument("--query-task", default=None)
    parser.add_argument("--document-task", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--truncate-dim", type=int, default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--code-revision", default=None)
    parser.add_argument("--native-model", action="store_true")
    parser.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable reranking; defaults to the runtime RERANK_ENABLED setting.",
    )
    parser.add_argument(
        "--hash-embeddings",
        action="store_true",
        help="Use deterministic hash embeddings (no Ollama required).",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    questions = _load_questions(args.questions)
    settings = get_settings()
    store = LanceDBRetrievalStore(args.lancedb_path or settings.lancedb_path)
    if args.hash_embeddings or args.embedding_provider == "hash":
        embeddings = HashEmbeddingProvider(dimensions=store.vector_dimensions() or 64)
        embedding_model_name = "hash"
    elif args.embedding_provider == "huggingface":
        if not args.embedding_model:
            parser.error("--embedding-model is required for Hugging Face embeddings")
        embeddings = HuggingFaceEmbeddingProvider(
            model_name=args.embedding_model,
            device=args.device,
            batch_size=args.batch_size,
            query_prefix=args.query_prefix or "",
            document_prefix=args.document_prefix or "",
            query_task=args.query_task,
            document_task=args.document_task,
            trust_remote_code=args.trust_remote_code,
            truncate_dim=args.truncate_dim,
            revision=args.revision,
            code_revision=args.code_revision,
            native_model=args.native_model,
        )
        embedding_model_name = args.embedding_model
    else:
        embeddings = OllamaEmbeddingProvider(
            host=settings.ollama_host,
            model=args.embedding_model or settings.embedding_model,
            timeout_seconds=settings.request_timeout_seconds,
            query_prefix=(args.query_prefix if args.query_prefix is not None else settings.embedding_query_prefix),
            document_prefix=(args.document_prefix if args.document_prefix is not None else settings.embedding_document_prefix),
        )
        embedding_model_name = args.embedding_model or settings.embedding_model
    rerank_enabled = (
        args.rerank
        if args.rerank is not None
        else bool(args.rerank_mode or args.cross_encoder_path or settings.rerank_enabled)
    )
    try:
        report = asyncio.run(
            _evaluate(
                questions,
                top_k=args.top_k,
                embeddings=embeddings,
                embedding_model_name=embedding_model_name,
                rerank=rerank_enabled,
                rerank_mode=args.rerank_mode or settings.rerank_mode,
                cross_encoder_model_path=(
                    args.cross_encoder_path
                    or settings.rerank_cross_encoder_path
                ),
                lancedb_path=args.lancedb_path,
            )
        )
    except LanceDBUnavailable as exc:
        raise SystemExit(f"LanceDB unavailable: {exc}") from exc

    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    for row in report["rows"]:
        status = "OK" if row[f"hit@{args.top_k}"] else "MISS"
        print(f"[{status}] {row['question']}")
        print(f"       expected={row['expected_document_id']}")
        print(f"       got={row['top_document_ids'][:3]}")

    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
