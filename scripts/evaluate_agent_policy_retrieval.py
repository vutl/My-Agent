#!/usr/bin/env python3
"""Measure Aya's deterministic retrieval orchestration without an LLM call.

This complements ``evaluate_dual_query_retrieval.py``.  The dual-query runner
measures the primitive RagService path; this runner measures the policy used by
the agent for multi-document comparison and multi-facet paper questions:
document-scoped branches and bounded facet branches are retrieved in parallel,
then merged with a deterministic round-robin quota before final top-k scoring.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for path in (BACKEND_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.core.config import get_settings  # noqa: E402
from app.db.sqlite import connect  # noqa: E402
from app.rag.paper_facets import facet_query_terms, requested_paper_facets  # noqa: E402
from app.rag.embeddings import EmbeddingError, OllamaEmbeddingProvider  # noqa: E402
from app.retrieval_store.lancedb_store import LanceDBRetrievalStore, LanceDBUnavailable  # noqa: E402
from app.services.rag_service import RagService  # noqa: E402
from evaluate_dual_query_retrieval import (  # noqa: E402
    CountingEmbeddings,
    EvaluationContractError,
    _aggregate,
    _failure_taxonomy,
    _index_fingerprint,
    _length_bucket,
    _runtime_document_scope,
    _run_final_stage,
    load_cases,
    resolve_case,
    score_results,
    suite_quality_diagnostics,
)


def _identity(item: dict[str, Any]) -> tuple[str, str] | None:
    for field in ("table_id", "figure_id", "chunk_id", "parent_chunk_id"):
        value = str(item.get(field) or "").strip()
        if value:
            return field, value
    return None


def _round_robin(branches: list[list[dict[str, Any]]], limit: int) -> list[dict[str, Any]]:
    """Merge ranked branches while reserving slots for every branch."""

    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    depth = 0
    while len(merged) < limit and any(depth < len(branch) for branch in branches):
        for branch in branches:
            if depth >= len(branch):
                continue
            item = dict(branch[depth])
            identity = _identity(item)
            if identity is not None and identity in seen:
                continue
            if identity is not None:
                seen.add(identity)
            merged.append(item)
            if len(merged) >= limit:
                break
        depth += 1
    return merged


def _branch_specs(case, document_ids: list[str] | None) -> list[tuple[str, str, list[str]]]:
    category = str(case.category)
    scoped = list(dict.fromkeys(document_ids or []))
    if category == "cross_document_compare" and len(scoped) >= 2:
        return [(f"document:{document_id}", case.standalone_full, [document_id]) for document_id in scoped]
    if category == "intra_document_multihop" and scoped:
        facets = requested_paper_facets(
            case.standalone_full,
            answer_intent="research",
            focused_document_count=1,
        )
        return [
            (f"facet:{facet}", f"{case.standalone_full} {facet_query_terms([facet])}", scoped)
            for facet in facets[:3]
        ]
    return [("single", case.standalone_full, scoped or None)]


def _serializable_quality(cases) -> dict[str, Any]:
    quality = suite_quality_diagnostics(cases)
    quality["language_context_matrix"] = {
        f"{language}:{'contextual' if contextual else 'standalone'}": count
        for (language, contextual), count in quality["language_context_matrix"].items()
    }
    return quality


async def _retrieve_policy_case(
    *,
    rag: RagService,
    store: LanceDBRetrievalStore,
    embeddings: CountingEmbeddings,
    settings: Any,
    case,
    document_ids: list[str] | None,
) -> tuple[dict[str, Any], float]:
    specs = _branch_specs(case, document_ids)
    started = time.perf_counter()

    async def run(spec: tuple[str, str, list[str] | None]) -> tuple[str, dict[str, Any]]:
        origin, query, scope = spec
        payload = await rag.search_hybrid(
            query=query,
            top_k=30,
            document_ids=scope,
            retrieval_store=store,
            embeddings=embeddings,
            rerank=False,
            rerank_mode=settings.rerank_mode,
            cross_encoder_model_path=settings.rerank_cross_encoder_path,
            rerank_max_candidates=settings.rerank_max_candidates,
            visual_boost=case.category == "figure_page",
            candidate_document_union=False,
        )
        return origin, payload

    results = await asyncio.gather(*(run(spec) for spec in specs))
    branch_results = [payload for _origin, payload in results]
    branch_candidates = [
        list(payload.get("candidate_results") or payload.get("results") or [])
        for payload in branch_results
    ]
    candidates = _round_robin(branch_candidates, 30) if len(branch_candidates) > 1 else branch_candidates[0]
    channel_candidates = {
        "fts": [item for payload in branch_results for item in (payload.get("channel_candidates") or {}).get("fts", [])],
        "vector": [item for payload in branch_results for item in (payload.get("channel_candidates") or {}).get("vector", [])],
    }
    payload = {
        "query": case.standalone_full,
        "queries": [query for _origin, query, _scope in specs],
        "query_origins": [origin for origin, _query, _scope in specs],
        "candidate_results": candidates,
        "results": candidates,
        "selected_document_ids": list(dict.fromkeys(
            document_id
            for branch in branch_results
            for document_id in branch.get("selected_document_ids") or []
        )),
        "channel_candidates": channel_candidates,
        "branch_results": [
            {
                "origin": origin,
                "results": list(branch.get("candidate_results") or branch.get("results") or []),
                "selected_document_ids": list(branch.get("selected_document_ids") or []),
                "channel_candidates": dict(branch.get("channel_candidates") or {}),
            }
            for (origin, _query, _scope), branch in zip(specs, branch_results, strict=True)
        ],
        "policy_branches": [origin for origin, _query, _scope in specs],
        "policy_mode": "per_document" if case.category == "cross_document_compare" and len(specs) > 1 else "facet_parallel" if len(specs) > 1 else "single",
        "timings_ms": {
            "branches": max((float((branch.get("timings_ms") or {}).get("total") or 0.0) for branch in branch_results), default=0.0),
            "total": round((time.perf_counter() - started) * 1000, 3),
        },
    }
    return payload, round((time.perf_counter() - started) * 1000, 3)


async def evaluate_policy(cases, *, db_path: Path, lancedb_path: Path, artifact_root: Path) -> dict[str, Any]:
    settings = get_settings()
    with connect(db_path) as connection:
        resolved = [resolve_case(connection, case) for case in cases]
    store = LanceDBRetrievalStore(lancedb_path)
    rag = RagService(db_path, artifact_root=artifact_root)
    provider = CountingEmbeddings(
        OllamaEmbeddingProvider(
            host=settings.ollama_host,
            model=settings.embedding_model,
            timeout_seconds=settings.request_timeout_seconds,
            query_prefix=settings.embedding_query_prefix,
            document_prefix=settings.embedding_document_prefix,
        )
    )
    await provider.embed_query(resolved[0].case.raw_query)
    rows: list[dict[str, Any]] = []
    for resolved_case in resolved:
        document_ids, scope_source = _runtime_document_scope(rag, resolved_case)
        before = provider.query_calls
        payload, latency = await _retrieve_policy_case(
            rag=rag,
            store=store,
            embeddings=provider,
            settings=settings,
            case=resolved_case.case,
            document_ids=document_ids,
        )
        candidates = list(payload.get("candidate_results") or [])[:30]
        candidate_score = score_results(resolved_case, candidates, cutoff=30)
        before_final = provider.query_calls
        final, final_ms = await _run_final_stage(
            query=resolved_case.case.standalone_full,
            candidates=candidates,
            embeddings=provider,
            settings=settings,
            required_document_ids=list(resolved_case.document_ids),
        )
        final_score = score_results(resolved_case, final, cutoff=10)
        payload["scope_leak"] = list(final_score["forbidden_documents"])
        rows.append(
            {
                "case_id": resolved_case.case.case_id,
                "category": resolved_case.case.category,
                "language": resolved_case.case.language,
                "positive": resolved_case.case.positive,
                "scope": resolved_case.case.scope,
                "effective_scope_source": scope_source,
                "contextual": bool(resolved_case.case.conversation_context),
                "policy_mode": payload["policy_mode"],
                "policy_branches": payload["policy_branches"],
                "candidate_recall": candidate_score["evidence_recall"],
                "evidence_recall": final_score["evidence_recall"],
                "document_recall": final_score["document_recall"],
                "multi_paper_coverage": final_score["multi_paper_coverage"],
                "modality_recall": final_score["modality_recall"],
                "hit_at_1": final_score["hit_at_1"],
                "hit_at_3": final_score["hit_at_3"],
                "hit_at_5": final_score["hit_at_5"],
                "hit_at_10": final_score["hit_at_10"],
                "mrr_at_10": final_score["mrr_at_10"],
                "ndcg_at_10": final_score["ndcg_at_10"],
                "forbidden_documents": final_score["forbidden_documents"],
                "failure_taxonomy": _failure_taxonomy(
                    resolved_case,
                    payload=payload,
                    candidates=candidates,
                    final_results=final,
                    rerank_applied=bool(settings.rerank_enabled),
                ),
                "timings_ms": {**payload["timings_ms"], "runner_retrieval": latency, "runner_final_stage": final_ms, "total": round(latency + final_ms, 3)},
                "latency_ms": round(latency + final_ms, 3),
                "embedding_query_calls": provider.query_calls - before,
                "retrieval_embedding_query_calls": before_final - before,
                "rerank_embedding_query_calls": provider.query_calls - before_final,
            }
        )
    metrics = _aggregate(rows)
    return {
        "schema_version": "agent-policy-retrieval-v1",
        "status": "draft_contract_smoke" if any(case.curation_status != "audited" for case in cases) else "audited_evaluation",
        "suite_quality": _serializable_quality(cases),
        "metrics": metrics,
        "rows": rows,
        "qrels_resolved": len(resolved),
        "model_calls": 0,
        "index_fingerprint": _index_fingerprint(db_path),
        "embedding_model": settings.embedding_model,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=PROJECT_ROOT / "evals/retrieval_v2/dev-v2.2.proposed.jsonl")
    parser.add_argument("--db-path", type=Path, required=True)
    parser.add_argument("--lancedb-path", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=PROJECT_ROOT / "data/artifacts")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-draft", action="store_true")
    args = parser.parse_args()
    cases = load_cases(args.suite, require_audited=not args.allow_draft)
    try:
        report = asyncio.run(evaluate_policy(cases, db_path=args.db_path, lancedb_path=args.lancedb_path, artifact_root=args.artifact_root))
    except LanceDBUnavailable as exc:
        raise SystemExit(f"LanceDB unavailable: {exc}") from exc
    except EmbeddingError as exc:
        raise SystemExit(f"Embedding provider unavailable: {exc}") from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
