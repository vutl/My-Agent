#!/usr/bin/env python3
"""Evaluate text, ColSmol visual, and late-fused retrieval without an LLM.

The public documents/artifacts are indexed in full before gold page/artifact
annotations are consulted.  The runner uses Aya's isolated production parser,
EmbeddingGemma/Lance path for T and the revision-scoped ColSmol index for V.
It never writes production state and never calls 9router.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterable

from evaluation_checkpoint import load_completed_checkpoint, write_checkpoint


def _visual_evaluator_fingerprint() -> str:
    """Fingerprint visual evaluator and shared public-case adapter code."""

    from evaluator_fingerprint import default_rag_dependency_files, fingerprint_files

    root = Path(__file__).resolve().parents[1]
    return fingerprint_files(
        root,
        [
            Path(__file__).resolve(),
            Path(__file__).with_name("public_multimodal_eval_lib.py").resolve(),
            Path(__file__).with_name("evaluate_public_multimodal_three_mode.py").resolve(),
            Path(__file__).with_name("evaluation_checkpoint.py").resolve(),
            *default_rag_dependency_files(root),
        ],
    )


VISUAL_EVALUATOR_FINGERPRINT = _visual_evaluator_fingerprint()


async def _bounded_nonblocking(awaitable: Any, timeout_seconds: float) -> Any:
    """Bound a request without waiting for a cancelled ``to_thread`` task.

    ``asyncio.wait_for`` waits for cancellation to propagate.  That is unsafe
    for the visual worker because its pipe request runs in a synchronous
    thread.  We cancel the task and return control immediately; the caller
    aborts the current frontier and closes the worker, which makes the thread
    observe EOF and unwind safely.
    """

    task = asyncio.create_task(awaitable)
    done, _pending = await asyncio.wait({task}, timeout=timeout_seconds)
    if not done:
        task.cancel()
        raise asyncio.TimeoutError
    return task.result()


def _write_checkpoint(
    checkpoint_path: Path,
    *,
    rows: list[dict[str, Any]],
    checkpoint_contract: dict[str, Any],
) -> None:
    """Persist the latest durable frontier state without scoring it.

    A checkpoint exists primarily so an interrupted cold visual index can be
    resumed safely.  Its in-flight row must therefore expose the *current*
    stage rather than retaining ``parse_or_projection`` while a renderer or
    visual worker consumes minutes of CPU.  Only completed ``ok`` rows are
    reused by ``--resume``; in-flight states remain diagnostic and retryable.
    """

    write_checkpoint(checkpoint_path, {
        "evaluator_fingerprint": VISUAL_EVALUATOR_FINGERPRINT,
        "checkpoint_contract": checkpoint_contract,
        "rows": rows,
    })


def _load_checkpoint(
    checkpoint_path: Path, *, checkpoint_contract: dict[str, Any],
    selected_ids: list[str],
) -> tuple[list[dict[str, Any]], str]:
    """Only reuse complete rows from the identical ordered evaluation frontier."""

    return load_completed_checkpoint(
        checkpoint_path, evaluator_fingerprint=VISUAL_EVALUATOR_FINGERPRINT,
        contract=checkpoint_contract, selected_ids=selected_ids,
    )


def _timeout_budget_for_stage(
    stage: str,
    *,
    prepare_timeout_seconds: float,
    visual_timeout_seconds: float,
) -> float | None:
    """Return the budget that actually bounded a failed evaluator stage."""

    if stage in {"prepare", "render_pages"}:
        return float(prepare_timeout_seconds)
    if stage in {"visual_index", "visual_search"}:
        return float(visual_timeout_seconds)
    return None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
for root in (BACKEND_ROOT, PROJECT_ROOT / "scripts"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from app.core.config import get_settings  # noqa: E402
from app.rag.embeddings import EmbeddingError  # noqa: E402
from app.db.sqlite import connect  # noqa: E402
from app.rag.visual_worker_client import ColSmolWorkerClient  # noqa: E402
from app.rag.document_coverage import detect_document_coverage  # noqa: E402
from app.services.query_rewrite_service import has_visual_intent  # noqa: E402
from app.rag.visual_retrieval import (  # noqa: E402
    page_records_from_document_rows,
    reciprocal_rank_fuse_keys,
    render_pdf_page_records,
)
from evaluate_public_multimodal_three_mode import IsolatedAyaPipeline  # noqa: E402
from public_multimodal_eval_lib import (  # noqa: E402
    BenchmarkCase,
    load_mmlong_cases,
    load_spiqa_test_c_cases,
    mmlong_document_profile,
    stable_balanced_sample,
    stable_mmlong_case_sample,
    stable_mmlong_profile_sample,
    stable_spiqa_case_sample,
    resolve_case_ids,
)


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def _ranking_metrics(ranking: list[str], required: set[str]) -> dict[str, Any]:
    if not required:
        return {"scored": False}
    ranks = [index for index, key in enumerate(ranking, start=1) if key in required]
    recall = len(set(ranking) & required) / len(required)
    dcg = sum(1.0 / math.log2(rank + 1) for rank in ranks)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(required), len(ranking)) + 1))
    return {
        "scored": True,
        "recall": round(recall, 6),
        "hit_at_1": bool(ranks and min(ranks) <= 1),
        "hit_at_3": bool(ranks and min(ranks) <= 3),
        "hit_at_5": bool(ranks and min(ranks) <= 5),
        "hit_at_10": bool(ranks and min(ranks) <= 10),
        "mrr": round(1.0 / min(ranks), 6) if ranks else 0.0,
        "ndcg": round(dcg / ideal, 6) if ideal else 0.0,
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 3)
    value = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return round(value, 3)


def _aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate scored rows without changing their evidence semantics."""

    positive = [
        row
        for row in rows
        if row.get("status") == "ok"
        and not row.get("unanswerable")
        and row.get("required_keys")
    ]
    aggregate: dict[str, dict[str, Any]] = {}
    for arm in ("T", "V", "T+V"):
        metrics = [row["metrics"][arm] for row in positive]
        latency = [
            float(row["latency_ms"][arm])
            for row in positive
            if arm in row.get("latency_ms", {})
        ]
        aggregate[arm] = {
            "positive_cases": len(metrics),
            "recall": round(statistics.mean(item["recall"] for item in metrics), 4)
            if metrics
            else None,
            "hit_at_1": round(statistics.mean(item["hit_at_1"] for item in metrics), 4)
            if metrics
            else None,
            "hit_at_3": round(statistics.mean(item["hit_at_3"] for item in metrics), 4)
            if metrics
            else None,
            "mrr": round(statistics.mean(item["mrr"] for item in metrics), 4)
            if metrics
            else None,
            "ndcg": round(statistics.mean(item["ndcg"] for item in metrics), 4)
            if metrics
            else None,
            "p50_ms": _percentile(latency, 0.5),
            "p95_ms": _percentile(latency, 0.95),
        }
    return aggregate


def _aggregate_by_key(
    rows: list[dict[str, Any]], key: str
) -> dict[str, dict[str, dict[str, Any]]]:
    """Expose aggregates by suite/profile without mixing metric scopes."""

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        value = str(row.get(key) or "unknown")
        groups.setdefault(value, []).append(row)
    return {value: _aggregate_metrics(group) for value, group in sorted(groups.items())}


def _visual_failure_taxonomy(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Classify misses without attributing them to a specific paper or case."""

    counts: Counter[str] = Counter()
    for row in rows:
        if row.get("status") != "ok" or row.get("unanswerable"):
            continue
        required = {str(value) for value in (row.get("required_keys") or []) if str(value)}
        if not required:
            continue
        rankings = row.get("rankings") or {}
        text_rank_key = (
            "T_artifact"
            if str(row.get("text_metric_scope") or "") == "artifact"
            or str(row.get("suite") or "") == "spiqa-test-c"
            else "T"
        )
        text = set(rankings.get(text_rank_key) or rankings.get("T") or [])
        visual = set(rankings.get("V") or [])
        fused = set(rankings.get("T+V") or [])
        if not (text & required):
            counts["text_no_required_candidate"] += 1
            if not text:
                counts["text_empty_ranking"] += 1
        elif len(text & required) < len(required):
            counts["text_partial_evidence"] += 1
        if not (visual & required):
            counts["visual_no_required_candidate"] += 1
        elif len(visual & required) < len(required):
            counts["visual_partial_evidence"] += 1
        if len(fused & required) < len(visual & required):
            counts["fusion_loss_vs_visual"] += 1
        if len(fused & required) < len(required):
            counts["fused_incomplete_evidence"] += 1
    return dict(sorted(counts.items()))


def visual_promotion_gate(
    rows: list[dict[str, Any]],
    aggregate: dict[str, dict[str, Any]],
    *,
    worker_health: dict[str, Any] | None,
    answer_quality: dict[str, Any] | None = None,
    minimum_positive_cases: int = 20,
    selected_case_ids: list[str] | None = None,
    production_text_baseline: bool = False,
) -> dict[str, Any]:
    """Decide whether a visual provider earned an isolated promotion trial.

    A visual smoke can look perfect by chance, especially when one document is
    sampled.  Promotion therefore requires a minimum, profile-balanced set,
    a measurable fused-recall gain, no per-case regression of the text
    baseline, and an explicit paired answer-quality contract. This is a gate
    for staging only; it never flips production feature flags.
    """

    successful = [row for row in rows if row.get("status") == "ok"]
    positive = [row for row in successful if not row.get("unanswerable") and row.get("required_keys")]
    profile_counts = Counter(str(row.get("profile") or "unknown") for row in positive)
    suites = {str(row.get("suite") or "") for row in positive}
    # SPIQA's figure/table strata validate visual artifact selection, but they
    # do not exercise PDF page coverage or mixed/image-only parsing.  A visual
    # provider cannot be promoted for the arbitrary-format product gate from a
    # paper-artifact-only sample.  Require the three MMLong PDF frontiers in
    # addition to the modality/profile checks below.
    required_pdf_profiles = {"native_text", "mixed_text_image", "image_only"}
    observed_pdf_profiles = profile_counts.keys() & required_pdf_profiles
    regressions = [
        str(row.get("case_id"))
        for row in positive
        if float((row.get("metrics") or {}).get("T+V", {}).get("recall") or 0.0)
        < float((row.get("metrics") or {}).get("T", {}).get("recall") or 0.0)
    ]
    text_recall = float((aggregate.get("T") or {}).get("recall") or 0.0)
    fused_recall = float((aggregate.get("T+V") or {}).get("recall") or 0.0)
    quality = answer_quality if isinstance(answer_quality, dict) else {}
    quality_status = str(quality.get("status") or "not_attached").casefold()
    try:
        quality_pairs = max(0, int(quality.get("paired_cases") or 0))
    except (TypeError, ValueError):
        quality_pairs = 0
    # Retrieval-only rows cannot establish that the visual branch improves
    # answers. Require a separately scored, explicit promotion-safe contract;
    # never infer it from page/artifact recall alone.
    answer_quality_passed = (
        quality_status == "complete"
        and quality_pairs >= minimum_positive_cases
        and quality.get("promotion_safe") is True
    )
    checks = {
        "complete_frontier": (
            bool(selected_case_ids)
            and (len(selected_case_ids) == len(set(selected_case_ids))
                and len(rows) == len(selected_case_ids)
                and {row.get("case_id") for row in rows} == set(selected_case_ids))
        ),
        "production_text_baseline": production_text_baseline,
        "minimum_positive_cases": len(positive) >= minimum_positive_cases,
        "multiple_profiles": len(profile_counts) >= 3,
        "profile_minimum": bool(profile_counts) and min(profile_counts.values()) >= 2,
        "public_suite_frontier": {"mmlongbench-doc", "spiqa-test-c"}.issubset(suites),
        "pdf_format_frontier": observed_pdf_profiles == required_pdf_profiles,
        "visual_gain": fused_recall >= text_recall + 0.02,
        "no_text_regression": not regressions,
        "all_successful": len(successful) == len(rows),
        "worker_healthy": bool(worker_health and worker_health.get("ok")),
        "answer_quality": answer_quality_passed,
    }
    return {
        "passed": all(checks.values()),
        "scope": "isolated_visual_trial",
        "production_eligible": False,
        "answer_quality_required": True,
        "answer_quality_status": quality_status,
        "answer_quality_pairs": quality_pairs,
        "checks": checks,
        "positive_cases": len(positive),
        "profile_counts": dict(sorted(profile_counts.items())),
        "suites": sorted(suites),
        "required_pdf_profiles": sorted(required_pdf_profiles),
        "observed_pdf_profiles": sorted(observed_pdf_profiles),
        "text_recall": round(text_recall, 4),
        "fused_recall": round(fused_recall, 4),
        "fused_gain": round(fused_recall - text_recall, 4),
        "regressions": regressions,
        "minimum_positive_cases": minimum_positive_cases,
        "note": (
            "eligible for an isolated visual trial only; answer-quality, RC and "
            "shadow validation are still required"
            if all(checks.values())
            else "diagnostic smoke remains non-promotable until all checks pass"
        ),
    }


def visual_report_promotion_gate(
    report: dict[str, Any], *, minimum_positive_cases: int = 20,
) -> dict[str, Any]:
    """Recompute eligibility without laundering an incomplete/diagnostic run.

    Postprocessors must preserve the original preselected frontier, not infer
    completeness from the successful rows left after filtering. Legacy reports
    can still be inspected, but missing completion evidence is fail-closed.
    """
    selection = report.get("selection")
    selected = selection.get("case_ids") if isinstance(selection, dict) else None
    valid_selection = (
        isinstance(selected, list)
        and bool(selected)
        and all(isinstance(value, str) and value for value in selected)
        and len(selected) == len(set(selected))
    )
    gate = visual_promotion_gate(
        list(report.get("rows") or []),
        dict(report.get("aggregate") or {}),
        worker_health=report.get("worker_health"),
        answer_quality=report.get("answer_quality"),
        minimum_positive_cases=minimum_positive_cases,
        selected_case_ids=selected if valid_selection else None,
        production_text_baseline=report.get("pdf_ocr_enabled") is True,
    )
    checks = gate["checks"]
    checks["report_complete"] = (
        report.get("score_valid") is True and report.get("suite_complete") is True
    )
    checks["no_posthoc_selection"] = not bool(report.get("posthoc_selection"))
    checks["llm_free_retrieval"] = report.get("no_llm_calls") is True
    gate["passed"] = all(checks.values())
    if not gate["passed"]:
        gate["note"] = "diagnostic report remains non-promotable until all checks pass"
    return gate


def _active_revision(db_path: Path, document_id: str) -> str:
    with connect(db_path) as connection:
        row = connection.execute(
            "SELECT active_revision_id FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
    revision_id = str(row["active_revision_id"] or "") if row else ""
    if not revision_id:
        raise RuntimeError(f"active_revision_missing:{document_id}")
    return revision_id


def _artifact_rows(db_path: Path, document_id: str, revision_id: str) -> list[dict[str, Any]]:
    with connect(db_path) as connection:
        return [
            dict(row)
            for row in connection.execute(
                """
                SELECT id, document_id, revision_id, figure_index, page_number,
                       image_path, caption, metadata_json
                FROM document_figures
                WHERE document_id = ? AND revision_id = ?
                ORDER BY figure_index
                """,
                (document_id, revision_id),
            ).fetchall()
        ]


def _artifact_identity_maps(
    db_path: Path,
    document_id: str,
    revision_id: str,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    identity_to_file: dict[str, str] = {}
    path_to_file: dict[str, str] = {}
    file_to_page: dict[str, str] = {}
    with connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT id, image_path, page_number, metadata_json FROM document_figures
            WHERE document_id = ? AND revision_id = ?
            UNION ALL
            SELECT id, NULL AS image_path, page_number, metadata_json FROM document_tables
            WHERE document_id = ? AND revision_id = ?
            """,
            (document_id, revision_id, document_id, revision_id),
        ).fetchall()
    for row in rows:
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            metadata = {}
        filename = str(metadata.get("external_artifact_file") or "").strip()
        if not filename:
            continue
        identity_to_file[str(row["id"])] = filename
        if row["page_number"] is not None:
            file_to_page[filename] = str(int(row["page_number"]))
        if row["image_path"]:
            path_to_file[str(Path(str(row["image_path"])).resolve())] = filename
    return identity_to_file, path_to_file, file_to_page


def _text_keys(
    case: BenchmarkCase,
    results: list[dict[str, Any]],
    *,
    identity_to_file: dict[str, str],
    path_to_file: dict[str, str],
    file_to_page: dict[str, str],
    key_mode: str = "artifact",
) -> list[str]:
    if key_mode not in {"artifact", "page"}:
        raise ValueError(f"unsupported text key mode: {key_mode!r}")
    if case.suite == "mmlongbench-doc" or key_mode == "page":
        return _dedupe(
            str(int(item["page_number"]))
            for item in results
            if item.get("page_number") is not None
        )
    if key_mode == "page":
        return []
    keys: list[str] = []
    for item in results:
        image_path = str(item.get("image_path") or "").strip()
        if image_path:
            key = path_to_file.get(str(Path(image_path).expanduser().resolve()))
            if key:
                keys.append(key)
                continue
        identifiers = [
            str(item.get(name) or "")
            for name in ("id", "source_id", "chunk_id", "figure_id", "table_id")
        ]
        matched = next(
            (
                filename
                for identity, filename in identity_to_file.items()
                if any(identity == value or identity in value for value in identifiers)
            ),
            None,
        )
        if matched:
            keys.append(matched)
    return _dedupe(keys)


def _visual_keys(case: BenchmarkCase, hits: list[Any]) -> list[str]:
    if case.suite == "mmlongbench-doc":
        return _dedupe(
            str(hit.get("page_number")) if isinstance(hit, dict) else str(hit.page_number)
            for hit in hits
            if (hit.get("page_number") if isinstance(hit, dict) else getattr(hit, "page_number", None)) is not None
        )
    return _dedupe(
        str(
            (hit.get("metadata") or {}).get("external_artifact_file")
            if isinstance(hit, dict)
            else hit.metadata.get("external_artifact_file")
            or ""
        )
        for hit in hits
    )


async def _text_search(
    pipeline: IsolatedAyaPipeline,
    case: BenchmarkCase,
    document_id: str,
    top_k: int,
) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    coverage_policy = detect_document_coverage(case.question)
    response = await pipeline.rag.search_hybrid(
        query=case.question,
        top_k=top_k,
        document_ids=[document_id],
        retrieval_store=pipeline.vector_index.retrieval_store,
        embeddings=pipeline.embeddings,
        # Match the user's production retrieval configuration.  The previous
        # evaluator forced the optional reranker on, making T slower and
        # changing the comparator's ranking semantics even though the report
        # described it as a production-parity arm.
        rerank=bool(getattr(pipeline, "rerank_enabled", False)),
        rerank_mode=str(getattr(pipeline, "rerank_mode", "embedding")),
        cross_encoder_model_path=getattr(pipeline, "rerank_cross_encoder_path", None),
        rerank_max_candidates=int(getattr(pipeline, "rerank_max_candidates", 20)),
        # Keep the text arm production-shaped: benchmark evidence annotations
        # are scoring-only and must not change its ranking.
        visual_boost=bool(has_visual_intent(case.question) or coverage_policy.enabled),
        coverage_mode=coverage_policy.enabled,
        coverage_page_limit=coverage_policy.page_limit,
    )
    results = list(response.get("results") or [])
    if coverage_policy.enabled:
        counts = pipeline.rag.document_page_counts([document_id])
        expected_page_count = int(counts.get(document_id) or 0)
        aggregation_limit = min(
            int(getattr(coverage_policy, "aggregation_page_limit", 64) or 64),
            64,
        )
        if 0 < expected_page_count <= aggregation_limit:
            exhaustive = pipeline.rag.list_page_coverage(
                document_ids=[document_id],
                page_limit=expected_page_count,
            )
            existing_pages = {
                (str(item.get("document_id") or ""), int(item["page_number"]))
                for item in results
                if item.get("document_id") and item.get("page_number") is not None
            }
            results.extend(
                item
                for item in exhaustive
                if (
                    str(item.get("document_id") or ""),
                    int(item.get("page_number")),
                ) not in existing_pages
            )
    return results, (time.perf_counter() - started) * 1000


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.mmlong_per_profile < 0 or args.spiqa_cases < 0:
        raise ValueError("case counts must be non-negative")
    if args.mmlong_per_profile + args.spiqa_cases < 1:
        raise ValueError("select at least one public case")
    settings = get_settings()
    mmlong_cases = load_mmlong_cases(version=args.mmlong_version)
    all_cases = [*mmlong_cases, *load_spiqa_test_c_cases()]
    case_manifest = getattr(args, "case_ids_file", None)
    if case_manifest:
        manifest_path = Path(case_manifest).expanduser().resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            requested_case_ids = payload.get("case_ids")
        else:
            requested_case_ids = payload
        if not isinstance(requested_case_ids, list):
            raise ValueError("case IDs file must contain a list or {case_ids: [...]}")
        cases = resolve_case_ids(all_cases, requested_case_ids)
        selection_source = str(manifest_path)
    else:
        cases = [
        *(
            (
                stable_mmlong_profile_sample(
                    mmlong_cases,
                    per_profile=args.mmlong_per_profile,
                )
                if args.mmlong_cases_per_document == 1
                else stable_mmlong_case_sample(
                    mmlong_cases,
                    per_profile=args.mmlong_per_profile,
                    cases_per_document=args.mmlong_cases_per_document,
                )
            )
            if args.mmlong_per_profile
            else []
        ),
        *(
            (
                stable_balanced_sample(
                    load_spiqa_test_c_cases(), limit=args.spiqa_cases
                )
                if args.spiqa_cases_per_document == 1
                else stable_spiqa_case_sample(
                    load_spiqa_test_c_cases(),
                    limit=args.spiqa_cases,
                    cases_per_document=args.spiqa_cases_per_document,
                )
            )
            if args.spiqa_cases
            else []
        ),
        ]
        selection_source = "stable_sampler"
    runtime_root = args.runtime_root.expanduser().resolve()
    runtime_root.mkdir(parents=True, exist_ok=True)
    text_runtime_root_arg = getattr(args, "text_runtime_root", None)
    text_runtime_root = (
        Path(text_runtime_root_arg).expanduser().resolve()
        if text_runtime_root_arg
        else runtime_root / "text"
    )
    pipeline = IsolatedAyaPipeline(
        runtime_root=text_runtime_root,
        settings=settings,
        enable_pdf_ocr=not args.disable_pdf_ocr,
        stage_version=args.stage_version,
    )
    # Do not cold-parse public PDFs when the canonical local embedding
    # provider is unavailable.  Visual retrieval is optional, but its T arm
    # and all fusion comparisons still require the same local search contract.
    provider_preflight = await pipeline.preflight_local_search()
    visual = ColSmolWorkerClient(
        python_executable=settings.visual_retrieval_worker_python,
        index_root=runtime_root / "visual" / "colsmol",
        model_name=settings.visual_retrieval_model,
        model_revision=settings.visual_retrieval_model_revision,
        base_model_name=settings.visual_retrieval_base_model,
        base_model_revision=settings.visual_retrieval_base_revision,
        device=args.device,
        local_files_only=args.local_files_only,
        max_image_pixels_per_request=args.batch_max_pixels,
    )

    rows: list[dict[str, Any]] = []
    checkpoint_path = args.output.with_suffix(args.output.suffix + ".partial")
    checkpoint_contract = {
        "evaluator_fingerprint": VISUAL_EVALUATOR_FINGERPRINT,
        "stage_version": pipeline.processing.stage_version,
        "case_ids": [case.case_id for case in cases],
        "top_k": int(args.top_k),
        "mmlong_version": args.mmlong_version,
        "model_fingerprint": visual.model_fingerprint,
        "embedding_model": settings.embedding_model,
        "embedding_query_prefix": settings.embedding_query_prefix,
        "embedding_document_prefix": settings.embedding_document_prefix,
        "rerank_enabled": pipeline.rerank_enabled,
        "rerank_mode": pipeline.rerank_mode,
        "rerank_max_candidates": pipeline.rerank_max_candidates,
        "rerank_cross_encoder_path": str(pipeline.rerank_cross_encoder_path),
        "runtime_root": str(runtime_root),
        "text_runtime_root": str(text_runtime_root),
        "pdf_ocr_enabled": bool(not args.disable_pdf_ocr),
        "render_scale": round(float(args.render_scale), 4),
        "render_max_pixels": int(args.render_max_pixels),
        "batch_max_pixels": int(args.batch_max_pixels),
        "device": str(args.device),
        "local_files_only": bool(args.local_files_only),
        "prepare_timeout_seconds": float(args.prepare_timeout_seconds),
        "visual_timeout_seconds": float(args.visual_timeout_seconds),
    }
    resume_status = "not_requested"
    if args.resume:
        rows, resume_status = _load_checkpoint(
            checkpoint_path, checkpoint_contract=checkpoint_contract,
            selected_ids=[case.case_id for case in cases],
        )
    # A failed checkpoint row is diagnostic state, not completed work.  This
    # matters when a provider comes back online: ``--resume`` must retry the
    # case instead of silently carrying the old failure into the aggregate.
    completed_ids = {
        str(row.get("case_id"))
        for row in rows
        if row.get("status") == "ok"
    }
    pending_cases = [case for case in cases if case.case_id not in completed_ids]
    if pending_cases:
        # Fail before parsing a potentially large scan.  The public evaluator
        # uses local EmbeddingGemma for its text arm; when Ollama is offline,
        # spending minutes on Docling/RapidOCR only to fail at vectorization is
        # neither useful nor a valid benchmark result.
        try:
            await pipeline.embeddings.embed_query("visual retrieval gate preflight")
        except Exception as exc:
            await visual.aclose()
            raise RuntimeError(f"embedding_preflight_failed: {exc}") from exc
    prepared_visual: dict[str, dict[str, Any]] = {}
    abort_after_timeout = False
    for case in cases:
        if case.case_id in completed_ids:
            continue
        # Persist intent before cold Docling/OCR/rendering starts.  A long
        # image-only document can be interrupted before the final score row;
        # retaining a visible ``preparing`` row makes that interruption
        # auditable and lets a later --resume run replace it rather than
        # pretending the case never existed.
        progress_row = {
            "case_id": case.case_id,
            "suite": case.suite,
            "document_id": case.document_id,
            "profile": (
                mmlong_document_profile(case)
                if case.suite == "mmlongbench-doc"
                else case.stratum
            ),
            "status": "preparing",
            "stage": "parse_or_projection",
        }
        rows = [
            existing
            for existing in rows
            if str(existing.get("case_id")) != case.case_id
        ]
        rows.append(progress_row)
        _write_checkpoint(
            checkpoint_path,
            rows=rows,
            checkpoint_contract=checkpoint_contract,
        )
        row_started = time.perf_counter()
        current_stage = "prepare"
        try:
            prepared = await _bounded_nonblocking(
                pipeline.prepare(case),
                float(args.prepare_timeout_seconds),
            )
            document_id = str(prepared["document_id"])
            revision_id = _active_revision(pipeline.db_path, document_id)
            visual_key = f"{document_id}:{revision_id}"
            visual_status = prepared_visual.get(visual_key)
            if visual_status is None:
                current_stage = "render_pages"
                progress_row["stage"] = current_stage
                _write_checkpoint(
                    checkpoint_path,
                    rows=rows,
                    checkpoint_contract=checkpoint_contract,
                )
                if case.source_path is not None:
                    isolated_source, _ = pipeline._source_for(case)  # noqa: SLF001
                    page_records = await asyncio.wait_for(
                        asyncio.to_thread(
                            render_pdf_page_records,
                            source_path=isolated_source,
                            document_id=document_id,
                            revision_id=revision_id,
                            output_root=runtime_root / "visual" / "rendered",
                            scale=args.render_scale,
                            max_render_pixels=args.render_max_pixels,
                        ),
                        timeout=float(args.prepare_timeout_seconds),
                    )
                else:
                    page_records = page_records_from_document_rows(
                        _artifact_rows(pipeline.db_path, document_id, revision_id)
                    )
                current_stage = "visual_index"
                progress_row["stage"] = current_stage
                _write_checkpoint(
                    checkpoint_path,
                    rows=rows,
                    checkpoint_contract=checkpoint_contract,
                )

                def persist_visual_batch(progress: dict[str, Any]) -> None:
                    """Expose durable per-batch progress without affecting ranking.

                    The worker commits every batch before emitting this callback,
                    so these numbers describe recoverable state rather than a
                    speculative progress bar.  Keep only scalar telemetry in
                    the checkpoint; detailed per-page errors remain in the
                    eventual visual-index result.
                    """

                    progress_row["visual_progress"] = {
                        key: progress.get(key)
                        for key in (
                            "requested_pages",
                            "processed_pages",
                            "remaining_pages",
                            "indexed_pages",
                            "reused_pages",
                            "failed_pages",
                            "batches",
                            "elapsed_ms",
                        )
                    }
                    _write_checkpoint(
                        checkpoint_path,
                        rows=rows,
                        checkpoint_contract=checkpoint_contract,
                    )

                visual_status = await _bounded_nonblocking(
                    visual.index_pages(
                        page_records,
                        progress_callback=persist_visual_batch,
                    ),
                    float(args.visual_timeout_seconds),
                )
                prepared_visual[visual_key] = visual_status

            identity_map, path_map, file_to_page = _artifact_identity_maps(
                pipeline.db_path, document_id, revision_id
            )
            current_stage = "text_search"
            progress_row["stage"] = current_stage
            _write_checkpoint(
                checkpoint_path,
                rows=rows,
                checkpoint_contract=checkpoint_contract,
            )
            text_results, text_ms = await asyncio.wait_for(
                _text_search(pipeline, case, document_id, args.top_k),
                timeout=float(args.visual_timeout_seconds),
            )
            visual_started = time.perf_counter()
            # Use the same RagService adapter as the live agent rather than
            # calling the provider directly. This preserves active-revision
            # filtering and explicit page/slide constraints in the comparator;
            # the visual model/ranking itself is unchanged.
            current_stage = "visual_search"
            progress_row["stage"] = current_stage
            _write_checkpoint(
                checkpoint_path,
                rows=rows,
                checkpoint_contract=checkpoint_contract,
            )
            visual_hits = await _bounded_nonblocking(
                pipeline.rag.search_visual_pages(
                    case.question,
                    provider=visual,
                    document_ids=[document_id],
                    top_k=args.top_k,
                ),
                float(args.visual_timeout_seconds),
            )
            visual_ms = (time.perf_counter() - visual_started) * 1000
            text_artifact_ranking = _text_keys(
                case,
                text_results,
                identity_to_file=identity_map,
                path_to_file=path_map,
                file_to_page=file_to_page,
                key_mode="artifact",
            )
            text_page_ranking = _text_keys(
                case,
                text_results,
                identity_to_file=identity_map,
                path_to_file=path_map,
                file_to_page=file_to_page,
                key_mode="page",
            )
            visual_ranking = _visual_keys(case, visual_hits)
            fusion_started = time.perf_counter()
            fused_ranking = reciprocal_rank_fuse_keys(
                [text_artifact_ranking, visual_ranking], top_k=args.top_k
            )
            fusion_ms = (time.perf_counter() - fusion_started) * 1000
            required = (
                {str(value) for value in case.evidence_pages}
                if case.suite == "mmlongbench-doc"
                else {str(value) for value in case.referred_artifacts}
            )
            text_page_required = (
                required
                if case.suite == "mmlongbench-doc"
                else {
                    file_to_page[filename]
                    for filename in required
                    if file_to_page.get(filename)
                }
            )
            # MMLong's native PDFs carry trustworthy page provenance, so its
            # text arm is scored at page level.  SPIQA's official extracted
            # Markdown intentionally flattens paper sections and assigns no
            # native page provenance; scoring that arm against the filename's
            # page would turn every text miss into a fake page-1 miss.  For
            # SPIQA, text therefore stays artifact-level (canonical table /
            # figure identity), while an explicit T_page diagnostic is marked
            # unscored rather than inventing page locations.
            text_metric_scope = "page" if case.suite == "mmlongbench-doc" else "artifact"
            text_metric_ranking = (
                text_page_ranking
                if text_metric_scope == "page"
                else text_artifact_ranking
            )
            text_metric_required = (
                required if text_metric_scope == "artifact" else text_page_required
            )
            unanswerable = any(
                answer.strip().casefold() == "not answerable" for answer in case.answers
            )
            row = {
                "case_id": case.case_id,
                "suite": case.suite,
                "document_id": case.document_id,
                "profile": (
                    mmlong_document_profile(case)
                    if case.suite == "mmlongbench-doc"
                    else case.stratum
                ),
                "question": case.question,
                "unanswerable": unanswerable,
                "required_keys": sorted(required),
                "text_required_keys": sorted(text_metric_required),
                "text_page_required_keys": sorted(text_page_required),
                "text_metric_scope": text_metric_scope,
                "rankings": {
                    "T": text_metric_ranking,
                    "T_artifact": text_artifact_ranking,
                    "T_page": text_page_ranking,
                    "V": visual_ranking,
                    "T+V": fused_ranking,
                },
                "metrics": {
                    "T": _ranking_metrics(text_metric_ranking, text_metric_required),
                    "T_artifact": _ranking_metrics(text_artifact_ranking, required),
                    "T_page": (
                        _ranking_metrics(text_page_ranking, text_page_required)
                        if case.suite == "mmlongbench-doc"
                        else {
                            "scored": False,
                            "reason": "source_has_no_native_page_provenance",
                        }
                    ),
                    "V": _ranking_metrics(visual_ranking, required),
                    "T+V": _ranking_metrics(fused_ranking, required),
                },
                "latency_ms": {
                    "T": round(text_ms, 3),
                    "V": round(visual_ms, 3),
                    "T+V": round(text_ms + visual_ms + fusion_ms, 3),
                    "fusion": round(fusion_ms, 3),
                },
                "visual_index": visual_status,
                "status": "ok",
            }
        except Exception as exc:
            row = {
                "case_id": case.case_id,
                "suite": case.suite,
                "document_id": case.document_id,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "failed_stage": current_stage,
                "timeout_seconds": (
                    _timeout_budget_for_stage(
                        current_stage,
                        prepare_timeout_seconds=args.prepare_timeout_seconds,
                        visual_timeout_seconds=args.visual_timeout_seconds,
                    )
                    if isinstance(exc, asyncio.TimeoutError)
                    else None
                ),
            }
            if isinstance(exc, asyncio.TimeoutError):
                abort_after_timeout = True
        row["wall_ms"] = round((time.perf_counter() - row_started) * 1000, 3)
        # Replace an earlier failed/partial checkpoint for the same case.  A
        # resumed run must contain one authoritative row per case, otherwise a
        # transient provider outage would double-count the case in metrics.
        rows = [
            existing
            for existing in rows
            if str(existing.get("case_id")) != case.case_id
        ]
        rows.append(row)
        _write_checkpoint(
            checkpoint_path,
            rows=rows,
            checkpoint_contract=checkpoint_contract,
        )
        if abort_after_timeout:
            break

    aggregate = _aggregate_metrics(rows)

    visual_root = runtime_root / "visual"
    index_bytes = sum(path.stat().st_size for path in visual_root.rglob("*") if path.is_file())
    worker_health: dict[str, Any] | None = None
    if abort_after_timeout:
        # A cancelled pipe request may still be unwinding in a thread.  Kill
        # the child before health telemetry, otherwise health would wait on
        # the same worker lock and defeat the timeout boundary.
        await visual.aclose()
        worker_health = {
            "ok": False,
            "error_type": "TimeoutError",
            "error": "worker terminated after stage timeout",
        }
    else:
        try:
            worker_health = await visual.health()
        except Exception as exc:
            # Health is telemetry, not a scoring dependency.  A completed
            # retrieval run remains valid if a worker exits while we collect
            # diagnostics; preserve the reason instead of hiding it.
            worker_health = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
    try:
        import psutil

        rss_mb = round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
    except Exception:
        rss_mb = None
    promotion = visual_promotion_gate(
        rows,
        aggregate,
        worker_health=worker_health,
        selected_case_ids=[case.case_id for case in cases],
        production_text_baseline=not args.disable_pdf_ocr,
    )
    suite_complete = len(rows) == len(cases) and all(row.get("status") == "ok" for row in rows)
    report = {
        "schema_version": "visual-retrieval-gate-v1",
        "promotion_contract_version": "visual-arbitrary-format-v2",
        "metric_contract_version": "visual-evidence-scope-v2",
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_root": str(runtime_root),
        "text_runtime_root": str(text_runtime_root),
        "model": settings.visual_retrieval_model,
        "evaluator_fingerprint": VISUAL_EVALUATOR_FINGERPRINT,
        "model_revision": settings.visual_retrieval_model_revision,
        "model_fingerprint": visual.model_fingerprint,
        "embedding_model_T": settings.embedding_model,
        "projection_stage_version": pipeline.processing.stage_version,
        "score_valid": suite_complete,
        "suite_complete": suite_complete,
        "selected_cases": len(cases),
        "resume_status": resume_status,
        "pdf_ocr_enabled": bool(not args.disable_pdf_ocr),
        "evaluation_configuration": {
            "render_scale": round(float(args.render_scale), 4),
            "render_max_pixels": int(args.render_max_pixels),
            "batch_max_pixels": int(args.batch_max_pixels),
            "device": str(args.device),
            "local_files_only": bool(args.local_files_only),
            "stage_version": pipeline.processing.stage_version,
        },
        "timeouts": {
            "prepare_seconds": float(args.prepare_timeout_seconds),
            "visual_seconds": float(args.visual_timeout_seconds),
        },
        "retrieval_configuration": {
            "rerank_enabled": bool(pipeline.rerank_enabled),
            "rerank_mode": pipeline.rerank_mode,
            "rerank_max_candidates": pipeline.rerank_max_candidates,
            "cross_encoder_model_path": str(pipeline.rerank_cross_encoder_path)
            if pipeline.rerank_cross_encoder_path
            else None,
        },
        "provider_preflight": provider_preflight,
        "no_llm_calls": True,
        "cases": len(rows),
        "successful": sum(row.get("status") == "ok" for row in rows),
        "aborted_after_timeout": bool(abort_after_timeout),
        "failures_by_type": dict(
            Counter(
                str(row.get("error_type") or "unknown")
                for row in rows
                if row.get("status") != "ok"
            )
        ),
        "aggregate": aggregate,
        "aggregate_by_suite": _aggregate_by_key(rows, "suite"),
        "aggregate_by_profile": _aggregate_by_key(rows, "profile"),
        "failure_taxonomy": _visual_failure_taxonomy(rows),
        "visual_index_size_bytes": index_bytes,
        "orchestrator_process_rss_mb": rss_mb,
        "worker_health": worker_health,
        "promotion": promotion,
        "selection": {
            "source": selection_source,
            "mmlong_version": args.mmlong_version,
            "mmlong_per_profile": args.mmlong_per_profile,
            "mmlong_cases_per_document": args.mmlong_cases_per_document,
            "spiqa_cases": args.spiqa_cases,
            "spiqa_cases_per_document": args.spiqa_cases_per_document,
            "case_ids": [case.case_id for case in cases],
        },
        "rows": rows,
        "note": (
            "Gold page/artifact annotations are scoring-only. T+V uses equal RRF over "
            "canonical page/artifact keys; incomparable raw scores are never mixed."
        ),
    }
    await visual.aclose()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmlong-per-profile", type=int, default=1)
    parser.add_argument(
        "--mmlong-cases-per-document",
        type=int,
        default=1,
        help="reuse deterministic questions from each selected PDF (diagnostic only)",
    )
    parser.add_argument(
        "--mmlong-version",
        choices=("v1", "v2"),
        default="v2",
        help="MMLongBench-Doc annotation release; v2 is recommended for new runs",
    )
    parser.add_argument("--spiqa-cases", type=int, default=4)
    parser.add_argument(
        "--case-ids-file",
        type=Path,
        default=None,
        help=(
            "optional JSON list or {case_ids: [...]} manifest; when present, "
            "run exactly this ordered frontier so retrieval and answer reports pair"
        ),
    )
    parser.add_argument(
        "--spiqa-cases-per-document",
        type=int,
        default=1,
        help="reuse a bounded number of deterministic questions per SPIQA paper",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--stage-version",
        default=None,
        help=(
            "explicit isolated projection contract to reuse; omit for the current "
            "foundation-v3 contract. Use an older value only with a matching "
            "isolated runtime, never for production ingestion."
        ),
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"))
    parser.add_argument("--render-scale", type=float, default=1.25)
    parser.add_argument(
        "--render-max-pixels",
        type=int,
        default=2_000_000,
        help="maximum pixels for one rendered PDF page before downscaling",
    )
    parser.add_argument(
        "--batch-max-pixels",
        type=int,
        default=4_000_000,
        help="maximum aggregate pixels sent to one ColSmol image forward batch",
    )
    parser.add_argument(
        "--prepare-timeout-seconds",
        type=float,
        default=900.0,
        help="maximum seconds for isolated parse/projection/page rendering per case",
    )
    parser.add_argument(
        "--visual-timeout-seconds",
        type=float,
        default=600.0,
        help="maximum seconds for visual indexing/search and text comparison per case",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--disable-pdf-ocr",
        action="store_true",
        help=(
            "skip derived PDF OCR in the isolated visual-only run; this keeps "
            "OCR cost separate from ColSmol recall and never changes production defaults"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume case rows from <output>.partial while reusing the visual manifest",
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=PROJECT_ROOT
        / "data/retrieval_eval/public/results/runtime/visual-retrieval-gate-v1",
    )
    parser.add_argument(
        "--text-runtime-root",
        type=Path,
        default=None,
        help=(
            "optional existing isolated text/index root; use this when the "
            "answer-quality and visual reports share a prepared projection"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "data/retrieval_eval/public/results/visual-retrieval-gate-v1.json",
    )
    args = parser.parse_args()
    if (
        args.top_k < 1
        or args.render_scale <= 0
        or args.render_max_pixels < 1
        or args.batch_max_pixels < 1
        or args.prepare_timeout_seconds <= 0
        or args.visual_timeout_seconds <= 0
        or args.mmlong_cases_per_document < 1
        or args.spiqa_cases_per_document < 1
    ):
        parser.error("top-k, render-scale and per-document case counts must be positive")
    try:
        report = asyncio.run(run(args))
    except EmbeddingError as exc:
        # The text arm is a required comparator for every visual run.  Fail
        # before cold visual indexing and make the missing local provider
        # explicit rather than producing an empty/non-comparable visual score.
        report = {
            "schema_version": "visual-retrieval-gate-v1",
            "created_at": datetime.now(UTC).isoformat(),
            "status": "provider_unavailable",
            "score_valid": False,
            "embedding_model_T": get_settings().embedding_model,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "promotion": {"passed": False, "reason": "text_provider_unavailable"},
            "note": "No visual aggregate is valid until the local embedding provider is available.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2
    except KeyboardInterrupt:
        # Cold page-image encoding can exceed an interactive shell's patience.
        # Keep the checkpoint auditable but never publish a partial aggregate
        # as a visual score or promotion decision.
        partial = args.output.with_suffix(args.output.suffix + ".partial")
        interrupted = {
            "schema_version": "visual-retrieval-gate-interrupted-v1",
            "created_at": datetime.now(UTC).isoformat(),
            "status": "interrupted",
            "score_valid": False,
            "partial_checkpoint": str(partial),
            "note": "No aggregate score is valid until the checkpoint is resumed to completion.",
        }
        interrupted_path = args.output.with_suffix(".interrupted.json")
        interrupted_path.parent.mkdir(parents=True, exist_ok=True)
        interrupted_path.write_text(
            json.dumps(interrupted, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(interrupted, ensure_ascii=False, indent=2))
        return 130
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in ("cases", "successful", "aggregate")}, indent=2))
    return 0 if report["suite_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
