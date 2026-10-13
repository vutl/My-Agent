#!/usr/bin/env python3
"""Run a no-LLM retrieval-only smoke on public MMLongBench/SPIQA cases.

The runner reuses the isolated production pipeline used by the multimodal
three-mode evaluator, but stops before answer generation.  Official evidence
annotations are used only after retrieval to compute diagnostics; they are not
passed into indexing or query construction.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import UTC, datetime
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluation_checkpoint import load_completed_checkpoint, write_checkpoint  # noqa: E402
from evaluator_fingerprint import default_rag_dependency_files, fingerprint_files  # noqa: E402

EVALUATOR_FINGERPRINT = fingerprint_files(PROJECT_ROOT, [
    Path(__file__).resolve(),
    Path(__file__).with_name("evaluation_checkpoint.py"),
    Path(__file__).with_name("public_multimodal_eval_lib.py"),
    Path(__file__).with_name("evaluate_public_multimodal_three_mode.py"),
    *default_rag_dependency_files(PROJECT_ROOT),
])

from evaluate_public_multimodal_three_mode import IsolatedAyaPipeline  # noqa: E402
from app.rag.embeddings import EmbeddingError  # noqa: E402
from public_multimodal_eval_lib import (  # noqa: E402
    BenchmarkCase,
    load_mmlong_cases,
    load_spiqa_test_c_cases,
    mmlong_document_profile,
    stable_mmlong_profile_sample,
    stable_balanced_sample,
    resolve_case_ids,
    read_spiqa_images,
    SPIQA_IMAGE_INPUT_POLICY,
)
from app.core.config import get_settings  # noqa: E402
from app.services.query_rewrite_service import (  # noqa: E402
    has_enumerated_evidence_intent,
    has_visual_intent,
)


def _norm(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def _checkpoint_contract(args: Any, pipeline: Any, settings: Any, cases: list[BenchmarkCase]) -> dict[str, Any]:
    # Hash annotations and inputs for reuse only; never inject them into a query.
    sources = hashlib.sha256()
    seen_paths: set[Path] = set()
    seen_spiqa_documents: set[str] = set()
    for case in cases:
        sources.update(json.dumps(asdict(case), sort_keys=True, default=str).encode())
        if case.source_path is not None:
            path = case.source_path.resolve()
            if path not in seen_paths:
                seen_paths.add(path)
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        sources.update(block)
        elif case.suite == "spiqa-test-c" and case.document_id not in seen_spiqa_documents:
            seen_spiqa_documents.add(case.document_id)
            # All authored artifacts, never only the gold-referred images.
            for name, payload in read_spiqa_images(
                case, [item["file"] for item in case.artifacts],
            ):
                sources.update(name.encode())
                sources.update(payload)
    return {
        "evaluator_fingerprint": EVALUATOR_FINGERPRINT,
        "case_ids": [case.case_id for case in cases],
        "source_annotations_sha256": sources.hexdigest(),
        "spiqa_image_input_policy": SPIQA_IMAGE_INPUT_POLICY,
        "runtime_root": str(Path(args.runtime_root).resolve()),
        "top_k": int(args.top_k),
        "mmlong_version": args.mmlong_version,
        "materialization_policy": pipeline.indexing.materialization_policy("pdf"),
        "stage_version": pipeline.processing.stage_version,
        "embedding_model": settings.embedding_model,
        "embedding_query_prefix": settings.embedding_query_prefix,
        "embedding_document_prefix": settings.embedding_document_prefix,
        "rerank_enabled": pipeline.rerank_enabled,
        "rerank_mode": pipeline.rerank_mode,
        "rerank_max_candidates": pipeline.rerank_max_candidates,
        "rerank_cross_encoder_path": str(pipeline.rerank_cross_encoder_path),
    }


def _phrase_hit(context: str, evidence: str) -> bool:
    evidence_norm = _norm(evidence)
    if not evidence_norm:
        return False
    # Exact short phrases are useful for parser diagnostics, but avoid calling
    # a long, whitespace-damaged PDF quote a miss when its informative tokens
    # are still present in the selected context.
    if evidence_norm in _norm(context):
        return True
    tokens = [token for token in re.findall(r"[\w%.-]+", evidence_norm) if len(token) > 2]
    if len(tokens) < 3:
        return False
    return sum(token in _norm(context) for token in tokens) / len(tokens) >= 0.65


def _answer_token_coverage(context: str, answer: str) -> float | None:
    """Post-retrieval diagnostic for malformed/annotation-only evidence spans."""

    tokens = [token for token in re.findall(r"[\w%.-]+", _norm(answer)) if len(token) > 2]
    if not tokens:
        return None
    normalized_context = _norm(context)
    return sum(token in normalized_context for token in tokens) / len(tokens)


def _artifact_recall(artifact_hits: dict[str, bool]) -> float | None:
    """Score artifact selection independently from OCR text quality."""

    values = [
        bool(hit)
        for name, hit in artifact_hits.items()
        if name != "annotation_present_but_name_unresolved"
    ]
    return (sum(values) / len(values)) if values else None


def _candidate_artifact_recall(
    referred_artifacts: list[str],
    candidate_artifact_names: list[str],
) -> tuple[float | None, dict[str, bool]]:
    """Measure artifact presence in the first-stage frontier independently.

    ``artifact_recall`` remains the final-context metric for backwards
    compatibility.  This paired metric answers the more useful question:
    did candidate generation find the image/table before context budgeting?
    """

    referred = [str(value).strip() for value in referred_artifacts if str(value).strip()]
    if not referred:
        return None, {}
    available = {str(value).strip() for value in candidate_artifact_names if str(value).strip()}
    hits = {name: name in available for name in referred}
    return sum(hits.values()) / len(hits), hits


def _query_requires_artifact_support(query: str) -> bool:
    """Mirror production modality cues without reading gold annotations."""

    normalized = " ".join(str(query or "").casefold().split())
    return bool(
        has_visual_intent(normalized)
        or has_enumerated_evidence_intent(normalized)
        or re.search(
            r"(?<!\w)(?:figure|fig\.?|table|bảng|hình)\s*"
            r"(?:#|no\.?|number|số)?\s*\d+(?!\w)",
            normalized,
        )
    )


def _case_result(
    case: BenchmarkCase,
    retrieved: dict[str, Any],
    *,
    document_profile: str | None = None,
) -> dict[str, Any]:
    context = str(retrieved.get("context") or "")
    pages = {int(value) for value in retrieved.get("retrieved_pages") or []}
    candidate_pages = {
        int(value)
        for value in retrieved.get("candidate_evidence_pages") or []
    }
    required_pages = {int(value) for value in case.evidence_pages}
    page_hit = bool(required_pages & pages) if required_pages else None
    unanswerable = (
        not required_pages
        and (
            str(case.answer_format or "").casefold() in {"", "none"}
            or any(str(answer).casefold().strip() == "not answerable" for answer in case.answers)
        )
    )
    evidence_hits = [_phrase_hit(context, value) for value in case.evidence_text]
    candidate_evidence_context = str(retrieved.get("candidate_evidence_text") or "")
    candidate_evidence_hits = [
        _phrase_hit(candidate_evidence_context, value)
        for value in case.evidence_text
    ]
    artifact_names = {
        str(item.get("file") or "")
        for item in case.artifacts
        if item.get("file")
    }
    referred = [name for name in case.referred_artifacts if name]
    artifact_hits = {
        name: (name in context or any(name == image_name for image_name, _ in retrieved.get("images", [])))
        for name in referred
    }
    if not artifact_hits and (
        case.metadata.get("figure_in_evidence") or case.metadata.get("table_in_evidence")
    ):
        # SPIQA may provide a prose caption rather than the filename in the
        # selected context; retain the evidence-text signal separately.
        artifact_hits = {"annotation_present_but_name_unresolved": bool(any(evidence_hits))}
    evidence_phrase_recall = (
        (sum(evidence_hits) / len(evidence_hits)) if evidence_hits else None
    )
    artifact_recall = _artifact_recall(artifact_hits)
    candidate_artifact_recall, candidate_artifact_hits = _candidate_artifact_recall(
        referred,
        list(retrieved.get("candidate_artifact_names") or []),
    )
    final_artifact_present = any(
        str(item.get("artifact_type") or item.get("chunk_type") or "").casefold()
        in {"table", "figure", "image"}
        or item.get("table_id")
        or item.get("figure_id")
        for item in (retrieved.get("source_summaries") or [])
        if isinstance(item, dict)
    ) or bool(retrieved.get("images"))
    artifact_support_required = _query_requires_artifact_support(case.question)
    support_candidates = [
        value
        for value in (evidence_phrase_recall, artifact_recall)
        if value is not None
    ]
    return {
        "case_id": case.case_id,
        "suite": case.suite,
        # Keep the input-format stratum separate from the benchmark's semantic
        # ``doc_type``.  This prevents an academic paper's profile from being
        # silently confused with the retrieval question category.
        "document_profile": document_profile,
        "stratum": case.metadata.get("doc_type") or (
            "table" if case.metadata.get("table_in_evidence") else
            "figure" if case.metadata.get("figure_in_evidence") else "text"
        ),
        "question": case.question,
        "required_pages": sorted(required_pages),
        "retrieved_pages": sorted(pages),
        "candidate_evidence_pages": sorted(candidate_pages),
        "candidate_page_recall": (
            (len(required_pages & candidate_pages) / len(required_pages))
            if required_pages
            else None
        ),
        "page_recall": page_hit,
        "unanswerable": unanswerable,
        "evidence_phrase_hits": evidence_hits,
        # Keep the historical phrase score for comparability, while exposing
        # modality-aware artifact selection and a separate support diagnostic.
        "evidence_recall": evidence_phrase_recall,
        "evidence_phrase_recall": evidence_phrase_recall,
        "candidate_evidence_hits": candidate_evidence_hits,
        "candidate_evidence_recall": (
            (sum(candidate_evidence_hits) / len(candidate_evidence_hits))
            if candidate_evidence_hits else None
        ),
        "artifact_recall": artifact_recall,
        "candidate_artifact_recall": candidate_artifact_recall,
        "candidate_artifact_hits": candidate_artifact_hits,
        "candidate_artifact_names": sorted(
            str(value) for value in (retrieved.get("candidate_artifact_names") or [])
        ),
        "candidate_result_count": retrieved.get("candidate_result_count"),
        "artifact_support_required_by_query": artifact_support_required,
        "final_artifact_support_hit": bool(final_artifact_present),
        "query_scoped_artifact_support": (
            1.0 if final_artifact_present else 0.0
        ) if artifact_support_required else None,
        "retrieval_support_recall": max(support_candidates) if support_candidates else None,
        "answer_token_coverage": _answer_token_coverage(context, case.answers[0] if case.answers else ""),
        "evidence_annotation_note": (
            "annotation_marker_or_non_content"
            if any(re.match(r"\s*(?:FLOAT|TEXT|IMAGE|TABLE|FIGURE)\s+(?:SELECTED|EVIDENCE)\s*:", value, re.IGNORECASE) for value in case.evidence_text)
            else None
        ),
        "artifact_names_available": sorted(artifact_names),
        "referred_artifact_hits": artifact_hits,
        "context_chars": len(context),
        "image_count": len(retrieved.get("images") or []),
        "index_ms": retrieved.get("index_ms"),
        "index_cached": retrieved.get("index_cached"),
        "retrieval_ms": retrieved.get("retrieval_ms"),
        "vector_status": retrieved.get("vector_status"),
        "retrieval_channels": retrieved.get("retrieval_channels"),
        "source_summaries": retrieved.get("source_summaries") or [],
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if (
        args.mmlong_cases < 0
        or args.spiqa_cases < 0
        or args.mmlong_per_profile < 0
        or (not args.mmlong_per_profile and args.mmlong_cases + args.spiqa_cases < 1)
    ):
        raise ValueError("select at least one case")
    settings = get_settings()
    all_mmlong_cases = load_mmlong_cases(version=args.mmlong_version)
    all_spiqa_cases = load_spiqa_test_c_cases()
    if args.case_ids_file is not None:
        manifest_path = Path(args.case_ids_file).expanduser().resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        invalidation_path = manifest_path.with_name(
            manifest_path.name.removesuffix(".case-ids.json") + ".invalid.json"
        )
        if invalidation_path.is_file() or (
            isinstance(payload, dict) and payload.get("status") == "invalid_for_promotion"
        ):
            raise ValueError("refusing invalidated public frontier: " + str(manifest_path))
        requested_ids = payload.get("case_ids") if isinstance(payload, dict) else payload
        if not isinstance(requested_ids, list):
            raise ValueError("case IDs file must contain a list or {case_ids: [...]}")
        cases = resolve_case_ids([*all_mmlong_cases, *all_spiqa_cases], requested_ids)
    else:
        mmlong_cases = (
            stable_mmlong_profile_sample(
                all_mmlong_cases,
                per_profile=args.mmlong_per_profile,
            )
            if args.mmlong_per_profile
            else stable_balanced_sample(all_mmlong_cases, limit=args.mmlong_cases)
            if args.mmlong_cases
            else []
        )
        cases = [
            *mmlong_cases,
            *(stable_balanced_sample(all_spiqa_cases, limit=args.spiqa_cases) if args.spiqa_cases else []),
        ]
    pipeline = IsolatedAyaPipeline(
        runtime_root=args.runtime_root,
        settings=settings,
        enable_pdf_ocr=not args.disable_pdf_ocr,
    )
    # Fail before expensive public parsing if local embeddings are offline.
    # This is a local provider preflight only; no 9router/LLM call or fallback
    # model is permitted in retrieval-only evaluation.
    provider_preflight = await pipeline.preflight_local_search()
    rows: list[dict[str, Any]] = []
    checkpoint_path = args.output.with_suffix(args.output.suffix + ".partial")
    contract = _checkpoint_contract(args, pipeline, settings, cases)
    resume_status = "not_requested"
    if args.resume:
        rows, resume_status = load_completed_checkpoint(
            checkpoint_path, evaluator_fingerprint=EVALUATOR_FINGERPRINT,
            contract=contract, selected_ids=[case.case_id for case in cases],
        )
    # A checkpoint is only complete for a case after a successful retrieval.
    # Failed rows remain useful audit evidence, but must be retried on resume;
    # otherwise a transient provider/parser failure would silently shrink the
    # evaluated population and could make the aggregate look healthier than it
    # is.  Keep this policy aligned with the visual-gate runner.
    completed_ids = {
        str(row.get("case_id"))
        for row in rows
        if str(row.get("status") or "") == "ok"
    }
    for case in cases:
        if case.case_id in completed_ids:
            continue
        started = time.perf_counter()
        document_profile = (
            mmlong_document_profile(case)
            if case.suite == "mmlongbench-doc"
            else "extracted_adapter"
        )
        try:
            retrieved = await pipeline.retrieve(case, top_k=args.top_k)
            row = _case_result(case, retrieved, document_profile=document_profile)
            row["status"] = "ok"
        except Exception as exc:  # keep a per-stratum failure instead of hiding it
            row = {
                "case_id": case.case_id,
                "suite": case.suite,
                "document_profile": document_profile,
                "stratum": case.metadata.get("doc_type") or "unknown",
                "unanswerable": not case.evidence_pages and str(case.answer_format or "").casefold() in {"", "none"},
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        row["wall_ms"] = round((time.perf_counter() - started) * 1000, 3)
        # Replace an older failed/interrupted row for this case rather than
        # accumulating duplicates across retries.  This keeps per-case and
        # aggregate metrics one-row-per-selected-case while retaining the
        # latest failure/success outcome in the checkpoint.
        rows = [
            existing
            for existing in rows
            if str(existing.get("case_id")) != case.case_id
        ]
        rows.append(row)
        write_checkpoint(checkpoint_path, {
                    "schema_version": "public-retrieval-only-v3-partial",
                    "created_at": datetime.now(UTC).isoformat(),
                    "selected_case_ids": [case.case_id for case in cases],
                    "evaluator_fingerprint": EVALUATOR_FINGERPRINT,
                    "checkpoint_contract": contract,
                    "rows": rows,
                    "note": "Checkpoint only; safe to resume after interruption.",
        })

    successful = [row for row in rows if row.get("status") == "ok"]
    page_values = [row["page_recall"] for row in successful if row.get("page_recall") is not None]
    evidence_values = [row["evidence_recall"] for row in successful if row.get("evidence_recall") is not None]
    artifact_values = [row["artifact_recall"] for row in successful if row.get("artifact_recall") is not None]
    candidate_page_values = [
        row["candidate_page_recall"]
        for row in successful
        if row.get("candidate_page_recall") is not None
    ]
    candidate_artifact_values = [
        row["candidate_artifact_recall"]
        for row in successful
        if row.get("candidate_artifact_recall") is not None
    ]
    candidate_evidence_values = [
        row["candidate_evidence_recall"]
        for row in successful
        if row.get("candidate_evidence_recall") is not None
    ]
    query_scoped_artifact_values = [
        row["query_scoped_artifact_support"]
        for row in successful
        if row.get("query_scoped_artifact_support") is not None
    ]
    support_values = [
        row["retrieval_support_recall"]
        for row in successful
        if row.get("retrieval_support_recall") is not None
    ]
    answer_coverage_values = [
        float(row["answer_token_coverage"])
        for row in successful
        if row.get("answer_token_coverage") is not None
    ]
    retrieval_values = [float(row["retrieval_ms"]) for row in successful if row.get("retrieval_ms") is not None]
    by_stratum: dict[str, dict[str, Any]] = {}
    for stratum in sorted({str(row.get("stratum") or "unknown") for row in rows}):
        subset = [row for row in successful if str(row.get("stratum") or "unknown") == stratum]
        by_stratum[stratum] = {
            "cases": len(subset),
            "unanswerable_cases": sum(1 for row in subset if row.get("unanswerable")),
            "page_recall": round(statistics.mean(row["page_recall"] for row in subset if row.get("page_recall") is not None), 4)
            if any(row.get("page_recall") is not None for row in subset) else None,
            "candidate_page_recall": round(
                statistics.mean(
                    row["candidate_page_recall"]
                    for row in subset
                    if row.get("candidate_page_recall") is not None
                ),
                4,
            ) if any(row.get("candidate_page_recall") is not None for row in subset) else None,
            "candidate_evidence_recall": round(
                statistics.mean(
                    row["candidate_evidence_recall"]
                    for row in subset
                    if row.get("candidate_evidence_recall") is not None
                ),
                4,
            ) if any(row.get("candidate_evidence_recall") is not None for row in subset) else None,
            "evidence_recall": round(statistics.mean(row["evidence_recall"] for row in subset if row.get("evidence_recall") is not None), 4)
            if any(row.get("evidence_recall") is not None for row in subset) else None,
            "artifact_recall": round(
                statistics.mean(
                    row["artifact_recall"]
                    for row in subset
                    if row.get("artifact_recall") is not None
                ),
                4,
            ) if any(row.get("artifact_recall") is not None for row in subset) else None,
            "candidate_artifact_recall": round(
                statistics.mean(
                    row["candidate_artifact_recall"]
                    for row in subset
                    if row.get("candidate_artifact_recall") is not None
                ),
                4,
            ) if any(row.get("candidate_artifact_recall") is not None for row in subset) else None,
            "query_scoped_artifact_support": round(
                statistics.mean(
                    row["query_scoped_artifact_support"]
                    for row in subset
                    if row.get("query_scoped_artifact_support") is not None
                ),
                4,
            ) if any(row.get("query_scoped_artifact_support") is not None for row in subset) else None,
            "retrieval_support_recall": round(
                statistics.mean(
                    row["retrieval_support_recall"]
                    for row in subset
                    if row.get("retrieval_support_recall") is not None
                ),
                4,
            ) if any(row.get("retrieval_support_recall") is not None for row in subset) else None,
            "answer_token_coverage": round(
                statistics.mean(row["answer_token_coverage"] for row in subset if row.get("answer_token_coverage") is not None),
                4,
            ) if any(row.get("answer_token_coverage") is not None for row in subset) else None,
            "failures": len([row for row in rows if str(row.get("stratum") or "unknown") == stratum and row.get("status") != "ok"]),
        }
    by_profile: dict[str, dict[str, Any]] = {}
    for profile in sorted({str(row.get("document_profile") or "unknown") for row in rows}):
        subset = [
            row for row in successful
            if str(row.get("document_profile") or "unknown") == profile
        ]
        profile_page_values = [row["page_recall"] for row in subset if row.get("page_recall") is not None]
        profile_answer_values = [
            float(row["answer_token_coverage"])
            for row in subset
            if row.get("answer_token_coverage") is not None
        ]
        profile_retrieval_values = [
            float(row["retrieval_ms"])
            for row in subset
            if row.get("retrieval_ms") is not None
        ]
        by_profile[profile] = {
            "cases": len(subset),
            "unanswerable_cases": sum(1 for row in subset if row.get("unanswerable")),
            "page_recall": round(statistics.mean(profile_page_values), 4) if profile_page_values else None,
            "answer_token_coverage": round(statistics.mean(profile_answer_values), 4) if profile_answer_values else None,
            "retrieval_p50_ms": round(statistics.median(profile_retrieval_values), 3)
            if profile_retrieval_values else None,
            "retrieval_max_ms": round(max(profile_retrieval_values), 3)
            if profile_retrieval_values else None,
            "failures": len([
                row for row in rows
                if str(row.get("document_profile") or "unknown") == profile
                and row.get("status") != "ok"
            ]),
        }
    return {
        "schema_version": "public-retrieval-only-v3",
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_root": str(args.runtime_root),
        "embedding_model": settings.embedding_model,
        "evaluator_fingerprint": EVALUATOR_FINGERPRINT,
        "checkpoint_contract": contract,
        "resume_status": resume_status,
        "suite_complete": len(rows) == len(cases) and len(successful) == len(cases),
        "score_valid": len(rows) == len(cases) and len(successful) == len(cases),
        "selected_cases": len(cases),
        "no_llm_calls": True,
        "provider_preflight": provider_preflight,
        "cases": len(rows),
        "successful": len(successful),
        "failed": len(rows) - len(successful),
        "aggregate": {
            "positive_page_cases": len(page_values),
            "unanswerable_cases": sum(1 for row in successful if row.get("unanswerable")),
            "page_recall": round(statistics.mean(page_values), 4) if page_values else None,
            "candidate_page_recall": round(statistics.mean(candidate_page_values), 4)
            if candidate_page_values else None,
            "candidate_evidence_recall": round(statistics.mean(candidate_evidence_values), 4)
            if candidate_evidence_values else None,
            "evidence_recall": round(statistics.mean(evidence_values), 4) if evidence_values else None,
            "artifact_recall": round(statistics.mean(artifact_values), 4) if artifact_values else None,
            "candidate_artifact_recall": round(statistics.mean(candidate_artifact_values), 4)
            if candidate_artifact_values else None,
            "query_scoped_artifact_support": round(
                statistics.mean(query_scoped_artifact_values), 4
            ) if query_scoped_artifact_values else None,
            "retrieval_support_recall": round(statistics.mean(support_values), 4) if support_values else None,
            "answer_token_coverage": round(statistics.mean(answer_coverage_values), 4) if answer_coverage_values else None,
            "retrieval_p50_ms": round(statistics.median(retrieval_values), 3) if retrieval_values else None,
            "retrieval_max_ms": round(max(retrieval_values), 3) if retrieval_values else None,
        },
        "by_stratum": by_stratum,
        "by_document_profile": by_profile,
        "selection": {
            "mmlong_cases": args.mmlong_cases,
            "mmlong_per_profile": args.mmlong_per_profile,
            "mmlong_version": args.mmlong_version,
            "spiqa_cases": args.spiqa_cases,
            "disable_pdf_ocr": bool(args.disable_pdf_ocr),
            "case_ids_file": str(args.case_ids_file) if args.case_ids_file else None,
            "case_ids": [case.case_id for case in cases],
        },
        "failures_by_type": dict(Counter(str(row.get("error_type") or "unknown") for row in rows if row.get("status") != "ok")),
        "rows": rows,
        "resumed_rows": len(completed_ids),
        "checkpoint_path": str(checkpoint_path),
        "note": (
            "Retrieval-only diagnostic. Official evidence annotations are used only after retrieval; "
            "phrase recall, final-context artifact recall and first-stage candidate artifact recall "
            "are reported separately so composition loss is not confused with retrieval loss. "
            "No LLM calls or production paths are used."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mmlong-cases", type=int, default=8)
    parser.add_argument(
        "--mmlong-version",
        choices=("v1", "v2"),
        default="v2",
        help="MMLongBench-Doc annotation release; v2 is recommended for new runs",
    )
    parser.add_argument(
        "--mmlong-per-profile",
        type=int,
        default=0,
        help="Select this many document-balanced cases from each native_text/mixed_text_image/image_only profile; overrides --mmlong-cases.",
    )
    parser.add_argument(
        "--case-ids-file",
        type=Path,
        default=None,
        help=(
            "optional JSON list or {case_ids: [...]} manifest; when present, "
            "run exactly this ordered frontier for pairing with another evaluator"
        ),
    )
    parser.add_argument("--spiqa-cases", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--disable-pdf-ocr",
        action="store_true",
        help=(
            "skip derived OCR in the isolated retrieval-only run; useful for "
            "separating native text retrieval from cold scan recovery"
        ),
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=PROJECT_ROOT / "data/retrieval_eval/public/results/runtime/public-retrieval-only-v2",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/retrieval_eval/public/results/public-retrieval-only-v2.json",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore an interrupted-run checkpoint next to --output.",
    )
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    args.runtime_root.mkdir(parents=True, exist_ok=True)
    try:
        report = asyncio.run(run(args))
    except EmbeddingError as exc:
        # Provider availability is an environment precondition, not a
        # per-document retrieval miss.  Emit a structured fail-closed report
        # so callers do not mistake an offline run for a scored benchmark.
        report = {
            "schema_version": "public-retrieval-only-v2",
            "created_at": datetime.now(UTC).isoformat(),
            "status": "provider_unavailable",
            "score_valid": False,
            "embedding_model": get_settings().embedding_model,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "note": "No aggregate score was emitted; start the configured local embedding provider and rerun.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2
    except KeyboardInterrupt:
        # A cold public PDF can legitimately outlive an interactive terminal.
        # Preserve an auditable checkpoint without emitting a misleading
        # aggregate or a traceback that looks like a retrieval failure.
        partial = args.output.with_suffix(args.output.suffix + ".partial")
        interrupted = {
            "schema_version": "public-retrieval-only-interrupted-v1",
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
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        key: report[key]
        for key in ("cases", "successful", "failed", "aggregate", "by_stratum", "by_document_profile")
    }, ensure_ascii=False, indent=2))
    return 0 if report["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
