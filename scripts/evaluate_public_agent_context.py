#!/usr/bin/env python3
"""Replay a fixed SPIQA frontier through the real agent retrieval helper.

No ingestion, rewrite, planner or generation; therefore not full agent E2E.
Questions and post-selection qrels are read from a preserved diagnostic run.
This tests production context parity, not semantic answer quality.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "scripts")]
from app.api.agent import _retrieve_legacy_for_agent, _visual_attachment_candidates
from app.core.config import get_settings
from app.services.query_rewrite_service import (
    _classify_answer_depth, _classify_answer_intent, has_result_table_intent, has_visual_intent,
)
from app.services.rag_service import RagService
from evaluation_checkpoint import write_checkpoint
from evaluator_fingerprint import default_rag_dependency_files, fingerprint_files


def artifact_names(sources: list[dict]) -> list[str]:
    return list(dict.fromkeys(
        str((source.get("metadata") or {}).get("external_artifact_file")
            or Path(str(source.get("image_path") or "")).name)
        for source in sources if source.get("image_path")
    ))


def artifact_recall(selected: list[str], required: list[str]) -> float | None:
    gold = set(required)
    return len(gold.intersection(selected)) / len(gold) if gold else None


def forbidden_model_call(*args, **kwargs):
    raise AssertionError("Retrieval-only context evaluation attempted an LLM call")


def catalog_snapshot(db_path: Path) -> list[dict]:
    with sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute(
            "SELECT id, filename, active_revision_id FROM documents ORDER BY id"
        )]


def artifact_identity_names(db_path: Path) -> dict[str, str]:
    """Read-only locator map for scoring after context selection."""
    result = {}
    with sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        for table in ("document_tables", "document_figures"):
            for artifact_id, metadata in connection.execute(f"SELECT id, metadata_json FROM {table}"):
                info = json.loads(metadata or "{}")
                name = info.get("external_artifact_file")
                if name:
                    # Image-backed tables may physically reside in figures.
                    # Map the locator, not the storage type or a new qrel.
                    for kind in ("table", "figure"):
                        result[f"{kind}:{artifact_id}"] = str(name)
    return result


def composer_artifact_boundaries(trace: dict, names: dict[str, str], required: list[str]) -> dict:
    events = trace.get("events") or []
    observed = {}
    for event in events:
        name = names.get(event.get("evidence_id"))
        if name:
            observed.setdefault(name, []).append(event["reason"])
    return {name: ("selected_in_composer" if "selected" in observed.get(name, [])
                   else sorted(set(observed[name])) if name in observed
                   else "absent_from_composer_input" if trace.get("complete")
                   else "unobserved_in_incomplete_trace") for name in required}


async def run(args) -> dict:
    runtime = args.runtime_root.resolve()
    if not (runtime / "evaluation-runtime.json").is_file():
        raise ValueError("Requires an existing owned isolated evaluation clone")
    baseline = json.loads(args.baseline.read_text())
    cases = baseline["rows"]
    if len(cases) != baseline["cases"] or len({r["case_id"] for r in cases}) != len(cases):
        raise ValueError("Incomplete or duplicate baseline frontier")
    settings = get_settings().model_copy(update={
        "data_dir": runtime, "dual_query_rrf_enabled": False,
        "visual_retrieval_enabled": False, "lightrag_enabled": False,
    })
    before = catalog_snapshot(settings.sqlite_db_path)
    identity_names = artifact_identity_names(settings.sqlite_db_path)
    by_filename = {row["filename"]: row["id"] for row in before}
    rag = RagService(settings.sqlite_db_path, artifact_root=settings.artifacts_path)
    rows = []
    report = {
        "schema_version": "public-agent-context-parity-v2", "cases": len(cases),
        "created_at": datetime.now(UTC).isoformat(), "runtime_root": str(runtime),
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "evaluator_fingerprint": fingerprint_files(ROOT, [Path(__file__),
            ROOT / "backend/app/api/agent.py", *default_rag_dependency_files(ROOT)]),
        "no_llm_calls": True, "promotion_eligible": False,
        "note": "Raw query, production helper only; no rewrite/planner/generation. Qrels are scoring-only; potential pixels are not uploaded.",
        "rows": rows,
    }
    # Both async and sync caller paths raise rather than returning a fake
    # completion. The embedding provider remains available locally.
    with patch("app.llm.openai_client.OpenAICompatibleClient.chat", forbidden_model_call), \
         patch("app.llm.openai_client.OpenAICompatibleClient.stream_chat", forbidden_model_call), \
         patch("app.llm.ollama_client.OllamaClient.chat", forbidden_model_call):
        for case in cases:
            if not case["case_id"].startswith("spiqa-c:"):
                raise ValueError("This preserved parity frontier supports SPIQA only")
            document = case["case_id"].split(":")[1]
            filename = "spiqa-" + document.replace(".", "_") + ".md"
            question = case["question"]
            started = time.perf_counter()
            result = await _retrieve_legacy_for_agent(
                rag=rag, settings=settings, query=question, collection_id=None,
                retrieval_mode="hybrid", focus_document_ids=[by_filename[filename]],
                answer_intent=_classify_answer_intent(question),
                answer_depth=_classify_answer_depth(question),
                include_visual_boost=has_visual_intent(question),
                prefer_tables=has_result_table_intent(question),
            )
            sources = result["documents"]
            names = artifact_names(sources)
            pixels = _visual_attachment_candidates(sources, enabled=True, max_images=4, query=question)
            pixel_names = artifact_names([source for _path, _mime, source in pixels])
            # Gold is deliberately read only after retrieval/projection.
            required = case["required_artifacts"]
            selection_trace = result.get("context_stats", {}).get("selection_trace", {})
            row = {
                "case_id": case["case_id"], "question": question,
                "selected_artifacts": names, "potential_pixel_artifacts": pixel_names,
                "required_artifacts": required,
                "selected_artifact_recall": artifact_recall(names, required),
                "potential_pixel_recall": artifact_recall(pixel_names, required),
                "source_count": len(sources), "retrieval_ms": round((time.perf_counter()-started)*1000, 3),
                "composer_selection_trace": selection_trace,
                "required_artifact_composer_boundary": composer_artifact_boundaries(
                    selection_trace, identity_names, required),
            }
            rows.append(row)
            write_checkpoint(args.output.with_suffix(".partial.json"), report)
            print(json.dumps({key: row[key] for key in (
                "case_id", "selected_artifact_recall", "source_count", "retrieval_ms",
                "required_artifact_composer_boundary",
            )}), flush=True)
    if catalog_snapshot(settings.sqlite_db_path) != before:
        raise RuntimeError("Active document catalog changed during retrieval-only replay")
    report.update({
        "selected_artifact_recall": statistics.mean(r["selected_artifact_recall"] for r in rows if r["selected_artifact_recall"] is not None),
        "potential_pixel_recall": statistics.mean(r["potential_pixel_recall"] for r in rows if r["potential_pixel_recall"] is not None),
        "retrieval_p50_ms": statistics.median(r["retrieval_ms"] for r in rows),
        "catalog_unchanged": True,
    })
    write_checkpoint(args.output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve completed reports; choose a new output")
    report = asyncio.run(run(args))
    print(json.dumps({k:v for k,v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
