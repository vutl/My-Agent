#!/usr/bin/env python3
"""Bounded three-mode answer diagnostic with pixel telemetry and fail-stop calls.

Reuses the existing production-shaped ingestion/retrieval and mode-input helpers.
This visible probe neither promotes a provider nor substitutes for shadow tests.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

from app.core.config import get_settings
from app.llm.openai_client import OpenAICompatibleClient
from evaluate_public_multimodal_three_mode import (
    APPROVED_9ROUTER_MODELS, IsolatedAyaPipeline, _generate, _mode_input, _score,
    _answer_quality_comparison, _bounded_nonblocking, build_eval_prompt, _safe_runtime_root,
    ensure_public_output_path,
)
from evaluation_checkpoint import write_checkpoint
from evaluator_fingerprint import fingerprint_files, default_rag_dependency_files
from public_multimodal_eval_lib import (
    load_mmlong_cases, load_spiqa_test_c_cases, resolve_case_ids,
    mmlong_document_profile, read_spiqa_images,
    SPIQA_IMAGE_INPUT_POLICY, supplied_image_provenance,
)


MODES = ("gold_evidence", "full_extracted_document", "aya_pipeline")
PROJECT = Path(__file__).resolve().parents[1]


def select_cases(frontier: list[Any], limit: int) -> list[Any]:
    """Stratify on input modality/profile only; never inspect answers or scores."""
    if limit < 1 or limit > len(frontier):
        raise ValueError("Probe limit must fit the frozen public frontier")
    buckets: dict[str, list[Any]] = defaultdict(list)
    for case in frontier:
        key = ("pdf:" + mmlong_document_profile(case) if case.suite == "mmlongbench-doc"
               else "spiqa:" + case.stratum)
        buckets[key].append(case)
    for values in buckets.values():
        values.sort(key=lambda case: hashlib.sha256(case.case_id.encode()).hexdigest())
    selected = []
    while len(selected) < limit:
        for key in sorted(buckets):
            if buckets[key] and len(selected) < limit:
                selected.append(buckets[key].pop(0))
    return selected


def source_fingerprint(cases: list[Any]) -> str:
    h = hashlib.sha256()
    seen = set()
    for case in cases:
        # Labels are hashed for reproducibility, never copied into Aya/full input.
        h.update(json.dumps(asdict(case), sort_keys=True, default=str).encode())
        if case.document_id in seen:
            continue
        seen.add(case.document_id)
        if case.source_path:
            with case.source_path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    h.update(block)
        else:
            for name, payload in read_spiqa_images(case, [a["file"] for a in case.artifacts]):
                h.update(name.encode())
                h.update(payload)
    return h.hexdigest()


async def run(args: Any) -> dict:
    if args.model not in APPROVED_9ROUTER_MODELS:
        raise ValueError("Unapproved model")
    manifest = json.loads(args.case_ids_file.read_text())
    ids = manifest.get("case_ids") if isinstance(manifest, dict) else manifest
    frontier = resolve_case_ids([*load_mmlong_cases(version="v2"), *load_spiqa_test_c_cases()], ids)
    cases = select_cases(frontier, args.limit)
    settings = get_settings()
    if settings.llm_provider != "openai_compatible":
        raise ValueError("Probe requires the approved 9router client")
    pipeline = IsolatedAyaPipeline(runtime_root=_safe_runtime_root(args.runtime_root),
                                   settings=settings)
    code_files = [Path(__file__), Path(__file__).with_name("evaluate_public_multimodal_three_mode.py"),
                  Path(__file__).with_name("public_multimodal_eval_lib.py"),
                  Path(__file__).with_name("evaluation_checkpoint.py"),
                  *default_rag_dependency_files(PROJECT)]
    contract = {
        "code": fingerprint_files(PROJECT, code_files), "sources": source_fingerprint(cases),
        "model": args.model, "case_ids": [c.case_id for c in cases], "modes": list(MODES),
        "runtime": str(pipeline.root.resolve()), "max_context_chars": args.max_context_chars,
        "max_context_tokens": args.max_context_tokens, "max_full_images": args.max_full_images,
        "timeout": args.timeout, "seed": args.seed, "top_k": args.top_k,
        "embedding": settings.embedding_model,
        "query_prefix": settings.embedding_query_prefix,
        "document_prefix": settings.embedding_document_prefix,
        "rerank": pipeline.rerank_enabled, "rerank_mode": pipeline.rerank_mode,
        "materialization_policy": pipeline.indexing.materialization_policy("pdf"),
        "spiqa_image_input_policy": SPIQA_IMAGE_INPUT_POLICY,
    }
    ordered = []
    rng = random.Random(args.seed)
    for case in cases:
        modes = list(MODES)
        rng.shuffle(modes)
        ordered.extend((case, mode) for mode in modes)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    rows = []
    if partial.exists():
        if not args.resume:
            raise ValueError("Existing probe checkpoint; use --resume or a new output")
        saved = json.loads(partial.read_text())
        if saved.get("contract") != contract:
            raise ValueError("Probe contract changed")
        if saved.get("inflight") or saved.get("error"):
            raise ValueError("Interrupted/failed provider attempt requires explicit new audited run")
        rows = saved.get("rows", [])
        actual = [(r["case_id"], r["mode"]) for r in rows]
        if actual != [(c.case_id, m) for c, m in ordered[:len(rows)]]:
            raise ValueError("Checkpoint is not an ordered case/mode prefix")
    elif args.resume:
        raise ValueError("Missing probe checkpoint")
    state = {"schema_version": "public-answer-probe-v1", "contract": contract,
             "rows": rows, "inflight": None, "error": None, "diagnostic_only": True,
             "selected_case_ids": contract["case_ids"], "complete": False,
             "selection_uses_answers_or_scores": False,
             "expected_generations": len(ordered), "completed_generations": len(rows),
             "limitations": [
                 "Visible bounded diagnostic, not shadow or promotion evidence",
                 "SPIQA token F1 is diagnostic, not semantic correctness",
                 "Full mode is bounded extracted context, not ChatGPT PDF upload",
                 "Aya uses the same no-LLM local retrieval baseline as public retrieval-only",
                 "No agent routing, persona, answer claim guard or live query planner in this probe",
             ]}
    write_checkpoint(partial, state)
    client = OpenAICompatibleClient(base_url=settings.openai_api_base,
                                    api_key=settings.openai_api_key,
                                    timeout_seconds=args.timeout)
    stage = "preflight"
    try:
        health = await client.health()
        if not health.get("reachable") or args.model not in set(health.get("models") or []):
            raise RuntimeError("Approved 9router model is unavailable")
        await pipeline.preflight_local_search()
        for case, mode in ordered[len(rows):]:
            state["inflight"] = {"case_id": case.case_id, "mode": mode}
            write_checkpoint(partial, state)
            stage = "retrieval_prepare"
            if mode == "aya_pipeline":
                retrieved = await _bounded_nonblocking(pipeline.retrieve(case, top_k=args.top_k), args.timeout)
                context, images = retrieved.pop("context"), retrieved.pop("images")
                meta = {"context_label": "aya_production_retrieval", **retrieved}
            else:
                context, images, meta = _mode_input(
                    case, mode=mode, max_context_chars=args.max_context_chars,
                    max_context_tokens=args.max_context_tokens,
                    max_full_images=args.max_full_images, output_reserve_tokens=16000)
            prompt = build_eval_prompt(question=case.question, context=context,
                                      context_label=meta["context_label"],
                                      answer_format=case.answer_format,
                                      visual_context=bool(images))
            stage = "generation"
            generated = await _bounded_nonblocking(_generate(
                client=client, model=args.model, prompt=prompt, images=images), args.timeout)
            rows.append({"status": "ok", "case_id": case.case_id, "mode": mode,
                         "suite": case.suite, "stratum": case.stratum,
                         "question": case.question, "answer": generated["answer"],
                         "score": _score(case, generated["answer"]), "generation": generated,
                         "input": {**meta, "context_chars": len(context),
                                   "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
                                   "supplied_images": [supplied_image_provenance(n, b)
                                                        for n, b in images]}})
            state.update(inflight=None, completed_generations=len(rows))
            write_checkpoint(partial, state)
            print(f"[{len(rows)}/{len(ordered)}] {case.case_id} {mode}", file=sys.stderr, flush=True)
        if fingerprint_files(PROJECT, code_files) != contract["code"] or source_fingerprint(cases) != contract["sources"]:
            raise RuntimeError("Code/source changed during probe")
        state["complete"] = True
        state["paired_quality"] = _answer_quality_comparison(cases, rows, MODES)
    except Exception as exc:
        # No continuation after any provider/preparation failure, no model
        # fallback and no replacement of the failed attempt on resume.
        state["error"] = {"type": type(exc).__name__, "stage": stage}
    write_checkpoint(partial, state)
    write_checkpoint(args.output, state)
    return state


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-ids-file", type=Path, required=True)
    p.add_argument("--runtime-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=12)
    p.add_argument("--model", default="cx/gpt-5.6-sol")
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--max-context-chars", type=int, default=60000)
    p.add_argument("--max-context-tokens", type=int, default=16000)
    p.add_argument("--max-full-images", type=int, default=4)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--seed", type=int, default=20261005)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--approve-public-9router-upload", action="store_true")
    args = p.parse_args()
    if not args.approve_public_9router_upload:
        p.error("Explicit public upload acknowledgement required")
    args.output = ensure_public_output_path(args.output)
    result = asyncio.run(run(args))
    print(json.dumps({k: result.get(k) for k in ("complete", "completed_generations", "error", "paired_quality")}, indent=2))
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
