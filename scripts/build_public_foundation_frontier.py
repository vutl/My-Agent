#!/usr/bin/env python3
"""Freeze a larger public dev frontier without answer-based selection."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re

from evaluation_checkpoint import write_checkpoint
from public_multimodal_eval_lib import (
    BenchmarkCase, load_mmlong_cases, load_spiqa_test_c_cases,
    mmlong_document_profile, stable_mmlong_case_sample, stable_spiqa_case_sample,
)


def document_family(document_id: str) -> str:
    name = Path(document_id).name.casefold()
    return re.sub(r"v\d+(?=\.pdf$|$)", "", name)


def previous_documents(case_ids: set[str], pool: list[BenchmarkCase]) -> set[str]:
    by_id = {case.case_id: case for case in pool}
    unknown = case_ids - by_id.keys()
    if unknown:
        raise ValueError("unresolved previous frontier cases: " + ", ".join(sorted(unknown)))
    return {by_id[case_id].document_id for case_id in case_ids}


def select_frontier(
    mmlong: list[BenchmarkCase], spiqa: list[BenchmarkCase], *,
    excluded_documents: set[str], total_cases: int = 60,
    mmlong_documents_per_profile: int = 4, mmlong_cases_per_document: int = 2,
) -> list[BenchmarkCase]:
    excluded = {document_family(value) for value in excluded_documents}
    eligible_mmlong = [case for case in mmlong if document_family(case.document_id) not in excluded]
    eligible_spiqa = [case for case in spiqa if document_family(case.document_id) not in excluded]
    selected_pdf = stable_mmlong_case_sample(
        eligible_mmlong, per_profile=mmlong_documents_per_profile,
        cases_per_document=mmlong_cases_per_document,
    )
    if not selected_pdf or len(selected_pdf) >= total_cases:
        raise ValueError("frontier must include both public corpora")
    selected_spiqa = stable_spiqa_case_sample(
        eligible_spiqa, limit=total_cases - len(selected_pdf), cases_per_document=1,
    )
    # Fast scientific adapters run first; long PDF ingestion remains in the
    # same frozen population and may not be dropped after seeing failures.
    selected = [*selected_spiqa, *selected_pdf]
    ids = [case.case_id for case in selected]
    if len(selected) != total_cases or len(set(ids)) != len(ids):
        raise ValueError("insufficient or duplicate public frontier")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=60)
    parser.add_argument("--previous-frontier", type=Path, default=Path("evals/retrieval_v2/gate-e-visual-frontier-v2.case-ids.json"))
    parser.add_argument("--shadow-manifest", type=Path, default=Path("evals/retrieval_v2/shadow-v3.manifest.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to replace a frozen frontier; choose a new version")
    shadow = json.loads(args.shadow_manifest.read_text())
    # Only public document reservations are read, never sealed questions/qrels.
    reserved = {str(row["document_id"]) for row in shadow["documents"]}
    prior = json.loads(args.previous_frontier.read_text())
    prior_ids = set(prior.get("case_ids", []) if isinstance(prior, dict) else prior)
    spiqa = load_spiqa_test_c_cases()
    legacy = load_mmlong_cases(version="v1")
    mmlong = load_mmlong_cases(version="v2")
    previous = previous_documents(prior_ids, [*legacy, *mmlong, *spiqa])
    cases = select_frontier(mmlong, spiqa, excluded_documents=reserved | previous, total_cases=args.cases)
    suite_hash = hashlib.sha256(json.dumps(
        [asdict(case) for case in cases], sort_keys=True, default=str,
    ).encode()).hexdigest()
    manifest = {
        "schema_version": "public-foundation-dev-frontier-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "purpose": "visible public dev; not sealed shadow or production acceptance by itself",
        "selection_policy": {
            "answer_based_selection": False,
            "ranking_based_selection": False,
            "document_disjoint_from_previous_frontier": True,
            "document_family_disjoint_from_shadow": True,
            "mmlong_version": "v2",
            "mmlong_documents_per_profile": 4,
            "mmlong_cases_per_document": 2,
            "spiqa_cases_per_document": 1,
            "ordering": "stable existing profile/stratum hash samplers; SPIQA then PDF",
        },
        "case_ids": [case.case_id for case in cases],
        "cases": len(cases),
        "documents": len({case.document_id for case in cases}),
        "source_annotation_sha256": suite_hash,
        "by_suite": dict(Counter(case.suite for case in cases)),
        "by_profile": dict(Counter(
            mmlong_document_profile(case) if case.source_path else "extracted_adapter"
            for case in cases
        )),
        "spiqa_strata": dict(Counter(case.stratum for case in cases if case.source_path is None)),
        "excluded_documents": sorted(reserved | previous),
        "exclusion_manifest_sha256": hashlib.sha256(args.shadow_manifest.read_bytes()).hexdigest(),
        "no_posthoc_selection": True,
        "limitations": ["No DOCX/MD/TXT external QA corpus in this frontier", "Retrieval-only is not agent or multimodal answer quality"],
    }
    write_checkpoint(args.output, manifest)
    print(json.dumps({key: manifest[key] for key in ("cases", "documents", "by_suite", "by_profile", "spiqa_strata")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
