#!/usr/bin/env python3
"""Build a sealed, answer-free public shadow manifest.

The manifest selects genuine MMLongBench-Doc/SPIQA cases from documents that
are absent from the visible Gate-E frontier.  It stores only stable case IDs
and document fingerprints; public answers/evidence annotations remain in the
upstream dataset and are never copied into the tracked manifest.  Selection
is deterministic and based on case/document IDs plus format strata, not on
answer text or retrieval outcomes.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "scripts", ROOT / "backend"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from public_multimodal_eval_lib import (  # noqa: E402
    BenchmarkCase,
    load_mmlong_cases,
    load_spiqa_test_c_cases,
    mmlong_document_profile,
    SPIQA_TEST_C_IMAGES,
)


BUCKETS = (
    ("mmlong:native_text", "mmlongbench-doc", "native_text"),
    ("mmlong:mixed_text_image", "mmlongbench-doc", "mixed_text_image"),
    ("mmlong:image_only", "mmlongbench-doc", "image_only"),
    ("spiqa:figure", "spiqa-test-c", "figure"),
    ("spiqa:table", "spiqa-test-c", "table"),
    ("spiqa:figure+table", "spiqa-test-c", "figure+table"),
)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_fingerprint(case: BenchmarkCase) -> tuple[str, str]:
    if case.source_path is not None and case.source_path.is_file():
        return "source_sha256", _hash_file(case.source_path)
    # SPIQA distributes extracted paper text and image annotations rather than
    # a PDF in this checkout.  Hash the canonical extracted adapter text as a
    # stable content fingerprint; no answer/evidence field participates.
    return "extracted_content_sha256", hashlib.sha256(
        case.full_text.encode("utf-8")
    ).hexdigest()


def _artifact_filename_fingerprint(filename: str) -> str:
    return hashlib.sha256(" ".join(str(filename).split()).casefold().encode("utf-8")).hexdigest()


def _artifact_image_sha256(filename: str) -> str | None:
    """Hash a SPIQA image member when the public archive is available."""

    normalized = str(filename or "").strip()
    if not normalized or not SPIQA_TEST_C_IMAGES.is_file():
        return None
    try:
        with ZipFile(SPIQA_TEST_C_IMAGES) as archive:
            candidates = [
                name for name in archive.namelist()
                if name.casefold().endswith(normalized.casefold())
            ]
            if len(candidates) != 1:
                return None
            return hashlib.sha256(archive.read(candidates[0])).hexdigest()
    except (OSError, KeyError, ValueError):
        return None


def _sort_key(case: BenchmarkCase) -> str:
    return hashlib.sha256(
        f"shadow-v3:{case.suite}:{case.document_id}:{case.case_id}".encode("utf-8")
    ).hexdigest()


def _bucket(case: BenchmarkCase) -> str | None:
    if case.suite == "mmlongbench-doc":
        return f"mmlong:{mmlong_document_profile(case)}"
    return f"spiqa:{case.stratum}"


def select_shadow_cases(
    cases: Iterable[BenchmarkCase],
    *,
    excluded_case_ids: set[str],
    excluded_document_ids: set[str],
    per_bucket: int = 4,
) -> list[BenchmarkCase]:
    if per_bucket < 1:
        raise ValueError("per_bucket must be positive")
    grouped: dict[str, list[BenchmarkCase]] = {key: [] for key, _s, _p in BUCKETS}
    for case in cases:
        if case.case_id in excluded_case_ids or case.document_id in excluded_document_ids:
            continue
        key = _bucket(case)
        if key in grouped:
            grouped[key].append(case)

    selected: list[BenchmarkCase] = []
    selected_documents: set[str] = set()
    for key, _suite, _profile in BUCKETS:
        candidates = sorted(grouped[key], key=_sort_key)
        count = 0
        for case in candidates:
            if case.document_id in selected_documents:
                continue
            selected.append(case)
            selected_documents.add(case.document_id)
            count += 1
            if count >= per_bucket:
                break
        if count < per_bucket:
            raise ValueError(
                f"shadow-v3 bucket {key!r} has only {count} eligible documents; "
                f"need {per_bucket}"
            )
    if len(selected) != per_bucket * len(BUCKETS):
        raise AssertionError("shadow-v3 selection cardinality mismatch")
    return selected


def build_manifest(
    *,
    output: Path,
    exclude_manifest: Path,
    per_bucket: int = 4,
    questions_output: Path | None = None,
    qrels_output: Path | None = None,
) -> dict:
    excluded_ids_raw = json.loads(exclude_manifest.read_text(encoding="utf-8"))
    if isinstance(excluded_ids_raw, dict):
        excluded_ids_raw = excluded_ids_raw.get("case_ids") or []
    excluded_case_ids = {str(value).strip() for value in excluded_ids_raw if str(value).strip()}
    all_cases = [*load_mmlong_cases(version="v2"), *load_spiqa_test_c_cases()]
    excluded_document_ids = {
        case.document_id for case in all_cases if case.case_id in excluded_case_ids
    }
    selected = select_shadow_cases(
        all_cases,
        excluded_case_ids=excluded_case_ids,
        excluded_document_ids=excluded_document_ids,
        per_bucket=per_bucket,
    )

    documents: dict[str, dict] = {}
    case_rows: list[dict] = []
    for case in selected:
        field, fingerprint = _content_fingerprint(case)
        row = {
            "case_id": case.case_id,
            "suite": case.suite,
            "document_id": case.document_id,
            "question": case.question,
            "answer_format": case.answer_format,
            "bucket": _bucket(case),
            "profile": mmlong_document_profile(case)
            if case.suite == "mmlongbench-doc"
            else case.stratum,
            "content_fingerprint_field": field,
            "content_fingerprint": fingerprint,
            "source_filename": case.source_path.name if case.source_path else None,
        }
        case_rows.append(row)
        documents.setdefault(
            case.document_id,
            {
                "document_id": case.document_id,
                "suite": case.suite,
                "source_filename": row["source_filename"],
                "content_fingerprint_field": field,
                "content_fingerprint": fingerprint,
                "case_ids": [],
            },
        )["case_ids"].append(case.case_id)

    manifest = {
        "schema_version": "shadow-v3-manifest-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "sealed": True,
        "opened": False,
        "open_policy": "run only with explicit --open-shadow after RC freeze",
        "selection_policy": {
            "per_bucket": per_bucket,
            "bucket_count": len(BUCKETS),
            "excluded_manifest": str(exclude_manifest.relative_to(ROOT)),
            "answer_or_qrel_fields_used": False,
        },
        "case_ids": [row["case_id"] for row in case_rows],
        "cases": case_rows,
        "documents": list(documents.values()),
        "qrels_location": "data/retrieval_eval/private/shadow-v3.qrels.jsonl",
        "note": "Sealed manifest only; no answer or qrel content is tracked here.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if questions_output is not None:
        questions_output.parent.mkdir(parents=True, exist_ok=True)
        questions_output.write_text(
            "".join(json.dumps({key: row[key] for key in (
                "case_id", "suite", "document_id", "question", "answer_format",
                "bucket", "profile",
            )}, ensure_ascii=False) + "\n" for row in case_rows),
            encoding="utf-8",
        )
    if qrels_output is not None:
        qrels_output.parent.mkdir(parents=True, exist_ok=True)
        qrel_rows: list[dict] = []
        for case in selected:
            document = documents[case.document_id]
            groups: list[dict] = []
            if case.suite == "mmlongbench-doc":
                for page in case.evidence_pages:
                    groups.append(
                        {
                            "kind": "page",
                            "page_number": int(page),
                            "evidence_sources": list(case.evidence_sources),
                        }
                    )
            else:
                artifact_by_name = {
                    str(item.get("file") or "").strip(): item
                    for item in case.artifacts
                    if str(item.get("file") or "").strip()
                }
                for filename in case.referred_artifacts:
                    item = artifact_by_name.get(filename) or {}
                    lower = filename.casefold()
                    kind = "table" if "table" in lower else "figure"
                    group = {
                        "kind": kind,
                        "artifact_filename": filename,
                        "artifact_filename_sha256": _artifact_filename_fingerprint(filename),
                    }
                    caption = str(item.get("caption") or "").strip()
                    if caption:
                        group["caption_sha256"] = hashlib.sha256(
                            " ".join(caption.split()).casefold().encode("utf-8")
                        ).hexdigest()
                    image_sha256 = _artifact_image_sha256(filename)
                    if image_sha256:
                        group["image_sha256"] = image_sha256
                    groups.append(group)
            is_unanswerable = any(
                str(answer).strip().casefold() == "not answerable"
                for answer in case.answers
            )
            if not groups and not is_unanswerable:
                raise ValueError(f"shadow-v3 qrel has no evidence groups: {case.case_id}")
            qrel = {
                    "schema_version": "shadow-v3-qrel-v1",
                    "case_id": case.case_id,
                    "suite": case.suite,
                    "document_id": case.document_id,
                    "document_fingerprint_field": document["content_fingerprint_field"],
                    "document_fingerprint": document["content_fingerprint"],
                    "evidence_groups": groups,
                    "positive": not is_unanswerable,
                    "expected_abstention": is_unanswerable,
                }
            qrel_rows.append(qrel)
        qrels_output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in qrel_rows),
            encoding="utf-8",
        )
        qrel_digest = _hash_file(qrels_output)
        manifest["qrels_sha256"] = qrel_digest
        manifest["qrels_case_count"] = len(qrel_rows)
        # The qrels themselves remain under ignored/private data; only their
        # checksum is recorded in the tracked manifest for reproducibility.
        output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evals" / "retrieval_v2" / "shadow-v3.manifest.json",
    )
    parser.add_argument(
        "--qrels-output",
        type=Path,
        default=ROOT / "data" / "retrieval_eval" / "private" / "shadow-v3.qrels.jsonl",
    )
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        default=ROOT / "evals" / "retrieval_v2" / "gate-e-visual-frontier-v2.case-ids.json",
    )
    parser.add_argument(
        "--questions-output",
        type=Path,
        default=ROOT / "evals" / "retrieval_v2" / "shadow-v3.questions.jsonl",
    )
    parser.add_argument("--per-bucket", type=int, default=4)
    args = parser.parse_args()
    manifest = build_manifest(
        output=args.output,
        exclude_manifest=args.exclude_manifest,
        per_bucket=args.per_bucket,
        questions_output=args.questions_output,
        qrels_output=args.qrels_output,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "questions_output": str(args.questions_output),
                "qrels_output": str(args.qrels_output),
                "qrels_case_count": manifest.get("qrels_case_count", 0),
                "cases": len(manifest["case_ids"]),
                "documents": len(manifest["documents"]),
                "buckets": sorted({row["bucket"] for row in manifest["cases"]}),
                "opened": manifest["opened"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
