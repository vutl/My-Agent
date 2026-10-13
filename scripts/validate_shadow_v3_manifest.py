#!/usr/bin/env python3
"""Fail-closed validation for the sealed shadow-v3 public frontier."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(
    manifest_path: Path,
    questions_path: Path,
    qrels_path: Path,
    *,
    open_shadow: bool = False,
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "shadow-v3-manifest-v1":
        raise ValueError("unsupported shadow manifest schema")
    if manifest.get("opened") is not False and not open_shadow:
        raise ValueError("shadow-v3 is already open or has an invalid opened state")
    if not manifest.get("sealed"):
        raise ValueError("shadow-v3 manifest is not sealed")

    questions = [
        json.loads(line)
        for line in questions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    qrels = [
        json.loads(line)
        for line in qrels_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    case_ids = [str(value) for value in manifest.get("case_ids") or []]
    question_ids = [str(row.get("case_id") or "") for row in questions]
    qrel_ids = [str(row.get("case_id") or "") for row in qrels]
    if len(case_ids) != 24 or len(set(case_ids)) != 24:
        raise ValueError("shadow-v3 must contain exactly 24 unique case IDs")
    if question_ids != case_ids or qrel_ids != case_ids:
        raise ValueError("manifest/questions/qrels case order does not match")
    documents = manifest.get("documents") or []
    document_ids = [str(row.get("document_id") or "") for row in documents]
    if len(document_ids) != 24 or len(set(document_ids)) != 24:
        raise ValueError("shadow-v3 must contain exactly 24 unique documents")
    if any(not str(row.get("question") or "").strip() for row in questions):
        raise ValueError("shadow-v3 contains an empty question")
    expected_buckets = {
        "mmlong:native_text",
        "mmlong:mixed_text_image",
        "mmlong:image_only",
        "spiqa:figure",
        "spiqa:table",
        "spiqa:figure+table",
    }
    bucket_counts: dict[str, int] = {}
    for row in questions:
        bucket = str(row.get("bucket") or "")
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    if set(bucket_counts) != expected_buckets or set(bucket_counts.values()) != {4}:
        raise ValueError(f"shadow-v3 bucket matrix mismatch: {bucket_counts}")
    if any("answer" in row or "evidence" in row or "qrel" in row for row in questions):
        raise ValueError("tracked shadow questions contain answer/evidence fields")
    if manifest.get("qrels_case_count") != len(qrels):
        raise ValueError("qrel case count mismatch")
    expected_qrel_hash = str(manifest.get("qrels_sha256") or "")
    if not expected_qrel_hash or _sha256(qrels_path) != expected_qrel_hash:
        raise ValueError("private qrel checksum mismatch")
    positive = 0
    negative = 0
    for row in qrels:
        groups = row.get("evidence_groups") or []
        if bool(row.get("positive")):
            if not groups:
                raise ValueError(f"positive qrel has no evidence groups: {row['case_id']}")
            positive += 1
        else:
            if not row.get("expected_abstention"):
                raise ValueError(f"negative qrel is not an abstention: {row['case_id']}")
            negative += 1
    if positive < 1 or negative < 1:
        raise ValueError("shadow-v3 must retain both positive and abstention cases")
    result = {
        "manifest": str(manifest_path),
        "questions": len(questions),
        "documents": len(documents),
        "positive_qrels": positive,
        "abstention_qrels": negative,
        "bucket_counts": bucket_counts,
        "opened": bool(open_shadow),
        "qrels_sha256": expected_qrel_hash,
        "source_fingerprints_present": all(
            str(row.get("content_fingerprint") or "").strip() for row in documents
        ),
    }
    if not result["source_fingerprints_present"]:
        raise ValueError("shadow-v3 document fingerprint missing")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "evals/retrieval_v2/shadow-v3.manifest.json")
    parser.add_argument("--questions", type=Path, default=ROOT / "evals/retrieval_v2/shadow-v3.questions.jsonl")
    parser.add_argument("--qrels", type=Path, default=ROOT / "data/retrieval_eval/private/shadow-v3.qrels.jsonl")
    parser.add_argument("--open-shadow", action="store_true")
    args = parser.parse_args()
    result = validate(args.manifest, args.questions, args.qrels, open_shadow=args.open_shadow)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
