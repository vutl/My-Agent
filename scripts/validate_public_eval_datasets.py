#!/usr/bin/env python3
"""Validate Aya's pinned public benchmark downloads without model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PUBLIC_ROOT = PROJECT_ROOT / "data" / "retrieval_eval" / "public"
MANIFEST_PATH = PUBLIC_ROOT / "manifest.json"


EXPECTED_COUNTS = {
    "mtrag_human_conversations": 110,
    "mtrag_synthetic_conversations": 200,
    "mtrag_un_reference_tasks": 507,
    "multichallenge_questions": 273,
    "mmlongbench_questions": 1082,
    "wildbench_v2_tasks": 1024,
    "chatrag_convfinqa": 1490,
    "chatrag_sqa": 3012,
    "chatrag_inscit": 502,
    "chatrag_doqa_cooking": 1797,
    "chatrag_doqa_movies": 1884,
    "chatrag_doqa_travel": 1713,
    "chatrag_hybridial": 1111,
    "spiqa_test_a_papers": 118,
    "spiqa_test_a_questions": 666,
    "spiqa_test_b_records": 65,
    "spiqa_test_b_questions": 228,
    "spiqa_test_c_records": 314,
    "spiqa_test_c_questions": 493,
}


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return ""
    return result.stdout.strip()


def collect_counts() -> dict[str, int]:
    raw = PUBLIC_ROOT / "raw"
    mtrag = raw / "github" / "mtrag"
    chatrag = raw / "huggingface" / "chatrag_bench" / "data"
    spiqa = raw / "huggingface" / "spiqa"

    test_a = _json(spiqa / "test-A" / "SPIQA_testA.json")
    test_b = _json(spiqa / "test-B" / "SPIQA_testB.json")
    test_c = _json(spiqa / "test-C" / "SPIQA_testC.json")

    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("pyarrow is required to validate WildBench parquet") from exc

    return {
        "mtrag_human_conversations": len(
            _json(mtrag / "mtrag-human" / "conversations" / "conversations.json")
        ),
        "mtrag_synthetic_conversations": len(
            _json(mtrag / "mtrag-synthetic" / "conversations" / "conversations.json")
        ),
        "mtrag_un_reference_tasks": len(
            _jsonl(mtrag / "mtragun-human" / "generation_tasks" / "reference.jsonl")
        ),
        "multichallenge_questions": len(
            _jsonl(
                raw
                / "github"
                / "multichallenge"
                / "data"
                / "benchmark_questions.jsonl"
            )
        ),
        "mmlongbench_questions": len(
            _json(raw / "github" / "mmlongbench_doc" / "data" / "samples.json")
        ),
        "wildbench_v2_tasks": parquet.read_metadata(
            raw
            / "huggingface"
            / "wildbench"
            / "v2"
            / "test-00000-of-00001.parquet"
        ).num_rows,
        "chatrag_convfinqa": len(_json(chatrag / "convfinqa" / "dev.json")),
        "chatrag_sqa": len(_json(chatrag / "sqa" / "test.json")),
        "chatrag_inscit": len(_json(chatrag / "inscit" / "dev.json")),
        "chatrag_doqa_cooking": len(_json(chatrag / "doqa" / "test_cooking.json")),
        "chatrag_doqa_movies": len(_json(chatrag / "doqa" / "test_movies.json")),
        "chatrag_doqa_travel": len(_json(chatrag / "doqa" / "test_travel.json")),
        "chatrag_hybridial": len(_json(chatrag / "hybridial" / "test.json")),
        "spiqa_test_a_papers": len(test_a),
        "spiqa_test_a_questions": sum(
            len(paper.get("qa") or []) for paper in test_a.values()
        ),
        "spiqa_test_b_records": len(test_b),
        "spiqa_test_b_questions": sum(
            len(item.get("question") or []) for item in test_b.values()
        ),
        "spiqa_test_c_records": len(test_c),
        "spiqa_test_c_questions": sum(
            len(item.get("question") or []) for item in test_c.values()
        ),
    }


def validate(*, full: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    manifest = _json(MANIFEST_PATH)

    revisions: dict[str, str] = {}
    for source in manifest["sources"]:
        relative = Path(source["local_path"])
        local_path = PUBLIC_ROOT / relative
        if not local_path.exists():
            errors.append(f"missing_source:{relative}")
            continue
        expected_revision = str(source.get("revision") or "")
        if relative.parts[:2] == ("raw", "github"):
            observed = _git_head(local_path)
            revisions[source["name"]] = observed
            if observed != expected_revision:
                errors.append(
                    f"revision_mismatch:{source['name']}:{observed}!={expected_revision}"
                )
        else:
            revisions[source["name"]] = expected_revision

    checked_hashes = 0
    for relative_text, expected in manifest["sha256"].items():
        relative = Path(relative_text)
        path = PUBLIC_ROOT / relative
        if not path.is_file():
            errors.append(f"missing_file:{relative}")
            continue
        if full:
            observed = _sha256(path)
            checked_hashes += 1
            if observed != expected:
                errors.append(f"sha256_mismatch:{relative}:{observed}!={expected}")

    try:
        counts = collect_counts()
    except Exception as exc:
        counts = {}
        errors.append(f"count_validation_failed:{type(exc).__name__}:{exc}")
    for name, expected in EXPECTED_COUNTS.items():
        observed = counts.get(name)
        if observed != expected:
            errors.append(f"count_mismatch:{name}:{observed}!={expected}")

    zip_archives_checked = 0
    if full:
        for archive in sorted(
            (PUBLIC_ROOT / "raw" / "huggingface" / "spiqa").glob(
                "test-*/SPIQA_test*_Images_224px.zip"
            )
        ):
            with ZipFile(archive) as handle:
                broken = handle.testzip()
            zip_archives_checked += 1
            if broken:
                errors.append(f"broken_zip_member:{archive.relative_to(PUBLIC_ROOT)}:{broken}")

    return {
        "ok": not errors,
        "full": full,
        "manifest": str(MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        "sources": len(manifest["sources"]),
        "revisions": revisions,
        "counts": counts,
        "hashes_checked": checked_hashes,
        "zip_archives_checked": zip_archives_checked,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Verify SHA-256 and ZIP payloads.")
    args = parser.parse_args()
    report = validate(full=args.full)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
