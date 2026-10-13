"""Stable source/content/revision identity helpers for local document ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def revision_fingerprint(
    *,
    raw_content_hash: str,
    parser_fingerprint: str,
    chunker_fingerprint: str,
    artifact_fingerprint: str,
) -> str:
    return stable_json_hash(
        {
            "raw_content_hash": raw_content_hash,
            "parser_fingerprint": parser_fingerprint,
            "chunker_fingerprint": chunker_fingerprint,
            "artifact_fingerprint": artifact_fingerprint,
        }
    )


@dataclass(frozen=True, slots=True)
class DocumentRevisionIdentity:
    revision_id: str
    document_id: str
    raw_content_hash: str
    extracted_content_hash: str
    parser_fingerprint: str
    chunker_fingerprint: str
    artifact_fingerprint: str

    @property
    def fingerprint(self) -> str:
        return revision_fingerprint(
            raw_content_hash=self.raw_content_hash,
            parser_fingerprint=self.parser_fingerprint,
            chunker_fingerprint=self.chunker_fingerprint,
            artifact_fingerprint=self.artifact_fingerprint,
        )
