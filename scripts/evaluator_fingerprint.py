"""Stable fingerprints for evaluation contracts and their code dependencies."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


def fingerprint_files(project_root: Path, files: Iterable[Path]) -> str:
    """Hash evaluator code plus relevant backend modules deterministically.

    Relative paths are included so two files with identical bytes cannot
    silently swap identities.  Missing files are rejected instead of creating
    a deceptively valid fingerprint.
    """

    root = project_root.resolve()
    normalized = sorted({path.resolve() for path in files})
    digest = hashlib.sha256()
    for path in normalized:
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def default_rag_dependency_files(project_root: Path) -> list[Path]:
    """Return code paths that can change the public retrieval contract."""

    root = project_root.resolve()
    backend = root / "backend" / "app"
    files: list[Path] = []
    for directory in (
        backend / "rag",
        backend / "retrieval_store",
        backend / "services",
    ):
        files.extend(directory.rglob("*.py"))
    files.extend(
        [
            backend / "db" / "sqlite.py",
            backend / "core" / "config.py",
        ]
    )
    return files

