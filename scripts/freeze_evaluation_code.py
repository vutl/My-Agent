#!/usr/bin/env python3
"""Snapshot dirty evaluation code without copying secrets or Python environments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from evaluation_checkpoint import write_checkpoint


def freeze_code(project: Path, target: Path) -> dict:
    project, target = project.resolve(), target.resolve()
    if target.exists():
        raise FileExistsError("Refusing to overwrite a frozen evaluation snapshot")
    (target / "backend").mkdir(parents=True)
    shutil.copytree(project / "backend" / "app", target / "backend" / "app",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".env*"))
    (target / "scripts").mkdir()
    for script in sorted((project / "scripts").glob("*.py")):
        shutil.copy2(script, target / "scripts" / script.name)
    # Share only the public dataset tree, not production SQLite/Lance/artifacts.
    # A symlink is not a filesystem read-only boundary: the evaluator still
    # enforces explicit isolated output/runtime roots. No .env is copied.
    (target / "data" / "retrieval_eval").mkdir(parents=True)
    (target / "data" / "retrieval_eval" / "public").symlink_to(
        project / "data" / "retrieval_eval" / "public", target_is_directory=True)
    h = hashlib.sha256()
    for path in sorted([*(target / "backend" / "app").rglob("*.py"),
                        *(target / "scripts").glob("*.py")]):
        h.update(str(path.relative_to(target)).encode())
        h.update(path.read_bytes())
    report = {"source_project": str(project), "frozen_root": str(target),
              "code_sha256": h.hexdigest(), "secrets_copied": False,
              "public_datasets_are_shared_inputs": True,
              "production_data_linked": False,
              "isolation_boundary": "explicit isolated evaluator runtime/output roots",
              "required_isolated_runtime": True}
    write_checkpoint(target / "frozen-code.json", report)
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(freeze_code(Path(__file__).resolve().parents[1], args.output_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
