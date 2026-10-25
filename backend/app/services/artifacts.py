"""Workspace-local storage for complete tool results; handles are relative IDs."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re

_HANDLE = re.compile(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+\.txt")


class ArtifactStore:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.expanduser().resolve()
        self.root = self.workspace / ".aya_runs" / "artifacts"

    def _path(self, handle: str) -> Path:
        if not isinstance(handle, str) or not _HANDLE.fullmatch(handle):
            raise ValueError("Invalid artifact handle; expected run_id/call_id.txt")
        root = self.root.resolve()
        path = (self.root / handle).resolve()
        if root != self.root or not path.is_relative_to(root):
            raise ValueError("Artifact handle escapes the artifact directory")
        return path

    def save(self, run_id: str, call_id: str, content: str) -> str:
        def identifier(value: str) -> str:
            return value if re.fullmatch(r"[A-Za-z0-9_-]+", value) else hashlib.sha256(value.encode()).hexdigest()

        stem = f"{identifier(run_id)}/{identifier(call_id)}"
        # Call ids repeat across task steps and retries; never overwrite an earlier result.
        for n in range(1, 10000):
            handle = f"{stem}.txt" if n == 1 else f"{stem}-{n}.txt"
            path = self._path(handle)
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with path.open("x", encoding="utf-8") as stream:
                    stream.write(content)
            except FileExistsError:
                continue
            return handle
        raise OSError(f"Could not allocate an artifact handle for {stem}")

    def read(self, handle: str, offset: int = 0, length: int = 4000) -> str:
        if type(offset) is not int or offset < 0 or type(length) is not int or not 1 <= length <= 6000:
            raise ValueError("offset must be non-negative; length must be between 1 and 6000 characters")
        # Offsets count Unicode characters, matching the model's preview.
        with self._path(handle).open(encoding="utf-8") as stream:
            remaining = offset
            while remaining:
                chunk = stream.read(min(remaining, 65536))
                if not chunk:
                    return ""
                remaining -= len(chunk)
            return stream.read(length)
