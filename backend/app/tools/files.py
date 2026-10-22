"""File tools: list, read, search, write, edit, move, trash.

Paths are resolved (symlinks followed) and checked against ``policy``. Writes
inside the workspace run directly; anywhere else in the home folder they wait
for approval. Deletion moves to the macOS Trash (reversible) and always asks.
File text is untrusted data for the model.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import fnmatch
import os
from pathlib import Path
import shutil
from typing import Any

from app.tools import policy
from app.tools.registry import Tool, ToolContext, ToolResult, ToolSource

_MAX_READ_CHARS = 60_000
_MAX_LIST = 200
_MAX_SEARCH_HITS = 60
_SKIP_DIRS = {".git", "node_modules", ".venv", "__pycache__", ".Trash", "Library"}
_DOCUMENT_SUFFIXES = {".pdf", ".docx"}


def _workspace(context: ToolContext) -> Path:
    context.workspace.mkdir(parents=True, exist_ok=True)
    return context.workspace.resolve()


def _fail(message: str) -> ToolResult:
    return ToolResult(content=message, summary=message[:120], ok=False)


def _display(path: Path) -> str:
    try:
        return "~/" + str(path.relative_to(policy.home()))
    except ValueError:
        return str(path)


async def list_dir(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    try:
        path = policy.resolve(arguments.get("path") or ".", _workspace(context))
        policy.check_readable(path)
    except policy.PathPolicyError as exc:
        return _fail(str(exc))
    if not path.is_dir():
        return _fail(f"{_display(path)} is not a folder")
    entries = []
    for child in sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))[:_MAX_LIST]:
        if policy.is_secret(child):
            continue
        try:
            stat = child.stat()
        except OSError:
            continue
        kind = "dir" if child.is_dir() else f"{stat.st_size:,} B"
        entries.append(f"{child.name}{'/' if child.is_dir() else ''}\t{kind}\t{datetime.fromtimestamp(stat.st_mtime):%Y-%m-%d %H:%M}")
    return ToolResult(content="\n".join(entries) or "(empty)", summary=f"Xem thư mục {_display(path)} — {len(entries)} mục",
                      sources=[ToolSource(title=_display(path), path=str(path))])


def _read_text(path: Path) -> str:
    if path.suffix.lower() in _DOCUMENT_SUFFIXES:
        from app.rag.parsers import parse_document

        return parse_document(path, enable_pdf_ocr=False).text
    raw = path.read_bytes()[: _MAX_READ_CHARS * 4]
    if b"\x00" in raw[:4096]:
        raise ValueError("binary file")
    return raw.decode("utf-8", errors="replace")


async def read_file(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    try:
        path = policy.resolve(arguments.get("path") or "", _workspace(context))
        policy.check_readable(path)
    except policy.PathPolicyError as exc:
        return _fail(str(exc))
    if not path.is_file():
        return _fail(f"{_display(path)} does not exist")
    try:
        text = await asyncio.to_thread(_read_text, path)
    except Exception as exc:  # noqa: BLE001
        return _fail(f"Cannot read {_display(path)} ({type(exc).__name__}: {exc})"[:200])
    clipped = len(text) > _MAX_READ_CHARS
    return ToolResult(
        content=text[:_MAX_READ_CHARS] + ("\n…(truncated)" if clipped else ""),
        summary=f"Đọc {_display(path)} ({len(text):,} ký tự)", untrusted=True,
        sources=[ToolSource(title=path.name, path=str(path), snippet=text[:200])],
    )


def _search(root: Path, pattern: str, glob: str) -> list[str]:
    hits: list[str] = []
    needle = pattern.lower()
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in _SKIP_DIRS and not name.startswith(".")
                       and not policy.is_secret(Path(directory) / name)]
        for filename in filenames:
            path = Path(directory) / filename
            if glob and not fnmatch.fnmatch(filename, glob):
                continue
            if policy.is_secret(path):
                continue
            if needle in filename.lower():
                hits.append(f"{_display(path)} (name)")
            elif path.suffix.lower() in {".txt", ".md", ".csv", ".json", ".py", ".ts", ".tsx", ".js", ".html", ".yml", ".yaml"}:
                try:
                    with path.open("r", encoding="utf-8", errors="ignore") as handle:
                        for number, line in enumerate(handle, 1):
                            if needle in line.lower():
                                hits.append(f"{_display(path)}:{number}: {line.strip()[:160]}")
                                break
                except OSError:
                    continue
            if len(hits) >= _MAX_SEARCH_HITS:
                return hits
    return hits


async def search_files(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    pattern = str(arguments.get("pattern") or "").strip()
    if not pattern:
        return _fail("empty pattern")
    try:
        root = policy.resolve(arguments.get("path") or "~", _workspace(context))
        policy.check_readable(root)
    except policy.PathPolicyError as exc:
        return _fail(str(exc))
    hits = await asyncio.to_thread(_search, root, pattern, str(arguments.get("glob") or ""))
    return ToolResult(content="\n".join(hits) or "No matches.", untrusted=True,
                      summary=f"Tìm “{pattern}” trong {_display(root)} — {len(hits)} kết quả")


def _target(arguments: dict[str, Any], context: ToolContext, key: str = "path") -> Path:
    path = policy.resolve(arguments.get(key) or "", context.workspace.resolve())
    policy.check_writable(path)
    return path


def _outside_workspace(*keys: str):
    def check(arguments: dict[str, Any], context: ToolContext) -> bool:
        try:
            return any(
                policy.write_needs_approval(policy.resolve(arguments.get(key) or "", context.workspace.resolve()),
                                            context.workspace.resolve())
                for key in keys
            )
        except policy.PathPolicyError:
            return False  # the handler reports the refusal

    return check


async def write_file(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    _workspace(context)
    try:
        path = _target(arguments, context)
    except policy.PathPolicyError as exc:
        return _fail(str(exc))
    content = str(arguments.get("content") or "")
    if path.exists() and not arguments.get("overwrite"):
        return _fail(f"{_display(path)} already exists; pass overwrite=true to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return ToolResult(content=f"Wrote {len(content):,} characters to {path}.", summary=f"Ghi {_display(path)}",
                      sources=[ToolSource(title=path.name, path=str(path))])


async def edit_file(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    try:
        path = _target(arguments, context)
    except policy.PathPolicyError as exc:
        return _fail(str(exc))
    if not path.is_file():
        return _fail(f"{_display(path)} does not exist")
    old, new = str(arguments.get("old_text") or ""), str(arguments.get("new_text") or "")
    text = path.read_text(encoding="utf-8", errors="replace")
    count = text.count(old) if old else 0
    if count != 1:
        return _fail(f"old_text must occur exactly once in {_display(path)} (found {count})")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return ToolResult(content=f"Edited {path}.", summary=f"Sửa {_display(path)}",
                      sources=[ToolSource(title=path.name, path=str(path))])


async def move_path(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    try:
        source = _target(arguments, context, "source")
        destination = _target(arguments, context, "destination")
    except policy.PathPolicyError as exc:
        return _fail(str(exc))
    if not source.exists():
        return _fail(f"{_display(source)} does not exist")
    if destination.exists() and destination.is_dir():
        destination = destination / source.name
    if destination.exists():
        return _fail(f"{_display(destination)} already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    return ToolResult(content=f"Moved {source} to {destination}.", summary=f"Chuyển {_display(source)} → {_display(destination)}")


async def trash_path(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    try:
        path = _target(arguments, context)
    except policy.PathPolicyError as exc:
        return _fail(str(exc))
    if not path.exists():
        return _fail(f"{_display(path)} does not exist")
    trash = policy.home() / ".Trash"
    trash.mkdir(exist_ok=True)
    destination = trash / path.name
    if destination.exists():
        destination = trash / f"{path.stem} {datetime.now():%H.%M.%S}{path.suffix}"
    shutil.move(str(path), str(destination))
    return ToolResult(content=f"Moved {path} to the Trash.", summary=f"Bỏ {_display(path)} vào Thùng rác")


_PATH = {"type": "string", "description": "Absolute, ~/… or workspace-relative path."}

LIST_DIR = Tool(name="list_dir", description="List a folder (names, sizes, modified times).",
                parameters={"type": "object", "properties": {"path": _PATH}}, handler=list_dir, label="Xem thư mục")
READ_FILE = Tool(name="read_file", description="Read a text, Markdown, CSV, code, PDF or DOCX file.",
                 parameters={"type": "object", "properties": {"path": _PATH}, "required": ["path"]},
                 handler=read_file, label="Đọc file")
SEARCH_FILES = Tool(
    name="search_files", description="Find files by name or by a text they contain under a folder (default: home).",
    parameters={"type": "object", "properties": {"pattern": {"type": "string"}, "path": _PATH,
                                                 "glob": {"type": "string", "description": "e.g. *.csv"}},
                "required": ["pattern"]},
    handler=search_files, label="Tìm file",
)
WRITE_FILE = Tool(
    name="write_file", description="Create a text file (inside the workspace without asking; elsewhere the user approves).",
    parameters={"type": "object", "properties": {"path": _PATH, "content": {"type": "string"},
                                                 "overwrite": {"type": "boolean"}}, "required": ["path", "content"]},
    handler=write_file, label="Ghi file", needs_approval=_outside_workspace("path"),
)
EDIT_FILE = Tool(
    name="edit_file", description="Replace one exact occurrence of old_text with new_text in a text file.",
    parameters={"type": "object", "properties": {"path": _PATH, "old_text": {"type": "string"},
                                                 "new_text": {"type": "string"}},
                "required": ["path", "old_text", "new_text"]},
    handler=edit_file, label="Sửa file", needs_approval=_outside_workspace("path"),
)
MOVE_PATH = Tool(
    name="move_path", description="Move or rename a file or folder.",
    parameters={"type": "object", "properties": {"source": _PATH, "destination": _PATH},
                "required": ["source", "destination"]},
    handler=move_path, label="Di chuyển", needs_approval=_outside_workspace("source", "destination"),
)
TRASH_PATH = Tool(
    name="trash_path", description="Move a file or folder to the macOS Trash (always asks the user).",
    parameters={"type": "object", "properties": {"path": _PATH}, "required": ["path"]},
    handler=trash_path, label="Bỏ vào Thùng rác", permission="ask", side_effects=True,
)

FILE_TOOLS = (LIST_DIR, READ_FILE, SEARCH_FILES, WRITE_FILE, EDIT_FILE, MOVE_PATH, TRASH_PATH)
