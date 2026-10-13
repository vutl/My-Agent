"""Durable SSE evaluation state bound to an isolated runtime and real history.

Interrupted in-flight turns are never silently replayed: that could duplicate
model calls or change the conversation which later cases are evaluating.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Callable
import uuid

from evaluation_checkpoint import write_checkpoint
from evaluator_fingerprint import fingerprint_files


class AgentCheckpointError(RuntimeError):
    pass


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def runtime_fingerprint(root: Path) -> str:
    """Hash canonical projections, excluding changing chat/job state."""
    db = root.resolve() / "sqlite" / "app.db"
    if not db.is_file():
        raise AgentCheckpointError("Isolated runtime database is missing")
    h = hashlib.sha256()
    with sqlite3.connect(db.as_uri() + "?mode=ro", uri=True) as connection:
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        for table in ("documents", "document_revisions", "rag_chunks",
                      "document_tables", "document_figures", "document_cards",
                      "paper_evidence_cards", "paper_evidence_card_revisions",
                      "paper_evidence_facet_revisions", "paper_evidence_item_revisions"):
            if table not in tables:
                continue
            h.update(table.encode())
            for row in connection.execute(f'SELECT * FROM "{table}" ORDER BY rowid'):
                h.update(json.dumps(list(row), default=str, ensure_ascii=False).encode())
    for path in sorted((root / "lancedb").rglob("*")):
        if path.is_file() and path.suffix != ".lock":
            h.update(str(path.relative_to(root)).encode())
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    h.update(block)
    for path in sorted((root / "artifacts").rglob("*")):
        if path.is_file():
            h.update(str(path.relative_to(root)).encode())
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    h.update(block)
    return h.hexdigest()


def evaluator_fingerprint(project: Path) -> str:
    files = list((project / "backend" / "app").rglob("*.py"))
    files.extend(project / "scripts" / name for name in (
        "evaluate_agent_conversations.py", "agent_evaluation_checkpoint.py",
        "evaluation_checkpoint.py", "evaluator_fingerprint.py"))
    return fingerprint_files(project, files)


def verify_runtime_binding(root: Path, project: Path,
                           get_json: Callable[[str], Any]) -> None:
    """Prove the HTTP server uses this clone before sending any model request.

    A temporary empty conversation is written ONLY to the declared clone and
    read back through HTTP. Production roots and symlink aliases are rejected.
    """
    root = root.resolve()
    db = root / "sqlite" / "app.db"
    production_db = (project / "data" / "sqlite" / "app.db").resolve()
    if not db.is_file() or db.resolve() == production_db:
        raise AgentCheckpointError("Checkpoint evaluation requires an existing isolated clone")
    if production_db.exists() and db.samefile(production_db):
        raise AgentCheckpointError("Runtime database aliases production")
    marker = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(db) as connection:
        connection.execute(
            "INSERT INTO conversations(id,title,created_at,updated_at,metadata_json) "
            "VALUES(?,?,?,?,?)", (marker, "[evaluation runtime binding]", now, now, "{}"))
    try:
        conversations = get_json("/chat/conversations")
        if not isinstance(conversations, list) or not any(
            item.get("id") == marker for item in conversations if isinstance(item, dict)
        ):
            raise AgentCheckpointError("HTTP backend is not serving the declared isolated clone")
    finally:
        with sqlite3.connect(db) as connection:
            connection.execute("DELETE FROM conversations WHERE id=?", (marker,))


class AgentCheckpoint:
    def __init__(self, path: Path, *, contract: dict[str, Any],
                 cases: list[dict[str, Any]], resume: bool,
                 get_json: Callable[[str], Any]):
        self.path, self.contract, self.cases, self.get_json = path, contract, cases, get_json
        self.rows: list[dict[str, Any]] = []
        self.histories: dict[str, dict[str, Any]] = {}
        self.inflight: dict[str, Any] | None = None
        self.resumed_cases = 0
        if path.exists():
            if not resume:
                raise AgentCheckpointError("Checkpoint exists; use --resume or a new output path")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("contract") != contract:
                raise AgentCheckpointError("Checkpoint code/index/options/case contract changed")
            self.rows = payload.get("results", [])
            self.histories = payload.get("histories", {})
            self.inflight = payload.get("inflight")
            ids = [row.get("case_id") for row in self.rows]
            expected = [case["case_id"] for case in cases]
            if ids != expected[:len(ids)] or len(ids) != len(set(ids)):
                raise AgentCheckpointError("Checkpoint rows are not the ordered suite prefix")
            if self.inflight:
                raise AgentCheckpointError(
                    "An interrupted turn remains in flight; no duplicate model call will be made. "
                    "Keep this audit file and use a new output for an explicit group replay.")
            if any(not row.get("run_completed") for row in self.rows):
                raise AgentCheckpointError("A failed provider/run checkpoint cannot resume as a clean run")
            groups = {row["conversation_group"] for row in self.rows}
            if set(self.histories) != groups or any(
                not row.get("conversation_id") or row["conversation_id"] !=
                self.histories[row["conversation_group"]].get("conversation_id")
                for row in self.rows
            ):
                raise AgentCheckpointError("Checkpoint conversation history binding is incomplete")
            self.verify_history()
            self.resumed_cases = len(self.rows)
        elif resume:
            raise AgentCheckpointError("Cannot resume a missing checkpoint")

    def verify_history(self) -> None:
        for group, saved in self.histories.items():
            messages = self.get_json(f"/chat/conversations/{saved['conversation_id']}/messages")
            if digest(messages) != saved["digest"]:
                raise AgentCheckpointError(f"Persisted conversation changed or disappeared: {group}")
        for row in self.rows:
            if row.get("run_completed"):
                run = self.get_json(f"/agent/runs/{row['run_id']}")
                if (run.get("status") != "completed"
                        or run.get("conversation_id") != row["conversation_id"]
                        or digest(run.get("final_answer") or "") != row["answer_sha256"]):
                    raise AgentCheckpointError("Persisted run does not match the measured SSE turn")

    def save(self, *, error: str | None = None) -> None:
        write_checkpoint(self.path, {
            "schema_version": "agent-sse-checkpoint-v1", "contract": self.contract,
            "results": self.rows, "histories": self.histories,
            "inflight": self.inflight, "resumed_cases": self.resumed_cases,
            "cases_requested": len(self.cases), "cases_completed": len(self.rows),
            "error": error, "updated_at": datetime.now(timezone.utc).isoformat(),
        })

    def begin(self, case: dict[str, Any], conversation_id: str | None) -> None:
        self.inflight = {"case_id": case["case_id"],
                         "conversation_group": case["conversation_group"],
                         "conversation_id": conversation_id, "run_id": None}
        self.save()

    def started(self, data: dict[str, Any]) -> None:
        if self.inflight is not None:
            self.inflight.update({"run_id": data.get("run_id"),
                                  "conversation_id": data.get("conversation_id")})
            self.save()

    def finish(self, row: dict[str, Any], *, conversation_id: str | None,
               answer: str, completed: bool) -> None:
        row.update(conversation_id=conversation_id, run_completed=completed,
                   answer_sha256=digest(answer))
        if completed:
            run = self.get_json(f"/agent/runs/{row['run_id']}")
            if (run.get("status") != "completed"
                    or run.get("conversation_id") != conversation_id
                    or digest(run.get("final_answer") or "") != digest(answer)):
                raise AgentCheckpointError("Completed SSE answer is not persisted identically")
        if conversation_id:
            messages = self.get_json(f"/chat/conversations/{conversation_id}/messages")
            self.histories[row["conversation_group"]] = {
                "conversation_id": conversation_id, "digest": digest(messages)}
        self.rows.append(row)
        self.inflight = None
        self.save()
