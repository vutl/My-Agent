from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import json
from pathlib import Path
from typing import Any
import uuid

from app.db.sqlite import connect


_UNSET = object()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


def _content_digest(*values: Any) -> str:
    raw = "\x1f".join(str(value or "") for value in values)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class AgentRunRecord:
    id: str
    conversation_id: str
    user_message_id: str | None
    mode: str
    status: str
    started_at: str
    ended_at: str | None
    plan: list[str]
    final_answer: str | None
    error_message: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ToolCallRecord:
    id: str
    run_id: str
    tool_name: str
    status: str
    started_at: str
    ended_at: str | None
    input: dict[str, Any] | None
    output: Any
    requires_confirmation: bool
    approved: bool | None
    error_message: str | None


@dataclass(frozen=True)
class AgentRunStore:
    db_path: Path

    def create_run(
        self,
        *,
        conversation_id: str,
        user_message_id: str | None,
        mode: str,
        metadata: dict[str, Any] | None = None,
    ) -> AgentRunRecord:
        run = AgentRunRecord(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            user_message_id=user_message_id,
            mode=mode,
            status="running",
            started_at=utc_now(),
            ended_at=None,
            plan=[],
            final_answer=None,
            error_message=None,
            metadata=metadata or {},
        )
        with connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO agent_runs (
                    id, conversation_id, user_message_id, mode, status,
                    started_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.id,
                    run.conversation_id,
                    run.user_message_id,
                    run.mode,
                    run.status,
                    run.started_at,
                    _json(run.metadata),
                ),
            )
        return run

    def update_plan(self, run_id: str, plan: list[str]) -> None:
        with connect(self.db_path) as connection:
            connection.execute(
                "UPDATE agent_runs SET plan_json = ? WHERE id = ?",
                (_json(plan), run_id),
            )

    def complete_run(self, run_id: str, final_answer: str) -> None:
        with connect(self.db_path) as connection:
            connection.execute(
                """
                UPDATE agent_runs
                SET status = 'completed', ended_at = ?, final_answer = ?, error_message = NULL
                WHERE id = ? AND status = 'running'
                """,
                (utc_now(), final_answer, run_id),
            )

    def fail_run(self, run_id: str, error_message: str) -> None:
        with connect(self.db_path) as connection:
            connection.execute(
                """
                UPDATE agent_runs
                SET status = 'failed', ended_at = ?, error_message = ?
                WHERE id = ? AND status = 'running'
                """,
                (utc_now(), error_message, run_id),
            )

    def cancel_run(self, run_id: str, reason: str = "stream_cancelled") -> None:
        with connect(self.db_path) as connection:
            connection.execute(
                """
                UPDATE agent_runs
                SET status = 'cancelled', ended_at = ?, error_message = ?
                WHERE id = ? AND status = 'running'
                """,
                (utc_now(), reason, run_id),
            )

    def fail_stale_running_runs(
        self,
        *,
        conversation_id: str | None = None,
        older_than_seconds: int = 900,
    ) -> int:
        """Recover abandoned runs left behind by a crashed/disconnected stream."""
        now = datetime.now(UTC)
        cutoff = (now - timedelta(seconds=max(1, older_than_seconds))).isoformat()
        query = """
            UPDATE agent_runs
            SET status = 'failed', ended_at = ?, error_message = 'stale_run_recovered'
            WHERE status = 'running' AND started_at < ?
        """
        params: list[Any] = [now.isoformat(), cutoff]
        if conversation_id:
            query += " AND conversation_id = ?"
            params.append(conversation_id)
        with connect(self.db_path) as connection:
            cursor = connection.execute(query, params)
            return max(0, cursor.rowcount)

    def record_tool_call(
        self,
        *,
        run_id: str,
        tool_name: str,
        input_payload: dict[str, Any] | None,
        output_payload: Any,
        status: str = "completed",
        requires_confirmation: bool = False,
        approved: bool | None = None,
        error_message: str | None = None,
        started_at: str | None = None,
    ) -> ToolCallRecord:
        now = utc_now()
        tool_call = ToolCallRecord(
            id=str(uuid.uuid4()),
            run_id=run_id,
            tool_name=tool_name,
            status=status,
            started_at=started_at or now,
            ended_at=now,
            input=input_payload,
            output=output_payload,
            requires_confirmation=requires_confirmation,
            approved=approved,
            error_message=error_message,
        )
        with connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO tool_calls (
                    id, run_id, tool_name, input_json, output_json, status,
                    started_at, ended_at, requires_confirmation, approved, error_message
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tool_call.id,
                    tool_call.run_id,
                    tool_call.tool_name,
                    _json(tool_call.input) if tool_call.input is not None else None,
                    _json(tool_call.output) if tool_call.output is not None else None,
                    tool_call.status,
                    tool_call.started_at,
                    tool_call.ended_at,
                    int(tool_call.requires_confirmation),
                    None if tool_call.approved is None else int(tool_call.approved),
                    tool_call.error_message,
                ),
            )
        return tool_call

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with connect(self.db_path) as connection:
            run = connection.execute(
                """
                SELECT id, conversation_id, user_message_id, mode, status, started_at,
                       ended_at, plan_json, final_answer, error_message, metadata_json
                FROM agent_runs
                WHERE id = ?
                """,
                (run_id,),
            ).fetchone()
            if run is None:
                return None

            tool_rows = connection.execute(
                """
                SELECT id, run_id, tool_name, input_json, output_json, status,
                       started_at, ended_at, requires_confirmation, approved, error_message
                FROM tool_calls
                WHERE run_id = ?
                ORDER BY started_at ASC
                """,
                (run_id,),
            ).fetchall()

        run_data = dict(run)
        run_data["plan"] = json.loads(run_data.pop("plan_json") or "[]")
        run_data["metadata"] = json.loads(run_data.pop("metadata_json") or "{}")
        run_data["tool_calls"] = [_decode_tool_call(dict(row)) for row in tool_rows]
        return run_data

    def upsert_debug_trace(
        self,
        *,
        run_id: str,
        payload: dict[str, Any],
        size_bytes: int,
        redaction_count: int,
        truncated: bool,
        retention_hours: int,
        max_runs: int,
        max_bytes: int,
    ) -> None:
        """Persist one already-redacted, hard-bounded snapshot for a run."""

        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        actual_bytes = len(raw.encode("utf-8"))
        if actual_bytes != size_bytes or actual_bytes > max_bytes:
            raise ValueError("debug trace payload violates UTF-8 byte cap")
        now = datetime.now(UTC)
        expires_at = (now + timedelta(hours=max(1, retention_hours))).isoformat()
        with connect(self.db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO agent_run_debug_traces (
                    run_id, schema_version, payload_json, size_bytes,
                    redaction_count, truncated, created_at, updated_at, expires_at
                )
                VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    size_bytes = excluded.size_bytes,
                    redaction_count = excluded.redaction_count,
                    truncated = excluded.truncated,
                    updated_at = excluded.updated_at,
                    expires_at = excluded.expires_at
                """,
                (
                    run_id,
                    raw,
                    actual_bytes,
                    max(0, redaction_count),
                    int(truncated),
                    now.isoformat(),
                    now.isoformat(),
                    expires_at,
                ),
            )
            connection.execute(
                "DELETE FROM agent_run_debug_traces WHERE expires_at <= ?",
                (now.isoformat(),),
            )
            keep = max(1, max_runs)
            connection.execute(
                """
                DELETE FROM agent_run_debug_traces
                WHERE run_id IN (
                    SELECT run_id
                    FROM agent_run_debug_traces
                    ORDER BY created_at DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (keep,),
            )

    def get_debug_trace(self, run_id: str) -> dict[str, Any] | None:
        now = utc_now()
        with connect(self.db_path) as connection:
            connection.execute(
                "DELETE FROM agent_run_debug_traces WHERE expires_at <= ?",
                (now,),
            )
            row = connection.execute(
                """
                SELECT run_id, schema_version, payload_json, size_bytes,
                       redaction_count, truncated, created_at, updated_at, expires_at
                FROM agent_run_debug_traces
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["payload"] = json.loads(result.pop("payload_json") or "{}")
        result["truncated"] = bool(result["truncated"])
        return result

    def purge_debug_traces(self, *, max_runs: int) -> int:
        now = utc_now()
        with connect(self.db_path) as connection:
            before = connection.execute(
                "SELECT COUNT(*) AS count FROM agent_run_debug_traces"
            ).fetchone()["count"]
            connection.execute(
                "DELETE FROM agent_run_debug_traces WHERE expires_at <= ?",
                (now,),
            )
            connection.execute(
                """
                DELETE FROM agent_run_debug_traces
                WHERE run_id IN (
                    SELECT run_id FROM agent_run_debug_traces
                    ORDER BY created_at DESC LIMIT -1 OFFSET ?
                )
                """,
                (max(1, max_runs),),
            )
            after = connection.execute(
                "SELECT COUNT(*) AS count FROM agent_run_debug_traces"
            ).fetchone()["count"]
        return max(0, int(before) - int(after))

    def latest_retrieved_document_ids(
        self,
        conversation_id: str,
        *,
        limit: int = 8,
        expected_collection_id: str | None | object = _UNSET,
    ) -> list[str]:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT tool_calls.input_json, tool_calls.output_json
                FROM tool_calls
                JOIN agent_runs ON agent_runs.id = tool_calls.run_id
                WHERE agent_runs.conversation_id = ?
                  AND agent_runs.status = 'completed'
                  AND tool_calls.tool_name = 'search_local_docs'
                  AND tool_calls.status = 'completed'
                  AND tool_calls.output_json IS NOT NULL
                ORDER BY tool_calls.started_at DESC
                LIMIT 20
                """,
                (conversation_id,),
            ).fetchall()

        for row in rows:
            input_payload = _parse_retrieval_payload(row["input_json"])
            if (
                expected_collection_id is not _UNSET
                and input_payload.get("collection_id") != expected_collection_id
            ):
                continue
            payload = _parse_retrieval_payload(row["output_json"])
            if not _is_valid_retrieval_payload(payload):
                continue
            seen: set[str] = set()
            document_ids: list[str] = []
            for document in payload.get("documents") or []:
                document_id = document.get("document_id")
                if document_id and document_id not in seen:
                    seen.add(document_id)
                    document_ids.append(document_id)
                    if len(document_ids) >= limit:
                        break
            if document_ids:
                return document_ids
        return []

    def latest_retrieval_output(
        self,
        conversation_id: str,
        *,
        expected_collection_id: str | None | object = _UNSET,
        expected_retrieval_mode: str | None = None,
        expected_index_fingerprint: str | None = None,
    ) -> dict[str, Any] | None:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT tool_calls.input_json, tool_calls.output_json
                FROM tool_calls
                JOIN agent_runs ON agent_runs.id = tool_calls.run_id
                WHERE agent_runs.conversation_id = ?
                  AND agent_runs.status = 'completed'
                  AND tool_calls.tool_name = 'search_local_docs'
                  AND tool_calls.status = 'completed'
                  AND tool_calls.output_json IS NOT NULL
                ORDER BY tool_calls.ended_at DESC, tool_calls.started_at DESC
                LIMIT 20
                """,
                (conversation_id,),
            ).fetchall()

        for row in rows:
            input_payload = _parse_retrieval_payload(row["input_json"])
            if (
                expected_collection_id is not _UNSET
                and input_payload.get("collection_id") != expected_collection_id
            ):
                continue
            if expected_retrieval_mode is not None and str(
                input_payload.get("retrieval_mode") or "auto"
            ).casefold() != expected_retrieval_mode.casefold():
                continue
            payload = _parse_retrieval_payload(row["output_json"])
            if (
                expected_index_fingerprint is not None
                and payload.get("index_fingerprint") != expected_index_fingerprint
            ):
                continue
            if _is_valid_retrieval_payload(payload):
                return payload
        return None

    def index_fingerprint(
        self,
        *,
        collection_id: str | None = None,
        document_ids: list[str] | None = None,
        configuration: dict[str, Any] | None = None,
    ) -> str:
        query = """
            SELECT documents.id, documents.content_hash, documents.indexed_at
            FROM documents
        """
        params: list[Any] = []
        conditions: list[str] = []
        if collection_id is not None:
            query += """
                JOIN collection_documents
                  ON collection_documents.document_id = documents.id
            """
            conditions.append("collection_documents.collection_id = ?")
            params.append(collection_id)
        if document_ids is not None:
            unique_document_ids = list(dict.fromkeys(str(value) for value in document_ids))
            if unique_document_ids:
                placeholders = ", ".join("?" for _ in unique_document_ids)
                conditions.append(f"documents.id IN ({placeholders})")
                params.extend(unique_document_ids)
            else:
                conditions.append("0 = 1")
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY documents.id"

        with connect(self.db_path) as connection:
            rows = connection.execute(query, params).fetchall()

        entries = [
            {
                "id": row["id"],
                "content_hash": row["content_hash"],
                "indexed_at": row["indexed_at"],
            }
            for row in rows
        ]
        scoped_document_ids = [str(row["id"]) for row in rows]
        derived_entries: list[dict[str, Any]] = []
        provenance_entries: list[dict[str, Any]] = []
        if scoped_document_ids:
            placeholders = ", ".join("?" for _ in scoped_document_ids)
            with connect(self.db_path) as connection:
                figures = connection.execute(
                    f"""
                    SELECT id, document_id, caption, visual_summary, metadata_json
                    FROM document_figures
                    WHERE document_id IN ({placeholders})
                    ORDER BY document_id, figure_index, id
                    """,
                    scoped_document_ids,
                ).fetchall()
                tables = connection.execute(
                    f"""
                    SELECT id, document_id, caption, markdown, metadata_json
                    FROM document_tables
                    WHERE document_id IN ({placeholders})
                    ORDER BY document_id, table_index, id
                    """,
                    scoped_document_ids,
                ).fetchall()
                provenance = connection.execute(
                    f"""
                    SELECT
                        lightrag_chunk_id,
                        parent_chunk_id,
                        document_id,
                        content_hash,
                        parent_content_hash,
                        overlap_chars,
                        canonical_method,
                        mapping_method,
                        mapping_score
                    FROM lightrag_chunk_parent_provenance
                    WHERE document_id IN ({placeholders})
                    ORDER BY document_id, lightrag_chunk_id, parent_chunk_id
                    """,
                    scoped_document_ids,
                ).fetchall()
            derived_entries = [
                {
                    "kind": "figure",
                    "id": row["id"],
                    "document_id": row["document_id"],
                    "digest": _content_digest(
                        row["caption"], row["visual_summary"], row["metadata_json"]
                    ),
                }
                for row in figures
            ] + [
                {
                    "kind": "table",
                    "id": row["id"],
                    "document_id": row["document_id"],
                    "digest": _content_digest(
                        row["caption"], row["markdown"], row["metadata_json"]
                    ),
                }
                for row in tables
            ]
            provenance_entries = [
                {
                    "lightrag_chunk_id": row["lightrag_chunk_id"],
                    "parent_chunk_id": row["parent_chunk_id"],
                    "document_id": row["document_id"],
                    "content_hash": row["content_hash"],
                    "parent_content_hash": row["parent_content_hash"],
                    "overlap_chars": row["overlap_chars"],
                    "canonical_method": row["canonical_method"],
                    "mapping_method": row["mapping_method"],
                    "mapping_score": row["mapping_score"],
                }
                for row in provenance
            ]
        raw = json.dumps(
            {
                "documents": entries,
                "derived": derived_entries,
                "lightrag_parent_provenance": provenance_entries,
                "configuration": configuration or {},
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get_retrieval_cache(self, cache_key: str) -> dict[str, Any] | None:
        with connect(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT output_json
                FROM retrieval_cache
                WHERE cache_key = ?
                  AND (expires_at IS NULL OR expires_at > ?)
                """,
                (cache_key, utc_now()),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """
                UPDATE retrieval_cache
                SET hit_count = hit_count + 1, last_hit_at = ?
                WHERE cache_key = ?
                """,
                (utc_now(), cache_key),
            )

        payload = json.loads(row["output_json"] or "{}")
        return payload if isinstance(payload, dict) else None

    def store_retrieval_cache(
        self,
        *,
        cache_key: str,
        normalized_query: str,
        collection_id: str | None,
        focus_document_ids: list[str],
        retrieval_mode: str,
        index_fingerprint: str,
        output_payload: dict[str, Any],
    ) -> None:
        with connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO retrieval_cache (
                    id, cache_key, normalized_query, collection_id,
                    focus_document_ids_json, retrieval_mode, index_fingerprint,
                    output_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    output_json = excluded.output_json,
                    index_fingerprint = excluded.index_fingerprint,
                    created_at = excluded.created_at
                """,
                (
                    str(uuid.uuid4()),
                    cache_key,
                    normalized_query,
                    collection_id,
                    _json(focus_document_ids),
                    retrieval_mode,
                    index_fingerprint,
                    _json(output_payload),
                    utc_now(),
                ),
            )


def _decode_tool_call(row: dict[str, Any]) -> dict[str, Any]:
    row["input"] = json.loads(row.pop("input_json") or "null")
    row["output"] = json.loads(row.pop("output_json") or "null")
    row["requires_confirmation"] = bool(row["requires_confirmation"])
    row["approved"] = None if row["approved"] is None else bool(row["approved"])
    return row


def _parse_retrieval_payload(raw: str | None) -> dict[str, Any]:
    try:
        payload = json.loads(raw or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _is_valid_retrieval_payload(payload: dict[str, Any]) -> bool:
    if not payload.get("documents"):
        return False
    validation = payload.get("evidence_validation")
    # Backward-compatible with historical completed calls written before the
    # validation field existed; all new calls include it explicitly.
    return not isinstance(validation, dict) or validation.get("valid") is True
