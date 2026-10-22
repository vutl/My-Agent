from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any
import sqlite3
import uuid

from app.db.sqlite import connect


MEMORY_QUEUE_VERSION = 1


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class ConversationSummary:
    id: str
    title: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class StoredMessage:
    id: str
    conversation_id: str
    role: str
    content: str
    model: str | None
    created_at: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)


class ConversationBusyError(RuntimeError):
    """A run is still streaming in the conversation."""


@dataclass(frozen=True)
class ChatHistory:
    db_path: Path

    def ensure_conversation(self, conversation_id: str | None, first_message: str) -> str:
        selected_id = conversation_id or str(uuid.uuid4())
        now = utc_now()
        title = self._title_from_message(first_message)

        with connect(self.db_path) as connection:
            existing = connection.execute(
                "SELECT id FROM conversations WHERE id = ?",
                (selected_id,),
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO conversations (
                        id, title, created_at, updated_at, metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        selected_id,
                        title,
                        now,
                        now,
                        json.dumps(
                            {"memory_queue_version": MEMORY_QUEUE_VERSION},
                            ensure_ascii=False,
                        ),
                    ),
                )
            else:
                connection.execute(
                    """
                    UPDATE conversations
                    SET updated_at = ?,
                        metadata_json = json_set(
                            CASE
                                WHEN json_valid(metadata_json)
                                     AND json_type(metadata_json) = 'object'
                                    THEN metadata_json
                                ELSE '{}'
                            END,
                            '$.memory_queue_version',
                            ?
                        )
                    WHERE id = ?
                    """,
                    (
                        now,
                        MEMORY_QUEUE_VERSION,
                        selected_id,
                    ),
                )

        return selected_id

    def save_message(
        self,
        *,
        conversation_id: str,
        role: str,
        content: str,
        model: str | None = None,
        sources: list[dict[str, Any]] | None = None,
        citations: list[dict[str, Any]] | None = None,
    ) -> StoredMessage:
        stored_sources = _json_safe_sources(sources)
        stored_citations = _json_safe_citations(citations)
        message = StoredMessage(
            id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role=role,
            content=content,
            model=model,
            created_at=utc_now(),
            sources=stored_sources,
            citations=stored_citations,
        )

        with connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO messages (
                    id, conversation_id, role, content, model, created_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.id,
                    message.conversation_id,
                    message.role,
                    message.content,
                    message.model,
                    message.created_at,
                    json.dumps(
                        {
                            **({"sources": stored_sources} if stored_sources else {}),
                            **({"citations": stored_citations} if stored_citations else {}),
                        },
                        ensure_ascii=False,
                    )
                    if stored_sources or stored_citations
                    else None,
                ),
            )
            connection.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (message.created_at, conversation_id),
            )

        return message

    def list_conversations(self, limit: int = 50) -> list[ConversationSummary]:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT id, title, created_at, updated_at
                FROM conversations
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [ConversationSummary(**dict(row)) for row in rows]

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation with its messages, runs, tool calls and the
        memory learned from it (foreign-key cascades). Reminders set from it are
        kept and unlinked. Refused while a run is still streaming in it."""
        with connect(self.db_path) as connection:
            if connection.execute("SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)).fetchone() is None:
                return False
            running = connection.execute(
                "SELECT 1 FROM agent_runs WHERE conversation_id = ? AND status = 'running' LIMIT 1",
                (conversation_id,),
            ).fetchone()
            if running:
                raise ConversationBusyError(conversation_id)
            connection.execute("DELETE FROM message_search WHERE conversation_id = ?", (conversation_id,))
            for table in ("reminders", "notifications"):
                try:
                    connection.execute(
                        f"UPDATE {table} SET conversation_id = NULL WHERE conversation_id = ?", (conversation_id,)
                    )
                except sqlite3.OperationalError:
                    pass  # table not created yet (no reminder ever set)
            connection.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        return True

    def list_messages(self, conversation_id: str) -> list[StoredMessage]:
        with connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT id, conversation_id, role, content, model, created_at, metadata_json
                FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC
                """,
                (conversation_id,),
            ).fetchall()

        messages: list[StoredMessage] = []
        for row in rows:
            item = dict(row)
            metadata = _parse_metadata(item.pop("metadata_json", None))
            messages.append(
                StoredMessage(
                    **item,
                    sources=_json_safe_sources(metadata.get("sources")),
                    citations=_json_safe_citations(metadata.get("citations")),
                )
            )
        return messages

    def _title_from_message(self, message: str) -> str:
        compact = " ".join(message.split())
        if not compact:
            return "New chat"
        return compact[:60]


def _parse_metadata(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _json_safe_citations(citations: object) -> list[dict[str, Any]]:
    if not isinstance(citations, (list, tuple)):
        return []
    safe = [
        json.loads(json.dumps(item, ensure_ascii=False, default=str))
        for item in citations
        if isinstance(item, dict) and item.get("id")
    ]
    return safe


def _json_safe_sources(sources: object) -> list[dict[str, Any]]:
    if not isinstance(sources, (list, tuple)):
        return []

    safe: list[dict[str, Any]] = []
    # History needs stable citations and visual attachments, not the entire
    # retrieval trace/context. Bound this payload so long-running threads do
    # not duplicate tens of KB of chunk metadata on every assistant turn.
    allowed_fields = {
        "chunk_id",
        "document_id",
        "source_path",
        "filename",
        "content",
        "citation_label",
        "score",
        "chunk_index",
        "page_number",
        "heading_path",
        "retrieval_channels",
        "source_id",
        "chunk_type",
        "artifact_type",
        "caption",
        "image_path",
        "image_url",
        "table_id",
        "table_index",
        "figure_id",
        "figure_index",
        "figure_label",
        "figure_number",
        "figure_type",
        "quality_status",
        "asset_kind",
        "is_content",
        "is_complete",
        "logical_group_id",
    }
    for source in sources[:16]:
        if not isinstance(source, dict):
            continue
        # Retrieval payloads should be JSON-native. ``default=str`` keeps a
        # provider-specific Path/enum from breaking message persistence.
        projected = {key: value for key, value in source.items() if key in allowed_fields}
        content = projected.get("content")
        if isinstance(content, str) and len(content) > 1600:
            projected["content"] = f"{content[:1599]}…"
        normalized = json.loads(json.dumps(projected, ensure_ascii=False, default=str))
        if isinstance(normalized, dict):
            safe.append(normalized)
    return safe
