"""Durable L2 conversation memory.

The answer path only records a completed turn and marks its conversation dirty.
An in-process coordinator later folds every durable, unsummarized turn into the
stable summary.  Recent beats are a prompt convenience, never the fold source.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import inspect
import json
import logging
from pathlib import Path
import re
from typing import Any, Protocol

from app.db.sqlite import connect
from app.services.chat_history import utc_now

MAX_RECENT_BEATS = 3
BEAT_USER_CHARS = 800
BEAT_ASSISTANT_CHARS = 1200
SUMMARY_MAX_CHARS = 12000
PENDING_PROMPT_MAX_CHARS = 16000
FOLD_INPUT_MAX_CHARS = 48000
MEMORY_FOLD_DEBOUNCE_SECONDS = 8.0
# The old synchronous-looking helper has a fast lazy-start path so its public
# timing contract and focused tests remain compatible. Production startup
# should call ``start_memory_fold_coordinator`` and gets the real idle debounce.
MEMORY_FOLD_COMPAT_DEBOUNCE_SECONDS = 0.01
MEMORY_FOLD_RETRY_BASE_SECONDS = 2.0
MEMORY_FOLD_RETRY_MAX_SECONDS = 60.0
MEMORY_L3_RETRY_BASE_SECONDS = 2.0
MEMORY_L3_RETRY_MAX_SECONDS = 60.0

logger = logging.getLogger(__name__)

_MEMORY_OP_ACTIONS = frozenset({"upsert", "forget"})
_MEMORY_OP_SCOPES = frozenset({"user", "conversation"})
_MEMORY_OP_KINDS = frozenset({"semantic", "episodic", "procedural"})
_MEMORY_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
_SENSITIVE_MEMORY_RE = re.compile(
    r"\b(password|passcode|api[_ -]?key|access[_ -]?token|refresh[_ -]?token|"
    r"bearer|private[_ -]?key|secret|mật khẩu|mat khau|khóa bí mật|khoa bi mat)\b",
    flags=re.IGNORECASE,
)
_SENSITIVE_MEMORY_VALUE_RE = re.compile(
    r"(?:"
    r"\bsk-[A-Za-z0-9_-]{12,}\b|"
    r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b|"
    r"\bAKIA[0-9A-Z]{16}\b|"
    r"\bAIza[0-9A-Za-z_-]{20,}\b|"
    r"\bxox[baprs]-[A-Za-z0-9-]{12,}\b|"
    r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r")",
    flags=re.IGNORECASE,
)

_FOLD_SYSTEM = """You maintain compact rolling memory for a personal AI chat thread.
Return ONLY one JSON object with this shape:
{"summary":"plain-text rolling summary", "memory_ops":[]}

Keep summary under 800 words. Preserve across digressions:
- Active paper/document focus (names, files)
- Random/side topics the user brought up (so they can resume later)
- User preferences and open questions
- Key facts already established
Do not invent details. Prefer Vietnamese if the conversation is Vietnamese.

memory_ops must be [] unless the user explicitly supplied a durable preference,
decision, open commitment, or procedural instruction. When warranted, each item
must have exactly this semantic contract:
{"action":"upsert|forget", "scope":"user|conversation",
 "kind":"semantic|episodic|procedural", "key":"stable_normalized_key",
 "content":"grounded user fact/rule/event", "confidence":0.0}
Never store passwords, secrets, API keys, or authentication data. Never promote
assistant statements, retrieved paper claims, or model inferences into global
user memory. Use [] whenever scope, grounding, sensitivity, or intent is unclear.
"""


class SupportsChat(Protocol):
    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        num_predict: int = 512,
    ) -> Any: ...


MemoryOpsCallback = Callable[
    [str, int, list[dict[str, Any]]],
    Awaitable[None] | None,
]


@dataclass(frozen=True)
class TurnBeat:
    user: str
    assistant: str
    at: str
    revision: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "user": self.user,
            "assistant": self.assistant,
            "at": self.at,
            "revision": self.revision,
        }

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> TurnBeat | None:
        user = str(raw.get("user") or "").strip()
        assistant = str(raw.get("assistant") or "").strip()
        if not user and not assistant:
            return None
        return TurnBeat(
            user=user,
            assistant=assistant,
            at=str(raw.get("at") or utc_now()),
            revision=_nonnegative_int(raw.get("revision")),
        )


@dataclass(frozen=True)
class CompletedMemoryTurn:
    conversation_id: str
    turn_seq: int
    user_text: str
    assistant_text: str
    completed_at: str
    user_message_id: str | None = None
    assistant_message_id: str | None = None
    working_topic: str | None = None
    working_filenames: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "turn_seq": self.turn_seq,
            "user_message_id": self.user_message_id,
            "assistant_message_id": self.assistant_message_id,
            "user_text": self.user_text,
            "assistant_text": self.assistant_text,
            "working_topic": self.working_topic,
            "working_filenames": list(self.working_filenames),
            "completed_at": self.completed_at,
        }


@dataclass(frozen=True)
class _RecoveredMessagePair:
    conversation_id: str
    user_message_id: str
    assistant_message_id: str
    user_text: str
    assistant_text: str
    completed_at: str


@dataclass(frozen=True)
class ConversationMemoryJob:
    conversation_id: str
    dirty_through_seq: int
    summary_through_seq: int
    status: str
    attempt_count: int
    next_attempt_at: str | None
    last_error: str | None
    updated_at: str

    @property
    def has_pending_work(self) -> bool:
        return self.dirty_through_seq > self.summary_through_seq

    @property
    def is_dormant(self) -> bool:
        return self.status == "dormant"


@dataclass(frozen=True)
class PendingMemoryOperations:
    conversation_id: str
    source_turn_seq: int
    operations: tuple[dict[str, Any], ...]
    attempt_count: int
    next_attempt_at: str | None
    last_error: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class FoldOutput:
    summary: str
    memory_ops: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class ConversationMemoryFoldResult:
    memory: ConversationMemory
    output: FoldOutput
    target_revision: int


@dataclass(frozen=True)
class ConversationMemory:
    summary: str | None = None
    recent_beats: tuple[TurnBeat, ...] = ()
    pending_turns: tuple[CompletedMemoryTurn, ...] = ()
    updated_at: str | None = None
    revision: int = 0
    summary_revision: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "recent_beats": [beat.to_dict() for beat in self.recent_beats],
            "pending_turns": [turn.to_dict() for turn in self.pending_turns],
            "updated_at": self.updated_at,
            "revision": self.revision,
            "summary_revision": self.summary_revision,
        }

    def prompt_block(self, *, include_summary: bool = True) -> str:
        parts: list[str] = []
        if include_summary and self.summary:
            parts.append(f"Conversation summary (stable compressed context):\n{self.summary.strip()}")
        pending_revisions = {turn.turn_seq for turn in self.pending_turns}
        visible_beats = [
            beat for beat in self.recent_beats if beat.revision not in pending_revisions
        ]
        if visible_beats:
            lines = ["Recent turn notes (last exchanges, including digressions):"]
            for index, beat in enumerate(visible_beats, start=1):
                lines.append(f"{index}. user: {beat.user}")
                if beat.assistant:
                    lines.append(f"   assistant: {beat.assistant}")
            parts.append("\n".join(lines))
        if self.pending_turns:
            parts.append(_pending_turns_prompt(self.pending_turns))
        return "\n\n".join(parts)


def empty_memory() -> ConversationMemory:
    return ConversationMemory()


@dataclass(frozen=True)
class ConversationMemoryStore:
    db_path: Path

    def ensure_schema(self) -> None:
        with connect(self.db_path) as connection:
            _ensure_memory_tables(connection)

    def get_memory(self, conversation_id: str) -> ConversationMemory:
        with connect(self.db_path) as connection:
            _ensure_memory_tables(connection)
            row = connection.execute(
                "SELECT summary, metadata_json FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if row is None:
                return empty_memory()
            job_row = connection.execute(
                "SELECT * FROM conversation_memory_jobs WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            metadata_memory = _memory_from_row(row)
            summary_revision = (
                _nonnegative_int(job_row["summary_through_seq"])
                if job_row is not None
                else metadata_memory.summary_revision
            )
            turn_rows = connection.execute(
                """
                SELECT conversation_id, turn_seq, user_message_id, assistant_message_id,
                       user_text, assistant_text, working_topic, working_filenames_json,
                       completed_at
                FROM conversation_memory_turns
                WHERE conversation_id = ? AND turn_seq > ?
                ORDER BY turn_seq ASC
                """,
                (conversation_id, summary_revision),
            ).fetchall()
        pending_turns = tuple(_completed_turn_from_row(item) for item in turn_rows)
        revision = max(
            metadata_memory.revision,
            job_row["dirty_through_seq"] if job_row is not None else 0,
            *(turn.turn_seq for turn in pending_turns),
        )
        return ConversationMemory(
            summary=metadata_memory.summary,
            recent_beats=metadata_memory.recent_beats,
            pending_turns=pending_turns,
            updated_at=metadata_memory.updated_at,
            revision=revision,
            summary_revision=min(summary_revision, revision) if revision else summary_revision,
        )

    def get_job(self, conversation_id: str) -> ConversationMemoryJob | None:
        with connect(self.db_path) as connection:
            _ensure_memory_tables(connection)
            row = connection.execute(
                "SELECT * FROM conversation_memory_jobs WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return _job_from_row(row) if row is not None else None

    def list_pending_conversation_ids(self, *, due_only: bool = False) -> list[str]:
        now = utc_now()
        due_clause = "AND (next_attempt_at IS NULL OR next_attempt_at <= ?)" if due_only else ""
        params: tuple[Any, ...] = (now,) if due_only else ()
        with connect(self.db_path) as connection:
            _ensure_memory_tables(connection)
            rows = connection.execute(
                f"""
                SELECT conversation_id
                FROM conversation_memory_jobs
                WHERE dirty_through_seq > summary_through_seq
                  AND status != 'dormant'
                  {due_clause}
                ORDER BY updated_at ASC
                """,
                params,
            ).fetchall()
        return [str(row["conversation_id"]) for row in rows]

    def list_pending_l3_conversation_ids(
        self,
        *,
        due_only: bool = False,
    ) -> list[str]:
        """List conversations with unacknowledged post-commit L3 operations."""
        now = utc_now()
        due_clause = "AND (next_attempt_at IS NULL OR next_attempt_at <= ?)" if due_only else ""
        params: tuple[Any, ...] = (now,) if due_only else ()
        with connect(self.db_path) as connection:
            _ensure_memory_tables(connection)
            rows = connection.execute(
                f"""
                SELECT conversation_id, MIN(source_turn_seq) AS first_source_turn_seq
                FROM conversation_memory_l3_outbox
                WHERE status = 'pending'
                  {due_clause}
                GROUP BY conversation_id
                ORDER BY MIN(updated_at) ASC, first_source_turn_seq ASC
                """,
                params,
            ).fetchall()
        return [str(row["conversation_id"]) for row in rows]

    def get_next_pending_l3_operations(
        self,
        conversation_id: str,
    ) -> PendingMemoryOperations | None:
        """Return the oldest unacknowledged extraction for ordered replay."""
        with connect(self.db_path) as connection:
            _ensure_memory_tables(connection)
            row = connection.execute(
                """
                SELECT *
                FROM conversation_memory_l3_outbox
                WHERE conversation_id = ? AND status = 'pending'
                ORDER BY source_turn_seq ASC
                LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()
        return _pending_memory_operations_from_row(row) if row is not None else None

    def mark_l3_operations_delivered(
        self,
        conversation_id: str,
        *,
        source_turn_seq: int,
    ) -> None:
        now = utc_now()
        with connect(self.db_path) as connection:
            _ensure_memory_tables(connection)
            connection.execute(
                """
                UPDATE conversation_memory_l3_outbox
                SET status = 'delivered', next_attempt_at = NULL,
                    last_error = NULL, updated_at = ?, delivered_at = ?
                WHERE conversation_id = ? AND source_turn_seq = ?
                  AND status = 'pending'
                """,
                (now, now, conversation_id, max(0, int(source_turn_seq))),
            )

    def mark_l3_operations_failed(
        self,
        conversation_id: str,
        *,
        source_turn_seq: int,
        error: str,
    ) -> PendingMemoryOperations | None:
        now_dt = datetime.now(UTC)
        with connect(self.db_path) as connection:
            _ensure_memory_tables(connection)
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT *
                FROM conversation_memory_l3_outbox
                WHERE conversation_id = ? AND source_turn_seq = ?
                  AND status = 'pending'
                """,
                (conversation_id, max(0, int(source_turn_seq))),
            ).fetchone()
            if row is None:
                return None
            attempt_count = _nonnegative_int(row["attempt_count"]) + 1
            delay = min(
                MEMORY_L3_RETRY_MAX_SECONDS,
                MEMORY_L3_RETRY_BASE_SECONDS * (2 ** max(0, attempt_count - 1)),
            )
            next_attempt_at = (now_dt + timedelta(seconds=delay)).isoformat()
            connection.execute(
                """
                UPDATE conversation_memory_l3_outbox
                SET attempt_count = ?, next_attempt_at = ?, last_error = ?,
                    updated_at = ?
                WHERE conversation_id = ? AND source_turn_seq = ?
                  AND status = 'pending'
                """,
                (
                    attempt_count,
                    next_attempt_at,
                    _clip_error(error),
                    now_dt.isoformat(),
                    conversation_id,
                    max(0, int(source_turn_seq)),
                ),
            )
        return self.get_next_pending_l3_operations(conversation_id)

    def list_turns(
        self,
        conversation_id: str,
        *,
        after_seq: int = 0,
        through_seq: int | None = None,
    ) -> list[CompletedMemoryTurn]:
        through_clause = "AND turn_seq <= ?" if through_seq is not None else ""
        params: list[Any] = [conversation_id, max(0, after_seq)]
        if through_seq is not None:
            params.append(max(0, through_seq))
        with connect(self.db_path) as connection:
            _ensure_memory_tables(connection)
            rows = connection.execute(
                f"""
                SELECT conversation_id, turn_seq, user_message_id, assistant_message_id,
                       user_text, assistant_text, working_topic, working_filenames_json,
                       completed_at
                FROM conversation_memory_turns
                WHERE conversation_id = ? AND turn_seq > ? {through_clause}
                ORDER BY turn_seq ASC
                """,
                params,
            ).fetchall()
        return [_completed_turn_from_row(row) for row in rows]

    def append_turn_beat(
        self,
        conversation_id: str,
        *,
        user_text: str,
        assistant_text: str,
    ) -> ConversationMemory:
        """Legacy/test helper that only advances the bounded prompt beat ring.

        Production completion uses :meth:`record_completed_turn`; keeping this
        helper metadata-only preserves its historical semantics for callers that
        merely want a transient note rather than a durable fold job.
        """
        now = utc_now()
        with connect(self.db_path) as connection:
            _ensure_memory_tables(connection)
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT summary, metadata_json FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if row is None:
                return empty_memory()
            memory = _memory_from_row(row)
            revision = memory.revision + 1
            beat = TurnBeat(
                user=_clip_sentence_boundary(user_text, BEAT_USER_CHARS),
                assistant=_clip_sentence_boundary(assistant_text, BEAT_ASSISTANT_CHARS),
                at=now,
                revision=revision,
            )
            beats = [*memory.recent_beats, beat][-MAX_RECENT_BEATS:]
            metadata = _parse_json_object(row["metadata_json"])
            metadata["memory"] = _memory_metadata(
                beats=beats,
                updated_at=now,
                revision=revision,
                summary_revision=memory.summary_revision,
            )
            connection.execute(
                """
                UPDATE conversations
                SET metadata_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (json.dumps(metadata, ensure_ascii=False), now, conversation_id),
            )
        return ConversationMemory(
            summary=memory.summary,
            recent_beats=tuple(beats),
            updated_at=now,
            revision=revision,
            summary_revision=memory.summary_revision,
        )

    def record_completed_turn(
        self,
        conversation_id: str,
        *,
        user_text: str,
        assistant_text: str,
        working_topic: str | None = None,
        working_filenames: list[str] | None = None,
        user_message_id: str | None = None,
        assistant_message_id: str | None = None,
        completed_at: str | None = None,
    ) -> ConversationMemory:
        """Atomically append a durable turn, recent beat, and dirty job cursor."""
        now = completed_at or utc_now()
        with connect(self.db_path) as connection:
            _ensure_memory_tables(connection)
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT summary, metadata_json FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if row is None:
                return empty_memory()
            memory = _memory_from_row(row)
            if user_message_id:
                existing = connection.execute(
                    """
                    SELECT turn_seq FROM conversation_memory_turns
                    WHERE conversation_id = ? AND user_message_id = ?
                    """,
                    (conversation_id, user_message_id),
                ).fetchone()
                if existing is not None:
                    connection.commit()
                    return self.get_memory(conversation_id)
            max_turn_row = connection.execute(
                """
                SELECT COALESCE(MAX(turn_seq), 0) AS max_seq
                FROM conversation_memory_turns WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()
            job_row = connection.execute(
                "SELECT * FROM conversation_memory_jobs WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            max_seq = max(
                memory.revision,
                _nonnegative_int(max_turn_row["max_seq"] if max_turn_row else 0),
                _nonnegative_int(job_row["dirty_through_seq"] if job_row else 0),
            )
            turn_seq = max_seq + 1
            filenames = _dedupe_strings(working_filenames or [])
            connection.execute(
                """
                INSERT INTO conversation_memory_turns (
                    conversation_id, turn_seq, user_message_id, assistant_message_id,
                    user_text, assistant_text, working_topic, working_filenames_json,
                    completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    turn_seq,
                    user_message_id,
                    assistant_message_id,
                    user_text,
                    assistant_text,
                    _string_or_none(working_topic),
                    json.dumps(filenames, ensure_ascii=False),
                    now,
                ),
            )
            beat = TurnBeat(
                user=_clip_sentence_boundary(user_text, BEAT_USER_CHARS),
                assistant=_clip_sentence_boundary(assistant_text, BEAT_ASSISTANT_CHARS),
                at=now,
                revision=turn_seq,
            )
            beats = [*memory.recent_beats, beat][-MAX_RECENT_BEATS:]
            metadata = _parse_json_object(row["metadata_json"])
            metadata["memory"] = _memory_metadata(
                beats=beats,
                updated_at=now,
                revision=turn_seq,
                summary_revision=memory.summary_revision,
            )
            connection.execute(
                """
                UPDATE conversations
                SET metadata_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (json.dumps(metadata, ensure_ascii=False), now, conversation_id),
            )
            initial_summary_revision = (
                _nonnegative_int(job_row["summary_through_seq"])
                if job_row is not None
                else min(memory.summary_revision, turn_seq)
            )
            connection.execute(
                """
                INSERT INTO conversation_memory_jobs (
                    conversation_id, dirty_through_seq, summary_through_seq,
                    status, attempt_count, next_attempt_at, last_error, updated_at
                ) VALUES (?, ?, ?, 'pending', 0, NULL, NULL, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    dirty_through_seq = MAX(conversation_memory_jobs.dirty_through_seq, excluded.dirty_through_seq),
                    status = 'pending',
                    next_attempt_at = NULL,
                    updated_at = excluded.updated_at
                """,
                (conversation_id, turn_seq, initial_summary_revision, now),
            )
        return self.get_memory(conversation_id)

    def recover_completed_runs(self) -> list[str]:
        """Backfill completed agent runs missed between finalization and enqueue.

        Exact user-message IDs are preferred. Legacy rows written before IDs were
        wired are matched by exact user/assistant text and backfilled in place.
        Ambiguous assistant-message matches are deliberately left NULL.
        """
        with connect(self.db_path) as connection:
            _ensure_memory_tables(connection)
            run_rows = connection.execute(
                """
                SELECT ar.conversation_id, ar.user_message_id, ar.final_answer,
                       ar.ended_at, um.content AS user_text, um.created_at AS user_created_at
                FROM agent_runs ar
                JOIN messages um ON um.id = ar.user_message_id
                WHERE ar.status = 'completed'
                  AND ar.user_message_id IS NOT NULL
                  AND ar.final_answer IS NOT NULL
                ORDER BY ar.conversation_id ASC, ar.ended_at ASC, ar.started_at ASC
                """
            ).fetchall()

        grouped: dict[str, list[Any]] = {}
        for row in run_rows:
            grouped.setdefault(str(row["conversation_id"]), []).append(row)
        affected: list[str] = []
        for conversation_id, runs in grouped.items():
            if self._recover_conversation_runs(conversation_id, runs):
                affected.append(conversation_id)
        return affected

    def recover_completed_message_pairs(self) -> list[str]:
        """Recover direct-chat pairs that have no agent-run crash journal.

        Only physically adjacent ``user`` -> ``assistant`` rows are eligible.
        Any intervening system/orphan/repeated-user row breaks the pair, and a
        user message owned by *any* agent run is left to agent-run recovery.
        Exact message IDs make this scan idempotent.
        """
        with connect(self.db_path) as connection:
            _ensure_memory_tables(connection)
            message_rows = connection.execute(
                """
                SELECT rowid AS storage_order, id, conversation_id, role,
                       content, created_at
                FROM messages
                ORDER BY conversation_id ASC, created_at ASC, storage_order ASC
                """
            ).fetchall()
            agent_user_ids = {
                str(row["user_message_id"])
                for row in connection.execute(
                    """
                    SELECT user_message_id FROM agent_runs
                    WHERE user_message_id IS NOT NULL
                    """
                ).fetchall()
            }
            recorded_message_ids = {
                str(value)
                for row in connection.execute(
                    """
                    SELECT user_message_id, assistant_message_id
                    FROM conversation_memory_turns
                    """
                ).fetchall()
                for value in (row["user_message_id"], row["assistant_message_id"])
                if value
            }

        grouped: dict[str, list[Any]] = {}
        for row in message_rows:
            grouped.setdefault(str(row["conversation_id"]), []).append(row)

        affected: list[str] = []
        for conversation_id, rows in grouped.items():
            pairs: list[_RecoveredMessagePair] = []
            index = 0
            while index + 1 < len(rows):
                user_row = rows[index]
                assistant_row = rows[index + 1]
                if (
                    str(user_row["role"]) == "user"
                    and str(assistant_row["role"]) == "assistant"
                ):
                    user_message_id = str(user_row["id"])
                    assistant_message_id = str(assistant_row["id"])
                    if (
                        user_message_id not in agent_user_ids
                        and user_message_id not in recorded_message_ids
                        and assistant_message_id not in recorded_message_ids
                        and str(user_row["content"] or "").strip()
                        and str(assistant_row["content"] or "").strip()
                    ):
                        pairs.append(
                            _RecoveredMessagePair(
                                conversation_id=conversation_id,
                                user_message_id=user_message_id,
                                assistant_message_id=assistant_message_id,
                                user_text=str(user_row["content"]),
                                assistant_text=str(assistant_row["content"]),
                                completed_at=str(
                                    assistant_row["created_at"] or utc_now()
                                ),
                            )
                        )
                    # This assistant belongs only to its physically adjacent
                    # user, even when the pair is excluded as agent-owned.
                    index += 2
                    continue
                index += 1
            if pairs and self._recover_conversation_message_pairs(
                conversation_id,
                pairs,
            ):
                affected.append(conversation_id)
        return affected

    def _recover_conversation_message_pairs(
        self,
        conversation_id: str,
        pairs: list[_RecoveredMessagePair],
    ) -> bool:
        now = utc_now()
        with connect(self.db_path) as connection:
            _ensure_memory_tables(connection)
            connection.execute("BEGIN IMMEDIATE")
            conversation_row = connection.execute(
                "SELECT summary, metadata_json FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if conversation_row is None:
                return False
            memory = _memory_from_row(conversation_row)
            job_row = connection.execute(
                "SELECT * FROM conversation_memory_jobs WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            existing_rows = connection.execute(
                """
                SELECT conversation_id, turn_seq, user_message_id,
                       assistant_message_id, user_text, assistant_text,
                       completed_at
                FROM conversation_memory_turns
                WHERE conversation_id = ?
                ORDER BY turn_seq ASC
                """,
                (conversation_id,),
            ).fetchall()
            existing_user_ids = {
                str(row["user_message_id"])
                for row in existing_rows
                if row["user_message_id"]
            }
            existing_assistant_ids = {
                str(row["assistant_message_id"])
                for row in existing_rows
                if row["assistant_message_id"]
            }
            agent_user_ids = {
                str(row["user_message_id"])
                for row in connection.execute(
                    """
                    SELECT user_message_id FROM agent_runs
                    WHERE conversation_id = ? AND user_message_id IS NOT NULL
                    """,
                    (conversation_id,),
                ).fetchall()
            }
            max_sequence = max(
                memory.revision,
                _nonnegative_int(
                    job_row["dirty_through_seq"] if job_row is not None else 0
                ),
                *(
                    _nonnegative_int(row["turn_seq"])
                    for row in existing_rows
                ),
            )
            inserted = False
            for pair in pairs:
                if (
                    pair.user_message_id in agent_user_ids
                    or pair.user_message_id in existing_user_ids
                    or pair.assistant_message_id in existing_assistant_ids
                ):
                    continue
                max_sequence += 1
                connection.execute(
                    """
                    INSERT INTO conversation_memory_turns (
                        conversation_id, turn_seq, user_message_id,
                        assistant_message_id, user_text, assistant_text,
                        working_topic, working_filenames_json, completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, NULL, '[]', ?)
                    """,
                    (
                        conversation_id,
                        max_sequence,
                        pair.user_message_id,
                        pair.assistant_message_id,
                        pair.user_text,
                        pair.assistant_text,
                        pair.completed_at,
                    ),
                )
                existing_user_ids.add(pair.user_message_id)
                existing_assistant_ids.add(pair.assistant_message_id)
                inserted = True
            if not inserted:
                return False

            refreshed_rows = connection.execute(
                """
                SELECT turn_seq, user_text, assistant_text, completed_at
                FROM conversation_memory_turns
                WHERE conversation_id = ?
                ORDER BY turn_seq ASC
                """,
                (conversation_id,),
            ).fetchall()
            dirty_through = max(
                memory.revision,
                *(
                    _nonnegative_int(row["turn_seq"])
                    for row in refreshed_rows
                ),
            )
            beats = [
                TurnBeat(
                    user=_clip_sentence_boundary(
                        str(row["user_text"] or ""),
                        BEAT_USER_CHARS,
                    ),
                    assistant=_clip_sentence_boundary(
                        str(row["assistant_text"] or ""),
                        BEAT_ASSISTANT_CHARS,
                    ),
                    at=str(row["completed_at"] or now),
                    revision=_nonnegative_int(row["turn_seq"]),
                )
                for row in refreshed_rows[-MAX_RECENT_BEATS:]
            ]
            metadata = _parse_json_object(conversation_row["metadata_json"])
            metadata["memory"] = _memory_metadata(
                beats=beats,
                updated_at=now,
                revision=dirty_through,
                summary_revision=min(memory.summary_revision, dirty_through),
            )
            connection.execute(
                """
                UPDATE conversations
                SET metadata_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (json.dumps(metadata, ensure_ascii=False), now, conversation_id),
            )
            summary_through = (
                _nonnegative_int(job_row["summary_through_seq"])
                if job_row is not None
                else min(memory.summary_revision, dirty_through)
            )
            unmarked_legacy = (
                _nonnegative_int(metadata.get("memory_queue_version")) == 0
            )
            legacy_dormant = unmarked_legacy and (
                (job_row is None and memory.revision == 0)
                or (
                    job_row is not None
                    and str(job_row["status"] or "") == "dormant"
                )
            )
            status = (
                "dormant"
                if legacy_dormant and dirty_through > summary_through
                else ("pending" if dirty_through > summary_through else "idle")
            )
            connection.execute(
                """
                INSERT INTO conversation_memory_jobs (
                    conversation_id, dirty_through_seq, summary_through_seq,
                    status, attempt_count, next_attempt_at, last_error, updated_at
                ) VALUES (?, ?, ?, ?, 0, NULL, NULL, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    dirty_through_seq = MAX(
                        conversation_memory_jobs.dirty_through_seq,
                        excluded.dirty_through_seq
                    ),
                    status = CASE
                        WHEN conversation_memory_jobs.status = 'dormant'
                             AND excluded.status = 'dormant' THEN 'dormant'
                        WHEN MAX(
                            conversation_memory_jobs.dirty_through_seq,
                            excluded.dirty_through_seq
                        ) > conversation_memory_jobs.summary_through_seq
                            THEN 'pending'
                        ELSE 'idle'
                    END,
                    next_attempt_at = NULL,
                    updated_at = excluded.updated_at
                """,
                (
                    conversation_id,
                    dirty_through,
                    summary_through,
                    status,
                    now,
                ),
            )
        return True

    def _recover_conversation_runs(self, conversation_id: str, runs: list[Any]) -> bool:
        now = utc_now()
        changed = False
        with connect(self.db_path) as connection:
            _ensure_memory_tables(connection)
            connection.execute("BEGIN IMMEDIATE")
            conversation_row = connection.execute(
                "SELECT summary, metadata_json FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if conversation_row is None:
                return False
            memory = _memory_from_row(conversation_row)
            existing_rows = connection.execute(
                """
                SELECT conversation_id, turn_seq, user_message_id, assistant_message_id,
                       user_text, assistant_text, working_topic, working_filenames_json,
                       completed_at
                FROM conversation_memory_turns
                WHERE conversation_id = ? ORDER BY turn_seq ASC
                """,
                (conversation_id,),
            ).fetchall()
            existing_by_user = {
                str(row["user_message_id"]): row
                for row in existing_rows
                if row["user_message_id"]
            }
            unmatched_existing = [row for row in existing_rows if not row["user_message_id"]]
            used_sequences = {_nonnegative_int(row["turn_seq"]) for row in existing_rows}
            next_sequence = max(memory.revision, max(used_sequences, default=0)) + 1

            for ordinal, run in enumerate(runs, start=1):
                user_message_id = str(run["user_message_id"])
                if user_message_id in existing_by_user:
                    continue
                user_text = str(run["user_text"] or "")
                assistant_text = str(run["final_answer"] or "")
                matched = next(
                    (
                        row
                        for row in unmatched_existing
                        if str(row["user_text"] or "") == user_text
                        and str(row["assistant_text"] or "") == assistant_text
                    ),
                    None,
                )
                assistant_message_id = _resolve_assistant_message_id(
                    connection,
                    conversation_id=conversation_id,
                    assistant_text=assistant_text,
                    user_created_at=str(run["user_created_at"] or ""),
                )
                if matched is not None:
                    connection.execute(
                        """
                        UPDATE conversation_memory_turns
                        SET user_message_id = ?,
                            assistant_message_id = COALESCE(assistant_message_id, ?)
                        WHERE conversation_id = ? AND turn_seq = ?
                        """,
                        (
                            user_message_id,
                            assistant_message_id,
                            conversation_id,
                            matched["turn_seq"],
                        ),
                    )
                    unmatched_existing.remove(matched)
                    changed = True
                    continue

                preferred_sequence = ordinal
                if preferred_sequence in used_sequences:
                    while next_sequence in used_sequences:
                        next_sequence += 1
                    preferred_sequence = next_sequence
                    next_sequence += 1
                used_sequences.add(preferred_sequence)
                connection.execute(
                    """
                    INSERT INTO conversation_memory_turns (
                        conversation_id, turn_seq, user_message_id, assistant_message_id,
                        user_text, assistant_text, working_topic, working_filenames_json,
                        completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, NULL, '[]', ?)
                    """,
                    (
                        conversation_id,
                        preferred_sequence,
                        user_message_id,
                        assistant_message_id,
                        user_text,
                        assistant_text,
                        str(run["ended_at"] or now),
                    ),
                )
                changed = True

            if not changed:
                return False
            refreshed_rows = connection.execute(
                """
                SELECT turn_seq, user_text, assistant_text, completed_at
                FROM conversation_memory_turns
                WHERE conversation_id = ? ORDER BY turn_seq ASC
                """,
                (conversation_id,),
            ).fetchall()
            dirty_through = max(
                memory.revision,
                *(_nonnegative_int(row["turn_seq"]) for row in refreshed_rows),
            )
            beats = [
                TurnBeat(
                    user=_clip_sentence_boundary(str(row["user_text"] or ""), BEAT_USER_CHARS),
                    assistant=_clip_sentence_boundary(
                        str(row["assistant_text"] or ""), BEAT_ASSISTANT_CHARS
                    ),
                    at=str(row["completed_at"] or now),
                    revision=_nonnegative_int(row["turn_seq"]),
                )
                for row in refreshed_rows[-MAX_RECENT_BEATS:]
            ]
            metadata = _parse_json_object(conversation_row["metadata_json"])
            metadata["memory"] = _memory_metadata(
                beats=beats,
                updated_at=now,
                revision=dirty_through,
                summary_revision=min(memory.summary_revision, dirty_through),
            )
            connection.execute(
                "UPDATE conversations SET metadata_json = ?, updated_at = ? WHERE id = ?",
                (json.dumps(metadata, ensure_ascii=False), now, conversation_id),
            )
            job_row = connection.execute(
                "SELECT * FROM conversation_memory_jobs WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            summary_through = (
                _nonnegative_int(job_row["summary_through_seq"])
                if job_row is not None
                else min(memory.summary_revision, dirty_through)
            )
            # Conversations created by the durable-memory runtime carry a
            # generation marker from birth.  If one of those has a completed
            # agent run but no queue row, the process crashed in the small gap
            # between answer finalization and ``record_completed_turn`` and it
            # must be recovered as pending.  Only unmarked pre-migration
            # conversations are lazy/dormant.
            unmarked_legacy = (
                _nonnegative_int(metadata.get("memory_queue_version")) == 0
            )
            legacy_dormant = unmarked_legacy and (
                (job_row is None and memory.revision == 0)
                or (
                    job_row is not None
                    and str(job_row["status"] or "") == "dormant"
                )
            )
            status = (
                "dormant"
                if legacy_dormant and dirty_through > summary_through
                else ("pending" if dirty_through > summary_through else "idle")
            )
            connection.execute(
                """
                INSERT INTO conversation_memory_jobs (
                    conversation_id, dirty_through_seq, summary_through_seq,
                    status, attempt_count, next_attempt_at, last_error, updated_at
                ) VALUES (?, ?, ?, ?, 0, NULL, NULL, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    dirty_through_seq = MAX(conversation_memory_jobs.dirty_through_seq, excluded.dirty_through_seq),
                    status = CASE
                        WHEN conversation_memory_jobs.status = 'dormant'
                             AND excluded.status = 'dormant' THEN 'dormant'
                        WHEN MAX(conversation_memory_jobs.dirty_through_seq, excluded.dirty_through_seq)
                             > conversation_memory_jobs.summary_through_seq THEN 'pending'
                        ELSE 'idle'
                    END,
                    next_attempt_at = NULL,
                    updated_at = excluded.updated_at
                """,
                (conversation_id, dirty_through, summary_through, status, now),
            )
        return True

    def set_summary(self, conversation_id: str, summary: str | None) -> ConversationMemory:
        """Force-set a summary at the current durable revision (admin/test helper)."""
        return self._write_summary(
            conversation_id,
            summary=summary,
            revision=None,
            only_if_newer=False,
        )

    def set_summary_if_newer(
        self,
        conversation_id: str,
        *,
        summary: str | None,
        revision: int,
        memory_ops: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    ) -> ConversationMemory:
        """Atomically commit a cumulative fold and its post-commit L3 outbox."""
        return self._write_summary(
            conversation_id,
            summary=summary,
            revision=revision,
            only_if_newer=True,
            memory_ops=memory_ops,
        )

    def _write_summary(
        self,
        conversation_id: str,
        *,
        summary: str | None,
        revision: int | None,
        only_if_newer: bool,
        memory_ops: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    ) -> ConversationMemory:
        now = utc_now()
        cleaned_summary = _clip_sentence_boundary(summary or "", SUMMARY_MAX_CHARS) or None
        with connect(self.db_path) as connection:
            _ensure_memory_tables(connection)
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT summary, metadata_json FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if row is None:
                return empty_memory()
            memory = _memory_from_row(row)
            job_row = connection.execute(
                "SELECT * FROM conversation_memory_jobs WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            current_summary_revision = max(
                memory.summary_revision,
                _nonnegative_int(job_row["summary_through_seq"] if job_row else 0),
            )
            target_revision = memory.revision if revision is None else max(0, revision)
            if only_if_newer and target_revision <= current_summary_revision:
                connection.commit()
                return self.get_memory(conversation_id)
            metadata = _parse_json_object(row["metadata_json"])
            next_revision = max(memory.revision, target_revision)
            next_summary_revision = max(current_summary_revision, target_revision)
            metadata["memory"] = _memory_metadata(
                beats=list(memory.recent_beats),
                updated_at=now,
                revision=next_revision,
                summary_revision=next_summary_revision,
            )
            connection.execute(
                """
                UPDATE conversations
                SET summary = ?, metadata_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    cleaned_summary,
                    json.dumps(metadata, ensure_ascii=False),
                    now,
                    conversation_id,
                ),
            )
            dirty_through = max(
                next_revision,
                _nonnegative_int(job_row["dirty_through_seq"] if job_row else 0),
            )
            status = "pending" if dirty_through > next_summary_revision else "idle"
            connection.execute(
                """
                INSERT INTO conversation_memory_jobs (
                    conversation_id, dirty_through_seq, summary_through_seq,
                    status, attempt_count, next_attempt_at, last_error, updated_at
                ) VALUES (?, ?, ?, ?, 0, NULL, NULL, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    dirty_through_seq = MAX(conversation_memory_jobs.dirty_through_seq, excluded.dirty_through_seq),
                    summary_through_seq = MAX(conversation_memory_jobs.summary_through_seq, excluded.summary_through_seq),
                    status = excluded.status,
                    attempt_count = 0,
                    next_attempt_at = NULL,
                    last_error = NULL,
                    updated_at = excluded.updated_at
                """,
                (
                    conversation_id,
                    dirty_through,
                    next_summary_revision,
                    status,
                    now,
                ),
            )
            if memory_ops:
                connection.execute(
                    """
                    INSERT INTO conversation_memory_l3_outbox (
                        conversation_id, source_turn_seq, operations_json,
                        status, attempt_count, next_attempt_at, last_error,
                        created_at, updated_at, delivered_at
                    ) VALUES (?, ?, ?, 'pending', 0, NULL, NULL, ?, ?, NULL)
                    ON CONFLICT(conversation_id, source_turn_seq) DO NOTHING
                    """,
                    (
                        conversation_id,
                        target_revision,
                        json.dumps(
                            list(memory_ops),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        now,
                        now,
                    ),
                )
        return self.get_memory(conversation_id)

    def mark_fold_running(self, conversation_id: str, *, through_seq: int) -> None:
        now = utc_now()
        with connect(self.db_path) as connection:
            _ensure_memory_tables(connection)
            connection.execute(
                """
                UPDATE conversation_memory_jobs
                SET status = 'running', updated_at = ?
                WHERE conversation_id = ?
                  AND dirty_through_seq >= ?
                  AND summary_through_seq < ?
                """,
                (now, conversation_id, through_seq, through_seq),
            )

    def mark_fold_deferred(self, conversation_id: str) -> None:
        with connect(self.db_path) as connection:
            _ensure_memory_tables(connection)
            connection.execute(
                """
                UPDATE conversation_memory_jobs
                SET status = CASE
                        WHEN dirty_through_seq > summary_through_seq THEN 'pending'
                        ELSE 'idle'
                    END,
                    updated_at = ?
                WHERE conversation_id = ?
                """,
                (utc_now(), conversation_id),
            )

    def mark_fold_failed(self, conversation_id: str, error: str) -> ConversationMemoryJob | None:
        now_dt = datetime.now(UTC)
        with connect(self.db_path) as connection:
            _ensure_memory_tables(connection)
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM conversation_memory_jobs WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            if row is None:
                return None
            attempt_count = _nonnegative_int(row["attempt_count"]) + 1
            delay = min(
                MEMORY_FOLD_RETRY_MAX_SECONDS,
                MEMORY_FOLD_RETRY_BASE_SECONDS * (2 ** max(0, attempt_count - 1)),
            )
            next_attempt_at = (now_dt + timedelta(seconds=delay)).isoformat()
            connection.execute(
                """
                UPDATE conversation_memory_jobs
                SET status = 'pending', attempt_count = ?, next_attempt_at = ?,
                    last_error = ?, updated_at = ?
                WHERE conversation_id = ?
                """,
                (
                    attempt_count,
                    next_attempt_at,
                    _clip_error(error),
                    now_dt.isoformat(),
                    conversation_id,
                ),
            )
        return self.get_job(conversation_id)

    def recover_interrupted_jobs(self) -> list[str]:
        """Turn pre-crash running jobs back into retryable pending jobs."""
        now = utc_now()
        with connect(self.db_path) as connection:
            _ensure_memory_tables(connection)
            connection.execute(
                """
                UPDATE conversation_memory_jobs
                SET status = CASE
                        WHEN dirty_through_seq > summary_through_seq THEN 'pending'
                        ELSE 'idle'
                    END,
                    next_attempt_at = CASE
                        WHEN dirty_through_seq > summary_through_seq THEN NULL
                        ELSE next_attempt_at
                    END,
                    last_error = CASE
                        WHEN dirty_through_seq > summary_through_seq
                            THEN COALESCE(last_error, 'recovered_interrupted_fold')
                        ELSE last_error
                    END,
                    updated_at = ?
                WHERE status = 'running'
                """,
                (now,),
            )
            rows = connection.execute(
                """
                SELECT conversation_id FROM conversation_memory_jobs
                WHERE dirty_through_seq > summary_through_seq
                  AND status != 'dormant'
                ORDER BY updated_at ASC
                """
            ).fetchall()
        return [str(row["conversation_id"]) for row in rows]


async def fold_conversation_summary_result(
    *,
    store: ConversationMemoryStore,
    conversation_id: str,
    client: SupportsChat,
    model: str,
    user_text: str = "",
    assistant_text: str = "",
    working_topic: str | None = None,
    working_filenames: list[str] | None = None,
    revision: int | None = None,
) -> ConversationMemoryFoldResult:
    """Fold every durable turn after the stable cursor through one target cursor."""
    memory = store.get_memory(conversation_id)
    job = store.get_job(conversation_id)
    requested_revision = (
        max(memory.revision, memory.summary_revision) + 1
        if revision is None and job is None
        else (
            job.dirty_through_seq
            if revision is None and job is not None
            else max(0, int(revision or 0))
        )
    )
    pending_turns = store.list_turns(
        conversation_id,
        after_seq=memory.summary_revision,
        through_seq=requested_revision,
    )
    if pending_turns:
        turn_budget = max(
            4000,
            FOLD_INPUT_MAX_CHARS - len(memory.summary or "") - 1200,
        )
        pending_turns = _fold_turn_batch(pending_turns, max_chars=turn_budget)
        target_revision = pending_turns[-1].turn_seq
        turns_block = "\n\n".join(
            f"turn {turn.turn_seq}:\n"
            f"user: {turn.user_text}\n"
            f"assistant: {turn.assistant_text}"
            for turn in pending_turns
        )
        latest_turn = pending_turns[-1]
        focus_topic = latest_turn.working_topic or working_topic or "(none)"
        focus_files = ", ".join(latest_turn.working_filenames or tuple(working_filenames or [])) or "(none)"
    else:
        if job is not None and requested_revision > memory.summary_revision:
            error = (
                "durable memory cursor gap: no turn rows after summary cursor "
                f"{memory.summary_revision} through requested cursor {requested_revision}"
            )
            store.mark_fold_failed(conversation_id, error)
            raise RuntimeError(error)
        # Compatibility for direct callers that have not recorded a turn yet.
        target_revision = requested_revision
        turns_block = f"user: {user_text}\nassistant: {assistant_text}"
        focus_topic = working_topic or "(none)"
        focus_files = ", ".join(working_filenames or []) or "(none)"
    user_content = (
        f"Active working focus: topic={focus_topic}; files={focus_files}\n\n"
        f"Previous stable summary through turn {memory.summary_revision}:\n"
        f"{memory.summary or '(empty)'}\n\n"
        f"New completed turns through turn {target_revision}:\n{turns_block}\n\n"
        "Return the updated rolling-memory JSON object."
    )
    store.mark_fold_running(conversation_id, through_seq=target_revision)
    try:
        completion = await client.chat(
            model=model,
            temperature=0.1,
            num_predict=1600,
            messages=[
                {"role": "system", "content": _FOLD_SYSTEM},
                {"role": "user", "content": user_content},
            ],
        )
        raw = getattr(completion, "message", None) or str(completion)
        output = _parse_fold_output(raw)
        if not output.summary.strip():
            raise RuntimeError("memory fold returned an empty summary")
    except asyncio.CancelledError:
        store.mark_fold_deferred(conversation_id)
        raise
    except Exception as exc:
        store.mark_fold_failed(conversation_id, str(exc))
        raise

    updated = store.set_summary_if_newer(
        conversation_id,
        summary=output.summary,
        revision=target_revision,
        memory_ops=output.memory_ops,
    )
    return ConversationMemoryFoldResult(
        memory=updated,
        output=output,
        target_revision=target_revision,
    )


async def fold_conversation_summary(
    *,
    store: ConversationMemoryStore,
    conversation_id: str,
    client: SupportsChat,
    model: str,
    user_text: str,
    assistant_text: str,
    working_topic: str | None = None,
    working_filenames: list[str] | None = None,
    revision: int | None = None,
) -> ConversationMemory:
    """Compatibility wrapper returning the updated memory object."""
    result = await fold_conversation_summary_result(
        store=store,
        conversation_id=conversation_id,
        client=client,
        model=model,
        user_text=user_text,
        assistant_text=assistant_text,
        working_topic=working_topic,
        working_filenames=working_filenames,
        revision=revision,
    )
    return result.memory


@dataclass
class _CoordinatorRegistration:
    store: ConversationMemoryStore
    client: SupportsChat
    model: str
    on_memory_ops: MemoryOpsCallback | None = None


class ConversationMemoryCoordinator:
    """One coalescing worker per conversation and one fold globally."""

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        debounce_seconds: float = MEMORY_FOLD_DEBOUNCE_SECONDS,
    ) -> None:
        self.loop = loop
        self.debounce_seconds = max(0.0, debounce_seconds)
        self._fold_semaphore = asyncio.Semaphore(1)
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._foreground_counts: dict[str, int] = {}
        self._process_foreground_depth = 0
        self._registration: _CoordinatorRegistration | None = None
        self._recovery_task: asyncio.Task[Any] | None = None
        self._started = False
        self._stopping = False

    def start(
        self,
        *,
        store: ConversationMemoryStore,
        client: SupportsChat,
        model: str,
        on_memory_ops: MemoryOpsCallback | None = None,
    ) -> None:
        """Configure the worker and launch the durable startup recovery scan."""
        retained_memory_ops = (
            on_memory_ops
            if on_memory_ops is not None
            else (
                self._registration.on_memory_ops
                if self._registration is not None
                else None
            )
        )
        self._registration = _CoordinatorRegistration(
            store,
            client,
            model,
            retained_memory_ops,
        )
        if not self._started:
            self._started = True
            self._recovery_task = self.loop.create_task(self._recover_and_enqueue())
        else:
            # A coordinator can be created by the compatibility scheduler
            # before application lifespan installs the L3 callback. Rebinding
            # that callback must wake any outbox rows produced in between.
            self._resume_pending_if_idle()

    def enqueue(self, conversation_id: str) -> None:
        if self._stopping or self._registration is None or self._has_foreground_activity:
            return
        job = self._registration.store.get_job(conversation_id)
        has_l2_work = (
            job is not None and job.has_pending_work and not job.is_dormant
        )
        has_l3_work = (
            self._registration.on_memory_ops is not None
            and self._registration.store.get_next_pending_l3_operations(
                conversation_id
            )
            is not None
        )
        if not has_l2_work and not has_l3_work:
            return
        current = self._tasks.get(conversation_id)
        if current is not None and not current.done():
            return
        task = self.loop.create_task(self._run_conversation(conversation_id))
        self._tasks[conversation_id] = task
        task.add_done_callback(
            lambda done, cid=conversation_id: self._task_done(cid, done)
        )

    def foreground_started(self, conversation_id: str) -> None:
        """Reference-count a foreground run and globally defer all GPT folds."""
        self._foreground_counts[conversation_id] = (
            self._foreground_counts.get(conversation_id, 0) + 1
        )
        self._cancel_running_folds()

    def foreground_finished(self, conversation_id: str) -> None:
        count = self._foreground_counts.get(conversation_id, 0)
        if count <= 1:
            self._foreground_counts.pop(conversation_id, None)
        else:
            self._foreground_counts[conversation_id] = count - 1
        self._resume_pending_if_idle()

    def global_foreground_started(self) -> None:
        """Runtime-gate hook: one more foreground answer is queued/running."""
        self._process_foreground_depth += 1
        self._cancel_running_folds()

    def global_foreground_finished(self) -> None:
        """Runtime-gate hook: resume folds only after the final answer is idle."""
        self._process_foreground_depth = max(0, self._process_foreground_depth - 1)
        self._resume_pending_if_idle()

    @property
    def _has_foreground_activity(self) -> bool:
        return self._process_foreground_depth > 0 or bool(self._foreground_counts)

    def _cancel_running_folds(self) -> None:
        for task in list(self._tasks.values()):
            if not task.done():
                task.cancel()

    def _resume_pending_if_idle(self) -> None:
        if self._has_foreground_activity or self._registration is None or self._stopping:
            return
        pending = self._registration.store.list_pending_conversation_ids()
        if self._registration.on_memory_ops is not None:
            pending.extend(
                self._registration.store.list_pending_l3_conversation_ids()
            )
        for conversation_id in _dedupe_strings(pending):
            self.enqueue(conversation_id)

    async def shutdown(self, *, timeout_seconds: float = 2.0) -> None:
        """Bounded drain; unfinished jobs remain durable and pending for restart."""
        self._stopping = True
        tasks = [task for task in self._tasks.values() if not task.done()]
        if self._recovery_task is not None and not self._recovery_task.done():
            tasks.append(self._recovery_task)
        if not tasks:
            return
        done, pending = await asyncio.wait(tasks, timeout=max(0.0, timeout_seconds))
        del done
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def _recover_and_enqueue(self) -> None:
        registration = self._registration
        if registration is None or self._stopping:
            return
        try:
            recovered_conversations = await asyncio.to_thread(
                registration.store.recover_completed_runs
            )
            recovered_message_pairs = await asyncio.to_thread(
                registration.store.recover_completed_message_pairs
            )
            interrupted = await asyncio.to_thread(
                registration.store.recover_interrupted_jobs
            )
            pending_l3 = (
                await asyncio.to_thread(
                    registration.store.list_pending_l3_conversation_ids
                )
                if registration.on_memory_ops is not None
                else []
            )
            for conversation_id in _dedupe_strings(
                [
                    *recovered_conversations,
                    *recovered_message_pairs,
                    *interrupted,
                    *pending_l3,
                ]
            ):
                self.enqueue(conversation_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("L2 startup recovery scan failed")

    async def _run_conversation(self, conversation_id: str) -> None:
        registration = self._registration
        if registration is None:
            return
        try:
            await asyncio.sleep(self.debounce_seconds)
            while not self._stopping:
                if self._has_foreground_activity:
                    registration.store.mark_fold_deferred(conversation_id)
                    return
                job = registration.store.get_job(conversation_id)
                has_l2_work = (
                    job is not None
                    and job.has_pending_work
                    and not job.is_dormant
                )
                l2_delay = (
                    _seconds_until(job.next_attempt_at)
                    if has_l2_work and job is not None
                    else None
                )
                pending_l3 = (
                    registration.store.get_next_pending_l3_operations(
                        conversation_id
                    )
                    if registration.on_memory_ops is not None
                    else None
                )
                l3_delay = (
                    _seconds_until(pending_l3.next_attempt_at)
                    if pending_l3 is not None
                    else None
                )

                # Replay the oldest extraction first. This preserves temporal
                # update/forget ordering across folds; receipts in the L3 store
                # make a callback that committed before an ack safe to replay.
                if pending_l3 is not None and (l3_delay or 0.0) <= 0:
                    try:
                        maybe_awaitable = registration.on_memory_ops(
                            conversation_id,
                            pending_l3.source_turn_seq,
                            list(pending_l3.operations),
                        )
                        if inspect.isawaitable(maybe_awaitable):
                            await maybe_awaitable
                        registration.store.mark_l3_operations_delivered(
                            conversation_id,
                            source_turn_seq=pending_l3.source_turn_seq,
                        )
                    except asyncio.CancelledError:
                        # No ack: startup/idle replay will retry this exact
                        # source cursor without another LLM fold.
                        raise
                    except Exception as exc:
                        registration.store.mark_l3_operations_failed(
                            conversation_id,
                            source_turn_seq=pending_l3.source_turn_seq,
                            error=str(exc),
                        )
                        logger.exception(
                            "Could not apply durable L3 memory operations for "
                            "conversation %s at turn %s",
                            conversation_id,
                            pending_l3.source_turn_seq,
                        )
                    continue

                if not has_l2_work:
                    if l3_delay is None:
                        return
                    await asyncio.sleep(max(0.0, l3_delay))
                    continue

                if l2_delay is not None and l2_delay > 0:
                    delays = [l2_delay]
                    if l3_delay is not None and l3_delay > 0:
                        delays.append(l3_delay)
                    await asyncio.sleep(min(delays))
                    continue

                async with self._fold_semaphore:
                    if self._has_foreground_activity:
                        registration.store.mark_fold_deferred(conversation_id)
                        return
                    try:
                        await fold_conversation_summary_result(
                            store=registration.store,
                            conversation_id=conversation_id,
                            client=registration.client,
                            model=registration.model,
                            revision=job.dirty_through_seq,
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        # The fold function already persisted error/backoff. Keep
                        # this worker alive so the durable job can retry.
                        continue
                await asyncio.sleep(self.debounce_seconds)
        except asyncio.CancelledError:
            registration.store.mark_fold_deferred(conversation_id)
            raise

    def _task_done(self, conversation_id: str, task: asyncio.Task[Any]) -> None:
        if self._tasks.get(conversation_id) is task:
            self._tasks.pop(conversation_id, None)
        if not task.cancelled():
            error = task.exception()  # consume unexpected errors
            if error is not None:
                logger.error(
                    "Unexpected L2 worker failure for conversation %s: %s",
                    conversation_id,
                    error,
                )
        if self._stopping or self._has_foreground_activity or self._registration is None:
            return
        try:
            job = self._registration.store.get_job(conversation_id)
            pending_l3 = (
                self._registration.store.get_next_pending_l3_operations(
                    conversation_id
                )
                if self._registration.on_memory_ops is not None
                else None
            )
        except Exception:
            logger.exception("Could not inspect L2 job after worker exit")
            return
        if (
            job is not None and job.has_pending_work and not job.is_dormant
        ) or pending_l3 is not None:
            self.enqueue(conversation_id)


_coordinators: dict[Path, ConversationMemoryCoordinator] = {}
_global_foreground_depth = 0


def start_memory_fold_coordinator(
    *,
    store: ConversationMemoryStore,
    client: SupportsChat,
    model: str,
    on_memory_ops: MemoryOpsCallback | None = None,
    debounce_seconds: float = MEMORY_FOLD_DEBOUNCE_SECONDS,
) -> ConversationMemoryCoordinator:
    """Start/reconfigure the current-loop coordinator and recover durable work."""
    loop = asyncio.get_running_loop()
    key = store.db_path.resolve()
    coordinator = _coordinators.get(key)
    if coordinator is None or coordinator.loop is not loop or coordinator.loop.is_closed():
        coordinator = ConversationMemoryCoordinator(
            loop=loop,
            debounce_seconds=debounce_seconds,
        )
        coordinator._process_foreground_depth = _global_foreground_depth
        _coordinators[key] = coordinator
    coordinator.start(
        store=store,
        client=client,
        model=model,
        on_memory_ops=on_memory_ops,
    )
    return coordinator


def mark_memory_foreground_active(
    *,
    store: ConversationMemoryStore,
    conversation_id: str,
) -> None:
    coordinator = _coordinators.get(store.db_path.resolve())
    if coordinator is not None:
        coordinator.foreground_started(conversation_id)


def mark_memory_foreground_idle(
    *,
    store: ConversationMemoryStore,
    conversation_id: str,
) -> None:
    coordinator = _coordinators.get(store.db_path.resolve())
    if coordinator is not None:
        coordinator.foreground_finished(conversation_id)


def global_foreground_started() -> None:
    """Process-wide answer-priority hook for a runtime gate busy transition."""
    global _global_foreground_depth
    _global_foreground_depth += 1
    for coordinator in list(_coordinators.values()):
        coordinator.global_foreground_started()


def global_foreground_finished() -> None:
    """Process-wide answer-priority hook for a runtime gate idle transition."""
    global _global_foreground_depth
    _global_foreground_depth = max(0, _global_foreground_depth - 1)
    for coordinator in list(_coordinators.values()):
        coordinator.global_foreground_finished()


async def shutdown_memory_fold_coordinators(*, timeout_seconds: float = 2.0) -> None:
    """Gracefully drain every coordinator owned by the current process."""
    global _global_foreground_depth
    coordinators = list(_coordinators.values())
    if coordinators:
        await asyncio.gather(
            *(
                coordinator.shutdown(timeout_seconds=timeout_seconds)
                for coordinator in coordinators
            ),
            return_exceptions=True,
        )
    _coordinators.clear()
    _global_foreground_depth = 0


def schedule_memory_fold(
    *,
    store: ConversationMemoryStore,
    conversation_id: str,
    client: SupportsChat | None,
    model: str,
    user_text: str,
    assistant_text: str,
    working_topic: str | None = None,
    working_filenames: list[str] | None = None,
    user_message_id: str | None = None,
    assistant_message_id: str | None = None,
    on_memory_ops: MemoryOpsCallback | None = None,
) -> None:
    """Persist a completed turn synchronously, then coalesce its async fold."""
    store.record_completed_turn(
        conversation_id,
        user_text=user_text,
        assistant_text=assistant_text,
        working_topic=working_topic,
        working_filenames=working_filenames,
        user_message_id=user_message_id,
        assistant_message_id=assistant_message_id,
    )
    if client is None:
        store.mark_fold_failed(conversation_id, "memory_fold_client_unavailable")
        return
    coordinator_key = store.db_path.resolve()
    existing_coordinator = _coordinators.get(coordinator_key)
    try:
        coordinator = start_memory_fold_coordinator(
            store=store,
            client=client,
            model=model,
            on_memory_ops=on_memory_ops,
            debounce_seconds=(
                existing_coordinator.debounce_seconds
                if existing_coordinator is not None
                else MEMORY_FOLD_COMPAT_DEBOUNCE_SECONDS
            ),
        )
    except RuntimeError:
        # No running loop: the durable job is intentionally left pending for
        # startup recovery instead of using a lossy local/model fallback.
        return
    coordinator.enqueue(conversation_id)


def _ensure_memory_tables(connection: Any) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS conversation_memory_turns (
            conversation_id TEXT NOT NULL,
            turn_seq INTEGER NOT NULL,
            user_message_id TEXT,
            assistant_message_id TEXT,
            user_text TEXT NOT NULL,
            assistant_text TEXT NOT NULL,
            working_topic TEXT,
            working_filenames_json TEXT NOT NULL DEFAULT '[]',
            completed_at TEXT NOT NULL,
            PRIMARY KEY (conversation_id, turn_seq),
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_turns_user_message
        ON conversation_memory_turns(user_message_id)
        WHERE user_message_id IS NOT NULL;

        CREATE INDEX IF NOT EXISTS idx_memory_turns_conversation_completed
        ON conversation_memory_turns(conversation_id, completed_at);

        CREATE TABLE IF NOT EXISTS conversation_memory_jobs (
            conversation_id TEXT PRIMARY KEY,
            dirty_through_seq INTEGER NOT NULL DEFAULT 0,
            summary_through_seq INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'pending',
            attempt_count INTEGER NOT NULL DEFAULT 0,
            next_attempt_at TEXT,
            last_error TEXT,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS conversation_memory_l3_outbox (
            conversation_id TEXT NOT NULL,
            source_turn_seq INTEGER NOT NULL,
            operations_json TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            attempt_count INTEGER NOT NULL DEFAULT 0,
            next_attempt_at TEXT,
            last_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            delivered_at TEXT,
            PRIMARY KEY (conversation_id, source_turn_seq),
            FOREIGN KEY (conversation_id)
                REFERENCES conversations(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_conversation_memory_l3_outbox_pending
        ON conversation_memory_l3_outbox(status, next_attempt_at, updated_at);
        """
    )


def _parse_fold_output(raw: str) -> FoldOutput:
    text = (raw or "").strip()
    candidate = text
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?", "", candidate, flags=re.IGNORECASE).strip()
        candidate = re.sub(r"```$", "", candidate).strip()
    try:
        parsed = json.loads(candidate)
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = None
    if isinstance(parsed, dict):
        summary = _clip_sentence_boundary(str(parsed.get("summary") or ""), SUMMARY_MAX_CHARS)
        raw_ops = parsed.get("memory_ops")
        operations = _validated_memory_ops(raw_ops)
        return FoldOutput(summary=summary, memory_ops=tuple(operations[:16]))
    # Old clients/tests may still return a plain-text summary. This is an output
    # format compatibility path, not a model or heuristic summary fallback.
    return FoldOutput(
        summary=_clip_sentence_boundary(text, SUMMARY_MAX_CHARS),
        memory_ops=(),
    )


def _validated_memory_ops(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    operations: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action") or "").strip().lower()
        scope = str(item.get("scope") or "").strip().lower()
        kind = str(item.get("kind") or "").strip().lower()
        key = str(item.get("key") or "").strip().lower()
        content = " ".join(str(item.get("content") or "").split())
        try:
            confidence = float(item.get("confidence"))
        except (TypeError, ValueError):
            continue
        contains_sensitive_value = bool(
            _SENSITIVE_MEMORY_RE.search(f"{key} {content}")
            or _SENSITIVE_MEMORY_VALUE_RE.search(content)
        )
        if (
            action not in _MEMORY_OP_ACTIONS
            or scope not in _MEMORY_OP_SCOPES
            or kind not in _MEMORY_OP_KINDS
            or not _MEMORY_KEY_RE.fullmatch(key)
            or not 0.0 <= confidence <= 1.0
            or (action == "upsert" and not content)
            or (action == "upsert" and contains_sensitive_value)
        ):
            continue
        operations.append(
            {
                "action": action,
                "scope": scope,
                "kind": kind,
                "key": key,
                # Forget operations identify a logical key and must never
                # persist any model-echoed credential value.
                "content": content if action == "upsert" else "",
                "confidence": confidence,
            }
        )
    return operations


def _pending_turns_prompt(turns: tuple[CompletedMemoryTurn, ...]) -> str:
    header = (
        "Completed turns not yet folded into the stable summary "
        "(treat these as newer context):"
    )
    remaining = max(0, PENDING_PROMPT_MAX_CHARS - len(header) - 2)
    selected_reversed: list[str] = []
    used = 0
    for turn in reversed(turns):
        block = _format_turn(turn)
        separator = 2 if selected_reversed else 0
        if used + separator + len(block) <= remaining:
            selected_reversed.append(block)
            used += separator + len(block)
            continue
        if not selected_reversed and remaining > 80:
            suffix = " [turn clipped to prompt budget; full text remains durable]"
            selected_reversed.append(
                _clip_sentence_boundary(block, max(40, remaining - len(suffix)))
                + suffix
            )
        break
    selected = list(reversed(selected_reversed))
    omitted = max(0, len(turns) - len(selected))
    lines = [header]
    if omitted:
        lines.append(
            f"[{omitted} older pending turn(s) omitted from this prompt budget; "
            "they remain durable and queued for folding.]"
        )
    lines.extend(selected)
    return "\n\n".join(lines)


def _fold_turn_batch(
    turns: list[CompletedMemoryTurn],
    *,
    max_chars: int,
) -> list[CompletedMemoryTurn]:
    """Take a chronological prefix so the committed cursor never skips a turn."""
    selected: list[CompletedMemoryTurn] = []
    used = 0
    for turn in turns:
        block = _format_turn(turn)
        separator = 2 if selected else 0
        if used + separator + len(block) <= max_chars:
            selected.append(turn)
            used += separator + len(block)
            continue
        if selected:
            break
        # A single pathological turn cannot be split by the cursor schema. Keep
        # sentence boundaries, mark the truncation, and advance only this turn.
        user_budget = max(1000, max_chars // 3)
        assistant_budget = max(1000, max_chars - user_budget - 300)
        selected.append(
            CompletedMemoryTurn(
                conversation_id=turn.conversation_id,
                turn_seq=turn.turn_seq,
                user_message_id=turn.user_message_id,
                assistant_message_id=turn.assistant_message_id,
                user_text=_clip_sentence_boundary(turn.user_text, user_budget),
                assistant_text=(
                    _clip_sentence_boundary(turn.assistant_text, assistant_budget)
                    + " [content clipped to fold budget]"
                ),
                working_topic=turn.working_topic,
                working_filenames=turn.working_filenames,
                completed_at=turn.completed_at,
            )
        )
        break
    return selected


def _format_turn(turn: CompletedMemoryTurn) -> str:
    return (
        f"turn {turn.turn_seq} user: {turn.user_text.strip()}\n"
        f"turn {turn.turn_seq} assistant: {turn.assistant_text.strip()}"
    )


def _clip_sentence_boundary(text: str, limit: int) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    window = compact[:limit]
    sentence_ends = [
        match.end()
        for match in re.finditer(r"[.!?。！？](?:[\"'’”\)\]]*)", window)
    ]
    usable = [end for end in sentence_ends if end >= max(32, limit // 3)]
    if usable:
        return window[: usable[-1]].rstrip() + " …"
    word_boundary = window.rfind(" ")
    if word_boundary >= max(16, limit // 3):
        window = window[:word_boundary]
    return window.rstrip(" ,;:-") + "…"


def _clip_error(error: str, limit: int = 800) -> str:
    compact = " ".join((error or "unknown memory fold error").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _parse_json_object(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _memory_from_row(row: Any) -> ConversationMemory:
    summary = _string_or_none(row["summary"])
    metadata = _parse_json_object(row["metadata_json"])
    memory_meta = metadata.get("memory")
    beats: list[TurnBeat] = []
    updated_at = None
    revision = 0
    summary_revision = 0
    if isinstance(memory_meta, dict):
        updated_at = _string_or_none(memory_meta.get("updated_at"))
        revision = _nonnegative_int(memory_meta.get("revision"))
        summary_revision = _nonnegative_int(memory_meta.get("summary_revision"))
        for item in memory_meta.get("recent_beats") or []:
            if isinstance(item, dict):
                beat = TurnBeat.from_dict(item)
                if beat is not None:
                    beats.append(beat)
    if beats:
        revision = max(revision, *(beat.revision for beat in beats))
    return ConversationMemory(
        summary=summary,
        recent_beats=tuple(beats[-MAX_RECENT_BEATS:]),
        updated_at=updated_at,
        revision=revision,
        summary_revision=min(summary_revision, revision) if revision else summary_revision,
    )


def _completed_turn_from_row(row: Any) -> CompletedMemoryTurn:
    raw_filenames = _parse_json_array(row["working_filenames_json"])
    return CompletedMemoryTurn(
        conversation_id=str(row["conversation_id"]),
        turn_seq=_nonnegative_int(row["turn_seq"]),
        user_message_id=_string_or_none(row["user_message_id"]),
        assistant_message_id=_string_or_none(row["assistant_message_id"]),
        user_text=str(row["user_text"] or ""),
        assistant_text=str(row["assistant_text"] or ""),
        working_topic=_string_or_none(row["working_topic"]),
        working_filenames=tuple(_dedupe_strings(raw_filenames)),
        completed_at=str(row["completed_at"] or utc_now()),
    )


def _job_from_row(row: Any) -> ConversationMemoryJob:
    return ConversationMemoryJob(
        conversation_id=str(row["conversation_id"]),
        dirty_through_seq=_nonnegative_int(row["dirty_through_seq"]),
        summary_through_seq=_nonnegative_int(row["summary_through_seq"]),
        status=str(row["status"] or "pending"),
        attempt_count=_nonnegative_int(row["attempt_count"]),
        next_attempt_at=_string_or_none(row["next_attempt_at"]),
        last_error=_string_or_none(row["last_error"]),
        updated_at=str(row["updated_at"] or utc_now()),
    )


def _pending_memory_operations_from_row(row: Any) -> PendingMemoryOperations:
    return PendingMemoryOperations(
        conversation_id=str(row["conversation_id"]),
        source_turn_seq=_nonnegative_int(row["source_turn_seq"]),
        operations=tuple(_parse_json_array_of_objects(row["operations_json"])),
        attempt_count=_nonnegative_int(row["attempt_count"]),
        next_attempt_at=_string_or_none(row["next_attempt_at"]),
        last_error=_string_or_none(row["last_error"]),
        created_at=str(row["created_at"] or utc_now()),
        updated_at=str(row["updated_at"] or utc_now()),
    )


def _memory_metadata(
    *,
    beats: list[TurnBeat],
    updated_at: str,
    revision: int,
    summary_revision: int,
) -> dict[str, Any]:
    return {
        "recent_beats": [beat.to_dict() for beat in beats[-MAX_RECENT_BEATS:]],
        "updated_at": updated_at,
        "revision": max(0, revision),
        "summary_revision": max(0, summary_revision),
    }


def _resolve_assistant_message_id(
    connection: Any,
    *,
    conversation_id: str,
    assistant_text: str,
    user_created_at: str,
) -> str | None:
    rows = connection.execute(
        """
        SELECT id FROM messages
        WHERE conversation_id = ? AND role = 'assistant' AND content = ?
          AND created_at >= ?
        ORDER BY created_at ASC
        LIMIT 2
        """,
        (conversation_id, assistant_text, user_created_at),
    ).fetchall()
    return str(rows[0]["id"]) if len(rows) == 1 else None


def _parse_json_array(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return [str(item) for item in value] if isinstance(value, list) else []


def _parse_json_array_of_objects(raw: str | None) -> list[dict[str, Any]]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _dedupe_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _seconds_until(timestamp: str | None) -> float:
    if not timestamp:
        return 0.0
    try:
        target = datetime.fromisoformat(timestamp)
        if target.tzinfo is None:
            target = target.replace(tzinfo=UTC)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, (target - datetime.now(UTC)).total_seconds())


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
