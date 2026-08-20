"""Deterministic L0 history retrieval and structured L3 memory.

This module deliberately contains no model or network calls.  L0 is recovered
from the immutable ``messages`` log; L3 is an explicit, versioned store whose
rows keep their provenance and validity interval.

Database assumptions are intentionally small:

* ``messages`` has ``id``, ``conversation_id``, ``role``, ``content`` and
  ``created_at`` columns.
* ``message_search`` is an optional FTS5 table with ``message_id``,
  ``conversation_id`` and ``content`` columns.  Search remains correct (but
  less efficient) when it is absent or temporarily stale.
* ``memory_items`` follows the schema documented by :class:`MemoryItem`.

Schema creation and FTS synchronization belong to the database migration/write
path, not this service.  Keeping those responsibilities separate makes it
possible to test degraded search when FTS is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Any, Sequence
import unicodedata
import uuid

from app.db.sqlite import connect
from app.services.chat_history import utc_now


MEMORY_KINDS = frozenset({"semantic", "episodic", "procedural"})

DEFAULT_HISTORY_PROMPT_CHARS = 3_200
DEFAULT_MEMORY_PROMPT_CHARS = 2_400
MAX_HISTORY_RESULTS = 20
MAX_MEMORY_RESULTS = 50

_WORD_RE = re.compile(r"[^\W_]+(?:[’'][^\W_]+)*", re.UNICODE)
_SENTENCE_END_RE = re.compile(r"[.!?。！？](?:[\"'’”)]*)")
_CLAUSE_END_RE = re.compile(r"[,;:，；：](?:[\"'’”)]*)")

# Removing very common glue words keeps an OR-based FTS query useful and stops
# token-overlap fallback from matching every chat merely because it contains
# words such as "the" or "và".  Domain terms and short identifiers are kept.
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "about",
        "at",
        "be",
        "chat",
        "conversation",
        "did",
        "discussed",
        "does",
        "earlier",
        "for",
        "from",
        "last",
        "how",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "previous",
        "previously",
        "said",
        "session",
        "that",
        "the",
        "this",
        "time",
        "to",
        "talked",
        "user",
        "was",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
        "you",
        "ban",
        "cai",
        "chat",
        "chung",
        "chuyen",
        "cuoc",
        "cua",
        "da",
        "doan",
        "do",
        "gi",
        "hom",
        "khi",
        "la",
        "lan",
        "luan",
        "luc",
        "minh",
        "mot",
        "nay",
        "nhung",
        "noi",
        "o",
        "qua",
        "ta",
        "thao",
        "toi",
        "tro",
        "truoc",
        "tung",
        "trong",
        "va",
        "ve",
    }
)

_CROSS_THREAD_HISTORY_CUES = (
    re.compile(
        r"\b(hom qua|hom truoc|lan truoc|cuoc tro chuyen truoc|"
        r"doan chat truoc|chat truoc|"
        r"(?:chung ta|chung minh|minh) (?:da |tung )?"
        r"(?:noi|thao luan|ban ve))\b"
    ),
    re.compile(
        r"\b(yesterday|last time|previous (?:conversation|chat|session)|"
        r"earlier (?:conversation|chat|session)|we (?:said|discussed|talked about)|"
        r"we (?:have |previously )?(?:said|discussed|talked about))\b"
    ),
)

_MEMORY_OPERATION_ACTIONS = frozenset({"upsert", "forget"})
_MEMORY_OPERATION_SCOPES = frozenset({"user", "conversation"})
_MEMORY_OPERATION_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
_SENSITIVE_MEMORY_RE = re.compile(
    r"\b(password|passcode|api[_ -]?key|access[_ -]?token|refresh[_ -]?token|"
    r"bearer|private[_ -]?key|secret|mật khẩu|mat khau|khóa bí mật|khoa bi mat)\b",
    re.IGNORECASE,
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
    re.IGNORECASE,
)


@dataclass(frozen=True)
class HistoricalMessage:
    """A source message included in a recovered conversational episode."""

    id: str
    conversation_id: str
    role: str
    content: str
    created_at: str

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class HistoricalEpisode:
    """A complete user/assistant exchange surrounding one or more search hits."""

    conversation_id: str
    messages: tuple[HistoricalMessage, ...]
    matched_message_ids: tuple[str, ...]
    score: float
    started_at: str
    ended_at: str

    @property
    def message_ids(self) -> tuple[str, ...]:
        return tuple(message.id for message in self.messages)

    @property
    def user_text(self) -> str:
        return "\n".join(
            message.content for message in self.messages if message.role == "user"
        )

    @property
    def assistant_text(self) -> str:
        return "\n".join(
            message.content for message in self.messages if message.role == "assistant"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "messages": [message.to_dict() for message in self.messages],
            "matched_message_ids": list(self.matched_message_ids),
            "score": self.score,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


def requests_cross_thread_history(query: str) -> bool:
    """Detect an explicit cue that permits recall beyond the current thread.

    The caller remains in control of scope: pass the current conversation ID
    to :meth:`HistoricalConversationSearch.search` normally, and pass ``None``
    only when this predicate (or an explicit UI intent) authorizes cross-thread
    history.  This avoids silently leaking an unrelated conversation into an
    ordinary same-thread follow-up.
    """

    normalized = _normalize_token(str(query or ""))
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized).strip()
    return bool(normalized) and any(
        pattern.search(normalized) for pattern in _CROSS_THREAD_HISTORY_CUES
    )


@dataclass(frozen=True)
class HistoricalConversationSearch:
    """Search older L0 messages and expand hits back to whole episodes.

    FTS5 is an optimization, not a source of truth.  Every FTS query is built
    only from quoted word tokens, and an ordinary SQL/token-overlap path is
    always available when FTS is missing, stale, or rejects a query.
    """

    db_path: Path

    def search_for_context(
        self,
        query: str,
        *,
        current_conversation_id: str,
        exclude_message_ids: Sequence[str] = (),
        limit: int = 4,
        candidate_limit: int = 80,
        allow_cross_thread: bool | None = None,
    ) -> list[HistoricalEpisode]:
        """Safe prompt-facing search: same thread unless history is explicit.

        ``allow_cross_thread`` lets an already-resolved router/UI intent
        override the lexical cue detector.  Leaving it as ``None`` applies the
        conservative built-in detector.
        """

        cross_thread = (
            requests_cross_thread_history(query)
            if allow_cross_thread is None
            else bool(allow_cross_thread)
        )
        return self.search(
            query,
            conversation_id=None if cross_thread else current_conversation_id,
            exclude_message_ids=exclude_message_ids,
            limit=limit,
            candidate_limit=candidate_limit,
        )

    def prompt_block_for_context(
        self,
        query: str,
        *,
        current_conversation_id: str,
        exclude_message_ids: Sequence[str] = (),
        limit: int = 4,
        max_chars: int = DEFAULT_HISTORY_PROMPT_CHARS,
        allow_cross_thread: bool | None = None,
    ) -> str:
        episodes = self.search_for_context(
            query,
            current_conversation_id=current_conversation_id,
            exclude_message_ids=exclude_message_ids,
            limit=limit,
            allow_cross_thread=allow_cross_thread,
        )
        return format_historical_episodes(episodes, max_chars=max_chars)

    def search(
        self,
        query: str,
        *,
        conversation_id: str | None = None,
        exclude_message_ids: Sequence[str] = (),
        limit: int = 4,
        candidate_limit: int = 80,
    ) -> list[HistoricalEpisode]:
        result_limit = max(0, min(int(limit), MAX_HISTORY_RESULTS))
        if result_limit == 0:
            return []
        raw_tokens = _raw_query_tokens(query)
        normalized_tokens = _normalized_query_tokens(query)
        if not raw_tokens and not normalized_tokens:
            return []

        excluded = {str(message_id) for message_id in exclude_message_ids if message_id}
        pool_limit = max(result_limit * 4, min(max(1, int(candidate_limit)), 500))

        with connect(self.db_path) as connection:
            candidate_scores = self._candidate_scores(
                connection,
                query=query,
                raw_tokens=raw_tokens,
                normalized_tokens=normalized_tokens,
                conversation_id=conversation_id,
                excluded=excluded,
                limit=pool_limit,
            )
            if not candidate_scores:
                return []
            episodes = self._expand_candidates(
                connection,
                candidate_scores=candidate_scores,
                query_tokens=normalized_tokens,
                excluded=excluded,
            )

        episodes.sort(
            key=lambda episode: (
                episode.score,
                _timestamp_sort_value(episode.ended_at),
                episode.conversation_id,
                episode.message_ids,
            ),
            reverse=True,
        )
        return episodes[:result_limit]

    def prompt_block(
        self,
        query: str,
        *,
        conversation_id: str | None = None,
        exclude_message_ids: Sequence[str] = (),
        limit: int = 4,
        max_chars: int = DEFAULT_HISTORY_PROMPT_CHARS,
    ) -> str:
        episodes = self.search(
            query,
            conversation_id=conversation_id,
            exclude_message_ids=exclude_message_ids,
            limit=limit,
        )
        return format_historical_episodes(episodes, max_chars=max_chars)

    def _candidate_scores(
        self,
        connection: sqlite3.Connection,
        *,
        query: str,
        raw_tokens: tuple[str, ...],
        normalized_tokens: tuple[str, ...],
        conversation_id: str | None,
        excluded: set[str],
        limit: int,
    ) -> dict[str, float]:
        scores: dict[str, float] = {}

        if _table_exists(connection, "message_search"):
            fts_query = _safe_fts_query(raw_tokens)
            if fts_query:
                sql = (
                    "SELECT message_id, bm25(message_search) AS rank "
                    "FROM message_search WHERE message_search MATCH ?"
                )
                params: list[Any] = [fts_query]
                if conversation_id is not None:
                    sql += " AND conversation_id = ?"
                    params.append(conversation_id)
                sql += " ORDER BY rank ASC LIMIT ?"
                params.append(limit)
                try:
                    rows = connection.execute(sql, params).fetchall()
                except sqlite3.OperationalError:
                    # An optional/stale FTS index must never break chat recall.
                    rows = []
                for position, row in enumerate(rows):
                    message_id = str(row["message_id"] or "")
                    if not message_id or message_id in excluded:
                        continue
                    # Rank magnitude varies with corpus size; ordering is the
                    # useful stable signal.  Lexical overlap is added below.
                    scores[message_id] = max(
                        scores.get(message_id, 0.0),
                        2.0 + 1.0 / (position + 1),
                    )

        # LIKE cheaply finds direct substrings and remains useful when the FTS
        # table has not yet been backfilled.  Tokens are parameters, never SQL.
        like_tokens = raw_tokens[:8]
        if like_tokens:
            clauses = ["LOWER(content) LIKE ? ESCAPE '\\'" for _ in like_tokens]
            sql = (
                "SELECT id, content FROM messages WHERE role IN ('user', 'assistant') "
                f"AND ({' OR '.join(clauses)})"
            )
            params = [f"%{_escape_like(token.casefold())}%" for token in like_tokens]
            if conversation_id is not None:
                sql += " AND conversation_id = ?"
                params.append(conversation_id)
            sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
            params.append(limit)
            for row in connection.execute(sql, params).fetchall():
                message_id = str(row["id"])
                if message_id in excluded:
                    continue
                overlap = _token_overlap_score(normalized_tokens, str(row["content"] or ""))
                scores[message_id] = max(scores.get(message_id, 0.0), 1.0 + overlap)

        # Accent-insensitive token overlap is the final deterministic fallback.
        # Bound the scan so a very large L0 log cannot monopolize the answer path.
        sql = (
            "SELECT id, content FROM messages "
            "WHERE role IN ('user', 'assistant')"
        )
        params = []
        if conversation_id is not None:
            sql += " AND conversation_id = ?"
            params.append(conversation_id)
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(min(max(limit * 8, 200), 1_000))
        for row in connection.execute(sql, params).fetchall():
            message_id = str(row["id"])
            if message_id in excluded:
                continue
            overlap = _token_overlap_score(normalized_tokens, str(row["content"] or ""))
            if overlap <= 0:
                continue
            scores[message_id] = max(scores.get(message_id, 0.0), overlap)

        # Stale FTS rows are filtered during episode expansion.  This bound also
        # prevents a broad OR query from expanding hundreds of conversations.
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return dict(ordered[:limit])

    def _expand_candidates(
        self,
        connection: sqlite3.Connection,
        *,
        candidate_scores: dict[str, float],
        query_tokens: tuple[str, ...],
        excluded: set[str],
    ) -> list[HistoricalEpisode]:
        candidate_ids = set(candidate_scores)
        placeholders = ",".join("?" for _ in candidate_ids)
        candidate_rows = connection.execute(
            f"SELECT id, conversation_id FROM messages WHERE id IN ({placeholders})",
            tuple(sorted(candidate_ids)),
        ).fetchall()
        by_conversation: dict[str, set[str]] = {}
        for row in candidate_rows:
            by_conversation.setdefault(str(row["conversation_id"]), set()).add(str(row["id"]))

        expanded: list[HistoricalEpisode] = []
        for conversation_id in sorted(by_conversation):
            rows = connection.execute(
                """
                SELECT id, conversation_id, role, content, created_at
                FROM messages
                WHERE conversation_id = ? AND role IN ('user', 'assistant')
                ORDER BY created_at ASC, id ASC
                """,
                (conversation_id,),
            ).fetchall()
            messages = tuple(
                HistoricalMessage(
                    id=str(row["id"]),
                    conversation_id=str(row["conversation_id"]),
                    role=str(row["role"]),
                    content=str(row["content"] or ""),
                    created_at=str(row["created_at"] or ""),
                )
                for row in rows
            )
            for episode_messages in _partition_episodes(messages):
                episode_ids = {message.id for message in episode_messages}
                matched_ids = tuple(sorted(episode_ids & by_conversation[conversation_id]))
                if not matched_ids or episode_ids & excluded:
                    # Avoid duplicating any part of the raw-recent prompt.  An
                    # episode is atomic: returning only its old half would lose
                    # exactly the user/assistant relationship L0 should recover.
                    continue
                joined = " ".join(message.content for message in episode_messages)
                lexical_overlap = _token_overlap_score(query_tokens, joined)
                if lexical_overlap <= 0:
                    # An FTS row can be temporarily stale after an interrupted
                    # migration or external write.  Candidate IDs are only an
                    # optimization; canonical message text must still match.
                    continue
                score = max(candidate_scores[message_id] for message_id in matched_ids)
                score += lexical_overlap
                expanded.append(
                    HistoricalEpisode(
                        conversation_id=conversation_id,
                        messages=episode_messages,
                        matched_message_ids=matched_ids,
                        score=round(score, 6),
                        started_at=episode_messages[0].created_at,
                        ended_at=episode_messages[-1].created_at,
                    )
                )
        return expanded


@dataclass(frozen=True)
class MemoryItem:
    """One immutable-content version of an L3 memory entry."""

    id: str
    scope: str
    conversation_id: str | None
    kind: str
    memory_key: str
    content: str
    status: str
    confidence: float
    valid_from: str | None
    valid_to: str | None
    source_conversation_id: str | None
    source_turn_seq: int | None
    supersedes_id: str | None
    created_at: str
    updated_at: str
    metadata: dict[str, Any]

    @property
    def is_current(self) -> bool:
        return self.status == "active" and self.valid_to is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scope": self.scope,
            "conversation_id": self.conversation_id,
            "kind": self.kind,
            "memory_key": self.memory_key,
            "content": self.content,
            "status": self.status,
            "confidence": self.confidence,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "source_conversation_id": self.source_conversation_id,
            "source_turn_seq": self.source_turn_seq,
            "supersedes_id": self.supersedes_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class MemoryItemStore:
    """Versioned semantic, episodic, and procedural L3 memory store."""

    db_path: Path

    def get(self, item_id: str) -> MemoryItem | None:
        with connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT * FROM memory_items WHERE id = ?",
                (item_id,),
            ).fetchone()
        return _memory_item_from_row(row) if row is not None else None

    def upsert(
        self,
        *,
        scope: str,
        kind: str,
        memory_key: str,
        content: str,
        conversation_id: str | None = None,
        confidence: float = 1.0,
        valid_from: str | None = None,
        source_conversation_id: str | None = None,
        source_turn_seq: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryItem:
        """Insert a new active version and close the previous active version.

        Even identical content is a new observation/version.  That preserves
        provenance and avoids mutating history behind an audit trail.
        """

        clean_scope = _validate_scope(scope)
        clean_kind = _validate_kind(kind)
        clean_key = _required_compact(memory_key, "memory_key")
        clean_content = _required_compact(content, "content")
        clean_conversation_id = _optional_compact(conversation_id)
        _validate_scope_conversation(clean_scope, clean_conversation_id)
        clean_source_conversation_id = _optional_compact(source_conversation_id)
        clean_turn_seq = _optional_nonnegative_int(source_turn_seq, "source_turn_seq")
        clean_confidence = _validate_confidence(confidence)
        clean_metadata = _json_object(metadata)
        now = utc_now()
        effective_at = _normalize_timestamp(valid_from, "valid_from") or now

        with connect(self.db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = _upsert_memory_item_in_connection(
                connection,
                scope=clean_scope,
                kind=clean_kind,
                memory_key=clean_key,
                content=clean_content,
                conversation_id=clean_conversation_id,
                confidence=clean_confidence,
                valid_from=effective_at,
                source_conversation_id=clean_source_conversation_id,
                source_turn_seq=clean_turn_seq,
                metadata=clean_metadata,
                now=now,
            )

        if row is None:  # pragma: no cover - protects against external DB triggers
            raise RuntimeError("memory item insert did not persist")
        return _memory_item_from_row(row)

    def forget(self, item_id: str, *, valid_to: str | None = None) -> MemoryItem | None:
        """Close one active memory version; repeated calls are idempotent."""

        forgotten_at = _normalize_timestamp(valid_to, "valid_to") or utc_now()
        now = utc_now()
        with connect(self.db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM memory_items WHERE id = ?",
                (item_id,),
            ).fetchone()
            if row is None:
                return None
            if str(row["status"]) == "active" and row["valid_to"] is None:
                connection.execute(
                    """
                    UPDATE memory_items
                    SET status = 'forgotten', valid_to = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (forgotten_at, now, item_id),
                )
                row = connection.execute(
                    "SELECT * FROM memory_items WHERE id = ?",
                    (item_id,),
                ).fetchone()
        return _memory_item_from_row(row)

    def forget_key(
        self,
        *,
        scope: str,
        kind: str,
        memory_key: str,
        conversation_id: str | None = None,
        valid_to: str | None = None,
    ) -> MemoryItem | None:
        """Forget the active version identified by its logical key."""

        clean_scope = _validate_scope(scope)
        clean_kind = _validate_kind(kind)
        clean_key = _required_compact(memory_key, "memory_key")
        clean_conversation_id = _optional_compact(conversation_id)
        _validate_scope_conversation(clean_scope, clean_conversation_id)
        with connect(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT id FROM memory_items
                WHERE scope = ? AND kind = ? AND memory_key = ? COLLATE NOCASE
                  AND status = 'active' AND valid_to IS NULL
                  AND (
                    (conversation_id IS NULL AND ? IS NULL)
                    OR conversation_id = ?
                  )
                ORDER BY updated_at DESC, id DESC
                LIMIT 1
                """,
                (
                    clean_scope,
                    clean_kind,
                    clean_key,
                    clean_conversation_id,
                    clean_conversation_id,
                ),
            ).fetchone()
        if row is None:
            return None
        return self.forget(str(row["id"]), valid_to=valid_to)

    def relevant(
        self,
        query: str,
        *,
        scopes: Sequence[str] | None = None,
        conversation_id: str | None = None,
        limit: int = 8,
        min_confidence: float = 0.0,
        as_of: str | None = None,
    ) -> list[MemoryItem]:
        """Return applicable procedures plus query-relevant facts/episodes.

        A supplied conversation includes global (``conversation_id IS NULL``)
        and matching thread-local items.  With no conversation supplied, only
        global items are eligible, preventing accidental cross-thread leakage.
        Superseded versions can be returned only for an ``as_of`` instant that
        falls inside their former validity interval.
        """

        result_limit = max(0, min(int(limit), MAX_MEMORY_RESULTS))
        if result_limit == 0:
            return []
        clean_confidence = _validate_confidence(min_confidence)
        query_tokens = _normalized_query_tokens(query)
        instant = _normalize_timestamp(as_of, "as_of") or utc_now()

        clauses = [
            "status IN ('active', 'superseded', 'forgotten')",
            "confidence >= ?",
            "(valid_from IS NULL OR valid_from <= ?)",
            "(valid_to IS NULL OR valid_to > ?)",
        ]
        params: list[Any] = [clean_confidence, instant, instant]
        if scopes is not None:
            clean_scopes = tuple(
                dict.fromkeys(_validate_scope(scope) for scope in scopes)
            )
            if not clean_scopes:
                return []
            placeholders = ",".join("?" for _ in clean_scopes)
            clauses.append(f"scope IN ({placeholders})")
            params.extend(clean_scopes)
        clean_conversation_id = _optional_compact(conversation_id)
        if clean_conversation_id is None:
            clauses.append("conversation_id IS NULL")
        else:
            clauses.append("(conversation_id IS NULL OR conversation_id = ?)")
            params.append(clean_conversation_id)

        candidate_limit = min(max(result_limit * 20, 100), 1_000)
        params.append(candidate_limit)
        with connect(self.db_path) as connection:
            rows = connection.execute(
                "SELECT * FROM memory_items WHERE "
                + " AND ".join(clauses)
                + " ORDER BY CASE kind WHEN 'procedural' THEN 0 ELSE 1 END, "
                "updated_at DESC, id DESC LIMIT ?",
                params,
            ).fetchall()

        ranked: list[tuple[MemoryItem, float]] = []
        for row in rows:
            item = _memory_item_from_row(row)
            if _contains_sensitive_memory_value(item.memory_key, item.content):
                # Defense in depth for imported/pre-migration rows. Automatic
                # extraction rejects these on write, but prompt construction
                # must never rely on every historical writer having done so.
                continue
            relevance = _memory_relevance(item, query, query_tokens)
            if item.kind != "procedural" and relevance <= 0:
                continue
            ranked.append((item, relevance))
        ranked.sort(
            key=lambda pair: (
                0 if pair[0].kind == "procedural" else 1,
                -pair[1],
                -pair[0].confidence,
                -_timestamp_sort_value(pair[0].updated_at),
                pair[0].id,
            )
        )
        return [item for item, _ in ranked[:result_limit]]

    # Descriptive alias for call sites that prefer an explicit verb phrase.
    retrieve_relevant = relevant

    def prompt_block(
        self,
        query: str,
        *,
        scopes: Sequence[str] | None = None,
        conversation_id: str | None = None,
        limit: int = 8,
        max_chars: int = DEFAULT_MEMORY_PROMPT_CHARS,
        min_confidence: float = 0.0,
        as_of: str | None = None,
    ) -> str:
        items = self.relevant(
            query,
            scopes=scopes,
            conversation_id=conversation_id,
            limit=limit,
            min_confidence=min_confidence,
            as_of=as_of,
        )
        return format_memory_items(items, max_chars=max_chars)

    def apply_operations(
        self,
        operations: Sequence[dict[str, Any]],
        *,
        source_conversation_id: str,
        source_turn_seq: int,
    ) -> list[MemoryItem]:
        """Apply validated L2 extraction operations idempotently.

        The coordinator may retry or restart around this boundary. An identical
        operation from the same source cursor therefore returns its existing
        version instead of creating another superseding row.
        """

        source_id = _required_compact(source_conversation_id, "source_conversation_id")
        source_seq = _optional_nonnegative_int(source_turn_seq, "source_turn_seq")
        if source_seq is None or source_seq <= 0:
            raise ValueError("source_turn_seq must be a positive integer")

        applied: list[MemoryItem] = []
        for raw in list(operations)[:16]:
            normalized = _normalize_memory_operation(raw)
            if normalized is None:
                continue
            scope = normalized["scope"]
            kind = normalized["kind"]
            key = normalized["key"]
            conversation_id = source_id if scope == "conversation" else None
            action = normalized["action"]
            content = (
                sentence_safe_clip(normalized["content"], 1_600)
                if action == "upsert"
                else ""
            )
            fingerprint = _memory_operation_fingerprint(
                action=action,
                scope=scope,
                kind=kind,
                key=key,
                content=content,
            )
            now = utc_now()
            with connect(self.db_path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                receipt = connection.execute(
                    """
                    SELECT result_item_id
                    FROM memory_operation_receipts
                    WHERE source_conversation_id = ? AND source_turn_seq = ?
                      AND operation_fingerprint = ?
                    """,
                    (
                        source_id,
                        source_seq,
                        fingerprint,
                    ),
                ).fetchone()
                if receipt is not None:
                    result_item_id = _optional_compact(receipt["result_item_id"])
                    row = (
                        connection.execute(
                            "SELECT * FROM memory_items WHERE id = ?",
                            (result_item_id,),
                        ).fetchone()
                        if result_item_id is not None
                        else None
                    )
                elif action == "forget":
                    row = _forget_memory_key_in_connection(
                        connection,
                        scope=scope,
                        kind=kind,
                        memory_key=key,
                        conversation_id=conversation_id,
                        valid_to=now,
                        now=now,
                    )
                    connection.execute(
                        """
                        INSERT INTO memory_operation_receipts (
                            source_conversation_id, source_turn_seq,
                            operation_fingerprint, action, result_item_id,
                            created_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            source_id,
                            source_seq,
                            fingerprint,
                            action,
                            str(row["id"]) if row is not None else None,
                            now,
                        ),
                    )
                else:
                    row = _upsert_memory_item_in_connection(
                        connection,
                        scope=scope,
                        kind=kind,
                        memory_key=key,
                        content=content,
                        conversation_id=conversation_id,
                        confidence=normalized["confidence"],
                        valid_from=now,
                        source_conversation_id=source_id,
                        source_turn_seq=source_seq,
                        metadata={"extracted_by": "l2_consolidation"},
                        now=now,
                    )
                    connection.execute(
                        """
                        INSERT INTO memory_operation_receipts (
                            source_conversation_id, source_turn_seq,
                            operation_fingerprint, action, result_item_id,
                            created_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            source_id,
                            source_seq,
                            fingerprint,
                            action,
                            str(row["id"]),
                            now,
                        ),
                    )
            if row is not None:
                applied.append(_memory_item_from_row(row))
        return applied


def _upsert_memory_item_in_connection(
    connection: sqlite3.Connection,
    *,
    scope: str,
    kind: str,
    memory_key: str,
    content: str,
    conversation_id: str | None,
    confidence: float,
    valid_from: str,
    source_conversation_id: str | None,
    source_turn_seq: int | None,
    metadata: dict[str, Any],
    now: str,
) -> sqlite3.Row:
    previous_row = connection.execute(
        """
        SELECT * FROM memory_items
        WHERE scope = ? AND kind = ? AND memory_key = ? COLLATE NOCASE
          AND status = 'active' AND valid_to IS NULL
          AND (
            (conversation_id IS NULL AND ? IS NULL)
            OR conversation_id = ?
          )
        ORDER BY updated_at DESC, id DESC
        LIMIT 1
        """,
        (
            scope,
            kind,
            memory_key,
            conversation_id,
            conversation_id,
        ),
    ).fetchone()
    supersedes_id = str(previous_row["id"]) if previous_row is not None else None
    if previous_row is not None:
        connection.execute(
            """
            UPDATE memory_items
            SET status = 'superseded', valid_to = ?, updated_at = ?
            WHERE id = ? AND status = 'active' AND valid_to IS NULL
            """,
            (valid_from, now, supersedes_id),
        )

    item_id = str(uuid.uuid4())
    connection.execute(
        """
        INSERT INTO memory_items (
            id, scope, conversation_id, kind, memory_key, content,
            status, confidence, valid_from, valid_to,
            source_conversation_id, source_turn_seq, supersedes_id,
            created_at, updated_at, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?, NULL, ?, ?, ?, ?, ?, ?)
        """,
        (
            item_id,
            scope,
            conversation_id,
            kind,
            memory_key,
            content,
            confidence,
            valid_from,
            source_conversation_id,
            source_turn_seq,
            supersedes_id,
            now,
            now,
            json.dumps(metadata, ensure_ascii=False, sort_keys=True),
        ),
    )
    row = connection.execute(
        "SELECT * FROM memory_items WHERE id = ?",
        (item_id,),
    ).fetchone()
    if row is None:  # pragma: no cover - protects against external DB triggers
        raise RuntimeError("memory item insert did not persist")
    return row


def _forget_memory_key_in_connection(
    connection: sqlite3.Connection,
    *,
    scope: str,
    kind: str,
    memory_key: str,
    conversation_id: str | None,
    valid_to: str,
    now: str,
) -> sqlite3.Row | None:
    row = connection.execute(
        """
        SELECT * FROM memory_items
        WHERE scope = ? AND kind = ? AND memory_key = ? COLLATE NOCASE
          AND status = 'active' AND valid_to IS NULL
          AND (
            (conversation_id IS NULL AND ? IS NULL)
            OR conversation_id = ?
          )
        ORDER BY updated_at DESC, id DESC
        LIMIT 1
        """,
        (
            scope,
            kind,
            memory_key,
            conversation_id,
            conversation_id,
        ),
    ).fetchone()
    if row is None:
        return None
    item_id = str(row["id"])
    connection.execute(
        """
        UPDATE memory_items
        SET status = 'forgotten', valid_to = ?, updated_at = ?
        WHERE id = ? AND status = 'active' AND valid_to IS NULL
        """,
        (valid_to, now, item_id),
    )
    return connection.execute(
        "SELECT * FROM memory_items WHERE id = ?",
        (item_id,),
    ).fetchone()


def _memory_operation_fingerprint(
    *,
    action: str,
    scope: str,
    kind: str,
    key: str,
    content: str,
) -> str:
    payload = {
        "action": action,
        "scope": scope,
        "kind": kind,
        "key": key.casefold(),
        # Confidence is deliberately omitted: a retry of the same source fact
        # with slightly different model confidence is still the same operation.
        "content": content if action == "upsert" else "",
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sentence_safe_clip(text: str, max_chars: int) -> str:
    """Compact and bound text, preferring a complete sentence or clause.

    A single sentence can itself exceed the budget.  In that unavoidable case
    the fallback stops on a word boundary and visibly marks the omission rather
    than slicing a Unicode word in half.
    """

    limit = max(0, int(max_chars))
    if limit == 0:
        return ""
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    if limit == 1:
        return "…"

    suffix = " …"
    body_limit = max(1, limit - len(suffix))
    prefix = compact[:body_limit]
    minimum_boundary = min(24, max(1, body_limit // 3))

    sentence_ends = [
        match.end()
        for match in _SENTENCE_END_RE.finditer(prefix)
        if match.end() >= minimum_boundary
    ]
    if sentence_ends:
        body = prefix[: sentence_ends[-1]].rstrip()
        return (body + suffix)[:limit]

    clause_ends = [
        match.end()
        for match in _CLAUSE_END_RE.finditer(prefix)
        if match.end() >= minimum_boundary
    ]
    if clause_ends:
        body = prefix[: clause_ends[-1]].rstrip()
        return (body + suffix)[:limit]

    word_boundary = prefix.rfind(" ")
    if word_boundary >= minimum_boundary:
        prefix = prefix[:word_boundary]
    return (prefix.rstrip() + suffix)[:limit]


def format_historical_episodes(
    episodes: Sequence[HistoricalEpisode],
    *,
    max_chars: int = DEFAULT_HISTORY_PROMPT_CHARS,
) -> str:
    """Render temporal L0 evidence without exceeding the prompt budget."""

    if not episodes or max_chars <= 0:
        return ""
    header = (
        "Relevant older conversation episodes (L0 history; timestamps and "
        "conversation provenance are authoritative):"
    )
    return _format_bounded_records(
        header=header,
        records=list(episodes),
        max_chars=max_chars,
        render_record=_render_episode,
    )


def format_memory_items(
    items: Sequence[MemoryItem],
    *,
    max_chars: int = DEFAULT_MEMORY_PROMPT_CHARS,
) -> str:
    """Render structured L3 memory, retaining kind, validity, and provenance."""

    if not items or max_chars <= 0:
        return ""
    header = (
        "Long-term memory (L3; semantic/episodic entries are context, while "
        "procedural entries are standing behavior rules):"
    )
    return _format_bounded_records(
        header=header,
        records=list(items),
        max_chars=max_chars,
        render_record=_render_memory_item,
    )


def _format_bounded_records(
    *,
    header: str,
    records: list[Any],
    max_chars: int,
    render_record: Any,
) -> str:
    limit = max(0, int(max_chars))
    if limit == 0:
        return ""
    if len(header) >= limit:
        return sentence_safe_clip(header, limit)

    parts = [header]
    omitted = 0
    for index, record in enumerate(records, start=1):
        separator_chars = 2
        remaining = limit - len("\n\n".join(parts)) - separator_chars
        # Leave enough room to disclose that later whole records were omitted.
        # Without this reserve, one large first record could consume the entire
        # budget and then be dropped merely to make an omission marker fit.
        marker_reserve = 64 if index < len(records) else 0
        record_budget = remaining - marker_reserve
        if record_budget < 48:
            omitted += len(records) - index + 1
            break
        block = render_record(record, index=index, max_chars=record_budget)
        if not block:
            omitted += 1
            continue
        parts.append(block)

    if omitted:
        while True:
            marker = (
                f"[{omitted} additional "
                f"{'record' if omitted == 1 else 'records'} omitted by prompt budget.]"
            )
            candidate = "\n\n".join([*parts, marker])
            if len(candidate) <= limit:
                parts.append(marker)
                break
            if len(parts) <= 1:
                # Extremely small caller budgets may fit only the clipped
                # header. Normal production budgets always retain this marker.
                break
            parts.pop()
            omitted += 1

    result = "\n\n".join(parts)
    return result if len(result) <= limit else sentence_safe_clip(result, limit)


def _render_episode(
    episode: HistoricalEpisode,
    *,
    index: int,
    max_chars: int,
) -> str:
    metadata = (
        f"Episode {index} | conversation={episode.conversation_id} | "
        f"{episode.started_at} -> {episode.ended_at}"
    )
    prefixes = [f"{message.role.title()} [{message.created_at}]: " for message in episode.messages]
    fixed = len(metadata) + sum(len(prefix) + 1 for prefix in prefixes)
    available = max_chars - fixed
    if available < max(16, 8 * len(episode.messages)):
        return ""
    per_message = max(8, available // max(1, len(episode.messages)))
    lines = [metadata]
    for message, prefix in zip(episode.messages, prefixes, strict=True):
        content = sentence_safe_clip(message.content, per_message)
        lines.append(prefix + content)
    block = "\n".join(lines)
    # Rounding the per-message split may leave a few characters over budget.
    if len(block) <= max_chars:
        return block
    return sentence_safe_clip(block, max_chars)


def _render_memory_item(item: MemoryItem, *, index: int, max_chars: int) -> str:
    del index
    source = ""
    if item.source_conversation_id:
        source = item.source_conversation_id
        if item.source_turn_seq is not None:
            source += f"#turn-{item.source_turn_seq}"
    else:
        source = "unspecified"
    validity = item.valid_from or "unspecified"
    metadata = (
        f"- [{item.kind}] {item.memory_key} "
        f"(confidence={item.confidence:.2f}; valid_from={validity}; source={source}): "
    )
    content_budget = max_chars - len(metadata)
    if content_budget < 8:
        return ""
    return metadata + sentence_safe_clip(item.content, content_budget)


def _partition_episodes(
    messages: tuple[HistoricalMessage, ...],
) -> tuple[tuple[HistoricalMessage, ...], ...]:
    episodes: list[tuple[HistoricalMessage, ...]] = []
    current: list[HistoricalMessage] = []
    for message in messages:
        if message.role == "user":
            if current:
                episodes.append(tuple(current))
            current = [message]
        elif message.role == "assistant":
            if current:
                current.append(message)
            else:
                # Imported histories occasionally begin with an assistant
                # message.  Keep it as its own auditable episode.
                current = [message]
    if current:
        episodes.append(tuple(current))
    return tuple(episodes)


def _safe_fts_query(tokens: Sequence[str]) -> str:
    safe: list[str] = []
    for token in tokens[:12]:
        # Tokens already come from _WORD_RE; quote escaping is defense in depth
        # and keeps this helper safe if it is reused independently later.
        escaped = token.replace('"', '""')
        if escaped:
            safe.append(f'"{escaped}"')
    return " OR ".join(safe)


def _raw_query_tokens(value: str) -> tuple[str, ...]:
    tokens: list[str] = []
    seen: set[str] = set()
    for match in _WORD_RE.finditer(str(value or "").casefold()):
        token = match.group(0)
        normalized = _normalize_token(token)
        if not normalized or normalized in _STOP_WORDS or normalized in seen:
            continue
        seen.add(normalized)
        tokens.append(token)
    return tuple(tokens[:24])


def _normalized_query_tokens(value: str) -> tuple[str, ...]:
    tokens: list[str] = []
    seen: set[str] = set()
    for match in _WORD_RE.finditer(str(value or "").casefold()):
        token = _normalize_token(match.group(0))
        if not token or token in _STOP_WORDS or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tuple(tokens[:24])


def _normalized_text_tokens(value: str) -> set[str]:
    return {
        token
        for match in _WORD_RE.finditer(str(value or "").casefold())
        if (token := _normalize_token(match.group(0))) and token not in _STOP_WORDS
    }


def _normalize_token(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold().replace("đ", "d"))
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def _token_overlap_score(query_tokens: Sequence[str], content: str) -> float:
    query_set = set(query_tokens)
    if not query_set:
        return 0.0
    content_tokens = _normalized_text_tokens(content)
    overlap = query_set & content_tokens
    if not overlap:
        return 0.0
    return len(overlap) / len(query_set)


def _memory_relevance(
    item: MemoryItem,
    query: str,
    query_tokens: Sequence[str],
) -> float:
    if item.kind == "procedural":
        # Procedures are applicable independent of lexical overlap.  A matching
        # procedure still ranks above another applicable procedure.
        base = 1.0
    else:
        base = 0.0
    query_set = set(query_tokens)
    key_tokens = _normalized_text_tokens(item.memory_key.replace("_", " "))
    content_tokens = _normalized_text_tokens(item.content)
    if query_set:
        base += 2.0 * len(query_set & key_tokens) / len(query_set)
        base += len(query_set & content_tokens) / len(query_set)
    normalized_query = " ".join(_normalized_query_tokens(query))
    normalized_content = " ".join(_normalized_query_tokens(f"{item.memory_key} {item.content}"))
    if normalized_query and normalized_query in normalized_content:
        base += 1.0
    return base


def _memory_item_from_row(row: Any) -> MemoryItem:
    metadata = _parse_json_object(row["metadata_json"])
    source_turn_seq = row["source_turn_seq"]
    return MemoryItem(
        id=str(row["id"]),
        scope=str(row["scope"]),
        conversation_id=_optional_compact(row["conversation_id"]),
        kind=str(row["kind"]),
        memory_key=str(row["memory_key"]),
        content=str(row["content"]),
        status=str(row["status"]),
        confidence=float(row["confidence"]),
        valid_from=_optional_compact(row["valid_from"]),
        valid_to=_optional_compact(row["valid_to"]),
        source_conversation_id=_optional_compact(row["source_conversation_id"]),
        source_turn_seq=int(source_turn_seq) if source_turn_seq is not None else None,
        supersedes_id=_optional_compact(row["supersedes_id"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        metadata=metadata,
    )


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE name = ? AND type IN ('table', 'view')",
        (table_name,),
    ).fetchone()
    return row is not None


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _required_compact(value: Any, field_name: str) -> str:
    compact = " ".join(str(value or "").split())
    if not compact:
        raise ValueError(f"{field_name} is required")
    return compact


def _optional_compact(value: Any) -> str | None:
    if value is None:
        return None
    compact = " ".join(str(value).split())
    return compact or None


def _validate_kind(value: str) -> str:
    kind = _required_compact(value, "kind").casefold()
    if kind not in MEMORY_KINDS:
        allowed = ", ".join(sorted(MEMORY_KINDS))
        raise ValueError(f"kind must be one of: {allowed}")
    return kind


def _validate_scope(value: str) -> str:
    scope = _required_compact(value, "scope").casefold()
    if scope not in _MEMORY_OPERATION_SCOPES:
        allowed = ", ".join(sorted(_MEMORY_OPERATION_SCOPES))
        raise ValueError(f"scope must be one of: {allowed}")
    return scope


def _validate_scope_conversation(scope: str, conversation_id: str | None) -> None:
    if scope == "conversation" and conversation_id is None:
        raise ValueError("conversation scope requires conversation_id")
    if scope == "user" and conversation_id is not None:
        raise ValueError("user scope must not set conversation_id")


def _normalize_timestamp(value: Any, field_name: str) -> str | None:
    compact = _optional_compact(value)
    if compact is None:
        return None
    candidate = compact[:-1] + "+00:00" if compact.endswith(("Z", "z")) else compact
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include a timezone")
    return parsed.astimezone(UTC).isoformat()


def _validate_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be a number from 0 to 1") from exc
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be a number from 0 to 1")
    return confidence


def _normalize_memory_operation(raw: object) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    action = str(raw.get("action") or "").strip().lower()
    scope = str(raw.get("scope") or "").strip().lower()
    kind = str(raw.get("kind") or "").strip().lower()
    key = str(raw.get("key") or "").strip().lower()
    content = " ".join(str(raw.get("content") or "").split())
    try:
        confidence = _validate_confidence(raw.get("confidence"))
    except (TypeError, ValueError):
        return None
    contains_sensitive_value = _contains_sensitive_memory_value(key, content)
    if (
        action not in _MEMORY_OPERATION_ACTIONS
        or scope not in _MEMORY_OPERATION_SCOPES
        or kind not in MEMORY_KINDS
        or not _MEMORY_OPERATION_KEY_RE.fullmatch(key)
        or (action == "upsert" and not content)
        or (action == "upsert" and contains_sensitive_value)
    ):
        return None
    return {
        "action": action,
        "scope": scope,
        "kind": kind,
        "key": key,
        # Forget operations address a logical key and never persist model text.
        "content": content if action == "upsert" else "",
        "confidence": confidence,
    }


def _contains_sensitive_memory_value(key: str, content: str) -> bool:
    return bool(
        _SENSITIVE_MEMORY_RE.search(f"{key} {content}")
        or _SENSITIVE_MEMORY_VALUE_RE.search(content)
    )


def _optional_nonnegative_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return parsed


def _json_object(value: dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("metadata must be a JSON object")
    try:
        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
        parsed = json.loads(serialized)
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must contain only JSON-compatible values") from exc
    return parsed if isinstance(parsed, dict) else {}


def _parse_json_object(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _timestamp_sort_value(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.timestamp()
    except (TypeError, ValueError, OverflowError):
        return 0.0
