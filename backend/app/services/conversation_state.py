"""L1 conversation working state — sticky session focus for RAG follow-ups."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any
import unicodedata

from app.db.sqlite import connect
from app.services.chat_history import utc_now


MAX_RECENT_DOCUMENT_THREADS = 4


@dataclass(frozen=True)
class DocumentThreadState:
    """A suspended document thread that can be resumed without searching globally."""

    document_ids: tuple[str, ...]
    topic: str | None = None
    filenames: tuple[str, ...] = ()
    last_answer_intent: str | None = None
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_ids": list(self.document_ids),
            "topic": self.topic,
            "filenames": list(self.filenames),
            "last_answer_intent": self.last_answer_intent,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class ConversationWorkingState:
    active_document_ids: list[str]
    active_topic: str | None = None
    active_filenames: list[str] | None = None
    last_answer_intent: str | None = None
    updated_at: str | None = None
    recent_document_threads: tuple[DocumentThreadState, ...] = ()
    # Last grounded multi-document comparison.  This is separate from the
    # currently active paper so "chúng/cả hai/both/them" can survive a later
    # single-paper turn or casual detour without guessing from a short window.
    referent_document_ids: list[str] | None = None
    referent_filenames: list[str] | None = None
    referent_topic: str | None = None
    referent_updated_at: str | None = None
    referent_source_turn_id: str | None = None

    @property
    def has_active_docs(self) -> bool:
        return bool(self.active_document_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_document_ids": list(self.active_document_ids),
            "active_topic": self.active_topic,
            "active_filenames": list(self.active_filenames or []),
            "last_answer_intent": self.last_answer_intent,
            "updated_at": self.updated_at,
            "recent_document_threads": [
                thread.to_dict() for thread in self.recent_document_threads
            ],
            "referent_document_ids": list(self.referent_document_ids or []),
            "referent_filenames": list(self.referent_filenames or []),
            "referent_topic": self.referent_topic,
            "referent_updated_at": self.referent_updated_at,
            "referent_source_turn_id": self.referent_source_turn_id,
        }

    def prompt_block(self) -> str:
        if not self.active_document_ids and not self.active_topic:
            return ""
        files = ", ".join(self.active_filenames or []) or "(ids only)"
        topic = self.active_topic or "(none)"
        block = (
            "Active conversation focus (working state):\n"
            f"- topic: {topic}\n"
            f"- documents: {files}\n"
            "Keep this paper/document focus across casual digressions unless the user "
            "clearly names a different paper or switches topic. Side chats are remembered "
            "in conversation summary / recent turn notes, not by clearing this focus."
        )
        if self.recent_document_threads:
            suspended = []
            for thread in reversed(self.recent_document_threads[-3:]):
                label = thread.topic or ", ".join(thread.filenames) or "(document ids only)"
                suspended.append(label)
            block += "\n- resumable previous document threads: " + "; ".join(suspended)
        if self.referent_document_ids:
            referents = ", ".join(self.referent_filenames or []) or "(ids only)"
            block += "\n- last grounded multi-document referents: " + referents
        return block


def empty_working_state() -> ConversationWorkingState:
    return ConversationWorkingState(active_document_ids=[])


@dataclass(frozen=True)
class ConversationStateStore:
    db_path: Path

    def get_working_state(self, conversation_id: str) -> ConversationWorkingState:
        with connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT metadata_json FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        if row is None:
            return empty_working_state()
        metadata = _parse_json_object(row["metadata_json"])
        working = metadata.get("working")
        if not isinstance(working, dict):
            return empty_working_state()
        doc_ids = [
            str(item)
            for item in (working.get("active_document_ids") or [])
            if item
        ]
        filenames = [
            str(item)
            for item in (working.get("active_filenames") or [])
            if item
        ]
        topic = working.get("active_topic")
        topic_text = str(topic).strip() if topic else None
        doc_ids = self._repair_document_ids(doc_ids, filenames, topic=topic_text)
        filenames = self._canonical_filenames(doc_ids, filenames)
        referent_filenames = _dedupe_str(working.get("referent_filenames") or [])
        referent_topic = _string_or_none(working.get("referent_topic"))
        referent_ids = self._repair_document_ids(
            _dedupe_str(working.get("referent_document_ids") or []),
            referent_filenames,
            topic=referent_topic,
        )
        referent_filenames = self._canonical_filenames(
            referent_ids,
            referent_filenames,
        )
        # Backward-compatible migration for conversations persisted before the
        # referent fields existed.
        if (
            len(referent_ids) < 2
            and len(doc_ids) >= 2
            and _string_or_none(working.get("last_answer_intent")) == "compare"
        ):
            referent_ids = list(doc_ids)
            referent_filenames = list(filenames)
            referent_topic = topic_text
        recent_threads: list[DocumentThreadState] = []
        for item in working.get("recent_document_threads") or []:
            if not isinstance(item, dict):
                continue
            thread_ids = tuple(_dedupe_str(item.get("document_ids") or []))
            thread_filenames = tuple(_dedupe_str(item.get("filenames") or []))
            thread_ids = tuple(
                self._repair_document_ids(
                    list(thread_ids),
                    list(thread_filenames),
                    topic=_string_or_none(item.get("topic")),
                )
            )
            thread_filenames = tuple(
                self._canonical_filenames(list(thread_ids), list(thread_filenames))
            )
            if not thread_ids:
                continue
            recent_threads.append(
                DocumentThreadState(
                    document_ids=thread_ids,
                    topic=_string_or_none(item.get("topic")),
                    filenames=thread_filenames,
                    last_answer_intent=_string_or_none(item.get("last_answer_intent")),
                    updated_at=_string_or_none(item.get("updated_at")),
                )
            )
        return ConversationWorkingState(
            active_document_ids=doc_ids,
            active_topic=topic_text or None,
            active_filenames=filenames,
            last_answer_intent=_string_or_none(working.get("last_answer_intent")),
            updated_at=_string_or_none(working.get("updated_at")),
            recent_document_threads=tuple(recent_threads[-MAX_RECENT_DOCUMENT_THREADS:]),
            referent_document_ids=referent_ids,
            referent_filenames=referent_filenames,
            referent_topic=referent_topic,
            referent_updated_at=_string_or_none(working.get("referent_updated_at")),
            referent_source_turn_id=_string_or_none(
                working.get("referent_source_turn_id")
            ),
        )

    def _repair_document_ids(
        self,
        document_ids: list[str],
        filenames: list[str],
        *,
        topic: str | None = None,
    ) -> list[str]:
        """Resolve stale pre-reindex UUIDs through unique canonical filenames."""
        with connect(self.db_path) as connection:
            existing_ids = {
                str(row["id"])
                for row in connection.execute("SELECT id FROM documents").fetchall()
            }
            if not existing_ids:
                return _dedupe_str(document_ids)
            repaired = [document_id for document_id in document_ids if document_id in existing_ids]
            missing_slots = max(0, len(document_ids) - len(repaired))
            for filename in filenames:
                if missing_slots <= 0:
                    break
                rows = connection.execute(
                    "SELECT id FROM documents WHERE filename = ? ORDER BY indexed_at DESC LIMIT 2",
                    (filename,),
                ).fetchall()
                if len(rows) != 1:
                    continue
                candidate = str(rows[0]["id"])
                if candidate not in repaired:
                    repaired.append(candidate)
                    missing_slots -= 1
            if missing_slots > 0 and _topic_can_resolve_document(topic):
                topic_key = _document_match_key(topic or "")
                candidates: list[str] = []
                rows = connection.execute(
                    """
                    SELECT d.id, d.filename, d.title, dc.title_guess
                    FROM documents d
                    LEFT JOIN document_cards dc ON dc.document_id = d.id
                    """
                ).fetchall()
                for row in rows:
                    searchable = " ".join(
                        str(row[field] or "")
                        for field in ("filename", "title", "title_guess")
                    )
                    if topic_key and topic_key in _document_match_key(searchable):
                        candidates.append(str(row["id"]))
                unique_candidates = _dedupe_str(candidates)
                if len(unique_candidates) == 1 and unique_candidates[0] not in repaired:
                    repaired.append(unique_candidates[0])
        return repaired

    def _canonical_filenames(
        self,
        document_ids: list[str],
        filenames: list[str],
    ) -> list[str]:
        repaired = _dedupe_str(filenames)
        if not document_ids:
            return repaired
        placeholders = ",".join("?" for _ in document_ids)
        with connect(self.db_path) as connection:
            rows = connection.execute(
                f"SELECT id, filename FROM documents WHERE id IN ({placeholders})",
                document_ids,
            ).fetchall()
        by_id = {str(row["id"]): str(row["filename"]) for row in rows}
        for document_id in document_ids:
            filename = by_id.get(document_id)
            if filename and filename not in repaired:
                repaired.append(filename)
        return repaired

    def get_effective_working_state(
        self,
        conversation_id: str,
        query: str,
    ) -> ConversationWorkingState:
        """Resolve an explicit "previous paper" request without mutating persisted L1.

        The selected thread is committed only after a grounded, completed retrieval turn.
        A casual detour never clears the active thread, so ordinary "back to the paper"
        requests continue to use the current state.
        """
        state = self.get_working_state(conversation_id)
        plural_referent_ids = resolve_plural_document_referent_ids(state, query)
        if len(plural_referent_ids) >= 2:
            plural_filenames = self._canonical_filenames(plural_referent_ids, [])
            plural_topic = _comparison_topic(plural_filenames) or state.referent_topic
            return ConversationWorkingState(
                active_document_ids=plural_referent_ids,
                active_topic=plural_topic or state.active_topic,
                active_filenames=plural_filenames,
                last_answer_intent="compare",
                updated_at=state.updated_at,
                recent_document_threads=state.recent_document_threads,
                referent_document_ids=plural_referent_ids,
                referent_filenames=plural_filenames,
                referent_topic=plural_topic,
                referent_updated_at=state.referent_updated_at,
                referent_source_turn_id=state.referent_source_turn_id,
            )
        if not requests_previous_document_thread(query) or not state.recent_document_threads:
            return state
        target = state.recent_document_threads[-1]
        return ConversationWorkingState(
            active_document_ids=list(target.document_ids),
            active_topic=target.topic,
            active_filenames=list(target.filenames),
            last_answer_intent=target.last_answer_intent,
            updated_at=target.updated_at,
            recent_document_threads=state.recent_document_threads,
            referent_document_ids=list(state.referent_document_ids or []),
            referent_filenames=list(state.referent_filenames or []),
            referent_topic=state.referent_topic,
            referent_updated_at=state.referent_updated_at,
            referent_source_turn_id=state.referent_source_turn_id,
        )

    def set_working_state(
        self,
        conversation_id: str,
        state: ConversationWorkingState,
    ) -> ConversationWorkingState:
        payload = state.to_dict()
        payload["updated_at"] = utc_now()
        with connect(self.db_path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT metadata_json FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            metadata = _parse_json_object(row["metadata_json"] if row else None)
            metadata["working"] = payload
            connection.execute(
                """
                UPDATE conversations
                SET metadata_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (json.dumps(metadata, ensure_ascii=False), payload["updated_at"], conversation_id),
            )
        return ConversationWorkingState(
            active_document_ids=list(payload["active_document_ids"]),
            active_topic=payload.get("active_topic"),
            active_filenames=list(payload.get("active_filenames") or []),
            last_answer_intent=payload.get("last_answer_intent"),
            updated_at=payload["updated_at"],
            recent_document_threads=state.recent_document_threads,
            referent_document_ids=list(payload.get("referent_document_ids") or []),
            referent_filenames=list(payload.get("referent_filenames") or []),
            referent_topic=payload.get("referent_topic"),
            referent_updated_at=payload.get("referent_updated_at"),
            referent_source_turn_id=payload.get("referent_source_turn_id"),
        )

    def update_from_retrieval(
        self,
        conversation_id: str,
        *,
        document_ids: list[str],
        topic: str | None = None,
        filenames: list[str] | None = None,
        answer_intent: str | None = None,
        source_turn_id: str | None = None,
    ) -> ConversationWorkingState:
        """Sync sticky focus after a successful local retrieval turn."""
        cleaned_ids = _dedupe_str(document_ids)
        if not cleaned_ids:
            return self.get_working_state(conversation_id)
        previous = self.get_working_state(conversation_id)
        recent_threads = list(previous.recent_document_threads)
        new_key = _document_thread_key(cleaned_ids)
        previous_key = _document_thread_key(previous.active_document_ids)
        # Re-activating a suspended paper removes its older duplicate from the stack.
        recent_threads = [
            thread
            for thread in recent_threads
            if _document_thread_key(list(thread.document_ids)) != new_key
        ]
        if previous.has_active_docs and previous_key != new_key:
            recent_threads.append(_thread_from_working_state(previous))
        canonical_filenames = self._canonical_filenames(cleaned_ids, filenames or [])
        if not canonical_filenames and previous_key == new_key:
            canonical_filenames = list(previous.active_filenames or [])
        referent_ids = list(previous.referent_document_ids or [])
        referent_filenames = list(previous.referent_filenames or [])
        referent_topic = previous.referent_topic
        referent_updated_at = previous.referent_updated_at
        referent_source_turn_id = previous.referent_source_turn_id
        if len(cleaned_ids) >= 2:
            referent_ids = cleaned_ids[:8]
            referent_filenames = canonical_filenames[:8]
            referent_topic = topic or _comparison_topic(referent_filenames)
            referent_updated_at = utc_now()
            referent_source_turn_id = source_turn_id
        return self.set_working_state(
            conversation_id,
            ConversationWorkingState(
                active_document_ids=cleaned_ids[:8],
                active_topic=(topic or previous.active_topic),
                active_filenames=canonical_filenames[:8],
                last_answer_intent=answer_intent or previous.last_answer_intent,
                recent_document_threads=tuple(
                    recent_threads[-MAX_RECENT_DOCUMENT_THREADS:]
                ),
                referent_document_ids=referent_ids,
                referent_filenames=referent_filenames,
                referent_topic=referent_topic,
                referent_updated_at=referent_updated_at,
                referent_source_turn_id=referent_source_turn_id,
            ),
        )


def looks_like_document_resume(query: str) -> bool:
    normalized = " ".join((query or "").lower().split())
    resume_marker = bool(
        re.search(
            r"\b(quay lại|trở lại|tiếp tục|quay về|back to|return to|resume|go back to)\b",
            normalized,
        )
    )
    document_marker = bool(
        re.search(
            r"\b(paper|bài(?: báo| nghiên cứu| lúc nãy| vừa rồi| trước)?|tài liệu|document|pdf|nghiên cứu)\b",
            normalized,
        )
    )
    return resume_marker and document_marker


def requests_previous_document_thread(query: str) -> bool:
    """True only for an explicit previous-paper request, not a casual detour resume."""
    normalized = " ".join((query or "").lower().split())
    if not looks_like_document_resume(query):
        return False
    return bool(
        re.search(
            r"\b(paper|bài(?: báo| nghiên cứu)?|tài liệu|document|pdf)\s+"
            r"(trước|trước đó|trước kia|previous|before)\b",
            normalized,
        )
        or re.search(r"\b(previous|prior)\s+(paper|document|pdf)\b", normalized)
    )


def requests_plural_document_referents(query: str) -> bool:
    """Detect anaphora that refers to a previously grounded document set.

    Patterns are deliberately language-level rather than paper/model-specific.
    Vietnamese first-person phrases such as ``chúng ta`` and ``chúng tôi`` are
    explicitly excluded.
    """

    normalized = " ".join((query or "").casefold().split())
    if not normalized:
        return False
    if re.search(r"\bchúng\b(?!\s+(?:ta|tôi|mình))", normalized):
        return True
    return bool(
        re.search(r"\bcả\s+(?:hai|2)\b", normalized)
        or re.search(
            r"\b(?:hai|2)\s+(?:bài(?:\s+báo)?|paper(?:s)?|tài\s+liệu|"
            r"document(?:s)?|model(?:s)?|mô\s+hình)(?:\s+(?:này|đó|trên|vừa\s+nói|lúc\s+nãy))?\b",
            normalized,
        )
        or re.search(
            r"\b(?:các|những)\s+(?:bài(?:\s+báo)?|paper(?:s)?|tài\s+liệu|"
            r"document(?:s)?|model(?:s)?|mô\s+hình)\s+(?:này|đó|trên|vừa\s+nói|lúc\s+nãy)\b",
            normalized,
        )
        or re.search(r"\b(?:both|them)\b", normalized)
        or re.search(r"\b(?:tụi|bọn)\s+(?:nó|này|đó|kia)\b", normalized)
        or re.search(
            r"\b(?:hai|2)\s+cái(?:\s+(?:này|đó|kia|trên|vừa\s+nói|lúc\s+nãy))?\b",
            normalized,
        )
        or re.search(
            r"\b(?:the\s+)?(?:two|2)\s+(?:papers?|documents?|models?)\b",
            normalized,
        )
        or re.search(r"\b(?:these|those)\s+(?:papers?|documents?|models?)\b", normalized)
        or re.search(
            r"\b(?:these|those)\s+(?:(?:two|2)\s+)?"
            r"(?:[a-z0-9_-]+\s+){0,3}(?:papers?|documents?|models?)\b",
            normalized,
        )
    )


def resolve_plural_document_referent_ids(
    state: ConversationWorkingState,
    query: str,
) -> list[str]:
    """Resolve plural anaphora from durable state or the nearest two threads.

    A grounded multi-document referent remains authoritative.  Before the user
    has compared a pair, an explicit count-two phrase (``hai bài đó``, ``both
    papers``) can still refer to the active document and the immediately
    suspended document thread.  Generic plural pronouns do not synthesize a
    new set from unrelated history.
    """

    if not requests_plural_document_referents(query):
        return []
    durable = _dedupe_str(state.referent_document_ids or [])
    if len(durable) >= 2:
        return durable[:8]
    normalized = " ".join((query or "").casefold().split())
    explicit_pair = bool(
        re.search(r"\b(?:cả\s+)?(?:hai|2)\b", normalized)
        or re.search(r"\b(?:both|(?:the\s+)?two|2)\b", normalized)
    )
    if not explicit_pair:
        return []
    nearest_previous: list[str] = []
    for thread in reversed(state.recent_document_threads):
        nearest_previous = _dedupe_str([*nearest_previous, *thread.document_ids])
        if nearest_previous:
            break
    # Preserve conversational chronology: the suspended paper was discussed
    # before the current one, so comparisons read in that same stable order.
    candidates = _dedupe_str([*nearest_previous, *state.active_document_ids])
    return candidates[:2] if len(candidates) >= 2 else []


def _thread_from_working_state(state: ConversationWorkingState) -> DocumentThreadState:
    return DocumentThreadState(
        document_ids=tuple(_dedupe_str(state.active_document_ids)),
        topic=state.active_topic,
        filenames=tuple(_dedupe_str(state.active_filenames or [])),
        last_answer_intent=state.last_answer_intent,
        updated_at=state.updated_at or utc_now(),
    )


def _document_thread_key(document_ids: list[str]) -> tuple[str, ...]:
    return tuple(sorted(_dedupe_str(document_ids)))


def _comparison_topic(filenames: list[str]) -> str | None:
    labels = [Path(filename).stem for filename in _dedupe_str(filenames)[:4]]
    return " vs ".join(labels) if len(labels) >= 2 else None


def _dedupe_str(values: list[str] | None) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values or []:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _parse_json_object(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _document_match_key(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold().replace("đ", "d"))
    asciiish = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "", asciiish)


def _topic_can_resolve_document(topic: str | None) -> bool:
    key = _document_match_key(topic or "")
    return len(key) >= 4 and key not in {
        "paper",
        "document",
        "tailieu",
        "nghiencuu",
        "architecture",
        "benchmark",
        "figure",
        "table",
    }
