"""L1 conversation working state tests."""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.db.sqlite import connect, init_db
from app.services.chat_history import ChatHistory
from app.services.conversation_state import (
    ConversationStateStore,
    ConversationWorkingState,
    DocumentThreadState,
    looks_like_document_resume,
    requests_plural_document_referents,
    requests_previous_document_thread,
    resolve_plural_document_referent_ids,
)
from app.services.query_rewrite_service import QueryRewriteService


class _Msg:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


class _NoLLM:
    async def chat(self, **kwargs):
        raise AssertionError("should not call LLM")


def test_working_state_roundtrip(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    history = ChatHistory(db_path)
    store = ConversationStateStore(db_path)
    conversation_id = history.ensure_conversation(None, "hello ASPIRE")

    empty = store.get_working_state(conversation_id)
    assert empty.active_document_ids == []

    updated = store.update_from_retrieval(
        conversation_id,
        document_ids=["doc-aspire"],
        topic="ASPIRE",
        filenames=["ASPIRE.pdf"],
        answer_intent="infer_structure",
    )
    assert updated.active_document_ids == ["doc-aspire"]
    assert updated.active_topic == "ASPIRE"
    assert updated.active_filenames == ["ASPIRE.pdf"]

    loaded = store.get_working_state(conversation_id)
    assert loaded.to_dict()["active_topic"] == "ASPIRE"
    assert "ASPIRE.pdf" in loaded.prompt_block()


def test_visual_followup_prefers_sticky_working_topic() -> None:
    messages = [
        _Msg("user", "Pitch-fusion architecture"),
        _Msg("assistant", "Pitch-fusion dùng Wav2Vec."),
        _Msg("user", "thế ASPIRE đi"),
        _Msg("assistant", "ASPIRE multimodal SER."),
    ]

    async def _run():
        return await QueryRewriteService(client=_NoLLM(), default_model="x").rewrite(
            query="có hình architecture không",
            previous_messages=messages,
            working_topic="ASPIRE",
            working_document_hint="ASPIRE.pdf",
        )

    result = asyncio.run(_run())
    assert result.current_topic == "ASPIRE"
    assert result.use_last_sources is True
    assert result.diagnostics.get("working_topic") == "ASPIRE"


def test_set_working_state_merges_metadata(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    history = ChatHistory(db_path)
    store = ConversationStateStore(db_path)
    conversation_id = history.ensure_conversation(None, "hi")

    store.set_working_state(
        conversation_id,
        ConversationWorkingState(
            active_document_ids=["a"],
            active_topic="A",
            active_filenames=["A.pdf"],
        ),
    )
    store.set_working_state(
        conversation_id,
        ConversationWorkingState(
            active_document_ids=["b"],
            active_topic="B",
            active_filenames=["B.pdf"],
        ),
    )
    loaded = store.get_working_state(conversation_id)
    assert loaded.active_topic == "B"
    assert loaded.active_document_ids == ["b"]


def test_document_thread_stack_resumes_previous_paper_without_early_commit(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    history = ChatHistory(db_path)
    store = ConversationStateStore(db_path)
    conversation_id = history.ensure_conversation(None, "ASPIRE")

    store.update_from_retrieval(
        conversation_id,
        document_ids=["doc-aspire"],
        topic="ASPIRE",
        filenames=["ASPIRE.pdf"],
    )
    store.update_from_retrieval(
        conversation_id,
        document_ids=["doc-whiser"],
        topic="WhiSER",
        filenames=["WhiSER.pdf"],
    )

    persisted = store.get_working_state(conversation_id)
    assert persisted.active_document_ids == ["doc-whiser"]
    assert [thread.topic for thread in persisted.recent_document_threads] == ["ASPIRE"]

    effective = store.get_effective_working_state(
        conversation_id,
        "Quay lại paper trước đi",
    )
    assert effective.active_document_ids == ["doc-aspire"]
    assert effective.active_topic == "ASPIRE"
    # Selection is read-only until a grounded turn commits it.
    assert store.get_working_state(conversation_id).active_document_ids == ["doc-whiser"]


def test_casual_detour_resume_keeps_current_document_thread(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    history = ChatHistory(db_path)
    store = ConversationStateStore(db_path)
    conversation_id = history.ensure_conversation(None, "ASPIRE")
    store.update_from_retrieval(
        conversation_id,
        document_ids=["doc-aspire"],
        topic="ASPIRE",
        filenames=["ASPIRE.pdf"],
    )

    effective = store.get_effective_working_state(
        conversation_id,
        "Quay lại paper lúc nãy, benchmark thế nào?",
    )
    assert effective.active_document_ids == ["doc-aspire"]
    assert effective.active_topic == "ASPIRE"


def test_natural_vietnamese_bai_resume_phrases() -> None:
    assert looks_like_document_resume("Quay lại bài lúc nãy đi")
    assert looks_like_document_resume("Quay lại bài vừa rồi")
    assert requests_previous_document_thread("Quay lại bài trước")
    assert not requests_previous_document_thread("Quay lại bài lúc nãy")


def test_plural_document_referent_phrases_exclude_first_person() -> None:
    for query in (
        "đưa bảng kết quả của chúng",
        "cả hai dùng dataset nào?",
        "hai bài này khác gì nhau?",
        "compare both",
        "what protocols do they use? show them",  # explicit English object pronoun
    ):
        assert requests_plural_document_referents(query), query
    for query in ("chúng ta làm tiếp", "chúng tôi cần test", "chúng mình đi thôi"):
        assert not requests_plural_document_referents(query), query


def test_explicit_two_referent_reconstructs_active_and_nearest_suspended_thread() -> None:
    state = ConversationWorkingState(
        active_document_ids=["doc-new"],
        recent_document_threads=(
            DocumentThreadState(document_ids=("doc-oldest",), topic="oldest"),
            DocumentThreadState(document_ids=("doc-previous",), topic="previous"),
        ),
    )

    assert resolve_plural_document_referent_ids(
        state,
        "So sánh kết quả hai bài đó.",
    ) == ["doc-previous", "doc-new"]
    assert resolve_plural_document_referent_ids(state, "tụi nó khác gì?") == []


def test_grounded_multi_document_referent_survives_later_single_document_turn(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    history = ChatHistory(db_path)
    store = ConversationStateStore(db_path)
    conversation_id = history.ensure_conversation(None, "compare")

    compared = store.update_from_retrieval(
        conversation_id,
        document_ids=["doc-a", "doc-b"],
        topic="A vs B",
        filenames=["A.pdf", "B.pdf"],
        answer_intent="direct_answer",
        source_turn_id="turn-compare",
    )
    assert compared.referent_document_ids == ["doc-a", "doc-b"]
    assert compared.referent_source_turn_id == "turn-compare"
    assert compared.referent_updated_at

    store.update_from_retrieval(
        conversation_id,
        document_ids=["doc-c"],
        topic="C",
        filenames=["C.pdf"],
        answer_intent="direct_answer",
        source_turn_id="turn-c",
    )
    persisted = store.get_working_state(conversation_id)
    assert persisted.active_document_ids == ["doc-c"]
    assert persisted.referent_document_ids == ["doc-a", "doc-b"]

    effective = store.get_effective_working_state(
        conversation_id,
        "bảng của cả hai thì sao?",
    )
    assert effective.active_document_ids == ["doc-a", "doc-b"]
    assert effective.last_answer_intent == "compare"


def test_stale_working_document_id_repairs_from_unique_filename(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    history = ChatHistory(db_path)
    store = ConversationStateStore(db_path)
    conversation_id = history.ensure_conversation(None, "ASPIRE")
    store.set_working_state(
        conversation_id,
        ConversationWorkingState(
            active_document_ids=["old-id"],
            active_topic="ASPIRE",
            active_filenames=["ASPIRE.pdf"],
        ),
    )
    with connect(db_path) as connection:
        connection.execute(
            """INSERT INTO indexed_folders
               (id, folder_path, recursive, file_types, created_at, updated_at)
               VALUES ('folder', '/papers', 0, '[]', 'now', 'now')"""
        )
        connection.execute(
            """INSERT INTO documents
               (id, folder_id, source_path, filename, file_type, content_hash,
                modified_at, indexed_at, chunk_count)
               VALUES ('new-id', 'folder', '/papers/ASPIRE.pdf', 'ASPIRE.pdf',
                       'pdf', 'hash', 'now', 'now', 0)"""
        )

    assert store.get_working_state(conversation_id).active_document_ids == ["new-id"]


def test_stale_working_document_id_repairs_from_unique_topic_without_filename(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    history = ChatHistory(db_path)
    store = ConversationStateStore(db_path)
    conversation_id = history.ensure_conversation(None, "ASPIRE")
    store.set_working_state(
        conversation_id,
        ConversationWorkingState(
            active_document_ids=["pre-migration-id"],
            active_topic="ASPIRE",
            active_filenames=[],
        ),
    )
    with connect(db_path) as connection:
        connection.execute(
            """INSERT INTO indexed_folders
               (id, folder_path, recursive, file_types, created_at, updated_at)
               VALUES ('folder', '/papers', 0, '[]', 'now', 'now')"""
        )
        connection.execute(
            """INSERT INTO documents
               (id, folder_id, source_path, filename, file_type, content_hash,
                modified_at, indexed_at, chunk_count)
               VALUES ('canonical-aspire', 'folder', '/papers/ASPIRE.pdf', 'ASPIRE.pdf',
                       'pdf', 'hash', 'now', 'now', 0)"""
        )

    loaded = store.get_working_state(conversation_id)
    assert loaded.active_document_ids == ["canonical-aspire"]
    assert loaded.active_filenames == ["ASPIRE.pdf"]
