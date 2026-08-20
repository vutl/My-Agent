"""Deterministic tests for L0 history retrieval and structured L3 memory."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.db.sqlite import connect, init_db
from app.services.long_term_memory import (
    HistoricalConversationSearch,
    HistoricalEpisode,
    HistoricalMessage,
    MemoryItemStore,
    format_historical_episodes,
    requests_cross_thread_history,
    sentence_safe_clip,
)


MEMORY_ITEMS_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_items (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    conversation_id TEXT,
    kind TEXT NOT NULL,
    memory_key TEXT NOT NULL,
    content TEXT NOT NULL,
    status TEXT NOT NULL,
    confidence REAL NOT NULL,
    valid_from TEXT,
    valid_to TEXT,
    source_conversation_id TEXT,
    source_turn_seq INTEGER,
    supersedes_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata_json TEXT
);
"""


def _prepare_db(tmp_path: Path, *, with_fts: bool) -> Path:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    with connect(db_path) as connection:
        connection.executescript(MEMORY_ITEMS_SCHEMA)
        if with_fts:
            connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS message_search USING fts5(
                    message_id UNINDEXED,
                    conversation_id UNINDEXED,
                    content
                )
                """
            )
        else:
            # init_db may provide the production FTS table/triggers.  Removing
            # them here deterministically exercises the documented fallback.
            connection.execute("DROP TRIGGER IF EXISTS messages_search_insert")
            connection.execute("DROP TRIGGER IF EXISTS messages_search_delete")
            connection.execute("DROP TRIGGER IF EXISTS messages_search_update")
            connection.execute("DROP TABLE IF EXISTS message_search")
        for conversation_id, title in (
            ("c-old", "Old research thread"),
            ("c-current", "Current thread"),
            ("c-other", "Other private thread"),
        ):
            connection.execute(
                """
                INSERT INTO conversations (id, title, created_at, updated_at)
                VALUES (?, ?, '2026-07-01T00:00:00+00:00', '2026-07-01T00:00:00+00:00')
                """,
                (conversation_id, title),
            )
    return db_path


def _insert_message(
    db_path: Path,
    *,
    message_id: str,
    conversation_id: str,
    role: str,
    content: str,
    created_at: str,
    index_fts: bool,
) -> None:
    with connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO messages (id, conversation_id, role, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (message_id, conversation_id, role, content, created_at),
        )
        if index_fts:
            existing = connection.execute(
                "SELECT 1 FROM message_search WHERE message_id = ? LIMIT 1",
                (message_id,),
            ).fetchone()
            if existing is None:
                columns = {
                    str(row[1])
                    for row in connection.execute("PRAGMA table_info(message_search)").fetchall()
                }
                if "role" in columns:
                    connection.execute(
                        """
                        INSERT INTO message_search (message_id, conversation_id, role, content)
                        VALUES (?, ?, ?, ?)
                        """,
                        (message_id, conversation_id, role, content),
                    )
                else:
                    connection.execute(
                        """
                        INSERT INTO message_search (message_id, conversation_id, content)
                        VALUES (?, ?, ?)
                        """,
                        (message_id, conversation_id, content),
                    )


def test_fts_search_is_safe_and_expands_a_hit_to_the_complete_episode(tmp_path: Path) -> None:
    db_path = _prepare_db(tmp_path, with_fts=True)
    _insert_message(
        db_path,
        message_id="u-old",
        conversation_id="c-old",
        role="user",
        content="Hãy lưu ý benchmark ASPIRE và hai metric Acc, F1.",
        created_at="2026-07-17T01:00:00+00:00",
        index_fts=True,
    )
    _insert_message(
        db_path,
        message_id="a-old",
        conversation_id="c-old",
        role="assistant",
        content="Đã rõ. Khi quay lại tôi sẽ giữ đúng paper và kiểm tra evidence.",
        created_at="2026-07-17T01:00:05+00:00",
        index_fts=True,
    )
    _insert_message(
        db_path,
        message_id="u-recent",
        conversation_id="c-current",
        role="user",
        content="ASPIRE đang được hỏi lại ngay bây giờ.",
        created_at="2026-07-19T01:00:00+00:00",
        index_fts=True,
    )
    _insert_message(
        db_path,
        message_id="a-recent",
        conversation_id="c-current",
        role="assistant",
        content="Đây là raw recent context.",
        created_at="2026-07-19T01:00:05+00:00",
        index_fts=True,
    )

    # Quotes/operators/wildcards from user input cannot become FTS syntax.
    episodes = HistoricalConversationSearch(db_path).search(
        'ASPIRE " OR * NEAR(',
        conversation_id=None,
        exclude_message_ids=("u-recent", "a-recent"),
        limit=3,
    )

    assert len(episodes) == 1
    episode = episodes[0]
    assert episode.conversation_id == "c-old"
    assert episode.message_ids == ("u-old", "a-old")
    assert episode.matched_message_ids == ("u-old",)
    assert episode.user_text.startswith("Hãy lưu ý benchmark ASPIRE")
    assert episode.assistant_text.startswith("Đã rõ")
    assert episode.started_at == "2026-07-17T01:00:00+00:00"
    assert episode.ended_at == "2026-07-17T01:00:05+00:00"


def test_stale_fts_candidate_cannot_return_nonmatching_canonical_episode(
    tmp_path: Path,
) -> None:
    db_path = _prepare_db(tmp_path, with_fts=True)
    _insert_message(
        db_path,
        message_id="u-stale",
        conversation_id="c-old",
        role="user",
        content="ASPIRE was the original indexed topic.",
        created_at="2026-07-17T01:00:00+00:00",
        index_fts=True,
    )
    with connect(db_path) as connection:
        connection.execute("DROP TRIGGER IF EXISTS messages_search_update")
        connection.execute(
            "UPDATE messages SET content = 'Coffee is now the canonical topic.' "
            "WHERE id = 'u-stale'"
        )

    search = HistoricalConversationSearch(db_path)

    assert search.search("ASPIRE", conversation_id="c-old") == []
    assert [
        episode.message_ids
        for episode in search.search("coffee", conversation_id="c-old")
    ] == [("u-stale",)]


def test_history_scope_fallback_and_exclusion_are_episode_atomic(tmp_path: Path) -> None:
    db_path = _prepare_db(tmp_path, with_fts=False)
    for values in (
        (
            "u-old",
            "c-old",
            "user",
            "Tối qua chúng ta đã chọn món nào?",
            "2026-07-17T02:00:00+00:00",
        ),
        (
            "a-old",
            "c-old",
            "assistant",
            "Chúng ta chọn pizza dứa và trà đá.",
            "2026-07-17T02:00:05+00:00",
        ),
        (
            "u-current",
            "c-current",
            "user",
            "Pizza dứa của thread này thì sao?",
            "2026-07-19T02:00:00+00:00",
        ),
        (
            "a-current",
            "c-current",
            "assistant",
            "Thread hiện tại cũng có nhắc đến món đó.",
            "2026-07-19T02:00:05+00:00",
        ),
        (
            "u-generic-history",
            "c-other",
            "user",
            "Hôm qua chúng ta đã nói chuyện rất lâu.",
            "2026-07-18T02:00:00+00:00",
        ),
        (
            "a-generic-history",
            "c-other",
            "assistant",
            "Đúng vậy, đây chỉ là một câu lịch sử chung chung.",
            "2026-07-18T02:00:05+00:00",
        ),
    ):
        _insert_message(
            db_path,
            message_id=values[0],
            conversation_id=values[1],
            role=values[2],
            content=values[3],
            created_at=values[4],
            index_fts=False,
        )

    search = HistoricalConversationSearch(db_path)
    same_thread = search.search("pizza dứa", conversation_id="c-current")
    cross_thread = search.search("pizza dứa", conversation_id=None, limit=5)
    safe_default = search.search_for_context(
        "Nhắc lại pizza dứa",
        current_conversation_id="c-current",
    )
    explicit_history = search.search_for_context(
        "Hôm qua chúng ta đã nói gì về pizza dứa?",
        current_conversation_id="c-current",
        limit=5,
    )

    assert [episode.conversation_id for episode in same_thread] == ["c-current"]
    assert {episode.conversation_id for episode in cross_thread} == {"c-old", "c-current"}
    assert ("u-old", "a-old") in {episode.message_ids for episode in cross_thread}
    assert [episode.conversation_id for episode in safe_default] == ["c-current"]
    assert {episode.conversation_id for episode in explicit_history} == {"c-old", "c-current"}
    assert all(episode.conversation_id != "c-other" for episode in explicit_history)

    # Excluding either half excludes the whole exchange, so the historical
    # prompt never duplicates a partial raw-recent episode.
    excluded = search.search(
        "pizza dứa",
        conversation_id="c-old",
        exclude_message_ids=("a-old",),
    )
    assert excluded == []
    assert requests_cross_thread_history("Hôm qua chúng ta đã nói gì về pizza?")
    assert requests_cross_thread_history("Chúng ta đã thảo luận benchmark này chưa?")
    assert requests_cross_thread_history("Check the previous conversation about pizza")
    assert not requests_cross_thread_history("Quay lại paper trong chat này")
    assert not requests_cross_thread_history("Paper đã nói gì về benchmark?")
    assert not requests_cross_thread_history("This method was previously proposed in the paper")


def test_historical_prompt_is_temporal_sentence_safe_and_bounded(tmp_path: Path) -> None:
    db_path = _prepare_db(tmp_path, with_fts=False)
    _insert_message(
        db_path,
        message_id="u1",
        conversation_id="c-old",
        role="user",
        content="First complete sentence. Second sentence contains unnecessary detail " * 8,
        created_at="2026-07-10T03:00:00+00:00",
        index_fts=False,
    )
    _insert_message(
        db_path,
        message_id="a1",
        conversation_id="c-old",
        role="assistant",
        content="A complete answer. More detail that should be bounded safely " * 8,
        created_at="2026-07-10T03:00:05+00:00",
        index_fts=False,
    )
    episodes = HistoricalConversationSearch(db_path).search(
        "complete sentence answer",
        conversation_id="c-old",
    )

    prompt = format_historical_episodes(episodes, max_chars=520)

    assert len(prompt) <= 520
    assert "conversation=c-old" in prompt
    assert "2026-07-10T03:00:00+00:00" in prompt
    assert "User [" in prompt and "Assistant [" in prompt
    clipped = sentence_safe_clip(
        "First complete sentence. Second sentence must not be cut in its middle.",
        35,
    )
    assert clipped == "First complete sentence. …"
    assert len(clipped) <= 35


def test_historical_prompt_marks_whole_episodes_omitted_by_budget() -> None:
    episodes = [
        HistoricalEpisode(
            conversation_id=f"c-{index}",
            messages=(
                HistoricalMessage(
                    id=f"u-{index}",
                    conversation_id=f"c-{index}",
                    role="user",
                    content=f"Question {index} with enough detail to consume prompt space.",
                    created_at=f"2026-07-{index:02d}T00:00:00+00:00",
                ),
                HistoricalMessage(
                    id=f"a-{index}",
                    conversation_id=f"c-{index}",
                    role="assistant",
                    content=f"Answer {index} with a complete sentence.",
                    created_at=f"2026-07-{index:02d}T00:00:05+00:00",
                ),
            ),
            matched_message_ids=(f"u-{index}",),
            score=1.0,
            started_at=f"2026-07-{index:02d}T00:00:00+00:00",
            ended_at=f"2026-07-{index:02d}T00:00:05+00:00",
        )
        for index in range(1, 6)
    ]

    prompt = format_historical_episodes(episodes, max_chars=500)

    assert len(prompt) <= 500
    assert "omitted by prompt budget" in prompt
    assert not prompt.endswith("consu")


def test_memory_upsert_creates_versions_with_validity_and_provenance(tmp_path: Path) -> None:
    db_path = _prepare_db(tmp_path, with_fts=False)
    store = MemoryItemStore(db_path)

    old = store.upsert(
        scope="user",
        kind="semantic",
        memory_key="preferred_tea",
        content="The user prefers green tea.",
        confidence=0.75,
        valid_from="2026-07-01T00:00:00+00:00",
        source_conversation_id="c-old",
        source_turn_seq=3,
        metadata={"evidence_message_id": "u-old"},
    )
    new = store.upsert(
        scope="user",
        kind="semantic",
        memory_key="preferred_tea",
        content="The user now prefers jasmine tea.",
        confidence=0.95,
        valid_from="2026-07-10T00:00:00+00:00",
        source_conversation_id="c-current",
        source_turn_seq=7,
        metadata={"evidence_message_id": "u-current"},
    )

    closed_old = store.get(old.id)
    assert closed_old is not None
    assert closed_old.status == "superseded"
    assert closed_old.valid_to == new.valid_from
    assert new.supersedes_id == old.id
    assert new.source_conversation_id == "c-current"
    assert new.source_turn_seq == 7
    assert new.metadata == {"evidence_message_id": "u-current"}

    historical = store.relevant(
        "tea preference",
        as_of="2026-07-05T00:00:00+00:00",
    )
    current = store.relevant(
        "tea preference",
        as_of="2026-07-15T00:00:00+00:00",
    )
    assert [item.id for item in historical] == [old.id]
    assert [item.id for item in current] == [new.id]


def test_memory_validity_normalizes_offsets_before_temporal_comparison(
    tmp_path: Path,
) -> None:
    db_path = _prepare_db(tmp_path, with_fts=False)
    store = MemoryItemStore(db_path)
    item = store.upsert(
        scope="user",
        kind="semantic",
        memory_key="timezone_preference",
        content="The user works in the UTC+7 timezone.",
        valid_from="2026-07-10T07:00:00+07:00",
    )

    assert item.valid_from == "2026-07-10T00:00:00+00:00"
    assert [
        memory.id
        for memory in store.relevant(
            "timezone preference",
            as_of="2026-07-10T00:30:00Z",
        )
    ] == [item.id]
    with pytest.raises(ValueError, match="must include a timezone"):
        store.relevant(
            "timezone preference",
            as_of="2026-07-10T00:30:00",
        )


def test_memory_relevance_always_includes_procedure_without_cross_thread_leak(tmp_path: Path) -> None:
    db_path = _prepare_db(tmp_path, with_fts=False)
    store = MemoryItemStore(db_path)
    common = {"valid_from": "2026-07-01T00:00:00+00:00"}

    procedure = store.upsert(
        scope="user",
        kind="procedural",
        memory_key="no_model_fallback",
        content="If GPT quota is exhausted, stop and report it; never switch models silently.",
        **common,
    )
    tea = store.upsert(
        scope="user",
        kind="semantic",
        memory_key="preferred_tea",
        content="The user likes jasmine tea.",
        **common,
    )
    unrelated = store.upsert(
        scope="user",
        kind="episodic",
        memory_key="pizza_chat",
        content="The user discussed pineapple pizza last week.",
        **common,
    )
    private_other_thread = store.upsert(
        scope="conversation",
        conversation_id="c-other",
        kind="semantic",
        memory_key="preferred_tea",
        content="A different thread mentioned black tea.",
        **common,
    )

    items = store.relevant(
        "Which tea does the user prefer?",
        conversation_id="c-current",
        as_of="2026-07-15T00:00:00+00:00",
    )
    ids = [item.id for item in items]

    assert ids[0] == procedure.id
    assert tea.id in ids
    assert unrelated.id not in ids
    assert private_other_thread.id not in ids


def test_memory_forget_and_prompt_block_are_auditable_and_bounded(tmp_path: Path) -> None:
    db_path = _prepare_db(tmp_path, with_fts=False)
    store = MemoryItemStore(db_path)
    item = store.upsert(
        scope="user",
        kind="semantic",
        memory_key="answer_style",
        content=(
            "The user prefers concise answers. "
            "This second sentence contains a large amount of optional elaboration. " * 8
        ),
        valid_from="2026-07-01T00:00:00+00:00",
        source_conversation_id="c-current",
        source_turn_seq=11,
    )

    prompt = store.prompt_block(
        "How should the answer style look?",
        conversation_id="c-current",
        max_chars=420,
        as_of="2026-07-15T00:00:00+00:00",
    )
    assert len(prompt) <= 420
    assert "[semantic] answer_style" in prompt
    assert "source=c-current#turn-11" in prompt

    forgotten = store.forget(item.id, valid_to="2026-07-16T00:00:00+00:00")
    assert forgotten is not None
    assert forgotten.status == "forgotten"
    assert forgotten.valid_to == "2026-07-16T00:00:00+00:00"
    assert [
        memory.id
        for memory in store.relevant(
            "answer style",
            conversation_id="c-current",
            as_of="2026-07-15T00:00:00+00:00",
        )
    ] == [item.id]
    assert store.relevant(
        "answer style",
        conversation_id="c-current",
        as_of="2026-07-17T00:00:00+00:00",
    ) == []


def test_l2_memory_operations_are_idempotent_scoped_and_sensitive_safe(tmp_path: Path) -> None:
    db_path = _prepare_db(tmp_path, with_fts=False)
    store = MemoryItemStore(db_path)
    operations = [
        {
            "action": "upsert",
            "scope": "user",
            "kind": "procedural",
            "key": "no_model_fallback",
            "content": "Stop and report quota errors; never switch models silently.",
            "confidence": 1.0,
        },
        {
            "action": "upsert",
            "scope": "conversation",
            "kind": "episodic",
            "key": "active_memory_refactor",
            "content": "The current thread is implementing durable memory.",
            "confidence": 0.9,
        },
        {
            "action": "upsert",
            "scope": "user",
            "kind": "semantic",
            "key": "api_key",
            "content": "secret-token-value",
            "confidence": 1.0,
        },
    ]

    first = store.apply_operations(
        operations,
        source_conversation_id="c-current",
        source_turn_seq=12,
    )
    repeated = store.apply_operations(
        operations,
        source_conversation_id="c-current",
        source_turn_seq=12,
    )

    assert len(first) == 2
    assert [item.id for item in repeated] == [item.id for item in first]
    assert first[0].conversation_id is None
    assert first[1].conversation_id == "c-current"
    with connect(db_path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM memory_items").fetchone()[0]
    assert count == 2

    updated = store.apply_operations(
        [
            {
                "action": "upsert",
                "scope": "user",
                "kind": "procedural",
                "key": "no_model_fallback",
                "content": "Stop immediately and surface the provider error.",
                "confidence": 1.0,
            }
        ],
        source_conversation_id="c-current",
        source_turn_seq=13,
    )[0]
    assert updated.supersedes_id == first[0].id
    assert store.get(first[0].id).status == "superseded"


def test_memory_scope_invariants_reject_cross_thread_storage_shapes(tmp_path: Path) -> None:
    db_path = _prepare_db(tmp_path, with_fts=False)
    store = MemoryItemStore(db_path)

    with pytest.raises(ValueError, match="conversation scope requires"):
        store.upsert(
            scope="conversation",
            kind="episodic",
            memory_key="decision",
            content="This must remain thread-local.",
        )
    with pytest.raises(ValueError, match="user scope must not"):
        store.upsert(
            scope="user",
            conversation_id="c-current",
            kind="semantic",
            memory_key="preference",
            content="A global item cannot carry a private thread ID.",
        )
    with pytest.raises(ValueError, match="scope must be one of"):
        store.upsert(
            scope="workspace",
            kind="semantic",
            memory_key="invalid_scope",
            content="Unsupported scope.",
        )


def test_forget_receipt_cannot_close_a_newer_version_on_replay(tmp_path: Path) -> None:
    db_path = _prepare_db(tmp_path, with_fts=False)
    store = MemoryItemStore(db_path)
    old = store.upsert(
        scope="user",
        kind="semantic",
        memory_key="api_key",
        content="Legacy sensitive value that must be purgeable.",
        valid_from="2026-07-01T00:00:00+00:00",
    )
    forget_operation = {
        "action": "forget",
        "scope": "user",
        "kind": "semantic",
        "key": "api_key",
        "content": "",
        "confidence": 1.0,
    }

    first = store.apply_operations(
        [forget_operation],
        source_conversation_id="c-current",
        source_turn_seq=20,
    )
    assert [item.id for item in first] == [old.id]
    assert first[0].status == "forgotten"

    newer = store.upsert(
        scope="user",
        kind="semantic",
        memory_key="api_key",
        content="A replacement created after the old forget was committed.",
        valid_from="2026-07-20T00:00:00+00:00",
    )
    replay = store.apply_operations(
        [forget_operation],
        source_conversation_id="c-current",
        source_turn_seq=20,
    )

    assert [item.id for item in replay] == [old.id]
    assert store.get(newer.id).status == "active"
    with connect(db_path) as connection:
        receipt_count = connection.execute(
            """
            SELECT COUNT(*) FROM memory_operation_receipts
            WHERE source_conversation_id = 'c-current' AND source_turn_seq = 20
            """
        ).fetchone()[0]
    assert receipt_count == 1


def test_memory_operations_reject_raw_secret_values_without_secret_labels(
    tmp_path: Path,
) -> None:
    db_path = _prepare_db(tmp_path, with_fts=False)
    store = MemoryItemStore(db_path)

    applied = store.apply_operations(
        [
            {
                "action": "upsert",
                "scope": "user",
                "kind": "semantic",
                "key": "project_credential",
                "content": "Use sk-proj-abcdefghijklmnopqrstuvwxyz012345 for this project.",
                "confidence": 1.0,
            }
        ],
        source_conversation_id="c-current",
        source_turn_seq=21,
    )

    assert applied == []
    with connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO memory_items (
                id, scope, kind, memory_key, content, status, confidence,
                valid_from, created_at, updated_at
            ) VALUES (
                'legacy-secret', 'user', 'semantic', 'project_credential',
                'Use sk-proj-abcdefghijklmnopqrstuvwxyz012345 for this project.',
                'active', 1.0, '2026-07-01T00:00:00+00:00',
                '2026-07-01T00:00:00+00:00', '2026-07-01T00:00:00+00:00'
            )
            """
        )
    assert store.relevant(
        "project credential",
        as_of="2026-07-15T00:00:00+00:00",
    ) == []
