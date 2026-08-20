import json

from app.db.sqlite import connect, init_db
from app.services.chat_history import ChatHistory


def test_chat_history_creates_conversation_and_stores_messages(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    history = ChatHistory(db_path)

    conversation_id = history.ensure_conversation(None, "Plan this project")
    history.save_message(
        conversation_id=conversation_id,
        role="user",
        content="Plan this project",
        model="qwen3.5:4b",
    )
    history.save_message(
        conversation_id=conversation_id,
        role="assistant",
        content="Start with Phase 0.",
        model="qwen3.5:4b",
    )

    conversations = history.list_conversations()
    messages = history.list_messages(conversation_id)

    assert conversations[0].id == conversation_id
    assert conversations[0].title == "Plan this project"
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[1].content == "Start with Phase 0."


def test_reopening_conversation_marks_memory_generation_without_losing_metadata(
    tmp_path,
) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    history = ChatHistory(db_path)
    conversation_id = history.ensure_conversation(None, "first")
    with connect(db_path) as connection:
        connection.execute(
            "UPDATE conversations SET metadata_json = ? WHERE id = ?",
            (
                json.dumps(
                    {
                        "working": {"active_topic": "ASPIRE"},
                        "memory": {"revision": 7},
                    }
                ),
                conversation_id,
            ),
        )

    history.ensure_conversation(conversation_id, "resume")

    with connect(db_path) as connection:
        raw = connection.execute(
            "SELECT metadata_json FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()[0]
    metadata = json.loads(raw)
    assert metadata["memory_queue_version"] == 1
    assert metadata["working"] == {"active_topic": "ASPIRE"}
    assert metadata["memory"] == {"revision": 7}


def test_chat_history_round_trips_retrieval_sources(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    history = ChatHistory(db_path)

    conversation_id = history.ensure_conversation(None, "Show figure 2")
    sources = [
        {
            "chunk_id": "figure:fig-2",
            "document_id": "paper-1",
            "source_path": "/papers/example.pdf",
            "filename": "example.pdf",
            "content": "Figure 2. Proposed architecture.",
            "score": 0.94,
            "figure_id": "fig-2",
            "figure_index": 1,
            "page_number": 4,
            "image_url": "/rag/figures/fig-2/image",
            "caption": "Figure 2. Proposed architecture.",
        }
    ]
    history.save_message(
        conversation_id=conversation_id,
        role="assistant",
        content="Đây là kiến trúc của paper.",
        model="cx/gpt-5.5",
        sources=sources,
    )

    stored = history.list_messages(conversation_id)[0]

    assert stored.sources == sources
    assert stored.sources[0]["figure_id"] == "fig-2"
    assert stored.sources[0]["image_url"] == "/rag/figures/fig-2/image"


def test_chat_history_preserves_table_identity(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    history = ChatHistory(db_path)

    conversation_id = history.ensure_conversation(None, "Show the benchmark table")
    source = {
        "chunk_id": "table:table-3",
        "document_id": "paper-1",
        "source_path": "/papers/example.pdf",
        "filename": "example.pdf",
        "content": "| Acc | F1 | CCC |\n|---:|---:|---:|\n| 75.86 | 76.31 | 0.714 |",
        "score": 0.98,
        "chunk_type": "table",
        "artifact_type": "table",
        "table_id": "table-3",
        "table_index": 2,
        "page_number": 5,
        "caption": "Table 3: Benchmark results",
    }
    history.save_message(
        conversation_id=conversation_id,
        role="assistant",
        content="Đây là bảng benchmark.",
        model="cx/gpt-5.5",
        sources=[source],
    )

    stored = history.list_messages(conversation_id)[0]

    assert stored.sources == [source]
    assert stored.sources[0]["table_id"] == "table-3"
    assert stored.sources[0]["table_index"] == 2


def test_chat_history_tolerates_malformed_message_metadata(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    history = ChatHistory(db_path)

    conversation_id = history.ensure_conversation(None, "Hello")
    message = history.save_message(
        conversation_id=conversation_id,
        role="assistant",
        content="Hi",
    )
    with connect(db_path) as connection:
        connection.execute(
            "UPDATE messages SET metadata_json = ? WHERE id = ?",
            ("{not json", message.id),
        )

    assert history.list_messages(conversation_id)[0].sources == []


def test_chat_history_compacts_retrieval_trace_fields(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    history = ChatHistory(db_path)
    conversation_id = history.ensure_conversation(None, "source")

    history.save_message(
        conversation_id=conversation_id,
        role="assistant",
        content="answer",
        sources=[
            {
                "chunk_id": "chunk",
                "document_id": "doc",
                "filename": "paper.pdf",
                "source_path": "/paper.pdf",
                "content": "x" * 5000,
                "score": 1.0,
                "expanded_content": "do not persist",
                "provider_raw": {"large": "trace"},
            }
        ],
    )

    source = history.list_messages(conversation_id)[0].sources[0]
    assert len(source["content"]) == 1600
    assert "expanded_content" not in source
    assert "provider_raw" not in source
