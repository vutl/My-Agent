from app.api.chat import _chat_memory_context
from app.db.sqlite import init_db
from app.services.chat_history import ChatHistory
from app.services.conversation_memory import ConversationMemoryStore
from app.services.long_term_memory import HistoricalConversationSearch, MemoryItemStore


def test_direct_chat_context_gets_pending_l2_and_relevant_l3(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    history = ChatHistory(db_path)
    conversation_id = history.ensure_conversation(None, "memory")
    memory = ConversationMemoryStore(db_path)
    memory.record_completed_turn(
        conversation_id,
        user_text="Do not silently switch models.",
        assistant_text="Understood.",
    )
    MemoryItemStore(db_path).upsert(
        scope="user",
        kind="procedural",
        memory_key="no_model_fallback",
        content="Stop and surface quota errors; never switch models silently.",
        source_conversation_id=conversation_id,
        source_turn_seq=1,
    )

    context = _chat_memory_context(
        conversation_id=conversation_id,
        query="What should happen on model quota?",
        previous_messages=history.list_messages(conversation_id),
        memory_store=memory,
        historical_search=HistoricalConversationSearch(db_path),
        long_term_memory=MemoryItemStore(db_path),
    )

    assert "no_model_fallback" in context
    assert "Completed turns not yet folded" in context
    assert "Do not silently switch models" in context
