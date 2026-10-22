from __future__ import annotations

from datetime import timedelta

from fastapi.testclient import TestClient
import pytest

from app.core.config import Settings, get_settings
from app.db.sqlite import connect, init_db
from app.main import app
from app.services.agent_run_store import AgentRunStore
from app.services.chat_history import ChatHistory, ConversationBusyError
from app.services.reminders import ReminderStore, utc_now


@pytest.fixture
def setup(tmp_path):
    settings = Settings(data_dir=tmp_path / "data")
    init_db(settings.sqlite_db_path)
    history = ChatHistory(settings.sqlite_db_path)
    conversation = history.ensure_conversation(None, "Giá vàng hôm nay?")
    user = history.save_message(conversation_id=conversation, role="user", content="Giá vàng hôm nay?")
    history.save_message(conversation_id=conversation, role="assistant", content="142,3 triệu [W1]")
    runs = AgentRunStore(settings.sqlite_db_path)
    run = runs.create_run(conversation_id=conversation, user_message_id=user.id, mode="auto")
    runs.record_tool_call(run_id=run.id, tool_name="web_search", input_payload={"query": "x"}, output_payload={})
    runs.complete_run(run.id, "ok")
    reminders = ReminderStore(settings.sqlite_db_path)
    reminder = reminders.create(text="Uống nước", due_at=utc_now() + timedelta(hours=1), conversation_id=conversation)
    app.dependency_overrides[get_settings] = lambda: settings
    yield settings, history, runs, conversation, reminder
    app.dependency_overrides.clear()


def _count(db, sql, *args) -> int:
    with connect(db) as connection:
        return connection.execute(sql, args).fetchone()[0]


def test_delete_removes_the_conversation_everywhere_but_keeps_reminders(setup) -> None:
    settings, history, _runs, conversation, reminder = setup
    db = settings.sqlite_db_path
    client = TestClient(app)
    assert client.delete(f"/chat/conversations/{conversation}").status_code == 204
    assert _count(db, "SELECT COUNT(*) FROM conversations WHERE id = ?", conversation) == 0
    assert _count(db, "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", conversation) == 0
    assert _count(db, "SELECT COUNT(*) FROM agent_runs WHERE conversation_id = ?", conversation) == 0
    assert _count(db, "SELECT COUNT(*) FROM tool_calls") == 0
    assert _count(db, "SELECT COUNT(*) FROM message_search WHERE conversation_id = ?", conversation) == 0
    pending = ReminderStore(db).pending()
    assert [item.id for item in pending] == [reminder.id] and pending[0].conversation_id is None
    assert client.delete(f"/chat/conversations/{conversation}").status_code == 404


def test_a_conversation_still_answering_is_not_deleted(setup) -> None:
    settings, history, runs, conversation, _reminder = setup
    runs.create_run(conversation_id=conversation, user_message_id=None, mode="auto")  # status running
    with pytest.raises(ConversationBusyError):
        history.delete_conversation(conversation)
    assert TestClient(app).delete(f"/chat/conversations/{conversation}").status_code == 409
    assert _count(settings.sqlite_db_path, "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", conversation) == 2
