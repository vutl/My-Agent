from app.api.health import _gateway_status, _memory_status
from app.core.config import Settings
from app.db.sqlite import connect, init_db


def test_default_answer_and_router_stay_on_9router_gpt() -> None:
    settings = Settings()
    assert settings.default_model == "cx/gpt-5.6-sol"
    assert settings.router_model == "cx/gpt-5.6-sol"
    assert settings.llm_provider == "openai_compatible"
    assert settings.router_llm_provider == "openai_compatible"


def test_gateway_health_requires_configured_model() -> None:
    status = _gateway_status(
        {"reachable": True, "models": ["cx/gpt-5.4"]},
        model="cx/gpt-5.6-sol",
        base_url="http://localhost:20128/v1",
        provider="openai_compatible",
    )
    assert status["provider"] == "9router"
    assert status["reachable"] is False
    assert status["model_available"] is False


def test_gateway_health_accepts_9router_model() -> None:
    status = _gateway_status(
        {"reachable": True, "models": ["cx/gpt-5.6-sol"]},
        model="cx/gpt-5.6-sol",
        base_url="http://localhost:20128/v1",
        provider="openai_compatible",
    )
    assert status["reachable"] is True


def test_memory_health_exposes_pending_retry_without_hiding_error(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    with connect(db_path) as connection:
        connection.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) "
            "VALUES ('c1', 'test', 'now', 'now')"
        )
        connection.execute(
            """
            INSERT INTO conversation_memory_jobs (
                conversation_id, dirty_through_seq, summary_through_seq,
                status, attempt_count, next_attempt_at, last_error, updated_at
            ) VALUES ('c1', 4, 2, 'pending', 1, 'later',
                      '429 usage limit reached', 'now')
            """
        )
        connection.execute(
            """
            INSERT INTO conversation_memory_l3_outbox (
                conversation_id, source_turn_seq, operations_json, status,
                attempt_count, next_attempt_at, last_error,
                created_at, updated_at, delivered_at
            ) VALUES (
                'c1', 2, '[]', 'pending', 1, 'l3-later',
                'temporary L3 write failure', 'now', 'now', NULL
            )
            """
        )

    status = _memory_status(db_path)

    assert status["pending_conversations"] == 1
    assert status["retrying_conversations"] == 1
    assert status["last_error"] == "429 usage limit reached"
    assert status["l3_outbox"] == {
        "pending_operations": 1,
        "pending_conversations": 1,
        "retrying_operations": 1,
        "last_error": "temporary L3 write failure",
        "next_attempt_at": "l3-later",
        "last_error_at": "now",
        "conversation_id": "c1",
        "source_turn_seq": 2,
    }
