import sqlite3

from app.db.sqlite import connect, init_db


def test_connections_enforce_foreign_keys_and_use_busy_timeout(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)

    with connect(db_path) as connection:
        foreign_keys = connection.execute("PRAGMA foreign_keys").fetchone()[0]
        busy_timeout = connection.execute("PRAGMA busy_timeout").fetchone()[0]

    assert foreign_keys == 1
    assert busy_timeout == 5000


def test_conversation_delete_cascades_messages(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)

    with connect(db_path) as connection:
        connection.execute(
            "INSERT INTO conversations (id, title, mode, created_at, updated_at) "
            "VALUES ('c1', 'test', 'chat', 'now', 'now')"
        )
        connection.execute(
            "INSERT INTO messages (id, conversation_id, role, content, created_at) "
            "VALUES ('m1', 'c1', 'user', 'hello', 'now')"
        )
        connection.execute("DELETE FROM conversations WHERE id = 'c1'")
        count = connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0]

    assert count == 0


def test_message_search_tracks_insert_and_cascade_delete(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)

    with connect(db_path) as connection:
        connection.execute(
            "INSERT INTO conversations (id, title, mode, created_at, updated_at) "
            "VALUES ('c1', 'test', 'chat', 'now', 'now')"
        )
        connection.execute(
            "INSERT INTO messages (id, conversation_id, role, content, created_at) "
            "VALUES ('m1', 'c1', 'user', 'ASPIRE benchmark discussion', 'now')"
        )
        hit = connection.execute(
            "SELECT message_id FROM message_search "
            "WHERE conversation_id = ? AND message_search MATCH ?",
            ("c1", "ASPIRE"),
        ).fetchone()
        assert hit[0] == "m1"

        connection.execute(
            "UPDATE messages SET content = 'WhiSER architecture discussion' "
            "WHERE id = 'm1'"
        )
        old_hit = connection.execute(
            "SELECT message_id FROM message_search "
            "WHERE conversation_id = ? AND message_search MATCH ?",
            ("c1", "ASPIRE"),
        ).fetchone()
        new_hit = connection.execute(
            "SELECT message_id FROM message_search "
            "WHERE conversation_id = ? AND message_search MATCH ?",
            ("c1", "WhiSER"),
        ).fetchone()
        assert old_hit is None
        assert new_hit[0] == "m1"

        connection.execute("DELETE FROM conversations WHERE id = 'c1'")
        remaining = connection.execute(
            "SELECT COUNT(*) FROM message_search WHERE message_id = 'm1'"
        ).fetchone()[0]

    assert remaining == 0


def test_init_db_rebuilds_legacy_fts_schema_and_backfills_messages(tmp_path) -> None:
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            INSERT INTO conversations (id, title, created_at, updated_at)
            VALUES ('legacy-c', 'Legacy', 'now', 'now');
            INSERT INTO messages (id, conversation_id, role, content, created_at)
            VALUES ('legacy-m', 'legacy-c', 'user', 'ASPIRE legacy context', 'now');
            CREATE VIRTUAL TABLE message_search USING fts5(
                message_id UNINDEXED,
                conversation_id UNINDEXED,
                content
            );
            """
        )

    init_db(db_path)

    with connect(db_path) as connection:
        columns = {
            str(row[1])
            for row in connection.execute(
                "PRAGMA table_info(message_search)"
            ).fetchall()
        }
        hit = connection.execute(
            """
            SELECT message_id, role FROM message_search
            WHERE conversation_id = 'legacy-c' AND message_search MATCH 'ASPIRE'
            """
        ).fetchone()
    assert columns == {"message_id", "conversation_id", "role", "content"}
    assert dict(hit) == {"message_id": "legacy-m", "role": "user"}


def test_init_db_reconciles_stale_compatible_fts_rows(tmp_path) -> None:
    db_path = tmp_path / "stale.db"
    init_db(db_path)
    with connect(db_path) as connection:
        connection.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) "
            "VALUES ('c1', 'test', 'now', 'now')"
        )
        connection.execute(
            "INSERT INTO messages (id, conversation_id, role, content, created_at) "
            "VALUES ('m1', 'c1', 'user', 'ASPIRE old text', 'now')"
        )
        connection.execute("DROP TRIGGER messages_search_update")
        connection.execute(
            "UPDATE messages SET content = 'WhiSER canonical text' WHERE id = 'm1'"
        )

    init_db(db_path)

    with connect(db_path) as connection:
        old_hit = connection.execute(
            "SELECT message_id FROM message_search WHERE message_search MATCH 'ASPIRE'"
        ).fetchone()
        new_hit = connection.execute(
            "SELECT message_id FROM message_search WHERE message_search MATCH 'WhiSER'"
        ).fetchone()
    assert old_hit is None
    assert new_hit[0] == "m1"


def test_memory_queue_and_l3_rows_cascade_with_conversation(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)

    with connect(db_path) as connection:
        connection.execute(
            "INSERT INTO conversations (id, title, mode, created_at, updated_at) "
            "VALUES ('c1', 'test', 'chat', 'now', 'now')"
        )
        connection.execute(
            """
            INSERT INTO conversation_memory_turns (
                conversation_id, turn_seq, user_text, assistant_text, completed_at
            ) VALUES ('c1', 1, 'user', 'assistant', 'now')
            """
        )
        connection.execute(
            """
            INSERT INTO conversation_memory_jobs (
                conversation_id, dirty_through_seq, summary_through_seq,
                status, updated_at
            ) VALUES ('c1', 1, 0, 'pending', 'now')
            """
        )
        connection.execute(
            """
            INSERT INTO conversation_memory_l3_outbox (
                conversation_id, source_turn_seq, operations_json,
                status, created_at, updated_at
            ) VALUES ('c1', 1, '[]', 'pending', 'now', 'now')
            """
        )
        connection.execute(
            """
            INSERT INTO memory_items (
                id, scope, conversation_id, kind, memory_key, content, status,
                confidence, valid_from, created_at, updated_at
            ) VALUES (
                'l3', 'conversation', 'c1', 'episodic', 'decision', 'remember',
                'active', 1.0, 'now', 'now', 'now'
            )
            """
        )
        connection.execute("DELETE FROM conversations WHERE id = 'c1'")
        counts = [
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "conversation_memory_turns",
                "conversation_memory_jobs",
                "conversation_memory_l3_outbox",
                "memory_items",
            )
        ]

    assert counts == [0, 0, 0, 0]
