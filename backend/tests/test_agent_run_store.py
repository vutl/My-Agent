from app.db.sqlite import connect, init_db
from app.services.agent_run_store import AgentRunStore
from app.services.chat_history import ChatHistory
from app.services.debug_trace_service import DebugTraceRecorder


def _seed_graph_fingerprint_data(db_path) -> None:
    with connect(db_path) as connection:
        connection.execute(
            """INSERT INTO indexed_folders
               (id, folder_path, recursive, file_types, created_at, updated_at)
               VALUES ('folder', '/papers', 0, '[]', 'now', 'now')"""
        )
        connection.executemany(
            """INSERT INTO documents
               (id, folder_id, source_path, filename, file_type, content_hash,
                modified_at, indexed_at, chunk_count)
               VALUES (?, 'folder', ?, ?, 'pdf', ?, 'now', 'now', 0)""",
            [
                ("doc-a", "/papers/a.pdf", "a.pdf", "document-a"),
                ("doc-b", "/papers/b.pdf", "b.pdf", "document-b"),
            ],
        )
        connection.execute(
            """INSERT INTO collections
               (id, name, type, scope_type, created_at, updated_at)
               VALUES ('collection-a', 'A', 'manual', 'manual', 'now', 'now')"""
        )
        connection.execute(
            """INSERT INTO collection_documents
               (collection_id, document_id, added_at)
               VALUES ('collection-a', 'doc-a', 'now')"""
        )
        connection.executemany(
            """INSERT INTO lightrag_chunk_parent_provenance (
                   lightrag_chunk_id, parent_chunk_id, document_id,
                   lightrag_full_doc_id, lightrag_file_path,
                   lightrag_chunk_order_index, parent_order_index,
                   content_hash, parent_content_hash,
                   document_char_start, document_char_end,
                   canonical_method, mapping_method, mapping_score, mapped_at
               )
               VALUES (?, ?, ?, ?, ?, 0, 0, ?, ?, 0, 100, ?, ?, ?, ?)""",
            [
                (
                    "graph-a",
                    "parent-a",
                    "doc-a",
                    "doc-a",
                    "/papers/a.pdf",
                    "chunk-hash-a",
                    "parent-hash-a",
                    "full_doc_id",
                    "exact_span",
                    1.0,
                    "2026-01-01T00:00:00+00:00",
                ),
                (
                    "graph-b",
                    "parent-b",
                    "doc-b",
                    "doc-b",
                    "/papers/b.pdf",
                    "chunk-hash-b",
                    "parent-hash-b",
                    "full_doc_id",
                    "exact_span",
                    1.0,
                    "2026-01-01T00:00:00+00:00",
                ),
            ],
        )


def test_agent_run_store_records_run_and_tool_call(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    history = ChatHistory(db_path)
    store = AgentRunStore(db_path)

    conversation_id = history.ensure_conversation(None, "Search local docs")
    message = history.save_message(
        conversation_id=conversation_id,
        role="user",
        content="Search local docs",
        model="qwen3.5:4b",
    )
    run = store.create_run(
        conversation_id=conversation_id,
        user_message_id=message.id,
        mode="research",
    )
    tool_call = store.record_tool_call(
        run_id=run.id,
        tool_name="search_local_docs",
        input_payload={"query": "docs"},
        output_payload={"documents": []},
    )
    store.update_plan(run.id, ["Search indexed docs"])
    store.complete_run(run.id, "No docs found.")

    saved = store.get_run(run.id)

    assert saved is not None
    assert saved["status"] == "completed"
    assert saved["plan"] == ["Search indexed docs"]
    assert saved["tool_calls"][0]["id"] == tool_call.id
    assert saved["tool_calls"][0]["input"] == {"query": "docs"}
    assert "debug_trace" not in saved
    assert store.get_debug_trace(run.id) is None


def test_debug_trace_is_redacted_bounded_separate_and_fk_cascaded(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    history = ChatHistory(db_path)
    store = AgentRunStore(db_path)
    conversation_id = history.ensure_conversation(None, "trace")
    run = store.create_run(
        conversation_id=conversation_id,
        user_message_id=None,
        mode="research",
    )
    secret = "sk-this-is-a-very-secret-token-123456"
    recorder = DebugTraceRecorder(
        store=store,
        run_id=run.id,
        enabled=True,
        max_bytes=8192,
        retention_hours=72,
        max_runs=25,
        exact_secrets=(secret,),
    )
    recorder.record_rewrite(
        {
            "authorization": f"Bearer {secret}",
            "prompt": (
                f"key={secret}\npath: /Users/example/private/paper.pdf\n"
                + "đây là prompt dài " * 2000
            ),
            "pem": "-----BEGIN PRIVATE KEY-----\nabc123\n-----END PRIVATE KEY-----",
        }
    )
    recorder.outcome("completed")

    trace = store.get_debug_trace(run.id)
    assert trace is not None
    raw = str(trace["payload"])
    assert secret not in raw
    assert "/Users/example/private" not in raw
    assert "BEGIN PRIVATE KEY" not in raw
    assert trace["size_bytes"] <= 8192
    assert trace["redaction_count"] >= 3
    assert trace["truncated"] is True
    assert "debug_trace" not in store.get_run(run.id)

    with connect(db_path) as connection:
        connection.execute("DELETE FROM agent_runs WHERE id = ?", (run.id,))
    assert store.get_debug_trace(run.id) is None


def test_debug_trace_retention_keeps_only_newest_configured_runs(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    history = ChatHistory(db_path)
    store = AgentRunStore(db_path)
    conversation_id = history.ensure_conversation(None, "trace retention")
    run_ids: list[str] = []
    for index in range(3):
        run = store.create_run(
            conversation_id=conversation_id,
            user_message_id=None,
            mode="research",
        )
        run_ids.append(run.id)
        recorder = DebugTraceRecorder(
            store=store,
            run_id=run.id,
            enabled=True,
            max_bytes=8192,
            max_runs=2,
        )
        recorder.record_route({"index": index})

    assert store.get_debug_trace(run_ids[0]) is None
    assert store.get_debug_trace(run_ids[1]) is not None
    assert store.get_debug_trace(run_ids[2]) is not None


def test_debug_trace_startup_purge_removes_expired_and_excess_rows(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    history = ChatHistory(db_path)
    store = AgentRunStore(db_path)
    conversation_id = history.ensure_conversation(None, "trace startup purge")
    run_ids: list[str] = []
    for index in range(3):
        run = store.create_run(
            conversation_id=conversation_id,
            user_message_id=None,
            mode="research",
        )
        run_ids.append(run.id)
        DebugTraceRecorder(
            store=store,
            run_id=run.id,
            enabled=True,
            max_bytes=8192,
            max_runs=10,
        ).record_route({"index": index})

    with connect(db_path) as connection:
        connection.execute(
            "UPDATE agent_run_debug_traces SET expires_at = ? WHERE run_id = ?",
            ("2000-01-01T00:00:00+00:00", run_ids[0]),
        )

    removed = store.purge_debug_traces(max_runs=1)

    assert removed == 2
    with connect(db_path) as connection:
        rows = connection.execute(
            "SELECT run_id FROM agent_run_debug_traces ORDER BY created_at DESC"
        ).fetchall()
    assert [row["run_id"] for row in rows] == [run_ids[2]]


def test_last_source_cache_ignores_failed_and_invalid_runs(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    history = ChatHistory(db_path)
    store = AgentRunStore(db_path)
    conversation_id = history.ensure_conversation(None, "ASPIRE")

    completed = store.create_run(
        conversation_id=conversation_id,
        user_message_id=None,
        mode="research",
    )
    valid_payload = {
        "documents": [{"document_id": "doc-aspire", "content": "ASPIRE"}],
        "evidence_validation": {"valid": True},
    }
    store.record_tool_call(
        run_id=completed.id,
        tool_name="search_local_docs",
        input_payload={"query": "ASPIRE"},
        output_payload=valid_payload,
    )
    store.complete_run(completed.id, "grounded")

    failed = store.create_run(
        conversation_id=conversation_id,
        user_message_id=None,
        mode="research",
    )
    store.record_tool_call(
        run_id=failed.id,
        tool_name="search_local_docs",
        input_payload={"query": "wrong"},
        output_payload={
            "documents": [{"document_id": "doc-wrong"}],
            "evidence_validation": {"valid": True},
        },
    )
    store.fail_run(failed.id, "generation failed")

    invalid = store.create_run(
        conversation_id=conversation_id,
        user_message_id=None,
        mode="research",
    )
    store.record_tool_call(
        run_id=invalid.id,
        tool_name="search_local_docs",
        input_payload={"query": "missing evidence"},
        output_payload={
            "documents": [{"document_id": "doc-invalid"}],
            "evidence_validation": {"valid": False},
        },
    )
    store.complete_run(invalid.id, "refusal")

    assert store.latest_retrieved_document_ids(conversation_id) == ["doc-aspire"]
    assert store.latest_retrieval_output(conversation_id) == valid_payload


def test_cancel_and_stale_run_lifecycle(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    history = ChatHistory(db_path)
    store = AgentRunStore(db_path)
    conversation_id = history.ensure_conversation(None, "hello")

    cancelled = store.create_run(
        conversation_id=conversation_id,
        user_message_id=None,
        mode="research",
    )
    store.cancel_run(cancelled.id)
    assert store.get_run(cancelled.id)["status"] == "cancelled"

    stale = store.create_run(
        conversation_id=conversation_id,
        user_message_id=None,
        mode="research",
    )
    with connect(db_path) as connection:
        connection.execute(
            "UPDATE agent_runs SET started_at = ? WHERE id = ?",
            ("2000-01-01T00:00:00+00:00", stale.id),
        )
    assert store.fail_stale_running_runs(conversation_id=conversation_id) == 1
    saved = store.get_run(stale.id)
    assert saved["status"] == "failed"
    assert saved["error_message"] == "stale_run_recovered"


def test_index_fingerprint_tracks_visual_table_and_retrieval_configuration(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    store = AgentRunStore(db_path)
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
               VALUES ('doc', 'folder', '/papers/a.pdf', 'a.pdf', 'pdf',
                       'content', 'now', 'now', 0)"""
        )
        connection.execute(
            """INSERT INTO document_figures
               (id, document_id, figure_index, visual_summary)
               VALUES ('fig', 'doc', 0, 'old summary')"""
        )
        connection.execute(
            """INSERT INTO document_tables
               (id, document_id, table_index, markdown)
               VALUES ('table', 'doc', 0, '| A |')"""
        )

    original = store.index_fingerprint(
        document_ids=["doc"], configuration={"embedding_model": "model-a"}
    )
    with connect(db_path) as connection:
        connection.execute(
            "UPDATE document_figures SET visual_summary = 'new summary' WHERE id = 'fig'"
        )
    visual_changed = store.index_fingerprint(
        document_ids=["doc"], configuration={"embedding_model": "model-a"}
    )
    model_changed = store.index_fingerprint(
        document_ids=["doc"], configuration={"embedding_model": "model-b"}
    )

    assert visual_changed != original
    assert model_changed != visual_changed


def test_index_fingerprint_tracks_scoped_graph_provenance_without_timestamp_churn(
    tmp_path,
) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    _seed_graph_fingerprint_data(db_path)
    store = AgentRunStore(db_path)

    original = store.index_fingerprint(document_ids=["doc-a"])
    with connect(db_path) as connection:
        connection.execute(
            """UPDATE lightrag_chunk_parent_provenance
               SET mapped_at = '2026-02-02T00:00:00+00:00'
               WHERE document_id = 'doc-a'"""
        )
    assert store.index_fingerprint(document_ids=["doc-a"]) == original

    with connect(db_path) as connection:
        connection.execute(
            """UPDATE lightrag_chunk_parent_provenance
               SET mapping_score = 0.5
               WHERE document_id = 'doc-b'"""
        )
    assert store.index_fingerprint(document_ids=["doc-a"]) == original

    tracked_updates = (
        ("content_hash", "changed-chunk-hash"),
        ("parent_content_hash", "changed-parent-hash"),
        ("overlap_chars", 321),
        ("canonical_method", "file_path"),
        ("mapping_method", "overlap"),
        ("mapping_score", 0.75),
    )
    for column, value in tracked_updates:
        with connect(db_path) as connection:
            connection.execute(
                f"""UPDATE lightrag_chunk_parent_provenance
                    SET {column} = ?
                    WHERE document_id = 'doc-a'""",
                (value,),
            )
        assert store.index_fingerprint(document_ids=["doc-a"]) != original
        with connect(db_path) as connection:
            connection.execute(
                """DELETE FROM lightrag_chunk_parent_provenance
                   WHERE document_id = 'doc-a'"""
            )
        _insert_original_doc_a_provenance(db_path)

    with connect(db_path) as connection:
        connection.execute(
            """INSERT INTO lightrag_chunk_parent_provenance (
                   lightrag_chunk_id, parent_chunk_id, document_id,
                   lightrag_full_doc_id, lightrag_file_path,
                   lightrag_chunk_order_index, parent_order_index,
                   content_hash, parent_content_hash,
                   document_char_start, document_char_end,
                   canonical_method, mapping_method, mapping_score, mapped_at
               )
               SELECT
                   'graph-a-2', 'parent-a-2', document_id,
                   lightrag_full_doc_id, lightrag_file_path,
                   1, 1, content_hash, parent_content_hash,
                   100, 200, canonical_method, mapping_method,
                   mapping_score, '2026-03-03T00:00:00+00:00'
               FROM lightrag_chunk_parent_provenance
               WHERE document_id = 'doc-a'"""
        )
    assert store.index_fingerprint(document_ids=["doc-a"]) != original


def test_index_fingerprint_intersects_collection_and_document_scope(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    _seed_graph_fingerprint_data(db_path)
    store = AgentRunStore(db_path)

    collection_fingerprint = store.index_fingerprint(collection_id="collection-a")
    empty_intersection = store.index_fingerprint(
        collection_id="collection-a",
        document_ids=["doc-b"],
    )
    with connect(db_path) as connection:
        connection.execute(
            """UPDATE lightrag_chunk_parent_provenance
               SET parent_content_hash = 'doc-b-changed'
               WHERE document_id = 'doc-b'"""
        )
        connection.execute(
            "UPDATE documents SET content_hash = 'doc-b-changed' WHERE id = 'doc-b'"
        )

    assert store.index_fingerprint(collection_id="collection-a") == collection_fingerprint
    assert (
        store.index_fingerprint(
            collection_id="collection-a",
            document_ids=["doc-b"],
        )
        == empty_intersection
    )

    with connect(db_path) as connection:
        connection.execute(
            """UPDATE lightrag_chunk_parent_provenance
               SET parent_content_hash = 'doc-a-changed'
               WHERE document_id = 'doc-a'"""
        )
    assert store.index_fingerprint(collection_id="collection-a") != collection_fingerprint


def _insert_original_doc_a_provenance(db_path) -> None:
    with connect(db_path) as connection:
        connection.execute(
            """INSERT INTO lightrag_chunk_parent_provenance (
                   lightrag_chunk_id, parent_chunk_id, document_id,
                   lightrag_full_doc_id, lightrag_file_path,
                   lightrag_chunk_order_index, parent_order_index,
                   content_hash, parent_content_hash,
                   document_char_start, document_char_end,
                   canonical_method, mapping_method, mapping_score, mapped_at
               )
               VALUES (
                   'graph-a', 'parent-a', 'doc-a', 'doc-a', '/papers/a.pdf',
                   0, 0, 'chunk-hash-a', 'parent-hash-a', 0, 100,
                   'full_doc_id', 'exact_span', 1.0,
                   '2026-02-02T00:00:00+00:00'
               )"""
        )


def test_last_source_reuse_requires_same_collection_mode_and_fingerprint(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    history = ChatHistory(db_path)
    store = AgentRunStore(db_path)
    conversation_id = history.ensure_conversation(None, "paper")
    run = store.create_run(conversation_id=conversation_id, user_message_id=None, mode="research")
    payload = {
        "documents": [{"document_id": "doc-a", "content": "evidence"}],
        "evidence_validation": {"valid": True},
        "index_fingerprint": "fingerprint-a",
    }
    store.record_tool_call(
        run_id=run.id,
        tool_name="search_local_docs",
        input_payload={"collection_id": "collection-a", "retrieval_mode": "auto"},
        output_payload=payload,
    )
    store.complete_run(run.id, "answer")

    assert store.latest_retrieval_output(
        conversation_id,
        expected_collection_id="collection-a",
        expected_retrieval_mode="auto",
        expected_index_fingerprint="fingerprint-a",
    ) == payload
    assert store.latest_retrieval_output(
        conversation_id,
        expected_collection_id="collection-b",
        expected_retrieval_mode="auto",
        expected_index_fingerprint="fingerprint-a",
    ) is None
    assert store.latest_retrieval_output(
        conversation_id,
        expected_collection_id="collection-a",
        expected_retrieval_mode="hybrid",
        expected_index_fingerprint="fingerprint-a",
    ) is None
    assert store.latest_retrieval_output(
        conversation_id,
        expected_collection_id="collection-a",
        expected_retrieval_mode="auto",
        expected_index_fingerprint="changed",
    ) is None
