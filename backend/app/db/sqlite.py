from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import sqlite3


SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'chat',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    summary TEXT,
    pinned INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    model TEXT,
    created_at TEXT NOT NULL,
    token_count INTEGER,
    metadata_json TEXT,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_created
ON messages(conversation_id, created_at);

-- Durable L2 event queue.  Raw chat messages remain L0's source of truth;
-- these rows pair a completed user/assistant exchange with a monotonic cursor
-- so background consolidation can be retried after a crash without guessing
-- which messages belong together.
CREATE TABLE IF NOT EXISTS conversation_memory_turns (
    conversation_id TEXT NOT NULL,
    turn_seq INTEGER NOT NULL,
    user_message_id TEXT,
    assistant_message_id TEXT,
    user_text TEXT NOT NULL,
    assistant_text TEXT NOT NULL,
    working_topic TEXT,
    working_filenames_json TEXT,
    completed_at TEXT NOT NULL,
    PRIMARY KEY (conversation_id, turn_seq),
    UNIQUE (assistant_message_id),
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
    FOREIGN KEY (user_message_id) REFERENCES messages(id) ON DELETE SET NULL,
    FOREIGN KEY (assistant_message_id) REFERENCES messages(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_conversation_memory_turns_pending
ON conversation_memory_turns(conversation_id, turn_seq);

CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_turns_user_message
ON conversation_memory_turns(user_message_id)
WHERE user_message_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_memory_turns_conversation_completed
ON conversation_memory_turns(conversation_id, completed_at);

-- One durable dirty cursor per conversation.  A worker may disappear at any
-- point; dirty_through_seq > summary_through_seq is sufficient for startup
-- recovery and does not depend on an in-memory asyncio task surviving.
CREATE TABLE IF NOT EXISTS conversation_memory_jobs (
    conversation_id TEXT PRIMARY KEY,
    dirty_through_seq INTEGER NOT NULL DEFAULT 0,
    summary_through_seq INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    attempt_count INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TEXT,
    last_error TEXT,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_conversation_memory_jobs_pending
ON conversation_memory_jobs(status, next_attempt_at, updated_at);

-- L2 summary commit and extracted L3 operations share one transaction.  The
-- callback that materializes those operations into memory_items may fail or be
-- interrupted independently, so the outbox remains pending until explicitly
-- acknowledged and is replayed after restart.
CREATE TABLE IF NOT EXISTS conversation_memory_l3_outbox (
    conversation_id TEXT NOT NULL,
    source_turn_seq INTEGER NOT NULL,
    operations_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempt_count INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TEXT,
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    delivered_at TEXT,
    PRIMARY KEY (conversation_id, source_turn_seq),
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_conversation_memory_l3_outbox_pending
ON conversation_memory_l3_outbox(status, next_attempt_at, updated_at);

-- Structured L3 memory.  Updates are append-only versions: the old active row
-- is closed/superseded and the replacement points back to it.
CREATE TABLE IF NOT EXISTS memory_items (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL DEFAULT 'user',
    conversation_id TEXT,
    kind TEXT NOT NULL,
    memory_key TEXT NOT NULL,
    content TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    confidence REAL NOT NULL DEFAULT 1.0,
    valid_from TEXT NOT NULL,
    valid_to TEXT,
    source_conversation_id TEXT,
    source_turn_seq INTEGER,
    supersedes_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata_json TEXT,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
    FOREIGN KEY (source_conversation_id) REFERENCES conversations(id) ON DELETE SET NULL,
    FOREIGN KEY (supersedes_id) REFERENCES memory_items(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_items_active_scope
ON memory_items(scope, conversation_id, status, kind, updated_at);

CREATE INDEX IF NOT EXISTS idx_memory_items_key_history
ON memory_items(scope, conversation_id, memory_key, created_at);

-- Durable receipts make L3 extraction replay-safe.  In particular, an old
-- ``forget`` operation must not close a newer value if a worker is retried
-- after the original operation already committed.
CREATE TABLE IF NOT EXISTS memory_operation_receipts (
    source_conversation_id TEXT NOT NULL,
    source_turn_seq INTEGER NOT NULL,
    operation_fingerprint TEXT NOT NULL,
    action TEXT NOT NULL,
    result_item_id TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (
        source_conversation_id,
        source_turn_seq,
        operation_fingerprint
    ),
    FOREIGN KEY (source_conversation_id)
        REFERENCES conversations(id) ON DELETE CASCADE,
    FOREIGN KEY (result_item_id)
        REFERENCES memory_items(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_operation_receipts_result
ON memory_operation_receipts(result_item_id);

-- FTS is an index, never the source of truth.  The triggers keep new chat
-- messages searchable while the migration below backfills existing histories.
CREATE VIRTUAL TABLE IF NOT EXISTS message_search USING fts5(
    message_id UNINDEXED,
    conversation_id UNINDEXED,
    role UNINDEXED,
    content,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS messages_search_insert
AFTER INSERT ON messages BEGIN
    INSERT INTO message_search(message_id, conversation_id, role, content)
    VALUES (new.id, new.conversation_id, new.role, new.content);
END;

CREATE TRIGGER IF NOT EXISTS messages_search_delete
AFTER DELETE ON messages BEGIN
    DELETE FROM message_search WHERE message_id = old.id;
END;

CREATE TRIGGER IF NOT EXISTS messages_search_update
AFTER UPDATE OF conversation_id, role, content ON messages BEGIN
    DELETE FROM message_search WHERE message_id = old.id;
    INSERT INTO message_search(message_id, conversation_id, role, content)
    VALUES (new.id, new.conversation_id, new.role, new.content);
END;

CREATE TABLE IF NOT EXISTS indexed_folders (
    id TEXT PRIMARY KEY,
    folder_path TEXT NOT NULL UNIQUE,
    recursive INTEGER NOT NULL,
    file_types TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS approved_folders (
    id TEXT PRIMARY KEY,
    folder_path TEXT NOT NULL UNIQUE,
    access_level TEXT NOT NULL DEFAULT 'read_index',
    recursive INTEGER NOT NULL DEFAULT 0,
    watch_enabled INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS files (
    id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL UNIQUE,
    filename TEXT NOT NULL,
    extension TEXT,
    mime_type TEXT,
    size_bytes INTEGER,
    created_at TEXT,
    modified_at TEXT,
    content_hash TEXT,
    parent_folder TEXT,
    approved INTEGER NOT NULL DEFAULT 0,
    last_seen_at TEXT,
    is_supported INTEGER NOT NULL DEFAULT 0,
    risk_level TEXT NOT NULL DEFAULT 'normal',
    metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_files_parent_folder
ON files(parent_folder);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    file_id TEXT,
    folder_id TEXT NOT NULL,
    source_path TEXT NOT NULL UNIQUE,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    title TEXT,
    doc_type TEXT,
    language TEXT,
    content_hash TEXT NOT NULL,
    modified_at TEXT NOT NULL,
    indexed_at TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,
    table_count INTEGER NOT NULL DEFAULT 0,
    figure_count INTEGER NOT NULL DEFAULT 0,
    parse_status TEXT NOT NULL DEFAULT 'parsed',
    index_status TEXT NOT NULL DEFAULT 'indexed',
    parser_name TEXT,
    parser_version TEXT,
    page_count INTEGER,
    error_message TEXT,
    updated_at TEXT,
    metadata_json TEXT,
    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE SET NULL,
    FOREIGN KEY (folder_id) REFERENCES indexed_folders(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_documents_folder_id
ON documents(folder_id);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    file_id TEXT,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    source_path TEXT NOT NULL,
    filename TEXT NOT NULL,
    chunk_type TEXT NOT NULL DEFAULT 'text',
    heading_path_json TEXT,
    token_count INTEGER,
    char_count INTEGER,
    order_index INTEGER,
    parent_chunk_id TEXT,
    page_number INTEGER,
    lance_table TEXT,
    lance_id TEXT,
    created_at TEXT,
    metadata_json TEXT,
    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE SET NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id
ON chunks(document_id);

-- Durable bridge from LightRAG's storage-owned chunk IDs back to canonical
-- parent passages in SQLite.  parent_chunk_id is intentionally not a foreign
-- key: parent passages are represented by one or more child rows in `chunks`,
-- rather than by a standalone parent row.  Resolvers verify the persisted
-- parent_content_hash against the current child metadata and fail closed when
-- a document has been re-indexed without a provenance rebuild.
CREATE TABLE IF NOT EXISTS lightrag_chunk_parent_provenance (
    lightrag_chunk_id TEXT NOT NULL,
    parent_chunk_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    lightrag_full_doc_id TEXT,
    lightrag_file_path TEXT,
    lightrag_chunk_order_index INTEGER NOT NULL,
    parent_order_index INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    parent_content_hash TEXT NOT NULL,
    overlap_chars INTEGER NOT NULL DEFAULT 0,
    document_char_start INTEGER NOT NULL,
    document_char_end INTEGER NOT NULL,
    canonical_method TEXT NOT NULL,
    mapping_method TEXT NOT NULL,
    mapping_score REAL NOT NULL,
    mapped_at TEXT NOT NULL,
    PRIMARY KEY (lightrag_chunk_id, parent_chunk_id),
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_lightrag_provenance_document
ON lightrag_chunk_parent_provenance(document_id);

CREATE INDEX IF NOT EXISTS idx_lightrag_provenance_parent
ON lightrag_chunk_parent_provenance(document_id, parent_chunk_id);

CREATE TABLE IF NOT EXISTS document_tables (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    file_id TEXT,
    table_index INTEGER NOT NULL,
    page_number INTEGER,
    caption TEXT,
    markdown TEXT,
    row_count INTEGER,
    column_count INTEGER,
    extraction_method TEXT,
    bbox_json TEXT,
    created_at TEXT,
    metadata_json TEXT,
    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE SET NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_document_tables_document_id
ON document_tables(document_id);

CREATE TABLE IF NOT EXISTS document_figures (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    file_id TEXT,
    figure_index INTEGER NOT NULL,
    page_number INTEGER,
    caption TEXT,
    image_path TEXT,
    visual_summary TEXT,
    extraction_method TEXT,
    bbox_json TEXT,
    created_at TEXT,
    metadata_json TEXT,
    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE SET NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_document_figures_document_id
ON document_figures(document_id);

-- Compact, provenance-backed paper summaries.  These are a navigation layer,
-- never an alternative source of truth: every item must resolve to a current
-- canonical chunk/table/figure owned by the same document.
CREATE TABLE IF NOT EXISTS paper_evidence_cards (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL UNIQUE,
    document_content_hash TEXT NOT NULL,
    parser_name TEXT,
    parser_version TEXT,
    schema_version TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    generator_provider TEXT NOT NULL,
    generator_model TEXT NOT NULL,
    status TEXT NOT NULL,
    coverage_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata_json TEXT,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_paper_evidence_cards_status
ON paper_evidence_cards(status, updated_at);

CREATE TABLE IF NOT EXISTS paper_evidence_facets (
    id TEXT PRIMARY KEY,
    card_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    facet TEXT NOT NULL,
    synopsis TEXT NOT NULL,
    status TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.0,
    source_count INTEGER NOT NULL DEFAULT 0,
    facet_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata_json TEXT,
    UNIQUE(card_id, facet),
    FOREIGN KEY (card_id) REFERENCES paper_evidence_cards(id) ON DELETE CASCADE,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_paper_evidence_facets_document
ON paper_evidence_facets(document_id, facet);

CREATE TABLE IF NOT EXISTS paper_evidence_items (
    id TEXT PRIMARY KEY,
    facet_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    claim_text TEXT NOT NULL,
    evidence_refs_json TEXT NOT NULL,
    validation_status TEXT NOT NULL,
    validation_reason TEXT,
    created_at TEXT NOT NULL,
    metadata_json TEXT,
    UNIQUE(facet_id, ordinal),
    FOREIGN KEY (facet_id) REFERENCES paper_evidence_facets(id) ON DELETE CASCADE,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_paper_evidence_items_document
ON paper_evidence_items(document_id, facet_id);

CREATE TABLE IF NOT EXISTS paper_evidence_build_jobs (
    document_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    requested_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    error_message TEXT,
    schema_version TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    generator_model TEXT NOT NULL,
    metadata_json TEXT,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_paper_evidence_build_jobs_status
ON paper_evidence_build_jobs(status, requested_at);

CREATE VIRTUAL TABLE IF NOT EXISTS rag_chunks_fts USING fts5(
    chunk_id UNINDEXED,
    document_id UNINDEXED,
    source_path UNINDEXED,
    filename UNINDEXED,
    content
);

CREATE TABLE IF NOT EXISTS document_cards (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL UNIQUE,
    title_guess TEXT,
    doc_type TEXT,
    language TEXT,
    short_summary TEXT,
    topic_tags_json TEXT,
    project_tags_json TEXT,
    people_json TEXT,
    orgs_json TEXT,
    dates_json TEXT,
    keywords_json TEXT,
    importance_score REAL NOT NULL DEFAULT 0.0,
    confidence REAL NOT NULL DEFAULT 0.0,
    should_deep_index INTEGER NOT NULL DEFAULT 0,
    lance_table TEXT,
    lance_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata_json TEXT,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_document_cards USING fts5(
    document_id UNINDEXED,
    title,
    short_summary,
    topic_tags,
    project_tags,
    keywords
);

CREATE VIRTUAL TABLE IF NOT EXISTS fts_files USING fts5(
    file_id UNINDEXED,
    filename,
    source_path,
    extension
);

CREATE TABLE IF NOT EXISTS collections (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    scope_type TEXT NOT NULL,
    scope_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_collections_scope
ON collections(scope_type, scope_id);

CREATE TABLE IF NOT EXISTS collection_documents (
    collection_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    added_at TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'reference',
    metadata_json TEXT,
    PRIMARY KEY (collection_id, document_id),
    FOREIGN KEY(collection_id) REFERENCES collections(id) ON DELETE CASCADE,
    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_collection_documents_document
ON collection_documents(document_id);

CREATE TABLE IF NOT EXISTS agent_runs (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    user_message_id TEXT,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    plan_json TEXT,
    final_answer TEXT,
    error_message TEXT,
    metadata_json TEXT,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_agent_runs_conversation_started
ON agent_runs(conversation_id, started_at);

CREATE TABLE IF NOT EXISTS agent_run_debug_traces (
    run_id TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL DEFAULT 1,
    payload_json TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    redaction_count INTEGER NOT NULL DEFAULT 0,
    truncated INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES agent_runs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_agent_run_debug_traces_expiry
ON agent_run_debug_traces(expires_at, created_at);

CREATE TABLE IF NOT EXISTS tool_calls (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    input_json TEXT,
    output_json TEXT,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    requires_confirmation INTEGER NOT NULL DEFAULT 0,
    approved INTEGER,
    error_message TEXT,
    FOREIGN KEY (run_id) REFERENCES agent_runs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tool_calls_run_started
ON tool_calls(run_id, started_at);

CREATE TABLE IF NOT EXISTS retrieval_cache (
    id TEXT PRIMARY KEY,
    cache_key TEXT NOT NULL UNIQUE,
    normalized_query TEXT NOT NULL,
    collection_id TEXT,
    focus_document_ids_json TEXT,
    retrieval_mode TEXT NOT NULL,
    index_fingerprint TEXT NOT NULL,
    output_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT,
    hit_count INTEGER NOT NULL DEFAULT 0,
    last_hit_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_retrieval_cache_key
ON retrieval_cache(cache_key);
"""


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        _configure_connection(connection)
        connection.executescript(SCHEMA)
        _migrate_existing_schema(connection)


def _migrate_existing_schema(connection: sqlite3.Connection) -> None:
    _add_missing_columns(
        connection,
        "conversations",
        {
            "mode": "TEXT NOT NULL DEFAULT 'chat'",
            "summary": "TEXT",
            "pinned": "INTEGER NOT NULL DEFAULT 0",
            "metadata_json": "TEXT",
        },
    )
    _add_missing_columns(
        connection,
        "messages",
        {
            "token_count": "INTEGER",
            "metadata_json": "TEXT",
        },
    )
    _add_missing_columns(
        connection,
        "documents",
        {
            "file_id": "TEXT",
            "title": "TEXT",
            "doc_type": "TEXT",
            "language": "TEXT",
            "table_count": "INTEGER NOT NULL DEFAULT 0",
            "figure_count": "INTEGER NOT NULL DEFAULT 0",
            "parse_status": "TEXT NOT NULL DEFAULT 'parsed'",
            "index_status": "TEXT NOT NULL DEFAULT 'indexed'",
            "parser_name": "TEXT",
            "parser_version": "TEXT",
            "page_count": "INTEGER",
            "error_message": "TEXT",
            "updated_at": "TEXT",
            "metadata_json": "TEXT",
        },
    )
    _add_missing_columns(
        connection,
        "chunks",
        {
            "file_id": "TEXT",
            "chunk_type": "TEXT NOT NULL DEFAULT 'text'",
            "heading_path_json": "TEXT",
            "token_count": "INTEGER",
            "char_count": "INTEGER",
            "order_index": "INTEGER",
            "parent_chunk_id": "TEXT",
            "page_number": "INTEGER",
            "lance_table": "TEXT",
            "lance_id": "TEXT",
            "created_at": "TEXT",
            "metadata_json": "TEXT",
        },
    )
    _add_missing_columns(
        connection,
        "lightrag_chunk_parent_provenance",
        {
            "overlap_chars": "INTEGER NOT NULL DEFAULT 0",
        },
    )
    _ensure_message_search_schema(connection)
    # SCHEMA creates the FTS table/triggers for new rows. Existing databases
    # need an idempotent reconciliation: remove orphan/stale index rows first,
    # then backfill anything missing from the immutable L0 source of truth.
    connection.execute(
        """
        DELETE FROM message_search
        WHERE NOT EXISTS (
            SELECT 1 FROM messages
            WHERE messages.id = message_search.message_id
        )
        OR EXISTS (
            SELECT 1 FROM messages
            WHERE messages.id = message_search.message_id
              AND (
                messages.conversation_id IS NOT message_search.conversation_id
                OR messages.role IS NOT message_search.role
                OR messages.content IS NOT message_search.content
              )
        )
        """
    )
    connection.execute(
        """
        INSERT INTO message_search(message_id, conversation_id, role, content)
        SELECT messages.id, messages.conversation_id, messages.role, messages.content
        FROM messages
        WHERE NOT EXISTS (
            SELECT 1 FROM message_search
            WHERE message_search.message_id = messages.id
        )
        """
    )


def _ensure_message_search_schema(connection: sqlite3.Connection) -> None:
    """Repair an intermediate/legacy FTS table before running the backfill.

    SQLite cannot add a column to an FTS5 virtual table.  ``messages`` is the
    source of truth, so rebuilding an incompatible index is both safer and
    simpler than trying to preserve stale index rows.  Trigger recreation is
    included because SQLite accepts trigger definitions that reference a
    missing virtual-table column and would otherwise fail only on the next
    message write.
    """

    columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(message_search)").fetchall()
    }
    expected = {"message_id", "conversation_id", "role", "content"}
    if columns == expected:
        return

    connection.executescript(
        """
        DROP TRIGGER IF EXISTS messages_search_insert;
        DROP TRIGGER IF EXISTS messages_search_delete;
        DROP TRIGGER IF EXISTS messages_search_update;
        DROP TABLE IF EXISTS message_search;

        CREATE VIRTUAL TABLE message_search USING fts5(
            message_id UNINDEXED,
            conversation_id UNINDEXED,
            role UNINDEXED,
            content,
            tokenize = 'unicode61 remove_diacritics 2'
        );

        CREATE TRIGGER messages_search_insert
        AFTER INSERT ON messages BEGIN
            INSERT INTO message_search(message_id, conversation_id, role, content)
            VALUES (new.id, new.conversation_id, new.role, new.content);
        END;

        CREATE TRIGGER messages_search_delete
        AFTER DELETE ON messages BEGIN
            DELETE FROM message_search WHERE message_id = old.id;
        END;

        CREATE TRIGGER messages_search_update
        AFTER UPDATE OF conversation_id, role, content ON messages BEGIN
            DELETE FROM message_search WHERE message_id = old.id;
            INSERT INTO message_search(message_id, conversation_id, role, content)
            VALUES (new.id, new.conversation_id, new.role, new.content);
        END;
        """
    )


def _add_missing_columns(
    connection: sqlite3.Connection,
    table_name: str,
    columns: dict[str, str],
) -> None:
    existing = {
        row[1]
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    for column_name, definition in columns.items():
        if column_name not in existing:
            connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(db_path)
    _configure_connection(connection)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def _configure_connection(connection: sqlite3.Connection) -> None:
    # Foreign-key enforcement is connection-local in SQLite. Without this,
    # cascade declarations in the schema are only documentation and stale
    # chat/index rows can survive parent deletion. A bounded busy wait also
    # makes concurrent SSE/memory writers cooperate with BEGIN IMMEDIATE.
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
