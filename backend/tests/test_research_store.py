"""Real SQLite migration, concurrent writes, stable labels, and file lifetime."""
from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest

from app.core.config import Settings
from app.db.sqlite import SCHEMA, connect, init_db
from app.services.research import ResearchStore, start_research
from app.services.tasks import TaskStore


@pytest.fixture
def env(tmp_path):
    settings = Settings(data_dir=tmp_path)
    init_db(settings.sqlite_db_path)
    task_id = start_research(question="Evidence for a generic question", settings=settings, depth="quick")
    return settings, task_id, ResearchStore(settings.sqlite_db_path, tmp_path)


def test_migrate_existing_tasks(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(SCHEMA.replace("    kind TEXT NOT NULL DEFAULT 'agent',\n", ""))
        connection.execute("INSERT INTO tasks(id,goal,status,created_at,updated_at) VALUES ('old','goal','planning','now','now')")
    init_db(path)
    init_db(path)
    assert TaskStore(path).get("old").kind == "agent"
    assert TaskStore(path).create("Research", kind="research").kind == "research"
    with pytest.raises(ValueError):
        TaskStore(path).create("Research", kind="unknown")


def test_atomic_counters_and_labels(env):
    settings, task_id, store = env
    def write(i):
        store.add_usage(task_id, pages=1, worker_calls=1)
        source = store.source(task_id, dict(url=f"https://example.org/{i}", title=str(i)))
        return store.mark_source(source["id"], text="read evidence " * 30)
    with ThreadPoolExecutor(5) as executor:
        sources = list(executor.map(write, range(12)))
    assert store.get(task_id)["usage"]["pages"] == 12
    assert {x["label"] for x in sources} == {f"W{i}" for i in range(1, 13)}
    for source in sources:
        assert store.mark_source(source["id"], text="updated text " * 30)["label"] == source["label"]
    library = store.source(task_id, dict(url="library:doc", title="doc", kind="library"))
    assert store.mark_source(library["id"], text="library text " * 30)["label"] == "S1"
    assert store.source(task_id, dict(url="https://www.example.org/0#fragment"))["id"] == next(s["id"] for s in sources if s["url"].endswith("/0"))


def test_evidence_unique_files_and_delete(env):
    settings, task_id, store = env
    store.replace_subquestions(task_id, [dict(text="A useful question")])
    subq = store.subquestions(task_id)[0]
    source = store.source(task_id, dict(url="https://example.org/source"))
    card = dict(claim="Claim", quote="quote", numbers=[], locator="page 1", stance="supports", confidence="high")
    assert not store.add_card(task_id, subq["id"], source["id"], 1, card)
    source = store.mark_source(source["id"], text="quote " * 100)
    assert store.text(source) == "quote " * 100
    assert store.add_card(task_id, subq["id"], source["id"], 1, card)
    assert not store.add_card(task_id, subq["id"], source["id"], 1, card)
    unreadable = store.mark_source(store.source(task_id, dict(url="https://example.org/empty"))["id"], text="short")
    assert unreadable["status"] == "unreadable" and not unreadable["label"]
    assert store.counts(task_id)["cards"] == 1
    with connect(store.db_path) as connection:
        row = dict(connection.execute("SELECT * FROM research_sources WHERE id=?", (source["id"],)).fetchone())
        assert "text" not in row
    folder = store.root / task_id
    assert folder.exists()
    store.delete(task_id)
    assert not folder.exists() and not store.get(task_id)
    assert store.sources(task_id) == [] and store.cards(task_id) == []


def test_research_model_defaults_follow_settings():
    settings = Settings(default_model="cx/gpt-6-sol", router_model="cx/gpt-6-luna")
    assert settings.research_lead_model == settings.default_model
    assert settings.research_worker_model == settings.router_model


def test_merge_library_chunks_retains_pages_and_label(env):
    _, task_id, store = env
    source = store.source(task_id, dict(url="library:doc", kind="library"))
    chunks = [f"Document: doc; page {page}\n" + f"Evidence on page {page}. " * 20 for page in (1, 2, 3)]
    first = store.merge_library_chunks(source["id"], chunks[:1])
    merged = store.merge_library_chunks(source["id"], chunks[1:] + chunks[:1])
    assert merged["label"] == first["label"] == "S1"
    assert all(store.text(merged).count(chunk.strip()) == 1 for chunk in chunks)
    assert store.merge_library_chunks(source["id"], chunks) == merged
