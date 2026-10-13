"""Durable fact checkpoints never overwrite a newer source/revision."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from app.db.sqlite import connect, init_db
from app.services import fact_extraction_service as service
from app.services.indexing_service import IndexingService


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    db = tmp_path / "app.db"
    init_db(db)
    source = tmp_path / "amounts.txt"
    source.write_text("Alpha revenue was 45 million.", encoding="utf-8")
    doc = IndexingService(db).index_file(source_path=str(source))
    with connect(db) as c:
        revision = c.execute("SELECT active_revision_id FROM documents WHERE id=?", (doc.id,)).fetchone()[0]
        c.execute("DELETE FROM chunks WHERE document_id=?", (doc.id,))
        for index, entity in enumerate(("Alpha", "Beta")):
            c.execute("INSERT INTO chunks (id,document_id,chunk_index,content,source_path,filename,revision_id) VALUES (?,?,?,?,?,?,?)",
                      (f"chunk-{index}", doc.id, index, f"{entity} revenue was 45 million.", str(source), source.name, revision))
    monkeypatch.setattr(service, "_MAX_BATCH_CHUNKS", 1)
    return db, doc.id


class Client:
    def __init__(self, fail_at=None, callback=None):
        self.calls = 0
        self.fail_at = fail_at
        self.callback = callback

    async def stream_chat(self, **kwargs):
        self.calls += 1
        if self.calls == self.fail_at:
            raise TimeoutError("provider failed")
        if self.callback:
            await self.callback(self.calls)
        excerpt = json.loads(kwargs["messages"][-1]["content"].split("\n", 1)[1])
        entity = excerpt["content"].split()[0]
        yield SimpleNamespace(content=json.dumps({"facts": [{"chunk": 0, "entity": entity,
            "attribute": "revenue", "value": "45", "unit": "million", "quote": excerpt["content"]}]}))


def run(corpus, client, **kwargs):
    db, doc = corpus
    return asyncio.run(service.build_document_facts(db, doc, client=client, model="cx/gpt-5.6-sol", **kwargs))


def test_restart_reuses_completed_batches(corpus):
    with pytest.raises(TimeoutError):
        run(corpus, Client(fail_at=2), retry_delay_seconds=0)
    with connect(corpus[0]) as c:
        assert c.execute("SELECT COUNT(*) FROM document_fact_batches").fetchone()[0] == 1
        assert c.execute("SELECT COUNT(*) FROM document_facts").fetchone()[0] == 0
    client = Client()
    assert run(corpus, client)["status"] == "complete"
    assert client.calls == 1
    assert run(corpus, client)["status"] == "skipped"
    assert client.calls == 1


def test_source_change_invalidates_fingerprint(corpus):
    run(corpus, Client())
    with connect(corpus[0]) as c:
        c.execute("UPDATE chunks SET content='Gamma revenue was 45 million.' WHERE id='chunk-0'")
    client = Client()
    assert run(corpus, client)["status"] == "complete"
    assert client.calls == 2
    assert service.FactStore(corpus[0]).facts_for_chunks(["chunk-0"])["chunk-0"][0]["entity"] == "Gamma"


def test_unchanged_quote_does_not_make_old_build_current(corpus):
    run(corpus, Client())
    with connect(corpus[0]) as c:
        c.execute("UPDATE chunks SET content=content || ' Another statement 99.' WHERE id='chunk-1'")
    assert service.FactStore(corpus[0]).facts_for_chunks(["chunk-0"]) == {}


def test_legacy_completion_does_not_skip_new_contract(corpus):
    with connect(corpus[0]) as c:
        revision = c.execute("SELECT active_revision_id FROM documents WHERE id=?", (corpus[1],)).fetchone()[0]
        c.execute("INSERT INTO document_fact_jobs (document_id,revision_id,status,prompt_version,updated_at) VALUES (?,?,'complete','facts-v1','old')",
                  (corpus[1], revision or ""))
    client = Client()
    assert run(corpus, client)["status"] == "complete" and client.calls == 2


def test_model_change_does_not_reuse_old_completion(corpus):
    run(corpus, Client())
    client = Client()
    result = asyncio.run(service.build_document_facts(corpus[0], corpus[1], client=client, model="cx/gpt-5.6-terra"))
    assert result["status"] == "complete" and client.calls == 2


def test_retry_delay_is_enforced(corpus):
    with pytest.raises(TimeoutError):
        run(corpus, Client(fail_at=1))
    client = Client()
    assert run(corpus, client)["status"] == "retry_wait"
    assert client.calls == 0


def test_concurrent_build_is_busy_and_heartbeat_renews(corpus):
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        async def pause(_):
            entered.set()
            await release.wait()
        worker = asyncio.create_task(service.build_document_facts(*corpus, client=Client(callback=pause),
            model="cx/gpt-5.6-sol", lease_seconds=1))
        await asyncio.wait_for(entered.wait(), timeout=2)
        with connect(corpus[0]) as connection:
            initial = connection.execute("SELECT lease_expires_at FROM document_fact_builds").fetchone()[0]
        # Observe renewal beyond the original expiry rather than assume that
        # a 120 ms lease survives CPU-heavy full-suite scheduling on this Mac.
        async def renewed():
            while True:
                with connect(corpus[0]) as connection:
                    row = connection.execute("SELECT heartbeat_at,lease_expires_at FROM document_fact_builds").fetchone()
                if row["heartbeat_at"] > initial and row["lease_expires_at"] > row["heartbeat_at"]:
                    return
                await asyncio.sleep(0.02)
        await asyncio.wait_for(renewed(), timeout=5)
        rival = Client()
        result = await service.build_document_facts(*corpus, client=rival, model="cx/gpt-5.6-sol", force=True)
        assert result["status"] == "busy" and rival.calls == 0
        release.set()
        assert (await worker)["status"] == "complete"
    asyncio.run(scenario())


def test_expired_lease_is_recovered(corpus):
    with pytest.raises(TimeoutError):
        run(corpus, Client(fail_at=2), retry_delay_seconds=0)
    with connect(corpus[0]) as c:
        c.execute("UPDATE document_fact_builds SET status='running',lease_owner='dead',lease_expires_at='2000-01-01',next_retry_at=NULL")
    client = Client()
    assert run(corpus, client)["status"] == "complete" and client.calls == 1


@pytest.mark.parametrize("change_revision", [False, True])
def test_change_during_build_preserves_previous_facts(corpus, change_revision):
    run(corpus, Client())
    with connect(corpus[0]) as c:
        previous = list(c.execute("SELECT id,quote FROM document_facts"))
    async def change(call):
        if call == 2:
            with connect(corpus[0]) as c:
                if change_revision:
                    c.execute("UPDATE documents SET active_revision_id='newer' WHERE id=?", (corpus[1],))
                else:
                    c.execute("UPDATE chunks SET content='changed source 99' WHERE id='chunk-0'")
    with pytest.raises(RuntimeError, match="activation refused"):
        run(corpus, Client(callback=change), force=True)
    with connect(corpus[0]) as c:
        assert list(c.execute("SELECT id,quote FROM document_facts")) == previous
    exposed = service.FactStore(corpus[0]).facts_for_chunks(["chunk-0", "chunk-1"])
    assert "chunk-0" not in exposed


def test_multi_value_quote_remains_unverified(corpus):
    fact = service.validate_fact({"entity": "Alpha", "attribute": "revenue", "value": "45",
        "quote": "Alpha revenue was 45 million in 2024."}, "Alpha revenue was 45 million in 2024.")
    assert fact.validation_status == "unverified"
    fact.chunk_id = "chunk-0"
    revision, _ = service.numeric_chunks(*corpus)
    service.FactStore(corpus[0]).replace(corpus[1], revision, service.ExtractionResult([fact]), model="cx/gpt-5.6-sol")
    assert service.FactStore(corpus[0]).facts_for_chunks(["chunk-0"]) == {}
