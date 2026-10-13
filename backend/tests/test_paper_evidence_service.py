from pathlib import Path
import asyncio
import json
from types import SimpleNamespace

import pytest

from app.api import agent as agent_api
from app.core.config import Settings
from app.db.sqlite import connect, init_db
from app.rag.paper_facets import CORE_PAPER_FACETS, extract_query_facets, requested_paper_facets
from app.services.paper_evidence_builder import PaperEvidenceBuilder, _parse_json_object
from app.services.indexing_service import IndexingService
from app.services.paper_evidence_service import (
    EvidenceFacetDraft,
    EvidenceItemDraft,
    EvidenceRefDraft,
    PaperEvidenceDraft,
    PaperEvidenceService,
    _quote_found,
)


def _seed(db_path: Path, *, document_id: str = "doc-a") -> None:
    init_db(db_path)
    with connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO indexed_folders(id, folder_path, recursive, file_types, created_at, updated_at)
            VALUES ('folder', '/tmp', 0, '[]', 'now', 'now')
            """
        )
        connection.execute(
            """
            INSERT INTO documents(
                id, folder_id, source_path, filename, file_type, title, doc_type,
                language, content_hash, modified_at, indexed_at, chunk_count,
                parser_name, parser_version
            ) VALUES (?, 'folder', ?, ?, 'pdf', ?, 'research_paper', 'en',
                      'hash-a', 'now', 'now', 1, 'docling', 'v9')
            """,
            (document_id, f"/tmp/{document_id}.pdf", f"{document_id}.pdf", f"Paper {document_id}"),
        )
        connection.execute(
            """
            INSERT INTO chunks(
                id, document_id, chunk_index, content, source_path, filename,
                chunk_type, order_index, page_number, created_at, metadata_json
            ) VALUES (?, ?, 0, ?, ?, ?, 'text', 0, 1, 'now', '{}')
            """,
            (
                f"chunk-{document_id}",
                document_id,
                "The paper addresses speech emotion recognition and proposes a compact fusion architecture. "
                "Its main contribution is a provenance-aware fusion method evaluated on a speech dataset.",
                f"/tmp/{document_id}.pdf",
                f"{document_id}.pdf",
            ),
        )
        connection.execute(
            """
            INSERT INTO document_tables(
                id, document_id, table_index, page_number, caption, markdown,
                created_at, metadata_json
            ) VALUES (?, ?, 0, 4, 'Table 1: Main results',
                      '| Model | Accuracy |\n|---|---|\n| Fusion | 80.0% |', 'now', '{}')
            """,
            (f"table-{document_id}", document_id),
        )


def _complete_draft(document_id: str = "doc-a") -> PaperEvidenceDraft:
    chunk = EvidenceRefDraft(
        source_kind="chunk",
        source_id=f"chunk-{document_id}",
        quote="The paper addresses speech emotion recognition",
    )
    table = EvidenceRefDraft(
        source_kind="table",
        source_id=f"table-{document_id}",
        quote="| Fusion | 80.0% |",
    )
    facets = []
    for facet in CORE_PAPER_FACETS:
        if facet == "benchmark_results":
            facets.append(
                EvidenceFacetDraft(
                    facet=facet,
                    synopsis="Fusion reaches 80.0% Accuracy.",
                    confidence=0.9,
                    items=[EvidenceItemDraft("Fusion reaches Accuracy 80.0%.", [table])],
                )
            )
        else:
            facets.append(
                EvidenceFacetDraft(
                    facet=facet,
                    synopsis=f"Canonical {facet} synopsis.",
                    confidence=0.8,
                    items=[
                        EvidenceItemDraft(
                            "The paper addresses speech emotion recognition.",
                            [chunk],
                        )
                    ],
                )
            )
    return PaperEvidenceDraft(
        document_id=document_id,
        facets=facets,
        generator_model="cx/gpt-5.6-sol",
    )


def test_shared_facet_taxonomy_covers_core_vietnamese_and_compare_defaults() -> None:
    assert extract_query_facets("kiến trúc, dataset và kết quả") == [
        "architecture",
        "dataset_setup",
        "benchmark_results",
    ]
    assert requested_paper_facets(
        "so sánh hai paper", answer_intent="compare", focused_document_count=2
    ) == list(CORE_PAPER_FACETS)
    assert requested_paper_facets("Bài này làm gì?") == ["task", "contributions"]
    assert requested_paper_facets("Explain its contribution") == [
        "contributions",
        "task",
    ]
    assert requested_paper_facets("What dataset is used?") == ["dataset_setup"]
    assert extract_query_facets("Compare their task and architecture") == [
        "task",
        "architecture",
    ]
    assert extract_query_facets("Compare their tasks and datasets") == [
        "task",
        "dataset_setup",
    ]
    assert extract_query_facets("Compare their tasks, architectures and datasets") == [
        "task",
        "architecture",
        "dataset_setup",
    ]
    assert extract_query_facets("multitask benchmark results") == [
        "benchmark_results"
    ]


def test_publish_batch_load_materialize_and_stale_detection(tmp_path: Path) -> None:
    db_path = tmp_path / "sqlite" / "app.db"
    _seed(db_path)
    service = PaperEvidenceService(db_path)

    published = service.publish(_complete_draft())
    assert published["status"] == "complete"
    assert published["stale"] is False
    assert len(published["facets"]) == 5
    task = next(facet for facet in published["facets"] if facet["facet"] == "task")
    assert task["synopsis"] == "The paper addresses speech emotion recognition."
    assert task["synopsis"] != "Canonical task synopsis."

    cards, matrix = service.coverage_matrix(
        ["doc-a"], ["task", "architecture", "benchmark_results"]
    )
    assert matrix[0].missing_facets == []
    sources = service.materialize_sources(
        cards, requested_facets=["task", "architecture", "benchmark_results"]
    )
    assert {source["artifact_type"] for source in sources} == {"text", "table"}
    assert all(source["card_evidence"] for source in sources)

    validation_tables = service.materialize_validation_tables(["doc-a"])
    assert [source["table_id"] for source in validation_tables] == ["table-doc-a"]
    assert validation_tables[0]["validation_only"] is True
    assert validation_tables[0]["retrieval_channels"] == ["canonical_validation"]

    validation_documents = agent_api._paper_card_validation_documents(
        settings=Settings(
            data_dir=tmp_path,
            paper_evidence_cards_enabled=True,
        ),
        retrieval={
            "diagnostics": {
                "selected_engine": "paper_evidence_cards",
                "requested_facets": ["task"],
            }
        },
        retrieved_docs=[sources[0]],
        focus_document_ids=["doc-a"],
    )
    assert {source.get("table_id") for source in validation_documents} == {
        None,
        "table-doc-a",
    }

    cached_without_card_diagnostics = agent_api._paper_card_validation_documents(
        settings=Settings(
            data_dir=tmp_path,
            paper_evidence_cards_enabled=True,
        ),
        retrieval={"mode": "retrieval_cache", "diagnostics": {"cache_hit": True}},
        retrieved_docs=[sources[0]],
        focus_document_ids=["doc-a"],
    )
    assert {source.get("table_id") for source in cached_without_card_diagnostics} == {
        None,
        "table-doc-a",
    }

    with connect(db_path) as connection:
        connection.execute("UPDATE documents SET content_hash = 'changed' WHERE id = 'doc-a'")
    stale = service.card_for_document("doc-a")
    assert stale is not None
    assert stale["stale"] is True
    assert "content_hash" in stale["stale_reasons"]


def test_paper_cards_are_revision_owned_and_active_selection_isolated(tmp_path: Path) -> None:
    db_path = tmp_path / "sqlite" / "app.db"
    _seed(db_path)
    service = PaperEvidenceService(db_path)
    first = service.publish(_complete_draft())
    assert first["status"] == "complete"

    with connect(db_path) as connection:
        document = connection.execute(
            "SELECT active_revision_id FROM documents WHERE id = 'doc-a'"
        ).fetchone()
        first_revision = str(document["active_revision_id"] or "legacy:doc-a")
        second_revision = "rev:doc-a:second"
        connection.execute(
            """
            INSERT INTO document_revisions(
                revision_id, document_id, raw_content_hash, extracted_content_hash,
                parser_fingerprint, chunker_fingerprint, artifact_fingerprint,
                materialization_fingerprint, status, created_at
            ) VALUES (?, 'doc-a', 'raw-second', 'hash-second', 'docling:v10',
                      'chunker:v2', 'artifact:v2', 'materialization-second',
                      'staged', 'later')
            """,
            (second_revision,),
        )
        connection.execute(
            """
            INSERT INTO chunks(
                id, document_id, revision_id, chunk_index, content, source_path,
                filename, chunk_type, order_index, page_number, created_at,
                metadata_json
            ) VALUES ('chunk-doc-a-second', 'doc-a', ?, 0,
                      'The second revision adds a new speech dataset.',
                      '/tmp/doc-a.pdf', 'doc-a.pdf', 'text', 0, 1, 'later', '{}')
            """,
            (second_revision,),
        )
        connection.execute(
            """
            UPDATE documents
            SET active_revision_id = ?, revision_status = 'active',
                content_hash = 'hash-second'
            WHERE id = 'doc-a'
            """,
            (second_revision,),
        )

    active_candidates = service.candidate_sources("doc-a")
    assert any(
        candidate["source_id"] == "chunk-doc-a-second"
        for candidate in active_candidates
    )
    assert all(
        candidate["source_id"] != "chunk-doc-a"
        for candidate in active_candidates
    )

    second = service.publish(
        PaperEvidenceDraft(
            document_id="doc-a",
            generator_model="cx/gpt-5.6-sol",
            facets=[
                EvidenceFacetDraft(
                    facet="task",
                    synopsis="second revision task",
                    items=[
                        EvidenceItemDraft(
                            "The second revision adds a new speech dataset.",
                            [
                                EvidenceRefDraft(
                                    source_kind="chunk",
                                    source_id="chunk-doc-a-second",
                                    quote="The second revision adds a new speech dataset.",
                                )
                            ],
                        )
                    ],
                )
            ],
        )
    )
    assert second["stale"] is False

    with connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT revision_id FROM paper_evidence_card_revisions
            WHERE document_id = ? ORDER BY revision_id
            """,
            ("doc-a",),
        ).fetchall()
        assert {str(row["revision_id"]) for row in rows} == {
            first_revision,
            "rev:doc-a:second",
        }
        connection.execute(
            """
            UPDATE documents
            SET active_revision_id = NULL, revision_status = 'active',
                content_hash = 'hash-a'
            WHERE id = 'doc-a'
            """,
        )

    active_again = service.card_for_document("doc-a")
    assert active_again is not None
    assert active_again["revision_id"] == first_revision
    assert active_again["stale"] is False


def test_cross_document_source_is_rejected_before_atomic_publish(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    _seed(db_path, document_id="doc-a")
    with connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO documents(
                id, folder_id, source_path, filename, file_type, title, doc_type,
                language, content_hash, modified_at, indexed_at, chunk_count,
                parser_name, parser_version
            ) VALUES ('doc-b', 'folder', '/tmp/doc-b.pdf', 'doc-b.pdf', 'pdf',
                      'Paper B', 'research_paper', 'en', 'hash-b', 'now', 'now', 1,
                      'docling', 'v9')
            """
        )
        connection.execute(
            """
            INSERT INTO chunks(
                id, document_id, chunk_index, content, source_path, filename,
                chunk_type, order_index, page_number, created_at, metadata_json
            ) VALUES ('chunk-doc-b', 'doc-b', 0, 'Foreign evidence text.',
                      '/tmp/doc-b.pdf', 'doc-b.pdf', 'text', 0, 1, 'now', '{}')
            """
        )
    service = PaperEvidenceService(db_path)
    bad = PaperEvidenceDraft(
        document_id="doc-a",
        generator_model="cx/gpt-5.6-sol",
        facets=[
            EvidenceFacetDraft(
                facet="task",
                synopsis="bad",
                items=[
                    EvidenceItemDraft(
                        "Foreign evidence text.",
                        [
                            EvidenceRefDraft(
                                source_kind="chunk",
                                source_id="chunk-doc-b",
                                quote="Foreign evidence text.",
                            )
                        ],
                    )
                ],
            )
        ],
    )
    with pytest.raises(ValueError, match="no valid canonical evidence"):
        service.publish(bad)
    assert service.card_for_document("doc-a") is None


def test_quote_matching_tolerates_layout_spacing_but_not_paraphrase_order() -> None:
    source = (
        "Theproposedframeworkemploys a structuredfusion approach and captures "
        "over- lapping speech under noisyconditions."
    )
    assert _quote_found(
        "The proposed framework employs a structured fusion approach",
        source,
    )
    assert _quote_found("overlapping speech under noisy conditions", source)
    assert not _quote_found(
        "Noisy conditions employ a proposed structured framework",
        source,
    )


class _FakeClient:
    async def chat(self, **_: object):
        source = "chunk-doc-a"
        facet = {
            "synopsis": "Grounded synopsis.",
            "confidence": 0.9,
            "items": [
                {
                    "claim": "The paper addresses speech emotion recognition.",
                    "refs": [
                        {
                            "source_id": source,
                            "quote": "The paper addresses speech emotion recognition",
                        }
                    ],
                }
            ],
        }
        payload = {"facets": {name: facet for name in CORE_PAPER_FACETS}}
        return type("Completion", (), {"message": json.dumps(payload)})()

    async def stream_chat(self, **kwargs):
        completion = await self.chat(**kwargs)
        yield type("Chunk", (), {"content": completion.message})()


def test_builder_uses_one_approved_model_call_and_publishes(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    _seed(db_path)
    service = PaperEvidenceService(db_path)
    builder = PaperEvidenceBuilder(
        service=service,
        client=_FakeClient(),
        model="cx/gpt-5.6-sol",
        max_concurrency=2,
    )
    result = asyncio.run(builder.build_document("doc-a"))
    assert result["status"] == "complete"
    assert service.get_status()["job_status"] == {"complete": 1}


def test_builder_resume_skips_fresh_partial_card_without_provider_retry(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "app.db"
    _seed(db_path)
    service = PaperEvidenceService(db_path)
    service.mark_job("doc-a", "building", model="cx/gpt-5.6-sol")
    complete = _complete_draft()
    partial = service.publish(
        PaperEvidenceDraft(
            document_id="doc-a",
            generator_model="cx/gpt-5.6-sol",
            facets=[facet for facet in complete.facets if facet.facet == "task"],
        )
    )
    assert partial["status"] == "partial"

    class _ProviderMustNotRun:
        async def chat(self, **_: object):
            raise AssertionError("fresh partial cards are terminal unless force=True")

    builder = PaperEvidenceBuilder(
        service=service,
        client=_ProviderMustNotRun(),
        model="cx/gpt-5.6-sol",
    )
    result = asyncio.run(builder.build_document("doc-a"))

    assert result["status"] == "skipped"
    assert result["card"]["status"] == "partial"
    assert service.get_status()["job_status"] == {"complete": 1}


@pytest.mark.parametrize(
    "rendered",
    (
        'Here is the result:\n```json\n{"facets":{"task":{}}}\n```',
        'Result follows:\n{"facets":{"task":{}}}\nDone.',
    ),
)
def test_evidence_card_parser_accepts_one_unambiguous_wrapped_object(
    rendered: str,
) -> None:
    assert _parse_json_object(rendered) == {"facets": {"task": {}}}


def test_evidence_card_parser_rejects_multiple_candidate_objects() -> None:
    with pytest.raises(ValueError, match="ambiguous multiple JSON objects"):
        _parse_json_object(
            '{"facets":{"task":{"synopsis":"first"}}}\n'
            '{"facets":{"task":{"synopsis":"second"}}}'
        )


def test_builder_retries_invalid_serialization_once_with_same_model(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "app.db"
    _seed(db_path)
    service = PaperEvidenceService(db_path)

    class _InvalidThenValidClient(_FakeClient):
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def chat(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return type("Completion", (), {"message": "truncated: {"})()
            return await super().chat(**kwargs)

    client = _InvalidThenValidClient()
    builder = PaperEvidenceBuilder(
        service=service,
        client=client,
        model="cx/gpt-5.6-sol",
    )
    result = asyncio.run(builder.build_document("doc-a"))

    assert result["status"] == "complete"
    assert result["card"]["metadata"]["generation_attempts"] == 2
    assert len(client.calls) == 2
    assert {call["model"] for call in client.calls} == {"cx/gpt-5.6-sol"}
    assert {call["response_format"]["type"] for call in client.calls} == {"json_object"}
    assert "at most 2 items per facet" in client.calls[1]["messages"][0]["content"]


def test_runtime_card_hit_bypasses_raw_retrieval(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    db_path = data_dir / "sqlite" / "app.db"
    _seed(db_path)
    service = PaperEvidenceService(db_path)
    service.publish(_complete_draft())
    settings = Settings(data_dir=data_dir, paper_evidence_cards_enabled=True)

    async def fail_raw(**_kwargs):
        raise AssertionError("complete card hit must not call raw retrieval")

    monkeypatch.setattr(agent_api, "_retrieve_legacy_for_agent", fail_raw)

    async def run():
        return await agent_api._retrieve_with_paper_evidence_cards(  # noqa: SLF001
            rag=object(),
            settings=settings,
            query="paper làm gì",
            original_task="paper làm gì",
            collection_id=None,
            retrieval_mode="auto",
            focus_document_ids=["doc-a"],
            answer_intent="direct_answer",
            answer_depth="normal",
            include_visual_boost=False,
            prefer_tables=False,
        )

    result = asyncio.run(run())
    assert result is not None
    retrieval, navigation, coverage = result
    assert retrieval["mode"] == "paper_evidence_cards"
    assert retrieval["diagnostics"]["raw_fallback_document_count"] == 0
    assert "PAPER CARD: doc-a.pdf" in navigation
    assert coverage[0]["missing_facets"] == []


def test_runtime_retrieves_only_missing_facets_for_each_paper(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    db_path = data_dir / "sqlite" / "app.db"
    _seed(db_path)
    service = PaperEvidenceService(db_path)
    complete = _complete_draft()
    service.publish(
        PaperEvidenceDraft(
            document_id="doc-a",
            generator_model="cx/gpt-5.6-sol",
            facets=[facet for facet in complete.facets if facet.facet == "task"],
        )
    )
    settings = Settings(data_dir=data_dir, paper_evidence_cards_enabled=True)
    calls = []

    async def fake_raw(**kwargs):
        calls.append(kwargs)
        return {
            "mode": "hybrid",
            "documents": [
                {
                    "chunk_id": "raw-a",
                    "document_id": "doc-a",
                    "filename": "doc-a.pdf",
                    "content": "Raw architecture and benchmark evidence.",
                }
            ],
            "context_text": "raw",
            "context_stats": {},
            "diagnostics": {},
        }

    monkeypatch.setattr(agent_api, "_retrieve_legacy_for_agent", fake_raw)

    async def run():
        return await agent_api._retrieve_with_paper_evidence_cards(  # noqa: SLF001
            rag=object(),
            settings=settings,
            query="architecture benchmark",
            original_task="kiến trúc và kết quả",
            collection_id=None,
            retrieval_mode="auto",
            focus_document_ids=["doc-a"],
            answer_intent="direct_answer",
            answer_depth="normal",
            include_visual_boost=False,
            prefer_tables=False,
        )

    result = asyncio.run(run())
    assert result is not None
    retrieval, _navigation, coverage = result
    assert len(calls) == 1
    assert calls[0]["focus_document_ids"] == ["doc-a"]
    assert "architecture pipeline" in calls[0]["query"]
    assert "benchmark results" in calls[0]["query"]
    assert retrieval["mode"] == "paper_evidence_cards+raw_fallback"
    assert coverage[0]["missing_facets"] == ["architecture", "benchmark_results"]
    raw = next(item for item in retrieval["documents"] if item.get("chunk_id") == "raw-a")
    assert raw["evidence_facets"] == ["architecture", "benchmark_results"]


def test_indexing_only_queues_evidence_build_without_calling_provider(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    init_db(db_path)
    source = tmp_path / "paper.md"
    source.write_text(
        "# Paper\n\n## Abstract\nA grounded research task.\n\n## Method\nA compact architecture.",
        encoding="utf-8",
    )
    indexed = IndexingService(
        db_path,
        paper_evidence_card_build_enabled=True,
        paper_evidence_card_model="cx/gpt-5.6-sol",
    ).index_file(source_path=str(source))
    with connect(db_path) as connection:
        job = connection.execute(
            "SELECT status, attempt_count, generator_model FROM paper_evidence_build_jobs WHERE document_id = ?",
            (indexed.id,),
        ).fetchone()
        assert job is not None
        assert dict(job) == {
            "status": "pending",
            "attempt_count": 0,
            "generator_model": "cx/gpt-5.6-sol",
        }
        assert connection.execute(
            "SELECT COUNT(*) FROM paper_evidence_cards"
        ).fetchone()[0] == 0


def test_cached_card_retrieval_restores_progressive_runtime_from_provenance(
    monkeypatch,
) -> None:
    class FakeCards:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def coverage_matrix(self, document_ids, requested_facets):
            assert list(document_ids) == ["doc-a", "doc-b"]
            assert list(requested_facets) == ["architecture", "benchmark_results"]
            cards = [{"document_id": "doc-a"}, {"document_id": "doc-b"}]
            coverage = [
                SimpleNamespace(
                    document_id=document_id,
                    requested_facets=list(requested_facets),
                    covered_facets=list(requested_facets),
                    missing_facets=[],
                    stale=False,
                    status="partial",
                )
                for document_id in document_ids
            ]
            return cards, coverage

        def render_navigation_context(self, cards, *, requested_facets):
            assert len(list(cards)) == 2
            assert list(requested_facets) == ["architecture", "benchmark_results"]
            return "rehydrated card navigation"

    monkeypatch.setattr(agent_api, "PaperEvidenceService", FakeCards)
    settings = SimpleNamespace(
        sqlite_db_path=Path("/tmp/unused.db"),
        paper_evidence_card_schema_version="v1",
        paper_evidence_card_prompt_version="v2",
    )
    navigation, coverage = agent_api._restore_cached_paper_evidence_runtime(
        settings=settings,
        retrieval={
            "documents": [],
            "diagnostics": {
                "selected_engine": "paper_evidence_cards",
                "requested_facets": ["architecture", "benchmark_results"],
                "paper_facet_coverage": [{"document_id": "doc-a"}],
            },
        },
        focus_document_ids=["doc-a", "doc-b"],
        original_task="Compare their architecture and benchmark results.",
        answer_intent="compare",
    )

    assert navigation == "rehydrated card navigation"
    assert [item["document_id"] for item in coverage] == ["doc-a", "doc-b"]


def test_cached_card_retrieval_recomputes_facets_for_the_current_followup(
    monkeypatch,
) -> None:
    observed: dict[str, list[str]] = {}

    class FakeCards:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def coverage_matrix(self, document_ids, requested_facets):
            observed["facets"] = list(requested_facets)
            coverage = [
                SimpleNamespace(
                    document_id=document_ids[0],
                    requested_facets=list(requested_facets),
                    covered_facets=[],
                    missing_facets=list(requested_facets),
                    stale=False,
                    status="partial",
                )
            ]
            return [{"document_id": document_ids[0]}], coverage

        def render_navigation_context(self, cards, *, requested_facets):
            return ""

    monkeypatch.setattr(agent_api, "PaperEvidenceService", FakeCards)
    settings = SimpleNamespace(
        sqlite_db_path=Path("/tmp/unused.db"),
        paper_evidence_card_schema_version="v1",
        paper_evidence_card_prompt_version="v2",
    )
    _navigation, coverage = agent_api._restore_cached_paper_evidence_runtime(
        settings=settings,
        retrieval={
            "documents": [],
            "diagnostics": {
                "selected_engine": "paper_evidence_cards",
                "requested_facets": ["task", "contributions"],
                "paper_facet_coverage": [{"document_id": "doc-a"}],
            },
        },
        focus_document_ids=["doc-a"],
        original_task="What architecture does that plan propose?",
        answer_intent="infer_structure",
    )

    assert observed["facets"] == ["architecture"]
    assert coverage[0]["requested_facets"] == ["architecture"]
