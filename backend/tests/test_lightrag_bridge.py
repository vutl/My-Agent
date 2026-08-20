import asyncio
import json

from app.api import lightrag_routes
from app.core.config import Settings
from app.db.sqlite import connect, init_db
from app.lightrag import bridge as bridge_module
from app.lightrag.bridge import (
    LightRAGBridge,
    _is_generic_graph_candidate,
    _source_chunk_ids,
)
from app.lightrag.provenance import (
    ProvenanceResolution,
    ResolvedParentPassage,
    sync_document_chunk_records,
)
from app.services.indexing_service import IndexingService


def _settings_and_duplicate_documents(tmp_path):
    data_dir = tmp_path / "data"
    db_path = data_dir / "sqlite" / "app.db"
    init_db(db_path)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first_path = first_dir / "paper.md"
    second_path = second_dir / "paper.md"
    first_path.write_text("# First Paper\n\nAlpha evidence.", encoding="utf-8")
    second_path.write_text("# Second Paper\n\nBeta evidence.", encoding="utf-8")
    indexing = IndexingService(db_path)
    first = indexing.index_file(source_path=str(first_path))
    second = indexing.index_file(source_path=str(second_path))
    return Settings(data_dir=data_dir), first, second


def test_lightrag_ambiguous_basename_is_not_relabelled_by_focus(tmp_path) -> None:
    settings, first, second = _settings_and_duplicate_documents(tmp_path)
    bridge = LightRAGBridge(settings)
    raw = {
        "data": {
            "chunks": [
                {
                    "chunk_id": "chunk-1",
                    "file_path": "paper.md",
                    "content": "focused evidence",
                }
            ]
        }
    }

    focused = bridge._to_retrieval_results(  # noqa: SLF001
        raw,
        focus_document_ids=[first.id],
    )
    unscoped = bridge._to_retrieval_results(  # noqa: SLF001
        raw,
        focus_document_ids=None,
    )

    assert focused == []
    assert unscoped == []
    assert second.id != first.id


def test_lightrag_mapping_accepts_exact_path_and_document_id(tmp_path) -> None:
    settings, first, second = _settings_and_duplicate_documents(tmp_path)
    bridge = LightRAGBridge(settings)
    raw = {
        "data": {
            "chunks": [
                {
                    "chunk_id": "exact-path",
                    "file_path": second.source_path,
                    "content": "second evidence",
                },
                {
                    "chunk_id": "direct-id",
                    "document_id": first.id,
                    "content": "first evidence",
                },
            ]
        }
    }

    mapped = bridge._to_retrieval_results(raw, focus_document_ids=None)  # noqa: SLF001

    assert [item["document_id"] for item in mapped] == [second.id, first.id]
    assert [item["source_path"] for item in mapped] == [second.source_path, first.source_path]

    wrong_focus = bridge._to_retrieval_results(  # noqa: SLF001
        raw,
        focus_document_ids=[first.id],
    )
    assert [item["chunk_id"] for item in wrong_focus] == ["direct-id"]


def test_lightrag_multi_source_entity_is_not_relabelled_by_focus(tmp_path) -> None:
    settings, first, second = _settings_and_duplicate_documents(tmp_path)
    bridge = LightRAGBridge(settings)
    raw = {
        "data": {
            "entities": [
                {
                    "reference_id": "entity-1",
                    "entity_name": "Shared concept",
                    "description": "Appears in multiple papers.",
                    "file_path": f"{first.source_path}<SEP>{second.source_path}",
                }
            ]
        }
    }

    focused = bridge._to_retrieval_results(  # noqa: SLF001
        raw,
        focus_document_ids=[first.id],
    )
    unscoped = bridge._to_retrieval_results(  # noqa: SLF001
        raw,
        focus_document_ids=None,
    )

    assert focused == []
    assert unscoped == []


def test_lightrag_debug_query_maps_without_focus(monkeypatch, tmp_path) -> None:
    settings, _, _ = _settings_and_duplicate_documents(tmp_path)

    async def fake_query(*_args, **_kwargs):
        return {"data": {"chunks": []}}

    monkeypatch.setattr(lightrag_routes, "query_lightrag", fake_query)
    result = asyncio.run(
        lightrag_routes.debug_query(
            lightrag_routes.LightRAGQueryRequest(query="architecture"),
            settings,
        )
    )

    assert result["mapped_results"] == []


def test_lightrag_explicit_empty_scope_never_widens_to_corpus(tmp_path) -> None:
    settings, first, _second = _settings_and_duplicate_documents(tmp_path)
    bridge = LightRAGBridge(settings)
    raw = {
        "data": {
            "chunks": [
                {
                    "chunk_id": "direct-id",
                    "document_id": first.id,
                    "content": "must stay out",
                }
            ]
        }
    }

    assert bridge._to_retrieval_results(raw, focus_document_ids=[]) == []  # noqa: SLF001


def test_graph_source_ids_are_split_and_deduplicated() -> None:
    assert _source_chunk_ids(
        {
            "source_id": (
                "doc-a-chunk-001<SEP>doc-b-chunk-002"
                "<SEP>doc-a-chunk-001<SEP>truncated"
            )
        }
    ) == ["doc-a-chunk-001", "doc-b-chunk-002"]


def test_structural_and_high_cardinality_hubs_are_not_bridge_candidates() -> None:
    assert _is_generic_graph_candidate(
        {
            "entity_name": "Figure 2",
            "source_id": "a<SEP>b",
        }
    )
    assert _is_generic_graph_candidate(
        {
            "entity_name": "Accuracy",
            "source_id": "<SEP>".join(f"chunk-{index}" for index in range(8)),
        }
    )
    assert not _is_generic_graph_candidate(
        {
            "entity_name": "Dual-branch Mamba fusion",
            "source_id": "a<SEP>b<SEP>c",
        }
    )


def _first_parent(settings, document):
    with connect(settings.sqlite_db_path) as connection:
        row = connection.execute(
            """
            SELECT parent_chunk_id, metadata_json
            FROM chunks
            WHERE document_id = ?
            ORDER BY chunk_index
            LIMIT 1
            """,
            (document.id,),
        ).fetchone()
    metadata = json.loads(row["metadata_json"])
    return str(row["parent_chunk_id"]), str(metadata["parent_content"])


def _sync_parent(settings, document, chunk_id):
    parent_chunk_id, parent_content = _first_parent(settings, document)
    sync_document_chunk_records(
        settings.sqlite_db_path,
        document.id,
        [
            {
                "_id": chunk_id,
                "content": parent_content,
                "full_doc_id": document.id,
                "file_path": document.source_path,
                "chunk_order_index": 0,
            }
        ],
    )
    return parent_chunk_id, parent_content


def test_graph_navigation_materializes_one_deduped_parent_with_anchors(
    monkeypatch,
    tmp_path,
) -> None:
    settings, first, _second = _settings_and_duplicate_documents(tmp_path)
    parent_chunk_id, parent_content = _sync_parent(settings, first, "lr-source")
    raw = {
        "data": {
            "chunks": [
                {
                    "chunk_id": "lr-source",
                    "file_path": first.source_path,
                    "content": parent_content,
                }
            ],
            "entities": [
                {
                    "entity_name": "Cross-modal Fusion",
                    "description": "Graph prose must never become evidence.",
                    "source_id": "lr-source",
                    "file_path": first.source_path,
                }
            ],
            "relationships": [
                {
                    "src_id": "Acoustic Encoder",
                    "tgt_id": "Emotion Classifier",
                    "description": "Relationship prose is navigation only.",
                    "source_id": "lr-source",
                    "file_path": first.source_path,
                }
            ],
        }
    }
    calls = []
    original_resolver = bridge_module.resolve_lightrag_chunk_parent_status

    def observing_resolver(db_path, chunk_ids, *, allowed_document_ids=None):
        calls.append(list(chunk_ids))
        return original_resolver(
            db_path,
            chunk_ids,
            allowed_document_ids=allowed_document_ids,
        )

    monkeypatch.setattr(
        bridge_module,
        "resolve_lightrag_chunk_parent_status",
        observing_resolver,
    )
    results, diagnostics = LightRAGBridge(settings)._materialize_retrieval_results(  # noqa: SLF001
        raw,
        focus_document_ids=[first.id],
    )

    assert calls == [["lr-source"]]
    assert len(results) == 1
    result = results[0]
    assert result["chunk_id"] == parent_chunk_id
    assert result["content"] == parent_content
    assert "Graph prose must never become evidence." not in result["content"]
    assert "Relationship prose is navigation only." not in result["content"]
    assert {"lightrag_parent", "graph_bridge", "lightrag_entity", "lightrag_relation"} <= set(
        result["retrieval_channels"]
    )
    assert result["document_id"] == first.id
    assert result["anchors"] == [
        "Cross-modal Fusion",
        "Acoustic Encoder",
        "Emotion Classifier",
    ]
    assert all(
        record["document_id"] == first.id
        for record in result["graph_bridge_metadata"]
    )
    assert diagnostics["provenance_requested_chunk_count"] == 1
    assert diagnostics["provenance_mapped_chunk_count"] == 1
    assert diagnostics["parent_passage_count"] == 1
    assert diagnostics["parent_deduplicated_count"] == 2
    assert len(diagnostics["graph_bridge_metadata"]) == 2


def test_graph_source_ids_are_scoped_by_canonical_parent_not_active_relabel(
    tmp_path,
) -> None:
    settings, first, second = _settings_and_duplicate_documents(tmp_path)
    _sync_parent(settings, first, "first-source")
    _sync_parent(settings, second, "second-source")
    raw = {
        "data": {
            "entities": [
                {
                    "entity_name": "Shared but specific concept",
                    "source_id": "first-source<SEP>second-source",
                    "file_path": (
                        f"{first.source_path}<SEP>{second.source_path}"
                    ),
                }
            ]
        }
    }

    results, diagnostics = LightRAGBridge(settings)._materialize_retrieval_results(  # noqa: SLF001
        raw,
        focus_document_ids=[first.id],
    )

    assert len(results) == 1
    assert results[0]["document_id"] == first.id
    assert results[0]["graph_bridge_metadata"][0]["document_id"] == first.id
    assert diagnostics["provenance_known_chunk_count"] == 2
    assert diagnostics["provenance_mapped_chunk_count"] == 1
    assert diagnostics["provenance_scoped_out_chunk_count"] == 1


def test_retrieve_exposes_graph_bridge_metadata_in_top_level_diagnostics(
    monkeypatch,
    tmp_path,
) -> None:
    settings, first, _second = _settings_and_duplicate_documents(tmp_path)
    _parent_chunk_id, parent_content = _sync_parent(
        settings,
        first,
        "lr-source",
    )
    raw = {
        "status": "success",
        "data": {
            "entities": [
                {
                    "entity_name": "Cross-modal Fusion",
                    "source_id": "lr-source",
                    "file_path": first.source_path,
                }
            ]
        },
    }

    async def fake_query(*_args, **_kwargs):
        return raw

    monkeypatch.setattr(bridge_module, "query_lightrag", fake_query)
    retrieval = asyncio.run(
        LightRAGBridge(settings).retrieve(
            "fusion",
            focus_document_ids=[first.id],
        )
    )

    assert retrieval["documents"][0]["content"] == " ".join(
        parent_content.split()
    )
    graph_metadata = retrieval["diagnostics"]["graph_bridge_metadata"]
    assert graph_metadata[0]["document_id"] == first.id
    assert graph_metadata[0]["anchors"] == ["Cross-modal Fusion"]
    assert retrieval["diagnostics"]["bridge"]["parent_passage_count"] == 1


def test_generic_graph_hubs_report_cardinality_and_never_resolve(
    monkeypatch,
    tmp_path,
) -> None:
    settings, _first, _second = _settings_and_duplicate_documents(tmp_path)
    calls = []

    def empty_resolver(db_path, chunk_ids, *, allowed_document_ids=None):
        calls.append(list(chunk_ids))
        return ProvenanceResolution({}, frozenset(), frozenset(), frozenset())

    monkeypatch.setattr(
        bridge_module,
        "resolve_lightrag_chunk_parent_status",
        empty_resolver,
    )
    raw = {
        "data": {
            "entities": [
                {
                    "entity_name": "Figure 2",
                    "source_id": "one",
                },
                {
                    "entity_name": "Accuracy",
                    "source_id": "<SEP>".join(
                        f"source-{index}" for index in range(8)
                    ),
                },
            ]
        }
    }

    results, diagnostics = LightRAGBridge(settings)._materialize_retrieval_results(  # noqa: SLF001
        raw,
        focus_document_ids=None,
    )

    assert results == []
    assert calls == [[]]
    assert diagnostics["graph_source_cardinality_total"] == 9
    assert diagnostics["graph_source_cardinality_max"] == 8
    assert diagnostics["suppression_reasons"] == {"generic_entity": 2}


def _resolved_parent(document, chunk_id, parent_index):
    return ResolvedParentPassage(
        lightrag_chunk_id=chunk_id,
        document_id=document.id,
        parent_chunk_id=f"{document.id}-parent-{parent_index}",
        parent_content=f"Grounded parent {parent_index} for {document.id}",
        source_path=document.source_path,
        filename=document.filename,
        lightrag_full_doc_id=document.id,
        lightrag_file_path=document.source_path,
        lightrag_chunk_order_index=parent_index,
        parent_order_index=parent_index,
        content_hash=f"chunk-hash-{parent_index}",
        parent_content_hash=f"parent-hash-{parent_index}",
        overlap_chars=100 - parent_index,
        canonical_method="full_doc_id",
        mapping_method="exact_offset",
        mapping_score=1.0,
        document_char_start=0,
        document_char_end=100,
        mapped_at="now",
        page_number=parent_index + 1,
        section_title="Method",
        heading_path=("Method",),
    )


def test_parent_quota_preserves_candidates_from_second_document(
    monkeypatch,
    tmp_path,
) -> None:
    settings, first, second = _settings_and_duplicate_documents(tmp_path)
    source_ids = [f"first-{index}" for index in range(5)] + ["second-0"]
    parents = {
        source_id: [
            _resolved_parent(
                first if source_id.startswith("first") else second,
                source_id,
                index if source_id.startswith("first") else 0,
            )
        ]
        for index, source_id in enumerate(source_ids)
    }

    def fake_resolver(db_path, chunk_ids, *, allowed_document_ids=None):
        return ProvenanceResolution(
            parents_by_chunk_id={
                chunk_id: parents.get(chunk_id, []) for chunk_id in chunk_ids
            },
            known_chunk_ids=frozenset(chunk_ids),
            stale_chunk_ids=frozenset(),
            scoped_out_chunk_ids=frozenset(),
        )

    monkeypatch.setattr(
        bridge_module,
        "resolve_lightrag_chunk_parent_status",
        fake_resolver,
    )
    raw = {
        "data": {
            "entities": [
                {
                    "entity_name": f"Specific Concept {index}",
                    "source_id": source_id,
                }
                for index, source_id in enumerate(source_ids)
            ]
        }
    }

    results, diagnostics = LightRAGBridge(settings)._materialize_retrieval_results(  # noqa: SLF001
        raw,
        focus_document_ids=None,
        max_parent_candidates_per_document=2,
    )

    assert [item["document_id"] for item in results].count(first.id) == 2
    assert [item["document_id"] for item in results].count(second.id) == 1
    assert diagnostics["suppression_reasons"]["parent_document_quota"] == 3
    assert diagnostics["parent_counts_by_document"] == {
        first.id: 2,
        second.id: 1,
    }


def test_stale_known_mapping_never_falls_back_to_raw_chunk(tmp_path) -> None:
    settings, first, _second = _settings_and_duplicate_documents(tmp_path)
    parent_chunk_id, parent_content = _sync_parent(settings, first, "stale")
    with connect(settings.sqlite_db_path) as connection:
        row = connection.execute(
            """
            SELECT metadata_json
            FROM chunks
            WHERE document_id = ? AND parent_chunk_id = ?
            LIMIT 1
            """,
            (first.id, parent_chunk_id),
        ).fetchone()
        metadata = json.loads(row["metadata_json"])
        metadata["parent_content"] = "Changed after provenance sync."
        connection.execute(
            """
            UPDATE chunks
            SET metadata_json = ?
            WHERE document_id = ? AND parent_chunk_id = ?
            """,
            (json.dumps(metadata), first.id, parent_chunk_id),
        )
    raw = {
        "data": {
            "chunks": [
                {
                    "chunk_id": "stale",
                    "file_path": first.source_path,
                    "content": parent_content,
                }
            ]
        }
    }

    results, diagnostics = LightRAGBridge(settings)._materialize_retrieval_results(  # noqa: SLF001
        raw,
        focus_document_ids=[first.id],
    )

    assert results == []
    assert diagnostics["provenance_known_chunk_count"] == 1
    assert diagnostics["provenance_stale_chunk_count"] == 1
    assert diagnostics["raw_fallback_count"] == 0
    assert diagnostics["suppression_reasons"] == {
        "raw_chunk_known_mapping_unresolved": 1
    }
