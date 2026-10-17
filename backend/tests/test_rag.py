from pathlib import Path
import asyncio
import json
from uuid import uuid4

from app.db.sqlite import connect, init_db
from app.rag.chunking import chunk_parsed_document, chunk_text, count_tokens
from app.rag.embeddings import HashEmbeddingProvider
from app.rag.parsers import ParsedDocument, ParsedPage, parse_document, parse_text_file
from app.services.indexing_service import IndexingService
from app.services.rag_service import RagService


def test_chunk_text_splits_long_text() -> None:
    text = "alpha " * 500

    chunks = chunk_text(text, chunk_size=200, overlap=20)

    assert len(chunks) > 1
    assert all(chunk for chunk in chunks)


def test_chunk_parsed_document_preserves_page_numbers() -> None:
    parsed = ParsedDocument(
        text="first page text\n\nsecond page text",
        parser_name="test",
        parser_version="1",
        page_count=2,
        pages=[
            ParsedPage(page_number=7, text="first page text"),
            ParsedPage(page_number=8, text="second page text"),
        ],
    )

    chunks = chunk_parsed_document(parsed, chunk_size=200, overlap=20)

    assert [chunk.page_number for chunk in chunks] == [7, 8]
    assert [chunk.text for chunk in chunks] == ["first page text", "second page text"]


def test_chunk_parsed_document_preserves_printed_page_aliases() -> None:
    parsed = ParsedDocument(
        text="first page text\n\nsecond page text",
        parser_name="test",
        parser_version="page-labels",
        page_count=2,
        pages=[
            ParsedPage(page_number=7, text="first page text", page_label="1"),
            ParsedPage(page_number=8, text="second page text", page_label="2"),
        ],
    )

    chunks = chunk_parsed_document(parsed, chunk_size=200, overlap=20)

    assert [chunk.page_label for chunk in chunks] == ["1", "2"]


def test_chunking_preserves_paragraph_boundaries_and_section_title() -> None:
    text = "\n\n".join(
        [
            "2 Method",
            "The first paragraph explains the architecture. It should stay together.",
            "The second paragraph adds training details. It should not be cut in the middle.",
        ]
    )

    chunks = chunk_parsed_document(
        ParsedDocument(text=text, parser_name="test", parser_version="1"),
        chunk_size=120,
        overlap=0,
    )

    assert chunks
    assert chunks[0].section_title == "2 Method"
    assert any("The first paragraph explains the architecture." in chunk.text for chunk in chunks)


def test_token_chunking_creates_bounded_children_with_parent_context() -> None:
    text = "\n\n".join(
        [
            "2 Method",
            "The architecture combines acoustic and visual encoders. " * 35,
            "The fusion block exchanges information across modalities. " * 35,
        ]
    )

    chunks = chunk_parsed_document(
        ParsedDocument(text=text, parser_name="test", parser_version="1"),
        max_tokens=96,
        overlap_tokens=12,
        parent_max_tokens=320,
    )

    assert len(chunks) > 1
    assert all((chunk.token_count or 0) <= 96 for chunk in chunks)
    assert all(chunk.parent_text for chunk in chunks)
    assert all((chunk.parent_token_count or 0) <= 320 for chunk in chunks)
    assert any(chunk.parent_text != chunk.text for chunk in chunks)
    assert all(count_tokens(chunk.text) == chunk.token_count for chunk in chunks)


def test_markdown_table_and_figure_caption_are_indexed(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "report.md").write_text(
        "\n".join(
            [
                "# Report",
                "",
                "Table 1: Metrics",
                "| metric | value |",
                "| --- | --- |",
                "| accuracy | 0.92 |",
                "",
                "Figure 1: Agent architecture diagram",
            ]
        ),
        encoding="utf-8",
    )

    init_db(db_path)
    indexing = IndexingService(db_path)
    rag = RagService(db_path)
    result = indexing.index_folder(
        folder_path=str(docs_dir),
        recursive=False,
        file_types=["md"],
    )
    document_id = result.documents[0].id
    documents = indexing.list_documents()

    assert documents[0]["table_count"] == 1
    assert documents[0]["figure_count"] == 1

    tables = rag.list_document_tables(document_id)
    figures = rag.list_document_figures(document_id)

    assert tables[0]["caption"] == "Table 1: Metrics"
    assert tables[0]["row_count"] == 1
    assert tables[0]["column_count"] == 2
    assert figures[0]["caption"] == "Figure 1: Agent architecture diagram"


def test_table_image_path_in_metadata_is_promoted_to_canonical_table_row(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    source = tmp_path / "report.md"
    source.write_text(
        "# Report\n\nTable 1: Metrics\n| metric | value |\n| --- | --- |\n| accuracy | 0.92 |\n",
        encoding="utf-8",
    )
    table_image = tmp_path / "table-1.png"
    table_image.write_bytes(b"table-image")
    init_db(db_path)
    indexed = IndexingService(db_path).index_file(source_path=str(source))
    with connect(db_path) as connection:
        row = connection.execute(
            "SELECT id, metadata_json FROM document_tables WHERE document_id = ?",
            (indexed.id,),
        ).fetchone()
        metadata = json.loads(row["metadata_json"] or "{}")
        metadata["table_image_path"] = str(table_image)
        connection.execute(
            "UPDATE document_tables SET metadata_json = ? WHERE id = ?",
            (json.dumps(metadata), row["id"]),
        )

    table = RagService(db_path).list_document_tables(indexed.id)[0]

    assert table["image_path"] == str(table_image)


def test_mislabeled_table_caption_is_reconciled_from_unique_same_page_text(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    source = tmp_path / "report.md"
    source.write_text(
        """# Report

Table 5: Existing results
| Model | F1 |
| --- | --- |
| Base | 0.50 |

Table 6: Target results
| Model | F1 |
| --- | --- |
| Ours | 0.80 |
""",
        encoding="utf-8",
    )
    init_db(db_path)
    indexed = IndexingService(db_path).index_file(source_path=str(source))
    with connect(db_path) as connection:
        connection.execute(
            """
            UPDATE document_tables
            SET caption = 'Figure 5: Nearby qualitative example'
            WHERE document_id = ? AND table_index = 1
            """,
            (indexed.id,),
        )

    tables = RagService(db_path).list_document_tables(indexed.id)

    assert [table["caption"] for table in tables] == [
        "Table 5: Existing results",
        "Table 6: Target results",
    ]
    repaired = tables[1]
    assert repaired["metadata"]["caption_reconciled"] is True
    assert repaired["metadata"]["caption_reconciled_number"] == 6
    assert (
        repaired["metadata"]["caption_reconciliation_source"]
        == "same_document_page_text"
    )


def test_mislabeled_table_caption_stays_filtered_when_page_match_is_ambiguous(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    source = tmp_path / "report.md"
    source.write_text(
        """# Report

Table 5: Existing results
| Model | F1 |
| --- | --- |
| Base | 0.50 |

Table 6: First unmatched caption
| Model | F1 |
| --- | --- |
| Ours | 0.80 |

Table 7: Second unmatched caption
| Model | F1 |
| --- | --- |
| Ours | 0.81 |
""",
        encoding="utf-8",
    )
    init_db(db_path)
    indexed = IndexingService(db_path).index_file(source_path=str(source))
    with connect(db_path) as connection:
        connection.execute(
            """
            UPDATE document_tables
            SET caption = 'Figure 5: Nearby qualitative example'
            WHERE document_id = ? AND table_index = 1
            """,
            (indexed.id,),
        )
        connection.execute(
            "DELETE FROM document_tables WHERE document_id = ? AND table_index = 2",
            (indexed.id,),
        )

    tables = RagService(db_path).list_document_tables(indexed.id)

    assert [table["caption"] for table in tables] == ["Table 5: Existing results"]


def test_mislabeled_table_caption_does_not_borrow_other_page_text(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    source = tmp_path / "report.md"
    source.write_text(
        """# Report

Table 6: Target results
| Model | F1 |
| --- | --- |
| Ours | 0.80 |
""",
        encoding="utf-8",
    )
    init_db(db_path)
    indexed = IndexingService(db_path).index_file(source_path=str(source))
    with connect(db_path) as connection:
        connection.execute(
            """
            UPDATE document_tables
            SET caption = 'Figure 5: Nearby qualitative example', page_number = 2
            WHERE document_id = ?
            """,
            (indexed.id,),
        )

    assert RagService(db_path).list_document_tables(indexed.id) == []


def test_index_folder_and_search_markdown(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "project.md").write_text(
        "# Project\n\nPhase three adds local document retrieval for planning.",
        encoding="utf-8",
    )

    init_db(db_path)
    indexing = IndexingService(db_path)
    rag = RagService(db_path)

    result = indexing.index_folder(
        folder_path=str(docs_dir),
        recursive=False,
        file_types=["md"],
    )
    chunks = rag.search("document retrieval", top_k=3)

    assert result.scanned_files == 1
    assert result.indexed_files == 1
    assert result.documents[0].filename == "project.md"
    assert chunks
    assert chunks[0].filename == "project.md"
    assert chunks[0].chunk_index == 0
    assert chunks[0].page_number == 1
    assert chunks[0].token_count is not None

    documents = indexing.list_documents()
    assert documents[0]["parser_name"] == "md_plain_text"
    assert documents[0]["index_status"] == "indexed"

    detail = rag.get_document(documents[0]["id"])
    assert detail is not None
    assert detail["filename"] == "project.md"

    stored_chunks = rag.list_document_chunks(documents[0]["id"])
    assert stored_chunks[0]["page_number"] == 1
    assert stored_chunks[0]["metadata"]["file_type"] == "md"
    assert stored_chunks[0]["metadata"]["page_number"] == 1
    assert stored_chunks[0]["metadata"]["context_prefix"].startswith("Document: project.md.")

    debug = rag.search_debug("document retrieval", top_k=3)
    assert debug["fts_query"]
    assert debug["retrieval_channels"] == ["sqlite_fts5"]


def test_visual_only_readiness_requires_visual_channel_on_active_projection(tmp_path) -> None:
    import json

    db_path = tmp_path / "app.db"
    source = tmp_path / "scan.md"
    source.write_text("# scan\n\nplaceholder", encoding="utf-8")
    init_db(db_path)
    indexed = IndexingService(db_path).index_file(source_path=str(source))
    with connect(db_path) as connection:
        connection.execute(
            "DELETE FROM chunks WHERE document_id = ?",
            (indexed.id,),
        )
        connection.execute(
            """
            UPDATE documents
            SET file_type = 'pdf', page_count = 3, figure_count = 1,
                table_count = 0, metadata_json = ?, revision_status = 'active'
            WHERE id = ?
            """,
            (
                json.dumps(
                    {
                        "document_readiness": {
                            "state": "visual_only",
                            "pages": [{"state": "visual_only"}],
                        }
                    }
                ),
                indexed.id,
            ),
        )

    rag = RagService(db_path)
    assert rag.documents_requiring_visual_channel([indexed.id]) == [indexed.id]

    with connect(db_path) as connection:
        connection.execute(
            "UPDATE documents SET revision_status = 'retired' WHERE id = ?",
            (indexed.id,),
        )
    assert rag.documents_requiring_visual_channel([indexed.id]) == []


def test_version_reindex_failure_keeps_previous_canonical_document(tmp_path, monkeypatch) -> None:
    import json
    import pytest

    from app.db.sqlite import connect
    from app.services import indexing_service as indexing_module

    db_path = tmp_path / "app.db"
    artifact_root = tmp_path / "artifacts"
    source = tmp_path / "paper.md"
    source.write_text("# Paper\n\nStable evidence.", encoding="utf-8")
    init_db(db_path)
    indexing = IndexingService(db_path, artifact_root=artifact_root)
    original = indexing.index_file(source_path=str(source))
    with connect(db_path) as connection:
        connection.execute(
            "UPDATE documents SET metadata_json = ? WHERE id = ?",
            (json.dumps({"page_chunk_version": 2, "artifact_extraction_version": 7}), original.id),
        )

    def fail_chunking(_parsed):
        raise RuntimeError("synthetic chunk failure")

    monkeypatch.setattr(indexing_module, "chunk_parsed_document", fail_chunking)
    with pytest.raises(RuntimeError, match="synthetic chunk failure"):
        indexing.index_file(source_path=str(source))

    documents = indexing.list_documents()
    assert [document["id"] for document in documents] == [original.id]
    assert RagService(db_path).search("Stable evidence", top_k=1)


def test_version_reindex_preserves_document_figure_and_table_identity(tmp_path) -> None:
    import json

    from app.db.sqlite import connect

    db_path = tmp_path / "app.db"
    source = tmp_path / "paper.md"
    source.write_text(
        "# Paper\n\nFigure 3: Stable architecture diagram.\n\n"
        "| Model | Acc |\n| --- | --- |\n| Aya | 91.2 |",
        encoding="utf-8",
    )
    init_db(db_path)
    indexing = IndexingService(db_path)
    original = indexing.index_file(source_path=str(source))
    original_figure = RagService(db_path).list_document_figures(original.id)[0]
    original_table = RagService(db_path).list_document_tables(original.id)[0]
    original_chunk_ids = [
        chunk["id"] for chunk in RagService(db_path).list_document_chunks(original.id)
    ]
    with connect(db_path) as connection:
        connection.execute(
            "UPDATE documents SET metadata_json = ? WHERE id = ?",
            (json.dumps({"page_chunk_version": 2, "artifact_extraction_version": 7}), original.id),
        )

    rebuilt = indexing.index_file(source_path=str(source))
    rebuilt_figure = RagService(db_path).list_document_figures(rebuilt.id)[0]
    rebuilt_table = RagService(db_path).list_document_tables(rebuilt.id)[0]
    rebuilt_chunk_ids = [
        chunk["id"] for chunk in RagService(db_path).list_document_chunks(rebuilt.id)
    ]

    assert rebuilt.id == original.id
    assert rebuilt_figure["id"] == original_figure["id"]
    assert rebuilt_table["id"] == original_table["id"]
    assert rebuilt_chunk_ids == original_chunk_ids


def test_text_only_rechunk_preserves_existing_artifacts(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    source = tmp_path / "paper.md"
    source.write_text(
        "# Paper\n\n"
        + "Method and retrieval context. " * 100
        + "\n\nFigure 3: Stable architecture diagram.\n\n"
        "| Model | Acc |\n| --- | --- |\n| Aya | 91.2 |",
        encoding="utf-8",
    )
    init_db(db_path)
    indexing = IndexingService(db_path)
    indexed = indexing.index_file(source_path=str(source))
    rag = RagService(db_path)
    figure_id = rag.list_document_figures(indexed.id)[0]["id"]
    table_id = rag.list_document_tables(indexed.id)[0]["id"]

    result = indexing.rechunk_document(indexed.id)

    assert result["chunking_version"] == 4
    assert rag.list_document_figures(indexed.id)[0]["id"] == figure_id
    assert rag.list_document_tables(indexed.id)[0]["id"] == table_id
    assert all(
        chunk["parent_chunk_id"]
        for chunk in rag.list_document_chunks(indexed.id)
    )


def test_rag_service_expands_neighbor_chunks(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "long.md").write_text(
        "\n\n".join(
            [
                "# Long Report",
                "Alpha architecture context. " * 80,
                "Beta retrieval target details. " * 80,
                "Gamma neighbor explanation. " * 80,
            ]
        ),
        encoding="utf-8",
    )

    init_db(db_path)
    indexing = IndexingService(db_path)
    rag = RagService(db_path)
    result = indexing.index_folder(
        folder_path=str(docs_dir),
        recursive=False,
        file_types=["md"],
    )
    chunks = rag.list_document_chunks(result.documents[0].id)
    middle = chunks[1]
    expanded = rag.expand_with_neighbor_chunks(
        [
            {
                "chunk_id": middle["id"],
                "document_id": middle["document_id"],
                "filename": middle["filename"],
                "source_path": middle["source_path"],
                "content": middle["content"],
                "chunk_index": middle["chunk_index"],
                "page_number": middle["page_number"],
                "chunk_type": "text",
            }
        ]
    )

    assert expanded[0]["expanded_content"]
    assert "retrieved chunk" in expanded[0]["expanded_content"]
    assert expanded[0]["neighbor_chunk_ids"]


def test_rag_service_expands_token_child_to_parent_context(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    source = tmp_path / "parent.md"
    source.write_text(
        "# Method\n\n"
        + "Acoustic encoder setup and preprocessing details. " * 80
        + "\n\n"
        + "Cross-modal fusion target evidence and implementation details. " * 80,
        encoding="utf-8",
    )
    init_db(db_path)
    indexed = IndexingService(db_path).index_file(source_path=str(source))
    rag = RagService(db_path)
    chunks = rag.list_document_chunks(indexed.id)
    child = next(chunk for chunk in chunks if chunk["parent_chunk_id"])

    expanded = rag.expand_with_neighbor_chunks(
        [
            {
                "chunk_id": child["id"],
                "document_id": child["document_id"],
                "filename": child["filename"],
                "source_path": child["source_path"],
                "content": child["content"],
                "chunk_index": child["chunk_index"],
                "page_number": child["page_number"],
                "chunk_type": "text",
            }
        ]
    )

    assert expanded[0]["parent_expanded"] is True
    assert expanded[0]["parent_chunk_id"] == child["parent_chunk_id"]
    assert "[retrieved chunk / child]" in expanded[0]["expanded_content"]
    assert "[parent section context]" in expanded[0]["expanded_content"]
    assert isinstance(expanded[0]["neighbor_chunk_ids"], list)


def test_rag_service_attaches_active_page_text_to_visual_hit(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    source = tmp_path / "page.md"
    source.write_text(
        "# Results\n\n"
        "The visual page shows the GCN comparison.\n\n"
        "The discussion concludes that the task is not simpler.\n",
        encoding="utf-8",
    )
    init_db(db_path)
    indexed = IndexingService(db_path).index_file(source_path=str(source))
    with connect(db_path) as connection:
        connection.execute(
            "UPDATE chunks SET page_number = 2 WHERE document_id = ?",
            (indexed.id,),
        )

    visual_hit = {
        "document_id": indexed.id,
        "page_number": 2,
        "chunk_id": f"page:{indexed.id}:2",
        "chunk_type": "page",
        "artifact_type": "page",
        "visual_page": True,
        "content": "Visual page evidence selected from the active revision.",
        "retrieval_channels": ["visual_page", "colsmol"],
    }
    expanded = RagService(db_path).expand_with_page_context(
        [visual_hit],
        query="Does the GCN comparison show that the task is simpler?",
    )

    assert "[same-page native text context]" in expanded[0]["expanded_content"]
    assert "task is not simpler" in expanded[0]["expanded_content"]
    assert expanded[0]["page_context_chunk_ids"]


def test_same_page_injection_never_inherits_other_artifact_payload(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    source = tmp_path / "report.md"
    source.write_text("# Report\n\nTable 3: Dataset statistics\n| Split | Count |\n|---|---|\n| Train | 42 |\n", encoding="utf-8")
    init_db(db_path)
    document = IndexingService(db_path).index_file(source_path=str(source))
    original = {
        # Page projection is only meaningful for paged formats (PDF).
        "document_id": document.id, "page_number": 1, "filename": "report.pdf",
        "chunk_id": "figure:backbone", "figure_id": "backbone", "artifact_type": "figure",
        "chunk_type": "figure", "caption": "Architecture of a baseline",
        "content": "A convolutional encoder.", "expanded_content": "Encoder with contrastive loss.",
        "figure_index": 8, "evidence_facets": ["architecture"], "parent_chunk_id": "old-parent",
        "page_context_chunk_ids": ["old-text"],
    }
    result = RagService(db_path).expand_with_page_artifacts([original], query="dataset statistics")
    assert result[0]["figure_id"] == "backbone"
    assert "[same-page canonical artifact context]" in result[0]["expanded_content"]
    table = next(item for item in result if item.get("table_id"))
    assert "Train" in table["content"]
    assert table["chunk_type"] == "table"
    assert table["table_index"] == 0
    assert table.get("revision_id")
    for key in ("expanded_content", "figure_index", "evidence_facets", "parent_chunk_id", "page_context_chunk_ids"):
        assert key not in table
    assert "Encoder" not in table["content"]


def test_unpaged_formats_do_not_project_same_page_artifacts(tmp_path) -> None:
    # Markdown/DOCX parsers put every artifact on page 1; projecting "same
    # page" artifacts would attach unrelated tables from the whole file.
    db_path = tmp_path / "app.db"
    source = tmp_path / "report.md"
    source.write_text("# Report\n\nTable 3: Dataset statistics\n| Split | Count |\n|---|---|\n| Train | 42 |\n", encoding="utf-8")
    init_db(db_path)
    document = IndexingService(db_path).index_file(source_path=str(source))
    figure = {"document_id": document.id, "page_number": 1, "filename": "report.md",
              "chunk_id": "figure:f", "figure_id": "f", "artifact_type": "figure",
              "chunk_type": "figure", "caption": "Architecture", "content": "Encoder."}
    assert RagService(db_path).expand_with_page_artifacts([figure], query="statistics") == [figure]


def test_unpaged_figures_are_not_deduplicated_by_page() -> None:
    from app.rag.context import _prioritize_results_for_context

    figures = [
        {"document_id": "d", "filename": "paper.md", "page_number": 1, "chunk_id": f"figure:{i}",
         "figure_id": str(i), "artifact_type": "figure", "score": 1.0 - i / 10}
        for i in range(2)
    ]
    ordered = _prioritize_results_for_context(
        figures, min_figures=2, min_tables=0, min_visual_pages=0, query="architecture",
    )
    assert [item["figure_id"] for item in ordered[:2]] == ["0", "1"]


def test_rag_service_attaches_active_page_artifact_context_to_visual_hit(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    source = tmp_path / "artifact-page.md"
    source.write_text("# Results\n\nA visual page contains a table.\n", encoding="utf-8")
    init_db(db_path)
    indexed = IndexingService(db_path).index_file(source_path=str(source))
    with connect(db_path) as connection:
        revision_id = connection.execute(
            "SELECT active_revision_id FROM documents WHERE id = ?",
            (indexed.id,),
        ).fetchone()[0]
        table_id = str(uuid4())
        connection.execute(
            """
            INSERT INTO document_tables
                (id, document_id, table_index, page_number, caption, markdown,
                 extraction_method, revision_id, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                table_id,
                indexed.id,
                0,
                1,
                "Table 1: External data sources",
                "Source | Dataset\\nRaw data | Gigaword\\nPOS | People's Daily",
                "test",
                revision_id,
                json.dumps({
                    "ocr_text": "Raw data Gigaword; POS People's Daily",
                    "artifact_context": (
                        "The evaluation compares external data sources used by the model. "
                        "Raw data comes from Gigaword and POS comes from People's Daily."
                    ),
                }),
            ),
        )

    visual_hit = {
        "document_id": indexed.id,
        "page_number": 1,
        "chunk_id": f"page:{indexed.id}:1",
        "chunk_type": "page",
        "artifact_type": "page",
        "visual_page": True,
        "content": "Visual page evidence selected from the active revision.",
        "retrieval_channels": ["visual_page", "colsmol"],
    }
    expanded = RagService(db_path).expand_with_page_artifacts(
        [visual_hit], query="What external data sources are used?"
    )

    artifact = next(item for item in expanded if item.get("table_id") == table_id)
    assert "Gigaword" in artifact["content"]
    assert "external data sources used" in artifact["content"]
    page = next(item for item in expanded if item.get("visual_page"))
    assert "[same-page canonical artifact context]" in page["expanded_content"]
    assert page["page_artifact_ids"] == [table_id]


def test_rag_service_does_not_attach_retired_page_artifact(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    source = tmp_path / "artifact-revision.md"
    source.write_text("# Results\n\nCurrent page.\n", encoding="utf-8")
    init_db(db_path)
    indexed = IndexingService(db_path).index_file(source_path=str(source))
    with connect(db_path) as connection:
        revision_id = connection.execute(
            "SELECT active_revision_id FROM documents WHERE id = ?",
            (indexed.id,),
        ).fetchone()[0]
        connection.execute(
            """
            INSERT INTO document_figures
                (id, document_id, figure_index, page_number, caption,
                 extraction_method, revision_id, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid4()), indexed.id, 0, 1, "Retired figure", "test",
                "retired-revision", json.dumps({"visual_summary": "stale"}),
            ),
        )
    expanded = RagService(db_path).expand_with_page_artifacts(
        [{
            "document_id": indexed.id,
            "page_number": 1,
            "chunk_type": "page",
            "artifact_type": "page",
            "visual_page": True,
            "content": "page",
            "retrieval_channels": ["visual_page"],
        }],
        query="What is shown?",
    )
    assert "same-page canonical artifact context" not in expanded[0].get("expanded_content", "")


def test_rag_service_search_active_artifacts_is_revision_scoped(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    source = tmp_path / "artifact-search.md"
    source.write_text("# Results\n\nCurrent revision.\n", encoding="utf-8")
    init_db(db_path)
    indexed = IndexingService(db_path).index_file(source_path=str(source))
    with connect(db_path) as connection:
        active_revision = connection.execute(
            "SELECT active_revision_id FROM documents WHERE id = ?",
            (indexed.id,),
        ).fetchone()[0]
        connection.execute(
            """
            INSERT INTO document_figures
                (id, document_id, figure_index, page_number, caption,
                 visual_summary, extraction_method, revision_id, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "active-figure",
                indexed.id,
                0,
                2,
                "Figure 1: Active metric comparison",
                "The active figure reports metric improvement.",
                "test",
                active_revision,
                json.dumps({"artifact_context": "active metric result"}),
            ),
        )
        connection.execute(
            """
            INSERT INTO document_figures
                (id, document_id, figure_index, page_number, caption,
                 visual_summary, extraction_method, revision_id, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "retired-figure",
                indexed.id,
                1,
                3,
                "Figure 2: Retired metric comparison",
                "The retired figure must never be retrieved.",
                "test",
                "retired-revision",
                json.dumps({"artifact_context": "retired metric result"}),
            ),
        )
        connection.execute(
            """
            INSERT INTO document_tables
                (id, document_id, table_index, page_number, caption,
                 markdown, extraction_method, revision_id, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "active-table",
                indexed.id,
                0,
                2,
                "Table 1: Active benchmark summary",
                "Model | Metric\nOurs | improved",
                "test",
                active_revision,
                json.dumps({"artifact_context": "active benchmark metric"}),
            ),
        )

    results = RagService(db_path).search_active_artifacts_lexical(
        "What metric improvement is shown?",
        document_ids=[indexed.id],
    )

    assert [item["figure_id"] for item in results if item["figure_id"]] == [
        "active-figure"
    ]
    assert results[0]["revision_id"] == active_revision
    assert results[0]["evidence_id"] == "figure:active-figure"

    table_results = RagService(db_path).search_active_artifacts_lexical(
        "Which benchmark summary table is active?",
        document_ids=[indexed.id],
    )
    assert [item["table_id"] for item in table_results if item["table_id"]] == [
        "active-table"
    ]


def test_artifact_lexical_score_prefers_caption_over_shared_document_context(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    source = tmp_path / "paper.md"
    source.write_text(
        """# Paper

Table 2: Hyper-parameter values
| parameter | value |
| --- | --- |
| beam | 5 |

Table 3: Statistics of external data sources
| source | count |
| --- | --- |
| Gigaword | 116m |
""",
        encoding="utf-8",
    )
    init_db(db_path)
    indexed = IndexingService(db_path).index_file(source_path=str(source))
    with connect(db_path) as connection:
        rows = connection.execute(
            "SELECT id, metadata_json FROM document_tables WHERE document_id = ?",
            (indexed.id,),
        ).fetchall()
        for row in rows:
            metadata = json.loads(row["metadata_json"] or "{}")
            metadata["artifact_context"] = (
                "Table 2 hyper-parameters; Table 3 external data sources."
            )
            connection.execute(
                "UPDATE document_tables SET metadata_json = ? WHERE id = ?",
                (json.dumps(metadata), row["id"]),
            )

    results = RagService(db_path).search_active_artifacts_lexical(
        "Which external data sources are used?",
        document_ids=[indexed.id],
        top_k=4,
    )

    assert results
    assert results[0]["caption"].startswith("Table 3")
    expanded = RagService(db_path).expand_with_page_artifacts(
        results,
        query="Which external data sources are used?",
        max_artifacts=3,
    )
    assert expanded[0]["caption"].startswith("Table 3")


def test_artifact_lexical_deduplicates_table_pixel_alias(tmp_path) -> None:
    """A table row and its mirrored image row consume one canonical rank."""

    db_path = tmp_path / "app.db"
    source = tmp_path / "paper.md"
    source.write_text("# Paper\n\nResults.\n", encoding="utf-8")
    init_db(db_path)
    indexed = IndexingService(db_path).index_file(source_path=str(source))
    with connect(db_path) as connection:
        revision_id = connection.execute(
            "SELECT active_revision_id FROM documents WHERE id = ?",
            (indexed.id,),
        ).fetchone()[0]
        shared = {
            "external_artifact_file": "7-Table1-1.png",
            "image_sha256": "same-image",
            "asset_type": "table",
        }
        connection.execute(
            """
            INSERT INTO document_tables
                (id, document_id, table_index, page_number, caption, markdown,
                 extraction_method, revision_id, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "table-canonical",
                indexed.id,
                0,
                7,
                "Table 1: Crosslingual similarity results",
                "| model | score |\n| --- | --- |\n| ours | .8 |",
                "adapter",
                revision_id,
                json.dumps(shared),
            ),
        )
        connection.execute(
            """
            INSERT INTO document_figures
                (id, document_id, figure_index, page_number, caption,
                 visual_summary, extraction_method, revision_id, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "figure-mirror",
                indexed.id,
                0,
                7,
                "7-Table1-1.png: Table 1: Crosslingual similarity results",
                "same pixels",
                "adapter",
                revision_id,
                json.dumps(shared),
            ),
        )

    results = RagService(db_path).search_active_artifacts_lexical(
        "Which crosslingual similarity results are reported?",
        document_ids=[indexed.id],
        top_k=4,
    )

    assert len(
        [item for item in results if item.get("metadata", {}).get("external_artifact_file") == "7-Table1-1.png"]
    ) == 1
    assert results[0]["artifact_type"] == "table"


def test_hybrid_search_routes_unscoped_explicit_artifact_locator(tmp_path) -> None:
    """A numbered artifact must route its owning document before vectors run."""

    db_path = tmp_path / "app.db"
    source = tmp_path / "coffee-manual.md"
    source.write_text("# Coffee manual\n\nGeneral product notes.\n", encoding="utf-8")
    init_db(db_path)
    indexed = IndexingService(db_path).index_file(source_path=str(source))
    with connect(db_path) as connection:
        revision_id = connection.execute(
            "SELECT active_revision_id FROM documents WHERE id = ?",
            (indexed.id,),
        ).fetchone()[0]
        connection.execute(
            """
            INSERT INTO document_figures
                (id, document_id, figure_index, page_number, caption,
                 visual_summary, extraction_method, revision_id, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "coffee-figure-89",
                indexed.id,
                88,
                118,
                "Figure 89: Coffee machine control panel",
                "The picture shows the coffee machine control panel.",
                "test",
                revision_id,
                json.dumps({"figure_number": 89, "figure_label": "Figure 89"}),
            ),
        )

    class EmptyRetrievalStore:
        async def search_document_cards(self, **_kwargs):
            return []

        async def search_text_chunks(self, **_kwargs):
            return []

        async def search_table_chunks(self, **_kwargs):
            return []

        async def search_figure_chunks(self, **_kwargs):
            return []

    result = asyncio.run(
        RagService(db_path).search_hybrid(
            query="In Figure 89, what is shown in the coffee machine picture?",
            top_k=5,
            retrieval_store=EmptyRetrievalStore(),
            embeddings=HashEmbeddingProvider(dimensions=16),
            rerank=False,
        )
    )

    assert result["selected_document_ids"] == [indexed.id]
    assert any(item.get("figure_id") == "coffee-figure-89" for item in result["results"])


def test_explicit_page_hint_candidates_cover_image_only_pages(tmp_path) -> None:
    """Page/slide hints become canonical candidates even without OCR text."""

    db_path = tmp_path / "app.db"
    source = tmp_path / "slides.md"
    source.write_text("# Slide deck\n\nOnly the cover has text.\n", encoding="utf-8")
    init_db(db_path)
    indexed = IndexingService(db_path).index_file(source_path=str(source))
    with connect(db_path) as connection:
        connection.execute(
            "UPDATE documents SET page_count = ? WHERE id = ?",
            (14, indexed.id),
        )

    candidates = RagService(db_path).explicit_page_hint_candidates(
        "Which stages are shown on slides 13 and 14?",
        document_ids=[indexed.id],
    )

    assert [item["page_number"] for item in candidates] == [13, 14]
    assert all(item["explicit_page_hint"] for item in candidates)
    assert all(item["revision_id"] for item in candidates)
    assert all(item["artifact_type"] == "page" for item in candidates)


def test_rag_service_keeps_exact_value_sibling_for_low_signal_text_hit(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    source = tmp_path / "form.md"
    source.write_text(
        "# Application\n\n"
        "APPLICATION FOR RETURN OF COLLATERAL\n\n"
        "Date\n\n"
        "10/06/14\n",
        encoding="utf-8",
    )
    init_db(db_path)
    indexed = IndexingService(db_path).index_file(source_path=str(source))
    with connect(db_path) as connection:
        rows = connection.execute(
            "SELECT id, page_number, content FROM chunks WHERE document_id = ? ORDER BY chunk_index",
            (indexed.id,),
        ).fetchall()
    heading = next(row for row in rows if "APPLICATION FOR RETURN" in row["content"])
    expanded = RagService(db_path).expand_with_page_context(
        [
            {
                "chunk_id": heading["id"],
                "id": heading["id"],
                "document_id": indexed.id,
                "page_number": 1,
                "chunk_type": "text",
                "content": heading["content"],
            }
        ],
        query="When was the application date? Format as YYYY-MM-DD",
        max_chars=1_200,
        max_chunks=2,
    )

    assert "10/06/14" in expanded[0]["expanded_content"]
    assert expanded[0]["page_context_chunk_ids"]


def test_rag_service_neighbor_expansion_respects_section_metadata(tmp_path) -> None:
    db_path = tmp_path / "app.db"
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "sections.md").write_text(
        "\n\n".join(
            [
                "# Related Work",
                "Prior work context. " * 80,
                "# Method",
                "Method target details. " * 80,
                "Method continuation details. " * 80,
            ]
        ),
        encoding="utf-8",
    )

    init_db(db_path)
    indexing = IndexingService(db_path)
    rag = RagService(db_path)
    result = indexing.index_folder(
        folder_path=str(docs_dir),
        recursive=False,
        file_types=["md"],
    )
    chunks = rag.list_document_chunks(result.documents[0].id)
    method_chunk = next(chunk for chunk in chunks if chunk["metadata"].get("section_title") == "Method")
    expanded = rag.expand_with_neighbor_chunks(
        [
            {
                "chunk_id": method_chunk["id"],
                "document_id": method_chunk["document_id"],
                "filename": method_chunk["filename"],
                "source_path": method_chunk["source_path"],
                "content": method_chunk["content"],
                "chunk_index": method_chunk["chunk_index"],
                "page_number": method_chunk["page_number"],
                "chunk_type": "text",
                "section_title": "Method",
            }
        ],
        window=2,
    )

    assert "Prior work context" not in expanded[0].get("expanded_content", "")


def test_parse_docx(tmp_path) -> None:
    from docx import Document

    path = tmp_path / "note.docx"
    document = Document()
    document.add_paragraph("Docx retrieval works.")
    document.save(path)

    assert "Docx retrieval works." in parse_text_file(path)


def test_pdf_image_extraction_saves_figure_artifact(tmp_path) -> None:
    import pytest

    fitz = pytest.importorskip("fitz")
    path = tmp_path / "visual.pdf"
    document = fitz.open()
    page = document.new_page()
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 40, 40), 0)
    pixmap.clear_with(0x22AA66)
    page.insert_image(fitz.Rect(50, 50, 120, 120), pixmap=pixmap)
    page.insert_text((50, 140), "Figure 1: Green architecture block")
    document.save(path)

    parsed = parse_document(
        path,
        artifact_dir=tmp_path / "artifacts",
        vision_summarizer=lambda image_path, caption, page_text: f"Visual summary for {caption}",
    )
    figures = [figure for figure in parsed.figures or [] if figure.image_path]

    assert figures
    assert parsed.parser_name in {"docling", "pypdf+pymupdf"}
    assert Path(figures[0].image_path or "").exists()
    assert figures[0].visual_summary == "Visual summary for Figure 1: Green architecture block"
    assert figures[0].extraction_method in {"docling_picture", "pymupdf_caption_crop"}
    assert figures[0].bbox is not None
    assert figures[0].metadata["source"] in {"docling", "pymupdf_caption_crop"}
    assert figures[0].metadata["width"] > 40


def test_pdf_ocr_gate_is_document_level_and_not_filename_specific() -> None:
    from app.rag.parsers import ParsedPage, _needs_pdf_ocr

    scanned = [ParsedPage(page_number=index, text="") for index in range(1, 11)]
    normal = [
        ParsedPage(page_number=index, text="searchable scientific paragraph " * 8)
        for index in range(1, 8)
    ] + [ParsedPage(page_number=8, text="")]
    sparse_but_usable = [
        ParsedPage(page_number=index, text="searchable content " * 4)
        for index in range(1, 5)
    ] + [ParsedPage(page_number=index, text="") for index in range(5, 11)]

    assert _needs_pdf_ocr(scanned) is True
    assert _needs_pdf_ocr(normal) is False
    # Four usable pages out of ten are enough to preserve the fast text path.
    assert _needs_pdf_ocr(sparse_but_usable) is False


def test_page_ocr_fallback_keeps_searchable_pages_and_page_provenance(
    tmp_path, monkeypatch
) -> None:
    from types import ModuleType, SimpleNamespace

    from app.rag.parsers import ParsedPage, _ocr_sparse_pdf_pages

    class FakePage:
        def get_pixmap(self, **kwargs):  # noqa: ANN003
            return SimpleNamespace(tobytes=lambda _: b"page-image")

    class FakeDocument:
        def __enter__(self):
            return self

        def __exit__(self, *args):  # noqa: ANN002
            return None

        def __iter__(self):
            return iter([FakePage(), FakePage()])

    fake_fitz = ModuleType("fitz")
    fake_fitz.open = lambda _: FakeDocument()
    fake_fitz.Matrix = lambda *_: object()

    class FakeRapidOCR:
        def __init__(self, **kwargs):  # noqa: ANN003
            pass

        def __call__(self, payload):  # noqa: ANN001
            return SimpleNamespace(txts=("OCR title", "OCR value 42"))

    monkeypatch.setitem(__import__("sys").modules, "fitz", fake_fitz)
    monkeypatch.setattr(
        "app.rag.parsers.get_rapidocr_engine",
        lambda: (FakeRapidOCR(), None),
    )

    pages = [
        ParsedPage(1, "existing searchable text " * 4),
        ParsedPage(2, ""),
    ]
    result = _ocr_sparse_pdf_pages(tmp_path / "scanned.pdf", pages)

    assert result[0] == pages[0]
    assert result[1].page_number == 2
    assert result[1].text == "OCR title\nOCR value 42"


def test_page_ocr_fallback_bounds_sparse_pages_with_even_coverage(
    tmp_path, monkeypatch
) -> None:
    from types import ModuleType, SimpleNamespace

    from app.rag.parsers import ParsedPage, _ocr_sparse_pdf_pages

    calls: list[bytes] = []

    class FakePage:
        def get_pixmap(self, **kwargs):  # noqa: ANN003
            return SimpleNamespace(tobytes=lambda _: b"page-image")

    class FakeDocument:
        def __enter__(self):
            return self

        def __exit__(self, *args):  # noqa: ANN002
            return None

        def __iter__(self):
            return iter([FakePage() for _ in range(10)])

    fake_fitz = ModuleType("fitz")
    fake_fitz.open = lambda _: FakeDocument()
    fake_fitz.Matrix = lambda *_: object()
    monkeypatch.setitem(__import__("sys").modules, "fitz", fake_fitz)

    class FakeRapidOCR:
        def __call__(self, payload):  # noqa: ANN001
            calls.append(payload)
            return SimpleNamespace(txts=("OCR page",))

    monkeypatch.setattr(
        "app.rag.parsers.get_rapidocr_engine",
        lambda: (FakeRapidOCR(), None),
    )
    pages = [ParsedPage(index, "") for index in range(1, 11)]
    result = _ocr_sparse_pdf_pages(tmp_path / "scanned.pdf", pages, max_pages=3)

    assert len(calls) == 3
    assert [page.page_number for page in result if page.text] == [1, 5, 10]
    assert sum(bool(page.text) for page in result) == 3


def test_ocr_repairs_high_confidence_identifier_split() -> None:
    from app.rag.parsers import _repair_ocr_token_boundaries

    assert _repair_ocr_token_boundaries("coba lt-42 and ordinary words") == (
        "cobalt-42 and ordinary words"
    )


def test_figure_identity_keeps_duplicate_labels_distinct() -> None:
    from types import SimpleNamespace

    from app.services.indexing_service import _stable_figure_id

    first = SimpleNamespace(
        figure_index=0,
        page_number=2,
        caption="Figure 1",
        image_path=None,
        visual_summary=None,
        extraction_method="pymupdf_caption_crop",
        bbox=None,
        metadata={"figure_label": "Figure 1"},
    )
    second = SimpleNamespace(**{**first.__dict__, "figure_index": 1})

    assert _stable_figure_id("doc", first, revision_id="rev") != _stable_figure_id(
        "doc", second, revision_id="rev"
    )


def test_figure_caption_requires_separator(tmp_path) -> None:
    path = tmp_path / "captions.md"
    path.write_text(
        "\n".join(
            [
                "Figure 2 complements Table 4 with additional results.",
                "Figure 3: Real architecture caption",
                "Fig. 4. Another real caption",
            ]
        ),
        encoding="utf-8",
    )

    parsed = parse_document(path)
    captions = [figure.caption for figure in parsed.figures or []]

    assert captions == ["Figure 3: Real architecture caption", "Fig. 4. Another real caption"]


def test_figure_caption_merges_continuation_lines(tmp_path) -> None:
    path = tmp_path / "multiline-caption.md"
    path.write_text(
        "\n".join(
            [
                "Fig. 3. The key-sparse attention in KS-Transformer.",
                "In this module, only top-k keys are kept.",
                "Fig. 4. The details of CCAB.",
            ]
        ),
        encoding="utf-8",
    )

    parsed = parse_document(path)
    captions = [figure.caption for figure in parsed.figures or []]

    assert captions == [
        "Fig. 3. The key-sparse attention in KS-Transformer. In this module, only top-k keys are kept.",
        "Fig. 4. The details of CCAB.",
    ]


def test_scanned_page_ocr_keeps_table_rows_from_the_layout() -> None:
    from app.rag.parsers import _ocr_layout_text

    class _Layout:
        txts = ("Federal funds rate", "3.8", "3.6", "March projection", "3.4", "3.1")

        def to_markdown(self) -> str:
            return "Federal funds rate    3.8 3.6\nMarch projection   3.4 3.1"

    class _Plain:
        txts = ("line one", " ", "line two")

    assert _ocr_layout_text(_Layout()).splitlines() == ["Federal funds rate    3.8 3.6", "March projection   3.4 3.1"]
    assert _ocr_layout_text(_Plain()) == "line one\nline two"
