from pathlib import Path
import json

import pytest

from app.db.sqlite import connect, init_db
from app.services.indexing_service import IndexingService
from app.services.rag_service import RagService
from app.rag.artifact_references import extract_artifact_references
from app.rag.context import compose_retrieval_context


@pytest.mark.parametrize("text, expected", [
    ("Figs. 2–4 and Table IX", [("figure", "2"), ("figure", "3"), ("figure", "4"), ("table", "9")]),
    ("Bảng số 2.1; Hình A1 và A2", [("table", "2.1"), ("figure", "A1"), ("figure", "A2")]),
    ("Tables IV, V and VI; Figure 2 & 3", [("table", "4"), ("table", "5"), ("table", "6"), ("figure", "2"), ("figure", "3")]),
    ("Table TABREF18 and Figure FIGREF7", []),
    ("Table IIV, Figure 0, table of contents, configuration", []),
    ("Figures 1–9999", [("figure", "1")]),
    ("Figure 2 and Table 3; Fig. 2", [("figure", "2"), ("table", "3")]),
    # "mô hình" is Vietnamese for "model", never a figure label.
    ("Mô hình 2 đạt độ chính xác cao hơn mô hình 1; xem hình 4", [("figure", "4")]),
    ("mô hình V-JEPA và mô hình MIX", []),
    # Lower-case or single non-IVX letters are words / appendix labels.
    ("Table iv, see Table C, the table, mix", []),
    # Sub-panels resolve to the parent artifact.
    ("Fig. 2a, Figures 3(b) and 3c", [("figure", "2"), ("figure", "3")]),
    # Abbreviated, business and Vietnamese chart labels; appendix numbering.
    ("Tab. 4, Exhibit 2 and Chart 7", [("table", "4"), ("table", "2"), ("figure", "7")]),
    ("Biểu đồ 5, sơ đồ 6 và đồ thị 8", [("figure", "5"), ("figure", "6"), ("figure", "8")]),
    ("Table B.3", [("table", "B3")]),
    # A year after "and" is not another artifact number.
    ("Table 3 and 2021 reports", [("table", "3")]),
])
def test_printed_reference_labels_are_bounded_and_not_guessed(text, expected):
    assert [(ref.kind, ref.number) for ref in extract_artifact_references(text)] == expected


def _corpus(tmp_path: Path):
    db = tmp_path / "app.db"
    init_db(db)
    source = tmp_path / "manual.md"
    source.write_text("# Manual\n\nA staged installation has independent safety steps.\n")
    doc = IndexingService(db).index_file(source_path=str(source))
    with connect(db) as c:
        revision = c.execute("SELECT active_revision_id FROM documents WHERE id=?", (doc.id,)).fetchone()[0]
        c.execute(
            """INSERT INTO document_tables
               (id,document_id,revision_id,table_index,page_number,caption,markdown,extraction_method,metadata_json)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            ("limits", doc.id, revision, 0, 8, "Table IX. Electrical safety limits",
             "| System | Voltage |\n|---|---|\n| ORION | 48 |", "test", "{}"),
        )
        for key, rev, caption in (
            ("sequence", revision, "Figure 4. Installation sequence"),
            ("retired", "old", "Figure 5. Retired installation sequence"),
            ("staged", "pending", "Figure 6. Unactivated installation sequence"),
        ):
            c.execute(
                """INSERT INTO document_figures
                   (id,document_id,revision_id,figure_index,page_number,caption,visual_summary,extraction_method,metadata_json)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (key, doc.id, rev, 12, 12, caption, "Connect, verify, then energize.", "test",
                 json.dumps({"image_path": "/not-read-by-retrieval/sequence.png"})),
            )
    return RagService(db), doc, revision


@pytest.mark.parametrize("reference", [
    "Installation is described in Fig. 4 and safety limits in Table IX.",
    "Quy trình lắp đặt ở Hình 4, giới hạn an toàn ở Bảng 9.",
])
def test_text_reference_resolves_cross_page_artifacts_without_visual_query(tmp_path, reference):
    rag, doc, revision = _corpus(tmp_path)
    source = {"document_id": doc.id, "revision_id": revision, "page_number": 2,
              "chunk_id": "procedure", "chunk_type": "text", "content": reference,
              "parent_chunk_id": "parent", "evidence_facets": ["task"]}
    expanded = rag.expand_with_referenced_artifacts([source], query="How is the installation performed?")
    assert expanded[0] is source
    figure = next(item for item in expanded if item.get("figure_id") == "sequence")
    table = next(item for item in expanded if item.get("table_id") == "limits")
    assert figure["page_number"] == 12 and table["page_number"] == 8
    assert figure["revision_id"] == table["revision_id"] == revision
    assert figure["figure_index"] == 12  # Never index+1 -> Figure N.
    assert figure["retrieval_channels"] == ["artifact_reference"]
    assert figure["artifact_reference_from"] == "procedure"
    assert figure.get("image_path")
    assert "parent_chunk_id" not in figure and "evidence_facets" not in table


def test_references_do_not_resolve_retired_staged_or_other_document(tmp_path):
    rag, doc, revision = _corpus(tmp_path)
    source = {"document_id": doc.id, "revision_id": revision, "chunk_type": "text",
              "content": "See Figures 5 and 6 for the procedure."}
    assert rag.expand_with_referenced_artifacts([source]) == [source]
    source["content"] = "See Figure 4."
    source["revision_id"] = "retired"
    assert rag.expand_with_referenced_artifacts([source]) == [source]
    source["revision_id"] = revision
    source["document_id"] = "not-in-this-corpus"
    assert rag.expand_with_referenced_artifacts([source]) == [source]


def test_unresolved_reference_tokens_and_ambiguous_labels_are_not_guessed(tmp_path):
    rag, doc, revision = _corpus(tmp_path)
    source = {"document_id": doc.id, "revision_id": revision, "chunk_type": "text",
              "content": "See Figure FIGREF4 and Table TABREF9."}
    assert rag.expand_with_referenced_artifacts([source]) == [source]
    with connect(rag.db_path) as c:
        c.execute(
            """INSERT INTO document_figures
               (id,document_id,revision_id,figure_index,page_number,caption,extraction_method,metadata_json)
               VALUES (?,?,?,?,?,?,?,?)""",
            ("ambiguous", doc.id, revision, 1, 3, "Figure 4. Another sequence", "test", "{}"),
        )
    source["content"] = "See Figure 4."
    assert rag.expand_with_referenced_artifacts([source]) == [source]


def test_parent_reference_deduplicates_existing_artifact_and_obeys_budget(tmp_path):
    rag, doc, revision = _corpus(tmp_path)
    source = {"document_id": doc.id, "revision_id": revision, "chunk_id": "child",
              "chunk_type": "text", "content": "Safety instructions",
              "expanded_content": "The installation sequence is in Figure 4; limits are in Table 9."}
    expanded = rag.expand_with_referenced_artifacts([source], max_artifacts=1)
    assert len(expanded) == 2
    expanded = rag.expand_with_referenced_artifacts(expanded, max_artifacts=1)
    assert sum(item.get("figure_id") == "sequence" for item in expanded) == 1
    assert sum(item.get("table_id") == "limits" for item in expanded) <= 1


@pytest.mark.parametrize("labels", [
    {"metadata": {"section_title": "Figures and tables"}},
    {"metadata": {"heading_path": ["Manual", "References"]}},
    {"heading_path": ["Manual", "References"]},
])
def test_catalog_and_bibliography_references_do_not_expand(tmp_path, labels):
    rag, doc, revision = _corpus(tmp_path)
    source = {"document_id": doc.id, "revision_id": revision, "chunk_type": "text",
              "content": "Figure 4. Installation sequence", **labels}
    assert rag.expand_with_referenced_artifacts([source]) == [source]


def test_referenced_image_reaches_final_context_without_image_request(tmp_path):
    rag, doc, revision = _corpus(tmp_path)
    source = {"document_id": doc.id, "revision_id": revision, "chunk_type": "text",
              "chunk_id": "installation", "page_number": 2,
              "content": "The installation procedure follows Figure 4. Verify the circuit before energizing."}
    expanded = rag.expand_with_referenced_artifacts([source], query="How is the installation performed?")
    context = compose_retrieval_context(
        expanded, query="How is the installation performed?", max_sources=2,
        required_document_ids=[doc.id],
    )
    figure = next(row for row in context.sources if row.get("figure_id") == "sequence")
    assert figure["image_path"] == "/not-read-by-retrieval/sequence.png"
    assert figure["page_number"] == 12
    assert "Connect, verify, then energize" in context.context_text


def test_reference_expansion_caps_per_paragraph_and_prefers_relevant_artifacts(tmp_path):
    rag, doc, revision = _corpus(tmp_path)
    with connect(rag.db_path) as c:
        for index, caption in ((1, "Table 1. Wiring colours"), (2, "Table 2. Torque values"),
                               (3, "Table 3. Grounding resistance")):
            c.execute(
                """INSERT INTO document_tables
                   (id,document_id,revision_id,table_index,page_number,caption,markdown,extraction_method,metadata_json)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (f"t{index}", doc.id, revision, index, 3, caption, "| A | B |\n|---|---|\n| x | 1 |", "test", "{}"),
            )
    source = {"document_id": doc.id, "revision_id": revision, "chunk_id": "p", "chunk_type": "text",
              "content": "See Tables 1, 2 and 3 and Table IX."}
    expanded = rag.expand_with_referenced_artifacts([source], query="What grounding resistance is required?")
    injected = [item["table_id"] for item in expanded[1:]]
    assert len(injected) == 2
    assert injected[0] == "t3"
