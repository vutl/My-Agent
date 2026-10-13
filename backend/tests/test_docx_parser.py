from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import fitz
import pytest

from app.rag.parsers import (
    ParsedPage,
    _docling_pages,
    _with_printed_page_labels,
    ocr_pdf_page_labels,
    parse_document,
)


def test_docling_page_export_prefers_materially_richer_native_fallback() -> None:
    class Page:
        def export_to_text(self) -> str:
            return "tiny"

    document = SimpleNamespace(pages={1: Page()})
    pages = _docling_pages(
        document,
        [ParsedPage(page_number=1, text="A" * 300, text_origin="native")],
    )

    assert pages[0].text == "A" * 300
    assert pages[0].text_origin == "native"


def test_printed_page_alias_requires_a_sequential_run() -> None:
    pages = [
        ParsedPage(page_number=1, text="Cover\n\n1"),
        ParsedPage(page_number=2, text="Method\n\n2"),
        ParsedPage(page_number=3, text="Results\n\n3"),
    ]
    labelled = _with_printed_page_labels(pages)
    assert [page.page_label for page in labelled] == ["1", "2", "3"]

    noisy = [
        ParsedPage(page_number=1, text="A table has\n\n42"),
        ParsedPage(page_number=2, text="No footer here"),
        ParsedPage(page_number=3, text="A metric is 7"),
    ]
    assert all(page.page_label is None for page in _with_printed_page_labels(noisy))

    roman = [
        ParsedPage(page_number=1, text="Preface\n\ni"),
        ParsedPage(page_number=2, text="Contents\n\nii"),
        ParsedPage(page_number=3, text="Introduction\n\niii"),
    ]
    assert [page.page_label for page in _with_printed_page_labels(roman)] == [
        "i",
        "ii",
        "iii",
    ]


def test_facing_page_spread_labels_are_preserved_and_infer_stride() -> None:
    pages = [
        ParsedPage(page_number=1, text="cover"),
        ParsedPage(page_number=2, text="04 05\ncontent"),
        ParsedPage(page_number=3, text="06 07\ncontent"),
        ParsedPage(page_number=4, text="08 09\ncontent"),
        ParsedPage(page_number=5, text="10 11\ncontent"),
    ]
    labelled = _with_printed_page_labels(pages)
    assert [page.page_label for page in labelled] == [None, "4-5", "6-7", "8-9", "10-11"]


def test_sparse_sequential_labels_interpolate_unobserved_physical_pages() -> None:
    pages = [
        ParsedPage(page_number=page_number, text="")
        for page_number in range(1, 17)
    ]
    labels = {1: "1", 6: "6", 11: "10", 16: "15"}
    pages = [
        ParsedPage(page_number=page.page_number, text=page.text, page_label=labels.get(page.page_number))
        for page in pages
    ]

    labelled = _with_printed_page_labels(pages)

    assert labelled[13].page_label == "13"
    assert labelled[11].page_label == "11"


def test_page_label_ocr_is_cached_for_ingestion_and_warm_reads_skip_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "scan.pdf"
    document = fitz.open()
    document.new_page()
    document.save(source)
    document.close()

    calls = {"count": 0}

    class FakeEngine:
        def __call__(self, _image: bytes) -> SimpleNamespace:
            calls["count"] += 1
            return SimpleNamespace(txts=["13"])

    monkeypatch.setattr(
        "app.rag.parsers.get_rapidocr_engine",
        lambda: (FakeEngine(), None),
    )
    cache_dir = tmp_path / "label-cache"
    assert ocr_pdf_page_labels(source, [1], cache_dir=cache_dir) == {1: "13"}
    assert calls["count"] == 1

    def fail_if_engine_initializes() -> tuple[None, str]:
        raise AssertionError("warm page-label cache must not initialize OCR")

    monkeypatch.setattr("app.rag.parsers.get_rapidocr_engine", fail_if_engine_initializes)
    assert ocr_pdf_page_labels(source, [1], cache_dir=cache_dir) == {1: "13"}


def test_docx_parser_preserves_ordered_table_and_embedded_image(tmp_path: Path) -> None:
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph("A short introduction")
    document.add_paragraph("Table 1: Results")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Metric"
    table.cell(0, 1).text = "Score"
    table.cell(1, 0).text = "Recall"
    table.cell(1, 1).text = "0.91"
    document.add_paragraph("Figure 1: Pipeline")
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    )
    document.add_picture(BytesIO(png))
    source = tmp_path / "report.docx"
    document.save(source)

    parsed = parse_document(source, artifact_dir=tmp_path / "artifacts")

    assert parsed.parser_version == "2-ordered-table-image"
    assert len(parsed.tables or []) == 1
    assert parsed.tables[0].caption == "Table 1: Results"
    assert "| Recall | 0.91 |" in parsed.tables[0].markdown
    assert len(parsed.figures or []) == 1
    assert parsed.figures[0].image_path
    assert Path(parsed.figures[0].image_path).is_file()
    assert parsed.text.index("A short introduction") < parsed.text.index("Metric")
    assert parsed.text.index("Metric") < parsed.text.index("Embedded figure")
