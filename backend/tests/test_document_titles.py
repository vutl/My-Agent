from pathlib import Path

import pymupdf

from app.catalog.titles import guess_title, pdf_title, text_title


def _pdf(tmp_path: Path, lines: list[tuple[str, float, bool]], *, stamp: str | None = None) -> Path:
    document = pymupdf.open()
    page = document.new_page()
    y = 72.0
    for text, size, bold in lines:
        page.insert_text((72, y), text, fontsize=size, fontname="hebo" if bold else "helv")
        y += size * 1.6
    if stamp:
        page.insert_text((20, 500), stamp, fontsize=20, rotate=90)
    path = tmp_path / "paper.pdf"
    document.save(path)
    return path


def test_largest_font_title_beats_author_and_body_lines(tmp_path):
    path = _pdf(tmp_path, [
        ("Graph Attention for Traffic Forecasting", 18, True),
        ("Alice Smith, Bob Lee", 11, False),
        ("Department of Computing, Some University", 10, False),
    ])
    assert pdf_title(path) == "Graph Attention for Traffic Forecasting"


def test_rotated_arxiv_stamp_is_not_the_title(tmp_path):
    path = _pdf(tmp_path, [("Sparse Attention for Speech", 12, True), ("Carol Wu", 12, False)],
                stamp="arXiv:2106.11532v3 [cs.SD] 27 Feb 2023")
    assert pdf_title(path) == "Sparse Attention for Speech"


def test_bold_title_wraps_but_stops_at_same_size_author_line(tmp_path):
    path = _pdf(tmp_path, [
        ("A ROBUST FUSION MODEL FOR SPEECH", 12, True),
        ("IN LOW RESOURCE LANGUAGES", 12, True),
        ("Nguyen Van A, Tran Thi B", 12, False),
    ])
    assert pdf_title(path) == "A Robust Fusion Model for Speech in Low Resource Languages"


def test_wrapped_title_from_extracted_pdf_text_stops_at_authors():
    text = "1\n\nCrab: Contrastive Supervision to\n\nImprove Recognition Under Noise\n\nLucas Ueda, Student Member, IEEE"
    assert text_title(text) == "Crab: Contrastive Supervision to Improve Recognition Under Noise"


def test_markdown_uses_h1_and_never_body_prose(tmp_path):
    assert text_title("intro line\n# Local RAG demo\n\nBody.", allow_plain_lines=False) == "Local RAG demo"
    no_heading = "> Version: v1\n\n## 0. Purpose\n\nThe parent plan defines the app."
    path = tmp_path / "PLAN_ADDENDUM.md"
    assert guess_title(path, no_heading) == "PLAN ADDENDUM"


def test_vietnamese_admin_headers_and_generic_doc_types():
    from app.catalog.titles import text_title

    text = ("BỘ TÀI CHÍNH\nCỤC THỐNG KÊ\nCỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\nĐộc lập - Tự do - Hạnh phúc\n"
            "Hà Nội, ngày 03 tháng 10 năm 2026\nTHÔNG CÁO BÁO CHÍ\n"
            "Về tình hình lao động, việc làm quý III và chín tháng năm 20261\n"
            "Trong quý III năm 2026, thị trường lao động thế giới nhìn chung duy trì sự ổn định về các chỉ báo việc làm.")
    assert text_title(text) == "THÔNG CÁO BÁO CHÍ Về tình hình lao động, việc làm quý III và chín tháng năm 2026"


def test_setext_heading_and_year_ending_titles(tmp_path):
    import docx
    from app.catalog.titles import docx_title, text_title

    assert text_title("ripgrep (rg)\n------------\nripgrep is a line-oriented search tool.", allow_plain_lines=False) == "ripgrep (rg)"
    document = docx.Document()
    document.core_properties.title = "BỘ KẾ HOẠCH VÀ ĐẦU TƯ"
    for line in ("BÁO CÁO", "TÌNH HÌNH KINH TẾ - XÃ HỘI", "QUÝ II VÀ 6 THÁNG ĐẦU NĂM 2026", "____________________",
                 "Trong sáu tháng đầu năm 2026, tình hình thế giới tiếp tục diễn biến phức tạp."):
        document.add_paragraph(line)
    path = tmp_path / "report.docx"
    document.save(path)
    assert docx_title(path) == "Báo Cáo Tình Hình Kinh Tế - Xã Hội Quý II Và 6 Tháng Đầu Năm 2026"


def test_metadata_line_closing_a_banner_is_not_a_setext_title():
    text = (
        "NOAA Climate Prediction Center - ENSO Diagnostic Discussions (monthly), compiled\n\n"
        "==============\nENSO Diagnostic Discussion issued 8 October 2026\n"
        "Source: https://www.cpc.ncep.noaa.gov/products/ensodisc.shtml\n==============\n"
    )
    assert text_title(text, allow_plain_lines=False) is None
    assert text_title("Lost in the Middle: How Models Use Context\n-----\n", allow_plain_lines=False) == (
        "Lost in the Middle: How Models Use Context"
    )


def test_titles_reject_legacy_encoding_blank_dates_and_section_names() -> None:
    from app.catalog.titles import _ADMIN_HEADER_RE, _plausible, text_title

    assert not _plausible("Bé N«ng nghiÖp vµ PTNT")  # TCVN3 read as Latin-1
    assert _ADMIN_HEADER_RE.match("Hà Nội, ngày    tháng     năm 2026")
    assert _ADMIN_HEADER_RE.match("VĂN BẢN QUY PHẠM PHÁP LUẬT")
    assert not _plausible("Chương I QUY ĐỊNH CHUNG")
    assert not _plausible("Latest Updates -- YOLOE released")
    assert text_title("PEP: 703\nTitle: Making the Global Interpreter Lock Optional in CPython\nAuthor: X") == (
        "Making the Global Interpreter Lock Optional in CPython"
    )
    assert text_title("Abstract\n========\n\nBody text.", allow_plain_lines=False) is None


def test_docx_generic_type_title_takes_its_subject_line(tmp_path) -> None:
    import docx

    from app.catalog.titles import docx_title

    document = docx.Document()
    document.add_paragraph("BỘ Y TẾ")
    document.add_paragraph("Số: 12/2026/TT-BYT")
    document.add_paragraph("THÔNG TƯ", style="Title")
    document.add_paragraph("Quy định Danh mục bệnh bẩm sinh cần sàng lọc trước sinh và sơ sinh")
    path = tmp_path / "tt.docx"
    document.save(str(path))
    assert docx_title(path) == "THÔNG TƯ Quy định Danh mục bệnh bẩm sinh cần sàng lọc trước sinh và sơ sinh"


def test_startup_refresh_rederives_titles_made_under_older_rules(tmp_path) -> None:
    import json

    from app.catalog.title_refresh import refresh_document_titles
    from app.db.sqlite import connect, init_db
    from app.services.indexing_service import IndexingService

    db_path = tmp_path / "app.db"
    source = tmp_path / "sot_ret.md"
    source.write_text("# Sốt rét\n\nSốt rét là bệnh truyền nhiễm do ký sinh trùng.\n", encoding="utf-8")
    init_db(db_path)
    document = IndexingService(db_path).index_file(source_path=str(source))
    with connect(db_path) as connection:  # as indexed under the old rules
        connection.execute(
            "UPDATE documents SET title = 'Abstract', metadata_json = '{}' WHERE id = ?", (document.id,)
        )
    assert refresh_document_titles(db_path) == 1
    assert refresh_document_titles(db_path) == 0  # marked once
    with connect(db_path) as connection:
        row = connection.execute("SELECT title, metadata_json FROM documents WHERE id = ?", (document.id,)).fetchone()
    assert row["title"] == "Sốt rét"
    assert json.loads(row["metadata_json"])["title_rules"]


def test_card_language_needs_a_real_share_of_vietnamese_letters() -> None:
    from app.catalog.document_card import _language

    english = "Observation of gravitational waves from a binary black hole merger. " * 20 + "Đặng"
    assert _language(english) == "en"
    assert _language("Thông tư này quy định danh mục bệnh bẩm sinh cần sàng lọc trước sinh và sơ sinh.") == "vi"


def test_markdown_title_drops_embedded_html() -> None:
    from app.catalog.titles import text_title

    heading = '# Materials Graph Library <img src="https://x/MatGL.png?raw=true" alt="matgl" width="30%">'
    assert text_title(heading, allow_plain_lines=False) == "Materials Graph Library"
