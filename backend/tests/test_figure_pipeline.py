from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.rag.parsers import ParsedPage, _docling_logical_figures


class FakePicture:
    def __init__(
        self,
        *,
        page_number: int,
        bbox: tuple[float, float, float, float],
        color: tuple[int, int, int],
        caption: str | None = None,
        size: tuple[int, int] = (220, 180),
    ) -> None:
        image_module = pytest.importorskip("PIL.Image")
        self._image = image_module.new("RGB", size, color)
        self._caption = caption
        self.prov = [
            SimpleNamespace(
                page_no=page_number,
                bbox=SimpleNamespace(
                    l=bbox[0],
                    t=bbox[1],
                    r=bbox[2],
                    b=bbox[3],
                    coord_origin=SimpleNamespace(value="BOTTOMLEFT"),
                ),
            )
        ]

    def get_image(self, _document):
        return self._image

    def caption_text(self, _document) -> str:
        return self._caption or ""


class FakeDocument:
    def __init__(self, pictures: list[FakePicture]) -> None:
        self.pictures = pictures

    def iterate_items(self):
        return iter((picture, 0) for picture in self.pictures)


def _blank_pdf(path: Path, *, pages: int = 9) -> None:
    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    for _ in range(pages):
        document.new_page(width=600, height=840)
    document.save(path)


def _cmdm_pictures() -> list[FakePicture]:
    return [
        FakePicture(page_number=9, bbox=(63.3, 697.3, 159.4, 603.0), color=(220, 20, 20)),
        FakePicture(page_number=9, bbox=(60.4, 587.5, 159.0, 493.1), color=(20, 220, 20)),
        FakePicture(page_number=9, bbox=(181.6, 696.1, 279.5, 602.9), color=(20, 20, 220)),
        FakePicture(page_number=9, bbox=(181.0, 587.4, 279.1, 494.3), color=(220, 180, 20)),
        # One crop containing the separate two-chart Figure 7.
        FakePicture(
            page_number=9,
            bbox=(319.1, 692.3, 546.8, 575.4),
            color=(120, 40, 180),
            caption="Figure 7: Training and validation curves",
        ),
    ]


def test_docling_cmdm_panels_become_stable_logical_figures(tmp_path: Path) -> None:
    path = tmp_path / "CMDM.pdf"
    _blank_pdf(path)
    pages = [ParsedPage(page_number=index, text=f"Page {index}") for index in range(1, 10)]
    captions = {
        9: [
            "Figure 6: Confusion matrices for four settings",
            "Figure 7: Training and validation curves",
        ]
    }

    original = _docling_logical_figures(
        path=path,
        document=FakeDocument(_cmdm_pictures()),
        picture_item_type=FakePicture,
        pages=pages,
        captions_by_page=captions,
        artifact_dir=tmp_path / "original",
        vision_summarizer=None,
    )
    shuffled = _docling_logical_figures(
        path=path,
        document=FakeDocument(list(reversed(_cmdm_pictures()))),
        picture_item_type=FakePicture,
        pages=pages,
        captions_by_page=captions,
        artifact_dir=tmp_path / "shuffled",
        vision_summarizer=None,
    )

    assert [figure.metadata["child_count"] for figure in original] == [4, 1]
    assert [figure.metadata["figure_number"] for figure in original] == [6, 7]
    assert [figure.metadata["caption_source"] for figure in original] == [
        "fallback_sequence",
        "docling_direct",
    ]
    assert original[0].extraction_method == "docling_logical_composite"
    assert Path(original[0].image_path or "").is_file()
    assert len(original[0].metadata["children"]) == 4
    assert {
        figure.metadata["logical_group_id"] for figure in original
    } == {
        figure.metadata["logical_group_id"] for figure in shuffled
    }


def test_repeated_logo_is_rejected_without_consuming_page_caption(tmp_path: Path) -> None:
    path = tmp_path / "paper.pdf"
    _blank_pdf(path)
    logo_color = (15, 15, 15)
    pictures = [
        FakePicture(
            page_number=8,
            bbox=(15, 835, 95, 812),
            color=logo_color,
            size=(240, 70),
        ),
        FakePicture(
            page_number=9,
            bbox=(15, 835, 95, 812),
            color=logo_color,
            size=(240, 70),
        ),
        FakePicture(
            page_number=9,
            bbox=(110, 620, 510, 340),
            color=(30, 130, 220),
            size=(500, 300),
        ),
    ]
    pages = [ParsedPage(page_number=index, text=f"Page {index}") for index in range(1, 10)]

    figures = _docling_logical_figures(
        path=path,
        document=FakeDocument(pictures),
        picture_item_type=FakePicture,
        pages=pages,
        captions_by_page={9: ["Figure 6: Main scientific result"]},
        artifact_dir=tmp_path / "artifacts",
        vision_summarizer=None,
    )

    logos = [figure for figure in figures if figure.metadata["asset_kind"] == "branding"]
    content = next(figure for figure in figures if figure.metadata.get("figure_number") == 6)
    assert len(logos) == 2
    assert all(figure.metadata["quality_status"] == "rejected" for figure in logos)
    assert all(figure.metadata["caption_source"] == "none" for figure in logos)
    assert content.metadata["caption_source"] == "fallback_sequence"
    assert content.caption == "Figure 6: Main scientific result"
