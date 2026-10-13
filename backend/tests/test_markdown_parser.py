from __future__ import annotations

from pathlib import Path

from app.rag.parsers import parse_document


def test_markdown_local_images_are_scoped_to_approved_root(tmp_path: Path) -> None:
    root = tmp_path / "approved"
    root.mkdir()
    image = root / "diagram.png"
    image.write_bytes(b"png-bytes")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    source = root / "notes.md"
    source.write_text(
        "![local](diagram.png)\n![outside](../outside.png)\n![remote](https://example.test/x.png)",
        encoding="utf-8",
    )

    parsed = parse_document(source, approved_root=root)

    assert len(parsed.figures or []) == 1
    assert parsed.figures[0].caption == "local"
    assert parsed.figures[0].extraction_method == "markdown_local_image"
    assert parsed.figures[0].metadata["approved_root"] == str(root.resolve())
