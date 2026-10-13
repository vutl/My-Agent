from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from app.rag.image_ocr import extract_image_ocr


def test_image_ocr_uses_sha_cache_without_reinvoking_engine(tmp_path: Path) -> None:
    image = tmp_path / "figure.png"
    image.write_bytes(b"stable-image-bytes")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / f"{digest}.json").write_text(
        json.dumps(
            {
                "schema_version": "image-ocr-v1",
                "image_sha256": digest,
                "text": "Table OCR 72.72",
                "status": "text",
                "engine": "rapidocr:test-cache",
            }
        ),
        encoding="utf-8",
    )

    result = extract_image_ocr(image, cache_dir=cache)

    assert result.text == "Table OCR 72.72"
    assert result.image_sha256 == digest
    assert result.engine == "rapidocr:test-cache"


def test_rapidocr_engine_is_shared_across_image_calls(tmp_path: Path, monkeypatch) -> None:
    import app.rag.image_ocr as image_ocr

    class FakeRapidOCR:
        instances = 0

        def __init__(self, **_kwargs):
            type(self).instances += 1

        def __call__(self, _payload):
            return SimpleNamespace(txts=("cached OCR",))

    fake_rapidocr = ModuleType("rapidocr")
    fake_rapidocr.RapidOCR = FakeRapidOCR
    fake_typings = ModuleType("rapidocr.utils.typings")
    fake_typings.EngineType = SimpleNamespace(TORCH="torch")
    monkeypatch.setitem(sys.modules, "rapidocr", fake_rapidocr)
    monkeypatch.setitem(sys.modules, "rapidocr.utils.typings", fake_typings)
    monkeypatch.setattr(image_ocr, "_ENGINE", None)
    monkeypatch.setattr(image_ocr, "_ENGINE_ERROR", None)

    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    first.write_bytes(b"first-image")
    second.write_bytes(b"second-image")

    assert extract_image_ocr(first).text == "cached OCR"
    assert extract_image_ocr(second).text == "cached OCR"
    assert FakeRapidOCR.instances == 1
