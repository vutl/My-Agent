from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

import fitz
import pytest
import torch

from app.rag.colsmol_retrieval import (
    ColSmolVisualRetrievalProvider,
    VisualRetrievalUnavailable,
    resolve_colsmol_device,
    resolve_colsmol_score_batch_size,
)
from app.rag.parsers import ParsedPage
from app.rag.visual_retrieval import (
    DisabledVisualRetrievalProvider,
    VisualPageRecord,
    VisualSearchResult,
    page_records_from_document_rows,
    reciprocal_rank_fuse_keys,
    render_pdf_page_records,
    visual_search_results_as_evidence,
)
from app.rag.visual_worker_client import (
    ColSmolWorkerClient,
    plan_colsmol_page_batches,
    resolve_colsmol_image_batch_pixels,
)
from app.rag.visual_worker import _handle
from app.db.sqlite import connect, init_db
from app.services.indexing_service import IndexingService
from app.services.rag_service import RagService, requested_visual_page_numbers


def test_disabled_visual_provider_is_explicit_and_fail_closed() -> None:
    provider = DisabledVisualRetrievalProvider()
    result = asyncio.run(provider.index_pages([]))
    assert result["status"] == "disabled"
    assert asyncio.run(provider.search("chart")) == []


def test_visual_worker_uses_sync_provider_boundary_for_abortable_requests() -> None:
    """The worker must not leave an executor thread during parent aborts."""

    class FakeProvider:
        def _index_pages_sync(self, pages):
            return {"ok": True, "count": len(pages)}

        def _search_sync(self, query, *, document_ids, revision_ids, top_k):
            return []

        def _get_pages_sync(self, *, page_numbers, document_ids, revision_ids):
            return []

        def _delete_document_sync(self, document_id, revision_id):
            return None

    provider = FakeProvider()
    result = asyncio.run(
        _handle(
            provider,
            {"op": "index", "pages": []},
        )
    )
    assert result == {"ok": True, "count": 0}


def test_colsmol_auto_device_is_cpu_safe_and_mps_requires_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("COLSMOL_ALLOW_MPS", raising=False)
    assert resolve_colsmol_device("auto", mps_available=True) == "cpu"
    monkeypatch.setenv("COLSMOL_ALLOW_MPS", "true")
    assert resolve_colsmol_device("auto", mps_available=True) == "mps"
    assert resolve_colsmol_device("cpu", mps_available=True) == "cpu"
    with pytest.raises(VisualRetrievalUnavailable):
        resolve_colsmol_device("mps", mps_available=False)


def test_colsmol_score_batch_size_is_conservative_and_configurable() -> None:
    assert resolve_colsmol_score_batch_size(None) == 16
    assert resolve_colsmol_score_batch_size("invalid") == 16
    assert resolve_colsmol_score_batch_size("0") == 1
    assert resolve_colsmol_score_batch_size("32") == 32
    assert resolve_colsmol_score_batch_size("999") == 128


def test_colsmol_image_batch_budget_is_pixel_aware(tmp_path: Path) -> None:
    from PIL import Image

    large = tmp_path / "large.png"
    small_one = tmp_path / "small-one.png"
    small_two = tmp_path / "small-two.png"
    Image.new("RGB", (200, 100)).save(large)  # 20k pixels: must stand alone.
    Image.new("RGB", (50, 50)).save(small_one)
    Image.new("RGB", (50, 50)).save(small_two)
    pages = [
        VisualPageRecord("large", "d", "r", 1, str(large)),
        VisualPageRecord("small-1", "d", "r", 2, str(small_one)),
        VisualPageRecord("small-2", "d", "r", 3, str(small_two)),
    ]
    batches = plan_colsmol_page_batches(pages, max_pages=8, max_pixels=6_000)
    assert [[page.page_id for page in batch] for batch in batches] == [
        ["large"],
        ["small-1", "small-2"],
    ]
    assert resolve_colsmol_image_batch_pixels("invalid") == 4_000_000
    assert resolve_colsmol_image_batch_pixels("999999999") == 32_000_000


def test_page_records_require_existing_image_and_active_revision(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"not-a-real-image-but-an-existing-test-asset")
    rows = [
        {
            "id": "figure-1",
            "document_id": "doc-1",
            "revision_id": "rev-1",
            "page_number": 2,
            "image_path": str(image),
            "caption": "Figure 1",
        },
        {
            "id": "figure-2",
            "document_id": "doc-1",
            "revision_id": "",
            "page_number": 3,
            "image_path": str(image),
        },
    ]
    records = page_records_from_document_rows(rows)
    assert len(records) == 1
    assert records[0].page_id == "figure-1"
    assert records[0].revision_id == "rev-1"


class _FakeMultiVectorEncoder:
    def __init__(self, fingerprint: str = "fake-visual-v1") -> None:
        self.model_fingerprint = fingerprint
        self.image_calls = 0
        self.query_calls = 0
        self.score_calls = 0

    def encode_image(self, image_path: Path) -> torch.Tensor:
        self.image_calls += 1
        value = float(image_path.read_bytes()[0])
        return torch.tensor([[value, 1.0]], dtype=torch.float32)

    def encode_query(self, query: str) -> torch.Tensor:
        self.query_calls += 1
        value = 2.0 if "second" in query else 1.0
        return torch.tensor([[value, 1.0]], dtype=torch.float32)

    def score(
        self,
        query_embedding: torch.Tensor,
        document_embeddings: list[torch.Tensor],
    ) -> list[float]:
        self.score_calls += 1
        return [
            float((query_embedding * document_embedding).sum().item())
            for document_embedding in document_embeddings
        ]


class _FakeBatchEncoder(_FakeMultiVectorEncoder):
    def __init__(self) -> None:
        super().__init__()
        self.batch_calls = 0

    def encode_images(self, image_paths: list[Path]) -> list[torch.Tensor]:
        self.batch_calls += 1
        return [self.encode_image(path) for path in image_paths]


def test_colsmol_provider_is_revision_scoped_resumable_and_searchable(
    tmp_path: Path,
) -> None:
    first_image = tmp_path / "first.png"
    second_image = tmp_path / "second.png"
    first_image.write_bytes(bytes([1]))
    second_image.write_bytes(bytes([2]))
    encoder = _FakeMultiVectorEncoder()
    provider = ColSmolVisualRetrievalProvider(
        index_root=tmp_path / "index",
        encoder=encoder,
    )
    pages = [
        VisualPageRecord("p1", "d1", "r1", 1, str(first_image)),
        VisualPageRecord("p2", "d2", "r2", 1, str(second_image)),
    ]

    first = asyncio.run(provider.index_pages(pages))
    assert first["status"] == "complete"
    assert first["indexed_pages"] == 2
    assert encoder.image_calls == 2

    repeated = asyncio.run(provider.index_pages(pages))
    assert repeated["reused_pages"] == 2
    assert encoder.image_calls == 2
    health = provider.health()
    assert health["index_entries"] == 2
    assert health["index_size_bytes"] > 0
    assert health["model_loaded"] is False

    hits = asyncio.run(provider.search("second page", top_k=2))
    assert [hit.page_id for hit in hits] == ["p2", "p1"]
    assert encoder.score_calls == 1
    scoped = asyncio.run(
        provider.search("second page", document_ids=["d1"], revision_ids=["r1"])
    )
    assert [hit.page_id for hit in scoped] == ["p1"]
    hinted = asyncio.run(
        provider.get_pages(page_numbers=[1], document_ids=["d2"], revision_ids=["r2"])
    )
    assert [hit.page_id for hit in hinted] == ["p2"]
    assert hinted[0].metadata["explicit_page_hint"] is True
    evidence = visual_search_results_as_evidence(scoped, filename_by_document={"d1": "a.pdf"})
    assert evidence[0]["artifact_type"] == "page"
    assert evidence[0]["revision_id"] == "r1"
    assert evidence[0]["filename"] == "a.pdf"

    asyncio.run(provider.delete_document("d1", "r1"))
    assert asyncio.run(provider.search("page", document_ids=["d1"])) == []


def test_colsmol_provider_batches_new_page_embeddings(tmp_path: Path) -> None:
    first_image = tmp_path / "first.png"
    second_image = tmp_path / "second.png"
    first_image.write_bytes(bytes([1]))
    second_image.write_bytes(bytes([2]))
    encoder = _FakeBatchEncoder()
    provider = ColSmolVisualRetrievalProvider(
        index_root=tmp_path / "index",
        encoder=encoder,
    )
    result = asyncio.run(
        provider.index_pages(
            [
                VisualPageRecord("p1", "d", "r", 1, str(first_image)),
                VisualPageRecord("p2", "d", "r", 2, str(second_image)),
            ]
        )
    )
    assert result["indexed_pages"] == 2
    assert result["failed_pages"] == 0
    assert encoder.batch_calls == 1
    assert encoder.image_calls == 2


def test_colsmol_provider_fails_closed_on_model_fingerprint_change(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(bytes([1]))
    root = tmp_path / "index"
    original = ColSmolVisualRetrievalProvider(
        index_root=root,
        encoder=_FakeMultiVectorEncoder("model-a"),
    )
    asyncio.run(
        original.index_pages([VisualPageRecord("p1", "d1", "r1", 1, str(image))])
    )
    changed = ColSmolVisualRetrievalProvider(
        index_root=root,
        encoder=_FakeMultiVectorEncoder("model-b"),
    )
    try:
        asyncio.run(changed.search("page"))
    except VisualRetrievalUnavailable as exc:
        assert "fingerprint mismatch" in str(exc)
    else:  # pragma: no cover - safety assertion
        raise AssertionError("model drift must not read the old visual index")


def test_pdf_page_rendering_is_revision_scoped_and_resumable(tmp_path: Path) -> None:
    source = tmp_path / "two-pages.pdf"
    pdf = fitz.open()
    for text in ("first", "second"):
        page = pdf.new_page()
        page.insert_text((72, 72), text)
    pdf.save(source)
    pdf.close()

    records = render_pdf_page_records(
        source_path=source,
        document_id="doc",
        revision_id="rev",
        output_root=tmp_path / "rendered",
        scale=1.0,
    )
    mtimes = [Path(record.image_path).stat().st_mtime_ns for record in records]
    repeated = render_pdf_page_records(
        source_path=source,
        document_id="doc",
        revision_id="rev",
        output_root=tmp_path / "rendered",
        scale=1.0,
    )
    assert [record.page_id for record in records] == ["page:doc:rev:1", "page:doc:rev:2"]
    assert [Path(record.image_path).stat().st_mtime_ns for record in repeated] == mtimes
    manifest = json.loads((Path(records[0].image_path).parent / "manifest.json").read_text())
    assert manifest["revision_id"] == "rev"
    assert manifest["status"] == "complete"
    assert manifest["completed_pages"] == manifest["page_count"] == 2
    assert len(manifest["pages"]) == 2


def test_pdf_page_rendering_caps_large_source_media_box(tmp_path: Path) -> None:
    source = tmp_path / "large.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=4_000, height=3_000)
    page.insert_text((72, 72), "large source page")
    pdf.save(source)
    pdf.close()

    records = render_pdf_page_records(
        source_path=source,
        document_id="doc",
        revision_id="rev",
        output_root=tmp_path / "rendered",
        scale=1.25,
        max_render_pixels=1_000_000,
    )
    from PIL import Image

    with Image.open(records[0].image_path) as image:
        assert image.width * image.height <= 1_005_000
    assert records[0].metadata["effective_render_scale"] < 1.25
    assert records[0].metadata["render_max_pixels"] == 1_000_000


def test_visual_text_rrf_is_deduplicated_and_stable() -> None:
    assert reciprocal_rank_fuse_keys(
        [["page-1", "page-1", "page-3"], ["page-2", "page-1"]],
        rrf_k=60,
        top_k=3,
    ) == ["page-1", "page-2", "page-3"]


def test_visual_worker_client_uses_ndjson_without_loading_colpali(tmp_path: Path) -> None:
    module = tmp_path / "fake_visual_worker.py"
    module.write_text(
        """import json, sys
for line in sys.stdin:
    request = json.loads(line)
    if request['op'] == 'health':
        result = {'ok': True, 'provider': 'fake'}
    elif request['op'] == 'search':
        result = [{'page_id':'p1','document_id':'d1','revision_id':'r1','page_number':1,'score':1.0,'image_path':'/tmp/p.png','metadata':{}}]
    elif request['op'] == 'shutdown':
        result = {'ok': True, 'shutdown': True}
    else:
        result = {'ok': True}
    print(json.dumps({'id': request['id'], 'ok': True, 'result': result}), flush=True)
    if result.get('shutdown'):
        break
""",
        encoding="utf-8",
    )
    client = ColSmolWorkerClient(
        python_executable=Path(sys.executable),
        index_root=tmp_path / "index",
        backend_root=tmp_path,
        worker_module="fake_visual_worker",
        response_timeout_seconds=5,
    )
    assert asyncio.run(client.health())["provider"] == "fake"
    hits = asyncio.run(client.search("chart", document_ids=["d1"]))
    assert hits[0].revision_id == "r1"
    client.close()


def test_visual_worker_auto_retries_cpu_after_device_abort(tmp_path: Path) -> None:
    state = tmp_path / "attempts"
    module = tmp_path / "device_fallback_worker.py"
    module.write_text(
        f"""import json, os, sys
state = {str(state)!r}
attempts = int(open(state).read()) if os.path.exists(state) else 0
open(state, 'w').write(str(attempts + 1))
if attempts == 0:
    os._exit(134)
for line in sys.stdin:
    request = json.loads(line)
    op = request['op']
    if op == 'health':
        result = {{'ok': True, 'provider': 'fake', 'effective_device': 'cpu'}}
    elif op == 'shutdown':
        result = {{'ok': True, 'shutdown': True}}
    else:
        result = {{'ok': True}}
    print(json.dumps({{'id': request['id'], 'ok': True, 'result': result}}), flush=True)
    if result.get('shutdown'):
        break
""",
        encoding="utf-8",
    )
    client = ColSmolWorkerClient(
        python_executable=Path(sys.executable),
        index_root=tmp_path / "index",
        backend_root=tmp_path,
        worker_module="device_fallback_worker",
        device="auto",
        response_timeout_seconds=5,
    )
    result = asyncio.run(client.health())
    assert result["provider"] == "fake"
    assert result["client_effective_device"] == "cpu"
    assert "exit=134" in str(result["device_fallback_reason"])
    assert state.read_text(encoding="utf-8") == "2"
    client.close()


def test_visual_worker_auto_retries_cpu_after_mps_segfault(tmp_path: Path) -> None:
    state = tmp_path / "attempts"
    module = tmp_path / "segfault_fallback_worker.py"
    module.write_text(
        f"""import json, os, signal, sys
state = {str(state)!r}
attempts = int(open(state).read()) if os.path.exists(state) else 0
open(state, 'w').write(str(attempts + 1))
if attempts == 0:
    os.kill(os.getpid(), signal.SIGSEGV)
for line in sys.stdin:
    request = json.loads(line)
    op = request['op']
    if op == 'health':
        result = {{'ok': True, 'provider': 'fake', 'effective_device': 'cpu'}}
    elif op == 'shutdown':
        result = {{'ok': True, 'shutdown': True}}
    else:
        result = {{'ok': True}}
    print(json.dumps({{'id': request['id'], 'ok': True, 'result': result}}), flush=True)
    if result.get('shutdown'):
        break
""",
        encoding="utf-8",
    )
    client = ColSmolWorkerClient(
        python_executable=Path(sys.executable),
        index_root=tmp_path / "index",
        backend_root=tmp_path,
        worker_module="segfault_fallback_worker",
        device="auto",
        response_timeout_seconds=5,
    )
    result = asyncio.run(client.health())
    assert result["provider"] == "fake"
    assert result["client_effective_device"] == "cpu"
    assert "exit=-11" in str(result["device_fallback_reason"])
    assert state.read_text(encoding="utf-8") == "2"
    client.close()


def test_visual_worker_client_batches_large_index_requests(tmp_path: Path) -> None:
    module = tmp_path / "batch_visual_worker.py"
    module.write_text(
        """import json, sys
for line in sys.stdin:
    request = json.loads(line)
    pages = request.get('pages') or []
    result = {'ok': True, 'status': 'complete', 'indexed_pages': len(pages),
              'reused_pages': 0, 'failed_pages': 0, 'errors': []}
    print(json.dumps({'id': request['id'], 'ok': True, 'result': result}), flush=True)
""",
        encoding="utf-8",
    )
    client = ColSmolWorkerClient(
        python_executable=Path(sys.executable),
        index_root=tmp_path / "index",
        backend_root=tmp_path,
        worker_module="batch_visual_worker",
        response_timeout_seconds=5,
        max_pages_per_request=2,
    )
    pages = [
        VisualPageRecord(f"p{i}", "d", "r", i, str(tmp_path / f"p{i}.png"))
        for i in range(1, 6)
    ]
    result = asyncio.run(client.index_pages(pages))
    assert result["batches"] == 3
    assert result["requested_pages"] == 5
    assert result["indexed_pages"] == 5
    assert result["failed_pages"] == 0
    client.close()


def test_visual_worker_client_reports_batch_progress_without_affecting_result(tmp_path: Path) -> None:
    module = tmp_path / "progress_visual_worker.py"
    module.write_text(
        """import json, sys
for line in sys.stdin:
    request = json.loads(line)
    pages = request.get('pages') or []
    result = {'ok': True, 'status': 'complete', 'indexed_pages': len(pages),
              'reused_pages': 0, 'failed_pages': 0, 'errors': []}
    print(json.dumps({'id': request['id'], 'ok': True, 'result': result}), flush=True)
""",
        encoding="utf-8",
    )
    client = ColSmolWorkerClient(
        python_executable=Path(sys.executable),
        index_root=tmp_path / "index",
        backend_root=tmp_path,
        worker_module="progress_visual_worker",
        response_timeout_seconds=5,
        max_pages_per_request=2,
    )
    pages = [
        VisualPageRecord(f"p{i}", "d", "r", i, str(tmp_path / f"p{i}.png"))
        for i in range(1, 6)
    ]
    progress: list[dict] = []
    result = asyncio.run(client.index_pages(pages, progress_callback=progress.append))
    assert result["indexed_pages"] == 5
    assert [item["processed_pages"] for item in progress] == [2, 4, 5]
    assert progress[-1]["remaining_pages"] == 0
    client.close()


def test_visual_worker_close_does_not_wait_for_cancelled_request(tmp_path: Path) -> None:
    module = tmp_path / "slow_visual_worker.py"
    module.write_text(
        """import json, sys, time
for line in sys.stdin:
    request = json.loads(line)
    if request['op'] == 'index':
        time.sleep(60)
    result = {'ok': True, 'shutdown': request['op'] == 'shutdown'}
    print(json.dumps({'id': request['id'], 'ok': True, 'result': result}), flush=True)
""",
        encoding="utf-8",
    )
    client = ColSmolWorkerClient(
        python_executable=Path(sys.executable),
        index_root=tmp_path / "index",
        backend_root=tmp_path,
        worker_module="slow_visual_worker",
        response_timeout_seconds=3_600,
    )
    pages = [VisualPageRecord("p1", "d", "r", 1, str(tmp_path / "p1.png"))]

    async def run() -> None:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(client.index_pages(pages), timeout=0.02)
        await asyncio.wait_for(client.aclose(), timeout=1.0)

    asyncio.run(run())


def test_rag_visual_search_rejects_stale_revision_hits(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    source = tmp_path / "paper.txt"
    source.write_text("active visual source", encoding="utf-8")
    init_db(db_path)
    document = IndexingService(db_path).index_file(source_path=str(source))
    assert document.revision_id

    class _Provider:
        async def search(self, _query, *, document_ids=None, revision_ids=None, top_k=10):
            assert document_ids == [document.id]
            assert revision_ids == [document.revision_id]
            assert top_k == 4
            return [
                VisualSearchResult(
                    page_id="live-page",
                    document_id=document.id,
                    revision_id=document.revision_id,
                    page_number=1,
                    score=0.9,
                    image_path=str(source),
                    metadata={},
                ),
                VisualSearchResult(
                    page_id="stale-page",
                    document_id=document.id,
                    revision_id="retired-revision",
                    page_number=1,
                    score=1.0,
                    image_path=str(source),
                    metadata={},
                ),
            ]

    results = asyncio.run(
        RagService(db_path).search_visual_pages(
            "chart",
            provider=_Provider(),
            top_k=4,
            document_ids=[document.id],
        )
    )
    assert [item["id"] for item in results] == ["live-page"]
    assert results[0]["revision_id"] == document.revision_id
    assert results[0]["filename"] == "paper.txt"


def test_rag_unscoped_visual_search_filters_active_revisions_before_provider_top_k(
    tmp_path: Path,
) -> None:
    """The provider must never rank retired visual pages into the frontier."""

    db_path = tmp_path / "app.db"
    first_source = tmp_path / "first.txt"
    second_source = tmp_path / "second.txt"
    first_source.write_text("first", encoding="utf-8")
    second_source.write_text("second", encoding="utf-8")
    init_db(db_path)
    service = IndexingService(db_path)
    first = service.index_file(source_path=str(first_source))
    second = service.index_file(source_path=str(second_source))

    observed: dict[str, object] = {}

    class _Provider:
        async def search(self, _query, *, document_ids=None, revision_ids=None, top_k=10):
            observed["document_ids"] = document_ids
            observed["revision_ids"] = revision_ids
            # If the provider receives the allow-list before ranking, this
            # stale hit cannot be returned by a conforming implementation.
            return [
                VisualSearchResult(
                    "active", first.id, first.revision_id, 1, 0.9, str(first_source), {}
                )
            ]

    results = asyncio.run(
        RagService(db_path).search_visual_pages(
            "chart", provider=_Provider(), top_k=4
        )
    )

    assert set(observed["document_ids"] or []) == {first.id, second.id}
    assert set(observed["revision_ids"] or []) == {
        first.revision_id,
        second.revision_id,
    }
    assert [item["id"] for item in results] == ["active"]


def test_rag_visual_search_promotes_explicit_slide_hint_without_embedding_call(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    source = tmp_path / "slides.txt"
    source.write_bytes(b"source")
    init_db(db_path)
    document = IndexingService(db_path).index_file(source_path=str(source))

    class _Provider:
        async def search(self, _query, *, document_ids=None, revision_ids=None, top_k=10):
            raise AssertionError("exact slide lookup must bypass semantic visual search")

        async def get_pages(self, *, page_numbers, document_ids=None, revision_ids=None):
            assert page_numbers == [6]
            return [
                VisualSearchResult(
                    "explicit", document.id, document.revision_id, 6, 1.0, str(source),
                    {"explicit_page_hint": True},
                )
            ]

    results = asyncio.run(
        RagService(db_path).search_visual_pages(
            "what is shown on slide 6?",
            provider=_Provider(),
            top_k=4,
            document_ids=[document.id],
        )
    )
    # An explicit slide is a hard scope constraint; adjacent semantic pages
    # must not leak into the visual evidence set.
    assert [item["id"] for item in results] == ["explicit"]
    assert results[0]["page_number"] == 6


def test_rag_visual_search_resolves_printed_page_label_per_active_document(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "printed-pages.db"
    source = tmp_path / "paper.txt"
    source.write_text("printed page 13", encoding="utf-8")
    init_db(db_path)
    document = IndexingService(db_path).index_file(source_path=str(source))
    with connect(db_path) as connection:
        rows = connection.execute(
            "SELECT id, metadata_json FROM chunks WHERE document_id = ? AND revision_id = ?",
            (document.id, document.revision_id),
        ).fetchall()
        assert rows
        for row in rows:
            metadata = json.loads(row["metadata_json"] or "{}")
            metadata["page_label"] = "13"
            connection.execute(
                "UPDATE chunks SET page_number = ?, metadata_json = ? WHERE id = ?",
                (14, json.dumps(metadata), row["id"]),
            )

    observed: dict[str, object] = {}

    class _Provider:
        async def search(self, _query, *, document_ids=None, revision_ids=None, top_k=10):
            return [
                VisualSearchResult(
                    "semantic", document.id, document.revision_id, 13, 0.99, str(source), {}
                )
            ]

        async def get_pages(self, *, page_numbers, document_ids=None, revision_ids=None):
            observed["page_numbers"] = page_numbers
            return [
                VisualSearchResult(
                    "printed-page-13",
                    document.id,
                    document.revision_id,
                    14,
                    1.0,
                    str(source),
                    {"explicit_page_hint": True},
                )
            ]

    results = asyncio.run(
        RagService(db_path).search_visual_pages(
            "what is shown on page 13?",
            provider=_Provider(),
            top_k=4,
            document_ids=[document.id],
        )
    )
    assert observed["page_numbers"] == [14]
    assert [item["id"] for item in results] == ["printed-page-13"]
    assert results[0]["page_number"] == 14


def test_empty_page_label_row_resolves_without_text_chunk(tmp_path: Path) -> None:
    db_path = tmp_path / "empty-page-label.db"
    source = tmp_path / "paper.txt"
    source.write_text("body", encoding="utf-8")
    init_db(db_path)
    document = IndexingService(db_path).index_file(source_path=str(source))
    indexing = IndexingService(db_path)
    with connect(db_path) as connection:
        indexing._insert_page_label_rows(
            connection,
            pages=[ParsedPage(page_number=14, text="", page_label="13")],
            existing_chunks=[],
            document_id=document.id,
            file_id=document.file_id,
            source_path=str(source),
            filename=source.name,
            revision_id=document.revision_id,
            now="2026-09-24T00:00:00+00:00",
        )
    candidates = RagService(db_path).explicit_page_hint_candidates(
        "what is shown on page 13?",
        document_ids=[document.id],
    )
    assert candidates
    assert candidates[0]["page_number"] == 14
    assert candidates[0]["metadata"]["page_label"] == "13"


def test_rag_visual_search_does_not_ocr_at_query_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "stored-printed-pages.db"
    source = tmp_path / "scan.pdf"
    document_pdf = fitz.open()
    for _ in range(14):
        document_pdf.new_page()
    document_pdf[0].insert_text((72, 72), "cover")
    document_pdf.save(source)
    document_pdf.close()
    init_db(db_path)
    indexed = IndexingService(db_path, enable_pdf_ocr=False).index_file(
        source_path=str(source)
    )
    with connect(db_path) as connection:
        connection.execute(
            "UPDATE documents SET page_count = ? WHERE id = ?",
            (14, indexed.id),
        )

    observed: dict[str, object] = {}

    def fail_if_query_time_ocr_runs(*_args, **_kwargs):
        raise AssertionError("page-label OCR must run during ingestion, not retrieval")

    monkeypatch.setattr(
        "app.rag.parsers.ocr_pdf_page_labels",
        fail_if_query_time_ocr_runs,
    )

    class _Provider:
        async def search(self, _query, *, document_ids=None, revision_ids=None, top_k=10):
            return []

        async def get_pages(self, *, page_numbers, document_ids=None, revision_ids=None):
            observed["page_numbers"] = page_numbers
            return []

    asyncio.run(
        RagService(db_path).search_visual_pages(
            "what is shown on page 13?",
            provider=_Provider(),
            top_k=4,
            document_ids=[indexed.id],
        )
    )

    # This intentionally stale/image-only revision has no persisted label, so
    # retrieval falls back to physical-page semantics without initializing OCR.
    assert observed["page_numbers"] == [13]


def test_rag_visual_search_resolves_facing_page_label_range(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "spread-pages.db"
    source = tmp_path / "spread.txt"
    source.write_text("spread page", encoding="utf-8")
    init_db(db_path)
    document = IndexingService(db_path).index_file(source_path=str(source))
    with connect(db_path) as connection:
        rows = connection.execute(
            "SELECT id, metadata_json FROM chunks WHERE document_id = ? AND revision_id = ?",
            (document.id, document.revision_id),
        ).fetchall()
        assert rows
        for row in rows:
            metadata = json.loads(row["metadata_json"] or "{}")
            metadata["page_label"] = "14-15"
            connection.execute(
                "UPDATE chunks SET page_number = ?, metadata_json = ? WHERE id = ?",
                (9, json.dumps(metadata), row["id"]),
            )

    observed: dict[str, object] = {}

    class _Provider:
        async def search(self, _query, *, document_ids=None, revision_ids=None, top_k=10):
            return []

        async def get_pages(self, *, page_numbers, document_ids=None, revision_ids=None):
            observed["page_numbers"] = page_numbers
            return []

    results = asyncio.run(
        RagService(db_path).search_visual_pages(
            "what is shown on page 14?",
            provider=_Provider(),
            top_k=4,
            document_ids=[document.id],
        )
    )
    assert results == []
    assert observed["page_numbers"] == [9]


def test_visual_page_label_resolution_ignores_retired_revision_metadata(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "retired-printed-pages.db"
    source = tmp_path / "paper.txt"
    source.write_text("old revision", encoding="utf-8")
    init_db(db_path)
    service = IndexingService(db_path)
    old_document = service.index_file(source_path=str(source))
    with connect(db_path) as connection:
        row = connection.execute(
            "SELECT id, metadata_json FROM chunks WHERE document_id = ? AND revision_id = ?",
            (old_document.id, old_document.revision_id),
        ).fetchone()
        assert row
        metadata = json.loads(row["metadata_json"] or "{}")
        metadata["page_label"] = "13"
        connection.execute(
            "UPDATE chunks SET page_number = ?, metadata_json = ? WHERE id = ?",
            (14, json.dumps(metadata), row["id"]),
        )
    source.write_text("new active revision", encoding="utf-8")
    active_document = service.index_file(source_path=str(source))
    assert active_document.id == old_document.id
    assert active_document.revision_id != old_document.revision_id

    observed: dict[str, object] = {}

    class _Provider:
        async def search(self, _query, *, document_ids=None, revision_ids=None, top_k=10):
            return []

        async def get_pages(self, *, page_numbers, document_ids=None, revision_ids=None):
            observed["page_numbers"] = page_numbers
            observed["revision_ids"] = revision_ids
            return []

    results = asyncio.run(
        RagService(db_path).search_visual_pages(
            "what is shown on page 13?",
            provider=_Provider(),
            top_k=4,
            document_ids=[active_document.id],
        )
    )
    assert results == []
    assert observed["page_numbers"] == [13]
    assert observed["revision_ids"] == [active_document.revision_id]


def test_requested_visual_page_numbers_accept_plural_page_and_slide_forms() -> None:
    assert requested_visual_page_numbers("Which stages are shown on slides 12 and 14?") == [12, 14]
    assert requested_visual_page_numbers("Compare pages 3 and page 7.") == [3, 7]
    assert requested_visual_page_numbers("Xem trên trang 5 và trangs 6.") == [5, 6]
    assert requested_visual_page_numbers("Compare pages xiii and xiv.") == [13, 14]
