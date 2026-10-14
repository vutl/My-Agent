from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from app.api.agent import _approved_answer_images, _visual_answer_guard_required
from app.core.config import Settings
from app.db.sqlite import connect, init_db
from app.rag.answer_images import approved_answer_image_sources


@pytest.fixture
def fixture(tmp_path):
    settings = Settings(data_dir=tmp_path, visual_retrieval_enabled=False)
    init_db(settings.sqlite_db_path)
    image = tmp_path / "artifacts" / "d" / "active" / "figure.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), "red").save(image)
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    with connect(settings.sqlite_db_path) as connection:
        connection.execute("INSERT INTO indexed_folders VALUES ('folder','/docs',0,'[]','now','now')")
        for document_id in ("d", "other"):
            connection.execute("""INSERT INTO documents
                (id,folder_id,source_path,filename,file_type,content_hash,modified_at,indexed_at,chunk_count,active_revision_id,metadata_json)
                VALUES (?,'folder',?,?,'pdf','hash','now','now',0,'active',?)""",
                (document_id, '/docs/'+document_id, document_id+'.pdf', '{"enrichment_approved":true}'))
        for revision in ("active", "retired", "staged"):
            connection.execute("""INSERT INTO document_figures
                (id,document_id,figure_index,page_number,image_path,revision_id,metadata_json)
                VALUES (?,'d',0,2,?,?,?)""", (revision, str(image), revision, json.dumps({"image_sha256":digest})))
        connection.execute("""INSERT INTO document_tables
            (id,document_id,table_index,page_number,revision_id,metadata_json)
            VALUES ('table','d',0,2,'active',?)""", (json.dumps({"image_path":str(image),"image_sha256":digest}),))
    source = {"document_id":"d", "revision_id":"active", "figure_id":"active",
              "artifact_type":"figure", "page_number":2, "image_path":str(image), "filename":"d.pdf"}
    return settings, image, source


def materialize(settings, sources):
    return _approved_answer_images(sources, settings=settings, max_images=4, query="explain the diagram")


def test_approved_canonical_pixels_do_not_require_visual_retriever(fixture):
    settings, image, source = fixture
    result = materialize(settings, [source])
    assert len(result["urls"]) == 1
    assert result["labels"] == ["attachment 1: page 2; file d.pdf"]
    assert result["sources"][0]["answer_image_sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()
    assert settings.visual_retrieval_enabled is False
    assert "answer_image_sha256" not in source


@pytest.mark.parametrize("policy", [{}, {"enrichment_approved":"true"},
    {"enrichment_approved":True,"answer_images_approved":False}, {"answer_images_approved":1}])
def test_answer_pixels_require_explicit_stored_permission(fixture, policy):
    settings, image, source = fixture
    source["metadata"] = {"enrichment_approved":True,"answer_images_approved":True}
    with connect(settings.sqlite_db_path) as connection:
        connection.execute("UPDATE documents SET metadata_json=?", (json.dumps(policy),))
    assert materialize(settings.model_copy(update={"visual_retrieval_enabled":True}), [source])["urls"] == []


@pytest.mark.parametrize("change", [{"revision_id":"retired"}, {"figure_id":"retired"},
    {"figure_id":"staged"}, {"document_id":"other"}, {"figure_id":"unknown"}])
def test_wrong_owner_or_revision_cannot_send_pixels(fixture, change):
    settings, _, source = fixture
    assert materialize(settings, [{**source, **change}])["urls"] == []


def test_path_override_and_changed_bytes_are_rejected(fixture):
    settings, image, source = fixture
    secret = image.parent / "not-canonical.png"
    Image.new("RGB", (8,8), "blue").save(secret)
    assert materialize(settings, [{**source, "image_path":str(secret)}])["urls"] == []
    image.write_bytes(secret.read_bytes())
    assert materialize(settings, [source])["urls"] == []


def test_approved_image_table_uses_canonical_metadata_and_exact_identity(fixture):
    settings, image, source = fixture
    table = {**source,"artifact_type":"table","table_id":"table"}
    table.pop("figure_id")
    table.pop("image_path")
    table["metadata"] = {"table_image_path":str(image)}
    assert len(materialize(settings, [table])["urls"]) == 1


def test_corrupt_canonical_image_without_digest_is_not_uploaded(fixture):
    settings, image, source = fixture
    with connect(settings.sqlite_db_path) as connection:
        connection.execute("UPDATE document_figures SET metadata_json='{}'")
    image.write_bytes(b"sensitive non-image text")
    assert materialize(settings, [source])["urls"] == []


def test_authorization_filter_keeps_image_labels_in_uploaded_order(fixture):
    settings, _, source = fixture
    blocked = {**source, "document_id":"other", "filename":"wrong.pdf"}
    result = materialize(settings, [blocked, source])
    assert len(result["urls"]) == 1
    assert result["labels"] == ["attachment 1: page 2; file d.pdf"]


def test_page_pixels_need_visual_flag_and_committed_active_manifest(fixture):
    settings, image, source = fixture
    page = settings.visual_retrieval_path / "rendered" / "fingerprint" / "page.png"
    page.parent.mkdir(parents=True)
    page.write_bytes(image.read_bytes())
    manifest = page.parent / "manifest.json"
    manifest.write_text(json.dumps({"document_id":"d","revision_id":"active","pages":[
        {"page_number":2,"image_path":str(page),"image_sha256":hashlib.sha256(page.read_bytes()).hexdigest()}
    ]}))
    visual = {**source,"artifact_type":"page","visual_page":True,"image_path":str(page)}
    visual.pop("figure_id")
    assert materialize(settings, [visual])["urls"] == []
    enabled = settings.model_copy(update={"visual_retrieval_enabled":True})
    assert len(materialize(enabled, [visual])["urls"]) == 1
    manifest.write_text(json.dumps({"document_id":"d","revision_id":"retired","pages":[]}))
    assert materialize(enabled, [visual])["urls"] == []


def test_missing_database_does_not_create_one(tmp_path):
    path = tmp_path / "missing.db"
    assert approved_answer_image_sources([], db_path=path, visual_root=tmp_path, visual_pages_enabled=True) == []
    assert not path.exists()


def test_activation_race_during_pixel_read_is_rejected(fixture, monkeypatch):
    settings, image, source = fixture
    original = Path.read_bytes

    def racing_read(path):
        payload = original(path)
        if path == image:
            with connect(settings.sqlite_db_path) as connection:
                connection.execute("UPDATE documents SET active_revision_id='new-active' WHERE id='d'")
        return payload

    monkeypatch.setattr(Path, "read_bytes", racing_read)
    assert materialize(settings, [source])["urls"] == []


def test_active_artifact_readers_keep_revision_identity(fixture):
    from app.services.rag_service import RagService
    settings, _, _ = fixture
    rag = RagService(settings.sqlite_db_path)
    assert rag.get_document("d")["active_revision_id"] == "active"
    assert rag.get_figure("active")["revision_id"] == "active"
    assert {item["revision_id"] for item in rag.list_document_figures("d")} == {"active"}
    assert {item["revision_id"] for item in rag.list_document_tables("d")} == {"active"}


def test_visual_appearance_needs_supplied_pixels_not_a_persisted_path(fixture):
    settings, _, source = fixture
    question = "Which color is the arrow in Figure 1?"
    assert _visual_answer_guard_required(question, [source], [])
    assert not _visual_answer_guard_required(question, [source], materialize(settings, [source])["urls"])
    assert not _visual_answer_guard_required("What is the caption of Figure 1?", [source], [])


def test_what_a_figure_shows_can_come_from_its_caption(fixture):
    _settings, _, source = fixture
    assert not _visual_answer_guard_required("What does Figure 1 show?", [source], [])
    assert _visual_answer_guard_required("Which color is the arrow in Figure 1?", [source], [])
