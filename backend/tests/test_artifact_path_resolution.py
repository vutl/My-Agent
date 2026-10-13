from pathlib import Path

from app.services.rag_service import RagService


def test_resolve_artifact_path_accepts_current_root(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    image = root / "doc-1" / "figures" / "figure.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"png")

    assert RagService(tmp_path / "app.db", artifact_root=root).resolve_artifact_path(image) == image


def test_resolve_artifact_path_remaps_legacy_absolute_root(tmp_path: Path) -> None:
    root = tmp_path / "isolated" / "artifacts"
    image = root / "doc-1" / "figures" / "figure.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"png")
    legacy = Path("/old/workspace/data/artifacts/doc-1/figures/figure.png")

    assert RagService(tmp_path / "app.db", artifact_root=root).resolve_artifact_path(legacy) == image


def test_resolve_artifact_path_rejects_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    outside = tmp_path / "secret.png"
    outside.write_bytes(b"not an artifact")

    assert RagService(tmp_path / "app.db", artifact_root=root).resolve_artifact_path(outside) is None
