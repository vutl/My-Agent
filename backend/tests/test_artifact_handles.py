from __future__ import annotations

from app.services.artifacts import ArtifactStore


def test_repeated_call_ids_never_overwrite_earlier_artifacts(tmp_path):
    store = ArtifactStore(tmp_path)
    first = store.save("run", "call_0", "first step output " * 500)
    second = store.save("run", "call_0", "second step output " * 500)
    retry = store.save("run", "call_0", "retried step output")
    assert len({first, second, retry}) == 3
    assert store.read(first, 0, 17) == "first step output"
    assert store.read(second, 0, 18) == "second step output"
    assert store.read(retry) == "retried step output"
