import asyncio
import io
import json
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import wave

from fastapi.testclient import TestClient
import numpy as np
import pytest

from aya_tts.characters import CHARACTERS, clone_characters, policy, voice_metadata
from aya_tts.models import parse_voice_id
from aya_tts.server import create_app
from aya_tts.vieneu_engine import VieneuEngine, patch_fetch, preset_voice
from aya_tts.voicevox_engine import VoicevoxEngine, manifest_voices


@pytest.mark.parametrize(("identity", "expected"), [
    ("vi:mai-anh", ("vi", "mai-anh")),
    ("vi-anime:kasukabe-tsumugi", ("vi-anime", "kasukabe-tsumugi")),
    ("ja:8", ("ja", "8")),
])
def test_voice_ids(identity, expected):
    assert parse_voice_id(identity) == expected


@pytest.mark.parametrize("identity", ["", "vi", "vi:", "xx:test", "ja:abc", "ja:-1", "ja:８", "vi:../escape", "vi:Mai Anh", "vi:a:b"])
def test_invalid_voice_ids(identity):
    with pytest.raises(ValueError):
        parse_voice_id(identity)


def test_preset_label():
    voice = preset_voice("⭐ Mai Anh — Nữ · Bắc · Phong cách tin tức")
    assert voice == {"id": "vi:mai-anh", "name": "Mai Anh", "kind": "vi", "language": "vi",
                     "gender": "female", "region": "Bắc", "description": "Phong cách tin tức",
                     "credit": None, "terms_url": None}
    assert preset_voice("Đăng — Nam · Nam · Trầm") == {
        "id": "vi:dang", "name": "Đăng", "kind": "vi", "language": "vi", "gender": "male",
        "region": "Nam", "description": "Trầm", "credit": None, "terms_url": None}
    assert preset_voice("No metadata")["region"] is None


@pytest.mark.parametrize("name", ["雨晴はう", "猫使ビィ", "猫使アル", "もち子さん", "栗田まろん", "琴詠ニア", "Unknown"])
def test_policy_forbids_excluded_clones(name):
    assert not policy(name).clone_allowed
    assert name not in dict(clone_characters())
    assert policy(name).ja_allowed


def test_clone_policy_and_credit():
    assert policy("櫻歌ミコ").clone_allowed
    assert all(item.gender == "female" for _, item in clone_characters())
    assert voice_metadata("もち子さん")["credit"] == "VOICEVOX:もち子(cv 明日葉よもぎ)"
    assert voice_metadata("Unknown")["credit"] == "VOICEVOX:Unknown"


def test_reference_discovery_obeys_policy_even_if_excluded_files_exist(tmp_path, monkeypatch):
    added = []
    refs = tmp_path / "refs"
    refs.mkdir()
    for item in CHARACTERS.values():
        (refs / f"{item.slug}.wav").touch()
    fake = SimpleNamespace(list_preset_voices=lambda: [("Mai Anh — Nữ · Bắc · Tin tức", "internal")],
                           add_voice=lambda name, path, **kwargs: added.append(name))
    monkeypatch.setattr("aya_tts.vieneu_engine.patch_fetch", lambda models: None)
    monkeypatch.setitem(sys.modules, "vieneu", SimpleNamespace(Vieneu=lambda **kwargs: fake))
    engine = VieneuEngine(tmp_path).load()
    assert set(added) == {f"vi-anime:{item.slug}" for _, item in clone_characters()}
    assert engine.voice_names["vi:mai-anh"] == "internal"
    assert all(voice["credit"] for voice in engine.voices if voice["kind"] == "vi-anime")


def test_vieneu_speed_pitch_only_when_requested(tmp_path, monkeypatch):
    operations = []
    engine = VieneuEngine(tmp_path)
    engine.voice_names = {"vi:test": "internal"}
    engine.tts = SimpleNamespace(infer=lambda text, voice: np.zeros(100))
    effects = SimpleNamespace(
        pitch_shift=lambda audio, **kwargs: operations.append(("pitch", kwargs)) or audio,
        time_stretch=lambda audio, **kwargs: operations.append(("speed", kwargs)) or audio,
    )
    monkeypatch.setitem(sys.modules, "librosa", SimpleNamespace(effects=effects))
    _, rate = engine.synthesize("Hi", "vi:test", 1.0, 0.0)
    assert rate == 48000
    assert operations == []
    engine.synthesize("Hi", "vi:test", 1.2, 3)
    assert operations == [("pitch", {"sr": 48000, "n_steps": 3}), ("speed", {"rate": 1.2})]


def test_fetch_uses_real_local_files_and_skips_present(tmp_path, monkeypatch):
    fake_class = type("OnnxV3LiteEngine", (), {})
    downloads = []

    def download(repo, filename, *, subfolder, local_dir):
        downloads.append((repo, filename, subfolder, local_dir))
        target = local_dir / subfolder / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"model")
        return str(target)

    monkeypatch.setitem(sys.modules, "vieneu._v3_turbo_engine.onnx_runtime_lite", SimpleNamespace(OnnxV3LiteEngine=fake_class))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=download))
    patch_fetch(tmp_path)
    directory = fake_class._fetch("org/repo", ["graph.onnx", "graph.onnx.data"], "onnx_update")
    assert directory == tmp_path / "vieneu/org--repo/onnx_update"
    assert len(downloads) == 2
    fake_class._fetch("org/repo", ["graph.onnx", "graph.onnx.data"], "onnx_update")
    assert len(downloads) == 2
    assert not (directory / "graph.onnx.data").is_symlink()


def test_voicevox_manifest_filters_singing_and_preserves_credits():
    voices = manifest_voices({"0.vvm": [
        {"name": "春日部つむぎ", "styles": [{"name": "ノーマル", "id": 8, "type": "talk"}]},
        {"name": "雨晴はう", "styles": [{"name": "ノーマル", "id": 10, "type": "talk"}]},
        {"name": "Unknown", "styles": [{"name": "talk", "id": 20}, {"name": "sing", "id": 21, "type": "sing"}]},
    ]})
    assert [voice["id"] for voice in voices] == ["ja:8", "ja:10", "ja:20"]
    assert voices[0]["credit"] == "VOICEVOX:春日部つむぎ"
    assert voices[1]["kind"] == "ja"


def test_voicevox_lazy_model_and_scales(tmp_path, monkeypatch):
    directory = tmp_path / "voicevox"
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps({"0.vvm": [
        {"name": "春日部つむぎ", "styles": [{"name": "ノーマル", "id": 8, "type": "talk"}]}]}))
    loads, queries = [], []

    class Model:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def synthesize(query, identity):
        queries.append(query)
        return b"wav"

    synth = SimpleNamespace(load_voice_model=lambda model: loads.append(model),
                            create_audio_query=lambda *args: SimpleNamespace(), synthesis=synthesize)
    monkeypatch.setitem(sys.modules, "voicevox_core", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "voicevox_core.blocking", SimpleNamespace(
        Onnxruntime=SimpleNamespace(load_once=lambda **kwargs: None), OpenJtalk=lambda path: None,
        Synthesizer=lambda *args: synth, VoiceModelFile=SimpleNamespace(open=lambda path: Model())))
    monkeypatch.setitem(sys.modules, "soundfile", SimpleNamespace(read=lambda *args, **kwargs: (np.zeros(10), 24000)))
    engine = VoicevoxEngine(tmp_path).load()
    assert loads == []
    engine.synthesize("こんにちは", "ja:8", 1.2, 6)
    engine.synthesize("こんにちは", "ja:8", 1, 0)
    assert len(loads) == 1
    assert queries[0].speed_scale == 1.2
    assert queries[0].pitch_scale == 0.5


class FakeEngine:
    voices = [{"id": "vi:test", "kind": "vi", "name": "Test"}]

    def synthesize(self, text, voice_id, speed, pitch):
        return np.array([[0.25, 0.25], [-0.25, -0.25]]), 24000


def wait_for_state(client, state):
    for _ in range(200):
        health = client.get("/health").json()
        if health["state"] == state:
            return health
        time.sleep(0.001)
    raise AssertionError(f"Health never became {state}: {health}")


def test_health_lifecycle_and_wav():
    gate = threading.Event()

    def loader():
        assert gate.wait(timeout=2)
        return [FakeEngine()]

    with TestClient(create_app(loader)) as client:
        assert client.get("/health").json()["state"] == "starting"
        assert client.get("/voices").status_code == 503
        gate.set()
        health = wait_for_state(client, "ready")
        assert health["load_seconds"] is not None
        assert client.get("/voices").json() == {"voices": FakeEngine.voices}
        assert client.post("/synthesize", json={"text": "Hi", "voice_id": "vi:missing"}).status_code == 404
        response = client.post("/synthesize", json={"text": "Hi", "voice_id": "vi:test"})
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        with wave.open(io.BytesIO(response.content)) as wav:
            assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 24000)
            assert wav.getnframes() == 2
            assert np.frombuffer(wav.readframes(2), dtype="<i2").tolist() == [8192, -8192]


def test_health_failure():
    def loader():
        raise RuntimeError("Model files missing")

    with TestClient(create_app(loader)) as client:
        assert wait_for_state(client, "error")["detail"] == "Model files missing"
        assert client.post("/synthesize", json={"text": "Hi", "voice_id": "vi:test"}).status_code == 503


@pytest.mark.parametrize("params", [{"speed": 0.6}, {"speed": 1.6}, {"pitch": -5}, {"pitch": 7},
                                    {"speed": "NaN"}, {"voice_id": "bad:id"}, {"text": ""}])
def test_synthesis_validation(params):
    with TestClient(create_app(lambda: [FakeEngine()])) as client:
        wait_for_state(client, "ready")
        assert client.post("/synthesize", json={"text": "Hi", "voice_id": "vi:test", **params}).status_code == 422


def test_import_does_not_load_models_or_audio_dependencies():
    code = """
import sys
import aya_tts.server, aya_tts.vieneu_engine, aya_tts.voicevox_engine
assert 'vieneu' not in sys.modules
assert 'voicevox_core' not in sys.modules
assert 'librosa' not in sys.modules
assert 'soundfile' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_synthesis_serializes_each_engine():
    import httpx

    class CountingEngine(FakeEngine):
        def __init__(self):
            self.active = 0
            self.maximum = 0

        def synthesize(self, *args):
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            time.sleep(0.01)
            self.active -= 1
            return super().synthesize(*args)

    engine = CountingEngine()
    app = create_app(lambda: [engine])

    async def run():
        async with app.router.lifespan_context(app):
            for _ in range(100):
                if app.state.health["state"] == "ready":
                    break
                await asyncio.sleep(0.001)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://sidecar") as client:
                responses = await asyncio.gather(*[
                    client.post("/synthesize", json={"text": "Hi", "voice_id": "vi:test"}) for _ in range(3)
                ])
                assert all(response.status_code == 200 for response in responses)

    asyncio.run(run())
    assert engine.maximum == 1


def test_setup_download_is_atomic_and_idempotent(tmp_path, monkeypatch):
    from aya_tts.setup_tts import download

    calls = []

    def retrieve(url, target):
        calls.append(url)
        target.write_bytes(b"model")

    monkeypatch.setattr("urllib.request.urlretrieve", retrieve)
    target = tmp_path / "model.vvm"
    download("https://example.org/model", target)
    download("https://example.org/model", target)
    assert calls == ["https://example.org/model"]
    assert target.read_bytes() == b"model"
    assert not target.with_suffix(".vvm.part").exists()


def test_setup_rejects_archive_traversal(tmp_path):
    import tarfile
    from aya_tts.setup_tts import extract

    archive = tmp_path / "bad.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        item = tarfile.TarInfo("../outside")
        item.size = 3
        tar.addfile(item, io.BytesIO(b"bad"))
    with pytest.raises(tarfile.FilterError):
        extract("https://example.org/bad.tgz", tmp_path, tmp_path / "expected")


def test_setup_manifest_references_and_rerun_are_offline_with_fakes(tmp_path, monkeypatch):
    from aya_tts import setup_tts

    downloads, rendered, unloaded = [], [], []
    normal = SimpleNamespace(id=8, name="ノーマル", type="talk")

    class Model:
        def __init__(self, path):
            self.id = path.name
            self.metas = [SimpleNamespace(name=name, styles=[normal]) for name in (
                ["春日部つむぎ", "雨晴はう"] if path.name == "0.vvm" else ["Unknown"])]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def retrieve(url, target):
        downloads.append(url)
        target.write_bytes(b"model")

    def extract(url, directory, expected):
        expected.parent.mkdir(parents=True, exist_ok=True)
        if expected.suffix:
            expected.touch()
        else:
            expected.mkdir(exist_ok=True)

    def render(text, identity):
        rendered.append(identity)
        return b"wav"

    def write(path, audio, rate, **kwargs):
        assert 5 * rate <= len(audio) <= 8 * rate
        path.write_bytes(b"reference")

    synthesizer = SimpleNamespace(load_voice_model=lambda model: None, tts=render,
                                  unload_voice_model=lambda identity: unloaded.append(identity))
    monkeypatch.setattr(setup_tts, "extract", extract)
    monkeypatch.setattr(setup_tts, "policy", lambda name: policy(name) if name != "Unknown" else SimpleNamespace(ja_allowed=False))
    monkeypatch.setattr("urllib.request.urlretrieve", retrieve)
    monkeypatch.setitem(sys.modules, "soundfile", SimpleNamespace(
        read=lambda *args, **kwargs: (np.zeros(900), 100), write=write))
    monkeypatch.setitem(sys.modules, "voicevox_core", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "voicevox_core.blocking", SimpleNamespace(
        Onnxruntime=SimpleNamespace(load_once=lambda **kwargs: None), OpenJtalk=lambda path: None,
        Synthesizer=lambda *args: synthesizer, VoiceModelFile=SimpleNamespace(open=Model)))
    monkeypatch.setattr(VieneuEngine, "load", lambda self: SimpleNamespace(voices=[{"kind": "vi-anime"}]))
    # Restore environment changes made by online setup when the test ends.
    for key in ["HF_HUB_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "HF_HOME", "HF_HUB_CACHE"]:
        monkeypatch.setenv(key, "unused")
    setup_tts.setup(tmp_path)
    manifest = json.loads((tmp_path / "voicevox/manifest.json").read_text())
    assert set(manifest) == {"0.vvm"}
    assert len(downloads) == 27
    assert rendered == [8]
    assert unloaded == ["0.vvm"]
    assert (tmp_path / "refs/kasukabe-tsumugi.wav").is_file()
    assert not (tmp_path / "refs/amehare-hau.wav").exists()
    assert not (tmp_path / "voicevox/1.vvm").exists()
    setup_tts.setup(tmp_path)
    assert len(downloads) == 27
    assert rendered == [8]
