import asyncio
from types import SimpleNamespace
from urllib.parse import unquote

from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx
import pytest

from app.api import speech as api
from app.core.config import Settings, get_settings
from app.services import speech


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.parametrize(("source", "expected"), [
    ("Xin **chào**, _cậu_ nhé! [S1]", "Xin chào, cậu nhé!"),
    ("# Tiêu đề\n- Một\n2. Hai\n> Ba", "Tiêu đề Một Hai Ba"),
    ("Trước\n```python\nprint('no')\n```\nSau", "Trước Sau"),
    ("Trước\n~~~js\nno\n~~~\nSau", "Trước Sau"),
    ("Trước\n```python\nunfinished", "Trước"),
    ("Trước\n```python", "Trước"),
    ("~~~javascript", ""),
    ("```inline code``` và lời nói\nSau", "và lời nói Sau"),
    ("Trước\n````python\n```\nstill code\n`````\nSau", "Trước Sau"),
    ("Trước\n```python\n```not a closing fence\nstill code\n```\nSau", "Trước Sau"),
    ("Trước\n~~~python\n```\nstill code\n~~~\nSau", "Trước Sau"),
    ("Trước\n- ```python\n  secret()\n  ```\nSau", "Trước Sau"),
    ("Trước\n12. ~~~python\n    secret()\n    ~~~\nSau", "Trước Sau"),
    ("Trước\n- Item\n  - ```python\n    secret()\n    ```\nSau", "Trước Item Sau"),
    ("Trước\n> - ```python\n>   secret()\n>   ```\nSau", "Trước Sau"),
    ("Trước\n> > ~~~python\n> > secret()\n> > ~~~\nSau", "Trước Sau"),
    ("Trước\n```python\n> ```\nsecret()\n```\nSau", "Trước Sau"),
    ("Trước\n- ```python\n  unfinished", "Trước"),
    ("`inline code` và ``a ` b``", "và"),
    ("Trước ``a ``` b`` Sau", "Trước Sau"),
    ("Công thức $x+y$ và $$E=mc^2$$ thôi", "Công thức và thôi"),
    (r"Công thức \(x+y\) và \[x^2\] thôi", "Công thức và thôi"),
    ("A [S1] B [W2, W3] C [S1, S2]", "A B C"),
    ("[S1][S2][W3]", ""),
    ("Xem [tài liệu](https://example.org/a_(b)) nhé", "Xem tài liệu nhé"),
    ("Xem [tài liệu](https://example.org/a_(b_(c))) nhé", "Xem tài liệu nhé"),
    ("Xem [tài [liệu]](https://example.org) nhé", "Xem tài [liệu] nhé"),
    (r"Xem [tài liệu](https://example.org/a\)b) nhé", "Xem tài liệu nhé"),
    ("![chú thích](https://example.org/a_(b_(c)).png)", "chú thích"),
    ("[tài liệu][ref]\n[ref]: https://example.org", "tài liệu"),
    ("[tài [liệu]][ref]\n[ref]: https://example.org", "tài [liệu]"),
    ("Vẫn giữ [chú thích] trong câu.", "Vẫn giữ [chú thích] trong câu."),
    ("https://example.org\nwww.example.org\n<https://example.org>", ""),
    ("ftp://example.org/file file:///tmp/report.pdf mailto:aya@example.org", ""),
    ("```python\nprint(1)\n```", ""),
    ("$x$ [S1] 123 — !!!", ""),
    ("###\n---\n***\n123", ""),
    ("Trước\n| A | B |\n| --- | :---: |\n| 1 | 2 |\n\nSau", "Trước Sau"),
    ("A | B\n--- | ---\nfoo | bar", ""),
    ("- [x] Đã xong\n- [ ] Chưa xong", "Đã xong Chưa xong"),
    ("\n  Tiếng Việt: Nguyễn, Đặng, Trần.  \n", "Tiếng Việt: Nguyễn, Đặng, Trần."),
    (r"Một \begin{align*}a &= b\end{align*} công thức", "Một công thức"),
    ("| Name | Value |\n| data | 123 |", ""),
    ("こんにちは！", "こんにちは！"),
    ("", ""),
])
def test_speech_text(source, expected):
    assert speech.speech_text(source) == expected


VOICES = [
    {"id": "vi:mai-anh", "kind": "vi", "gender": "female"},
    {"id": "vi-anime:kasukabe-tsumugi", "kind": "vi-anime", "gender": "female"},
    {"id": "ja:8", "kind": "ja", "gender": "female"},
]


@pytest.fixture
def speech_api():
    calls = []

    def handler(request):
        import json

        calls.append((request.url.path, json.loads(request.content) if request.content else None))
        if request.url.path == "/voices":
            return httpx.Response(200, json={"voices": VOICES})
        return httpx.Response(200, content=b"RIFFfake-wav", headers={"Content-Type": "audio/wav"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    supervisor = SimpleNamespace(state="ready", client=client, base_url="http://sidecar")
    supervisor.status = lambda: {"state": supervisor.state, "detail": None}
    app = FastAPI()
    app.include_router(api.router)
    app.dependency_overrides[api.get_supervisor] = lambda: supervisor
    with TestClient(app) as test_client:
        yield test_client, supervisor, calls
    asyncio.run(client.aclose())


def test_status_and_voices(speech_api, monkeypatch):
    client, supervisor, calls = speech_api
    monkeypatch.setattr(api, "get_settings", lambda: Settings())
    assert client.get("/speech/status").json() == {"state": "ready", "detail": None}
    assert client.get("/speech/voices").json()["default_voice_id"] == "vi-anime:kasukabe-tsumugi"
    monkeypatch.setattr(api, "get_settings", lambda: Settings(tts_default_voice="missing"))
    assert client.get("/speech/voices").json()["default_voice_id"] == "vi:mai-anh"
    supervisor.state = "starting"
    assert client.get("/speech/voices").status_code == 503
    assert client.post("/speech/synthesize", json={"text": "hi", "voice_id": "ja:8"}).status_code == 503


def test_clean_wav_and_spoken_header(speech_api):
    client, _, calls = speech_api
    response = client.post("/speech/synthesize", json={"text": "**Xin chào** [S1]", "voice_id": "vi:mai-anh"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert unquote(response.headers["X-Aya-Spoken-Text"]) == "Xin chào"
    assert response.content == b"RIFFfake-wav"
    assert calls[-1] == ("/synthesize", {"text": "Xin chào", "voice_id": "vi:mai-anh", "speed": 1.0, "pitch": 0.0})


def test_ja_translation_and_failure(speech_api, monkeypatch):
    client, _, calls = speech_api
    translated = []

    async def translate(text):
        translated.append(text)
        return "こんにちはだよ"

    monkeypatch.setattr(api, "translate_ja", translate)
    response = client.post("/speech/synthesize", json={"text": "**Hello** [W2]", "voice_id": "ja:8"})
    assert response.status_code == 200
    assert translated == ["Hello"]
    assert calls[-1][1]["text"] == "こんにちはだよ"
    assert unquote(response.headers["X-Aya-Spoken-Text"]) == "こんにちはだよ"

    async def fail(text):
        raise TimeoutError("LLM timeout")

    monkeypatch.setattr(api, "translate_ja", fail)
    assert client.post("/speech/synthesize", json={"text": "Hello", "voice_id": "ja:8"}).status_code == 502


def test_empty_and_unknown(speech_api):
    client, _, calls = speech_api
    response = client.post("/speech/synthesize", json={"text": "https://example.org [S1]", "voice_id": "vi:mai-anh"})
    assert response.status_code == 204
    assert response.content == b""
    assert all(path != "/synthesize" for path, _ in calls)
    assert client.post("/speech/synthesize", json={"text": "Hi", "voice_id": "vi:missing"}).status_code == 404


def test_list_code_is_silent_without_translation_or_synthesis(speech_api, monkeypatch):
    client, _, calls = speech_api

    async def forbidden(text):
        raise AssertionError("Code must never be sent to the translator")

    monkeypatch.setattr(api, "translate_ja", forbidden)
    response = client.post("/speech/synthesize", json={
        "text": "- ```python\n  print('not speech')\n  ```", "voice_id": "ja:8",
    })
    assert response.status_code == 204
    assert response.content == b""
    assert all(path != "/synthesize" for path, _ in calls)


@pytest.mark.parametrize("params", [{"speed": 0.69}, {"speed": 1.51}, {"pitch": -4.1}, {"pitch": 6.1},
                                    {"speed": "NaN"}, {"pitch": "Infinity"}, {"voice_id": ""}, {"text": None}])
def test_bad_params(speech_api, params):
    client, _, _ = speech_api
    response = client.post("/speech/synthesize", json={"text": "Hi", "voice_id": "vi:mai-anh", **params})
    assert response.status_code == 422


@pytest.mark.parametrize("status", [404, 422, 503, 500])
def test_sidecar_errors(speech_api, status):
    client, supervisor, _ = speech_api

    async def request(*args, **kwargs):
        return httpx.Response(status)

    supervisor.client.request = request
    assert client.get("/speech/voices").status_code == (502 if status == 500 else status)


def test_sidecar_transport_failure(speech_api):
    client, supervisor, _ = speech_api

    async def fail(*args, **kwargs):
        raise httpx.ConnectError("sidecar disappeared")

    supervisor.client.request = fail
    assert client.get("/speech/voices").status_code == 503


def test_no_supervisor_is_off():
    app = FastAPI()
    app.include_router(api.router)
    with TestClient(app) as client:
        assert client.get("/speech/status").json() == {"state": "off", "detail": None}
        assert client.get("/speech/voices").status_code == 503


@pytest.mark.anyio
async def test_translation_uses_router_and_lru(monkeypatch):
    calls = []
    clients = []
    settings = Settings(router_model="cx/gpt-6-luna")

    async def chat(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(message="こんにちはだよ")

    def factory(**kwargs):
        clients.append(kwargs)
        return SimpleNamespace(chat=chat)

    speech._translation_cache.clear()
    monkeypatch.setattr(speech, "get_settings", lambda: settings)
    monkeypatch.setattr(speech, "get_llm_client", factory)
    assert await speech.translate_ja("Hello") == "こんにちはだよ"
    assert await speech.translate_ja("Hello") == "こんにちはだよ"
    assert len(calls) == 1
    assert calls[0]["model"] == settings.router_model
    assert clients[0]["provider"] == settings.router_llm_provider
    assert clients[0]["timeout_seconds"] == 20
    for index in range(130):
        await speech.translate_ja(f"Hello {index}")
    assert len(speech._translation_cache) == 128
    assert not any(key[-1] == "Hello" for key in speech._translation_cache)
    speech._translation_cache.clear()


@pytest.mark.anyio
async def test_translation_failure_not_cached(monkeypatch):
    speech._translation_cache.clear()

    async def chat(**kwargs):
        return SimpleNamespace(message="")

    monkeypatch.setattr(speech, "get_llm_client", lambda **kwargs: SimpleNamespace(chat=chat))
    with pytest.raises(ValueError):
        await speech.translate_ja("Hi")
    assert not speech._translation_cache


class FakeProcess:
    returncode = None
    terminated = False
    killed = False

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def kill(self):
        self.killed = True
        self.returncode = -9

    async def wait(self):
        return self.returncode


def supervisor_settings(tmp_path, **kwargs):
    python = tmp_path / "python"
    python.touch()
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    return Settings(tts_python=python, tts_models_dir=models, data_dir=tmp_path, **kwargs)


@pytest.mark.anyio
async def test_supervisor_disabled_and_missing(tmp_path):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(503))) as client:
        async def forbidden(*args, **kwargs):
            raise AssertionError("No process may spawn")

        for settings in [Settings(tts_enabled=False), Settings(tts_python=tmp_path / "missing")]:
            supervisor = speech.SidecarSupervisor(settings, client=client, spawn=forbidden)
            await supervisor.start()
            assert supervisor.state == "off"
            assert supervisor.task is None
            await supervisor.stop()


@pytest.mark.anyio
async def test_supervisor_borrows_existing_process(tmp_path):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"state": "ready", "detail": None}))) as client:
        async def forbidden(*args, **kwargs):
            raise AssertionError("Existing process must be reused")

        supervisor = speech.SidecarSupervisor(Settings(), client=client, spawn=forbidden)
        await supervisor.start()
        await asyncio.sleep(0)
        assert supervisor.status() == {"state": "ready", "detail": None}
        await supervisor.stop()
        assert not client.is_closed


@pytest.mark.anyio
async def test_supervisor_spawn_start_ready_stop(tmp_path):
    spawns = []
    process = FakeProcess()

    def health(req):
        return httpx.Response(200, json={"state": "ready", "detail": None}) if spawns else httpx.Response(503)

    async def spawn(*args, **kwargs):
        spawns.append((args, kwargs))
        return process

    async with httpx.AsyncClient(transport=httpx.MockTransport(health)) as client:
        supervisor = speech.SidecarSupervisor(supervisor_settings(tmp_path), client=client, spawn=spawn, poll_seconds=0.001)
        await supervisor.start()
        assert supervisor.state == "starting"
        for _ in range(20):
            await asyncio.sleep(0.001)
            if supervisor.state == "ready":
                break
        assert supervisor.state == "ready"
        assert spawns[0][0][1:3] == ("-m", "aya_tts.server")
        assert spawns[0][1]["env"]["HF_HUB_OFFLINE"] == "1"
        assert spawns[0][1]["env"]["AYA_TTS_MODELS"] == str(tmp_path / "models")
        await supervisor.stop()
        assert process.terminated
        assert supervisor.log is None


@pytest.mark.anyio
async def test_supervisor_resolves_relative_installation_paths_before_child_cwd(tmp_path, monkeypatch):
    from pathlib import Path

    settings = supervisor_settings(tmp_path)
    # A venv interpreter is normally a symlink. Preserve that entry point:
    # resolving it to the base interpreter would lose the venv's packages.
    interpreter = tmp_path / "base-python"
    interpreter.touch()
    settings.tts_python.unlink()
    settings.tts_python.symlink_to(interpreter)
    monkeypatch.chdir(tmp_path)
    settings.tts_python = Path("python")
    settings.tts_models_dir = Path("models")
    spawns = []
    spawned = asyncio.Event()

    async def spawn(*args, **kwargs):
        spawns.append((args, kwargs))
        spawned.set()
        return FakeProcess()

    def health(request):
        return httpx.Response(200, json={"state": "ready"}) if spawns else httpx.Response(503)

    async with httpx.AsyncClient(transport=httpx.MockTransport(health)) as client:
        supervisor = speech.SidecarSupervisor(settings, client=client, spawn=spawn)
        await supervisor.start()
        try:
            await asyncio.wait_for(spawned.wait(), timeout=1)
            args, kwargs = spawns[0]
            assert args[0] == str(tmp_path / "python")
            assert kwargs["env"]["AYA_TTS_MODELS"] == str(tmp_path / "models")
            assert kwargs["cwd"] != str(tmp_path)
        finally:
            await supervisor.stop()


@pytest.mark.anyio
async def test_supervisor_bounded_crash_restarts(tmp_path):
    spawns = []

    async def spawn(*args, **kwargs):
        process = FakeProcess()
        process.returncode = 1
        spawns.append(process)
        return process

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(503))) as client:
        supervisor = speech.SidecarSupervisor(supervisor_settings(tmp_path), client=client, spawn=spawn,
                                              poll_seconds=0.001, restart_delay=0.001)
        await supervisor.start()
        await asyncio.wait_for(supervisor.task, timeout=1)
        assert len(spawns) == 4  # Initial start + at most three quick restarts.
        assert supervisor.state == "error"
        assert "exited (1)" in supervisor.detail
        await supervisor.stop()


@pytest.mark.anyio
async def test_supervisor_startup_error_is_visible(tmp_path):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, json={"state": "error", "detail": "Missing model"}))) as client:
        supervisor = speech.SidecarSupervisor(Settings(), client=client)
        await supervisor.start()
        await supervisor.task
        assert supervisor.status() == {"state": "error", "detail": "Missing model"}
        await supervisor.stop()


def test_settings_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("TTS_ENABLED", "false")
    monkeypatch.setenv("TTS_PORT", "7899")
    monkeypatch.setenv("TTS_PYTHON", str(tmp_path / "python"))
    monkeypatch.setenv("TTS_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("TTS_DEFAULT_VOICE", "sayo")
    get_settings.cache_clear()
    try:
        settings = get_settings()
        assert not settings.tts_enabled
        assert settings.tts_port == 7899
        assert settings.tts_python == tmp_path / "python"
        assert settings.tts_models_dir == tmp_path / "models"
        assert settings.tts_default_voice == "sayo"
    finally:
        get_settings.cache_clear()


def test_app_import_and_creation_do_not_start_sidecar(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("Sidecar must only start in lifespan")

    monkeypatch.setattr(speech.SidecarSupervisor, "start", forbidden)
    from app import main

    monkeypatch.setattr(main, "get_settings", lambda: Settings(data_dir=tmp_path))
    app = main.create_app()
    assert not hasattr(app.state, "speech_supervisor")


def test_lifespan_without_tts_is_off_and_app_works(monkeypatch, tmp_path):
    from app import main
    from app.api import health

    settings = Settings(data_dir=tmp_path, lightrag_enabled=False, tts_python=tmp_path / "missing",
                        telegram_bot_token="", telegram_owner_id=0)

    async def empty(*args, **kwargs):
        return None

    async def unavailable(*args, **kwargs):
        return {"reachable": False}

    async def worker(stop, **kwargs):
        await stop.wait()

    processing = SimpleNamespace(worker_loop=worker, close=empty)
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(health, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "_refresh_catalog_titles", lambda *args: None)
    monkeypatch.setattr(main, "get_llm_client", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(main, "start_memory_fold_coordinator", lambda **kwargs: None)
    monkeypatch.setattr(main, "shutdown_memory_fold_coordinators", empty)
    monkeypatch.setattr(main.DocumentProcessingService, "from_settings", lambda settings: processing)
    monkeypatch.setattr("app.services.reminders.reminder_worker", lambda store, stop, **kwargs: worker(stop))
    monkeypatch.setattr(speech.SidecarSupervisor, "_health", empty)
    monkeypatch.setattr(health.OllamaClient, "health", unavailable)
    monkeypatch.setattr(health.OpenAICompatibleClient, "health", unavailable)
    app = main.create_app()
    with TestClient(app) as client:
        assert client.get("/speech/status").json()["state"] == "off"
        assert client.get("/speech/voices").status_code == 503
        assert client.get("/openapi.json").status_code == 200
        assert client.get("/health").json()["tts"]["state"] == "off"
    assert app.state.speech_supervisor.state == "off"


@pytest.mark.anyio
async def test_supervisor_times_out_loading_and_cleans_processes(tmp_path):
    spawns = []

    def health(request):
        return httpx.Response(200, json={"state": "starting"}) if spawns else httpx.Response(503)

    async def spawn(*args, **kwargs):
        process = FakeProcess()
        spawns.append(process)
        return process

    async with httpx.AsyncClient(transport=httpx.MockTransport(health)) as client:
        supervisor = speech.SidecarSupervisor(supervisor_settings(tmp_path), client=client, spawn=spawn,
                                              startup_timeout=0, restart_delay=0)
        await supervisor.start()
        await asyncio.wait_for(supervisor.task, timeout=1)
        assert len(spawns) == 4
        assert all(process.terminated for process in spawns)
        assert supervisor.state == "error"
        assert "startup timed out" in supervisor.detail
        await supervisor.stop()
