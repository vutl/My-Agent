import os

# Set before importing any engine: runtime must never check the network.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import asyncio
from contextlib import asynccontextmanager
import io
import logging
import threading
import time
import wave

from fastapi import FastAPI, HTTPException, Response

from .config import DEFAULT_PORT, models_dir

os.environ["HF_HOME"] = str(models_dir() / "huggingface")
os.environ["HF_HUB_CACHE"] = str(models_dir() / "huggingface/hub")
from .models import SynthesisRequest

logger = logging.getLogger(__name__)


def load_engines():
    from .vieneu_engine import VieneuEngine
    from .voicevox_engine import VoicevoxEngine

    return [VieneuEngine(models_dir()).load(), VoicevoxEngine(models_dir()).load()]


def create_app(loader=load_engines):
    @asynccontextmanager
    async def lifespan(app):
        app.state.health = {"state": "starting", "detail": None, "load_seconds": None}

        async def load():
            started = time.monotonic()
            try:
                engines = await asyncio.to_thread(loader)
                app.state.voices = {voice["id"]: (voice, engine) for engine in engines for voice in engine.voices}
                app.state.locks = {id(engine): threading.Lock() for engine in engines}
                app.state.health.update(state="ready")
            except Exception as exc:
                logger.exception("TTS engine load failed")
                app.state.health.update(state="error", detail=str(exc))
            finally:
                app.state.health["load_seconds"] = time.monotonic() - started

        task = asyncio.create_task(load())
        try:
            yield
        finally:
            # Engine loading is a bounded offline operation; do not abandon its thread.
            await task

    app = FastAPI(lifespan=lifespan)
    app.state.health = {"state": "starting", "detail": None, "load_seconds": None}

    def ready():
        if app.state.health["state"] != "ready":
            raise HTTPException(503, "Speech engines are not ready")

    @app.get("/health")
    async def health():
        return app.state.health

    @app.get("/voices")
    async def voices():
        ready()
        return {"voices": [voice for voice, _ in app.state.voices.values()]}

    @app.post("/synthesize")
    async def synthesize(body: SynthesisRequest):
        ready()
        entry = app.state.voices.get(body.voice_id)
        if entry is None:
            raise HTTPException(404, "Unknown voice")
        _, engine = entry

        def render():
            import numpy as np

            with app.state.locks[id(engine)]:
                audio, rate = engine.synthesize(body.text, body.voice_id, body.speed, body.pitch)
            audio = np.asarray(audio)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            output = io.BytesIO()
            # The API boundary needs only PCM WAV; stdlib encoding also lets
            # backend-env tests exercise the real encoder without audio libraries.
            pcm = np.rint(np.clip(audio, -1.0, 32767 / 32768) * 32768).astype("<i2")
            with wave.open(output, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(rate)
                wav.writeframes(pcm.tobytes())
            return output.getvalue()

        return Response(await asyncio.to_thread(render), media_type="audio/wav")

    return app


app = create_app()


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
