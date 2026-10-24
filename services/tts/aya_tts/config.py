import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def models_dir() -> Path:
    return Path(os.environ.get("AYA_TTS_MODELS", str(REPO_ROOT / "data/models/tts"))).expanduser()


DEFAULT_PORT = 7801
