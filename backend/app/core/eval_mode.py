"""Agent evaluation mode: env-only switches, off by default.

``AYA_EVAL_MODE=1`` lets the evaluation runner point an isolated backend at
local fixtures. ``AYA_EVAL_WEB_ALLOW=host:port[,...]`` lets ``web_fetch`` reach
exactly those loopback servers (every other private host stays blocked), and
``AYA_EVAL_STUB_MACOS=1`` makes the macOS integrations record ``{tool, args}``
to ``$APP_DATA_DIR/eval_calls.jsonl`` instead of touching the Mac. Eval mode
refuses to start on the production data directory.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlparse

from app.core.config import PROJECT_ROOT, Settings

logger = logging.getLogger(__name__)

CALLS_FILE = "eval_calls.jsonl"
NETWORK_FILE = "eval_network.jsonl"
_TRUE = {"1", "true", "yes", "on"}


class EvalModeError(RuntimeError):
    pass


def _flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUE


def eval_mode_enabled() -> bool:
    return _flag("AYA_EVAL_MODE")


def stub_macos_enabled() -> bool:
    return eval_mode_enabled() and _flag("AYA_EVAL_STUB_MACOS")


def web_allowed(host: str | None, port: int) -> bool:
    """Exact host:port match against AYA_EVAL_WEB_ALLOW; never a prefix or wildcard."""
    if not eval_mode_enabled() or not host:
        return False
    allowed = {item.strip().lower() for item in os.getenv("AYA_EVAL_WEB_ALLOW", "").split(",") if item.strip()}
    return f"{host.lower()}:{port}" in allowed


def production_data_dir() -> Path:
    return (PROJECT_ROOT / "data").resolve()


def validate_eval_mode(settings: Settings) -> None:
    """Warn when eval mode is on; refuse to run it against the production data directory."""
    if not eval_mode_enabled():
        return
    if Path(settings.data_dir).expanduser().resolve() == production_data_dir():
        raise EvalModeError("AYA_EVAL_MODE refuses to run on the default production data directory; set APP_DATA_DIR")
    logger.warning(
        "AYA EVAL MODE is ON (web_allow=%s stub_macos=%s data_dir=%s): not for normal use",
        os.getenv("AYA_EVAL_WEB_ALLOW", ""), stub_macos_enabled(), settings.data_dir,
    )


def _append(name: str, entry: dict[str, Any]) -> None:
    from app.core.config import get_settings

    path = Path(get_settings().data_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"ts": time.time(), **entry}, ensure_ascii=False) + "\n")


def record_call(tool: str, args: dict[str, Any]) -> None:
    """Append one stubbed call so the evaluator can see what Aya tried to do."""
    _append(CALLS_FILE, {"tool": tool, "args": args})


def record_network(url: str, kind: str) -> None:
    """Eval mode only: audit one outbound request actually issued (every redirect hop is its own entry)."""
    if not eval_mode_enabled():
        return
    parsed = urlparse(url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    _append(NETWORK_FILE, {"kind": kind, "url": url, "host": f"{(parsed.hostname or '').lower()}:{port}"})
