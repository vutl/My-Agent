"""Where Aya's file and code tools may reach.

One policy for both: the workspace (``~/Aya`` by default) is Aya's own space
(read and write without asking); the rest of the home folder can be read, and
written only with the user's approval; nothing outside the home folder is
written. Secrets are never readable: SSH/GPG keys, ``~/Library`` (keychains,
browser cookies, mail, messages), cloud and assistant credentials, ``.env``
files and private keys.
"""

from __future__ import annotations

from pathlib import Path
import re

_SECRET_DIRS = (
    ".ssh", ".gnupg", ".aws", ".azure", ".kube", ".docker", ".config/gcloud", ".config/gh",
    ".codex", ".codex-vutlpro", ".codex-uclab", ".claude", "Library",
)
_SECRET_NAME_RE = re.compile(
    r"(?:^\.env(?:\..*)?$|^\.netrc$|^\.npmrc$|^\.pypirc$|id_(?:rsa|ed25519|ecdsa|dsa)(?:\.pub)?$|\.(?:pem|key|p12|pfx|keychain-db)$)",
    re.IGNORECASE,
)


class PathPolicyError(PermissionError):
    pass


def home() -> Path:
    return Path.home().resolve()


def secret_roots() -> list[Path]:
    return [home() / name for name in _SECRET_DIRS]


def resolve(raw: str, workspace: Path) -> Path:
    """Absolute, symlink-free path; relative paths are inside the workspace."""
    text = str(raw or "").strip()
    if not text:
        raise PathPolicyError("empty path")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = workspace / path
    return path.resolve()


def is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
        return True
    except ValueError:
        return False


def is_secret(path: Path) -> bool:
    return any(is_inside(path, root) for root in secret_roots()) or bool(_SECRET_NAME_RE.search(path.name))


def check_readable(path: Path) -> None:
    if is_secret(path):
        raise PathPolicyError(f"{path} is private (keys, credentials, Library) and never read by Aya")


def check_writable(path: Path) -> None:
    check_readable(path)
    if not is_inside(path, home()):
        raise PathPolicyError(f"{path} is outside the home folder")


def write_needs_approval(path: Path, workspace: Path) -> bool:
    return not is_inside(path, workspace)
