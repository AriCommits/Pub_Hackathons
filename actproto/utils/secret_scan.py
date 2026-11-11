"""Shared basename + content secret checks for export paths."""

from __future__ import annotations

import re
from pathlib import Path

SECRET = re.compile(
    r"(?:-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|"
    r"AKIA[A-Z0-9]{16}|hf_[A-Za-z0-9]{20,})\b)"
)
SECRET_FIELD = re.compile(
    r"(?i)[\"'](?:api[_-]?key|secret|password|access[_-]?token|authorization|credential)"
    r"[\"']\s*:\s*[\"'][^\"']+[\"']"
)

FORBIDDEN_NAMES = {
    ".env",
    "HF_TOKEN",
    ".hf_token",
    "secrets.yaml",
    "secrets.json",
    "credentials",
    "credentials.json",
    "resume-token.txt",
}
FORBIDDEN_SUFFIXES = {".env", ".pem", ".key", ".p12", ".pfx"}


def is_forbidden_name(path: Path) -> bool:
    name = path.name
    lower = name.lower()
    return lower in {n.lower() for n in FORBIDDEN_NAMES} or path.suffix.lower() in FORBIDDEN_SUFFIXES


def scan_file_for_secrets(path: Path) -> str | None:
    """Return a reason string if the file looks like a secret, else None."""
    if is_forbidden_name(path):
        return f"forbidden name: {path.name}"
    with path.open("rb") as file:
        tail = b""
        for block in iter(lambda: file.read(1024 * 1024), b""):
            chunk = (tail + block).decode("latin-1")
            if SECRET.search(chunk) or SECRET_FIELD.search(chunk):
                return f"credential-like material in {path}"
            tail = (tail + block)[-512:]
    return None


def assert_paths_clean(files: list[Path], *, raise_system_exit: bool = False) -> None:
    bad: list[str] = []
    for path in files:
        reason = scan_file_for_secrets(path)
        if reason:
            bad.append(reason)
    if not bad:
        return
    message = "refusing to export secrets: " + "; ".join(bad)
    if raise_system_exit:
        raise SystemExit(message)
    raise ValueError(message)


def write_secret_file(path: Path, text: str, *, mode: int = 0o600) -> None:
    """Write a secret file and restrict permissions when the OS supports it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    try:
        path.chmod(mode)
    except OSError:
        pass
