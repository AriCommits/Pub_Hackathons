"""Project-root path containment helpers."""

from __future__ import annotations

from pathlib import Path


class PathEscapeError(ValueError):
    """A path resolved outside the allowed project root."""


def contain_under(root: Path, path: str | Path, *, label: str = "path") -> Path:
    """Resolve ``path`` and require it to stay under ``root``.

    Absolute paths and ``..`` segments that escape the project are rejected.
    """
    root = root.resolve()
    candidate = Path(path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise PathEscapeError(f"{label} escapes project root: {path}")
    return resolved
