"""Load generated tasks for eval splits."""

from __future__ import annotations

from actproto.data.generator import generate_tasks
from actproto.data.types import Task


def load_tasks(split: str, limit: int | None = None) -> list[Task]:
    tasks = [t for t in generate_tasks() if t.split == split]
    if limit is not None:
        tasks = tasks[:limit]
    return tasks
