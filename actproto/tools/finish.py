"""Terminal action: store the agent's final answer."""

from __future__ import annotations

from typing import Any


def finish(answer: str = "") -> dict[str, Any]:
    if not isinstance(answer, str):
        return {"ok": False, "error": {"type": "invalid_answer", "message": "answer must be a string"}}
    return {"ok": True, "data": {"answer": answer, "done": True}}
