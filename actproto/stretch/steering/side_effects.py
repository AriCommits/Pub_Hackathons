"""Flag empty/invalid JSON and loops after a steering intervention."""

from __future__ import annotations

from typing import Any

from actproto.agent.parse import ParseError, parse_tool_call


def flag_output(text: str, pred_tools: list[str] | None = None) -> dict[str, Any]:
    parsed = parse_tool_call(text)
    invalid = isinstance(parsed, ParseError)
    empty = not (text or "").strip()
    loops = 0
    if pred_tools:
        loops = sum(1 for i in range(1, len(pred_tools)) if pred_tools[i] == pred_tools[i - 1])
    return {"empty": empty, "invalid_json": invalid, "loops": loops}
