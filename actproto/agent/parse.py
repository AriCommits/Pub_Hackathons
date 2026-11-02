"""Extract a single {name, arguments} tool call. Fail closed on garbage."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


@dataclass
class ParseError:
    message: str
    raw: str


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


ParseResult = ToolCall | ParseError


def _as_call(obj: Any) -> ToolCall | None:
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    arguments = obj.get("arguments", {})
    if not isinstance(name, str) or not name:
        return None
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return None
    return ToolCall(name=name, arguments=arguments)


def parse_tool_call(text: str) -> ParseResult:
    if not isinstance(text, str) or not text.strip():
        return ParseError("empty model output", raw=text or "")

    candidates: list[str] = []
    tag_hits = TOOL_CALL_RE.findall(text)
    fence_hits = FENCE_RE.findall(text)
    if len(tag_hits) > 1:
        return ParseError("multiple tool calls in one turn; refusing to guess", raw=text)
    candidates.extend(tag_hits)
    candidates.extend(fence_hits)

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        candidates.append(stripped)

    parsed_objects: list[Any] = []
    for blob in candidates:
        try:
            parsed_objects.append(json.loads(blob))
        except json.JSONDecodeError:
            continue

    calls = [c for c in (_as_call(obj) for obj in parsed_objects) if c is not None]
    if len(calls) == 1:
        return calls[0]
    if len(calls) > 1:
        return ParseError("multiple tool calls in one turn; refusing to guess", raw=text)
    return ParseError("no valid tool call object", raw=text)
