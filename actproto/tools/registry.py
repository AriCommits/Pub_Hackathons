"""Name → callable map and AgentTune-friendly harness dict.

`no_tool` is an alias of `finish`: both mean "answer or stop without another
tool". Prototypes and gold labels should canonicalize to `finish`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from actproto.tools.calculator import calculator
from actproto.tools.finish import finish
from actproto.tools.lookup_record import lookup_record
from actproto.tools.query_database import query_database
from actproto.tools.schemas import SCHEMAS, validate_args
from actproto.tools.search_docs import search_docs

PRIMARY_TOOLS = (
    "calculator",
    "search_docs",
    "query_database",
    "lookup_record",
    "finish",
)

TOOLS: dict[str, Callable[..., dict[str, Any]]] = {
    "calculator": calculator,
    "search_docs": search_docs,
    "query_database": query_database,
    "lookup_record": lookup_record,
    "finish": finish,
    "no_tool": finish,
}


def canonical_tool_name(name: str) -> str:
    return "finish" if name == "no_tool" else name


def tool_names() -> list[str]:
    return list(PRIMARY_TOOLS)


def as_harness_dict() -> dict[str, Callable[..., dict[str, Any]]]:
    """Map AgentTune DictToolHarness expects: name → callable."""
    return dict(TOOLS)


def call_tool(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    arguments = arguments or {}
    if name not in TOOLS:
        return {"ok": False, "error": {"type": "unknown_tool", "message": f"unknown tool {name}"}}
    ok, reason = validate_args(name, arguments)
    if not ok:
        return {"ok": False, "error": {"type": "invalid_arguments", "message": reason}}
    return TOOLS[name](**arguments)


__all__ = [
    "PRIMARY_TOOLS",
    "SCHEMAS",
    "TOOLS",
    "as_harness_dict",
    "call_tool",
    "canonical_tool_name",
    "tool_names",
    "validate_args",
]
