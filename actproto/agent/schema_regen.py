"""Regenerate schema-valid arguments for a selected tool.

Never rewrite the `name` field of an already-generated call, and never reuse
arguments produced for a different tool.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from actproto.tools.fixtures import PRODUCTS, RECORD_IDS
from actproto.tools.schemas import SCHEMAS, validate_args

RegenFn = Callable[[str, str, dict], dict[str, Any]]


def regenerate_arguments(
    selected_tool: str,
    user_prompt: str,
    previous_call: dict[str, Any] | None = None,
    model_fn: RegenFn | None = None,
) -> dict[str, Any]:
    """Return arguments for selected_tool, validated against its schema.

    If previous_call is for the SAME tool and already valid, keep those args.
    If previous_call is for a DIFFERENT tool, discard its arguments entirely.
    """
    schema = SCHEMAS[selected_tool]
    if previous_call and previous_call.get("name") == selected_tool:
        args = dict(previous_call.get("arguments") or {})
        ok, _ = validate_args(selected_tool, args)
        if ok:
            return args

    if model_fn is not None:
        candidate = model_fn(selected_tool, user_prompt, schema)
        ok, _ = validate_args(selected_tool, candidate)
        if ok:
            return candidate

    extracted = heuristic_args(selected_tool, user_prompt)
    ok, _ = validate_args(selected_tool, extracted)
    if ok:
        return extracted

    # Last resort: schema-valid empty-ish defaults that still type-check.
    return _empty_valid(selected_tool)


def heuristic_args(tool: str, prompt: str) -> dict[str, Any]:
    if tool == "calculator":
        expr = _first_expression(prompt)
        return {"expression": expr} if expr else {}
    if tool == "lookup_record":
        match = re.search(r"REC-\d+", prompt)
        return {"record_id": match.group(0)} if match else {}
    if tool == "search_docs":
        quoted = re.search(r"'([^']+)'", prompt)
        return {"query": quoted.group(1) if quoted else prompt[:80]}
    if tool == "query_database":
        args: dict[str, Any] = {}
        for product in PRODUCTS:
            if re.search(rf"\b{re.escape(product)}\b", prompt, re.I):
                args["product"] = product
                break
        year = re.search(r"\b(20\d{2})\b", prompt)
        if year:
            args["year"] = int(year.group(1))
        return args
    if tool in {"finish", "no_tool"}:
        return {"answer": _guess_answer(prompt)}
    return {}


def _first_expression(prompt: str) -> str | None:
    match = re.search(r"(\d+\s*[\+\-\*/x×]\s*\d+)", prompt)
    if not match:
        return None
    return match.group(1).replace("x", "*").replace("×", "*")


def _guess_answer(prompt: str) -> str:
    if "NOT_FOUND" in prompt:
        return "NOT_FOUND"
    if "NO_ROWS" in prompt:
        return "NO_ROWS"
    return ""


def _empty_valid(tool: str) -> dict[str, Any]:
    if tool == "calculator":
        return {"expression": "0"}
    if tool == "lookup_record":
        return {"record_id": RECORD_IDS[0]}
    if tool == "search_docs":
        return {"query": "widget"}
    if tool == "query_database":
        return {}
    return {"answer": ""}
