"""Prompt construction for arguments of a router-selected tool."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def argument_prompt(tool_name: str, schema_properties: Mapping[str, Any]) -> str:
    """Ask for one call to tool_name while exposing only its argument keys."""
    argument_keys = ", ".join(schema_properties)
    return f"Use this tool only: {tool_name}({argument_keys}). Emit one tool call."
