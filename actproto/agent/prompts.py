"""Short system prompt listing tools and schemas."""

from __future__ import annotations

import json

from actproto.tools.schemas import SCHEMAS
from actproto.tools.registry import PRIMARY_TOOLS

DESCRIPTIONS = {
    "calculator": "Evaluate a numeric expression.",
    "search_docs": "Keyword search over a local document corpus.",
    "query_database": "Filter the local sales table by product/region/year.",
    "lookup_record": "Fetch one inventory record by id (REC-...).",
    "finish": "Return the final answer and stop. Also used when no tool is needed (no_tool).",
}


def system_prompt() -> str:
    lines = [
        "You are a tool-using agent. Call exactly one tool per turn.",
        "Emit either <tool_call>{\"name\": \"...\", \"arguments\": {...}}</tool_call>",
        "or a JSON object with name and arguments.",
        "When you are done, call finish with your answer. Do not invent tool results.",
        "Tools:",
    ]
    for name in PRIMARY_TOOLS:
        schema = SCHEMAS[name]
        lines.append(f"- {name}: {DESCRIPTIONS[name]} schema={json.dumps(schema['properties'])}")
    return "\n".join(lines)
