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


def messages_for_template(messages: list[dict]) -> list[dict]:
    """Render log messages for the Cohere chat template.

    That template only alternates user and assistant turns. Tool results are
    sent back as user text. OpenAI-style tool_calls are turned into the same
    ``<tool_call>`` string the parser reads.
    """
    from actproto.training.sft_dataset import format_tool_call

    converted: list[dict] = []
    for msg in messages:
        role = str(msg.get("role") or "").lower()
        if role == "tool":
            converted.append({"role": "user", "content": f"Tool result:\n{msg.get('content') or ''}"})
            continue
        if role == "assistant" and msg.get("tool_calls"):
            call = msg["tool_calls"][0]["function"]
            args = call.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"raw": args}
            converted.append({"role": "assistant", "content": format_tool_call(str(call.get("name")), args)})
            continue
        content = msg.get("content")
        if content is None:
            continue
        if role in {"system", "user", "assistant"}:
            converted.append({"role": role, "content": str(content)})
    return converted
