"""EventLog helpers and OpenAI-style message conversion for AuditKIT."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from actproto.agent.harness import ToolResult


@dataclass
class EventLog:
    task: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    final_state: dict[str, Any] = field(default_factory=dict)
    final_answer: str | None = None
    steps: int = 0

    def as_openai_messages(self) -> list[dict[str, Any]]:
        return list(self.messages)


def eventlog_to_openai_messages(log: EventLog) -> list[dict[str, Any]]:
    return log.as_openai_messages()


def append_tool_turn(log: EventLog, result: ToolResult, call_id: str) -> None:
    log.messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": result.name,
                        "arguments": _dump_args(result.arguments),
                    },
                }
            ],
        }
    )
    log.messages.append(
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": _dump_args(result.output),
        }
    )
    log.tool_results.append(result)
    log.steps += 1


def _dump_args(obj: Any) -> str:
    import json

    return json.dumps(obj)
