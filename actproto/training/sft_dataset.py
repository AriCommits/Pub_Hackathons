"""Gold trajectories → AgentTune SFT row shape {messages, segment_weights, loss_mask}."""

from __future__ import annotations

import json
from typing import Any

from actproto.data.types import Task
from actproto.tools.registry import call_tool, canonical_tool_name


def format_tool_call(name: str, arguments: dict[str, Any]) -> str:
    payload = {"name": canonical_tool_name(name), "arguments": arguments}
    return f"<tool_call>{json.dumps(payload)}</tool_call>"


def gold_to_sft_row(task: Task, replay_tools: bool = True) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [{"role": "user", "content": task.prompt}]
    segment_weights = [0.0]
    loss_mask = [True]
    for name, args in zip(task.gold_tools, task.gold_args, strict=True):
        messages.append({"role": "assistant", "content": format_tool_call(name, args)})
        segment_weights.append(2.0)
        loss_mask.append(False)
        if canonical_tool_name(name) == "finish":
            break
        content = ""
        if replay_tools:
            result = call_tool(name, args)
            content = json.dumps(result)
        messages.append({"role": "tool", "content": content})
        segment_weights.append(0.0)
        loss_mask.append(True)
    assert len(messages) == len(segment_weights) == len(loss_mask)
    assert all((w == 0.0) == m for w, m in zip(segment_weights, loss_mask))
    return {"messages": messages, "segment_weights": segment_weights, "loss_mask": loss_mask}


def tasks_to_sft_rows(tasks: list[Task], replay_tools: bool = True) -> list[dict[str, Any]]:
    return [gold_to_sft_row(t, replay_tools=replay_tools) for t in tasks]
