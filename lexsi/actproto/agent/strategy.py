"""Agent policies.

Activation contract: the tool decision is made from the hidden state at
config.activation.token (prompt_last or pre_tool_name) BEFORE tool-name
tokens are sampled. H1 hooks that location. This module does not register
hooks.

ReActStrategy accepts a policy callable: state -> {name, arguments}.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from actproto.agent.harness import DictToolHarness
from actproto.agent.parse import ParseError, parse_tool_call
from actproto.agent.prompts import system_prompt
from actproto.agent.trajectory import EventLog, append_tool_turn


@dataclass
class AgentState:
    task: str
    step: int
    log: EventLog
    last_result: dict[str, Any] | None = None


Policy = Callable[[AgentState], dict[str, Any]]


@dataclass
class ReActStrategy:
    policy: Policy
    harness: DictToolHarness
    max_steps: int | None = None

    def run(self, task: str) -> EventLog:
        self.harness.reset()
        log = EventLog(
            task=task,
            messages=[
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": task},
            ],
        )
        limit = self.max_steps or self.harness.max_steps
        for step in range(limit):
            state = AgentState(task=task, step=step, log=log, last_result=log.tool_results[-1].output if log.tool_results else None)
            action = self.policy(state)
            name = action.get("name")
            arguments = action.get("arguments") or {}
            if not isinstance(name, str):
                break
            result = self.harness.execute(name, arguments)
            append_tool_turn(log, result, call_id=f"c{step}")
            if name in {"finish", "no_tool"}:
                log.final_answer = (result.output.get("data") or {}).get("answer")
                break
        log.final_state = dict(self.harness.final_state)
        return log


def scripted_policy(steps: list[dict[str, Any]]) -> Policy:
    remaining = list(steps)

    def _policy(_state: AgentState) -> dict[str, Any]:
        if remaining:
            return remaining.pop(0)
        return {"name": "finish", "arguments": {"answer": ""}}

    return _policy


def model_policy(generate_fn: Callable[[list[dict[str, Any]]], str]) -> Policy:
    """Policy that asks a generate_fn for the next tool call and fails closed on parse errors."""

    def _policy(state: AgentState) -> dict[str, Any]:
        text = generate_fn(state.log.messages)
        parsed = parse_tool_call(text)
        if isinstance(parsed, ParseError):
            return {"name": "finish", "arguments": {"answer": f"PARSE_ERROR: {parsed.message}"}}
        return {"name": parsed.name, "arguments": parsed.arguments}

    return _policy
