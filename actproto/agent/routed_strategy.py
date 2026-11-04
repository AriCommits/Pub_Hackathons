"""Wrap A1 ReActStrategy with prototype routing. Do not edit strategy.py."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from actproto.activations.hooks import ActivationHook
from actproto.agent.abstention import Threshold
from actproto.agent.arg_prompt import argument_prompt
from actproto.agent.harness import DictToolHarness
from actproto.agent.parse import ParseError, parse_tool_call
from actproto.agent.project import Project
from actproto.agent.prompts import messages_for_template
from actproto.agent.router import route_vector
from actproto.agent.schema_regen import regenerate_arguments
from actproto.agent.strategy import AgentState, Policy, ReActStrategy
from actproto.configs.schema import ExperimentConfig, load_default
from actproto.tools.schemas import validate_args

_FINISH_RETRY = "The finish answer must be non-empty. Emit one tool call."

GenerateFn = Callable[[list[dict[str, Any]]], str]


def _append_user_line(messages: list[dict[str, Any]], line: str) -> list[dict[str, Any]]:
    copied = [dict(message) for message in messages]
    if copied and copied[-1].get("role") == "user":
        copied[-1]["content"] = f"{copied[-1]['content']}\n\n{line}"
    else:
        copied.append({"role": "user", "content": line})
    return copied


def _valid_finish(parsed: Any) -> bool:
    if isinstance(parsed, ParseError) or parsed.name != "finish":
        return False
    answer = parsed.arguments.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return False
    ok, _reason = validate_args("finish", parsed.arguments)
    return ok


def routed_policy(
    inner: Policy,
    *,
    prototypes: dict[str, np.ndarray],
    threshold: Threshold | None,
    hook: ActivationHook | None = None,
    vector_fn=None,
    generate_fn: GenerateFn | None = None,
) -> Policy:
    def _policy(state: AgentState) -> dict[str, Any]:
        vector = None
        if vector_fn is not None:
            vector = vector_fn(state)
        elif hook is not None and hook.last is not None:
            vector = hook.numpy_vector()
        if vector is None:
            return inner(state)
        decision = route_vector(np.asarray(vector), prototypes, threshold)
        if decision["abstain"] or not decision["tool"]:
            return inner(state)

        selected_tool = decision["tool"]
        model_fn = None
        if generate_fn is not None:

            def model_fn(_tool: str, _prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
                prompt = argument_prompt(selected_tool, schema.get("properties", {}))
                messages = _append_user_line(messages_for_template(state.log.messages), prompt)
                parsed = parse_tool_call(generate_fn(messages))
                if selected_tool != "finish":
                    if isinstance(parsed, ParseError) or parsed.name != selected_tool:
                        return {"__invalid__": True}
                    return parsed.arguments
                if _valid_finish(parsed):
                    return parsed.arguments
                retry = _append_user_line(messages, _FINISH_RETRY)
                parsed = parse_tool_call(generate_fn(retry))
                if _valid_finish(parsed):
                    return parsed.arguments
                return {"__invalid__": True}

        args = regenerate_arguments(selected_tool, state.task, model_fn=model_fn)
        last_call = state.log.tool_results[-1] if state.log.tool_results else None
        if (
            last_call is not None
            and last_call.output.get("ok") is True
            and last_call.name == selected_tool
            and last_call.arguments == args
        ):
            return inner(state)
        return {"name": selected_tool, "arguments": args}

    return _policy


def build_routed_project(
    config: ExperimentConfig | None = None,
    *,
    prototypes: dict[str, np.ndarray],
    threshold: Threshold | None,
    inner_policy: Policy,
    vector_fn=None,
    generate_fn: GenerateFn | None = None,
) -> Project:
    config = config or load_default()
    harness = DictToolHarness(max_steps=config.max_agent_steps)
    policy = routed_policy(
        inner_policy,
        prototypes=prototypes,
        threshold=threshold,
        vector_fn=vector_fn,
        generate_fn=generate_fn,
    )
    strategy = ReActStrategy(policy=policy, harness=harness, max_steps=config.max_agent_steps)
    return Project(strategy=strategy, harness=harness)
