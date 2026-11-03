"""Wrap A1 ReActStrategy with prototype routing. Do not edit strategy.py."""

from __future__ import annotations

from typing import Any

import numpy as np

from actproto.activations.hooks import ActivationHook
from actproto.agent.abstention import Threshold
from actproto.agent.harness import DictToolHarness
from actproto.agent.project import Project
from actproto.agent.router import route_vector
from actproto.agent.schema_regen import regenerate_arguments
from actproto.agent.strategy import AgentState, Policy, ReActStrategy
from actproto.configs.schema import ExperimentConfig, load_default


def routed_policy(
    inner: Policy,
    *,
    prototypes: dict[str, np.ndarray],
    threshold: Threshold | None,
    hook: ActivationHook | None = None,
    vector_fn=None,
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
        inner_call = inner(state)
        previous = inner_call if isinstance(inner_call, dict) else None
        args = regenerate_arguments(decision["tool"], state.task, previous_call=previous)
        last_call = state.log.tool_results[-1] if state.log.tool_results else None
        if (
            last_call is not None
            and last_call.output.get("ok") is True
            and last_call.name == decision["tool"]
            and last_call.arguments == args
        ):
            return inner_call
        return {"name": decision["tool"], "arguments": args}

    return _policy


def build_routed_project(
    config: ExperimentConfig | None = None,
    *,
    prototypes: dict[str, np.ndarray],
    threshold: Threshold | None,
    inner_policy: Policy,
    vector_fn=None,
) -> Project:
    config = config or load_default()
    harness = DictToolHarness(max_steps=config.max_agent_steps)
    policy = routed_policy(inner_policy, prototypes=prototypes, threshold=threshold, vector_fn=vector_fn)
    strategy = ReActStrategy(policy=policy, harness=harness, max_steps=config.max_agent_steps)
    return Project(strategy=strategy, harness=harness)
