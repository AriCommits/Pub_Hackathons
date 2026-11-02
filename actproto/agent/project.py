"""Project factory: AgentTune Project if installed, otherwise a local stand-in."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from actproto.agent.harness import DictToolHarness, build_harness
from actproto.agent.strategy import Policy, ReActStrategy, model_policy, scripted_policy
from actproto.agent.trajectory import EventLog
from actproto.configs.schema import ExperimentConfig, load_default


@dataclass
class Project:
    strategy: ReActStrategy
    harness: DictToolHarness

    def infer(self, task: str) -> EventLog:
        return self.strategy.run(task)


def build_project(
    config: ExperimentConfig | None = None,
    strategy: ReActStrategy | None = None,
    policy: Policy | None = None,
) -> Project:
    config = config or load_default()
    harness = build_harness(max_steps=config.max_agent_steps)
    if strategy is None:
        if policy is None:
            raise ValueError("pass strategy= or policy= (scripted or model-backed)")
        strategy = ReActStrategy(policy=policy, harness=harness, max_steps=config.max_agent_steps)
    return Project(strategy=strategy, harness=harness)


def build_scripted_project(steps: list[dict[str, Any]], max_steps: int = 6) -> Project:
    harness = DictToolHarness(max_steps=max_steps)
    strategy = ReActStrategy(policy=scripted_policy(steps), harness=harness, max_steps=max_steps)
    return Project(strategy=strategy, harness=harness)
