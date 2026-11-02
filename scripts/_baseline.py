"""Shared gold-replay / dry-run helpers for B0–B2 scripts. Not a public API."""

from __future__ import annotations

import json
from pathlib import Path

from actproto.agent.project import build_scripted_project
from actproto.data.generator import generate_tasks
from actproto.data.types import Task
from actproto.evaluation.cases import task_to_case
from actproto.evaluation.episodes import eventlog_to_episode
from actproto.evaluation.report import write_report
from actproto.evaluation.runner import AgentEvalRunner, AgentEvalSpec
from actproto.tools.registry import canonical_tool_name


def load_tasks(split: str, limit: int | None) -> list[Task]:
    tasks = [t for t in generate_tasks() if t.split == split]
    if limit is not None:
        tasks = tasks[:limit]
    return tasks


def gold_steps(task: Task) -> list[dict]:
    return [{"name": canonical_tool_name(n), "arguments": a} for n, a in zip(task.gold_tools, task.gold_args)]


def replay_eval(tasks: list[Task], out: str) -> dict:
    cases = [task_to_case(t) for t in tasks]
    episodes = []
    for task in tasks:
        proj = build_scripted_project(gold_steps(task))
        log = proj.infer(task.prompt)
        # Gold replay should satisfy oracles that only look at finish answer / done.
        episodes.append(eventlog_to_episode(log, task.id))
    spec = AgentEvalSpec(cases=cases, mode="recorded", episodes=episodes)
    result = AgentEvalRunner().run(spec)
    return write_report(result, out)


def write_json(path: str, payload: dict) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
