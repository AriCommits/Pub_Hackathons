"""Isolated, provenance-capturing v2 fixture execution for rollout collection."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from actproto.experiments.contracts import TaskSpec


class ToolSession:
    def __init__(self, task: TaskSpec, environment: Any, *, allow_unscoped_fixture=False):
        from actproto.training.rl_dataset import VerifiedTrainFixtures
        self.unscoped_fixture = not isinstance(environment, VerifiedTrainFixtures)
        if not self.unscoped_fixture:
            # ScopedFixtures returns a shared view; copy all mutable backing data
            # so tool state and nested fixture dictionaries cannot cross episodes.
            environment = deepcopy(environment.for_task(task))
        elif not allow_unscoped_fixture:
            raise ValueError("training collection requires manifest-verified ScopedFixtures")
        if environment.identifier != task.environment_id or environment.digest != task.fixture_hash:
            raise ValueError("fixture environment mismatch")
        self.task = task
        self.environment = environment
        self.state: dict[str, Any] = {}
        self.events: list[dict[str, Any]] = []

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        before = deepcopy(self.state)
        try:
            output = self.environment.call(self.task.environment_id, name, deepcopy(arguments))
            if not isinstance(output, dict) or type(output.get("ok")) is not bool:
                raise ValueError("tool output does not conform to v2")
            output = deepcopy(output)
        except Exception as exc:
            output = {"ok": False, "error": {"type": "tool_exception", "message": str(exc)}}
        if name in ("finish", "no_tool") and output.get("ok"):
            data = output.get("data")
            if not isinstance(data, dict) or not isinstance(data.get("answer"), str) or data.get("done") is not True:
                output = {"ok": False, "error": {"type": "invalid_tool_output", "message": "invalid finish result"}}
            else:
                self.state.update({"answer": data["answer"], "done": True})
        self.events.append({"kind": "tool_call", "source": "harness", "tool": name,
                            "arguments": deepcopy(arguments), "output": deepcopy(output),
                            "state_before": before, "state_after": deepcopy(self.state)})
        return output


def new_session(task: TaskSpec, environment_factory: Callable[[], Any], *, allow_unscoped_fixture=False) -> ToolSession:
    return ToolSession(task, environment_factory(), allow_unscoped_fixture=allow_unscoped_fixture)
