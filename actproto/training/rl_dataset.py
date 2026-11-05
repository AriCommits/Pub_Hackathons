"""Load v2 train tasks without ever exposing private task fields to a policy."""
from __future__ import annotations

import json
from pathlib import Path

from actproto.experiments.contracts import TaskSpec


def load_train_tasks(path: str | Path) -> list[TaskSpec]:
    tasks = []
    seen = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                raw = row.get("task", row)
                task = TaskSpec.from_dict(raw)
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                raise ValueError(f"invalid task at line {line_number}") from exc
            if task.split != "train":
                raise ValueError(f"non-train task at line {line_number}")
            if not task.task_id or task.task_id in seen:
                raise ValueError(f"missing or duplicate task ID at line {line_number}")
            seen.add(task.task_id)
            tasks.append(task)
    if not tasks:
        raise ValueError("training task dataset is empty")
    return tasks


class VerifiedTrainFixtures:
    """Manifest-verified split scope with the exact authorized train tasks."""
    def __init__(self, manifest_path: str | Path):
        from actproto.evaluation.experiment_runner import load_data
        self._scope, tasks = load_data(manifest_path, 'train')
        self._tasks = {task.task_id: task for task in tasks}
        if len(self._tasks) != len(tasks):
            raise ValueError('duplicate task IDs in training manifest')
        self.identifier, self.digest = self._scope.identifier, self._scope.digest

    def for_task(self, task: TaskSpec):
        if self._tasks.get(task.task_id) != task:
            raise ValueError('task is not the manifest-bound training task')
        return self._scope.for_task(task.task_id)

    @property
    def tasks(self):
        return list(self._tasks.values())


def load_scoped_train_manifest(path: str | Path):
    """Load the sealed train split and its split-scoped fixture environment."""
    scope = VerifiedTrainFixtures(path)
    return scope, scope.tasks
