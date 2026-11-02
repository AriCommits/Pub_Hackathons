#!/usr/bin/env python3
"""Run recorded-mode AuditKIT-style eval on JSONL tasks + optional trajectory dump."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from actproto.data.generator import generate_tasks  # noqa: E402
from actproto.data.types import Task  # noqa: E402
from actproto.evaluation.cases import task_to_case  # noqa: E402
from actproto.evaluation.episodes import Episode  # noqa: E402
from actproto.evaluation.report import write_report  # noqa: E402
from actproto.evaluation.runner import AgentEvalRunner, AgentEvalSpec  # noqa: E402


def _load_tasks(split: str, path: str | None) -> list[Task]:
    if path:
        rows = []
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(Task.from_dict(json.loads(line)))
        return rows
    return [t for t in generate_tasks() if t.split == split]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test")
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--trajectories", default=None, help="JSON list of episodes")
    parser.add_argument("--out", default="results/eval")
    args = parser.parse_args(argv)
    tasks = _load_tasks(args.split, args.tasks)
    cases = [task_to_case(t) for t in tasks]
    episodes: list[Episode] = []
    if args.trajectories:
        raw = json.loads(Path(args.trajectories).read_text(encoding="utf-8"))
        for item in raw:
            episodes.append(Episode(**item) if isinstance(item, dict) else item)
    spec = AgentEvalSpec(cases=cases, mode="recorded", episodes=episodes, trials=1)
    result = AgentEvalRunner().run(spec)
    payload = write_report(result, args.out)
    print(payload["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
