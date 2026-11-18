#!/usr/bin/env python3
"""Score held-out tool combinations on the paired adapter. Does not touch the 104-task split."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from actproto.data.combinations import combination_tasks  # noqa: E402
from actproto.evaluation.live_runner import run_live_trial  # noqa: E402
from scripts._cli import add_trial_argument, trial_from_args  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate generated tool-combination tasks as B2combo.")
    add_trial_argument(parser)
    args = parser.parse_args(argv)
    trial = trial_from_args(args)
    tasks = combination_tasks()
    payload = run_live_trial("B2combo", trial, tasks)
    print(payload.get("summary") or payload.get("n_tasks"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
