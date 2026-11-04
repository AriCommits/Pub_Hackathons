#!/usr/bin/env python3
"""Score one copy of every order-prompt format. Does not touch the 104-task split."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from actproto.data.order_curriculum import ALL_FORMATS, order_tasks  # noqa: E402
from actproto.evaluation.live_runner import run_live_trial  # noqa: E402
from scripts._cli import add_trial_argument, trial_from_args  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate every prompt format as B2fmteval.")
    add_trial_argument(parser)
    args = parser.parse_args(argv)
    trial = trial_from_args(args)
    payload = run_live_trial("B2fmteval", trial, order_tasks(1, formats=ALL_FORMATS))
    print(payload.get("summary") or payload.get("n_tasks"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
