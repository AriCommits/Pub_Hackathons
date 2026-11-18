#!/usr/bin/env python3
"""Archived diagnostic train-echo: one copy of each curriculum order (not held-out)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from actproto.data.order_curriculum import order_tasks  # noqa: E402
from actproto.evaluation.live_runner import run_live_trial  # noqa: E402
from scripts._cli import add_trial_argument, trial_from_args  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Diagnostic train-echo eval of order_tasks(1) as B2ordereval (not held-out)."
    )
    add_trial_argument(parser)
    args = parser.parse_args(argv)
    trial = trial_from_args(args)
    print("NOTE: B2ordereval is diagnostic_train_echo; prompts overlap the order curriculum.", flush=True)
    payload = run_live_trial("B2ordereval", trial, order_tasks(1))
    print(payload.get("summary") or payload.get("n_tasks"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
