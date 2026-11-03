#!/usr/bin/env python3
"""Run B0, B1, or B2 live eval using paths and run matrix from trial.yaml."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._cli import add_trial_argument, trial_from_args  # noqa: E402

from actproto.evaluation.live_runner import run_live_trial  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        required=True,
        choices=["B0", "B1", "B2", "all"],
        help="Which baseline(s) to execute with Tiny Aya (+ adapter/router per trial.yaml).",
    )
    add_trial_argument(parser)
    args = parser.parse_args(argv)
    trial = trial_from_args(args)
    run_ids = ["B0", "B1", "B2"] if args.run == "all" else [args.run]
    summaries = []
    for run_id in run_ids:
        payload = run_live_trial(run_id, trial)
        summaries.append(
            {
                "run": run_id,
                "n": payload["n_tasks"],
                "summary": payload["summary"],
                "out": str(trial.report_dir(run_id)),
            }
        )
    print(json.dumps({"runs": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
