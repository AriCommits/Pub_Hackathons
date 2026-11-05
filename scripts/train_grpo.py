#!/usr/bin/env python3
"""Run a fail-closed GRPO lifecycle. A missing mode, dataset, or baseline is an error."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from actproto.training.grpo_run import run_collect_only, run_plan, run_resume, run_smoke, run_train  # noqa: E402
from actproto.training.rl_config import GrpoConfigError, load_grpo_config, resolve_output  # noqa: E402

CONFIG = ROOT / "actproto" / "configs" / "grpo_v2.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--collect-only", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--resume", type=Path)
    parser.add_argument("--output", help="relative directory inside results/rl_v2")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args(argv)
    try:
        config = load_grpo_config(args.config)
        if args.plan_only:
            payload = run_plan(config, ROOT)
        elif args.collect_only:
            payload = run_collect_only(config, ROOT)
        elif args.train:
            payload = run_train(config, ROOT)
        else:
            relative = args.output or config.output_dir
            output = resolve_output(relative, ROOT)
            steps = args.max_steps or config.max_steps
            if args.smoke:
                payload = run_smoke(config, ROOT, output, max_steps=steps)
            else:
                payload = run_resume(config, ROOT, output, args.resume, max_steps=steps)
    except (GrpoConfigError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
