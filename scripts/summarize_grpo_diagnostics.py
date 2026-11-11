#!/usr/bin/env python3
"""CPU helper: aggregate completion_reason counts from step_diagnostics.jsonl.

Use after a lab or Colab GRPO train so a constant -0.1 can be attributed to
parse_error vs truncation vs template_mismatch without re-running the model.

  python scripts/summarize_grpo_diagnostics.py results/rl_v2/grpo_gpu_study/step_diagnostics.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from actproto.training.grpo_run import summarize_step_diagnostics  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="step_diagnostics.jsonl from a GRPO run")
    parser.add_argument("--output", type=Path, help="optional JSON summary path")
    args = parser.parse_args(argv)
    if not args.path.is_file():
        raise SystemExit(f"missing diagnostics file: {args.path}")
    payload = summarize_step_diagnostics(args.path)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        payload["wrote"] = str(args.output)
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
