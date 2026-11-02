#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts._baseline import load_tasks, replay_eval  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="B2: LoRA + prototype router.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="results/b0_b2/B2")
    parser.add_argument("--adapter", default="results/adapters/b1")
    parser.add_argument("--prototypes", default="results/prototypes.npz")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    tasks = load_tasks(args.split, args.limit)
    if not args.dry_run:
        print("B2 live infer needs adapter + prototypes. Use --dry-run.")
        return 2
    payload = replay_eval(tasks, args.out)
    print(json.dumps({"run": "B2", "n": len(tasks), "summary": payload["summary"], "prototypes": args.prototypes}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
