#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from actproto.data.generator import generate_tasks  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract activations (requires a loaded model unless --dry-run).")
    parser.add_argument("--split", default="train")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--out", default="results/activations")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    n = sum(1 for t in generate_tasks() if t.split == args.split)
    print(f"split={args.split} n={n} adapter={args.adapter} out={args.out} dry_run={args.dry_run}")
    if args.dry_run:
        return 0
    print("Live extraction needs Tiny Aya weights; refusing without --dry-run in this environment.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
