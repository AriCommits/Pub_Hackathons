#!/usr/bin/env python3
"""Generate leakage-aware train/val/test JSONL under data/processed/."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from actproto.data.generator import bucket_counts, generate_tasks, write_splits  # noqa: E402
from actproto.utils.seed import seed_everything  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "data" / "processed"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    seed_everything(args.seed)
    tasks = generate_tasks()
    counts = write_splits(tasks, args.out)
    print(json.dumps({"splits": counts, "buckets": bucket_counts(tasks)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
