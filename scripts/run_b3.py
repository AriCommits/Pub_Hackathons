#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from actproto.steering.steer import contrast_direction  # noqa: E402
from actproto.steering.sweep import sweep_alphas  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="B3 activation steering sweep (stretch).")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args(argv)
    hidden = np.zeros(8, dtype=np.float32)
    target = np.ones(8, dtype=np.float32)
    mean = np.zeros(8, dtype=np.float32)
    direction = contrast_direction(target, mean)
    prototypes = {"calculator": target, "finish": mean}
    rows = sweep_alphas(hidden, direction, prototypes)
    print(json.dumps({"limit": args.limit, "rows": [{k: v for k, v in r.items() if k != "scores"} | {"scores": r["scores"]} for r in rows]}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
