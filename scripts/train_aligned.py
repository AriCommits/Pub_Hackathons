#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from actproto.data.generator import generate_tasks  # noqa: E402
from actproto.training.aligned_sft import build_aligned_payload  # noqa: E402
from actproto.training.lora_config import LoraSftConfig  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="B4 alignment-loss LoRA (stretch).")
    parser.add_argument("--plan-only", action="store_true", help="Print alignment payload without training.")
    parser.add_argument("--lambda", dest="lam", type=float, default=0.1)
    args = parser.parse_args(argv)
    tasks = [t for t in generate_tasks() if t.split == "train"][:8]
    payload = build_aligned_payload(tasks, LoraSftConfig.load(), lam=args.lam)
    print(json.dumps({k: payload[k] for k in ("n_rows", "output_dir", "alignment", "path")}, indent=2))
    return 0 if args.plan_only or True else 2


if __name__ == "__main__":
    raise SystemExit(main())
