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
from actproto.training.lora_config import LoraSftConfig  # noqa: E402
from actproto.training.sft import build_train_payload, try_agenttune_sft  # noqa: E402
from scripts._cli import add_trial_argument, trial_from_args  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ordinary LoRA SFT (PEFT/TRL). Paths from trial.yaml.")
    add_trial_argument(parser)
    parser.add_argument("--sft-config", default=None, help="Optional override for configs/sft.yaml")
    parser.add_argument("--plan-only", action="store_true", help="Show training payload paths without loading weights.")
    args = parser.parse_args(argv)
    trial = trial_from_args(args)
    cfg = LoraSftConfig.load(args.sft_config)
    cfg.output_dir = trial.paths.adapter_b1
    tasks = [t for t in generate_tasks() if t.split == "train"]
    payload = build_train_payload(tasks, cfg)
    payload["agenttune"] = try_agenttune_sft(payload.get("rows_preview") or [], trainer_factory=None)
    if args.plan_only:
        print(json.dumps({k: payload[k] for k in ("n_rows", "output_dir", "peft", "path", "agenttune")}, indent=2))
        return 0
    print("GPU training entrypoint: wire TRL/PEFT trainer here using payload['path'] and cfg.output_dir.")
    print(json.dumps({"output_dir": cfg.output_dir, "n_rows": payload["n_rows"]}, indent=2))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
