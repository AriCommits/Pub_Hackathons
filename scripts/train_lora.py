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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ordinary LoRA SFT (PEFT/TRL). Use --dry-run without GPU.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-model-load", action="store_true")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    cfg = LoraSftConfig.load(args.config)
    tasks = [t for t in generate_tasks() if t.split == "train"]
    payload = build_train_payload(tasks, cfg)
    payload["agenttune"] = try_agenttune_sft(payload.get("rows_preview") or [], trainer_factory=None)
    if args.dry_run or args.skip_model_load:
        print(json.dumps({k: payload[k] for k in ("n_rows", "output_dir", "peft", "path", "agenttune")}, indent=2))
        return 0
    print("Live training requires Tiny Aya weights and a GPU; refusing without --dry-run.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
