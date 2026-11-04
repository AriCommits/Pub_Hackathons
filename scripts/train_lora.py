#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from actproto.data.generator import generate_tasks  # noqa: E402
from actproto.training.lora_config import LoraSftConfig  # noqa: E402
from actproto.configs.schema import load_default  # noqa: E402
from actproto.training.lora_run import run_lora_sft  # noqa: E402
from actproto.training.sft import build_train_payload, try_agenttune_sft  # noqa: E402
from scripts._cli import add_trial_argument, trial_from_args  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ordinary LoRA SFT (PEFT/TRL). Paths from trial.yaml.")
    add_trial_argument(parser)
    parser.add_argument("--sft-config", default=None, help="Optional override for configs/sft.yaml")
    parser.add_argument("--output-dir", default=None, help="Adapter directory. Defaults to trial.yaml adapter_b1.")
    parser.add_argument("--orders", action="store_true", help="Append the order curriculum (4 copies) to the train split.")
    parser.add_argument("--plan-only", action="store_true", help="Show training payload paths without loading weights.")
    args = parser.parse_args(argv)
    trial = trial_from_args(args)
    cfg = LoraSftConfig.load(args.sft_config)
    cfg.output_dir = args.output_dir or trial.paths.adapter_b1
    tasks = [t for t in generate_tasks() if t.split == "train"]
    if args.orders:
        from actproto.data.order_curriculum import order_tasks

        tasks = tasks + order_tasks(4)
    payload = build_train_payload(tasks, cfg)
    payload["agenttune"] = try_agenttune_sft(payload.get("rows_preview") or [], trainer_factory=None)
    if args.plan_only:
        print(json.dumps({k: payload[k] for k in ("n_rows", "output_dir", "peft", "path", "agenttune")}, indent=2))
        return 0
    _pin_largest_gpu()
    experiment = load_default()
    result = run_lora_sft(tasks, cfg, experiment.model_id, experiment.revision)
    print(json.dumps(result, indent=2))
    return 0


def _pin_largest_gpu() -> None:
    """Hide every card except the one with the most memory. Must run before torch is imported here."""
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        print(f"CUDA_VISIBLE_DEVICES already {os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)
        return
    code = (
        "import torch\n"
        "if not torch.cuda.is_available():\n"
        "    raise SystemExit(0)\n"
        "best = max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.get_device_properties(i).total_memory)\n"
        "print(best)\n"
        "print(torch.cuda.get_device_name(best), file=__import__('sys').stderr)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    index = proc.stdout.strip().splitlines()[-1]
    os.environ["CUDA_VISIBLE_DEVICES"] = index
    print(f"pinned training to CUDA_VISIBLE_DEVICES={index} ({proc.stderr.strip()})", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
