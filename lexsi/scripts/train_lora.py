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
from actproto.configs.schema import load_default  # noqa: E402
from actproto.training.lora_run import run_lora_sft  # noqa: E402
from actproto.training.sft import build_train_payload, try_agenttune_sft  # noqa: E402
from actproto.utils.devices import pin_5060_ti_visible  # noqa: E402
from scripts._cli import add_trial_argument, trial_from_args  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ordinary LoRA SFT (PEFT/TRL). Paths from trial.yaml.")
    add_trial_argument(parser)
    parser.add_argument("--sft-config", default=None, help="Optional override for configs/sft.yaml")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Adapter directory. Defaults to sft.yaml output_dir (not the historical b1 path).",
    )
    parser.add_argument("--orders", action="store_true", help="Append the sequence-format order curriculum (4 copies) to the train split.")
    parser.add_argument("--formats", action="store_true", help="Append every prompt format (2 copies) to the train split.")
    parser.add_argument("--plan-only", action="store_true", help="Show training payload paths without loading weights.")
    args = parser.parse_args(argv)
    trial = trial_from_args(args)
    cfg = LoraSftConfig.load(args.sft_config)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    # Refuse to overwrite the preserved historical B1 adapter by accident.
    if Path(cfg.output_dir).name == "b1" and Path(cfg.output_dir).as_posix().endswith("adapters/b1"):
        print(
            "refusing to train into results/adapters/b1; pass --output-dir for a new run "
            "(historical B1 stays at trial.paths.adapter_b1)",
            file=sys.stderr,
        )
        return 1
    tasks = [t for t in generate_tasks() if t.split == "train"]
    if args.formats:
        from actproto.data.order_curriculum import ALL_FORMATS, order_tasks

        tasks = tasks + order_tasks(2, formats=ALL_FORMATS)
    elif args.orders:
        from actproto.data.order_curriculum import order_tasks

        tasks = tasks + order_tasks(4)
    payload = build_train_payload(tasks, cfg)
    payload["agenttune"] = try_agenttune_sft(payload.get("rows_preview") or [], trainer_factory=None)
    if args.plan_only:
        print(json.dumps({k: payload[k] for k in ("n_rows", "output_dir", "peft", "path", "agenttune")}, indent=2))
        return 0
    from actproto.training.rl_device import DeviceSelectionError

    try:
        ordinal = pin_5060_ti_visible()
    except DeviceSelectionError as exc:
        raise SystemExit(str(exc)) from exc
    if ordinal is not None:
        print(f"pinned training to CUDA_VISIBLE_DEVICES={ordinal}", flush=True)
    experiment = load_default()
    result = run_lora_sft(tasks, cfg, experiment.model_id, experiment.revision)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
