#!/usr/bin/env python3
"""Run a fail-closed GRPO lifecycle. A missing mode, dataset, or baseline is an error."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from actproto.training.grpo_run import run_collect_only, run_gpu_smoke, run_plan, run_resume, run_smoke, run_train  # noqa: E402
from actproto.training.rl_config import GRPO_GPU_PROFILE, RTX_5060_TI_UUID, GrpoConfigError, load_grpo_config, resolve_output  # noqa: E402
from actproto.training.rl_device import normalize_uuid, ordinal_from_nvidia_rows, parse_nvidia_gpu_csv  # noqa: E402

CONFIG = ROOT / "actproto" / "configs" / "grpo_v2.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--collect-only", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--resume", type=Path)
    parser.add_argument("--output", help="relative directory inside results/rl_v2")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--live-gpu", action="store_true", help="run the gated CUDA fixture smoke")
    args = parser.parse_args(argv)
    try:
        config = load_grpo_config(args.config)
        if args.plan_only:
            payload = run_plan(config, ROOT)
        elif args.collect_only:
            payload = run_collect_only(config, ROOT)
        elif args.train:
            payload = run_train(config, ROOT)
        elif config.schema_version == GRPO_GPU_PROFILE and args.resume:
            raise GrpoConfigError("GPU smoke resume is closed")
        elif args.smoke and config.schema_version == GRPO_GPU_PROFILE:
            if not args.live_gpu:
                raise GrpoConfigError("live CUDA smoke is gated; pass --live-gpu")
            if os.environ.get("LEXSI_RL_GPU_CHILD") == "1":
                relative = args.output or config.output_dir
                output = resolve_output(relative, ROOT)
                steps = args.max_steps or config.max_steps
                payload = run_gpu_smoke(config, ROOT, output, max_steps=steps)
            else:
                forwarded = list(argv) if argv is not None else sys.argv[1:]
                return launch_gpu_smoke(forwarded)
        else:
            if args.live_gpu:
                raise GrpoConfigError("--live-gpu applies only to the grpo_v2_gpu profile")
            relative = args.output or config.output_dir
            output = resolve_output(relative, ROOT)
            steps = args.max_steps or config.max_steps
            if args.smoke:
                payload = run_smoke(config, ROOT, output, max_steps=steps)
            else:
                payload = run_resume(config, ROOT, output, args.resume, max_steps=steps)
    except (GrpoConfigError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def launch_gpu_smoke(argv: list[str]) -> int:
    """Start the smoke in a child whose only visible GPU is the 5060 Ti."""
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise GrpoConfigError("CUDA_VISIBLE_DEVICES is already set")
    assert_5060_idle()
    ordinal = visible_ordinal_for_5060()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ordinal
    env["LEXSI_RL_GPU_CHILD"] = "1"
    return subprocess.call([sys.executable, str(Path(__file__).resolve()), *argv], env=env)


def visible_ordinal_for_5060() -> str:
    """Find the mask value that still shows the 5060 Ti UUID after torch reloads."""
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    )
    rows = parse_nvidia_gpu_csv(completed.stdout)
    preferred = ordinal_from_nvidia_rows(rows, RTX_5060_TI_UUID)
    candidates = [preferred] + [index for index, _raw in rows if index != preferred]
    for ordinal in candidates:
        if _mask_keeps_5060(ordinal):
            return ordinal
    raise GrpoConfigError("no device mask kept the RTX 5060 Ti UUID")


def _mask_keeps_5060(ordinal: str) -> bool:
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["CUDA_VISIBLE_DEVICES"] = ordinal
    probe = (
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "import torch\n"
        "from actproto.training.rl_device import DeviceRecord, DeviceSelectionError, normalize_uuid, select_5060_ti\n"
        "records=[]\n"
        "for index in range(torch.cuda.device_count()):\n"
        "    props=torch.cuda.get_device_properties(index)\n"
        "    records.append(DeviceRecord(index, props.name, normalize_uuid(props.uuid), int(props.total_memory)))\n"
        "try:\n"
        "    select_5060_ti(records)\n"
        "except DeviceSelectionError:\n"
        "    sys.exit(1)\n"
        "sys.exit(0)\n"
    )
    completed = subprocess.run([sys.executable, "-c", probe], env=env, check=False)
    return completed.returncode == 0


def assert_5060_idle() -> None:
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    )
    for line in gpu.stdout.splitlines():
        if "," not in line:
            continue
        raw_uuid, used = [part.strip() for part in line.split(",", 1)]
        if normalize_uuid(raw_uuid) != RTX_5060_TI_UUID:
            continue
        number = used.split()[0]
        if number.isdigit() and int(number) > 4096:
            raise GrpoConfigError("RTX 5060 Ti is busy; refusing to start another GPU job")
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,process_name", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    )
    for line in apps.stdout.splitlines():
        if "," not in line:
            continue
        raw_uuid, name = [part.strip() for part in line.split(",", 1)]
        if normalize_uuid(raw_uuid) == RTX_5060_TI_UUID and "python" in name.lower():
            raise GrpoConfigError("RTX 5060 Ti already has a Python compute process")


if __name__ == "__main__":
    raise SystemExit(main())
