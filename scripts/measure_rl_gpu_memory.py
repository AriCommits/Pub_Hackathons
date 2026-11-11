"""Measure a Tiny Aya bf16 load on the RTX 5060 Ti. This does not train."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from actproto.training.rl_config import RTX_4060_UUID, preferred_device_uuid  # noqa: E402
from actproto.training.rl_device import normalize_uuid  # noqa: E402

OUTPUT = ROOT / "results" / "rl_v2" / "gpu_memory" / "tiny_aya_bf16.json"


def build_memory_record(
    *,
    device_name: str,
    device_uuid: str,
    pytorch_index: int,
    torch_version: str,
    weight_dtype: str,
    allocated_bytes: int,
    reserved_bytes: int,
    peak_allocated_bytes: int,
    lora_rank: int,
    lora_alpha: int,
    forward_peak_allocated_bytes: int | None = None,
    forward_error: str | None = None,
) -> dict:
    """Assemble the measurement JSON. A fake snapshot is enough for the schema test."""
    uuid = normalize_uuid(device_uuid)
    if uuid == RTX_4060_UUID or "4060" in device_name:
        raise ValueError("the RTX 4060 cannot host the Tiny Aya memory probe")
    if uuid != preferred_device_uuid() or "RTX 5060 Ti" not in device_name:
        raise ValueError("memory probe requires the RTX 5060 Ti")
    if min(allocated_bytes, reserved_bytes, peak_allocated_bytes) < 0:
        raise ValueError("memory byte counts must be non-negative")
    return {
        "device_name": device_name,
        "device_uuid": uuid,
        "pytorch_index": pytorch_index,
        "torch_version": torch_version,
        "weight_dtype": weight_dtype,
        "allocated_bytes": allocated_bytes,
        "reserved_bytes": reserved_bytes,
        "peak_allocated_bytes": peak_allocated_bytes,
        "forward_peak_allocated_bytes": forward_peak_allocated_bytes,
        "forward_error": forward_error,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_target_modules": None,
        "adapter_loaded": False,
        "optimizer_started": False,
        "grpo_fit_claimed": False,
    }


def main(argv: list[str] | None = None) -> int:
    del argv
    if os.environ.get("LEXSI_RL_MEMORY_CHILD") == "1":
        return _measure()
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        print("CUDA_VISIBLE_DEVICES is already set; refusing to continue", file=sys.stderr)
        return 2
    from scripts.train_grpo import assert_5060_idle, visible_ordinal_for_5060

    try:
        assert_5060_idle()
        ordinal = visible_ordinal_for_5060()
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ordinal
    env["LEXSI_RL_MEMORY_CHILD"] = "1"
    return subprocess.call([sys.executable, str(Path(__file__).resolve())], env=env)


def _measure() -> int:
    import torch
    import yaml
    from transformers import AutoModelForCausalLM

    from actproto.training.rl_device import DeviceRecord, select_5060_ti

    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        print("masked child is missing CUDA_VISIBLE_DEVICES", file=sys.stderr)
        return 2
    records = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        records.append(DeviceRecord(index, props.name, normalize_uuid(props.uuid), int(props.total_memory)))
    try:
        chosen_index = select_5060_ti(records)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    chosen = next(record for record in records if record.index == chosen_index)
    device = torch.device("cuda", chosen.index)
    torch.cuda.set_device(device)
    config = yaml.safe_load((ROOT / "actproto" / "configs" / "sft_v2.yaml").read_text(encoding="utf-8"))
    torch.cuda.reset_peak_memory_stats(device)
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        revision=config["revision"],
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    model = model.to(device)
    model.eval()
    forward_peak = None
    forward_error = None
    try:
        token = torch.ones((1, 1), dtype=torch.long, device=device)
        with torch.no_grad():
            model(input_ids=token)
        forward_peak = int(torch.cuda.max_memory_allocated(device))
    except RuntimeError as exc:
        forward_error = f"{type(exc).__name__}: {exc}"
    record = build_memory_record(
        device_name=chosen.name,
        device_uuid=chosen.uuid,
        pytorch_index=chosen.index,
        torch_version=torch.__version__,
        weight_dtype="bfloat16",
        allocated_bytes=int(torch.cuda.memory_allocated(device)),
        reserved_bytes=int(torch.cuda.memory_reserved(device)),
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
        lora_rank=int(config["rank"]),
        lora_alpha=int(config["alpha"]),
        forward_peak_allocated_bytes=forward_peak,
        forward_error=forward_error,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
