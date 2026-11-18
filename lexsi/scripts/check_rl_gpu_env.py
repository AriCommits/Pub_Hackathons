"""Print the CUDA RL interpreter's devices and require the RTX 5060 Ti by UUID.

This script loads no model weights. It is not a training run.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from actproto.training.rl_device import DeviceRecord, DeviceSelectionError, normalize_uuid, select_5060_ti  # noqa: E402

REQUIRED_TORCH = "2.11.0+cu128"


def main() -> int:
    parent_mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if parent_mask:
        print("CUDA_VISIBLE_DEVICES is already set; refusing to continue", file=sys.stderr)
        return 2
    import torch

    print(f"torch {torch.__version__}")
    print(f"cuda_available {torch.cuda.is_available()}")
    records: list[DeviceRecord] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        uuid = normalize_uuid(props.uuid)
        print(f"device {index} name={props.name} uuid={uuid} memory={props.total_memory}")
        records.append(DeviceRecord(index, props.name, uuid, int(props.total_memory)))
    if torch.__version__ != REQUIRED_TORCH or not torch.cuda.is_available():
        print(f"expected {REQUIRED_TORCH} with CUDA available", file=sys.stderr)
        return 1
    try:
        chosen = select_5060_ti(records)
    except DeviceSelectionError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"selected {chosen} uuid={records[chosen].uuid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
