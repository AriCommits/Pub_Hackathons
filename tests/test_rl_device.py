import os
import subprocess
import sys
from pathlib import Path

import pytest

from actproto.training.rl_device import DeviceRecord, DeviceSelectionError, select_5060_ti

TI = "134c34ef-c155-b001-4e3a-b9134b072c21"
TI_NAME = "NVIDIA GeForce RTX 5060 Ti"
CARD_4060 = "5b885e12-623f-268c-f07c-ee33db379969"
NAME_4060 = "NVIDIA GeForce RTX 4060"
SCRIPT = Path("scripts/check_rl_gpu_env.py")


def test_known_two_card_map_returns_the_pytorch_5060_index():
    devices = [
        DeviceRecord(0, NAME_4060, CARD_4060, 8_000_000_000),
        DeviceRecord(1, TI_NAME, TI, 16_000_000_000),
    ]
    assert select_5060_ti(devices) == 1


def test_swapped_index_map_does_not_follow_memory_or_nvidia_smi_order():
    devices = [
        DeviceRecord(0, TI_NAME, f"GPU-{TI}", 8_000_000_000),
        DeviceRecord(1, NAME_4060, CARD_4060, 32_000_000_000),
    ]
    assert select_5060_ti(devices) == 0


def test_4060_only_map_is_rejected():
    devices = [DeviceRecord(0, NAME_4060, CARD_4060, 8_000_000_000)]
    with pytest.raises(DeviceSelectionError, match="missing"):
        select_5060_ti(devices)


def test_matching_uuid_with_the_wrong_name_is_rejected():
    devices = [DeviceRecord(0, NAME_4060, TI, 16_000_000_000)]
    with pytest.raises(DeviceSelectionError, match="name"):
        select_5060_ti(devices)


def test_parent_cuda_visible_devices_exits_nonzero_without_changing_parent():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in (None, "")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    result = subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "CUDA_VISIBLE_DEVICES" in result.stderr
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in (None, "")


def test_nvidia_ordinal_follows_the_uuid_rather_than_pytorch_index():
    from actproto.training.rl_device import ordinal_from_nvidia_rows, parse_nvidia_gpu_csv
    text = "\n".join([
        "0, GPU-5b885e12-623f-268c-f07c-ee33db379969",
        "1, GPU-134c34ef-c155-b001-4e3a-b9134b072c21",
    ])
    rows = parse_nvidia_gpu_csv(text)
    assert ordinal_from_nvidia_rows(rows, "134c34ef-c155-b001-4e3a-b9134b072c21") == "1"
    swapped = parse_nvidia_gpu_csv("0, GPU-134c34ef-c155-b001-4e3a-b9134b072c21\n1, GPU-5b885e12-623f-268c-f07c-ee33db379969\n")
    assert ordinal_from_nvidia_rows(swapped, "134c34ef-c155-b001-4e3a-b9134b072c21") == "0"
