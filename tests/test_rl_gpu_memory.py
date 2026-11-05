import os

import pytest

from scripts.measure_rl_gpu_memory import build_memory_record, main

TI = "134c34ef-c155-b001-4e3a-b9134b072c21"
CARD_4060 = "5b885e12-623f-268c-f07c-ee33db379969"


def test_parent_cuda_visible_devices_refuses_before_load(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert main([]) == 2
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_memory_record_uses_the_fake_snapshot():
    record = build_memory_record(
        device_name="NVIDIA GeForce RTX 5060 Ti",
        device_uuid=TI,
        pytorch_index=1,
        torch_version="2.11.0+cu128",
        weight_dtype="bfloat16",
        allocated_bytes=100,
        reserved_bytes=200,
        peak_allocated_bytes=300,
        lora_rank=16,
        lora_alpha=32,
        forward_peak_allocated_bytes=400,
    )
    assert record["device_uuid"] == TI
    assert record["pytorch_index"] == 1
    assert record["allocated_bytes"] == 100
    assert record["reserved_bytes"] == 200
    assert record["peak_allocated_bytes"] == 300
    assert record["forward_peak_allocated_bytes"] == 400
    assert record["weight_dtype"] == "bfloat16"
    assert record["lora_rank"] == 16
    assert record["lora_alpha"] == 32
    assert record["lora_target_modules"] is None
    assert record["adapter_loaded"] is False
    assert record["optimizer_started"] is False
    assert record["grpo_fit_claimed"] is False


def test_4060_snapshot_is_rejected():
    with pytest.raises(ValueError, match="4060"):
        build_memory_record(
            device_name="NVIDIA GeForce RTX 4060",
            device_uuid=CARD_4060,
            pytorch_index=0,
            torch_version="2.11.0+cu128",
            weight_dtype="bfloat16",
            allocated_bytes=1,
            reserved_bytes=1,
            peak_allocated_bytes=1,
            lora_rank=16,
            lora_alpha=32,
        )
