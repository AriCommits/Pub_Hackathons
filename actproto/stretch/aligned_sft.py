"""Wrap ordinary SFT with a training-only non-detached hook. Do not edit sft.py."""

from __future__ import annotations

from typing import Any

from actproto.stretch.alignment_loss import total_loss
from actproto.training.lora_config import LoraSftConfig
from actproto.training.sft import build_train_payload


def build_aligned_payload(tasks, config: LoraSftConfig, lam: float = 0.1) -> dict[str, Any]:
    payload = build_train_payload(tasks, config)
    payload["alignment"] = {
        "lambda": lam,
        "detach": False,
        "loss": "contrastive_prototypes",
        "note": "register ActivationHook(..., detach=False) during training only",
    }
    payload["output_dir"] = "results/adapters/b4"
    return payload


def combine_losses(sft_loss: float, align_loss: float, lam: float) -> float:
    return total_loss(sft_loss, align_loss, lam)
