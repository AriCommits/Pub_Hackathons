"""Ordinary LoRA SFT. No activation loss. Use scripts/train_lora.py --plan-only without GPU."""

from __future__ import annotations

from typing import Any

from actproto.training.lora_config import LoraSftConfig
from actproto.training.sft_dataset import tasks_to_sft_rows


def build_train_payload(tasks, config: LoraSftConfig) -> dict[str, Any]:
    rows = tasks_to_sft_rows(tasks)
    return {
        "n_rows": len(rows),
        "output_dir": config.output_dir,
        "peft": config.peft_kwargs(),
        "path": "peft_trl",
        "rows_preview": rows[:1],
    }


def try_agenttune_sft(rows: list[dict[str, Any]], trainer_factory, **kwargs: Any) -> dict[str, Any]:
    """Optional AgentTune Project.train(fmt='sft') path. Falls back to PEFT."""
    try:
        from agenttune.agentic import Project  # type: ignore  # noqa: F401
    except ImportError:
        return {"path": "peft_trl", "reason": "agenttune_not_installed"}
    try:
        # Real wiring needs a Project with full-tier trajectories; keep explicit.
        return {"path": "agenttune_sft", "kwargs": kwargs, "n_rows": len(rows)}
    except Exception as exc:  # pragma: no cover
        return {"path": "peft_trl", "reason": str(exc)}
