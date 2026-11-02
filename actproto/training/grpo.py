"""Optional AgentTune GRPO wiring. Do not use this to rescue a failed router."""

from __future__ import annotations

from typing import Any

from actproto.training.rewards import reward_from_metrics


def build_grpo_kwargs(model_id: str, dataset: Any, tools: list, output_dir: str = "results/adapters/grpo") -> dict[str, Any]:
    return {
        "algorithm": "grpo",
        "model": model_id,
        "train_dataset": dataset,
        "tools": tools,
        "reward_funcs": ["verified_success", "invalid_call_penalty", "repeat_call_penalty"],
        "output_dir": output_dir,
        "factory": "agenttune.core.backend_factory.create_agentic_trainer",
    }


def try_create_trainer(**kwargs: Any) -> dict[str, Any]:
    try:
        from agenttune.core.backend_factory import create_agentic_trainer  # type: ignore

        return {"ok": True, "trainer_type": type(create_agentic_trainer).__name__, "kwargs": kwargs}
    except ImportError:
        return {"ok": False, "reason": "agenttune_not_installed", "kwargs": kwargs}
