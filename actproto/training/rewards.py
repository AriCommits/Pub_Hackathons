"""GRPO reward pieces based on verified oracles, not LLM judges."""

from __future__ import annotations

from typing import Any


def reward(
    *,
    oracle_success: bool,
    invalid_calls: int = 0,
    repeats: int = 0,
    success_weight: float = 1.0,
    invalid_penalty: float = 0.5,
    repeat_penalty: float = 0.25,
) -> float:
    r = success_weight if oracle_success else 0.0
    r -= invalid_penalty * invalid_calls
    r -= repeat_penalty * repeats
    return float(r)


def reward_from_metrics(metrics: dict[str, Any]) -> float:
    validity = float(metrics.get("tool_call_validity") or 0.0)
    steps = int(metrics.get("step_count") or 0)
    invalid = int(round((1.0 - validity) * steps))
    return reward(
        oracle_success=bool(metrics.get("task_success")),
        invalid_calls=invalid,
        repeats=int(metrics.get("repeated_calls") or 0),
    )
