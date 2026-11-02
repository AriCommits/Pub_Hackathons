"""Deterministic oracles: no LLM required."""

from __future__ import annotations

from typing import Any

from actproto.data.types import Oracle, Task
from actproto.tools.registry import call_tool, canonical_tool_name


def answers_equal(expected: str | None, actual: str | None, tolerance: float) -> bool:
    if expected is None:
        return True
    if actual is None:
        return False
    e = expected.strip()
    a = actual.strip()
    if e == a:
        return True
    try:
        return abs(float(e) - float(a)) <= tolerance
    except ValueError:
        return e.lower() == a.lower()


def check_final_state(oracle: Oracle, state: dict[str, Any]) -> dict[str, Any]:
    for key, expected in oracle.final_state.items():
        actual = state.get(key)
        if expected != actual:
            if isinstance(expected, float) or isinstance(actual, float):
                try:
                    if abs(float(expected) - float(actual)) <= oracle.numeric_tolerance:
                        continue
                except (TypeError, ValueError):
                    pass
            return {
                "verdict": "failure",
                "reason": f"{key}: expected {expected!r} got {actual!r}",
            }
    if oracle.answer is not None:
        actual_answer = state.get("answer")
        if not answers_equal(oracle.answer, actual_answer if actual_answer is None else str(actual_answer), oracle.numeric_tolerance):
            return {
                "verdict": "failure",
                "reason": f"answer: expected {oracle.answer!r} got {actual_answer!r}",
            }
    return {"verdict": "success", "reason": "final state matches"}


def replay_gold(task: Task) -> dict[str, Any]:
    """Execute gold tools against the real harness; return last finish payload + state."""
    state: dict[str, Any] = {}
    last: dict[str, Any] = {}
    for name, args in zip(task.gold_tools, task.gold_args, strict=True):
        result = call_tool(canonical_tool_name(name), args)
        last = result
        if name in {"finish", "no_tool"} and result.get("ok"):
            state["answer"] = result["data"]["answer"]
            state["done"] = True
        if not result.get("ok") and name == "lookup_record":
            state["lookup_error"] = True
    state.setdefault("done", False)
    return {"last": last, "state": state}


def yoy_growth(values: list[float]) -> float:
    if len(values) < 2:
        raise ValueError("need at least two values")
    changes = [(values[i] - values[i - 1]) / values[i - 1] for i in range(1, len(values))]
    return sum(changes) / len(changes)
