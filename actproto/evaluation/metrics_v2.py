"""Operational metrics over a verified tooluse_v2 episode."""
from __future__ import annotations

import json
from collections import Counter
from typing import Any

from actproto.experiments.contracts import ExecutionTrace, TaskSpec, VerificationResult
from actproto.evaluation.verification_v2 import _call_match
from actproto.tools.schemas import validate_args


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def episode_metrics(task: TaskSpec, trace: ExecutionTrace, result: VerificationResult,
                    *, minimum_calls: int | None = None) -> dict:
    """Count repeated calls with state, distinct cycles, and explicit failure costs.

    `minimum_calls`, if supplied, must be a sound task-specific lower bound,
    including finish. Omit when unknown; an arbitrary reference length is not a
    sound lower bound when alternative valid trajectories exist.
    """
    if minimum_calls is not None and (isinstance(minimum_calls, bool) or not isinstance(minimum_calls, int) or minimum_calls < 0):
        raise ValueError("minimum_calls must be a nonnegative integer")
    calls = [e for e in trace.events if e.get("kind") == "tool_call"]
    signatures = [_canonical([e.get("tool"), e.get("arguments"), e.get("state_before")]) for e in calls]
    counts = Counter(signatures)
    exact_duplicates = sum(n - 1 for n in counts.values() if n > 1)
    # Count the repetitions of completed adjacent periods, excluding period 1.
    cycles = 0
    i = 0
    while i < len(signatures):
        period = next((p for p in range(2, (len(signatures) - i) // 2 + 1)
                       if signatures[i:i+p] == signatures[i+p:i+2*p]), None)
        if period is None:
            i += 1
        else:
            cycles += 1
            i += period * 2
    dependencies = task.tool_dependencies
    reference_first = dependencies[0] if dependencies else None
    first = calls[0] if calls else None
    first_tool = None if reference_first is None else bool(first and first.get("tool") == reference_first.get("tool"))
    first_argument = None if reference_first is None else bool(first and first.get("tool") == reference_first.get("tool") and first.get("arguments") == reference_first.get("arguments"))
    # This is a dependency coverage metric, independent of first-tool accuracy.
    matched = sum(any(_call_match(call, dep) for call in calls) for dep in dependencies)
    failures = sum(isinstance(e.get("output"), dict) and e["output"].get("ok") is False for e in calls)
    def invalid_call(event):
        tool = event.get("tool")
        args = event.get("arguments")
        valid = isinstance(tool, str) and validate_args(tool, args)[0]
        output = event.get("output")
        expected_error = any(d.get("expected_error") is not None and _call_match(event, d) for d in dependencies)
        return (not valid or event.get("source") != "harness" or not isinstance(output, dict)
                or type(output.get("ok")) is not bool
                or (output.get("ok") is False and not expected_error))
    return {"answer_success": result.answer_success, "execution_compliance": result.execution_compliance,
            "verified_success_v2": result.verified_success_v2, "tool_calls": len(calls),
            "invalid_parsing": sum(e.get("kind") == "parse_error" for e in trace.events),
            "context_overflows": sum(e.get("kind") == "context_overflow" for e in trace.events),
            "truncations": sum(e.get("kind") == "truncation" for e in trace.events),
            "failed_execution": failures,
            "noncompliant_calls": sum(invalid_call(e) for e in calls),
            "model_invocations": trace.model_invocations, "generated_tokens": trace.generated_tokens,
            "exact_duplicate_calls": exact_duplicates, "cycles": cycles,
            "excess_calls": max(0, len(calls) - minimum_calls) if minimum_calls is not None else None,
            "first_tool_accuracy": first_tool, "first_argument_correctness": first_argument,
            "dependency_argument_coverage": matched / len(dependencies) if dependencies else None}
