"""Strict execution-grounded verification for tooluse_v2.

Event schema: a tool_call has source=harness, tool, arguments, output,
state_before and state_after; parse_error events have kind=parse_error. Only
records captured by the local harness are trusted: string provenance fields
alone cannot authenticate records submitted by a model or an external caller.

A dependency has a unique id, tool, exact arguments, optional `after` IDs and
optional `expected_error` type. Independent dependencies may occur in any order.
Task oracles specify answer (or answers), optionally final_state, and an optional
list of forbidden_tools. Answer
normalization is explicit: exact, strip, casefold, or numeric with an optional
literal unit and absolute tolerance. No substring or implicit ID normalization.
"""
from __future__ import annotations

import math
import re
from typing import Any

from actproto.experiments.contracts import ExecutionTrace, TaskSpec, VerificationResult
from actproto.tools.schemas import validate_args


def _answer_matches(actual: Any, expected: Any, rules: dict) -> bool:
    kind = rules.get("type", "exact")
    if kind == "exact":
        return type(actual) is type(expected) and actual == expected
    if not isinstance(actual, str) or not isinstance(expected, str):
        return False
    if kind == "strip":
        return actual.strip() == expected.strip()
    if kind == "casefold":
        return actual.strip().casefold() == expected.strip().casefold()
    if kind != "numeric":
        raise ValueError(f"unsupported answer normalization: {kind}")
    unit = rules.get("unit", "")
    number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    pattern = rf"\s*({number})\s*{re.escape(unit)}\s*"
    left, right = re.fullmatch(pattern, actual), re.fullmatch(pattern, expected)
    if not left or not right:
        return False
    a, b = float(left.group(1)), float(right.group(1))
    tolerance = rules.get("tolerance", 0.0)
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("invalid numeric tolerance")
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance


def _call_match(event: dict, dep: dict) -> bool:
    if event.get("tool") != dep.get("tool") or event.get("arguments") != dep.get("arguments"):
        return False
    output = event.get("output", {})
    if not isinstance(output, dict):
        return False
    for field in ("state_before", "state_after", "output"):
        if field in dep and event.get(field) != dep[field]:
            return False
    error = dep.get("expected_error")
    if error is None:
        return output.get("ok") is True
    actual_error = output.get("error")
    return output.get("ok") is False and isinstance(actual_error, dict) and actual_error.get("type") == error


def _match_dependencies(events: list[dict], dependencies: list[dict]) -> set[int] | None:
    """Find a one-to-one matching satisfying the declared partial order."""
    ids = [d.get("id") for d in dependencies]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("dependency IDs must be unique nonempty strings")
    known = set(ids)
    for d in dependencies:
        after = d.get("after", [])
        if not isinstance(after, list) or any(x not in known or x == d["id"] for x in after):
            raise ValueError("invalid dependency reference")
    candidates = [[i for i, event in enumerate(events) if _call_match(event, d)] for d in dependencies]
    def solve(assigned: dict[str, int]) -> dict[str, int] | None:
        if len(assigned) == len(ids):
            return assigned
        remaining = [j for j, d in enumerate(dependencies) if d["id"] not in assigned and all(x in assigned for x in d.get("after", []))]
        if not remaining:
            return None
        j = min(remaining, key=lambda k: len(candidates[k]))
        dep = dependencies[j]
        for index in candidates[j]:
            if index not in assigned.values() and all(assigned[x] < index for x in dep.get("after", [])):
                match = solve({**assigned, dep["id"]: index})
                if match is not None:
                    return match
        return None
    match = solve({})
    return set(match.values()) if match is not None else None


def verify(task: TaskSpec, trace: ExecutionTrace) -> VerificationResult:
    """Verify a trusted local execution trace; never use model-declared observations."""
    reasons: list[str] = []
    events = trace.events
    if trace.task_id != task.task_id or trace.provenance != "harness":
        reasons.append("untrusted or mismatched trace")
    if trace.completion_reason != "finish":
        reasons.append("incomplete termination")
    calls = []
    previous_state = None
    for event in events:
        if event.get("kind") != "tool_call":
            reasons.append("invalid parsing or unknown event")
            continue
        calls.append(event)
        if event.get("source") != "harness":
            reasons.append("untrusted tool observation")
        tool, arguments, output = event.get("tool"), event.get("arguments"), event.get("output")
        valid, _ = validate_args(tool, arguments) if isinstance(tool, str) else (False, "invalid tool")
        # Python bool is an int, but it is not a JSON Schema integer.
        if isinstance(arguments, dict) and any(isinstance(x, bool) for x in arguments.values()):
            valid = False
        if not valid or not isinstance(output, dict) or type(output.get("ok")) is not bool:
            reasons.append("invalid call or tool result")
        before, after = event.get("state_before"), event.get("state_after")
        if not isinstance(before, dict) or not isinstance(after, dict):
            reasons.append("missing state provenance")
        elif previous_state is not None and before != previous_state:
            reasons.append("stale or inconsistent state")
        previous_state = after
    if len(calls) > task.call_budget:
        reasons.append("call budget exceeded")
    finishes = [i for i, event in enumerate(calls) if event.get("tool") in {"finish", "no_tool"}]
    if finishes != [len(calls) - 1] or not calls:
        reasons.append("missing, repeated, or premature finish")
    terminal = calls[-1] if calls and finishes == [len(calls) - 1] else {}
    terminal_args = terminal.get("arguments", {})
    terminal_out = terminal.get("output", {})
    if not isinstance(terminal_args, dict) or not isinstance(terminal_out, dict):
        terminal_args, terminal_out = {}, {}
    answer = terminal_args.get("answer")
    data = terminal_out.get("data", {})
    if (terminal_out.get("ok") is not True or not isinstance(data, dict)
            or data.get("answer") != answer or data.get("done") is not True
            or trace.final_answer != answer or trace.final_state != previous_state
            or trace.final_state.get("answer") != answer or trace.final_state.get("done") is not True):
        reasons.append("inconsistent finish result")
    forbidden_tools = task.oracle.get("forbidden_tools", [])
    if not isinstance(forbidden_tools, list) or any(not isinstance(tool, str) for tool in forbidden_tools):
        raise ValueError("forbidden_tools must be a list of tool names")
    if any(event.get("tool") in forbidden_tools for event in calls):
        reasons.append("forbidden tool called")
    dependencies = task.tool_dependencies if task.mode == "required_tools" else []
    matched = _match_dependencies(calls, dependencies)
    if matched is None:
        reasons.append("required calls or dependencies unsatisfied")
        matched = set()
    for index, event in enumerate(calls):
        output = event.get("output")
        if isinstance(output, dict) and output.get("ok") is False and index not in matched:
            reasons.append("unexpected tool error")
    expected = task.oracle.get("answers", [task.oracle.get("answer")])
    if not isinstance(expected, list):
        raise ValueError("oracle answers must be a list")
    answer_success = any(_answer_matches(answer, value, task.normalization) for value in expected)
    expected_state = task.oracle.get("final_state", {})
    if not isinstance(expected_state, dict):
        raise ValueError("oracle final_state must be an object")
    answer_success = answer_success and all(trace.final_state.get(k) == v for k, v in expected_state.items())
    if not answer_success:
        reasons.append("incorrect outcome")
    compliance = not any(reason != "incorrect outcome" for reason in reasons)
    return VerificationResult(answer_success, compliance, answer_success and compliance, reasons)
