"""Construct execution events at the tool-harness trust boundary.

The evaluator must receive these records directly, never from model messages. The
source label is descriptive, not a cryptographic signature: accepting externally
supplied JSON as a trusted trace defeats this boundary.
"""
from copy import deepcopy
from typing import Any

from actproto.experiments.contracts import ExecutionTrace


def execute_event(harness: Any, tool: str, arguments: dict) -> dict:
    """Execute once and record its real result and pre/post harness state."""
    before = deepcopy(harness.final_state)
    result = harness.execute(tool, deepcopy(arguments))
    return {"kind": "tool_call", "source": "harness", "tool": result.name,
            "arguments": deepcopy(result.arguments), "output": deepcopy(result.output),
            "state_before": before, "state_after": deepcopy(harness.final_state)}


def build_trace(task_id: str, events: list[dict], completion_reason: str,
                *, model_spans=None, generated_tokens=0, model_invocations=0) -> ExecutionTrace:
    """Build a trace from locally recorded events, not model-provided observations."""
    recorded = deepcopy(events)
    state = deepcopy(recorded[-1].get("state_after", {})) if recorded else {}
    return ExecutionTrace(task_id, recorded, completion_reason, "harness",
                          model_spans=deepcopy(model_spans or []),
                          generated_tokens=generated_tokens, model_invocations=model_invocations,
                          final_answer=state.get("answer"), final_state=state)
