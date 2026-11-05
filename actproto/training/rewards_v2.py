"""Execution-grounded, versioned rewards for the v2 tool-use protocol.

Only a trusted in-process caller may create a session and supply its harness.
The session records executions itself; it never accepts model-written traces or
serialized rollout JSON. Python objects are not a security sandbox, so trainer
integration must keep this API on the trusted side of the policy boundary.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from typing import Any

from actproto.evaluation.trace_adapter import execute_event
from actproto.evaluation.verification_v2 import _call_match, _match_dependencies, verify
from actproto.experiments.contracts import ExecutionTrace, TaskSpec
from actproto.tools.schemas import validate_args

SPARSE_VERSION = 'verified_success_v2'
SHAPED_VERSION = 'bounded_shaped_v1'


@dataclass(frozen=True)
class RewardLedger:
    task_id: str
    version: str
    reward: float
    verified_success: bool
    answer_success: bool
    execution_compliance: bool
    invalid_attempts: int
    duplicate_calls: int
    excess_calls: int | None
    excess_omitted_reason: str | None
    reasons: tuple[str, ...]


class TrustedRewardSession:
    """Trusted runtime owns the harness and supplies only calls to execute.

    Do not initialize a session from a model, external JSON, or a rehydrated
    trace. `harness` must be a fresh locally controlled harness; provenance
    labels are not an authentication mechanism.
    """
    def __init__(self, task: TaskSpec, harness: Any):
        if not isinstance(task, TaskSpec) or not callable(getattr(harness, 'execute', None)):
            raise TypeError('a TaskSpec and locally controlled harness are required')
        if getattr(harness, 'final_state', None) != {}:
            raise ValueError('harness must have a fresh empty state')
        self._task = deepcopy(task)
        self._harness = harness
        self._events: list[dict[str, Any]] = []
        self._expected_state: dict[str, Any] = {}
        self._tampered = False
        self._finished = False

    def _state_tampered(self) -> bool:
        if self._harness.final_state != self._expected_state:
            if not self._tampered:
                self._events.append({'kind': 'state_tampering'})
                self._tampered = True
            return True
        return self._tampered

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._finished:
            raise ValueError('session is finalized')
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise TypeError('tool name and arguments must be a string and object')
        if self._state_tampered():
            raise ValueError('harness state tampering detected')
        event = execute_event(self._harness, name, arguments)
        self._events.append(event)
        self._expected_state = deepcopy(event['state_after'])
        return deepcopy(event['output'])

    def captured_tool_events(self) -> list[dict[str, Any]]:
        """Return copies of tool events captured by this live session."""
        return deepcopy([event for event in self._events if event.get('kind') == 'tool_call'])

    def invalid_attempt(self, reason: str = 'invalid output') -> None:
        if self._finished:
            raise ValueError('session is finalized')
        self._events.append({'kind': 'parse_error', 'reason': str(reason)})

    def score(self, *, completion_reason: str, version: str = SPARSE_VERSION) -> RewardLedger:
        if self._finished:
            raise ValueError('session is finalized')
        if version not in (SPARSE_VERSION, SHAPED_VERSION):
            raise ValueError('unsupported reward version')
        self._finished = True
        self._state_tampered()
        state = deepcopy(self._harness.final_state)
        if not isinstance(state, dict):
            state = {}  # tampering is recorded; verifier must still fail closed
        trace = ExecutionTrace(self._task.task_id, deepcopy(self._events), completion_reason, 'harness',
                               final_answer=state.get('answer'), final_state=state)
        verdict = verify(self._task, trace)
        calls = [event for event in trace.events if event.get('kind') == 'tool_call']
        dependencies = self._task.tool_dependencies if self._task.mode == 'required_tools' else []
        matched = _match_dependencies(calls, dependencies)
        if matched is None:
            # A declared, correctly ordered expected error is recoverable even
            # when a later required call was never reached. Match each declared
            # dependency once; do not excuse repeated errors indefinitely.
            matched = set()
            completed_dependencies = set()
            for index, event in enumerate(calls):
                for dependency in dependencies:
                    if (dependency['id'] not in completed_dependencies
                            and all(parent in completed_dependencies for parent in dependency.get('after', []))
                            and _call_match(event, dependency)):
                        matched.add(index)
                        completed_dependencies.add(dependency['id'])
                        break
        invalid = sum(event.get('kind') != 'tool_call' for event in trace.events)
        forbidden = set(self._task.oracle.get('forbidden_tools', []))
        seen = set()
        duplicates = 0
        for index, event in enumerate(calls):
            name, arguments, output = event.get('tool'), event.get('arguments'), event.get('output')
            valid, _ = validate_args(name, arguments) if isinstance(name, str) else (False, '')
            if isinstance(arguments, dict) and any(isinstance(v, bool) for v in arguments.values()):
                valid = False
            if (not valid or name in forbidden or not isinstance(output, dict)
                    or type(output.get('ok')) is not bool or (output.get('ok') is False and index not in matched)):
                invalid += 1
            try:
                key = json.dumps([name, arguments, event.get('state_before')], sort_keys=True,
                                 separators=(',', ':'), allow_nan=False)
            except (TypeError, ValueError):
                # Unserializable/non-finite inputs remain invalid and must not
                # crash reward computation or masquerade as an identical loop.
                key = None
            if key is not None:
                if key in seen:
                    duplicates += 1
                seen.add(key)
        bound = self._task.oracle.get('valid_call_bound')
        if bound is None:
            excess, omitted = None, 'task does not declare valid_call_bound'
        elif type(bound) is not int or bound < 0:
            raise ValueError('valid_call_bound must be a nonnegative integer')
        else:
            excess, omitted = max(len(calls) - bound, 0), None
        success = verdict.verified_success_v2
        value = float(success)
        if version == SHAPED_VERSION:
            value -= .10 * min(invalid, 3) + .05 * min(duplicates, 3)
            value -= .05 * int(success) * min(excess or 0, 4)
        return RewardLedger(self._task.task_id, version, value, success, verdict.answer_success,
                            verdict.execution_compliance, invalid, duplicates, excess, omitted,
                            tuple(verdict.reasons))


def score_executed_session(session: TrustedRewardSession, *, completion_reason: str) -> float:
    """Score one live session with the checked-in shaped verifier reward.

    There is no token-length term. Serialized traces are not accepted.
    """
    if not isinstance(session, TrustedRewardSession):
        raise TypeError("a live TrustedRewardSession is required")
    if not isinstance(completion_reason, str) or not completion_reason:
        raise ValueError("completion_reason is required")
    return session.score(completion_reason=completion_reason, version=SHAPED_VERSION).reward


def group_advantages(rewards: list[float]) -> tuple[list[float], dict[str, Any]]:
    """Population-standardized advantages; all-equal groups are exactly zero."""
    if len(rewards) < 2 or any(isinstance(x, bool) or not isinstance(x, (int, float))
                               or not math.isfinite(x) for x in rewards):
        raise ValueError('at least two finite rewards are required')
    values = [float(x) for x in rewards]
    # Scale first to avoid overflow for finite extreme values.
    scale = max(abs(x) for x in values)
    normalized = [x / scale for x in values] if scale else values
    center = math.fsum(normalized) / len(normalized)
    variance = math.fsum((x - center) ** 2 for x in normalized) / len(normalized)
    if variance == 0:
        return [0.0] * len(values), {'zero_variance_groups': 1, 'groups': 1}
    std = math.sqrt(variance)
    return [(x - center) / std for x in normalized], {'zero_variance_groups': 0, 'groups': 1}
