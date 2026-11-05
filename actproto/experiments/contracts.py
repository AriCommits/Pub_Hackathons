"""Versioned private task, public policy, and trusted trace contracts."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

VERSION = "tooluse_v2"


def _version(value: str) -> None:
    if value != VERSION:
        raise ValueError(f"unsupported contract version: {value!r}")


class Contract:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]):
        _version(value.get("schema_version"))
        return cls(**value)


@dataclass
class PolicyInput(Contract):
    """Only these fields may be presented to the model."""
    prompt: str
    tool_schemas: list[dict[str, Any]]
    observations: list[dict[str, Any]] = field(default_factory=list)
    schema_version: str = VERSION

    def __post_init__(self):
        _version(self.schema_version)


@dataclass
class TaskSpec(Contract):
    task_id: str
    group_id: str
    split: str
    prompt: str
    mode: str
    oracle: dict[str, Any]
    tool_dependencies: list[dict[str, Any]]
    normalization: dict[str, Any]
    call_budget: int
    environment_id: str
    fixture_hash: str
    stratum: str = ""
    schema_version: str = VERSION

    def __post_init__(self):
        _version(self.schema_version)
        if self.mode not in {"outcome", "required_tools"}:
            raise ValueError("invalid task mode")
        if self.split not in {"train", "val", "test"} or self.call_budget < 1:
            raise ValueError("invalid split or call budget")

    def policy_input(self, tool_schemas: list[dict[str, Any]], observations=None) -> PolicyInput:
        return PolicyInput(self.prompt, tool_schemas, list(observations or []))


@dataclass
class ExecutionTrace(Contract):
    task_id: str
    events: list[dict[str, Any]]
    completion_reason: str
    provenance: str
    model_spans: list[dict[str, Any]] = field(default_factory=list)
    generated_tokens: int = 0
    model_invocations: int = 0
    schema_version: str = VERSION
    final_answer: str | None = None
    final_state: dict[str, Any] = field(default_factory=dict)
    legacy_messages: list[dict[str, Any]] = field(default_factory=list)
    legacy_steps: int | None = None

    def __post_init__(self):
        _version(self.schema_version)
        if self.provenance not in {"harness", "legacy_untrusted"}:
            raise ValueError("invalid trace provenance")
        if self.generated_tokens < 0 or self.model_invocations < 0:
            raise ValueError("negative counts")


@dataclass
class VerificationResult(Contract):
    answer_success: bool
    execution_compliance: bool
    verified_success_v2: bool
    reasons: list[str] = field(default_factory=list)
    schema_version: str = VERSION

    def __post_init__(self):
        _version(self.schema_version)
        if self.verified_success_v2 != (self.answer_success and self.execution_compliance):
            raise ValueError("verified success must combine answer and compliance")


def task_from_legacy(task, *, group_id, mode, normalization, call_budget, environment_id, fixture_hash, tool_dependencies):
    """Adapt a historical Task; callers declare constraints instead of inferring them."""
    if task.split is None:
        raise ValueError("legacy task must have a split")
    return TaskSpec(task.id, group_id, task.split, task.prompt, mode,
                    {"answer": task.oracle.answer, "final_state": task.oracle.final_state,
                     "numeric_tolerance": task.oracle.numeric_tolerance},
                    tool_dependencies, normalization, call_budget, environment_id,
                    fixture_hash, task.bucket)


def trace_from_legacy(log, *, task_id, completion_reason):
    """Legacy logs lack enough provenance to certify trusted execution."""
    from copy import deepcopy
    return ExecutionTrace(task_id, [{"tool": r.name, "arguments": deepcopy(r.arguments), "output": deepcopy(r.output)}
                                   for r in log.tool_results], completion_reason, "legacy_untrusted",
                          final_answer=log.final_answer, final_state=deepcopy(log.final_state),
                          legacy_messages=deepcopy(log.messages), legacy_steps=log.steps)
