"""Map D1 tasks to AgentCase objects (AuditKIT-compatible)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from actproto.data.types import Task


class Outcome(Protocol):
    def check(self, *, final_state: dict[str, Any], final_answer: str | None) -> dict[str, str]: ...


@dataclass
class FinalStateAssertion:
    key: str
    equals: Any

    def check(self, *, final_state: dict[str, Any], final_answer: str | None = None) -> dict[str, str]:
        actual = final_state.get(self.key)
        if actual == self.equals:
            return {"verdict": "success", "reason": f"{self.key}=={self.equals!r}"}
        return {"verdict": "failure", "reason": f"{self.key}: expected {self.equals!r} got {actual!r}"}


@dataclass
class AnswerAssertion:
    equals: str

    def check(self, *, final_state: dict[str, Any], final_answer: str | None = None) -> dict[str, str]:
        actual = final_answer if final_answer is not None else final_state.get("answer")
        if str(actual) == self.equals:
            return {"verdict": "success", "reason": "answer matches"}
        return {"verdict": "failure", "reason": f"answer expected {self.equals!r} got {actual!r}"}


@dataclass
class CompoundOutcome:
    parts: list[Any]

    def check(self, *, final_state: dict[str, Any], final_answer: str | None = None) -> dict[str, str]:
        for part in self.parts:
            result = part.check(final_state=final_state, final_answer=final_answer)
            if result["verdict"] != "success":
                return result
        return {"verdict": "success", "reason": "all assertions passed"}


@dataclass
class AgentCase:
    id: str
    task: str
    category: str
    allowed_tools: list[str]
    reference_turns: list[list[dict[str, Any]]]
    outcome: Any
    metadata: dict[str, Any] = field(default_factory=dict)


def task_to_case(task: Task) -> AgentCase:
    parts: list[Any] = []
    for key, value in task.oracle.final_state.items():
        parts.append(FinalStateAssertion(key, equals=value))
    if task.oracle.answer is not None:
        parts.append(AnswerAssertion(equals=task.oracle.answer))
    outcome: Any = parts[0] if len(parts) == 1 else CompoundOutcome(parts)
    ref = [[{"name": n, "arguments": a} for n, a in zip(task.gold_tools, task.gold_args, strict=True)]]
    return AgentCase(
        id=task.id,
        task=task.prompt,
        category=task.bucket,
        allowed_tools=sorted(set(task.gold_tools) | {"finish"}),
        reference_turns=ref,
        outcome=outcome,
        metadata={"template_family": task.template_family, "entity_ids": task.entity_ids},
    )
