"""Task and split types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Bucket = Literal["easy", "confusable", "multi_step", "no_tool", "error_recovery"]
Split = Literal["train", "val", "test"]


@dataclass
class Oracle:
    final_state: dict[str, Any]
    answer: str | None = None
    numeric_tolerance: float = 1e-6


@dataclass
class Task:
    id: str
    bucket: Bucket
    template_family: str
    entity_ids: list[str]
    prompt: str
    gold_tools: list[str]
    gold_args: list[dict[str, Any]]
    oracle: Oracle
    split: Split | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        payload = dict(data)
        raw_oracle = payload.pop("oracle")
        oracle = Oracle(**raw_oracle) if isinstance(raw_oracle, dict) else raw_oracle
        return cls(oracle=oracle, **payload)
