"""Route only when similarity and margin clear validation-chosen cutoffs.

Missing threshold files fail open (let the model decide).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Threshold:
    min_score: float
    min_margin: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Threshold:
        return cls(min_score=float(data["min_score"]), min_margin=float(data["min_margin"]))

    @classmethod
    def load(cls, path: str | Path) -> Threshold | None:
        p = Path(path)
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        inner = data.get("threshold", data)
        if "min_score" not in inner:
            return None
        return cls.from_dict(inner)


def should_route(scores: dict[str, Any], threshold: Threshold | None) -> bool:
    if threshold is None:
        return False
    top = float(scores.get("top1_score", float("-inf")))
    margin = float(scores.get("margin", 0.0))
    return top >= threshold.min_score and margin >= threshold.min_margin
