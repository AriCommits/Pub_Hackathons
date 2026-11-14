"""Small alpha sweep including 0."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from actproto.activations.similarity import cosine_scores
from actproto.stretch.steering.steer import steer


def default_alphas() -> list[float]:
    return [0.0, 0.1, 0.25, 0.5, 1.0]


def sweep_alphas(
    hidden: np.ndarray,
    direction: np.ndarray,
    prototypes: dict[str, np.ndarray],
    alphas: list[float] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for alpha in alphas or default_alphas():
        steered = steer(hidden, direction, alpha)
        scores = cosine_scores(steered, prototypes)
        top = max(scores, key=scores.get) if scores else None
        rows.append({"alpha": alpha, "top": top, "scores": scores})
    return rows
