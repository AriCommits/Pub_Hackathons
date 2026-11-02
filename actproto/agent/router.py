"""Score an activation against prototypes and optionally select a tool."""

from __future__ import annotations

from typing import Any

import numpy as np

from actproto.activations.similarity import cosine_scores, rank_scores
from actproto.agent.abstention import Threshold, should_route


def route_vector(
    vector: np.ndarray,
    prototypes: dict[str, np.ndarray],
    threshold: Threshold | None,
) -> dict[str, Any]:
    ranked = rank_scores(cosine_scores(vector, prototypes))
    abstain = not should_route(ranked, threshold)
    return {
        "tool": None if abstain else ranked["top1"],
        "confidence": ranked["top1_score"],
        "margin": ranked["margin"],
        "abstain": abstain,
        **ranked,
    }
