"""Choose an abstention threshold on the validation split only."""

from __future__ import annotations

from typing import Any

import numpy as np

from actproto.activations.similarity import cosine_scores, rank_scores


def accuracy_coverage(
    vectors: np.ndarray,
    labels: list[str],
    prototypes: dict[str, np.ndarray],
    min_score: float,
    min_margin: float,
) -> dict[str, float]:
    routed = 0
    correct = 0
    for vec, label in zip(vectors, labels, strict=True):
        ranked = rank_scores(cosine_scores(vec, prototypes))
        if ranked["top1_score"] >= min_score and ranked["margin"] >= min_margin:
            routed += 1
            if ranked["top1"] == label:
                correct += 1
    n = len(labels) or 1
    coverage = routed / n
    accuracy = (correct / routed) if routed else 0.0
    return {"coverage": coverage, "accuracy": accuracy, "n": float(n), "routed": float(routed)}


def choose_threshold(
    val_vectors: np.ndarray,
    val_labels: list[str],
    prototypes: dict[str, np.ndarray],
    *,
    min_coverage: float = 0.5,
    score_grid: list[float] | None = None,
    margin_grid: list[float] | None = None,
) -> dict[str, Any]:
    score_grid = score_grid or [round(x, 2) for x in np.linspace(-0.2, 0.95, 24)]
    margin_grid = margin_grid or [round(x, 2) for x in np.linspace(0.0, 0.6, 13)]
    best: dict[str, Any] | None = None
    curve = []
    for min_score in score_grid:
        for min_margin in margin_grid:
            stats = accuracy_coverage(val_vectors, val_labels, prototypes, min_score, min_margin)
            point = {"min_score": float(min_score), "min_margin": float(min_margin), **stats}
            curve.append(point)
            if stats["coverage"] + 1e-9 < min_coverage:
                continue
            if best is None or stats["accuracy"] > best["accuracy"] or (
                stats["accuracy"] == best["accuracy"] and stats["coverage"] > best["coverage"]
            ):
                best = point
    if best is None:
        # Fail open: route never (threshold impossible to meet) if coverage cannot be met.
        best = {
            "min_score": 2.0,
            "min_margin": 2.0,
            "coverage": 0.0,
            "accuracy": 0.0,
            "n": float(len(val_labels)),
            "routed": 0.0,
        }
    best["curve_len"] = len(curve)
    best["min_coverage_constraint"] = min_coverage
    return best
