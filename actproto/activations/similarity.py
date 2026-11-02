"""Cosine similarity against tool prototypes."""

from __future__ import annotations

from typing import Any

import numpy as np


def l2_normalize(vectors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vec = np.asarray(vectors, dtype=np.float64)
    if vec.ndim == 1:
        norm = np.linalg.norm(vec) + eps
        return (vec / norm).astype(np.float32)
    norms = np.linalg.norm(vec, axis=1, keepdims=True) + eps
    return (vec / norms).astype(np.float32)


def cosine_scores(vector: np.ndarray, prototypes: dict[str, np.ndarray]) -> dict[str, float]:
    q = l2_normalize(vector)
    scores = {}
    for name, proto in prototypes.items():
        p = l2_normalize(proto)
        scores[name] = float(np.dot(q, p))
    return scores


def rank_scores(scores: dict[str, float]) -> dict[str, Any]:
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top1 = ordered[0] if ordered else ("", 0.0)
    top2 = ordered[1] if len(ordered) > 1 else ("", float("-inf"))
    margin = top1[1] - (top2[1] if np.isfinite(top2[1]) else 0.0)
    return {
        "top1": top1[0],
        "top1_score": top1[1],
        "top2": top2[0],
        "top2_score": top2[1] if np.isfinite(top2[1]) else None,
        "margin": margin,
        "scores": scores,
    }
