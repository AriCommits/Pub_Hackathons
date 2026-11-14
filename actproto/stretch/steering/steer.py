"""h ← h + α * d_tool. Direction is a prototype (or contrast vs mean)."""

from __future__ import annotations

import numpy as np


def steer(hidden: np.ndarray, direction: np.ndarray, alpha: float) -> np.ndarray:
    h = np.asarray(hidden, dtype=np.float32)
    d = np.asarray(direction, dtype=np.float32)
    return h + float(alpha) * d


def contrast_direction(target: np.ndarray, mean: np.ndarray) -> np.ndarray:
    d = np.asarray(target, dtype=np.float32) - np.asarray(mean, dtype=np.float32)
    norm = np.linalg.norm(d) + 1e-8
    return d / norm
