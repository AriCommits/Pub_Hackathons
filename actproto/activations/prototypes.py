"""Mean L2-normalized prototypes per gold tool, including finish."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from actproto.activations.similarity import l2_normalize
from actproto.tools.registry import canonical_tool_name


def fit_prototypes(vectors: np.ndarray, labels: list[str]) -> dict[str, np.ndarray]:
    buckets: dict[str, list[np.ndarray]] = defaultdict(list)
    for vec, label in zip(vectors, labels, strict=True):
        buckets[canonical_tool_name(label)].append(l2_normalize(vec))
    prototypes = {}
    for name, items in buckets.items():
        stacked = np.stack(items)
        prototypes[name] = l2_normalize(stacked.mean(axis=0))
    return prototypes


def labels_from_index(index: list[dict]) -> list[str]:
    return [canonical_tool_name(row["gold_tool"]) for row in index]
