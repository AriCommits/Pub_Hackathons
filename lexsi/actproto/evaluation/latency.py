"""Wall-clock per episode. CUDA peak is optional and must not fail the run."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator


@contextmanager
def track_latency() -> Iterator[dict[str, Any]]:
    stats: dict[str, Any] = {"latency_s": None, "cuda_peak_bytes": None}
    t0 = time.perf_counter()
    try:
        yield stats
    finally:
        stats["latency_s"] = time.perf_counter() - t0
        try:
            import torch

            if torch.cuda.is_available():
                stats["cuda_peak_bytes"] = int(torch.cuda.max_memory_allocated())
        except Exception:
            stats["cuda_peak_bytes"] = None


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    idx = min(len(ordered) - 1, max(0, int(round((p / 100) * (len(ordered) - 1)))))
    return ordered[idx]
