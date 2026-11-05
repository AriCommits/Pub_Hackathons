"""Validation summaries and clustered uncertainty for the v2 SFT study."""
from __future__ import annotations

import math
import random


def _success(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("verified success must be a finite number")
    return float(value)


def macro_verified_success(rows: list[dict]) -> float:
    """Unweighted mean of per-stratum success. An empty stratum cannot be skipped."""
    if not rows:
        raise ValueError("no rows")
    buckets: dict[str, list[float]] = {}
    for row in rows:
        stratum = row.get("stratum")
        if not isinstance(stratum, str) or not stratum:
            raise ValueError("every row needs a stratum")
        buckets.setdefault(stratum, []).append(_success(row.get("verified_success_v2")))
    return sum(sum(values) / len(values) for values in buckets.values()) / len(buckets)


def paired_group_bootstrap(left: list[dict], right: list[dict], *, draws: int = 1000, seed: int = 42) -> dict:
    """Resample task groups, not individual tasks, and return a 95% interval for the mean difference."""
    if draws < 100 or len(left) != len(right) or not left:
        raise ValueError("paired bootstrap needs aligned rows and at least 100 draws")
    grouped: dict[str, list[tuple[float, float]]] = {}
    for first, second in zip(left, right):
        if first.get("task_id") != second.get("task_id") or first.get("group_id") != second.get("group_id"):
            raise ValueError("bootstrap pairs must be the same task and group")
        group = first.get("group_id")
        if not isinstance(group, str) or not group:
            raise ValueError("group id is required")
        grouped.setdefault(group, []).append((_success(first.get("verified_success_v2")), _success(second.get("verified_success_v2"))))
    groups = list(grouped.values())

    def difference(sample):
        total = count = 0.0
        for group in sample:
            for before, after in group:
                total += after - before
                count += 1
        return total / count

    observed = difference(groups)
    if not math.isfinite(observed):
        raise ValueError("observed difference is not finite")
    rng = random.Random(seed)
    samples = sorted(difference([groups[rng.randrange(len(groups))] for _ in groups]) for _ in range(draws))
    return {
        "difference": observed,
        "low": samples[int(0.025 * (draws - 1))],
        "high": samples[math.ceil(0.975 * (draws - 1)) - 1],
        "draws": draws,
        "groups": len(groups),
    }
