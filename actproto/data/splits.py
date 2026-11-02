"""Leakage-aware split assignment: (template_family, entity_id) is unique to one split."""

from __future__ import annotations

from collections import defaultdict

from actproto.data.types import Split, Task


def leakage_keys(task: Task) -> set[tuple[str, str]]:
    return {(task.template_family, eid) for eid in task.entity_ids}


def assert_no_leakage(train: list[Task], val: list[Task], test: list[Task]) -> None:
    groups = {"train": train, "val": val, "test": test}
    keys: dict[str, set[tuple[str, str]]] = {name: set() for name in groups}
    for name, tasks in groups.items():
        for task in tasks:
            keys[name].update(leakage_keys(task))
    overlaps = []
    names = list(keys)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = keys[a] & keys[b]
            if overlap:
                overlaps.append((a, b, sorted(overlap)[:5], len(overlap)))
    if overlaps:
        raise AssertionError(f"split leakage: {overlaps}")


def split_counts(tasks: list[Task]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for t in tasks:
        counts[t.split or "unset"] += 1
    return dict(counts)


def assign_split_by_index(n: int, i: int) -> Split:
    """Deterministic 60/20/20 by position in a sorted entity list."""
    if n <= 0:
        return "train"
    t1 = max(1, int(n * 0.6)) if n >= 5 else max(1, n - 2)
    t2 = max(t1 + 1, int(n * 0.8)) if n >= 5 else max(t1, n - 1)
    if n < 3:
        return ("train", "val", "test")[min(i, n - 1)]
    if i < t1:
        return "train"
    if i < t2:
        return "val"
    return "test"
