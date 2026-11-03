"""Pick local CUDA devices. PyTorch index 0 is not always the larger card."""

from __future__ import annotations

import os


def _cuda_indices() -> list[int]:
    import torch

    if not torch.cuda.is_available():
        return []
    allowed = os.environ.get("ACTPROTO_CUDA_DEVICES")
    if not allowed:
        return list(range(torch.cuda.device_count()))
    chosen = []
    for part in allowed.split(","):
        part = part.strip()
        if part:
            chosen.append(int(part))
    return [i for i in chosen if 0 <= i < torch.cuda.device_count()]


def preferred_device() -> str:
    """Device string for a single replica. Largest allowed GPU, else CPU."""
    import torch

    indices = _cuda_indices()
    if not indices:
        return "cpu"
    best = max(indices, key=lambda i: torch.cuda.get_device_properties(i).total_memory)
    return f"cuda:{best}"


def replica_plan(
    *,
    bytes_per_replica: int = int(6.5 * 1024**3),
    reserve_bytes: int = int(0.45 * 1024**3),
    cap_per_gpu: int = 2,
) -> list[str]:
    """One CUDA index string per resident bf16 replica that should fit.

    Measured Tiny Aya Global bf16 footprint is about 6.24 GB. The defaults
    allow two copies on a 16 GB card and one on an 8 GB card (~3 replicas).
    """
    import torch

    indices = _cuda_indices()
    if not indices:
        return ["cpu"]
    plan: list[str] = []
    for index in indices:
        free, _total = torch.cuda.mem_get_info(index)
        count = int((free - reserve_bytes) // bytes_per_replica)
        count = max(0, min(count, cap_per_gpu))
        plan.extend([str(index)] * count)
    if plan:
        return plan
    # Budget said nothing fits; still try the largest card once.
    only = preferred_device()
    if only.startswith("cuda:"):
        return [only.split(":", 1)[1]]
    return ["cpu"]
