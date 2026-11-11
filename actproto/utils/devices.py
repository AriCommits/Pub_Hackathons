"""Pick local CUDA devices. Prefer the RTX 5060 Ti UUID; never trust memory size alone."""

from __future__ import annotations

import os
import subprocess


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
    """Device string for a single replica. Prefer the 5060 Ti UUID among visible GPUs."""
    import torch

    from actproto.training.rl_config import preferred_device_uuid
    from actproto.training.rl_device import DeviceRecord, DeviceSelectionError, normalize_uuid, select_5060_ti

    indices = _cuda_indices()
    if not indices:
        return "cpu"
    records = []
    for index in indices:
        props = torch.cuda.get_device_properties(index)
        records.append(
            DeviceRecord(index, props.name, normalize_uuid(props.uuid), int(props.total_memory))
        )
    try:
        return f"cuda:{select_5060_ti(records)}"
    except DeviceSelectionError:
        if len(indices) == 1:
            return f"cuda:{indices[0]}"
        raise DeviceSelectionError(
            f"RTX 5060 Ti UUID {preferred_device_uuid()} is not among visible devices"
        ) from None


def pin_5060_ti_visible() -> str | None:
    """Set CUDA_VISIBLE_DEVICES to the nvidia-smi ordinal for the RTX 5060 Ti.

    Returns the ordinal string, or None when CUDA is unavailable. Raises if the
    UUID cannot be selected. Leaves an already-set CUDA_VISIBLE_DEVICES alone
    only after verifying the visible torch device still matches the UUID.
    """
    from actproto.training.rl_config import preferred_device_uuid
    from actproto.training.rl_device import (
        DeviceRecord,
        DeviceSelectionError,
        normalize_uuid,
        ordinal_from_nvidia_rows,
        parse_nvidia_gpu_csv,
        select_5060_ti,
    )

    existing = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing is not None and existing != "":
        import torch

        if not torch.cuda.is_available():
            raise DeviceSelectionError("CUDA_VISIBLE_DEVICES is set but CUDA is unavailable")
        records = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            records.append(
                DeviceRecord(index, props.name, normalize_uuid(props.uuid), int(props.total_memory))
            )
        select_5060_ti(records)
        return existing

    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        # No nvidia-smi: allow CPU-only hosts to continue without pinning.
        import torch

        if not torch.cuda.is_available():
            return None
        raise DeviceSelectionError("nvidia-smi is required to pin the RTX 5060 Ti") from exc

    rows = parse_nvidia_gpu_csv(completed.stdout)
    ordinal = ordinal_from_nvidia_rows(rows, preferred_device_uuid())
    os.environ["CUDA_VISIBLE_DEVICES"] = ordinal
    return ordinal

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
    only = preferred_device()
    if only.startswith("cuda:"):
        return [only.split(":", 1)[1]]
    return ["cpu"]
