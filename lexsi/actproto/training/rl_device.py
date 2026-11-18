"""Select the RTX 5060 Ti by UUID. Memory size and nvidia-smi order are not used."""
from __future__ import annotations

from dataclasses import dataclass
import os

from actproto.training.rl_config import RTX_4060_UUID, preferred_device_uuid

ALLOW_ANY_GPU_ENV = "LEXSI_ALLOW_ANY_GPU"
DIAGNOSTIC_GPU_WARNING = (
    "DIAGNOSTIC-ONLY: LEXSI_ALLOW_ANY_GPU / --allow-any-gpu bypasses the RTX 5060 Ti "
    "UUID pin. Do not use this for claimed study runs or submission evidence."
)


class DeviceSelectionError(ValueError):
    """The visible devices do not contain the required RTX 5060 Ti."""


@dataclass(frozen=True)
class DeviceRecord:
    index: int
    name: str
    uuid: str
    total_memory: int


def normalize_uuid(value: object) -> str:
    text = str(value).strip().lower()
    if text.startswith("gpu-"):
        text = text[4:]
    return text


def allow_any_gpu_enabled(explicit: bool | None = None) -> bool:
    """True when diagnostic-only any-GPU override is active."""
    if explicit is True:
        return True
    if explicit is False:
        return False
    return os.environ.get(ALLOW_ANY_GPU_ENV, "").strip().lower() in {"1", "true", "yes"}


def select_5060_ti(devices: list[DeviceRecord]) -> int:
    """Return the record index whose UUID is the RTX 5060 Ti."""
    expected = preferred_device_uuid()
    matches = [device for device in devices if normalize_uuid(device.uuid) == expected]
    if len(matches) != 1:
        raise DeviceSelectionError("RTX 5060 Ti UUID is missing")
    chosen = matches[0]
    if normalize_uuid(chosen.uuid) == RTX_4060_UUID:
        raise DeviceSelectionError("the selected device is the RTX 4060")
    if "RTX 5060 Ti" not in chosen.name:
        raise DeviceSelectionError("chosen device name does not identify an RTX 5060 Ti")
    return chosen.index


def select_cuda_device(devices: list[DeviceRecord], *, allow_any_gpu: bool | None = None) -> int:
    """Select the pinned 5060 Ti, or any visible GPU when the diagnostic override is on.

    Without the override this fails closed when the lab UUID is absent (Colab, laptop).
    """
    try:
        return select_5060_ti(devices)
    except DeviceSelectionError:
        if not allow_any_gpu_enabled(allow_any_gpu):
            raise DeviceSelectionError(
                "RTX 5060 Ti UUID is missing; pass --allow-any-gpu or set "
                f"{ALLOW_ANY_GPU_ENV}=1 for diagnostic-only Colab/laptop runs"
            ) from None
        if not devices:
            raise DeviceSelectionError("no CUDA devices are visible") from None
        return devices[0].index


def ordinal_from_nvidia_rows(rows: list[tuple[str, str]], uuid: str) -> str:
    """Return the nvidia-smi ordinal for a UUID. This is not a PyTorch index."""
    target = normalize_uuid(uuid)
    matches = [index.strip() for index, raw in rows if normalize_uuid(raw) == target]
    if len(matches) != 1:
        raise DeviceSelectionError("nvidia-smi has no single RTX 5060 Ti UUID")
    return matches[0]


def parse_nvidia_gpu_csv(text: str) -> list[tuple[str, str]]:
    rows = []
    for line in text.splitlines():
        if not line.strip() or "," not in line:
            continue
        index, raw = [part.strip() for part in line.split(",", 1)]
        if index.lower() == "index":
            continue
        rows.append((index, raw))
    return rows
