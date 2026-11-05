"""Select the RTX 5060 Ti by UUID. Memory size and nvidia-smi order are not used."""
from __future__ import annotations

from dataclasses import dataclass

from actproto.training.rl_config import RTX_4060_UUID, RTX_5060_TI_UUID


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


def select_5060_ti(devices: list[DeviceRecord]) -> int:
    """Return the record index whose UUID is the RTX 5060 Ti."""
    matches = [device for device in devices if normalize_uuid(device.uuid) == RTX_5060_TI_UUID]
    if len(matches) != 1:
        raise DeviceSelectionError("RTX 5060 Ti UUID is missing")
    chosen = matches[0]
    if normalize_uuid(chosen.uuid) == RTX_4060_UUID:
        raise DeviceSelectionError("the selected device is the RTX 4060")
    if "RTX 5060 Ti" not in chosen.name:
        raise DeviceSelectionError("chosen device name does not identify an RTX 5060 Ti")
    return chosen.index
