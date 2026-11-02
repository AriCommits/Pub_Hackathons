"""Pick a residual-stream layer from a small validation sweep."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def candidate_layers(n: int) -> list[int]:
    return sorted({max(0, n // 2), max(0, (2 * n) // 3), max(0, n - 2 if n >= 2 else 0)})


def pick_layer(scores: dict[int, float]) -> tuple[int, float]:
    if not scores:
        raise ValueError("no layer scores")
    layer = max(scores.items(), key=lambda kv: kv[1])[0]
    return layer, float(scores[layer])


def write_layer_config(path: str | Path, layer: int, method: str, score: float) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        yaml.safe_dump({"layer": int(layer), "method": method, "score": float(score)}, sort_keys=False),
        encoding="utf-8",
    )


def try_circuitkit() -> bool:
    try:
        import circuitkit  # noqa: F401

        return True
    except ImportError:
        return False
