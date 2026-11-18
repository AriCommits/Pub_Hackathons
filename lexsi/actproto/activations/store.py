"""Persist activation dumps: vectors.npz + index.jsonl."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def write_dump(path: str | Path, records: list[dict[str, Any]]) -> None:
    dest = Path(path)
    dest.mkdir(parents=True, exist_ok=True)
    if not records:
        np.savez(dest / "vectors.npz", vectors=np.zeros((0, 0), dtype=np.float32))
        (dest / "index.jsonl").write_text("", encoding="utf-8")
        return
    vectors = np.stack([np.asarray(r["vector"], dtype=np.float32) for r in records])
    np.savez(dest / "vectors.npz", vectors=vectors)
    with (dest / "index.jsonl").open("w", encoding="utf-8") as fh:
        for i, rec in enumerate(records):
            meta = {k: v for k, v in rec.items() if k != "vector"}
            meta["row"] = i
            fh.write(json.dumps(meta) + "\n")


def read_dump(path: str | Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
    dest = Path(path)
    vectors = np.load(dest / "vectors.npz", allow_pickle=False)["vectors"]
    index = []
    with (dest / "index.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                index.append(json.loads(line))
    return vectors, index
