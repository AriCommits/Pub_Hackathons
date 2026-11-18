"""Extract labeled residual-stream vectors from a hookable model."""

from __future__ import annotations

from typing import Any

import numpy as np

from actproto.activations.hooks import hooked
from actproto.activations.store import write_dump
from actproto.data.types import Task
from actproto.tools.registry import canonical_tool_name


def extract_from_hidden(hidden: Any, gold_tool: str, **meta: Any) -> dict[str, Any]:
    vector = np.asarray(hidden, dtype=np.float32).reshape(-1)
    return {
        "gold_tool": canonical_tool_name(gold_tool),
        "vector": vector,
        **meta,
    }


def extract_tasks(
    model: Any,
    tasks: list[Task],
    *,
    encode_fn,
    layer: int | str = "auto",
    detach: bool = True,
    token: str = "prompt_last",
    model_id: str = "toy",
    revision: str | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """encode_fn(task) -> input that model() accepts (e.g. dict of tensors)."""
    records: list[dict[str, Any]] = []
    with hooked(model, layer, detach=detach, token=token) as hook:
        for task in tasks:
            gold = task.gold_tools[0] if task.gold_tools else "finish"
            batch = encode_fn(task)
            seq_len = None
            if isinstance(batch, dict) and "input_ids" in batch:
                seq_len = int(batch["input_ids"].shape[-1])
            model(**batch) if isinstance(batch, dict) else model(batch)
            vector = hook.numpy_vector(seq_len)
            records.append(
                extract_from_hidden(
                    vector,
                    gold,
                    task_id=task.id,
                    step=0,
                    pred_tool=None,
                    layer=hook.layer_index,
                    token_index=seq_len - 1 if seq_len else 0,
                    model_id=model_id,
                    revision=revision,
                    seed=seed,
                    split=task.split,
                )
            )
    return records


def extract_and_write(model: Any, tasks: list[Task], out_dir: str, **kwargs: Any) -> None:
    records = extract_tasks(model, tasks, **kwargs)
    write_dump(out_dir, records)
