"""Resolve layer index and token position from config."""

from __future__ import annotations

from typing import Any, Literal

TokenMode = Literal["prompt_last", "pre_tool_name"]


def _walk_modules(model: Any):
    current = model
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        nxt = None
        for attr in ("base_model", "model"):
            candidate = getattr(current, attr, None)
            if candidate is not None and id(candidate) not in seen:
                nxt = candidate
                break
        current = nxt


def n_layers(model: Any) -> int:
    for current in _walk_modules(model):
        if hasattr(current, "n_layers"):
            return int(current.n_layers)
        if hasattr(current, "blocks"):
            return len(current.blocks)
        if hasattr(current, "layers"):
            return len(current.layers)
    raise AttributeError("cannot resolve layer count")


def resolve_layer(model: Any, layer: int | str = "auto") -> int:
    total = n_layers(model)
    if layer == "auto" or layer is None:
        return max(0, (2 * total) // 3)
    idx = int(layer)
    if idx < 0:
        idx = total + idx
    if not 0 <= idx < total:
        raise IndexError(f"layer {idx} out of range for {total} layers")
    return idx


def get_block(model: Any, layer_index: int) -> Any:
    for current in _walk_modules(model):
        if hasattr(current, "blocks"):
            return current.blocks[layer_index]
        if hasattr(current, "layers"):
            return current.layers[layer_index]
    raise AttributeError("cannot resolve transformer block")


def prompt_last_index(seq_len: int) -> int:
    if seq_len <= 0:
        raise ValueError("empty sequence")
    return seq_len - 1


def resolve_token_index(seq_len: int, mode: TokenMode = "prompt_last") -> int:
    if mode == "prompt_last":
        return prompt_last_index(seq_len)
    if mode == "pre_tool_name":
        # Implemented as last prompt token for generation-time capture.
        # A true pre-tool-name hook during sampling is a generate() callback.
        return prompt_last_index(seq_len)
    raise NotImplementedError(f"unknown token mode {mode}")
