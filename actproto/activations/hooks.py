"""Residual-stream forward hooks.

Inference hooks detach by default. Pass detach=False for a training-only path
(ST2). Do not leave hooks registered after capture.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from actproto.activations.positions import get_block, resolve_layer, resolve_token_index


class ActivationHook:
    def __init__(self, model: Any, layer: int | str = "auto", *, detach: bool = True, token: str = "prompt_last"):
        self.model = model
        self.layer_index = resolve_layer(model, layer)
        self.detach = detach
        self.token = token
        self._handle: Any = None
        self.last: Any = None

    def _callback(self, _module: Any, _inputs: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        vec = hidden
        if self.detach and hasattr(vec, "detach"):
            vec = vec.detach()
        if hasattr(vec, "cpu"):
            captured = vec.cpu()
        else:
            captured = vec
        self.last = captured

    def enable(self) -> None:
        if self._handle is not None:
            return
        block = get_block(self.model, self.layer_index)
        self._handle = block.register_forward_hook(self._callback)

    def disable(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def captured_token_vector(self, seq_len: int | None = None):
        if self.last is None:
            raise RuntimeError("no activation captured")
        tensor = self.last
        if tensor.dim() == 3:
            # [batch, seq, hidden]
            length = seq_len if seq_len is not None else tensor.size(1)
            idx = resolve_token_index(length, self.token)  # type: ignore[arg-type]
            tensor = tensor[:, idx, :]
        if hasattr(tensor, "float"):
            tensor = tensor.float()
        return tensor

    def numpy_vector(self, seq_len: int | None = None):
        vec = self.captured_token_vector(seq_len)
        if vec.dim() > 1:
            vec = vec[0]
        return vec.numpy()


@contextmanager
def hooked(model: Any, layer: int | str = "auto", *, detach: bool = True, token: str = "prompt_last") -> Iterator[ActivationHook]:
    hook = ActivationHook(model, layer, detach=detach, token=token)
    hook.enable()
    try:
        yield hook
    finally:
        hook.disable()
