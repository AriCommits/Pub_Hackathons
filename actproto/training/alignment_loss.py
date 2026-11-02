"""Contrastive alignment loss over hooked activations vs tool prototypes.

Must stay in the training graph (no detached cosine printed after generate).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def contrastive_loss_numpy(
    z: np.ndarray,
    prototypes: np.ndarray,
    label_index: np.ndarray,
    temperature: float = 0.07,
) -> float:
    """InfoNCE-style loss. z [B,D], prototypes [C,D], labels [B]."""
    z = _norm(z)
    p = _norm(prototypes)
    logits = (z @ p.T) / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    denom = exp.sum(axis=1)
    gather = exp[np.arange(len(label_index)), label_index]
    nll = -np.log(gather / denom)
    return float(nll.mean())


def _norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True) + eps
    return x / n


def total_loss(sft_loss: float, align_loss: float, lam: float) -> float:
    return float(sft_loss) + float(lam) * float(align_loss)


def contrastive_loss_torch(z: Any, prototypes: Any, labels: Any, temperature: float = 0.07) -> Any:
    import torch
    import torch.nn.functional as F

    z = F.normalize(z, dim=-1)
    p = F.normalize(prototypes, dim=-1)
    logits = z @ p.T / temperature
    return F.cross_entropy(logits, labels)
