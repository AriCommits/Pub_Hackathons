"""Rejected legacy GRPO helpers — not the trainer.

Do not import this module for training. Use ``scripts/train_grpo.py`` which
calls ``actproto.training.grpo_run``. Every function here raises
``GrpoConfigError`` so a metadata-only or import-only path cannot look like a
successful optimization.
"""
from __future__ import annotations

from actproto.training.rl_config import GrpoConfigError


def build_grpo_kwargs(*args, **kwargs):
    """Reject the old metadata-only wiring instead of reporting a successful trainer."""
    del args, kwargs
    raise GrpoConfigError("metadata-only GRPO setup is not a training run; use scripts/train_grpo.py")


def try_create_trainer(**kwargs):
    """Importing AgentTune is not a completed optimization."""
    del kwargs
    raise GrpoConfigError("importing the trainer factory does not train; use scripts/train_grpo.py")
