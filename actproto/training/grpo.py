"""GRPO entry points.

The previous helpers built a metadata payload and reported success when AgentTune
could be imported, even with an empty dataset and no call to ``trainer.train``.
That stub is gone. ``scripts/train_grpo.py`` now uses ``grpo_run``: plan-only and
collect-only do not optimize, and a real run fails until Plan 9 freezes the SFT
baseline. A beta of zero is an explicit named configuration, not a silent
substitution for the frozen SFT reference. The named GPU profile may train
from that adapter with beta 0; that load is not a KL reference.
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
