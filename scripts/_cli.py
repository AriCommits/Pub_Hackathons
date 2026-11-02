"""Shared CLI helpers for trial.yaml-backed scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from actproto.configs.trial_schema import TrialConfig, load_trial  # noqa: E402


def add_trial_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=None,
        help="Trial YAML (default: actproto/configs/trial.yaml). Paths in the file are relative to the lexsi project root.",
    )


def trial_from_args(args: argparse.Namespace) -> TrialConfig:
    return load_trial(args.config)
