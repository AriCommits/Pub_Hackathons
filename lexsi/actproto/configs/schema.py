"""Typed YAML config loader. Keep this module free of re-exports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import yaml

TokenPosition = Literal["prompt_last", "pre_tool_name"]


@dataclass
class ActivationConfig:
    layer: int | Literal["auto"] = "auto"
    token: TokenPosition = "prompt_last"
    detach: bool = True


@dataclass
class ExperimentConfig:
    model_id: str = "CohereLabs/tiny-aya-global"
    revision: str | None = None
    dtype: str = "bfloat16"
    max_new_tokens: int = 256
    max_agent_steps: int = 6
    seed: int = 42
    adapter_path: str | None = None
    tool_schema_version: str = "v1"
    load_in_4bit: bool = False
    activation: ActivationConfig = field(default_factory=ActivationConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        payload = dict(data)
        raw_act = payload.pop("activation", {}) or {}
        known = {f.name for f in fields(cls) if f.name != "activation"}
        filtered = {k: v for k, v in payload.items() if k in known}
        act_fields = {f.name for f in fields(ActivationConfig)}
        activation = ActivationConfig(**{k: v for k, v in raw_act.items() if k in act_fields})
        return cls(activation=activation, **filtered)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    loaded = yaml.safe_load(text) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config {path} must be a mapping")
    return loaded


def load_config(*paths: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    """Merge YAML files left-to-right, then optional overrides."""
    merged: dict[str, Any] = {}
    for path in paths:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        chunk = _read_yaml(p)
        act = merged.get("activation", {})
        if "activation" in chunk and isinstance(chunk["activation"], dict):
            act = {**act, **chunk["activation"]}
        merged.update(chunk)
        if act:
            merged["activation"] = act
    if overrides:
        merged.update(overrides)
    return ExperimentConfig.from_dict(merged)


CONFIG_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.yaml"
MODEL_CONFIG_PATH = CONFIG_DIR / "model.yaml"


def load_default() -> ExperimentConfig:
    return load_config(DEFAULT_CONFIG_PATH, MODEL_CONFIG_PATH)
