"""Bounded GRPO v2 settings. Unsupported options fail before any trainer is built."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import yaml


class GrpoConfigError(ValueError):
    """The requested GRPO run is outside the verified configuration."""


_REQUIRED = {
    "schema_version": "grpo_v2",
    "reference_policy": "beta_zero",
    "use_cpu": True,
    "use_vllm": False,
    "bf16": False,
    "fp16": False,
    "report_to": "none",
    "gradient_checkpointing": False,
}


@dataclass(frozen=True)
class GrpoV2Config:
    reference_policy: str
    beta: float
    learning_rate: float
    num_generations: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_completion_length: int
    temperature: float
    top_p: float
    max_steps: int
    seed: int
    output_dir: str
    baseline_manifest: str
    train_manifest: str

    @property
    def generation_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    def public_dict(self) -> dict:
        return {
            "schema_version": "grpo_v2",
            "reference_policy": self.reference_policy,
            "beta": self.beta,
            "learning_rate": self.learning_rate,
            "num_generations": self.num_generations,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "generation_batch_size": self.generation_batch_size,
            "max_completion_length": self.max_completion_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_steps": self.max_steps,
            "seed": self.seed,
            "output_dir": self.output_dir,
            "baseline_manifest": self.baseline_manifest,
            "train_manifest": self.train_manifest,
            "use_cpu": True,
            "use_vllm": False,
            "quantization": None,
            "vllm": False,
        }


def load_grpo_config(path: str | Path) -> GrpoV2Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise GrpoConfigError("GRPO config must be a mapping")
    unknown = set(raw) - set(_REQUIRED) - {
        "beta", "learning_rate", "num_generations", "per_device_train_batch_size",
        "gradient_accumulation_steps", "max_completion_length", "temperature", "top_p",
        "max_steps", "seed", "output_dir", "baseline_manifest", "train_manifest",
    }
    if unknown:
        raise GrpoConfigError(f"unsupported GRPO options: {', '.join(sorted(unknown))}")
    for key, expected in _REQUIRED.items():
        if raw.get(key) != expected:
            raise GrpoConfigError(f"{key} must be {expected!r}; refusing to drop or rewrite it")
    beta = raw.get("beta")
    if isinstance(beta, bool) or not isinstance(beta, (int, float)) or float(beta) != 0.0:
        raise GrpoConfigError("beta must be 0.0 until a frozen SFT reference is proven; do not label another policy as that reference")
    learning_rate = _positive_float(raw.get("learning_rate"), "learning_rate")
    temperature = _positive_float(raw.get("temperature"), "temperature")
    top_p = _positive_float(raw.get("top_p"), "top_p")
    if top_p > 1:
        raise GrpoConfigError("top_p must be at most 1")
    generations = _positive_int(raw.get("num_generations"), "num_generations")
    batch = _positive_int(raw.get("per_device_train_batch_size"), "per_device_train_batch_size")
    accumulation = _positive_int(raw.get("gradient_accumulation_steps"), "gradient_accumulation_steps")
    if generations < 2:
        raise GrpoConfigError("num_generations must be at least 2")
    if (batch * accumulation) % generations != 0:
        raise GrpoConfigError("per_device_train_batch_size * gradient_accumulation_steps must be divisible by num_generations")
    return GrpoV2Config(
        reference_policy="beta_zero",
        beta=0.0,
        learning_rate=learning_rate,
        num_generations=generations,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=accumulation,
        max_completion_length=_positive_int(raw.get("max_completion_length"), "max_completion_length"),
        temperature=temperature,
        top_p=top_p,
        max_steps=_positive_int(raw.get("max_steps"), "max_steps"),
        seed=_positive_int(raw.get("seed"), "seed"),
        output_dir=_relative_output(raw.get("output_dir")),
        baseline_manifest=str(raw.get("baseline_manifest") or ""),
        train_manifest=str(raw.get("train_manifest") or ""),
    )


def resolve_output(relative: str, root: Path) -> Path:
    """Resolve a results/rl_v2 output and reject paths that escape it."""
    if not relative or Path(relative).is_absolute():
        raise GrpoConfigError("output_dir must be a relative path under results/rl_v2")
    project_root = root.resolve()
    allowed = (project_root / "results" / "rl_v2").resolve()
    target = (project_root / relative).resolve()
    if target == allowed or allowed not in target.parents:
        raise GrpoConfigError("output_dir must be a file directory inside results/rl_v2")
    return target


def _relative_output(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GrpoConfigError("output_dir is required")
    return value.strip()


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise GrpoConfigError(f"{name} must be a positive integer")
    return value


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
        raise GrpoConfigError(f"{name} must be a positive finite number")
    return float(value)
