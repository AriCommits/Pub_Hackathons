"""Bounded GRPO v2 settings. Unsupported options fail before any trainer is built."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import yaml


class GrpoConfigError(ValueError):
    """The requested GRPO run is outside the verified configuration."""


GRPO_GPU_PROFILE = "grpo_v2_gpu"
RTX_5060_TI_UUID = "134c34ef-c155-b001-4e3a-b9134b072c21"
RTX_4060_UUID = "5b885e12-623f-268c-f07c-ee33db379969"
TOOL_REWARD_LEARNING_RATE = 5e-6
E2_LORA_TARGETS = frozenset({
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
})


_CPU_REQUIRED = {
    "schema_version": "grpo_v2",
    "reference_policy": "beta_zero",
    "use_cpu": True,
    "use_vllm": False,
    "bf16": False,
    "fp16": False,
    "report_to": "none",
    "gradient_checkpointing": False,
}
_GPU_REQUIRED = {
    "schema_version": GRPO_GPU_PROFILE,
    "reference_policy": "beta_zero",
    "use_cpu": False,
    "use_vllm": False,
    "bf16": True,
    "fp16": False,
    "report_to": "none",
    "gradient_checkpointing": True,
}
_GPU_DEVICE_KEYS = {"device_name", "device_uuid"}
_SHARED_KEYS = {
    "beta", "learning_rate", "num_generations", "per_device_train_batch_size",
    "gradient_accumulation_steps", "max_completion_length", "temperature", "top_p",
    "max_steps", "seed", "output_dir", "baseline_manifest", "train_manifest",
    "per_turn_new_tokens",
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
    schema_version: str = "grpo_v2"
    use_cpu: bool = True
    bf16: bool = False
    fp16: bool = False
    gradient_checkpointing: bool = False
    device_name: str = ""
    device_uuid: str = ""
    per_turn_new_tokens: int = 256

    @property
    def generation_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    def public_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
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
            "use_cpu": self.use_cpu,
            "bf16": self.bf16,
            "fp16": self.fp16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "device_name": self.device_name,
            "device_uuid": self.device_uuid,
            "use_vllm": False,
            "quantization": None,
            "vllm": False,
            "per_turn_new_tokens": self.per_turn_new_tokens,
        }


def load_grpo_config(path: str | Path) -> GrpoV2Config:
    """Load a CPU or named GPU profile. A manifest path is stored, never opened."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise GrpoConfigError("GRPO config must be a mapping")
    schema = raw.get("schema_version")
    if schema == "grpo_v2":
        required = _CPU_REQUIRED
        extra: set[str] = set()
    elif schema == "grpo_v2_gpu":
        required = _GPU_REQUIRED
        extra = set(_GPU_DEVICE_KEYS)
    else:
        raise GrpoConfigError("schema_version must be 'grpo_v2' or 'grpo_v2_gpu'")
    unknown = set(raw) - set(required) - _SHARED_KEYS - extra
    if unknown:
        raise GrpoConfigError(f"unsupported GRPO options: {', '.join(sorted(unknown))}")
    for key, expected in required.items():
        if raw.get(key) != expected:
            raise GrpoConfigError(f"{key} must be {expected!r}; refusing to drop or rewrite it")
    device_name = ""
    device_uuid = ""
    if schema == "grpo_v2_gpu":
        device_name, device_uuid = _require_5060_ti(raw)
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
    per_turn = raw.get("per_turn_new_tokens", 256)
    if isinstance(per_turn, bool) or type(per_turn) is not int or per_turn != 256:
        raise GrpoConfigError("per_turn_new_tokens must be 256")
    max_steps = _positive_int(raw.get("max_steps"), "max_steps")
    output_dir = _relative_output(raw.get("output_dir"))
    if Path(output_dir).name == "grpo_gpu_prefix" and max_steps > 20:
        raise GrpoConfigError("grpo_gpu_prefix max_steps must be at most 20")
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
        max_steps=max_steps,
        seed=_positive_int(raw.get("seed"), "seed"),
        output_dir=output_dir,
        per_turn_new_tokens=per_turn,
        baseline_manifest=str(raw.get("baseline_manifest") or ""),
        train_manifest=str(raw.get("train_manifest") or ""),
        schema_version=schema,
        use_cpu=bool(required["use_cpu"]),
        bf16=bool(required["bf16"]),
        fp16=bool(required["fp16"]),
        gradient_checkpointing=bool(required["gradient_checkpointing"]),
        device_name=device_name,
        device_uuid=device_uuid,
    )


def guard_tool_sampling(config: GrpoV2Config) -> None:
    """Keep tool-reward sampling at group size 2, temperature 0.8, and top-p 0.95.

    A larger group of identical rewards is still advantage 0. ``LEXSI_RL_GROUP4_FIT``
    is not consulted and is not set.
    """
    if config.temperature != 0.8 or config.top_p != 0.95:
        raise GrpoConfigError("the tool-reward run keeps temperature 0.8 and top_p 0.95")
    if config.num_generations != 2:
        raise GrpoConfigError(
            "num_generations must stay 2; a larger group of identical rewards is still advantage 0"
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


def _require_5060_ti(raw: dict) -> tuple[str, str]:
    name = raw.get("device_name")
    uuid = raw.get("device_uuid")
    if not isinstance(name, str) or "RTX 5060 Ti" not in name:
        raise GrpoConfigError("device_name must identify an RTX 5060 Ti")
    if uuid == RTX_4060_UUID:
        raise GrpoConfigError("the RTX 4060 cannot host this GPU profile")
    if not isinstance(uuid, str) or uuid != RTX_5060_TI_UUID:
        raise GrpoConfigError("device_uuid must be the RTX 5060 Ti")
    return name, uuid


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
