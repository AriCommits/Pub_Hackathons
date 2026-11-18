"""LoRA / SFT hyperparameters loaded from configs/sft.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SFT_PATH = Path(__file__).resolve().parents[1] / "configs" / "sft.yaml"


@dataclass
class LoraSftConfig:
    output_dir: str = "results/adapters/lora_run"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    max_seq_len: int = 2048
    learning_rate: float = 2e-4
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    load_in_4bit: bool = True
    logging_steps: int = 10
    save_steps: int = 100

    @classmethod
    def load(cls, path: str | Path | None = None) -> LoraSftConfig:
        p = Path(path) if path else DEFAULT_SFT_PATH
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def peft_kwargs(self) -> dict[str, Any]:
        return {
            "r": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": list(self.target_modules),
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
