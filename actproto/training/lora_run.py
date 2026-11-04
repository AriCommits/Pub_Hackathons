"""GPU LoRA SFT entry used by scripts/train_lora.py."""

from __future__ import annotations

from typing import Any

from actproto.training.lora_config import LoraSftConfig
from actproto.training.sft_dataset import tasks_to_sft_rows


_GENERATION_NEEDLE = "<|START_RESPONSE|>{{ message['content'] }}<|END_RESPONSE|>"
_GENERATION_MARKED = (
    "<|START_RESPONSE|>{% generation %}{{ message['content'] }}{% endgeneration %}<|END_RESPONSE|>"
)


def _template_text(chat_template: Any) -> str:
    """Return the default chat template string.

    Tiny Aya stores the template as a named list. ``str(...)`` on that list
    escapes quotes, so the generation needle never matches.
    """
    if isinstance(chat_template, str):
        return chat_template
    candidates: list[Any] = []
    if isinstance(chat_template, dict):
        for key in ("default", "tool_use"):
            if key in chat_template:
                candidates.append(chat_template[key])
        candidates.extend(chat_template.values())
    elif isinstance(chat_template, (list, tuple)):
        candidates.extend(chat_template)
    else:
        return str(chat_template)
    for item in candidates:
        if isinstance(item, str) and "<|START_RESPONSE|>" in item:
            return item
        if isinstance(item, dict):
            text = item.get("template") or item.get("chat_template")
            if isinstance(text, str) and "<|START_RESPONSE|>" in text:
                return text
    return str(chat_template)


def _with_generation_markers(template: str) -> str | None:
    if _GENERATION_NEEDLE not in template:
        return None
    return template.replace(_GENERATION_NEEDLE, _GENERATION_MARKED)


def run_lora_sft(tasks, config: LoraSftConfig, model_id: str, revision: str | None = None) -> dict[str, Any]:
    """LoRA SFT on the only visible GPU. Pin CUDA_VISIBLE_DEVICES before calling."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from actproto.agent.prompts import messages_for_template, system_prompt

    rows = tasks_to_sft_rows(tasks)
    system = {"role": "system", "content": system_prompt()}
    conversations = [messages_for_template([system, *row["messages"]]) for row in rows]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    template = _template_text(tokenizer.chat_template)
    def _ntok(msgs: list) -> int:
        encoded = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=False, return_dict=True
        )
        ids = encoded["input_ids"]
        if hasattr(ids, "shape"):
            return int(ids.shape[-1])
        if ids and isinstance(ids[0], list):
            return len(ids[0])
        return len(ids)

    lengths = [_ntok(msgs) for msgs in conversations]
    seq_max = max(lengths)
    max_length = min(8192, max(256, ((seq_max + 63) // 64) * 64))
    marked = _with_generation_markers(template)
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(
        {
            "n_rows": len(rows),
            "seq_max": seq_max,
            "seq_p50": sorted(lengths)[len(lengths) // 2],
            "max_length": max_length,
            "assistant_only": marked is not None,
            "device": device_name,
        },
        flush=True,
    )

    def _load_model():
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16)
        if torch.cuda.is_available():
            model = model.to("cuda")
        model.config.use_cache = False
        return model

    def _fit(assistant_only: bool) -> None:
        tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        if assistant_only and marked is not None:
            tok.chat_template = marked
        args = SFTConfig(
            output_dir=config.output_dir,
            learning_rate=config.learning_rate,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_length=max_length,
            assistant_only_loss=assistant_only,
            logging_steps=config.logging_steps,
            save_strategy="no",
            bf16=torch.cuda.is_available(),
            gradient_checkpointing=True,
            report_to="none",
            seed=42,
        )
        trainer = SFTTrainer(
            model=_load_model(),
            args=args,
            train_dataset=Dataset.from_dict({"messages": conversations}),
            processing_class=tok,
            peft_config=LoraConfig(**config.peft_kwargs()),
        )
        trainer.train()
        trainer.save_model(config.output_dir)

    if marked is None:
        raise RuntimeError("assistant-only loss marker was not found in the chat template; refusing full-sequence loss")
    _fit(True)
    return {
        "n_rows": len(rows),
        "output_dir": config.output_dir,
        "seq_max": seq_max,
        "max_length": max_length,
        "device": device_name,
    }
