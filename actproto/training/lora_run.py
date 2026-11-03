"""GPU LoRA SFT entry used by scripts/train_lora.py."""

from __future__ import annotations

from typing import Any

from actproto.training.lora_config import LoraSftConfig
from actproto.training.sft_dataset import tasks_to_sft_rows


def _with_generation_markers(template: str) -> str | None:
    needle = "<|START_RESPONSE|>{{ message['content'] }}<|END_RESPONSE|>"
    if needle not in template:
        return None
    return template.replace(
        needle,
        "<|START_RESPONSE|>{% generation %}{{ message['content'] }}{% endgeneration %}<|END_RESPONSE|>",
    )


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
    template = tokenizer.chat_template if isinstance(tokenizer.chat_template, str) else str(tokenizer.chat_template)
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

    try:
        _fit(marked is not None)
    except Exception as exc:
        if marked is None:
            raise
        print(f"assistant-only loss failed ({exc}); retrying with loss on the full sequence", flush=True)
        _fit(False)
    return {
        "n_rows": len(rows),
        "output_dir": config.output_dir,
        "seq_max": seq_max,
        "max_length": max_length,
        "device": device_name,
    }
