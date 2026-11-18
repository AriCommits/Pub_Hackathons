"""Tiny Aya loader. Tests must not call load_model (downloads gated weights)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from actproto.configs.schema import ExperimentConfig


@dataclass
class LoadedModel:
    tokenizer: Any
    model: Any
    config: ExperimentConfig


def load_model(config: ExperimentConfig, device: str | None = None) -> LoadedModel:
    """Load CohereLabs/tiny-aya-global. Requires HF_TOKEN for the gated repo.

    ``device`` is a torch device string (``cuda:1``, ``cpu``). The default is
    the CUDA device with the most memory.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise ImportError("transformers is required to load Tiny Aya") from exc

    from actproto.utils.devices import preferred_device

    if device is None:
        device = preferred_device()
    kwargs: dict[str, Any] = {}
    if config.revision:
        kwargs["revision"] = config.revision
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, **kwargs)
    model_kwargs: dict[str, Any] = dict(kwargs)
    if config.load_in_4bit:
        model_kwargs["load_in_4bit"] = True
    else:
        model_kwargs["dtype"] = config.dtype if config.dtype != "auto" else "auto"
    if device and device != "cpu":
        model_kwargs["device_map"] = {"": device}
    model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)
    if config.adapter_path:
        try:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, config.adapter_path)
        except ImportError as exc:  # pragma: no cover
            raise ImportError("peft is required to load adapters") from exc
    model.eval()
    return LoadedModel(tokenizer=tokenizer, model=model, config=config)


def _tool_call_stopping_criteria(tokenizer: Any, prompt_len: int):
    from actproto.agent.stop import completion_should_stop

    def _stop_flag(input_ids, scores, **kwargs):
        del scores, kwargs
        new_ids = input_ids[0, prompt_len:]
        text = tokenizer.decode(new_ids, skip_special_tokens=False)
        return completion_should_stop(text)

    try:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList
    except ImportError:
        class _CallableCriteria(list):
            def __call__(self, input_ids, scores, **kwargs):
                return _stop_flag(input_ids, scores, **kwargs)

        return _CallableCriteria([_stop_flag])

    class _StopAfterOneToolCall(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            # transformers StoppingCriteriaList ORs BoolTensors; a bare bool is fragile.
            done = bool(_stop_flag(input_ids, scores, **kwargs))
            return torch.full((input_ids.shape[0],), done, device=input_ids.device, dtype=torch.bool)

    return StoppingCriteriaList([_StopAfterOneToolCall()])

def _parameter_device(model: Any):
    try:
        return model.device
    except Exception:
        return next(model.parameters()).device


def generate_text(loaded: LoadedModel, messages: list[dict[str, Any]], max_new_tokens: int | None = None) -> str:
    import torch

    from actproto.agent.prompts import messages_for_template

    tokenizer = loaded.tokenizer
    model = loaded.model
    encoded = tokenizer.apply_chat_template(
        messages_for_template(messages),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    device = _parameter_device(model)
    encoded = {k: v.to(device) for k, v in encoded.items() if hasattr(v, "to")}
    prompt_len = int(encoded["input_ids"].shape[-1])
    limit = max_new_tokens or loaded.config.max_new_tokens
    with torch.inference_mode():
        out = model.generate(
            **encoded,
            max_new_tokens=limit,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=_tool_call_stopping_criteria(tokenizer, prompt_len),
        )
    new_tokens = out[0, encoded["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
