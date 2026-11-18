"""Deterministic v2 policy interface and strict one-call parsing."""
from __future__ import annotations
from dataclasses import dataclass
import json

from actproto.agent.prompts import messages_for_template
from actproto.training.sft_masks import selected_template


class ContextLimitExceeded(ValueError):
    def __init__(self, tokens, limit):
        self.tokens, self.limit = tokens, limit
        super().__init__(f'prompt length {tokens} exceeds available context {limit}')


@dataclass(frozen=True)
class Generation:
    text: str
    tokens: int
    truncated: bool = False


def parse_single_call(text):
    """Accept exactly one canonical tagged call or one bare JSON call."""
    if not isinstance(text, str):
        raise ValueError('model output is not text')
    value = text.strip()
    if value.startswith('<tool_call>') and value.endswith('</tool_call>'):
        value = value[len('<tool_call>'):-len('</tool_call>')].strip()
    try:
        obj = json.loads(value)
    except (ValueError, TypeError) as exc:
        raise ValueError('invalid single tool call') from exc
    if (not isinstance(obj, dict) or set(obj) != {'name', 'arguments'}
            or not isinstance(obj['name'], str) or not isinstance(obj['arguments'], dict)):
        raise ValueError('invalid tool call shape')
    return obj


class ModelPolicy:
    def __init__(self, model, tokenizer, *, template_name='default', max_new_tokens=256):
        if max_new_tokens < 1 or max_new_tokens > 256:
            raise ValueError('max_new_tokens must be between 1 and 256')
        self.model, self.tokenizer = model, tokenizer
        self.template = selected_template(tokenizer, template_name)
        self.max_new_tokens = max_new_tokens

    def __call__(self, messages):
        import torch
        from actproto.agent.model import _tool_call_stopping_criteria, _parameter_device
        encoded = self.tokenizer.apply_chat_template(messages_for_template(messages),
            chat_template=self.template, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors='pt')
        prompt_len = int(encoded['input_ids'].shape[-1])
        context_limit = getattr(self.model.config, 'max_position_embeddings', None)
        if not isinstance(context_limit, int) or context_limit < 1:
            raise ValueError('model context limit unavailable')
        if prompt_len + self.max_new_tokens > context_limit:
            raise ContextLimitExceeded(prompt_len, context_limit - self.max_new_tokens)
        encoded = {k: v.to(_parameter_device(self.model)) for k, v in encoded.items() if hasattr(v, 'to')}
        with torch.inference_mode():
            output = self.model.generate(**encoded, max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                stopping_criteria=_tool_call_stopping_criteria(self.tokenizer, prompt_len))
        ids = output[0, prompt_len:]
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        return Generation(text, int(len(ids)), len(ids) >= self.max_new_tokens)
