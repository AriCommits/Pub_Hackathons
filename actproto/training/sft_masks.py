"""Fail-closed token ownership checks for the pinned Tiny Aya template."""
from __future__ import annotations

from copy import deepcopy

_BRANCH = ("{% if role == 'user' %}<|USER_TOKEN|>{{ message['content'] }}"
           "{% elif role == 'assistant' or role == 'chatbot' %}<|CHATBOT_TOKEN|>"
           "<|START_RESPONSE|>{{ message['content'] }}<|END_RESPONSE|>"
           "{% elif role == 'system' %}<|SYSTEM_TOKEN|>{{ message['content'] }}"
           "{% endif %}<|END_OF_TURN_TOKEN|>")
_MARKED = ("{% if role == 'user' %}<|USER_TOKEN|>{{ message['content'] }}<|END_OF_TURN_TOKEN|>"
           "{% elif role == 'assistant' or role == 'chatbot' %}<|CHATBOT_TOKEN|><|START_RESPONSE|>"
           "{% generation %}{{ message['content'] }}<|END_RESPONSE|><|END_OF_TURN_TOKEN|>{% endgeneration %}"
           "{% elif role == 'system' %}<|SYSTEM_TOKEN|>{{ message['content'] }}<|END_OF_TURN_TOKEN|>"
           "{% else %}<|END_OF_TURN_TOKEN|>{% endif %}")
_ASSISTANT_HEADER = '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>'


def selected_template(tokenizer, name='default'):
    templates = tokenizer.chat_template
    if not isinstance(templates, dict) or name not in templates:
        raise ValueError(f'named chat template {name!r} unavailable')
    template = templates[name]
    if not isinstance(template, str) or template.count(_BRANCH) != 1:
        raise ValueError('unexpected Tiny Aya template; assistant branch not found exactly once')
    return template


def marked_template(template):
    if template.count(_BRANCH) != 1:
        raise ValueError('assistant branch not found exactly once')
    return template.replace(_BRANCH, _MARKED)


def _ids(tokenizer, text):
    return tokenizer(text, add_special_tokens=False)['input_ids']


def encode_checked(tokenizer, messages, max_length, template_name='default'):
    """Render twice and compare every mask bit against independent turn boundaries.

    No tokenizer truncation is requested; oversized rows are rejected before training.
    Only already-public role/content message pairs are accepted.
    """
    if not messages or messages[-1].get('role') != 'assistant':
        raise ValueError('conversation must end with an assistant finish turn')
    if any(set(m) != {'role', 'content'} or m['role'] not in {'system', 'user', 'assistant'}
           or not isinstance(m['content'], str) for m in messages):
        raise ValueError('messages must contain only public role/content strings')
    template = selected_template(tokenizer, template_name)
    marked = marked_template(template)
    kwargs = dict(tokenize=False, add_generation_prompt=False)
    original = tokenizer.apply_chat_template(messages, chat_template=template, **kwargs)
    rendered = tokenizer.apply_chat_template(messages, chat_template=marked, **kwargs)
    if original.encode('utf-8') != rendered.encode('utf-8'):
        raise ValueError('generation markers changed rendered template bytes')
    result = tokenizer.apply_chat_template(messages, chat_template=marked, tokenize=True,
        add_generation_prompt=False, return_dict=True, return_assistant_tokens_mask=True)
    ids = list(result['input_ids'])
    mask = list(result.get('assistant_masks', []))
    if ids != list(_ids(tokenizer, original)) or len(mask) != len(ids):
        raise ValueError('tokenization or assistant mask alignment mismatch')
    expected = [0] * len(ids)
    for i, message in enumerate(messages):
        if message['role'] != 'assistant':
            continue
        prior = tokenizer.apply_chat_template(messages[:i], chat_template=template, **kwargs)
        current = tokenizer.apply_chat_template(messages[:i + 1], chat_template=template, **kwargs)
        if not current.startswith(prior + _ASSISTANT_HEADER):
            raise ValueError('cannot establish assistant turn boundaries')
        start = len(_ids(tokenizer, prior + _ASSISTANT_HEADER))
        end = len(_ids(tokenizer, current))
        if end <= start or end > len(ids):
            raise ValueError('empty or invalid assistant span')
        expected[start:end] = [1] * (end - start)
    if mask != expected or not any(mask):
        raise ValueError('assistant token mask is empty or does not match turn ownership')
    if len(ids) > max_length:
        raise ValueError(f'conversation length {len(ids)} exceeds max_length {max_length}; no truncation applied')
    return {'input_ids': ids, 'attention_mask': [1] * len(ids),
            'labels': [token if owned else -100 for token, owned in zip(ids, mask)],
            'assistant_mask': mask}


def pad_batch(rows, pad_token_id):
    if not rows:
        raise ValueError('empty batch')
    width = max(len(row['input_ids']) for row in rows)
    batch = {key: [] for key in ('input_ids', 'attention_mask', 'labels')}
    for row in rows:
        n = width - len(row['input_ids'])
        for key, fill in [('input_ids', pad_token_id), ('attention_mask', 0), ('labels', -100)]:
            batch[key].append(deepcopy(row[key]) + [fill] * n)
    return batch
