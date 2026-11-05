from copy import deepcopy

import pytest

from actproto.training.sft_masks import _BRANCH, encode_checked, pad_batch, selected_template


class ToyTokenizer:
    chat_template = {'default': _BRANCH}

    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': list(map(ord, text))}

    def apply_chat_template(self, messages, chat_template, tokenize=False,
                            add_generation_prompt=False, return_dict=False,
                            return_assistant_tokens_mask=False):
        text, mask = 'SYSTEM', [0] * len('SYSTEM')
        for message in messages:
            if message['role'] == 'system':
                continue
            assistant = message['role'] == 'assistant'
            header = ('<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>'
                      if assistant else '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>')
            tail = ('<|END_RESPONSE|>' if assistant else '') + '<|END_OF_TURN_TOKEN|>'
            content = message['content'] + tail
            text += header + content
            mask += [0] * len(header) + [int(assistant)] * len(content)
        if not tokenize:
            return text
        return {'input_ids': list(map(ord, text)), 'assistant_masks': mask}


def conversation():
    return [{'role': 'user', 'content': 'question'}, {'role': 'assistant', 'content': 'call'},
            {'role': 'user', 'content': 'observation'}, {'role': 'assistant', 'content': 'finish'}]


def test_assistant_content_and_both_delimiters_owned_and_padding_masked():
    tokenizer = ToyTokenizer()
    result = encode_checked(tokenizer, conversation(), 5000)
    labels = ''.join(chr(x) for x in result['labels'] if x != -100)
    assert labels == ('call<|END_RESPONSE|><|END_OF_TURN_TOKEN|>'
                      'finish<|END_RESPONSE|><|END_OF_TURN_TOKEN|>')
    batch = pad_batch([result, encode_checked(tokenizer, conversation()[:2], 5000)], 0)
    assert batch['labels'][1][-1] == -100
    assert batch['attention_mask'][1][-1] == 0


def test_marker_alignment_and_overflow_fail_closed():
    tokenizer = ToyTokenizer()
    with pytest.raises(ValueError, match='exceeds'):
        encode_checked(tokenizer, conversation(), 2)
    with pytest.raises(ValueError, match='named'):
        selected_template(tokenizer, 'missing')
    bad = deepcopy(conversation())
    bad[0]['oracle'] = 'secret'
    with pytest.raises(ValueError, match='public'):
        encode_checked(tokenizer, bad, 5000)
    tokenizer.chat_template = {'default': 'unknown'}
    with pytest.raises(ValueError, match='branch'):
        selected_template(tokenizer)


def test_token_loss_has_no_gradient_for_unowned_positions():
    torch = pytest.importorskip('torch')
    result = encode_checked(ToyTokenizer(), conversation(), 5000)
    # Independent position logits make masked-position gradient ownership observable.
    logits = torch.randn(len(result['labels']), 256, requires_grad=True)
    labels = torch.tensor(result['labels'])
    torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100).backward()
    assert torch.all(logits.grad[labels == -100] == 0)
    assert torch.all(logits.grad[labels != -100].abs().sum(dim=1) > 0)


def test_cached_real_tokenizer_preflight():
    transformers = pytest.importorskip('transformers')
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained('CohereLabs/tiny-aya-global',
            revision='af89d219b53ed9b13b8a4645f9c8028973510324', local_files_only=True)
    except OSError:
        pytest.skip('pinned Tiny Aya tokenizer not cached')
    result = encode_checked(tokenizer, conversation(), 8192)
    assert sum(result['assistant_mask']) > 0
    assert len(result['labels']) == len(result['input_ids'])
