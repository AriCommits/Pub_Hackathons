import pytest
from actproto.integrations.agenttune_backend import BackendCompatibilityError, episode_token_mask


class CharacterTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        text = ''.join(f'<{m["role"]}>{m["content"]}</{m["role"]}>' for m in messages)
        if add_generation_prompt:
            text += '<assistant>'
        return [ord(c) for c in text]


def test_episode_masks_tool_observation():
    ids, mask = episode_token_mask(CharacterTokenizer(), [{'role': 'user', 'content': 'question'}],
                                   'CALL', 'SECRET_OBSERVATION', 'FINISH')
    text = ''.join(map(chr, ids))
    for value, expected in [('CALL', 1), ('SECRET_OBSERVATION', 0), ('FINISH', 1)]:
        start = text.index(value)
        assert mask[start:start + len(value)] == [expected] * len(value)


def test_rejects_template_that_rewrites_history():
    class RewritingTokenizer(CharacterTokenizer):
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            result = super().apply_chat_template(messages, tokenize, add_generation_prompt)
            return [len(messages)] + result
    with pytest.raises(BackendCompatibilityError, match='rewrites'):
        episode_token_mask(RewritingTokenizer(), [{'role': 'user', 'content': 'q'}], 'call', 'obs', 'finish')
