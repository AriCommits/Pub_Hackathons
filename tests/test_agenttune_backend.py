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


def test_assembled_episode_preserves_sampled_tokens_and_masks_template():
    from actproto.integrations.agenttune_backend import assemble_episode_tokens
    ids, mask = assemble_episode_tokens(CharacterTokenizer(), [{'role': 'user', 'content': 'question'}], [
        ('CALL', list(map(ord, 'CALL')), 'SECRET_OBSERVATION'),
        ('FINISH', list(map(ord, 'FINISH')), None),
    ])
    text = ''.join(map(chr, ids))
    for value, expected in [('CALL', 1), ('SECRET_OBSERVATION', 0), ('FINISH', 1), ('</assistant>', 0)]:
        start = text.index(value)
        assert mask[start:start + len(value)] == [expected] * len(value)


def test_rejects_sampled_tokens_that_differ_from_rendered_text():
    from actproto.integrations.agenttune_backend import assemble_episode_tokens
    with pytest.raises(BackendCompatibilityError, match='Sampled action'):
        assemble_episode_tokens(CharacterTokenizer(), [{'role': 'user', 'content': 'q'}], [('finish', [123], None)])


def test_rejects_nonterminal_episode():
    from actproto.integrations.agenttune_backend import assemble_episode_tokens
    with pytest.raises(BackendCompatibilityError, match='terminal'):
        assemble_episode_tokens(CharacterTokenizer(), [{'role': 'user', 'content': 'q'}], [('call', [1], 'obs')])


def test_fixture_episode_feeds_observation_and_stops_at_finish():
    from actproto.integrations.agenttune_backend import run_fixture_episode
    tokenizer = CharacterTokenizer()
    actions = iter(['{"name":"calculator","arguments":{"expression":"1+1"}}',
                    '{"name":"finish","arguments":{"answer":"2"}}'])
    prompts, calls = [], []
    def generate(prompt):
        prompts.append(''.join(map(chr, prompt)))
        action = next(actions)
        return action, list(map(ord, action))
    def execute(name, arguments):
        calls.append((name, arguments))
        return '2'
    result = run_fixture_episode(tokenizer, [{'role': 'user', 'content': 'calculate'}], generate, execute)
    assert len(prompts) == 2
    assert 'Tool result:\n2' in prompts[1]
    assert calls == [('calculator', {'expression': '1+1'})]
    assert result['finish']['name'] == 'finish'
    assert result['observations'] == ['2']


def test_adapter_aligns_genuine_sampling_logprobs_and_tool_spans():
    from actproto.integrations.agenttune_backend import AgenticRolloutAdapter
    tokenizer = CharacterTokenizer()
    actions = iter(['{"name":"calculator","arguments":{"expression":"1+1"}}',
                    '{"name":"finish","arguments":{"answer":"2"}}'])
    def generate(ids, trainer):
        action = next(actions)
        return action, list(map(ord, action)), [-0.5] * len(action)
    adapter = AgenticRolloutAdapter(tokenizer, generate, lambda name, args: '2')
    output = adapter(['calculate'], object())
    mask, logprobs = output['env_mask'][0], output['logprobs'][0]
    assert len(mask) == len(logprobs) == len(output['completion_ids'][0])
    assert all(value == (-0.5 if active else 0.0) for active, value in zip(mask, logprobs))
    assert output['observations'] == [['2']]
    assert output['finish'][0]['name'] == 'finish'


def test_adapter_rejects_missing_or_nonfinite_sampling_logprobs():
    from actproto.integrations.agenttune_backend import AgenticRolloutAdapter
    action = '{"name":"finish","arguments":{}}'
    for probs in ([], [float('nan')] * len(action)):
        adapter = AgenticRolloutAdapter(CharacterTokenizer(),
            lambda ids, trainer: (action, list(map(ord, action)), probs), lambda name, args: None)
        with pytest.raises(BackendCompatibilityError):
            adapter(['q'], object())


def test_adapter_rejects_nonpublic_prompt_metadata():
    from actproto.integrations.agenttune_backend import AgenticRolloutAdapter
    adapter = AgenticRolloutAdapter(CharacterTokenizer(), None, None)
    with pytest.raises(BackendCompatibilityError, match='public'):
        adapter([{'prompt': 'question', 'answer': 'private'}], object())


def test_source_provenance_accepts_only_pinned_git_commit(monkeypatch):
    import json
    import actproto.integrations.agenttune_backend as backend
    revision = 'd4db4c12d52342227c4113a70183d0f64aa2f345'
    class Distribution:
        def __init__(self, value):
            self.value = value
        def read_text(self, name):
            return json.dumps({'vcs_info': {'vcs': 'git', 'commit_id': self.value}})
    monkeypatch.setattr(backend.importlib.metadata, 'distribution', lambda name: Distribution(revision))
    assert backend.validate_agenttune_source() == revision
    monkeypatch.setattr(backend.importlib.metadata, 'distribution', lambda name: Distribution('wrong'))
    with pytest.raises(BackendCompatibilityError, match='revision'):
        backend.validate_agenttune_source()


def test_mismatched_sample_is_rejected_before_tool_execution():
    from actproto.integrations.agenttune_backend import run_fixture_episode
    actions, calls = [], []
    def generate(ids):
        actions.append(ids)
        return '{"name":"calculator","arguments":{}}', [999]
    with pytest.raises(BackendCompatibilityError, match='Sampled action'):
        run_fixture_episode(CharacterTokenizer(), [{'role': 'user', 'content': 'q'}],
                            generate, lambda name, args: calls.append(name))
    assert len(actions) == 1
    assert calls == []
