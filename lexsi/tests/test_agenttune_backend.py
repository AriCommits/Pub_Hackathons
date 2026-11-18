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


_AGENTTUNE = __import__("importlib").util.find_spec("agenttune")


@pytest.mark.skipif(not _AGENTTUNE, reason="AgentTune is installed in the isolated RL environment")
def test_bare_use_cpu_false_stays_rejected():
    from actproto.integrations.agenttune_backend import create_trainer
    with pytest.raises(BackendCompatibilityError, match="Only CPU trainer construction is currently verified"):
        create_trainer(use_cpu=False)


@pytest.mark.skipif(not _AGENTTUNE, reason="AgentTune is installed in the isolated RL environment")
def test_omitted_use_cpu_stays_rejected():
    from actproto.integrations.agenttune_backend import create_trainer
    with pytest.raises(BackendCompatibilityError, match="Only CPU trainer construction is currently verified"):
        create_trainer()


@pytest.mark.skipif(not _AGENTTUNE, reason="AgentTune is installed in the isolated RL environment")
def test_gpu_profile_passes_the_cpu_gate_and_rejects_vllm():
    from actproto.integrations.agenttune_backend import create_trainer
    with pytest.raises(BackendCompatibilityError, match="vLLM is outside the verified baseline"):
        create_trainer(use_cpu=False, gpu_profile="grpo_v2_gpu", use_vllm=True)
    with pytest.raises(BackendCompatibilityError, match="dataset"):
        create_trainer(use_cpu=False, gpu_profile="grpo_v2_gpu", use_vllm=False)


def test_named_default_template_is_explicit_and_a_missing_name_fails():
    import inspect
    from types import SimpleNamespace

    from actproto.agent.inference_v2 import ModelPolicy
    from actproto.integrations.agenttune_backend import render_tokens
    from actproto.training.sft_masks import _BRANCH

    class NamedTokenizer:
        def __init__(self):
            self.chat_template = {
                "default": "DEFAULT-TEMPLATE\n" + _BRANCH,
                "alternate": "ALTERNATE-TEMPLATE\n" + _BRANCH,
            }
            self.seen = []

        def apply_chat_template(self, messages, tokenize, add_generation_prompt, chat_template=None):
            self.seen.append(chat_template)
            text = "".join(f"<{message['role']}>{message['content']}</{message['role']}>" for message in messages)
            if add_generation_prompt:
                text += "<assistant>"
            return list(map(ord, text))

    tokenizer = NamedTokenizer()
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=8192))
    policy = ModelPolicy(model, tokenizer)
    assert policy.template == tokenizer.chat_template["default"]
    assert policy.max_new_tokens == 256
    assert "do_sample=False" in inspect.getsource(ModelPolicy.__call__)
    ids = render_tokens(tokenizer, [{"role": "user", "content": "question"}], generation=True,
                        chat_template=policy.template)
    assert ids
    assert tokenizer.seen == [policy.template]
    with pytest.raises(BackendCompatibilityError, match="named chat template"):
        render_tokens(tokenizer, [{"role": "user", "content": "question"}], generation=True)

    class StringTemplate(CharacterTokenizer):
        chat_template = "plain string template"

        def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
            assert "chat_template" not in kwargs
            return super().apply_chat_template(messages, tokenize, add_generation_prompt)

    assert render_tokens(StringTemplate(), [{"role": "user", "content": "question"}], generation=True)


@pytest.mark.skipif(not _AGENTTUNE, reason="AgentTune is installed in the isolated RL environment")
def test_cpu_construction_cannot_be_relabelled_as_the_gpu_profile():
    from actproto.integrations.agenttune_backend import create_trainer
    with pytest.raises(BackendCompatibilityError, match="use_cpu false"):
        create_trainer(use_cpu=True, gpu_profile="grpo_v2_gpu")

