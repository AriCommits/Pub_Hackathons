import json
from pathlib import Path

import pytest

from actproto.agent.inference_v2 import Generation, ModelPolicy
from actproto.data.benchmark_v2 import _make_example
from actproto.data.fixtures_v2 import build_environment
from actproto.evaluation.experiment_runner import evaluate, load_data, validation_evaluator
from actproto.experiments.manifest import hash_file


def make_dataset(path):
    env = build_environment(3)
    fixture = {'identifier': env.identifier, 'records': env.records, 'sales': env.sales, 'documents': env.documents}
    (path / 'fixtures.json').write_text(json.dumps(fixture), encoding='utf-8')
    item = _make_example(0, 'val', 'single_tool', env, env.digest)
    item['task']['split'] = 'val'
    (path / 'val.jsonl').write_text(json.dumps(item) + '\n', encoding='utf-8')
    manifest = {'schema_version': 'tooluse_v2', 'environment_id': env.identifier, 'fixture_hash': env.digest,
                'files': {name: hash_file(path / name) for name in ('fixtures.json', 'val.jsonl')}}
    (path / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return path / 'manifest.json', item


def test_manifest_checked_data_and_shared_scorer(tmp_path):
    path, item = make_dataset(tmp_path)
    env, tasks = load_data(path, 'val')
    calls = iter(item['gold_calls'])
    def policy(messages):
        call = next(calls)
        return Generation(json.dumps({'name': call['tool'], 'arguments': call['arguments']}), 1)
    records, metrics = evaluate(tasks, env, policy)
    assert metrics['macro_verified_success_v2'] == 1 and records[0]['verification']['verified_success_v2']
    (tmp_path / 'val.jsonl').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='hash mismatch'):
        load_data(path, 'val')


def test_test_split_guards_and_callback(tmp_path, monkeypatch):
    path, _ = make_dataset(tmp_path)
    with pytest.raises(ValueError, match='unlock'):
        load_data(path, 'test')
    with pytest.raises(ValueError, match='validation'):
        load_data(path, 'test', callback=True)
    monkeypatch.setenv('ACTPROTO_V2_MANIFEST', str(path))
    with pytest.raises(ValueError, match='validation'):
        validation_evaluator('checkpoint', split='test', model=None, tokenizer=None)


def test_real_tokenizer_prompt_contract_when_cached():
    transformers = pytest.importorskip('transformers')
    try:
        tok = transformers.AutoTokenizer.from_pretrained('CohereLabs/tiny-aya-global',
            revision='af89d219b53ed9b13b8a4645f9c8028973510324', local_files_only=True)
    except OSError:
        pytest.skip('pinned Tiny Aya tokenizer not cached')
    policy = ModelPolicy(None, tok)
    messages = [{'role':'system','content':'tools'}, {'role':'user','content':'question'}]
    rendered = tok.apply_chat_template(messages, chat_template=policy.template,
                                       tokenize=False, add_generation_prompt=True)
    assert 'question' in rendered and 'tools' in rendered


def test_fixture_visibility_is_split_scoped_and_full_hash_bound(tmp_path):
    env = build_environment(3)
    fixture = {'identifier': env.identifier, 'records': env.records, 'sales': env.sales, 'documents': env.documents}
    (tmp_path / 'fixtures.json').write_text(json.dumps(fixture), encoding='utf-8')
    items = []
    for idx, split in ((0, 'train'), (1, 'val')):
        item = _make_example(idx, split, 'single_tool', env, env.digest)
        item['task']['split'] = split
        (tmp_path / f'{split}.jsonl').write_text(json.dumps(item) + '\n', encoding='utf-8')
        items.append(item)
    files = {name: hash_file(tmp_path / name) for name in ('fixtures.json', 'train.jsonl', 'val.jsonl')}
    manifest = {'schema_version': 'tooluse_v2', 'environment_id': env.identifier,
                'fixture_hash': env.digest, 'files': files}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    scoped, tasks = load_data(path, 'train')
    assert scoped.digest == env.digest
    view = scoped.for_task(tasks[0].task_id)
    hidden = env.records[1]
    assert view.call(env.identifier, 'lookup_record', {'record_id': hidden['id']})['ok'] is False
    assert view.call(env.identifier, 'query_database', {})['data']['count'] == 1
    assert view.call(env.identifier, 'search_docs', {'query': hidden['product']})['data']['count'] == 0
    gold = iter(items[0]['gold_calls'])
    def policy(messages):
        call = next(gold)
        return Generation(json.dumps({'name': call['tool'], 'arguments': call['arguments']}), 1)
    assert evaluate(tasks, scoped, policy)[1]['micro_verified_success_v2'] == 1


def test_cached_tokenizer_oversize_detected_before_model_generation():
    transformers = pytest.importorskip('transformers')
    try:
        tok = transformers.AutoTokenizer.from_pretrained('CohereLabs/tiny-aya-global',
            revision='af89d219b53ed9b13b8a4645f9c8028973510324', local_files_only=True)
    except OSError:
        pytest.skip('pinned Tiny Aya tokenizer not cached')
    from types import SimpleNamespace
    from actproto.agent.inference_v2 import ContextLimitExceeded
    class NeverGenerate:
        config = SimpleNamespace(max_position_embeddings=8192)
        def generate(self, **kwargs):
            raise AssertionError('generation should not run')
    env = build_environment(1800)
    output = env.call(env.identifier, 'query_database', {})
    messages = [{'role': 'user', 'content': 'Sales?'},
                {'role': 'assistant', 'content': '<tool_call>{"name":"query_database","arguments":{}}</tool_call>'},
                {'role': 'tool', 'content': json.dumps(output)}]
    with pytest.raises(ContextLimitExceeded) as exc:
        ModelPolicy(NeverGenerate(), tok)(messages)
    assert exc.value.tokens > 8192 and exc.value.limit == 8192 - 256
