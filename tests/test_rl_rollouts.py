import json
from dataclasses import replace

import pytest

from actproto.experiments.contracts import ExecutionTrace, TaskSpec
from actproto.evaluation.verification_v2 import verify
from actproto.training.rl_rollouts import PolicySample, append_new_rollouts, collect_episode, collect_group


class CharacterTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        text = ''.join(f'<{m["role"]}>{m["content"]}</{m["role"]}>' for m in messages)
        if add_generation_prompt:
            text += '<assistant>'
        return list(map(ord, text))


class StatefulEnvironment:
    identifier = 'fixture'
    digest = 'digest'
    def __init__(self):
        self.calls = 0
    def call(self, environment_id, name, arguments):
        self.calls += 1
        if name == 'calculator':
            return {'ok': True, 'data': {'value': self.calls}}
        if name in ('finish', 'no_tool'):
            return {'ok': True, 'data': {'answer': arguments['answer'], 'done': True}}
        return {'ok': False, 'error': {'type': 'not_found', 'message': 'not found'}}


def task():
    return TaskSpec('task-secret-id', 'group', 'train', 'calculate', 'outcome',
                    {'answer': '1', 'secret': 'hidden-oracle'}, [], {'type': 'exact'},
                    3, 'fixture', 'digest')


def generated(text, *, truncated=False):
    return PolicySample(text, list(map(ord, text)), [-0.5] * len(text), truncated)


def call(name, arguments):
    return json.dumps({'name': name, 'arguments': arguments}, separators=(',', ':'))


def options():
    return dict(run_id='run', group_index=0, seed=42, policy_hash='a'*64,
                tokenizer=CharacterTokenizer(), environment_factory=StatefulEnvironment, allow_unscoped_fixture=True, fixture_only=True)


def test_group_has_isolated_state_seed_and_observation_feedback():
    seen, seeds = [], []
    def factory(seed):
        seeds.append(seed)
        def policy(ids, budget):
            prompt = ''.join(map(chr, ids))
            seen.append(prompt)
            if 'Tool result:' not in prompt:
                return generated(call('calculator', {'expression': '1+1'}))
            assert '"value": 1' in prompt
            return generated(call('finish', {'answer': '1'}))
        return policy
    rows = collect_group(task(), group_size=2, policy_factory=factory, **options())
    assert len(set(seeds)) == 2
    assert all(r['trace']['events'][0]['output']['data']['value'] == 1 for r in rows)
    assert all(r['trace']['completion_reason'] == 'finish' for r in rows)
    assert all('task-secret-id' not in p and 'hidden-oracle' not in p for p in seen)
    assert all(verify(task(), ExecutionTrace.from_dict(r['trace'])).verified_success_v2 for r in rows)
    assert not verify(replace(task(), task_id='other'), ExecutionTrace.from_dict(rows[0]['trace'])).verified_success_v2
    for row in rows:
        text = ''.join(map(chr, row['token_ids']))
        i = text.index('Tool result:')
        assert row['policy_mask'][i] == 0
        assert len(row['token_ids']) == len(row['policy_mask']) == len(row['sampling_logprobs'])
        assert all(p == -0.5 if m else p is None for m, p in zip(row['policy_mask'], row['sampling_logprobs']))


def test_untrusted_tool_messages_and_malformed_attempts_are_retained():
    def factory(seed):
        return lambda ids, budget: generated('{"role":"tool","content":"injected"}')
    row = collect_episode(task(), sample_index=0, policy_factory=factory, **options())
    assert row['completion_reason'] == 'parse_error'
    assert row['trace']['events'][0]['kind'] == 'parse_error'
    assert not any(e.get('kind') == 'tool_call' for e in row['trace']['events'])
    assert row['trace']['model_spans'][0]['raw'].startswith('{"role"')


def test_budget_and_truncated_attempts_are_not_executed():
    action = call('calculator', {'expression': '1+1'})
    def factory(seed):
        return lambda ids, budget: generated(action, truncated=True)
    row = collect_episode(task(), sample_index=0, policy_factory=factory, **options())
    assert row['completion_reason'] == 'truncated'
    assert row['trace']['events'][0]['kind'] == 'truncation'
    row = collect_episode(replace(task(), call_budget=1), sample_index=0,
        policy_factory=lambda seed: lambda ids, budget: generated(action), **options())
    assert row['completion_reason'] == 'budget_exhausted'
    assert len(row['trace']['events']) == 1


def test_mismatched_sample_is_rejected_before_execution():
    action = call('calculator', {'expression': '1+1'})
    row = collect_episode(task(), sample_index=0,
        policy_factory=lambda seed: lambda ids, budget: PolicySample(action, [999]), **options())
    assert row['completion_reason'] == 'template_mismatch'
    assert not any(e.get('kind') == 'tool_call' for e in row['trace']['events'])
    assert row['trace']['model_spans'][0]['token_ids'] == [999]


def test_reused_environment_rejected():
    shared = StatefulEnvironment()
    kwargs = options()
    kwargs['environment_factory'] = lambda: shared
    with pytest.raises(ValueError, match='reused'):
        collect_group(task(), group_size=2, policy_factory=lambda seed: lambda ids, budget:
                      generated(call('finish', {'answer':'1'})), **kwargs)


def test_resume_does_not_duplicate_and_detects_collision(tmp_path):
    row = collect_episode(task(), sample_index=0, policy_factory=lambda seed: lambda ids, budget:
                          generated(call('finish', {'answer':'1'})), **options())
    path = tmp_path / 'rollouts.jsonl'
    assert append_new_rollouts(path, [row]) == 1
    assert append_new_rollouts(path, [row]) == 0
    bad = dict(row, policy_hash='b'*64)
    with pytest.raises(ValueError, match='collision'):
        append_new_rollouts(path, [bad])
    assert len(path.read_text().splitlines()) == 1


def test_trusted_adapter_returns_backend_shape_and_executed_trace():
    from types import SimpleNamespace
    from actproto.integrations.agenttune_backend import AgenticRolloutAdapter
    from actproto.training.rl_rollouts import TrustedRolloutAdapter
    records, seen = [], []
    def factory(seed, trainer):
        def policy(ids, budget):
            prompt = ''.join(map(chr, ids))
            seen.append(prompt)
            if 'Tool result:' not in prompt:
                return generated(call('calculator', {'expression': '1+1'}))
            return generated(call('finish', {'answer': '1'}))
        return policy
    adapter = TrustedRolloutAdapter(CharacterTokenizer(), [task()], factory, StatefulEnvironment,
        run_id='run', policy_hash='a'*64, record_sink=lambda row: (records.append(row) or 1), sampling_mode='synthetic_fixture', batch_event_sink=lambda event: 1, abandonment_sink=lambda event: 1)
    assert isinstance(adapter, AgenticRolloutAdapter)
    output = adapter(['calculate', 'calculate'], SimpleNamespace(state=SimpleNamespace(global_step=4)))
    assert len(records) == 2 and all(r['policy_step'] == 4 for r in records)
    assert len(set(output['rollout_ids'])) == 2
    assert all(t['events'][0]['tool'] == 'calculator' and t['events'][1]['tool'] == 'finish'
               for t in output['execution_traces'])
    assert all(len(ids) == len(mask) == len(probs) for ids, mask, probs in
               zip(output['completion_ids'], output['env_mask'], output['logprobs']))
    assert all('task-secret-id' not in prompt and 'hidden-oracle' not in prompt for prompt in seen)


def test_adapter_saves_bad_token_accounting_before_rejecting():
    from types import SimpleNamespace
    from actproto.training.rl_rollouts import TrustedRolloutAdapter
    records = []
    action = call('calculator', {'expression': '1+1'})
    adapter = TrustedRolloutAdapter(CharacterTokenizer(), [task()],
        lambda seed, trainer: lambda ids, budget: PolicySample(action, [999], [-1.0]), StatefulEnvironment,
        run_id='run', policy_hash='a'*64, record_sink=lambda row: (records.append(row) or 1), sampling_mode='synthetic_fixture', batch_event_sink=lambda event: 1, abandonment_sink=lambda event: 1)
    with pytest.raises(ValueError, match='safely'):
        adapter(['calculate', 'calculate'], SimpleNamespace(state=SimpleNamespace(global_step=0)))
    assert records[0]['completion_reason'] == 'template_mismatch'
    assert not any(event.get('kind') == 'tool_call' for event in records[0]['trace']['events'])


def test_adapter_rejects_ambiguous_task_prompts():
    from actproto.training.rl_rollouts import TrustedRolloutAdapter
    with pytest.raises(ValueError, match='unique'):
        TrustedRolloutAdapter(CharacterTokenizer(), [task(), replace(task(), task_id='other')],
            None, StatefulEnvironment, run_id='run', policy_hash='a'*64, record_sink=lambda row: None, sampling_mode='synthetic_fixture', batch_event_sink=lambda event: 1, abandonment_sink=lambda event: 1)


def test_adapter_rejects_mixed_or_partial_groups_before_sampling():
    from types import SimpleNamespace
    from actproto.training.rl_rollouts import TrustedRolloutAdapter
    other = replace(task(), task_id='other', prompt='other prompt')
    adapter = TrustedRolloutAdapter(CharacterTokenizer(), [task(), other], None, StatefulEnvironment,
        run_id='run', policy_hash='a'*64, record_sink=lambda row: None, sampling_mode='synthetic_fixture', batch_event_sink=lambda event: 1, abandonment_sink=lambda event: 1)
    trainer = SimpleNamespace(state=SimpleNamespace(global_step=0))
    with pytest.raises(ValueError, match='complete'):
        adapter(['calculate'], trainer)
    with pytest.raises(ValueError, match='share one'):
        adapter(['calculate', 'other prompt'], trainer)


def test_adapter_resume_resolves_ready_and_partial_batches():
    from types import SimpleNamespace
    from actproto.training.rl_rollouts import TrustedRolloutAdapter
    records, batches, resolutions = [], [], []
    factory = lambda seed, trainer: lambda ids, budget: generated(call('finish', {'answer':'1'}))
    args = (CharacterTokenizer(), [task()], factory, StatefulEnvironment)
    kwargs = dict(run_id='run', policy_hash='a'*64,
        record_sink=lambda row: (records.append(row) or 1), sampling_mode='synthetic_fixture',
        batch_event_sink=lambda event: (batches.append(event) or 1),
        abandonment_sink=lambda event: (resolutions.append(event) or 1))
    trainer = SimpleNamespace(state=SimpleNamespace(global_step=4))
    first = TrustedRolloutAdapter(*args, **kwargs)
    first(['calculate', 'calculate'], trainer)
    ready_records = {r['rollout_id']: r for r in records}
    ready_events = list(batches)
    resumed = TrustedRolloutAdapter(*args, **kwargs, resume_records=ready_records, resume_batch_events=ready_events)
    assert resolutions[-1]['reason'] == 'ready_consumption_unknown'
    resumed(['calculate', 'calculate'], trainer)
    assert [r['group_index'] for r in records] == [0, 0, 1, 1]
    partial = TrustedRolloutAdapter(*args, **kwargs,
        resume_records={records[0]['rollout_id']: records[0]}, resume_batch_events=[ready_events[0]])
    assert resolutions[-1]['reason'] == 'batch_not_ready'
    assert len(resolutions[-1]['rollout_ids']) == 2
    partial(['calculate', 'calculate'], trainer)
    assert [r['group_index'] for r in records[-2:]] == [1, 1]
    bad = dict(records[1], policy_step=5)
    with pytest.raises(ValueError, match='provenance'):
        TrustedRolloutAdapter(*args, **kwargs,
            resume_records={records[0]['rollout_id']: records[0], bad['rollout_id']: bad},
            resume_batch_events=ready_events)


def test_adapter_keeps_two_tasks_in_separate_groups():
    from types import SimpleNamespace
    from actproto.training.rl_rollouts import TrustedRolloutAdapter
    other = replace(task(), task_id='other', prompt='other prompt')
    records = []
    policy = lambda seed, trainer: lambda ids, budget: generated(call('finish', {'answer':'1'}))
    adapter = TrustedRolloutAdapter(CharacterTokenizer(), [task(), other], policy, StatefulEnvironment,
        run_id='run', policy_hash='a'*64, record_sink=lambda row: (records.append(row) or 1), sampling_mode='synthetic_fixture', batch_event_sink=lambda event: 1, abandonment_sink=lambda event: 1)
    output = adapter(['calculate','calculate','other prompt','other prompt'],
                     SimpleNamespace(state=SimpleNamespace(global_step=1)))
    assert output['task_ids'] == ['task-secret-id','task-secret-id','other','other']
    assert [trace['task_id'] for trace in output['execution_traces']] == output['task_ids']


def test_adapter_refuses_unpersisted_or_duplicate_record():
    from types import SimpleNamespace
    from actproto.training.rl_rollouts import TrustedRolloutAdapter
    adapter = TrustedRolloutAdapter(CharacterTokenizer(), [task()],
        lambda seed, trainer: lambda ids, budget: generated(call('finish', {'answer':'1'})),
        StatefulEnvironment, run_id='run', policy_hash='a'*64, record_sink=lambda row: 0, sampling_mode='synthetic_fixture', batch_event_sink=lambda event: 1, abandonment_sink=lambda event: 1)
    with pytest.raises(ValueError, match='did not persist'):
        adapter(['calculate','calculate'], SimpleNamespace(state=SimpleNamespace(global_step=0)))


def test_missing_sampling_logprobs_are_retained_but_rejected_by_backend():
    from types import SimpleNamespace
    from actproto.training.rl_rollouts import TrustedRolloutAdapter
    records = []
    action = call('finish', {'answer': '1'})
    adapter = TrustedRolloutAdapter(CharacterTokenizer(), [task()],
        lambda seed, trainer: lambda ids, budget: PolicySample(action, list(map(ord, action))),
        StatefulEnvironment, run_id='run', policy_hash='a'*64,
        record_sink=lambda row: (records.append(row) or 1), sampling_mode='synthetic_fixture', batch_event_sink=lambda event: 1, abandonment_sink=lambda event: 1)
    with pytest.raises(ValueError, match='safely'):
        adapter(['calculate','calculate'], SimpleNamespace(state=SimpleNamespace(global_step=0)))
    assert records[0]['completion_reason'] == 'finish'
    assert records[0]['trace']['events'][-1]['tool'] == 'finish'


def test_elapsed_budget_retains_sample_without_executing_tool(monkeypatch):
    import actproto.training.rl_rollouts as module
    times = iter([0.0, 0.0, 2.0, 2.0])
    monkeypatch.setattr(module, 'monotonic', lambda: next(times))
    row = collect_episode(task(), sample_index=0, max_seconds=1,
        policy_factory=lambda seed: lambda ids, budget:
            generated(call('calculator', {'expression':'1+1'})), **options())
    assert row['completion_reason'] == 'timeout'
    assert row['trace']['model_spans']
    assert not any(e.get('kind') == 'tool_call' for e in row['trace']['events'])


def test_recoverable_tool_error_is_recorded_and_passed_back_to_policy():
    def factory(seed):
        def policy(ids, budget):
            if 'Tool result:' not in ''.join(map(chr, ids)):
                return generated(call('lookup_record', {'record_id': 'missing'}))
            assert 'not_found' in ''.join(map(chr, ids))
            return generated(call('finish', {'answer': '1'}))
        return policy
    row = collect_episode(task(), sample_index=0, policy_factory=factory, **options())
    assert row['completion_reason'] == 'finish'
    assert row['trace']['events'][0]['output']['error']['type'] == 'not_found'
    assert row['trace']['events'][1]['tool'] == 'finish'


def test_verified_train_scope_hides_test_only_fixtures_and_isolates_backing_state(tmp_path):
    from actproto.data.benchmark_v2 import _make_example
    from actproto.data.fixtures_v2 import build_environment
    from actproto.experiments.manifest import hash_file
    from actproto.training.rl_dataset import load_scoped_train_manifest
    from actproto.training.rl_tools import new_session
    env = build_environment(2)
    fixture = {'identifier': env.identifier, 'records': env.records, 'sales': env.sales, 'documents': env.documents}
    (tmp_path / 'fixtures.json').write_text(json.dumps(fixture), encoding='utf8')
    item = _make_example(0, 'train', 'single_tool', env, env.digest)
    (tmp_path / 'train.jsonl').write_text(json.dumps(item) + '\n', encoding='utf8')
    manifest = {'schema_version':'tooluse_v2', 'environment_id':env.identifier, 'fixture_hash':env.digest,
                'files':{n:hash_file(tmp_path/n) for n in ('fixtures.json','train.jsonl')}}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest), encoding='utf8')
    scoped, tasks = load_scoped_train_manifest(tmp_path / 'manifest.json')
    first, second = new_session(tasks[0], lambda: scoped), new_session(tasks[0], lambda: scoped)
    hidden = env.records[1]
    assert first.call('lookup_record', {'record_id': hidden['id']})['ok'] is False
    assert first.call('query_database', {})['data']['count'] == 1
    assert first.call('search_docs', {'query': hidden['product']})['data']['count'] == 0
    first.environment._scoped_env.records[0]['id'] = 'tampered'
    assert second.environment._scoped_env.records[0]['id'] == env.records[0]['id']
    with pytest.raises(ValueError, match='manifest-bound'):
        new_session(replace(tasks[0], prompt='forged'), lambda: scoped)
    with pytest.raises(ValueError, match='ScopedFixtures'):
        new_session(tasks[0], lambda: env)


def test_policy_overrun_is_retained_but_never_forwarded():
    from types import SimpleNamespace
    from actproto.training.rl_rollouts import TrustedRolloutAdapter
    records = []
    action = call('calculator', {'expression': '1+1'})
    adapter = TrustedRolloutAdapter(CharacterTokenizer(), [task()],
        lambda seed, trainer: lambda ids, budget: generated(action), StatefulEnvironment,
        run_id='run', policy_hash='a'*64, record_sink=lambda row: (records.append(row) or 1),
        sampling_mode='synthetic_fixture', batch_event_sink=lambda event: 1, abandonment_sink=lambda event: 1, max_generated_tokens=3)
    with pytest.raises(ValueError, match='safely'):
        adapter(['calculate', 'calculate'], SimpleNamespace(state=SimpleNamespace(global_step=0)))
    row = records[0]
    assert row['completion_reason'] == 'policy_budget_overrun'
    assert row['trace']['model_spans'][0]['token_ids'] == list(map(ord, action))
    assert not any(row['policy_mask'])
    assert not any(event.get('kind') == 'tool_call' for event in row['trace']['events'])


def test_policy_factory_errors_and_invalid_logprobs_are_retained():
    def bad_factory(seed):
        raise RuntimeError('factory failed')
    row = collect_episode(task(), sample_index=0, policy_factory=bad_factory, **options())
    assert row['completion_reason'] == 'policy_error'
    assert 'factory failed' in row['trace']['events'][0]['error']
    action = call('finish', {'answer':'1'})
    row = collect_episode(task(), sample_index=0,
        policy_factory=lambda seed: lambda ids, budget: PolicySample(action, list(map(ord, action)), 42), **options())
    assert row['completion_reason'] == 'policy_error'
    assert 'logprobs' in row['trace']['events'][0]['error']


def test_behavior_collection_requires_hard_deadline_executor():
    from actproto.training.rl_rollouts import TrustedRolloutAdapter
    opts = options()
    opts['fixture_only'] = False
    invoked = []
    with pytest.raises(ValueError, match='hard-deadline'):
        collect_episode(task(), sample_index=0,
            policy_factory=lambda seed: invoked.append(seed), **opts)
    assert invoked == []
    with pytest.raises(ValueError, match='hard-deadline'):
        TrustedRolloutAdapter(CharacterTokenizer(), [task()], None, StatefulEnvironment,
            run_id='run', policy_hash='a'*64, record_sink=lambda row: 1)


def test_abandonment_sidecar_is_durable_idempotent_and_collision_checked(tmp_path):
    from actproto.training.rl_rollouts import append_abandoned_group
    event = {'schema_version':'rl_rollout_group_resolution_v1', 'run_id':'run', 'task_id':'t',
             'group_index':0, 'rollout_ids':['sample'], 'reason':'partial_group_not_returned_to_trainer'}
    path = tmp_path / 'abandoned.jsonl'
    assert append_abandoned_group(path, event) == 1
    assert append_abandoned_group(path, event) == 0
    assert json.loads(path.read_text(encoding='utf8')) == event
    with pytest.raises(ValueError, match='collision'):
        append_abandoned_group(path, dict(event, rollout_ids=['other']))


def test_resume_requires_batch_journal():
    from types import SimpleNamespace
    from actproto.training.rl_rollouts import TrustedRolloutAdapter
    records = []
    factory = lambda seed, trainer: lambda ids, budget: generated(call('finish', {'answer':'1'}))
    args = (CharacterTokenizer(), [task()], factory, StatefulEnvironment)
    kwargs = dict(run_id='run', policy_hash='a'*64,
        record_sink=lambda row: (records.append(row) or 1), sampling_mode='synthetic_fixture',
        batch_event_sink=lambda event: 1, abandonment_sink=lambda event: 1)
    adapter = TrustedRolloutAdapter(*args, **kwargs)
    adapter(['calculate','calculate'], SimpleNamespace(state=SimpleNamespace(global_step=0)))
    with pytest.raises(ValueError, match='batch journal'):
        TrustedRolloutAdapter(*args, **kwargs, resume_records={records[0]['rollout_id']: records[0]})


@pytest.mark.parametrize('bad_sample', [2, 3])
def test_failed_batch_resolves_every_prepared_group_and_never_reuses_ids(tmp_path, bad_sample):
    from types import SimpleNamespace
    from actproto.training.rl_rollouts import (TrustedRolloutAdapter, append_abandoned_group,
        append_batch_event, append_new_rollouts, read_batch_events, read_rollouts)
    records_path, journal_path, resolution_path = [tmp_path / name for name in ('rows.jsonl', 'batches.jsonl', 'resolution.jsonl')]
    count = [0]
    fail = [True]
    def factory(seed, trainer):
        count[0] += 1
        def policy(ids, budget):
            action = call('finish', {'answer': '1'})
            if fail[0] and count[0] == bad_sample:
                return PolicySample(action, [999], [-1.0])
            return generated(action)
        return policy
    kwargs = dict(run_id='run', policy_hash='a'*64, sampling_mode='synthetic_fixture',
        record_sink=lambda row: append_new_rollouts(records_path, [row]),
        batch_event_sink=lambda event: append_batch_event(journal_path, event),
        abandonment_sink=lambda event: append_abandoned_group(resolution_path, event))
    args = (CharacterTokenizer(), [task()], factory, StatefulEnvironment)
    adapter = TrustedRolloutAdapter(*args, **kwargs)
    trainer = SimpleNamespace(state=SimpleNamespace(global_step=0))
    with pytest.raises(ValueError, match='safely'):
        adapter(['calculate'] * 4, trainer)
    events = read_batch_events(journal_path)
    assert [event['phase'] for event in events] == ['prepare', 'aborted']
    groups = [json.loads(line) for line in resolution_path.read_text(encoding='utf8').splitlines()]
    assert len(groups) == 2 and all(g['reason'] == 'batch_not_ready' for g in groups)
    assert len(read_rollouts(records_path)) == bad_sample
    fail[0] = False
    restarted = TrustedRolloutAdapter(*args, **kwargs, resume_records=read_rollouts(records_path),
                                      resume_batch_events=events)
    restarted(['calculate', 'calculate'], trainer)
    rows = list(read_rollouts(records_path).values())
    assert [r['group_index'] for r in rows[-2:]] == [2, 2]
    assert len({r['rollout_id'] for r in rows}) == len(rows)


def test_ready_before_crash_is_consumption_unknown_and_never_replayed(tmp_path):
    from types import SimpleNamespace
    from actproto.training.rl_rollouts import (TrustedRolloutAdapter, append_abandoned_group,
        append_batch_event, append_new_rollouts, read_batch_events, read_rollouts)
    records_path, journal_path, resolution_path = [tmp_path / n for n in ('rows.jsonl','batch.jsonl','resolved.jsonl')]
    policy = lambda seed, trainer: lambda ids, budget: generated(call('finish', {'answer':'1'}))
    args = (CharacterTokenizer(), [task()], policy, StatefulEnvironment)
    kwargs = dict(run_id='run', policy_hash='a'*64, sampling_mode='synthetic_fixture',
        record_sink=lambda row: append_new_rollouts(records_path, [row]),
        batch_event_sink=lambda event: append_batch_event(journal_path, event),
        abandonment_sink=lambda event: append_abandoned_group(resolution_path, event))
    trainer = SimpleNamespace(state=SimpleNamespace(global_step=0))
    TrustedRolloutAdapter(*args, **kwargs)(['calculate', 'calculate'], trainer)
    # Treat the ready record as the last durable event before a possible crash;
    # the journal has no trainer-consumption acknowledgement.
    resumed = TrustedRolloutAdapter(*args, **kwargs, resume_records=read_rollouts(records_path),
                                    resume_batch_events=read_batch_events(journal_path))
    resolution = json.loads(resolution_path.read_text(encoding='utf8'))
    assert resolution['reason'] == 'ready_consumption_unknown'
    resumed(['calculate', 'calculate'], trainer)
    assert [r['group_index'] for r in read_rollouts(records_path).values()] == [0, 0, 1, 1]


def test_crash_after_prepare_reserves_and_resolves_all_empty_groups(tmp_path):
    from types import SimpleNamespace
    from actproto.training.rl_rollouts import (TrustedRolloutAdapter, append_abandoned_group,
        append_batch_event, append_new_rollouts, read_batch_events, read_rollouts)
    records, journal, resolution = [tmp_path / name for name in ('rows.jsonl', 'batch.jsonl', 'resolved.jsonl')]
    crash = [True]
    def batch_sink(event):
        result = append_batch_event(journal, event)
        if crash[0] and event['phase'] == 'prepare':
            raise SystemExit('simulated process death after durable prepare')
        return result
    policy = lambda seed, trainer: lambda ids, budget: generated(call('finish', {'answer':'1'}))
    args = (CharacterTokenizer(), [task()], policy, StatefulEnvironment)
    kwargs = dict(run_id='run', policy_hash='a'*64, sampling_mode='synthetic_fixture',
        record_sink=lambda row: append_new_rollouts(records, [row]), batch_event_sink=batch_sink,
        abandonment_sink=lambda event: append_abandoned_group(resolution, event))
    trainer = SimpleNamespace(state=SimpleNamespace(global_step=0))
    with pytest.raises(SystemExit):
        TrustedRolloutAdapter(*args, **kwargs)(['calculate'] * 4, trainer)
    assert len(read_batch_events(journal)) == 1
    assert read_rollouts(records) == {}
    crash[0] = False
    resumed = TrustedRolloutAdapter(*args, **kwargs, resume_records=read_rollouts(records),
                                    resume_batch_events=read_batch_events(journal))
    resolved = [json.loads(line) for line in resolution.read_text(encoding='utf8').splitlines()]
    assert [row['group_index'] for row in resolved] == [0, 1]
    assert all(row['reason'] == 'batch_not_ready' for row in resolved)
    resumed(['calculate','calculate'], trainer)
    assert [row['group_index'] for row in read_rollouts(records).values()] == [2, 2]
