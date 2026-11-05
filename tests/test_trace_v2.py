import json
from dataclasses import replace

import pytest

from actproto.agent.inference_v2 import Generation, parse_single_call
from actproto.agent.trace_v2 import rollout
from actproto.data.benchmark_v2 import _make_example
from actproto.data.fixtures_v2 import build_environment
from actproto.evaluation.verification_v2 import verify
from actproto.experiments.contracts import TaskSpec


def example():
    env = build_environment(10)
    item = _make_example(0, 'train', 'dependent', env, env.digest)
    return env, item, TaskSpec.from_dict(item['task'])


def test_real_gold_execution_and_public_messages():
    env, item, task = example()
    calls = iter(item['gold_calls'])
    observed = []
    def policy(messages):
        observed.append(messages)
        call = next(calls)
        return Generation(json.dumps({'name': call['tool'], 'arguments': call['arguments']}), 5)
    trace, _ = rollout(task, env, policy)
    assert verify(task, trace).verified_success_v2
    assert len(trace.events) == len(item['gold_calls']) and trace.model_invocations == len(item['gold_calls'])
    assert task.task_id not in json.dumps(observed) and 'oracle' not in json.dumps(observed)
    assert all('Tool result:' not in message['content'] for message in observed[0])


def test_shortcut_fails_and_bad_output_is_visible():
    env, item, task = example()
    short = lambda messages: Generation(json.dumps({'name':'finish', 'arguments':{'answer': task.oracle['answer']}}), 1)
    trace, _ = rollout(task, env, short)
    assert verify(task, trace).answer_success and not verify(task, trace).verified_success_v2
    bad, _ = rollout(task, env, lambda messages: Generation('{"garbage":true}', 2))
    assert bad.events[0]['kind'] == 'parse_error' and bad.model_spans[0]['raw'] == '{"garbage":true}'
    assert not verify(task, bad).verified_success_v2


def test_budget_multiple_calls_truncation_and_environment_binding():
    env, _, task = example()
    call = '{"name":"calculator","arguments":{"expression":"2+2"}}'
    trace, _ = rollout(replace(task, call_budget=2), env, lambda messages: Generation(call, 3))
    assert trace.completion_reason == 'budget_exhausted' and len(trace.events) == 2
    assert not verify(task, trace).verified_success_v2
    trace, _ = rollout(task, env, lambda messages: Generation(call + call, 3))
    assert trace.completion_reason == 'parse_error'
    trace, _ = rollout(task, env, lambda messages: Generation(call, 256, True))
    assert trace.completion_reason == 'truncated'
    with pytest.raises(ValueError, match='fixture'):
        rollout(replace(task, fixture_hash='wrong'), env, lambda m: Generation(call, 1))
    with pytest.raises(ValueError):
        parse_single_call('[{},{}]')


def test_deployed_retry_logs_original_failure_and_extra_generation():
    env, item, task = example()
    responses = iter([Generation('bad output', 2), Generation(json.dumps({'name': 'finish',
                       'arguments': {'answer': task.oracle['answer']}}), 5)])
    seen = []
    def policy(messages):
        seen.append(messages)
        return next(responses)
    trace, _ = rollout(task, env, policy, interventions=('retry_parse',))
    assert trace.model_invocations == 2 and trace.generated_tokens == 7
    assert trace.events[0]['kind'] == 'parse_error'
    assert trace.model_spans[0]['raw'] == 'bad output'
    assert trace.model_spans[1]['interventions'] == ['retry_parse']
    assert seen[1][-2]['content'] == 'bad output'
    assert not verify(task, trace).verified_success_v2
