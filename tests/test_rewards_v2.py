from dataclasses import replace
import math
import pytest

from actproto.agent.harness import DictToolHarness
from actproto.evaluation.trace_adapter import build_trace, execute_event
from actproto.evaluation.verification_v2 import verify
from actproto.experiments.contracts import TaskSpec
from actproto.training.rewards_v2 import (SHAPED_VERSION, SPARSE_VERSION, TrustedRewardSession, group_advantages)


def task(mode='required_tools', dependencies=None, answer='4', budget=20, oracle=None):
    deps = [{'id': 'calc', 'tool': 'calculator', 'arguments': {'expression': '2+2'}}] if dependencies is None else dependencies
    return TaskSpec('t', 'g', 'train', 'What is 2+2?', mode, oracle or {'answer': answer},
                    deps, {'type': 'exact'}, budget, 'env', 'fixture')


def collect(spec, calls, reason='finish', version=SPARSE_VERSION):
    harness = DictToolHarness()
    session = TrustedRewardSession(spec, harness)
    for name, args in calls:
        session.execute(name, args)
    return session.score(completion_reason=reason, version=version)


def test_sparse_reward_matches_verifier():
    spec = task()
    calls = [('calculator', {'expression': '2+2'}), ('finish', {'answer': '4'})]
    harness = DictToolHarness()
    events = [execute_event(harness, name, args) for name, args in calls]
    verdict = verify(spec, build_trace('t', events, 'finish'))
    ledger = collect(spec, calls)
    assert ledger.reward == float(verdict.verified_success_v2) == 1.0
    assert ledger.answer_success == verdict.answer_success
    assert ledger.execution_compliance == verdict.execution_compliance
    assert ledger.excess_calls is None and ledger.excess_omitted_reason


def test_expected_recoverable_error_is_not_invalid():
    harness = DictToolHarness()
    event = execute_event(harness, 'lookup_record', {'record_id': 'missing'})
    error_type = event['output']['error']['type']
    dep = {'id': 'lookup', 'tool': 'lookup_record', 'arguments': {'record_id': 'missing'}, 'expected_error': error_type}
    spec = task(dependencies=[dep])
    ledger = collect(spec, [('lookup_record', {'record_id': 'missing'}), ('finish', {'answer': '4'})], version=SHAPED_VERSION)
    assert ledger.verified_success and ledger.invalid_attempts == 0 and ledger.reward == 1.0


def test_bounded_shaping_and_state_aware_duplicates():
    spec = task(mode='outcome', budget=50, oracle={'answer': '4', 'valid_call_bound': 1})
    calc = ('calculator', {'expression': '2+2'})
    finish = ('finish', {'answer': '4'})
    success = collect(spec, [calc] * 10 + [finish], version=SHAPED_VERSION)
    failure = collect(spec, [calc] * 10 + [('finish', {'answer': 'wrong'})], version=SHAPED_VERSION)
    assert success.duplicate_calls == 9 and success.excess_calls == 10
    assert success.reward == pytest.approx(1 - .15 - .20)
    assert failure.reward <= 0 and success.reward > failure.reward
    changed = collect(spec, [('calculator', {'expression': '1+1'}), calc, finish], version=SHAPED_VERSION)
    assert changed.duplicate_calls == 0


def test_zero_variance_and_finite_extreme_advantages():
    assert group_advantages([2., 2.]) == ([0., 0.], {'zero_variance_groups': 1, 'groups': 1})
    advantages, diagnostic = group_advantages([1e308, -1e308])
    assert all(math.isfinite(v) for v in advantages) and advantages == [1., -1.]
    assert diagnostic['zero_variance_groups'] == 0
    for values in ([1.], [0., float('nan')], [0., float('inf')], [True, 1.]):
        with pytest.raises(ValueError):
            group_advantages(values)


def test_session_requires_trusted_runtime_harness_and_finalizes_once():
    with pytest.raises(TypeError):
        TrustedRewardSession(task(), {'trace': 'harness'})
    session = TrustedRewardSession(task(mode='outcome'), DictToolHarness())
    session.execute('finish', {'answer': '4'})
    session.score(completion_reason='finish')
    with pytest.raises(ValueError):
        session.score(completion_reason='finish')
    with pytest.raises(ValueError):
        session.execute('finish', {'answer': '4'})


def test_valid_bound_validation():
    spec = task(mode='outcome', oracle={'answer': '4', 'valid_call_bound': True})
    with pytest.raises(ValueError, match='valid_call_bound'):
        collect(spec, [('finish', {'answer': '4'})])


def test_expected_error_does_not_become_invalid_when_later_dependency_missing():
    harness = DictToolHarness()
    event = execute_event(harness, 'lookup_record', {'record_id': 'missing'})
    error_type = event['output']['error']['type']
    deps = [
        {'id': 'lookup', 'tool': 'lookup_record', 'arguments': {'record_id': 'missing'}, 'expected_error': error_type},
        {'id': 'calc', 'tool': 'calculator', 'arguments': {'expression': '2+2'}, 'after': ['lookup']},
    ]
    ledger = collect(task(dependencies=deps), [('lookup_record', {'record_id': 'missing'}), ('finish', {'answer': '4'})], version=SHAPED_VERSION)
    assert not ledger.verified_success
    assert ledger.invalid_attempts == 0
    assert ledger.reward == 0


def test_success_always_outranks_failure_across_penalty_caps():
    spec = task(mode='outcome', budget=50, oracle={'answer': '4', 'valid_call_bound': 1})
    scores = []
    failures = []
    for repeats in range(9):
        actions = [('calculator', {'expression': '2+2'})] * repeats
        scores.append(collect(spec, actions + [('finish', {'answer': '4'})], version=SHAPED_VERSION).reward)
        failures.append(collect(spec, actions + [('finish', {'answer': 'wrong'})], version=SHAPED_VERSION).reward)
    assert min(scores) > max(failures)


def test_shaped_session_wrapper_covers_success_invalid_duplicate_and_excess():
    from actproto.training.rewards_v2 import score_executed_session

    spec = task(mode='outcome', oracle={'answer': '4', 'valid_call_bound': 1})
    success = TrustedRewardSession(spec, DictToolHarness())
    success.execute('calculator', {'expression': '2+2'})
    success.execute('finish', {'answer': '4'})
    # One extra call beyond the bound, and only because the episode verified.
    assert score_executed_session(success, completion_reason='finish') == pytest.approx(1 - 0.05)

    invalid = TrustedRewardSession(spec, DictToolHarness())
    invalid.invalid_attempt('parse_error')
    assert score_executed_session(invalid, completion_reason='parse_error') == pytest.approx(-0.10)

    duplicate = TrustedRewardSession(spec, DictToolHarness())
    duplicate.execute('calculator', {'expression': '2+2'})
    duplicate.execute('calculator', {'expression': '2+2'})
    duplicate.execute('finish', {'answer': 'wrong'})
    failure = score_executed_session(duplicate, completion_reason='finish')
    assert failure == pytest.approx(-0.05)
    clean = TrustedRewardSession(spec, DictToolHarness())
    clean.execute('finish', {'answer': 'wrong'})
    clean_reward = score_executed_session(clean, completion_reason='finish')
    assert clean_reward == 0
    assert (1 - 0.05) > clean_reward
    assert group_advantages([clean_reward, clean_reward]) == ([0.0, 0.0], {'zero_variance_groups': 1, 'groups': 1})
    with pytest.raises(TypeError):
        score_executed_session({'reward': 1}, completion_reason='finish')


def test_no_tool_alias_is_a_valid_zero_penalty_outcome():
    spec = task(mode='outcome')
    ledger = collect(spec, [('no_tool', {'answer': '4'})], version=SHAPED_VERSION)
    assert ledger.verified_success and ledger.invalid_attempts == 0 and ledger.reward == 1.0
