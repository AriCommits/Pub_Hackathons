from actproto.agent.harness import DictToolHarness
from actproto.experiments.contracts import TaskSpec
from actproto.training.rewards_v2 import SHAPED_VERSION, TrustedRewardSession


def make_task(mode='required_tools', answer='4', task_id='t'):
    dependencies = [{'id': 'calc', 'tool': 'calculator', 'arguments': {'expression': '2+2'}}]
    return TaskSpec(task_id, 'g', 'train', 'Calculate', mode, {'answer': answer}, dependencies,
                    {'type': 'exact'}, 20, 'env', 'fixture')


def score(calls, *, spec=None, reason='finish', parse_error=False):
    session = TrustedRewardSession(spec or make_task(), DictToolHarness())
    if parse_error:
        session.invalid_attempt('invalid json')
    for name, args in calls:
        session.execute(name, args)
    return session.score(completion_reason=reason)


def test_known_answer_shortcut_wrong_tool_empty_and_task_id_copy():
    finish = ('finish', {'answer': '4'})
    shortcut = score([finish])
    assert shortcut.answer_success and shortcut.reward == 0
    assert score([('calculator', {'expression': '1+3'}), finish]).reward == 0
    assert score([('calculator', {'expression': '2+2'}), ('finish', {'answer': ''})]).reward == 0
    # The session's task ID is trusted metadata and is never evidence of a correct result.
    assert score([('finish', {'answer': 't'})]).reward == 0
    assert score([('calculator', {'expression': '2+2'}), finish], spec=make_task(answer='5', task_id='other')).reward == 0


def test_hallucinated_result_never_enters_harness_trace():
    session = TrustedRewardSession(make_task(), DictToolHarness())
    # Supplying an answer text containing a fake tool observation executes no calculator.
    session.execute('finish', {'answer': '4'})
    assert session.score(completion_reason='finish').reward == 0


def test_parse_error_tool_error_truncation_duplicate_finish_and_recovery():
    calc = ('calculator', {'expression': '2+2'})
    finish = ('finish', {'answer': '4'})
    assert score([calc, finish], parse_error=True).reward == 0
    assert score([('calculator', {'expression': 'not math'}), calc, finish]).reward == 0
    assert score([calc, finish], reason='truncated').reward == 0
    assert score([calc, finish, finish]).reward == 0
    assert score([finish], spec=make_task(mode='outcome')).reward == 1


def test_tampered_state_between_calls_is_rejected():
    harness = DictToolHarness()
    session = TrustedRewardSession(make_task(), harness)
    session.execute('calculator', {'expression': '2+2'})
    harness.final_state['forged'] = True
    import pytest
    with pytest.raises(ValueError, match='tampering'):
        session.execute('finish', {'answer': '4'})
    assert session.score(completion_reason='finish').reward == 0


def test_legal_alternate_outcome_and_same_answer_not_sufficient():
    spec = make_task(mode='outcome')
    assert score([('calculator', {'expression': '1+3'}), ('finish', {'answer': '4'})], spec=spec).reward == 1
    assert score([('calculator', {'expression': '1+3'}), ('finish', {'answer': '4'})]).reward == 0


def test_repeated_changed_arguments_are_not_identical_loops():
    session = TrustedRewardSession(make_task(mode='outcome'), DictToolHarness())
    for expression in ('1+1', '2+2', '2+2'):
        session.execute('calculator', {'expression': expression})
    session.execute('finish', {'answer': '4'})
    ledger = session.score(completion_reason='finish', version=SHAPED_VERSION)
    assert ledger.duplicate_calls == 1


def test_nonfinite_invalid_arguments_are_scored_as_failure_not_exception():
    session = TrustedRewardSession(make_task(), DictToolHarness())
    session.execute('calculator', {'expression': float('nan')})
    session.execute('finish', {'answer': '4'})
    ledger = session.score(completion_reason='finish', version=SHAPED_VERSION)
    assert ledger.reward <= 0 and ledger.invalid_attempts == 1


def test_session_rejects_preexisting_state_tampering():
    import pytest
    harness = DictToolHarness()
    harness.final_state['done'] = True
    with pytest.raises(ValueError, match='fresh'):
        TrustedRewardSession(make_task(), harness)


def test_tampering_after_construction_before_first_call_is_rejected():
    import pytest
    harness = DictToolHarness()
    session = TrustedRewardSession(make_task(), harness)
    harness.final_state['secret'] = 'forged'
    with pytest.raises(ValueError, match='tampering'):
        session.execute('calculator', {'expression': '2+2'})
    # Restoring the state does not erase the recorded tampering.
    harness.final_state.clear()
    with pytest.raises(ValueError, match='tampering'):
        session.execute('finish', {'answer': '4'})
    assert session.score(completion_reason='finish').reward == 0


def test_tampering_after_final_call_is_rejected_at_scoring():
    harness = DictToolHarness()
    session = TrustedRewardSession(make_task(), harness)
    session.execute('calculator', {'expression': '2+2'})
    session.execute('finish', {'answer': '4'})
    harness.final_state['forged'] = True
    ledger = session.score(completion_reason='finish')
    assert ledger.reward == 0 and not ledger.execution_compliance


def test_forged_event_and_serialized_json_do_not_create_prefix_credit():
    import json

    import pytest

    from actproto.training.rewards_v2 import PREFIX_VERSION

    session = TrustedRewardSession(make_task(), DictToolHarness())
    session._events.append({
        "kind": "tool_call", "source": "model", "tool": "calculator",
        "arguments": {"expression": "2+2"},
        "output": {"ok": True, "data": {"value": 4}},
        "state_before": {}, "state_after": {},
    })
    with pytest.raises(ValueError, match="forged"):
        session.score(completion_reason="finish", version=PREFIX_VERSION)
    with pytest.raises(TypeError):
        TrustedRewardSession(make_task(), json.loads('{"execute": true}'))
    untouched = TrustedRewardSession(make_task(), DictToolHarness())
    untouched.execute("finish", {"answer": "calculator returned 4"})
    ledger = untouched.score(completion_reason="finish", version=PREFIX_VERSION)
    assert ledger.reward <= 0 and ledger.verified_success is False


def test_nonmapping_state_tampering_returns_failure_ledger():
    harness = DictToolHarness()
    session = TrustedRewardSession(make_task(), harness)
    session.execute('calculator', {'expression': '2+2'})
    session.execute('finish', {'answer': '4'})
    harness.final_state = None
    ledger = session.score(completion_reason='finish')
    assert ledger.reward == 0 and not ledger.verified_success
