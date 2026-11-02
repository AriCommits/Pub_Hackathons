from actproto.training.rewards import reward, reward_from_metrics


def test_success_without_penalties():
    assert reward(oracle_success=True) == 1.0
    assert reward(oracle_success=False) == 0.0


def test_penalties_on_invalid_and_repeats():
    r = reward(oracle_success=True, invalid_calls=2, repeats=2)
    assert r == 1.0 - 1.0 - 0.5


def test_from_metrics():
    r = reward_from_metrics({"task_success": True, "tool_call_validity": 1.0, "step_count": 2, "repeated_calls": 0})
    assert r == 1.0
