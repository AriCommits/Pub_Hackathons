from copy import deepcopy

from actproto.evaluation.metrics_v2 import episode_metrics
from actproto.evaluation.verification_v2 import verify
from test_verification_v2 import run, spec


def test_counts_repeat_calls_and_model_tokens_without_inventing_bound():
    calc = ("calculator", {"expression": "2+2"})
    trace = run(calc, calc, ("finish", {"answer": "4"}))
    trace.generated_tokens, trace.model_invocations = 21, 3
    values = episode_metrics(spec(), trace, verify(spec(), trace))
    assert values["exact_duplicate_calls"] == 1
    assert values["model_invocations"] == 3 and values["generated_tokens"] == 21
    assert values["excess_calls"] is None
    assert episode_metrics(spec(), trace, verify(spec(), trace), minimum_calls=2)["excess_calls"] == 1


def test_state_change_makes_repeat_useful_and_cycle_is_separate():
    a = ("calculator", {"expression": "2+2"})
    b = ("calculator", {"expression": "3+3"})
    trace = run(a, b, a, b, ("finish", {"answer": "4"}))
    values = episode_metrics(spec(), trace, verify(spec(), trace))
    assert values["cycles"] == 1 and values["exact_duplicate_calls"] == 2
    altered = deepcopy(trace)
    altered.events[2]["state_before"] = {"revision": 2}
    assert episode_metrics(spec(), altered, verify(spec(), altered))["exact_duplicate_calls"] == 1
