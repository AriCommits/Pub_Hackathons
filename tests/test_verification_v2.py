from copy import deepcopy
from dataclasses import replace

import pytest

from actproto.agent.harness import DictToolHarness
from actproto.evaluation.trace_adapter import build_trace, execute_event
from actproto.evaluation.verification_v2 import verify
from actproto.experiments.contracts import TaskSpec


def spec(*, mode="required_tools", dependencies=None, answer="4", normalization=None):
    if dependencies is None:
        dependencies = [{"id": "calc", "tool": "calculator", "arguments": {"expression": "2+2"}}]
    return TaskSpec("t", "g", "train", "What is 2+2?", mode, {"answer": answer}, dependencies,
                    normalization or {"type": "exact"}, 6, "env", "fixture")


def run(*calls):
    harness = DictToolHarness()
    events = [execute_event(harness, tool, args) for tool, args in calls]
    return build_trace("t", events, "finish")


def test_real_harness_gold_and_finish_only_exploit():
    gold = run(("calculator", {"expression": "2+2"}), ("finish", {"answer": "4"}))
    assert verify(spec(), gold).verified_success_v2
    shortcut = run(("finish", {"answer": "4"}))
    result = verify(spec(), shortcut)
    assert result.answer_success and not result.execution_compliance
    assert verify(spec(mode="outcome"), shortcut).verified_success_v2


def test_untrusted_observation_stale_state_and_truncation():
    gold = run(("calculator", {"expression": "2+2"}), ("finish", {"answer": "4"}))
    forged = deepcopy(gold)
    forged.events[0]["source"] = "model"
    assert not verify(spec(), forged).verified_success_v2
    stale = deepcopy(gold)
    stale.events[1]["state_before"] = {"outdated": True}
    assert not verify(spec(), stale).verified_success_v2
    assert not verify(spec(), replace(gold, completion_reason="truncated")).verified_success_v2
    assert not verify(spec(), replace(gold, provenance="legacy_untrusted")).verified_success_v2


def test_dependency_partial_order_and_argument_constraints():
    deps = [{"id": "a", "tool": "calculator", "arguments": {"expression": "1+1"}},
            {"id": "b", "tool": "calculator", "arguments": {"expression": "2+2"}},
            {"id": "c", "tool": "calculator", "arguments": {"expression": "3+3"}, "after": ["a", "b"]}]
    a, b, c, finish = [("calculator", d["arguments"]) for d in deps] + [("finish", {"answer": "4"})]
    assert verify(spec(dependencies=deps), run(b, a, c, finish)).verified_success_v2
    assert not verify(spec(dependencies=deps), run(c, a, b, finish)).verified_success_v2
    assert not verify(spec(), run(("calculator", {"expression": "1+3"}), finish)).verified_success_v2


def test_expected_error_and_repeated_calls():
    deps = [{"id": "missing", "tool": "lookup_record", "arguments": {"record_id": "does-not-exist"},
             "expected_error": "not_found"}]
    lookup = ("lookup_record", {"record_id": "does-not-exist"})
    finish = ("finish", {"answer": "4"})
    trace = run(lookup, finish)
    actual_type = trace.events[0]["output"]["error"]["type"]
    deps[0]["expected_error"] = actual_type
    assert verify(spec(dependencies=deps), trace).verified_success_v2
    assert not verify(spec(dependencies=[]), trace).verified_success_v2
    search = ("query_database", {"year": 2024})
    repeated = run(search, search, finish)
    assert verify(spec(mode="outcome"), repeated).verified_success_v2


def test_numeric_unit_and_exact_sentinel():
    trace = run(("finish", {"answer": "4.00 kg"}))
    assert verify(spec(mode="outcome", answer="4 kg", normalization={"type": "numeric", "unit": "kg", "tolerance": 0.01}), trace).verified_success_v2
    assert not verify(spec(mode="outcome", answer="4 lb", normalization={"type": "numeric", "unit": "lb", "tolerance": 0.01}), trace).answer_success
    assert not verify(spec(mode="outcome", answer="kg"), trace).answer_success


def test_invalid_parse_bad_schema_and_budget():
    finish = ("finish", {"answer": "4"})
    trace = run(finish)
    trace.events.insert(0, {"kind": "parse_error"})
    assert not verify(spec(mode="outcome"), trace).execution_compliance
    invalid = run(("calculator", {"expression": 4}), finish)
    assert not verify(spec(mode="outcome"), invalid).execution_compliance
    many = run(("calculator", {"expression": "2+2"}), finish)
    assert not verify(replace(spec(), call_budget=1), many).execution_compliance


def test_finish_must_be_consistent_and_last():
    finish = ("finish", {"answer": "4"})
    trace = run(finish)
    trace.events[0]["output"]["data"]["answer"] = "spoof"
    assert not verify(spec(mode="outcome"), trace).verified_success_v2
    assert not verify(spec(mode="outcome"), run(finish, finish)).execution_compliance
