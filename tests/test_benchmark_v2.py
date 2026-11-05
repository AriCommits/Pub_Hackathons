import json

import pytest

from actproto.agent.parse import parse_tool_call
from actproto.data.benchmark_v2 import QUOTAS, STRATA, build_benchmark, replay_gold
from actproto.data.curriculum_v2 import repair_mix, select_probe
from actproto.data.splits_v2 import validate_splits
from scripts.build_benchmark_v2 import training_row


@pytest.fixture(scope="module")
def dataset():
    return build_benchmark()


def test_targets_fingerprints_and_replay(dataset):
    env, examples, rejects, manifest = dataset
    assert not rejects
    assert manifest["counts"] == {axis: sum(quotas) for axis, quotas in QUOTAS.items()}
    assert len(examples) == 1800
    validate_splits(examples)
    for axis, quotas in QUOTAS.items():
        for stratum, quota in zip(STRATA, quotas):
            assert sum(e["axis"] == axis and e["task"]["stratum"] == stratum for e in examples) == quota
    for example in examples:
        trace = replay_gold(example, env)
        assert trace.final_answer == example["task"]["oracle"]["answer"]
        assert len(trace.events) <= example["task"]["call_budget"]
        assert "final answer" in example["task"]["prompt"]


def test_split_guard_and_fixture_identity(dataset):
    env, examples, _, _ = dataset
    copy = json.loads(json.dumps(examples[0]))
    copy["task"]["split"] = "val" if examples[0]["task"]["split"] != "val" else "train"
    with pytest.raises(ValueError, match="crosses splits"):
        validate_splits([examples[0], copy])
    with pytest.raises(ValueError, match="environment"):
        env.call("wrong", "lookup_record", {"record_id": "x"})


def test_training_public_data_and_repair_limits(dataset):
    _, examples, _, _ = dataset
    sample = next(e for e in examples if e["axis"] == "train")
    row = training_row(sample)
    assert row["split"] == "train" and row["messages"][-1]["role"] == "assistant"
    assert parse_tool_call(row["messages"][-1]["content"]).name == "finish"
    assert sample["task"]["task_id"] not in json.dumps(row)
    assert "oracle" not in json.dumps(row)
    train = [e for e in examples if e["axis"] == "train"]
    assert len(select_probe(examples)) == 100
    assert select_probe(examples) == select_probe(examples)
    variant = dict(train[0], verified=True)
    assert len(repair_mix(train, [variant], lambda e: e.get("verified", False))) == len(train) + 1
    with pytest.raises(ValueError):
        repair_mix(train, [dict(next(e for e in examples if e["axis"] == "val"), verified=True)], lambda e: True)

    with pytest.raises(ValueError):
        repair_mix(train, [variant], lambda e: False)
