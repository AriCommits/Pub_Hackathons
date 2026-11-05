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

def test_gold_traces_pass_shared_strict_verifier(dataset):
    from actproto.evaluation.verification_v2 import verify
    from actproto.experiments.contracts import TaskSpec

    env, examples, _, _ = dataset
    failures = []
    for example in examples:
        task = TaskSpec.from_dict(example["task"])
        result = verify(task, replay_gold(example, env))
        if not result.verified_success_v2:
            failures.append((task.task_id, result.reasons))
    assert not failures, failures[:10]


def test_generated_train_rows_pass_real_tokenizer_preflight(dataset, tmp_path):
    transformers = pytest.importorskip("transformers")
    yaml = pytest.importorskip("yaml")
    from pathlib import Path
    from actproto.training.sft_v2 import preflight, read_rows

    config = yaml.safe_load(Path("actproto/configs/sft_v2.yaml").read_text(encoding="utf-8"))
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config["model_id"], revision=config["revision"], local_files_only=True)
    except OSError:
        pytest.skip("pinned tokenizer not cached")
    _, examples, _, _ = dataset
    train = [e for e in examples if e["axis"] == "train"]
    samples = [next(e for e in train if e["task"]["stratum"] == name) for name in STRATA]
    samples += [next(e for e in train if e["task"]["stratum"] == "recovery_no_tool"
                     and len(e["gold_calls"]) == count) for count in (1, 3)]
    path = tmp_path / "train.jsonl"
    path.write_text("".join(json.dumps(training_row(e)) + "\n" for e in samples), encoding="utf-8")
    encoded, report = preflight(tokenizer, read_rows(path), config)
    assert report["accepted_rows"] == len(samples)
    assert report["excluded_rows"] == 0
    assert all(any(label != -100 for label in item["labels"]) for item in encoded)
