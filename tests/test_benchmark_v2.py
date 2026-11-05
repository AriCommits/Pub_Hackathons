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


def test_explicit_subset_exclusions_are_enforced(dataset):
    from actproto.evaluation.verification_v2 import verify
    from actproto.experiments.contracts import TaskSpec

    env, examples, _, _ = dataset
    args_by_tool = {"calculator": {"expression": "1 + 1"},
                    "search_docs": {"query": "item0000"},
                    "query_database": {"product": "item0000"}}
    for forbidden in args_by_tool:
        item = next(e for e in examples if forbidden in e["task"]["oracle"]["forbidden_tools"]
                    and len(e["gold_calls"]) < e["task"]["call_budget"])
        task = TaskSpec.from_dict(item["task"])
        trace = replay_gold(item, env)
        args = args_by_tool[forbidden]
        trace.events.insert(-1, {"kind": "tool_call", "tool": forbidden, "arguments": args,
                                 "output": env.call(env.identifier, forbidden, args),
                                 "source": "harness", "state_before": {}, "state_after": {}})
        result = verify(task, trace)
        assert result.answer_success and not result.execution_compliance
        assert "forbidden tool called" in result.reasons


def test_outcome_tasks_allow_equivalent_query_and_no_tool_tasks_reject_tools(dataset):
    from actproto.evaluation.verification_v2 import verify
    from actproto.experiments.contracts import TaskSpec

    env, examples, _, _ = dataset
    item = next(e for e in examples if e["task"]["stratum"] == "single_tool"
                and e["gold_calls"][0]["tool"] == "query_database")
    task = TaskSpec.from_dict(item["task"])
    trace = replay_gold(item, env)
    event = trace.events[0]
    args = {"product": event["arguments"]["product"]}
    event["arguments"] = args
    event["output"] = env.call(env.identifier, "query_database", args)
    assert len(event["output"]["data"]["rows"]) == 1
    assert task.mode == "outcome" and verify(task, trace).verified_success_v2

    no_tool = next(e for e in examples if e["task"]["stratum"] == "recovery_no_tool"
                   and len(e["gold_calls"]) == 1)
    task = TaskSpec.from_dict(no_tool["task"])
    trace = replay_gold(no_tool, env)
    args = {"expression": "1 + 1"}
    trace.events.insert(0, {"kind": "tool_call", "tool": "calculator", "arguments": args,
                            "output": env.call(env.identifier, "calculator", args),
                            "source": "harness", "state_before": {}, "state_after": {}})
    result = verify(task, trace)
    assert result.answer_success and not result.execution_compliance
    assert "forbidden tool called" in result.reasons


def test_fixture_identifiers_and_values_use_independent_seeded_pools(dataset):
    from actproto.data.fixtures_v2 import build_environment

    env, _, _, _ = dataset
    other = build_environment(len(env.records), seed=43)
    assert env.digest == build_environment(len(env.records), seed=42).digest
    assert env.digest != other.digest
    record_ids = [int(r["id"].removeprefix("V2-")) for r in env.records]
    product_ids = [int(r["product"].removeprefix("item")) for r in env.records]
    doc_ids = [int(d["id"].removeprefix("doc-")) for d in env.documents]
    assert len(set(record_ids)) == len(set(product_ids)) == len(set(doc_ids)) == len(record_ids)
    assert set(record_ids).isdisjoint(product_ids)
    assert set(record_ids).isdisjoint(doc_ids)
    assert set(product_ids).isdisjoint(doc_ids)
    # An index or identifier must not yield the sales value by the former formula.
    assert sum(row["sales"] == 1000 + ((i * 137) % 9000) for i, row in enumerate(env.sales)) < 5


def test_explicit_order_cannot_be_reversed(dataset):
    from actproto.evaluation.verification_v2 import verify
    from actproto.experiments.contracts import TaskSpec

    env, examples, _, _ = dataset
    item = next(e for e in examples if e["task"]["stratum"] == "ordered_subset")
    task = TaskSpec.from_dict(item["task"])
    trace = replay_gold(item, env)
    assert verify(task, trace).verified_success_v2
    trace.events[0], trace.events[1] = trace.events[1], trace.events[0]
    result = verify(task, trace)
    assert not result.execution_compliance
    assert "required calls or dependencies unsatisfied" in result.reasons
