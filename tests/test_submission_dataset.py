import json
from pathlib import Path

from scripts.curate_submission_dataset import curate


def _row(axis, prompt, answer, stratum="single_tool"):
    return {
        "axis": axis,
        "family": f"{axis}_{stratum}",
        "gold_calls": [{"tool": "finish", "arguments": {"answer": answer}, "output": {}}],
        "task": {
            "prompt": prompt,
            "split": "test" if axis in {"iid", "stress"} else axis,
            "stratum": stratum,
            "oracle": {"answer": answer},
        },
    }


def test_curator_drops_oracles_test_and_stress(tmp_path: Path):
    build = tmp_path / "build"
    build.mkdir()
    train = _row("train", "Which product is listed?", "item-train")
    val = _row("val", "How many sales?", "SECRET-VAL-ANSWER")
    stress = _row("stress", "Audit record 1.", "SECRET-STRESS")
    test = _row("iid", "Held out prompt.", "SECRET-TEST")
    (build / "train.jsonl").write_text(json.dumps(train) + "\n", encoding="utf-8")
    (build / "val.jsonl").write_text(
        json.dumps(val) + "\n" + json.dumps(stress) + "\n" + json.dumps(test) + "\n",
        encoding="utf-8",
    )
    (build / "train_messages.jsonl").write_text(
        json.dumps({
            "split": "train",
            "messages": [
                {"role": "user", "content": "Which product is listed?"},
                {"role": "assistant", "content": "<tool_call>{\"name\": \"finish\", \"arguments\": {\"answer\": \"supervised\"}}</tool_call>"},
            ],
        }) + "\n" + json.dumps({"split": "test", "oracle": {"answer": "SECRET-TEST"}, "messages": [{"role": "user", "content": "nope"}]}) + "\n",
        encoding="utf-8",
    )
    (build / "manifest.json").write_text(
        json.dumps({"seed": 42, "environment_id": "tooluse_v2.fixtures.1", "target_counts": {"train": 1, "val": 1, "iid": 1, "stress": 1}}),
        encoding="utf-8",
    )

    out = tmp_path / "out"
    manifest = curate(build, out)
    blob = "\n".join(path.read_text(encoding="utf-8") for path in out.glob("*.jsonl"))
    assert "SECRET-VAL-ANSWER" not in blob
    assert "SECRET-STRESS" not in blob
    assert "SECRET-TEST" not in blob
    assert '"oracle"' not in blob
    assert '"gold_calls"' not in blob
    assert "stress" not in blob
    assert '"split": "test"' not in blob
    assert manifest["train_messages"] == 1
    assert manifest["val_prompts"] == 1
    assert manifest["withheld"] == ["test", "stress", "oracles", "fixtures", "traces"]
    val_line = json.loads((tmp_path / "out" / "val_prompts.jsonl").read_text(encoding="utf-8"))
    assert val_line == {"prompt": "How many sales?", "task_type": "single_tool"}
