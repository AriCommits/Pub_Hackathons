import json
import os
import subprocess
import sys
from dataclasses import replace

import pytest

from actproto.experiments.contracts import ExecutionTrace, PolicyInput, TaskSpec, VerificationResult
from actproto.experiments.manifest import GpuLease, RunManifest, hash_file, hash_object, prepare_run, snapshot_files


def manifest():
    return RunManifest("r1", *[hash_object(i) for i in range(6)], code_revision="abc123")


def task():
    return TaskSpec("case-secret", "group-secret", "train", "public question", "required_tools",
                    {"answer": "secret"}, [{"tool": "calculator", "args": {"x": 4}}],
                    {"type": "exact"}, 4, "env", hash_object("fixtures"))


def test_contract_round_trip_and_public_allowlist():
    original = task()
    assert TaskSpec.from_dict(original.to_dict()) == original
    public = original.policy_input([{"name": "calculator"}]).to_dict()
    assert PolicyInput.from_dict(public).to_dict() == public
    assert "secret" not in json.dumps(public)
    assert "split" not in public and "tool_dependencies" not in public
    with pytest.raises(ValueError, match="unsupported"):
        TaskSpec.from_dict(dict(original.to_dict(), schema_version="future"))
    with pytest.raises(TypeError):
        PolicyInput.from_dict(dict(public, oracle={}))


def test_trace_and_result_round_trip():
    trace = ExecutionTrace("id", [], "finish", "harness", [{"start": 1, "end": 2}], 1, 1)
    assert ExecutionTrace.from_dict(trace.to_dict()) == trace
    result = VerificationResult(True, False, False, ["missing calls"])
    assert VerificationResult.from_dict(result.to_dict()) == result
    with pytest.raises(ValueError):
        VerificationResult(True, False, True)


def test_immutable_run_and_resume(tmp_path):
    value = manifest()
    path, token = prepare_run(tmp_path, value)
    with pytest.raises(FileExistsError):
        prepare_run(tmp_path, value)
    with pytest.raises(ValueError, match="token"):
        prepare_run(tmp_path, value, "wrong")
    with pytest.raises(ValueError, match="manifest"):
        prepare_run(tmp_path, replace(value, data_hash=hash_object("other")), token)
    assert prepare_run(tmp_path, value, token) == (path, token)
    for field in ("fixture_hash", "template_hash", "verifier_hash"):
        assert replace(value, **{field: hash_object(field)}).digest != value.digest
    with pytest.raises(ValueError):
        replace(value, run_id="../escape")


def test_snapshot(tmp_path):
    p = tmp_path / "large-weights.bin"
    p.write_bytes(b"contents")
    assert snapshot_files([p]) == {str(p): hash_file(p)}


def test_gpu_lease_blocks_other_process_and_releases(tmp_path):
    code = ("from actproto.experiments.manifest import GpuLease; "
            "import sys; "
            "GpuLease(sys.argv[1], 'GPU-abc').acquire()")
    with GpuLease(tmp_path, "GPU-abc") as lease:
        lease.heartbeat()
        meta = json.loads((tmp_path / "GPU-abc.json").read_text())
        assert meta["pid"] == os.getpid() and meta["process_start"]
        result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True)
        assert result.returncode != 0 and "GPU lease held" in result.stderr
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr



def test_legacy_adapters_preserve_data_without_certifying_provenance():
    from actproto.agent.harness import ToolResult
    from actproto.agent.trajectory import EventLog
    from actproto.data.types import Oracle, Task
    from actproto.experiments.contracts import task_from_legacy, trace_from_legacy

    legacy = Task("old", "easy", "family", ["entity"], "question", ["calculator"],
                  [{"expression": "2+2"}], Oracle({"answer": "4"}, "4"), "train")
    spec = task_from_legacy(legacy, group_id="g", mode="required_tools", normalization={"type": "exact"},
                            call_budget=4, environment_id="v2", fixture_hash=hash_object("fixture"),
                            tool_dependencies=[{"tool": "calculator"}])
    assert spec.oracle["answer"] == "4" and spec.oracle["final_state"] == {"answer": "4"}
    assert "4" not in json.dumps(spec.policy_input([]).to_dict())
    legacy.split = None
    with pytest.raises(ValueError, match="split"):
        task_from_legacy(legacy, group_id="g", mode="outcome", normalization={}, call_budget=4,
                         environment_id="v2", fixture_hash=hash_object("fixture"), tool_dependencies=[])
    log = EventLog("question", messages=[{"role": "user", "content": "question"}],
                   tool_results=[ToolResult("calculator", {"expression": "2+2"}, {"ok": True})],
                   final_state={"answer": "4", "done": True}, final_answer="4", steps=1)
    trace = trace_from_legacy(log, task_id="old", completion_reason="finish")
    assert trace.provenance == "legacy_untrusted"
    assert trace.final_answer == "4" and trace.final_state == log.final_state
    assert trace.legacy_messages == log.messages and trace.legacy_steps == 1
    assert trace.events[0]["arguments"] == {"expression": "2+2"}
    assert ExecutionTrace.from_dict(trace.to_dict()) == trace
    log.final_state["answer"] = "changed"
    assert trace.final_state["answer"] == "4"
