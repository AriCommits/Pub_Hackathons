from copy import deepcopy
from dataclasses import replace
import importlib.metadata
import json

import pytest

from actproto.agent.harness import DictToolHarness
from actproto.evaluation.trace_adapter import execute_event, build_trace
from actproto.evaluation.verification_v2 import verify
from actproto.experiments.contracts import TaskSpec
from actproto.integrations.auditkit_v2 import evaluate_episode, write_evaluation_manifest


def case(mode="required_tools"):
    deps = [{"id": "calc", "tool": "calculator", "arguments": {"expression": "2+2"}}]
    return TaskSpec("t", "g", "train", "Compute 2+2", mode, {"answer": "4"}, deps,
                    {"type": "exact"}, 6, "env", "fixture")


def episode(*calls):
    harness = DictToolHarness()
    events = [execute_event(harness, tool, arguments) for tool, arguments in calls]
    return build_trace("t", events, "finish")


def specimens():
    finish = ("finish", {"answer": "4"})
    success = episode(("calculator", {"expression": "2+2"}), finish)
    fail = episode(("calculator", {"expression": "2+2"}), ("finish", {"answer": "5"}))
    alternate = episode(("calculator", {"expression": "1+3"}), finish)
    forged = deepcopy(success)
    forged.events[0]["source"] = "model"
    shortcut = episode(finish)
    return [(case(), success), (case(), fail), (case("outcome"), alternate),
            (case(), forged), (case(), shortcut)]


def test_local_backend_and_manifest(tmp_path):
    task, trace = specimens()[0]
    result = evaluate_episode(task, trace, backend="local")
    assert result["verified_success_v2"]
    out = tmp_path / "manifest.json"
    write_evaluation_manifest(out, [result])
    assert json.loads(out.read_text())["evaluations"][0]["backend"] == "local"
    with pytest.raises(FileExistsError):
        write_evaluation_manifest(out, [result])
    with pytest.raises(ValueError):
        evaluate_episode(task, trace, backend="unknown")


def test_real_auditkit_recorded_custom_predicate_parity():
    try:
        importlib.metadata.distribution("auditkit")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("install pinned AuditKIT in isolated environment for submission integration")
    for task, trace in specimens():
        expected = verify(task, trace).verified_success_v2
        result = evaluate_episode(task, trace, backend="auditkit")
        assert result["verified_success_v2"] == expected
        assert result["toolkit"]["revision"] == "71316da345784ff220d5ed59337441a59ab31526"
        assert "AgentEvalRunner.run" in result["invocation"]


def test_missing_auditkit_fails_closed(monkeypatch):
    import actproto.integrations.auditkit_v2 as bridge
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(bridge.importlib.metadata, "distribution", missing)
    task, trace = specimens()[0]
    with pytest.raises(RuntimeError, match="unavailable"):
        evaluate_episode(task, trace, backend="auditkit")
    assert evaluate_episode(task, trace, backend="local")["verified_success_v2"]


def test_recorded_rescore_requires_exact_selected_split(tmp_path, monkeypatch):
    from scripts import check_toolkit_compat
    task, trace = specimens()[0]
    monkeypatch.setattr(check_toolkit_compat, "load_data", lambda manifest, split: (None, [task]))
    traces = tmp_path / "traces.jsonl"
    traces.write_text(json.dumps({"task_id": task.task_id, "trace": trace.to_dict()}) + "\n")
    manifest = tmp_path / "recorded.json"
    check_toolkit_compat.main(["--backend", "local", "--dataset-manifest", "input.json",
                               "--traces", str(traces), "--run-manifest", str(manifest)])
    assert json.loads(manifest.read_text())["evaluations"][0]["verified_success_v2"]
    traces.write_text(traces.read_text() * 2)
    with pytest.raises(ValueError, match="exactly match"):
        check_toolkit_compat.main(["--backend", "local", "--dataset-manifest", "input.json",
                                   "--traces", str(traces), "--run-manifest", str(tmp_path / "x.json")])
