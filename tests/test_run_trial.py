from pathlib import Path
from types import SimpleNamespace

from actproto.configs.schema import ExperimentConfig
from actproto.configs.trial_schema import RunSpec, TrialConfig
from actproto.evaluation import live_runner
from scripts import run_trial


class StubTrial:
    def report_dir(self, run_id: str) -> Path:
        return Path("results") / run_id


def test_b2d1_cli_runs_only_depth1(monkeypatch):
    calls = []
    monkeypatch.setattr(run_trial, "trial_from_args", lambda args: StubTrial())
    monkeypatch.setattr(
        run_trial,
        "run_live_trial",
        lambda run_id, trial: calls.append(run_id)
        or {"n_tasks": 104, "summary": "ok"},
    )

    assert run_trial.main(["--run", "B2d1"]) == 0
    assert calls == ["B2d1"]


def test_all_cli_still_runs_only_baselines(monkeypatch):
    calls = []
    monkeypatch.setattr(run_trial, "trial_from_args", lambda args: StubTrial())
    monkeypatch.setattr(
        run_trial,
        "run_live_trial",
        lambda run_id, trial: calls.append(run_id)
        or {"n_tasks": 104, "summary": "ok"},
    )

    assert run_trial.main(["--run", "all"]) == 0
    assert calls == ["B0", "B1", "B2"]


def test_run_step_limit_is_applied_to_live_project(monkeypatch):
    trial = TrialConfig(
        runs={"limited": RunSpec(max_agent_steps=2)},
        _root=Path.cwd(),
    )
    captured = {}
    monkeypatch.setattr(live_runner, "load_default", ExperimentConfig)

    def fake_build_project(config, *, policy):
        captured["config"] = config
        return object()

    monkeypatch.setattr(live_runner, "build_project", fake_build_project)

    live_runner.build_live_project("limited", trial, object())

    assert captured["config"].max_agent_steps == 2


def test_b2d1_routes_only_the_first_hop(monkeypatch, tmp_path):
    prototypes_path = tmp_path / "prototypes.npz"
    prototypes_path.touch()
    trial = TrialConfig(
        runs={"B2d1": RunSpec(router=True, max_agent_steps=2, route_first_hop_only=True)},
        _root=tmp_path,
    )
    captured = {}
    monkeypatch.setattr(live_runner, "load_default", ExperimentConfig)
    monkeypatch.setattr(
        live_runner,
        "merge_run_paths",
        lambda _trial, _spec: (None, "prototypes.npz", None),
    )
    monkeypatch.setattr(live_runner, "load_prototypes_npz", lambda _path: {})
    monkeypatch.setattr(
        live_runner,
        "make_vector_fn",
        lambda _loaded, _config: lambda _state: "routed",
    )

    def fake_build_routed_project(_config, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(live_runner, "build_routed_project", fake_build_routed_project)

    live_runner.build_live_project("B2d1", trial, object())

    vector_fn = captured["vector_fn"]
    assert vector_fn(SimpleNamespace(step=0)) == "routed"
    assert vector_fn(SimpleNamespace(step=1)) is None


def test_b2chain_keeps_routing_after_step_zero(monkeypatch, tmp_path):
    prototypes_path = tmp_path / "prototypes.npz"
    prototypes_path.touch()
    trial = TrialConfig(
        runs={"B2chain": RunSpec(router=True, max_agent_steps=4, route_first_hop_only=False)},
        _root=tmp_path,
    )
    captured = {}
    monkeypatch.setattr(live_runner, "load_default", ExperimentConfig)
    monkeypatch.setattr(
        live_runner,
        "merge_run_paths",
        lambda _trial, _spec: (None, "prototypes.npz", None),
    )
    monkeypatch.setattr(live_runner, "load_prototypes_npz", lambda _path: {})
    monkeypatch.setattr(
        live_runner,
        "make_vector_fn",
        lambda _loaded, _config: lambda _state: "routed",
    )

    def fake_build_routed_project(config, **kwargs):
        captured["project_config"] = config
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(live_runner, "build_routed_project", fake_build_routed_project)

    live_runner.build_live_project("B2chain", trial, object())

    vector_fn = captured["vector_fn"]
    assert vector_fn(SimpleNamespace(step=0)) == "routed"
    assert vector_fn(SimpleNamespace(step=1)) == "routed"
    assert captured["project_config"].max_agent_steps == 4


def test_followup_run_ids_are_cli_choices(monkeypatch):
    calls = []
    monkeypatch.setattr(run_trial, "trial_from_args", lambda args: StubTrial())
    monkeypatch.setattr(
        run_trial,
        "run_live_trial",
        lambda run_id, trial: calls.append(run_id) or {"n_tasks": 104, "summary": "ok"},
    )

    assert run_trial.main(["--run", "B2d1r"]) == 0
    assert run_trial.main(["--run", "B2chain"]) == 0
    assert calls == ["B2d1r", "B2chain"]


def test_replica_limit_caps_detected_plan(monkeypatch):
    payload = '{"plan": ["1", "1"], "gpus": []}\n'
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout=payload),
    )

    assert live_runner._replica_indices(1) == ["1"]
