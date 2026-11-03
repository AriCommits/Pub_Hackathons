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


def test_replica_limit_caps_detected_plan(monkeypatch):
    payload = '{"plan": ["1", "1"], "gpus": []}\n'
    monkeypatch.setattr(
        "subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout=payload),
    )

    assert live_runner._replica_indices(1) == ["1"]
