from pathlib import Path

from actproto.configs.trial_schema import load_trial, merge_run_paths


def test_default_trial_paths_and_runs():
    trial = load_trial()
    root = trial.root
    assert (root / "actproto" / "configs" / "trial.yaml").exists()
    assert trial.eval.split == "test"
    assert set(trial.runs) >= {"B0", "B1", "B2", "B2d1"}
    assert trial.resolve(trial.paths.reports_dir) == (root / "results" / "b0_b2").resolve()
    assert trial.run_spec("B2").router is True
    b2_adapter, protos, thresh = merge_run_paths(trial, trial.run_spec("B2"))
    assert b2_adapter == "results/adapters/b1"
    assert protos.endswith("prototypes.npz")
    assert thresh.endswith("prototype_metrics.json")
    b2d1 = trial.run_spec("B2d1")
    assert b2d1.adapter == "results/adapters/b1"
    assert b2d1.router is True
    assert b2d1.max_agent_steps == 2
    assert b2d1.replicas == 1
    assert b2d1.route_first_hop_only is True
    for run_id in ("B0", "B1", "B2"):
        spec = trial.run_spec(run_id)
        assert spec.route_first_hop_only is False
        assert spec.max_agent_steps is None
    b2d1r = trial.run_spec("B2d1r")
    assert b2d1r.route_first_hop_only is True
    assert b2d1r.max_agent_steps == 2
    assert b2d1r.replicas == 1
    assert b2d1r.adapter == "results/adapters/b1"
    log_run = trial.run_spec("B2d1log")
    assert log_run.route_first_hop_only is True
    assert log_run.max_agent_steps == 2
    assert log_run.replicas == 1
    assert log_run.adapter == "results/adapters/b1"
    assistant = trial.run_spec("B2d1a")
    assert assistant.adapter == "results/adapters/b1_assistant_only"
    assert assistant.route_first_hop_only is True
    assert assistant.max_agent_steps == 2
    assert assistant.replicas == 1
    chain = trial.run_spec("B2chain")
    assert chain.route_first_hop_only is False
    assert chain.max_agent_steps == 4
    assert chain.replicas == 1
    assert chain.router is True
    assert chain.adapter == "results/adapters/b1"


def test_hypothesis_defaults():
    trial = load_trial()
    assert trial.hypothesis.correctness_drop_pp == 3.0
    assert trial.hypothesis.latency_ratio == 1.5
