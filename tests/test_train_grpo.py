import json
from pathlib import Path

import pytest

from actproto.experiments.contracts import TaskSpec
from actproto.training.grpo import build_grpo_kwargs, try_create_trainer
from actproto.training.grpo_run import completed_marker, public_prompt_rows, run_collect_only, run_plan, run_smoke, run_train
from actproto.training.rl_config import GrpoConfigError, load_grpo_config
from scripts.train_grpo import main

CONFIG = Path("actproto/configs/grpo_v2.yaml")


def test_plan_only_does_not_optimize_or_write_a_completed_marker(tmp_path):
    config = load_grpo_config(CONFIG)
    payload = run_plan(config, Path("."))
    assert payload["optimizes"] is False
    assert payload["completed_marker"] is False
    assert payload["baseline_present"] is False
    assert not (tmp_path / "completed.json").exists()
    assert main(["--plan-only", "--config", str(CONFIG)]) == 0


def test_old_metadata_helper_is_no_longer_a_successful_run():
    with pytest.raises(GrpoConfigError, match="not a training run"):
        build_grpo_kwargs("model", dataset=[], tools=[])
    with pytest.raises(GrpoConfigError, match="does not train"):
        try_create_trainer(train_dataset=[])


def test_missing_baseline_and_empty_prompts_fail(tmp_path):
    config = load_grpo_config(CONFIG)
    with pytest.raises(GrpoConfigError, match="S7"):
        run_train(config, Path("."))
    assert main(["--train", "--config", str(CONFIG)]) == 2
    with pytest.raises(GrpoConfigError, match="nonempty"):
        public_prompt_rows([])
    secret = TaskSpec("t", "g", "train", "What is 1+1?", "outcome", {"answer": "private-oracle"}, [],
                      {"type": "exact"}, 2, "env", "hash")
    rows = public_prompt_rows([secret])
    assert "private-oracle" not in json.dumps(rows)
    assert main(["--collect-only", "--config", str(CONFIG)]) == 2
    with pytest.raises(GrpoConfigError, match="does not optimize"):
        run_collect_only(config, tmp_path)


def test_unfrozen_manifest_cannot_start_training(tmp_path):
    config = load_grpo_config(CONFIG)
    manifest = tmp_path / "docs"
    manifest.mkdir()
    (manifest / "SFT_BASELINE_MANIFEST.json").write_text(json.dumps({"frozen": False}), encoding="utf-8")
    edited = config.__class__(**{**config.__dict__, "baseline_manifest": "docs/SFT_BASELINE_MANIFEST.json"})
    with pytest.raises(GrpoConfigError, match="frozen"):
        run_train(edited, tmp_path)


@pytest.mark.skipif(not __import__("importlib").util.find_spec("agenttune"), reason="AgentTune is installed in the isolated RL environment")
def test_cpu_smoke_updates_only_lora_weights_and_resume_keeps_the_first_checkpoint(tmp_path):
    config = load_grpo_config(CONFIG)
    output = tmp_path / "results" / "rl_v2" / "smoke"
    first = run_smoke(config, tmp_path, output, max_steps=1)
    assert first["global_step"] == 1
    assert first["frozen_changed"] == 0
    assert first["trainable_changed"] > 0
    assert first["baseline_loaded"] is False
    marker = completed_marker(output)
    assert json.loads(marker.read_text(encoding="utf-8"))["mode"] == "smoke"
    checkpoint = output / "checkpoint-1"
    before = {path.relative_to(checkpoint): path.read_bytes() for path in checkpoint.rglob("*") if path.is_file()}
    resumed = run_resume(config, tmp_path, output, checkpoint, max_steps=2)
    assert resumed["global_step"] == 2
    assert resumed["preserved_checkpoint_files"] == len(before)
    for relative, data in before.items():
        assert (checkpoint / relative).read_bytes() == data
