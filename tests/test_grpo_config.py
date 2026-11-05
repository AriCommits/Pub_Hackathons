from pathlib import Path

import pytest
import yaml

from actproto.training.rl_config import GrpoConfigError, load_grpo_config, resolve_output

CONFIG = Path("actproto/configs/grpo_v2.yaml")


def test_checked_in_config_is_the_named_beta_zero_cpu_run():
    config = load_grpo_config(CONFIG)
    assert config.reference_policy == "beta_zero"
    assert config.beta == 0.0
    assert config.learning_rate == 5e-6
    assert config.num_generations == 2
    assert config.generation_batch_size % config.num_generations == 0
    assert config.public_dict()["use_vllm"] is False
    assert config.public_dict()["quantization"] is None


def test_config_rejects_rewritten_reference_and_unknown_keys(tmp_path):
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["beta"] = 0.02
    path = tmp_path / "grpo.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(GrpoConfigError, match="beta"):
        load_grpo_config(path)
    raw["beta"] = 0.0
    raw["reference_policy"] = "base_model"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(GrpoConfigError, match="reference_policy"):
        load_grpo_config(path)
    raw["reference_policy"] = "beta_zero"
    raw["use_vllm"] = True
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(GrpoConfigError, match="use_vllm"):
        load_grpo_config(path)
    del raw["use_vllm"]
    raw["quantization"] = "4bit"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(GrpoConfigError, match="unsupported"):
        load_grpo_config(path)


def test_config_rejects_a_group_that_does_not_divide_the_batch(tmp_path):
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["num_generations"] = 4
    raw["per_device_train_batch_size"] = 2
    raw["gradient_accumulation_steps"] = 1
    path = tmp_path / "grpo.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(GrpoConfigError, match="divisible"):
        load_grpo_config(path)


def test_output_must_stay_inside_the_rl_results_directory(tmp_path):
    root = tmp_path / "project"
    (root / "results" / "rl_v2").mkdir(parents=True)
    assert resolve_output("results/rl_v2/smoke", root).is_relative_to(root / "results" / "rl_v2")
    with pytest.raises(GrpoConfigError, match="results/rl_v2"):
        resolve_output("results/adapters/b1", root)
    with pytest.raises(GrpoConfigError, match="results/rl_v2"):
        resolve_output("results/rl_v2/../adapters/b1", root)
