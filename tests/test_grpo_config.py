from pathlib import Path

import pytest
import yaml

from actproto.training.rl_config import GrpoConfigError, load_grpo_config, resolve_output

CONFIG = Path("actproto/configs/grpo_v2.yaml")


GPU_CONFIG = Path("actproto/configs/grpo_v2_gpu.yaml")
RTX_4060_UUID = "5b885e12-623f-268c-f07c-ee33db379969"


def test_checked_in_config_is_the_named_beta_zero_cpu_run():
    config = load_grpo_config(CONFIG)
    assert config.schema_version == "grpo_v2"
    assert config.reference_policy == "beta_zero"
    assert config.beta == 0.0
    assert config.learning_rate == 5e-6
    assert config.num_generations == 2
    assert config.generation_batch_size % config.num_generations == 0
    assert config.use_cpu is True
    assert config.gradient_checkpointing is False
    public = config.public_dict()
    assert public["use_cpu"] is True
    assert public["reference_policy"] == "beta_zero"
    assert public["schema_version"] == "grpo_v2"
    assert public["use_vllm"] is False
    assert public["quantization"] is None


E2_CONFIG = Path("actproto/configs/grpo_v2_gpu_e2.yaml")


def test_e2_gpu_config_is_the_bounded_5060_run():
    config = load_grpo_config(E2_CONFIG)
    assert config.schema_version == "grpo_v2_gpu"
    assert config.reference_policy == "beta_zero"
    assert config.beta == 0.0
    assert config.use_cpu is False
    assert config.bf16 is True
    assert config.gradient_checkpointing is True
    assert config.max_steps == 20
    assert config.num_generations == 2
    assert config.per_device_train_batch_size == 1
    assert config.gradient_accumulation_steps == 2
    assert config.generation_batch_size % config.num_generations == 0
    assert config.max_completion_length == 256
    assert config.output_dir == "results/rl_v2/grpo_gpu_e2"
    assert config.train_manifest == "data/benchmarks/tooluse_v2/manifest.json"
    assert config.public_dict()["use_vllm"] is False


def test_gpu_config_loads_only_as_the_named_5060_profile():
    config = load_grpo_config(GPU_CONFIG)
    assert config.schema_version == "grpo_v2_gpu"
    assert config.reference_policy == "beta_zero"
    assert config.beta == 0.0
    assert config.use_cpu is False
    assert config.bf16 is True
    assert config.gradient_checkpointing is True
    assert config.max_steps == 1
    assert config.output_dir == "results/rl_v2/grpo_gpu_smoke"
    assert config.device_uuid == "134c34ef-c155-b001-4e3a-b9134b072c21"
    assert "RTX 5060 Ti" in config.device_name
    assert config.baseline_manifest == "docs/SFT_BASELINE_MANIFEST.json"
    assert config.train_manifest == ""
    public = config.public_dict()
    assert public["use_cpu"] is False
    assert public["schema_version"] == "grpo_v2_gpu"
    assert public["reference_policy"] == "beta_zero"


def _dump(tmp_path, raw):
    path = tmp_path / "grpo.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_cross_rewrites_between_cpu_and_gpu_profiles_fail(tmp_path):
    cpu = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cpu["use_cpu"] = False
    with pytest.raises(GrpoConfigError, match="use_cpu"):
        load_grpo_config(_dump(tmp_path, cpu))
    cpu["use_cpu"] = True
    cpu["schema_version"] = "grpo_v2_gpu"
    with pytest.raises(GrpoConfigError):
        load_grpo_config(_dump(tmp_path, cpu))

    gpu = yaml.safe_load(GPU_CONFIG.read_text(encoding="utf-8"))
    gpu["use_cpu"] = True
    with pytest.raises(GrpoConfigError, match="use_cpu"):
        load_grpo_config(_dump(tmp_path, gpu))
    gpu["use_cpu"] = False
    gpu["schema_version"] = "grpo_v2"
    with pytest.raises(GrpoConfigError):
        load_grpo_config(_dump(tmp_path, gpu))
    gpu["schema_version"] = "grpo_v2_gpu"
    gpu["device_uuid"] = RTX_4060_UUID
    gpu["device_name"] = "NVIDIA GeForce RTX 5060 Ti"
    with pytest.raises(GrpoConfigError, match="4060"):
        load_grpo_config(_dump(tmp_path, gpu))
    gpu["device_name"] = "NVIDIA GeForce RTX 4060"
    with pytest.raises(GrpoConfigError, match="5060"):
        load_grpo_config(_dump(tmp_path, gpu))
    gpu["device_uuid"] = "134c34ef-c155-b001-4e3a-b9134b072c21"
    gpu["device_name"] = "NVIDIA GeForce RTX 5060 Ti"
    gpu["beta"] = 0.04
    with pytest.raises(GrpoConfigError, match="beta"):
        load_grpo_config(_dump(tmp_path, gpu))
    gpu["beta"] = 0.0
    gpu["reference_policy"] = "sft_adapter"
    with pytest.raises(GrpoConfigError, match="reference_policy"):
        load_grpo_config(_dump(tmp_path, gpu))
    gpu["reference_policy"] = "beta_zero"
    gpu["use_vllm"] = True
    with pytest.raises(GrpoConfigError, match="use_vllm"):
        load_grpo_config(_dump(tmp_path, gpu))
    gpu["use_vllm"] = False
    gpu["extra_flag"] = True
    with pytest.raises(GrpoConfigError, match="unsupported"):
        load_grpo_config(_dump(tmp_path, gpu))


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
