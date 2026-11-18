import inspect
import json
import os
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from actproto.experiments.contracts import TaskSpec
from actproto.training.grpo import build_grpo_kwargs, try_create_trainer
from actproto.training.grpo_run import completed_marker, public_prompt_rows, run_collect_only, run_plan, run_resume, run_smoke, run_train
from actproto.training.rl_config import GrpoConfigError, load_grpo_config
from lexsi.scripts.train_grpo import main

CONFIG = Path("actproto/configs/grpo_v2.yaml")
LEXSI = Path(__file__).resolve().parents[1]


def _git_lfs_pointer(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as handle:
        return handle.read(128).startswith(b"version https://git-lfs.github.com/spec/v1")


def _directory_sha256(adapter: Path) -> str:
    digest = sha256()
    files = sorted(path for path in adapter.rglob("*") if path.is_file())
    for file in files:
        digest.update(file.relative_to(adapter).as_posix().encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def _write_baseline(root: Path, payload: dict) -> None:
    docs = root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "SFT_BASELINE_MANIFEST.json").write_text(json.dumps(payload), encoding="utf-8")


def test_plan_only_does_not_optimize_or_write_a_completed_marker(tmp_path):
    config = load_grpo_config(CONFIG)
    payload = run_plan(config, tmp_path)
    assert payload["optimizes"] is False
    assert payload["completed_marker"] is False
    assert payload["baseline_present"] is False
    assert not list(tmp_path.rglob("completed.json"))

    _write_baseline(tmp_path, {"frozen": True})
    present = run_plan(config, tmp_path)
    assert present["baseline_present"] is True
    assert present["optimizes"] is False
    assert present["completed_marker"] is False
    assert not list(tmp_path.rglob("completed.json"))

    live = run_plan(config, LEXSI)
    assert live["optimizes"] is False
    assert live["completed_marker"] is False
    marker = LEXSI / "results" / "rl_v2" / "grpo_v2" / "completed.json"
    existed = marker.exists()
    assert main(["--plan-only", "--config", str(CONFIG)]) == 0
    assert marker.exists() is existed
    assert not list(tmp_path.rglob("completed.json"))


def test_old_metadata_helper_is_no_longer_a_successful_run():
    with pytest.raises(GrpoConfigError, match="not a training run"):
        build_grpo_kwargs("model", dataset=[], tools=[])
    with pytest.raises(GrpoConfigError, match="does not train"):
        try_create_trainer(train_dataset=[])


def test_missing_baseline_and_empty_prompts_fail(tmp_path):
    config = load_grpo_config(CONFIG)
    with pytest.raises(GrpoConfigError, match="S7"):
        run_train(config, tmp_path)

    missing_root = tmp_path / "missing-adapter"
    _write_baseline(missing_root, {
        "frozen": True,
        "adapter_path": str(missing_root / "no-such-adapter"),
        "adapter_sha256": "ab" * 32,
    })
    with pytest.raises(GrpoConfigError, match="adapter path does not exist"):
        run_train(config, missing_root)

    shipped = LEXSI / "docs" / "SFT_BASELINE_MANIFEST.json"
    if shipped.is_file():
        recorded = json.loads(shipped.read_text(encoding="utf-8"))
        if not Path(recorded["adapter_path"]).is_dir():
            with pytest.raises(GrpoConfigError, match="adapter path does not exist"):
                run_train(config, LEXSI)

    from lexsi.scripts.run_experiment_v2 import adapter_digest

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "weights.bin").write_bytes(b"fixture-adapter")
    matched = tmp_path / "matched"
    _write_baseline(matched, {
        "frozen": True,
        "adapter_path": str(adapter),
        "adapter_sha256": adapter_digest(adapter),
    })
    with pytest.raises(GrpoConfigError, match="nonempty"):
        run_train(config, matched)

    mismatched = tmp_path / "mismatched"
    _write_baseline(mismatched, {
        "frozen": True,
        "adapter_path": str(adapter),
        "adapter_sha256": "cd" * 32,
    })
    with pytest.raises(GrpoConfigError, match="hash does not match"):
        run_train(config, mismatched)

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


GPU_CONFIG = Path("actproto/configs/grpo_v2_gpu.yaml")
E2_CONFIG = Path("actproto/configs/archive/grpo_v2_gpu_e2.yaml")


def test_gpu_schema_does_not_enter_cpu_fit(monkeypatch):
    monkeypatch.setattr("actproto.training.grpo_run._fit", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("entered _fit")))
    config = load_grpo_config(GPU_CONFIG)
    with pytest.raises(GrpoConfigError, match="run_gpu_smoke"):
        run_smoke(config, Path("."), Path("unused"))


def test_gpu_train_and_resume_fail_closed(tmp_path):
    config = load_grpo_config(GPU_CONFIG)
    with pytest.raises(GrpoConfigError, match="S7"):
        run_train(config, tmp_path)
    with pytest.raises(GrpoConfigError, match="resume is closed"):
        run_resume(config, tmp_path, tmp_path / "results" / "rl_v2" / "grpo_gpu_smoke", tmp_path, max_steps=2)
    assert main(["--train", "--config", str(GPU_CONFIG)]) == 2
    assert main(["--resume", str(tmp_path), "--config", str(GPU_CONFIG)]) == 2


def test_gpu_plan_only_writes_nothing(tmp_path):
    config = load_grpo_config(GPU_CONFIG)
    payload = run_plan(config, tmp_path)
    assert payload["optimizes"] is False
    assert payload["completed_marker"] is False
    assert payload["config"]["schema_version"] == "grpo_v2_gpu"
    assert payload["config"]["use_cpu"] is False
    assert main(["--plan-only", "--config", str(GPU_CONFIG)]) == 0
    assert not list(tmp_path.rglob("completed.json"))


def test_gpu_smoke_without_the_live_flag_does_not_train(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("GPU smoke started without the live gate")
    monkeypatch.setattr("scripts.train_grpo.run_gpu_smoke", fail_if_called)
    monkeypatch.setattr("scripts.train_grpo.launch_gpu_smoke", fail_if_called)
    monkeypatch.setattr("actproto.training.grpo_run._fit", fail_if_called)
    assert main(["--smoke", "--config", str(GPU_CONFIG)]) == 2


def test_parent_cuda_mask_blocks_the_live_launch(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert main(["--smoke", "--live-gpu", "--config", str(GPU_CONFIG)]) == 2
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_live_launch_puts_the_nvidia_ordinal_only_in_the_child(monkeypatch):
    import os
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr("scripts.train_grpo.assert_5060_idle", lambda: None)
    monkeypatch.setattr("scripts.train_grpo.visible_ordinal_for_5060", lambda: "0")
    seen = {}

    def fake_call(cmd, env):
        seen["cmd"] = cmd
        seen["env"] = env
        return 0

    monkeypatch.setattr("scripts.train_grpo.subprocess.call", fake_call)
    assert main(["--smoke", "--live-gpu", "--config", str(GPU_CONFIG)]) == 0
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert seen["env"]["LEXSI_RL_GPU_CHILD"] == "1"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in (None, "")
    assert "--live-gpu" in seen["cmd"]


def test_gpu_output_refuses_the_cpu_smoke_and_adapter_paths(tmp_path):
    from actproto.training.grpo_run import guard_gpu_output
    root = tmp_path / "project"
    (root / "results" / "rl_v2" / "grpo_gpu_smoke").mkdir(parents=True)
    (root / "results" / "rl_v2" / "grpo_cpu_smoke").mkdir()
    fresh = root / "results" / "rl_v2" / "grpo_gpu_smoke_other"
    # The checked-in profile writes grpo_gpu_smoke. An existing directory is refused.
    with pytest.raises(GrpoConfigError, match="already exists"):
        guard_gpu_output(root / "results" / "rl_v2" / "grpo_gpu_smoke", root)
    with pytest.raises(GrpoConfigError, match="grpo_cpu_smoke"):
        guard_gpu_output(root / "results" / "rl_v2" / "grpo_cpu_smoke", root)
    with pytest.raises(GrpoConfigError, match="adapter"):
        guard_gpu_output(root / "results" / "rl_v2" / "adapters" / "b1", root)
    assert guard_gpu_output(fresh, root) == fresh.resolve()


def _saved_adapter(root: Path) -> str:
    from lexsi.scripts.run_experiment_v2 import adapter_digest

    adapter = root / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "weights.bin").write_bytes(b"fixture-adapter")
    return adapter_digest(adapter)


def _project_with_baseline(tmp_path: Path, digest: str, adapter: Path) -> Path:
    root = tmp_path / "project"
    _write_baseline(root, {"frozen": True, "adapter_path": str(adapter), "adapter_sha256": digest})
    manifest = root / "data" / "benchmarks" / "tooluse_v2"
    manifest.mkdir(parents=True)
    (manifest / "manifest.json").write_text("{}\n", encoding="utf-8")
    (root / "results" / "rl_v2").mkdir(parents=True)
    return root


def test_baseline_checker_rejects_the_concatenation_hash(tmp_path):
    from lexsi.scripts.run_experiment_v2 import adapter_digest

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "weights.bin").write_bytes(b"fixture-adapter")
    assert adapter_digest(adapter) != _directory_sha256(adapter)
    root = _project_with_baseline(tmp_path, _directory_sha256(adapter), adapter)
    config = load_grpo_config(CONFIG)
    with pytest.raises(GrpoConfigError, match="hash does not match"):
        run_train(config, root)


def test_shipped_baseline_matches_adapter_digest_not_concatenation():
    from actproto.training.grpo_run import _require_baseline
    from lexsi.scripts.run_experiment_v2 import adapter_digest

    shipped = json.loads((LEXSI / "docs" / "SFT_BASELINE_MANIFEST.json").read_text(encoding="utf-8"))
    adapter = Path(shipped["adapter_path"])
    if not adapter.is_dir():
        pytest.skip("frozen adapter directory is not in this checkout")
    if _git_lfs_pointer(adapter / "adapter_model.safetensors"):
        pytest.skip("adapter weights are a Git LFS pointer; smudge the blob before checking the digest")
    assert adapter_digest(adapter) == shipped["adapter_sha256"]
    assert _directory_sha256(adapter) != shipped["adapter_sha256"]
    manifest = _require_baseline(LEXSI, load_grpo_config(CONFIG))
    assert manifest["adapter_sha256"] == shipped["adapter_sha256"]


def test_cpu_profile_still_refuses_optimization_after_a_matching_baseline(tmp_path, monkeypatch):
    adapter = tmp_path / "adapter"
    digest = _saved_adapter(tmp_path)
    root = _project_with_baseline(tmp_path, digest, adapter)
    task = TaskSpec("t", "g", "train", "What is 1+1?", "outcome", {"answer": "private-oracle"}, [],
                    {"type": "exact"}, 2, "env", "hash")
    monkeypatch.setattr(
        "actproto.training.rl_dataset.load_scoped_train_manifest",
        lambda path: (object(), [task]),
    )

    def fail(*args, **kwargs):
        raise AssertionError("Tiny Aya training started on the CPU profile")

    monkeypatch.setattr("actproto.training.grpo_run._train_tiny_aya", fail)
    config = load_grpo_config(CONFIG)
    edited = config.__class__(**{**config.__dict__, "train_manifest": "data/benchmarks/tooluse_v2/manifest.json"})
    with pytest.raises(GrpoConfigError, match="beta_zero is not an SFT reference"):
        run_train(edited, root)


def test_named_gpu_train_reaches_tiny_aya_only_in_the_masked_child(tmp_path, monkeypatch):
    adapter = tmp_path / "adapter"
    digest = _saved_adapter(tmp_path)
    root = _project_with_baseline(tmp_path, digest, adapter)
    task = TaskSpec("t", "g", "train", "What is 1+1?", "outcome", {"answer": "private-oracle"}, [],
                    {"type": "exact"}, 2, "env", "hash")
    monkeypatch.setattr(
        "actproto.training.rl_dataset.load_scoped_train_manifest",
        lambda path: (object(), [task]),
    )
    seen = {}

    def fake_train(*args, **kwargs):
        seen["rows"] = args[3]
        seen["steps"] = kwargs["max_steps"]
        return {"mode": "train", "reference_policy": "beta_zero", "baseline_loaded": True}

    monkeypatch.setattr("actproto.training.grpo_run._train_tiny_aya", fake_train)
    config = load_grpo_config(E2_CONFIG, allow_historical=True)
    with pytest.raises(GrpoConfigError, match="masked child"):
        run_train(config, root)
    assert "rows" not in seen
    monkeypatch.setenv("LEXSI_RL_GPU_CHILD", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    payload = run_train(config, root)
    assert seen["steps"] == 20
    assert payload["reference_policy"] == "beta_zero"
    assert payload["baseline_loaded"] is True
    encoded = json.dumps(seen["rows"])
    assert "private-oracle" not in encoded
    assert set(seen["rows"][0]) == {"prompt"}


def test_gpu_child_still_rejects_a_missing_or_mismatched_adapter(tmp_path, monkeypatch):
    monkeypatch.setenv("LEXSI_RL_GPU_CHILD", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        "actproto.training.grpo_run._train_tiny_aya",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("training started")),
    )
    config = load_grpo_config(E2_CONFIG, allow_historical=True)
    missing = tmp_path / "missing"
    _write_baseline(missing, {
        "frozen": True,
        "adapter_path": str(missing / "no-such-adapter"),
        "adapter_sha256": "ab" * 32,
    })
    with pytest.raises(GrpoConfigError, match="adapter path does not exist"):
        run_train(config, missing)

    adapter = tmp_path / "adapter"
    _saved_adapter(tmp_path)
    mismatched = _project_with_baseline(tmp_path, "cd" * 32, adapter)
    with pytest.raises(GrpoConfigError, match="hash does not match"):
        run_train(config, mismatched)


def test_train_payload_records_beta_zero_only_after_the_adapter_loads():
    from actproto.training.grpo_run import _train_payload

    payload = _train_payload(global_step=1, baseline_loaded=True, reference_model_built=False)
    assert payload["reference_policy"] == "beta_zero"
    assert payload["baseline_loaded"] is True
    assert payload["reference_model_built"] is False
    assert payload["beta"] == 0.0
    with pytest.raises(GrpoConfigError, match="frozen adapter"):
        _train_payload(global_step=1, baseline_loaded=False, reference_model_built=False)
    with pytest.raises(GrpoConfigError, match="reference model"):
        _train_payload(global_step=1, baseline_loaded=True, reference_model_built=True)


def test_gpu_train_without_the_live_flag_does_not_launch(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("training launched")

    monkeypatch.setattr("scripts.train_grpo.launch_gpu_smoke", fail)
    monkeypatch.setattr("actproto.training.grpo_run._train_tiny_aya", fail)
    assert main(["--train", "--config", str(E2_CONFIG)]) == 2


def test_live_train_launch_masks_only_the_child(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr("scripts.train_grpo.assert_5060_idle", lambda: None)
    monkeypatch.setattr("scripts.train_grpo.visible_ordinal_for_5060", lambda: "0")
    seen = {}

    def fake_call(cmd, env):
        seen["cmd"] = cmd
        seen["env"] = env
        return 0

    monkeypatch.setattr("scripts.train_grpo.subprocess.call", fake_call)
    assert main(["--train", "--live-gpu", "--allow-historical", "--config", str(E2_CONFIG)]) == 0
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert seen["env"]["LEXSI_RL_GPU_CHILD"] == "1"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in (None, "")
    assert "--train" in seen["cmd"]
    assert "--live-gpu" in seen["cmd"]
    assert "--allow-historical" in seen["cmd"]


def test_finish_string_without_a_tool_trace_is_not_a_training_reward():
    from actproto.training import grpo_run

    rewards = grpo_run._executed_trace_reward(["I will finish the task"])
    assert rewards == [0.0]
    conversational = grpo_run._executed_trace_reward(
        [[{"role": "assistant", "content": "finish"}]],
    )
    assert conversational == [0.0]
    assert all(reward <= 0 for reward in rewards)
    source = inspect.getsource(grpo_run._train_tiny_aya)
    assert "_executed_trace_reward" in source
    assert "_public_completion_reward" not in source
    assert "AgenticRolloutAdapter" in grpo_run._fit.__code__.co_names
    assert "_executed_trace_reward" not in grpo_run._fit.__code__.co_names
    assert "_public_completion_reward" not in grpo_run._fit.__code__.co_names
    assert "AgenticRolloutAdapter" in grpo_run._gpu_attempt.__code__.co_names
    assert "_executed_trace_reward" not in grpo_run._gpu_attempt.__code__.co_names
    assert 0.0 in grpo_run._fit.__code__.co_consts
    assert 0.0 in grpo_run._gpu_attempt.__code__.co_consts
    with pytest.raises(GrpoConfigError, match="serialized rollout JSON"):
        grpo_run._executed_trace_reward(["finish"], reward_sessions=[{"trace": "finish"}])
    assert "_study_rollout" in grpo_run._train_tiny_aya.__code__.co_names
    assert "rollout_func=rollout" in inspect.getsource(grpo_run._train_tiny_aya)
    assert "_study_rollout" not in grpo_run._fit.__code__.co_names
    assert "_study_rollout" not in grpo_run._gpu_attempt.__code__.co_names


def test_val_comparison_rejects_a_sealed_test_payload(tmp_path):
    from lexsi.scripts.compare_experiments_v2 import compare_candidates, load_candidates

    path = tmp_path / "sealed.json"
    path.write_text(json.dumps({"split": "test", "status": "completed", "id": "sealed"}), encoding="utf-8")
    with pytest.raises(ValueError, match="validation"):
        load_candidates([path])
    with pytest.raises(ValueError, match="validation"):
        compare_candidates([{"split": "test", "status": "completed", "id": "sealed"}], "e2")


def test_loaded_policy_must_be_the_e2_lora_without_a_ref_adapter():
    from actproto.training import grpo_run
    from actproto.training.grpo_run import assert_loaded_e2_policy

    class Target:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    class Flag:
        def __init__(self, requires_grad):
            self.requires_grad = requires_grad

    def model(peft_config):
        return SimpleNamespace(
            peft_config=peft_config,
            named_parameters=lambda: [("base.weight", Flag(False)), ("lora_A.default", Flag(True))],
        )

    config = load_grpo_config(E2_CONFIG, allow_historical=True)
    assert_loaded_e2_policy(model({"default": Target()}), config)
    with pytest.raises(GrpoConfigError, match="ref adapter"):
        assert_loaded_e2_policy(model({"default": Target(), "ref": Target()}), config)
    with pytest.raises(GrpoConfigError, match="existing adapter"):
        assert_loaded_e2_policy(model({}), config)
    other = config.__class__(**{**config.__dict__, "learning_rate": 2e-4})
    with pytest.raises(GrpoConfigError, match="5e-6"):
        assert_loaded_e2_policy(model({"default": Target()}), other)
    full = model({"default": Target()})
    full.named_parameters = lambda: [("base.weight", Flag(True))]
    with pytest.raises(GrpoConfigError, match="LoRA matrices"):
        assert_loaded_e2_policy(full, config)
    source = inspect.getsource(grpo_run._train_tiny_aya)
    assert "is_trainable=True" in source
    assert "get_peft_model" not in source
    assert "merge_and_unload" not in source


def test_tied_pair_records_zero_variance_and_does_not_change_weights():
    from actproto.training import grpo_run
    from actproto.training.grpo_run import format_step_log, tied_step_leaves_weights

    weights = {"lora_A": 0.25, "lora_B": -1.0}
    unchanged, record = tied_step_leaves_weights(weights, [0.0, 0.0], 2)
    assert unchanged == weights
    assert record["zero_variance_groups"] == 1
    assert record["advantages"] == [0.0, 0.0]
    assert record["reward_std"] == 0.0
    assert record["grad_norm"] == 0.0
    assert record["all_groups_tied"] is True
    line = format_step_log({}, record)
    assert "zero_variance_groups=1" in line
    assert "advantages=0" in line
    assert "scale_rewards" not in inspect.getsource(grpo_run._train_tiny_aya)
    with pytest.raises(GrpoConfigError, match="all tied"):
        tied_step_leaves_weights(weights, [0.0, 1.0], 2)


def test_live_reward_scores_the_shaped_trace_and_ignores_finish_text():
    from actproto.agent.harness import DictToolHarness
    from actproto.training.grpo_run import _executed_trace_reward
    from actproto.training.rewards_v2 import TrustedRewardSession, group_advantages

    spec = TaskSpec("t", "g", "train", "What is 2+2?", "outcome", {"answer": "4", "valid_call_bound": 1}, [],
                    {"type": "exact"}, 4, "env", "hash")
    success = TrustedRewardSession(spec, DictToolHarness())
    success.execute("calculator", {"expression": "2+2"})
    success.execute("finish", {"answer": "4"})
    invalid = TrustedRewardSession(spec, DictToolHarness())
    invalid.invalid_attempt("parse_error")
    other = TrustedRewardSession(spec, DictToolHarness())
    other.invalid_attempt("parse_error")
    long_text = "finish " * 40
    rewards = _executed_trace_reward(
        ["finish", long_text, "finish"],
        [success, invalid, other],
        ["finish", "parse_error", "parse_error"],
    )
    assert rewards[0] == pytest.approx(0.95)
    assert rewards[1] == pytest.approx(-0.10)
    assert rewards[1] == rewards[2]
    assert rewards[0] > 0 > rewards[1]
    assert group_advantages([rewards[1], rewards[2]])[0] == [0.0, 0.0]


def test_trainer_reward_uses_prefix_credit_and_ignores_finish_text_and_length():
    from actproto.agent.harness import DictToolHarness
    from actproto.training.grpo_run import _executed_trace_reward
    from actproto.training.rewards_v2 import PREFIX_VERSION, SHAPED_VERSION, TrustedRewardSession, score_executed_session

    spec = TaskSpec("t", "g", "train", "What is 2+2?", "required_tools", {"answer": "4"},
                    [{"id": "calc", "tool": "calculator", "arguments": {"expression": "2+2"}}],
                    {"type": "exact"}, 4, "env", "hash")

    def partial(text):
        session = TrustedRewardSession(spec, DictToolHarness())
        session.execute("calculator", {"expression": "2+2"})
        session.execute("finish", {"answer": "wrong"})
        return _executed_trace_reward([text], [session], ["finish"])[0]

    short = partial("finish")
    long = partial("finish " * 80)
    assert short == pytest.approx(0.30)
    assert long == short
    assert short > 0
    shaped = TrustedRewardSession(spec, DictToolHarness())
    shaped.execute("calculator", {"expression": "2+2"})
    shaped.execute("finish", {"answer": "wrong"})
    assert shaped.score(completion_reason="finish", version=SHAPED_VERSION).reward == 0
    assert _executed_trace_reward(["finish the task without a session"]) == [0.0]
    assert _executed_trace_reward(["finish"], [None], ["parse_error"]) == [0.0]

    live = TrustedRewardSession(spec, DictToolHarness())
    live.execute("calculator", {"expression": "2+2"})
    live.execute("finish", {"answer": "4"})
    first = live.score(completion_reason="finish", version=PREFIX_VERSION)
    assert first.version == PREFIX_VERSION and first.reward == 1.0

    def score_again(*args, **kwargs):
        raise AssertionError("finalized session was scored a second time")

    live.score = score_again
    assert _executed_trace_reward(["x"], [live], ["finish"]) == [first.reward]
    other = TrustedRewardSession(spec, DictToolHarness())
    other.invalid_attempt("parse_error")
    assert score_executed_session(other, completion_reason="parse_error") == pytest.approx(-0.10)


def test_study_rollout_executes_tools_and_refuses_historical_directories(monkeypatch):
    import json

    from actproto.integrations.agenttune_backend import AgenticRolloutAdapter
    from actproto.training import grpo_run
    from actproto.training.rewards_v2 import SHAPED_VERSION
    from actproto.training.rl_dataset import VerifiedTrainFixtures

    from actproto.training.sft_masks import _BRANCH

    class CharacterTokenizer:
        def __init__(self):
            self.chat_template = {
                "default": "DEFAULT-TEMPLATE\n" + _BRANCH,
                "alternate": "ALTERNATE-TEMPLATE\n" + _BRANCH,
            }
            self.seen_templates = []

        def apply_chat_template(self, messages, tokenize, add_generation_prompt, chat_template=None):
            self.seen_templates.append(chat_template)
            text = "".join(f"<{message['role']}>{message['content']}</{message['role']}>" for message in messages)
            if add_generation_prompt:
                text += "<assistant>"
            return list(map(ord, text))

    class StatefulEnvironment:
        identifier = "env"
        digest = "hash"
        def call(self, environment_id, name, arguments):
            if name == "calculator":
                return {"ok": True, "data": {"value": 2}}
            if name == "finish":
                return {"ok": True, "data": {"answer": arguments["answer"], "done": True}}
            return {"ok": False, "error": {"type": "not_found", "message": "missing"}}

    spec = TaskSpec("t", "g", "train", "What is 1+1?", "outcome", {"answer": "2"}, [],
                    {"type": "exact"}, 2, "env", "hash")
    scope = object.__new__(VerifiedTrainFixtures)
    scope._tasks = {spec.task_id: spec}

    def for_task(task):
        if scope._tasks.get(task.task_id) != task:
            raise ValueError("task is not the manifest-bound training task")
        return StatefulEnvironment()

    scope.for_task = for_task
    seen = []
    sampled_caps = []
    tokenizer = CharacterTokenizer()

    def fake_sample(model, tokenizer, prompt_ids, **kwargs):
        del tokenizer
        seen.append(model)
        sampled_caps.append(kwargs.get("max_new_tokens"))
        assert kwargs.get("temperature") == pytest.approx(0.8)
        text = "".join(map(chr, prompt_ids))
        action = {"name": "finish", "arguments": {"answer": "2"}} if "Tool result:" in text else {
            "name": "calculator", "arguments": {"expression": "1+1"},
        }
        raw = json.dumps(action, separators=(",", ":"))
        ids = list(map(ord, raw))
        return raw, ids, [-0.2] * len(ids), False

    class PolicyModel:
        def __init__(self):
            self.config = SimpleNamespace(max_position_embeddings=8192)

    trainer_policy = PolicyModel()
    monkeypatch.setattr(grpo_run, "_sample_peft_action", fake_sample)
    config = load_grpo_config(E2_CONFIG, allow_historical=True)
    assert config.temperature == pytest.approx(0.8)
    rollout = grpo_run._study_rollout("loaded-peft", tokenizer, [spec], scope, config, "grpo_gpu_tools")
    assert isinstance(rollout, AgenticRolloutAdapter)
    result = rollout([[{"role": "user", "content": spec.prompt}]], trainer=SimpleNamespace(model=trainer_policy))
    assert seen == [trainer_policy, trainer_policy]
    assert sampled_caps == [256, 256]
    assert tokenizer.seen_templates
    assert set(tokenizer.seen_templates) == {tokenizer.chat_template["default"]}
    ledger = result["reward_sessions"][0].score(completion_reason=result["completion_reasons"][0], version=SHAPED_VERSION)
    assert result["completion_reasons"] == ["finish"]
    assert ledger.verified_success is True
    assert [event["tool"] for event in result["reward_sessions"][0].captured_tool_events()] == ["calculator", "finish"]
    with pytest.raises(GrpoConfigError, match="new results"):
        grpo_run.refuse_historical_output(Path("results/rl_v2/grpo_gpu_e2"))
    with pytest.raises(GrpoConfigError, match="new results"):
        grpo_run.refuse_historical_output(Path("results/rl_v2/grpo_gpu_smoke"))
    for name in ("grpo_gpu_tools", "grpo_gpu_tools_r2", "grpo_gpu_study"):
        with pytest.raises(GrpoConfigError, match="new results"):
            grpo_run.refuse_historical_output(Path("results/rl_v2") / name)


def test_peft_sampler_logprobs_follow_returned_scores():
    torch = pytest.importorskip("torch")

    class Tokenizer:
        eos_token_id = 1
        pad_token_id = 0
        def decode(self, ids, skip_special_tokens=True):
            del skip_special_tokens
            return "call"

    class Model:
        def __init__(self):
            self.training = True
            self._modes = []
        def parameters(self):
            yield torch.zeros(1)
        def eval(self):
            self.training = False
            self._modes.append("eval")
        def train(self, mode=True):
            self.training = bool(mode)
            self._modes.append("train" if mode else "eval")
        def generate(self, **kwargs):
            # Sampling must run under eval(); train()+dropout/ckpt collapses Tiny Aya.
            assert self.training is False
            new = torch.tensor([[4, 1]])
            sequences = torch.cat([kwargs["input_ids"], new], dim=-1)
            scores = (torch.zeros(1, 6), torch.zeros(1, 6))
            scores[0][0, 4] = 2
            return SimpleNamespace(sequences=sequences, scores=scores)

    from actproto.training.grpo_run import _sample_peft_action

    model = Model()
    text, ids, logprobs, truncated = _sample_peft_action(
        model, Tokenizer(), [3], temperature=1.0, top_p=1.0, max_new_tokens=4,
    )
    assert text == "call"
    assert ids == [4]
    assert truncated is False
    assert model.training is True
    assert model._modes == ["eval", "train"]

    class EosModel(Model):
        def generate(self, **kwargs):
            sequences = torch.cat([kwargs["input_ids"], torch.tensor([[1]])], dim=-1)
            scores = (torch.zeros(1, 6),)
            scores[0][0, 1] = 1.5
            return SimpleNamespace(sequences=sequences, scores=scores)

    eos_text, eos_ids, eos_logprobs, eos_truncated = _sample_peft_action(
        EosModel(), Tokenizer(), [3], temperature=1.0, top_p=1.0, max_new_tokens=4,
    )
    assert eos_ids == [1]
    assert eos_text == Tokenizer().decode([1], skip_special_tokens=False)
    assert eos_logprobs and eos_truncated is False
    expected = float(torch.log_softmax(torch.tensor([0.0, 0.0, 0.0, 0.0, 2.0, 0.0]), dim=-1)[4])
    assert logprobs == pytest.approx([expected])
    with pytest.raises(GrpoConfigError, match="max_new_tokens"):
        _sample_peft_action(Model(), Tokenizer(), [3], temperature=1.0, top_p=1.0, max_new_tokens=257)


def _contains_ids(haystack, needle):
    width = len(needle)
    return any(haystack[index:index + width] == needle for index in range(len(haystack) - width + 1))


def test_study_rollout_returns_a_completion_longer_than_the_loss_denominator(monkeypatch):
    import json

    from actproto.agent.inference_v2 import ModelPolicy
    from actproto.training import grpo_run
    from actproto.training.rl_dataset import VerifiedTrainFixtures
    from actproto.training.sft_masks import _BRANCH

    class CharacterTokenizer:
        def __init__(self):
            self.chat_template = {
                "default": "DEFAULT-TEMPLATE\n" + _BRANCH,
                "alternate": "ALTERNATE-TEMPLATE\n" + _BRANCH,
            }

        def apply_chat_template(self, messages, tokenize, add_generation_prompt, chat_template=None):
            assert chat_template == self.chat_template["default"]
            text = "".join(f"<{message['role']}>{message['content']}</{message['role']}>" for message in messages)
            if add_generation_prompt:
                text += "<assistant>"
            return list(map(ord, text))

    class StatefulEnvironment:
        identifier = "env"
        digest = "hash"
        def call(self, environment_id, name, arguments):
            if name == "calculator":
                return {"ok": True, "data": {"value": 1}}
            if name == "finish":
                return {"ok": True, "data": {"answer": arguments["answer"], "done": True}}
            return {"ok": False, "error": {"type": "not_found", "message": "missing"}}

    spec = TaskSpec("long", "g", "train", "Add the padded numbers", "outcome", {"answer": "1"}, [],
                    {"type": "exact"}, 2, "env", "hash")
    scope = object.__new__(VerifiedTrainFixtures)
    scope._tasks = {spec.task_id: spec}

    def for_task(task):
        if scope._tasks.get(task.task_id) != task:
            raise ValueError("task is not the manifest-bound training task")
        return StatefulEnvironment()

    scope.for_task = for_task
    expression = "1+" + ("1" * 180)
    first = json.dumps({"name": "calculator", "arguments": {"expression": expression}}, separators=(",", ":"))
    second = json.dumps({"name": "finish", "arguments": {"answer": "1" + ("0" * 180)}}, separators=(",", ":"))
    actions = iter((first, second))
    caps = []

    def fake_sample(model, tokenizer, prompt_ids, **kwargs):
        del model, tokenizer, prompt_ids
        caps.append(kwargs["max_new_tokens"])
        raw = next(actions)
        ids = list(map(ord, raw))
        assert len(ids) <= 256
        return raw, ids, [-0.2] * len(ids), False

    monkeypatch.setattr(grpo_run, "_sample_peft_action", fake_sample)
    config = load_grpo_config(E2_CONFIG, allow_historical=True)
    assert config.max_completion_length == 256
    tokenizer = CharacterTokenizer()
    policy = ModelPolicy(SimpleNamespace(config=SimpleNamespace(max_position_embeddings=8192)), tokenizer)
    assert policy.template == tokenizer.chat_template["default"]
    assert policy.max_new_tokens == 256
    rollout = grpo_run._study_rollout(
        SimpleNamespace(config=SimpleNamespace(max_position_embeddings=8192)),
        tokenizer, [spec], scope, config, "grpo_gpu_tools",
    )
    result = rollout([[{"role": "user", "content": spec.prompt}]], trainer=None)
    completion = result["completion_ids"][0]
    assert caps == [256, 256]
    assert len(completion) > 256
    assert len(completion) > config.max_completion_length
    assert _contains_ids(completion, list(map(ord, first)))
    assert _contains_ids(completion, list(map(ord, second)))
    assert "collect_episode" not in grpo_run._study_rollout.__code__.co_names


def _prefix_scope(spec):
    from actproto.training.rl_dataset import VerifiedTrainFixtures

    class StatefulEnvironment:
        identifier = spec.environment_id
        digest = spec.fixture_hash
        def call(self, environment_id, name, arguments):
            if name == "calculator":
                return {"ok": True, "data": {"value": 1}}
            if name == "finish":
                return {"ok": True, "data": {"answer": arguments["answer"], "done": True}}
            return {"ok": False, "error": {"type": "not_found", "message": "missing"}}

    scope = object.__new__(VerifiedTrainFixtures)
    scope._tasks = {spec.task_id: spec}

    def for_task(task):
        if scope._tasks.get(task.task_id) != task:
            raise ValueError("task is not the manifest-bound training task")
        return StatefulEnvironment()

    scope.for_task = for_task
    return scope


def test_tied_batch_redraws_once_and_a_second_tie_skips_without_moving_weights(tmp_path):
    import json

    from actproto.training.grpo_run import (
        _LAST_STEP, _executed_trace_reward, diagnostic_record, tied_step_leaves_weights, zero_tied_gradients,
    )
    from actproto.training.rewards_v2 import group_advantages
    from actproto.training.rl_rollouts import BehaviorToolRollout, PolicySample

    deps = [
        {"id": "a", "tool": "calculator", "arguments": {"expression": "1+1"}},
        {"id": "b", "tool": "calculator", "arguments": {"expression": "2+2"}, "after": ["a"]},
    ]
    spec = TaskSpec("prefix-task", "g", "train", "chain the calculators", "required_tools",
                    {"answer": "4", "secret": "hidden-oracle"}, deps, {"type": "exact"}, 4, "env", "hash")
    scope = _prefix_scope(spec)
    made = {"n": 0}

    def varying_factory(seed, trainer):
        del seed, trainer
        episode_index = made["n"]
        made["n"] += 1

        def policy(ids, budget):
            assert budget == 256
            prompt = "".join(map(chr, ids))
            if episode_index < 2:
                return PolicySample("not a call", list(map(ord, "not a call")), [-0.2] * len("not a call"), False)
            if episode_index == 2:
                action = call_json("finish", {"answer": "no"}) if "Tool result:" in prompt else call_json(
                    "calculator", {"expression": "1+1"})
                return PolicySample(action, list(map(ord, action)), [-0.2] * len(action), False)
            if "2+2" in prompt or prompt.count("Tool result:") >= 2:
                action = call_json("finish", {"answer": "no"})
            elif "Tool result:" in prompt:
                action = call_json("calculator", {"expression": "2+2"})
            else:
                action = call_json("calculator", {"expression": "1+1"})
            return PolicySample(action, list(map(ord, action)), [-0.2] * len(action), False)

        return policy

    rollout = BehaviorToolRollout(
        _CharTokenizer(), [spec], varying_factory, lambda: scope,
        deadline_seconds=5, max_generated_tokens=256, group_size=2,
        context_limit_reader=lambda trainer: 8192,
    )
    varied = rollout([spec.prompt, spec.prompt], trainer=None)
    assert varied["skipped_tied_batch"] is False
    assert made["n"] == 4
    rewards = _executed_trace_reward(
        ["finish", "finish"], varied["reward_sessions"], varied["completion_reasons"],
        skipped_tied_batch=varied["skipped_tied_batch"],
    )
    assert rewards[0] != rewards[1]
    assert any(item != 0.0 for item in group_advantages(rewards)[0])
    assert _LAST_STEP["skipped_tied_batch"] is False

    made["n"] = 0

    def always_tied(seed, trainer):
        del seed, trainer
        episode_index = made["n"]
        made["n"] += 1

        def policy(ids, budget):
            assert budget == 256
            if episode_index % 2 == 0:
                text = "not a call"
                return PolicySample(text, list(map(ord, text)), [-0.2] * len(text), False)
            text = "z" * 256
            return PolicySample(text, list(map(ord, text)), [-0.1] * 256, True)

        return policy

    tied = BehaviorToolRollout(
        _CharTokenizer(), [spec], always_tied, lambda: scope,
        deadline_seconds=5, max_generated_tokens=256, group_size=2,
        context_limit_reader=lambda trainer: 8192,
    )
    stuck = tied([spec.prompt, spec.prompt], trainer=None)
    assert stuck["skipped_tied_batch"] is True
    assert made["n"] == 4
    long_completion = "finish " * 30
    tied_rewards = _executed_trace_reward(
        ["not a call", long_completion], stuck["reward_sessions"], stuck["completion_reasons"],
        skipped_tied_batch=True,
    )
    assert tied_rewards == pytest.approx([-0.1, -0.1])
    weights = {"lora_A": 0.25, "lora_B": -1.0}
    unchanged, record = tied_step_leaves_weights(weights, tied_rewards, 2)
    assert unchanged == weights
    assert record["grad_norm"] == 0.0
    previews = [row.get("decoded_completion_preview") for row in _LAST_STEP["completions"]]
    assert previews[0] == "not a call"
    assert previews[1] == long_completion
    assert _LAST_STEP["completions"][0]["completion_reason"] == "parse_error"
    assert _LAST_STEP["completions"][1]["completion_reason"] == "truncated"

    class _Grad:
        def __init__(self):
            self.value = 3.0
        def zero_(self):
            self.value = 0.0

    class _Parameter:
        def __init__(self):
            self.data = 0.25
            self.grad = _Grad()

    class _Model:
        def __init__(self):
            self.param = _Parameter()
        def parameters(self):
            yield self.param

    model = _Model()
    zero_tied_gradients(model)
    assert model.param.data == 0.25
    assert model.param.grad.value == 0.0
    reasons = [row["completion_reason"] for row in stuck["reward_components"]]
    assert reasons == ["parse_error", "truncated"]
    for row in stuck["reward_components"]:
        assert set(row) == {
            "reward", "verified_success", "invalid_attempts", "duplicate_calls",
            "excess_calls", "excess_omitted_reason", "completion_reason",
        }
    trainer = SimpleNamespace(_metrics={"train": {"frac_reward_zero_std": [1.0]}})
    logged = diagnostic_record(
        {"loss": 0.0, "completions/clipped_ratio": 1.0, "completions/mean_terminated_length": 0.0},
        _LAST_STEP, trainer,
    )
    assert logged["skipped_tied_batch"] is True
    assert logged["grad_norm"] == 0.0
    assert logged["frac_reward_zero_std"] == 1.0
    assert logged["eos_based_clip"]["completions/clipped_ratio"] == 1.0
    assert "eos_based_clip" in logged
    assert logged["completion_reason_counts"] == {"parse_error": 1, "truncated": 1}
    assert logged["completions"][0]["decoded_completion_preview"] == "not a call"
    text = json.dumps(logged)
    assert "oracle" not in text and "hidden-oracle" not in text
    path = tmp_path / "step_diagnostics.jsonl"
    path.write_text(text + "\n", encoding="utf-8")
    assert "skipped_tied_batch" in path.read_text(encoding="utf-8")
    from_log = diagnostic_record({"frac_reward_zero_std": 0.5, "loss": 1.0}, _LAST_STEP, trainer)
    assert from_log["frac_reward_zero_std"] == 0.5
    from actproto.training.grpo_run import completion_preview, summarize_step_diagnostics

    assert completion_preview("x" * 600).endswith("x") and len(completion_preview("x" * 600)) == 512
    summary = summarize_step_diagnostics(path)
    assert summary["steps"] == 1
    assert summary["completion_reason_counts"] == {"parse_error": 1, "truncated": 1}


class _CharTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, chat_template=None):
        del tokenize, chat_template
        text = "".join(f"<{message['role']}>{message['content']}</{message['role']}>" for message in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return list(map(ord, text))


def test_val_comparison_rejects_a_test_split_and_unit_tests_do_not_launch_the_smoke(tmp_path, monkeypatch):
    import json

    from lexsi.scripts.compare_experiments_v2 import compare_candidates, load_candidates
    import lexsi.scripts.train_grpo as train_grpo

    launched = []
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    monkeypatch.setattr(train_grpo, "main", lambda *args, **kwargs: launched.append(args))
    payload = {"split": "test", "status": "completed", "id": "heldout",
               "metrics": {"macro_verified_success_v2": 1.0}}
    path = tmp_path / "heldout.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="validation"):
        load_candidates([path])
    with pytest.raises(ValueError, match="validation"):
        compare_candidates([payload], "e2")
    procedure = Path("docs/RL_BACKEND_COMPATIBILITY.md").read_text(encoding="utf-8")
    assert "actproto/configs/grpo_v2_gpu_prefix.yaml" in procedure
    assert "134c34ef-c155-b001-4e3a-b9134b072c21" in procedure
    assert "5b885e12-623f-268c-f07c-ee33db379969" in procedure
    assert "results/rl_v2/grpo_gpu_study" in procedure
    assert "CUDA_VISIBLE_DEVICES" in procedure
    assert "2da2899c0fd2f5ca0bffc32c2e9201e0b7bab8b1758d2fde429ff19cf50be8bd" in procedure
    assert "results/rl_v2/grpo_gpu_tools_r2" in procedure
    assert "--split val" in procedure
    assert "--final-eval" in procedure
    assert launched == []
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == visible


def call_json(name, arguments):
    import json
    return json.dumps({"name": name, "arguments": arguments}, separators=(",", ":"))
