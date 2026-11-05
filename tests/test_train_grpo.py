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
from scripts.train_grpo import main

CONFIG = Path("actproto/configs/grpo_v2.yaml")
LEXSI = Path(__file__).resolve().parents[1]


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

    from scripts.run_experiment_v2 import adapter_digest

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
E2_CONFIG = Path("actproto/configs/grpo_v2_gpu_e2.yaml")


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
    from scripts.run_experiment_v2 import adapter_digest

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
    from scripts.run_experiment_v2 import adapter_digest

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
    from scripts.run_experiment_v2 import adapter_digest

    shipped = json.loads((LEXSI / "docs" / "SFT_BASELINE_MANIFEST.json").read_text(encoding="utf-8"))
    adapter = Path(shipped["adapter_path"])
    if not adapter.is_dir():
        pytest.fail("frozen adapter path does not exist")
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
    config = load_grpo_config(E2_CONFIG)
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
    config = load_grpo_config(E2_CONFIG)
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
    assert main(["--train", "--live-gpu", "--config", str(E2_CONFIG)]) == 0
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert seen["env"]["LEXSI_RL_GPU_CHILD"] == "1"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in (None, "")
    assert "--train" in seen["cmd"]
    assert "--live-gpu" in seen["cmd"]


def test_finish_string_without_a_tool_trace_is_not_a_training_reward():
    from actproto.training import grpo_run

    rewards = grpo_run._executed_trace_reward(["I will finish the task"])
    assert rewards == [0.0]
    conversational = grpo_run._executed_trace_reward(
        [[{"role": "assistant", "content": "finish"}]],
    )
    assert conversational == [0.0]
    assert all(reward <= 0 for reward in rewards)
    assert "_executed_trace_reward" in grpo_run._train_tiny_aya.__code__.co_names
    assert "_public_completion_reward" not in grpo_run._train_tiny_aya.__code__.co_names
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


def test_study_rollout_executes_tools_and_refuses_historical_directories(monkeypatch):
    import json

    from actproto.integrations.agenttune_backend import AgenticRolloutAdapter
    from actproto.training import grpo_run
    from actproto.training.rewards_v2 import SHAPED_VERSION
    from actproto.training.rl_dataset import VerifiedTrainFixtures

    class CharacterTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
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

    def fake_sample(model, tokenizer, prompt_ids, **kwargs):
        del tokenizer, kwargs
        seen.append(model)
        text = "".join(map(chr, prompt_ids))
        action = {"name": "finish", "arguments": {"answer": "2"}} if "Tool result:" in text else {
            "name": "calculator", "arguments": {"expression": "1+1"},
        }
        raw = json.dumps(action, separators=(",", ":"))
        ids = list(map(ord, raw))
        return raw, ids, [-0.2] * len(ids), False

    monkeypatch.setattr(grpo_run, "_sample_peft_action", fake_sample)
    config = load_grpo_config(E2_CONFIG)
    rollout = grpo_run._study_rollout("loaded-peft", CharacterTokenizer(), [spec], scope, config, "grpo_gpu_tools")
    assert isinstance(rollout, AgenticRolloutAdapter)
    result = rollout([[{"role": "user", "content": spec.prompt}]], trainer=SimpleNamespace(model="trainer-policy"))
    assert seen == ["trainer-policy", "trainer-policy"]
    ledger = result["reward_sessions"][0].score(completion_reason=result["completion_reasons"][0], version=SHAPED_VERSION)
    assert result["completion_reasons"] == ["finish"]
    assert ledger.verified_success is True
    assert [event["tool"] for event in result["reward_sessions"][0].captured_tool_events()] == ["calculator", "finish"]
    with pytest.raises(GrpoConfigError, match="new results"):
        grpo_run.refuse_historical_output(Path("results/rl_v2/grpo_gpu_e2"))
    with pytest.raises(GrpoConfigError, match="new results"):
        grpo_run.refuse_historical_output(Path("results/rl_v2/grpo_gpu_smoke"))


def test_peft_sampler_logprobs_follow_returned_scores():
    torch = pytest.importorskip("torch")

    class Tokenizer:
        eos_token_id = 1
        pad_token_id = 0
        def decode(self, ids, skip_special_tokens=True):
            del skip_special_tokens
            return "call"

    class Model:
        def parameters(self):
            yield torch.zeros(1)
        def generate(self, **kwargs):
            new = torch.tensor([[4, 1]])
            sequences = torch.cat([kwargs["input_ids"], new], dim=-1)
            scores = (torch.zeros(1, 6), torch.zeros(1, 6))
            scores[0][0, 4] = 2
            return SimpleNamespace(sequences=sequences, scores=scores)

    from actproto.training.grpo_run import _sample_peft_action

    text, ids, logprobs, truncated = _sample_peft_action(
        Model(), Tokenizer(), [3], temperature=1.0, top_p=1.0, max_new_tokens=4,
    )
    assert text == "call"
    assert ids == [4]
    assert truncated is False
    expected = float(torch.log_softmax(torch.tensor([0.0, 0.0, 0.0, 0.0, 2.0, 0.0]), dim=-1)[4])
    assert logprobs == pytest.approx([expected])
    with pytest.raises(GrpoConfigError, match="max_new_tokens"):
        _sample_peft_action(Model(), Tokenizer(), [3], temperature=1.0, top_p=1.0, max_new_tokens=257)
