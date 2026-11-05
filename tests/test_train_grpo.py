import json
import os
from hashlib import sha256
from pathlib import Path

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

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "weights.bin").write_bytes(b"fixture-adapter")
    matched = tmp_path / "matched"
    _write_baseline(matched, {
        "frozen": True,
        "adapter_path": str(adapter),
        "adapter_sha256": _directory_sha256(adapter),
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


def test_gpu_schema_does_not_enter_cpu_fit(monkeypatch):
    monkeypatch.setattr("actproto.training.grpo_run._fit", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("entered _fit")))
    config = load_grpo_config(GPU_CONFIG)
    with pytest.raises(GrpoConfigError, match="run_gpu_smoke"):
        run_smoke(config, Path("."), Path("unused"))


def test_gpu_train_and_resume_fail_closed(tmp_path):
    config = load_grpo_config(GPU_CONFIG)
    with pytest.raises(GrpoConfigError, match="Tiny Aya"):
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
