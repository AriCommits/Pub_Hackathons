"""GRPO lifecycle. Plan and collect modes do not optimize; real training is fail-closed."""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import shutil

from actproto.experiments.contracts import TaskSpec
from actproto.training.rl_config import GRPO_GPU_PROFILE, GrpoConfigError, GrpoV2Config, resolve_output

COMPLETED_NAME = "completed.json"
# The fixture action is longer than a one-token probe. This bound is only for that
# CPU mechanics smoke; it does not change the study config's max_completion_length.
SMOKE_COMPLETION_LENGTH = 512


def completed_marker(output_dir: Path) -> Path:
    return output_dir / COMPLETED_NAME


def run_plan(config: GrpoV2Config, project_root: Path) -> dict:
    """Describe the run without creating a trainer, an output directory, or a completed marker."""
    baseline = project_root / config.baseline_manifest if config.baseline_manifest else None
    return {
        "mode": "plan-only",
        "optimizes": False,
        "completed_marker": False,
        "reference_policy": config.reference_policy,
        "beta": config.beta,
        "baseline_present": bool(baseline and baseline.is_file()),
        "config": config.public_dict(),
    }


def run_collect_only(config: GrpoV2Config, project_root: Path) -> dict:
    """Validate that a train manifest exists. Do not build a trainer or mark a run complete."""
    if not config.train_manifest:
        raise GrpoConfigError("collect-only requires a train manifest and does not optimize")
    path = project_root / config.train_manifest
    if not path.is_file():
        raise GrpoConfigError("train manifest is missing; collect-only did not optimize")
    from actproto.training.rl_dataset import load_scoped_train_manifest

    _scope, tasks = load_scoped_train_manifest(path)
    prompts = public_prompt_rows(tasks)
    return {"mode": "collect-only", "optimizes": False, "tasks": len(prompts), "optimizer_steps": 0}


def run_train(config: GrpoV2Config, project_root: Path) -> dict:
    """Refuse real optimization. A GPU profile does not open Tiny Aya training."""
    if config.schema_version == GRPO_GPU_PROFILE:
        raise GrpoConfigError("a GPU config does not open Tiny Aya optimization")
    manifest = _require_baseline(project_root, config)
    if not config.train_manifest:
        raise GrpoConfigError("real training requires a nonempty train manifest")
    train_path = project_root / config.train_manifest
    if not train_path.is_file():
        raise GrpoConfigError("train manifest is missing")
    from actproto.training.rl_dataset import load_scoped_train_manifest

    _scope, tasks = load_scoped_train_manifest(train_path)
    public_prompt_rows(tasks)
    adapter = Path(manifest["adapter_path"])
    output = resolve_output(config.output_dir, project_root)
    if output.resolve() == adapter.resolve() or adapter.resolve() in output.resolve().parents:
        raise GrpoConfigError("training output must not overwrite the frozen SFT adapter")
    raise GrpoConfigError(
        "real GRPO optimization is blocked until the frozen SFT baseline is loaded on a verified GPU configuration; "
        "beta_zero is not an SFT reference"
    )


def public_prompt_rows(tasks: list[TaskSpec]) -> list[dict]:
    """Expose only public prompts. Oracle fields stay out of the trainer dataset."""
    if not tasks:
        raise GrpoConfigError("a nonempty training dataset is required")
    rows = []
    for task in tasks:
        if not isinstance(task, TaskSpec) or task.split != "train" or not task.prompt:
            raise GrpoConfigError("training rows must be train-split tasks with a public prompt")
        rows.append({"prompt": [{"role": "user", "content": task.prompt}]})
    encoded = json.dumps(rows)
    for task in tasks:
        secret = json.dumps(task.oracle, sort_keys=True)
        if secret and secret in encoded:
            raise GrpoConfigError("training prompts leaked private oracle data")
    return rows


def run_smoke(config: GrpoV2Config, project_root: Path, output_dir: Path, *, max_steps: int = 1) -> dict:
    """One real CPU update on a tiny LoRA model. This does not load or replace the SFT baseline."""
    if config.schema_version == GRPO_GPU_PROFILE:
        raise GrpoConfigError("grpo_v2_gpu smoke must use run_gpu_smoke and must not enter the CPU fit")
    del project_root
    if max_steps < 1:
        raise GrpoConfigError("smoke requires at least one update")
    _require_fresh(output_dir)
    return _fit(config, output_dir, max_steps=max_steps, resume_from=None)


def run_resume(config: GrpoV2Config, project_root: Path, output_dir: Path, checkpoint: Path, *, max_steps: int) -> dict:
    """Continue a smoke checkpoint without removing its earlier files."""
    if config.schema_version == GRPO_GPU_PROFILE:
        raise GrpoConfigError("GPU smoke resume is closed")
    if not checkpoint.is_dir() or output_dir.resolve() not in checkpoint.resolve().parents:
        raise GrpoConfigError("resume checkpoint must be an existing directory inside the run output")
    if max_steps < 2:
        raise GrpoConfigError("resume must continue past the first update")
    preserved = {path: sha256(path.read_bytes()).hexdigest() for path in checkpoint.rglob("*") if path.is_file()}
    if not preserved:
        raise GrpoConfigError("resume checkpoint is empty")
    result = _fit(config, output_dir, max_steps=max_steps, resume_from=checkpoint)
    for path, digest in preserved.items():
        if not path.is_file() or sha256(path.read_bytes()).hexdigest() != digest:
            raise GrpoConfigError(f"resume replaced an earlier checkpoint file: {path.name}")
    if int(result["global_step"]) <= 1:
        raise GrpoConfigError("resume did not advance the update counter")
    result["preserved_checkpoint_files"] = len(preserved)
    return result


def _fit(config: GrpoV2Config, output_dir: Path, *, max_steps: int, resume_from: Path | None) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, TrainerCallback

    from actproto.integrations.agenttune_backend import AgenticRolloutAdapter, create_trainer

    output_dir.mkdir(parents=True, exist_ok=True)
    if resume_from is None and completed_marker(output_dir).exists():
        raise GrpoConfigError("a completed marker already exists; refusing to overwrite the run")
    tokenizer = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global", local_files_only=True)
    actions = [
        '{"name":"calculator","arguments":{"expression":"1+1"}}',
        '{"name":"finish","arguments":{"answer":"2"}}',
    ]

    def policy(prompt_ids, trainer):
        del trainer
        text = tokenizer.decode(prompt_ids)
        action = actions[1] if "Tool result:" in text else actions[0]
        sampled = tokenizer.encode(action, add_special_tokens=False)
        return action, sampled, [-0.4] * len(sampled)

    def execute(name, arguments):
        if name == "calculator" and arguments.get("expression") == "1+1":
            return "2"
        if name == "finish":
            return arguments.get("answer", "")
        raise ValueError("smoke tool received an unexpected call")

    adapter = AgenticRolloutAdapter(tokenizer, policy, execute, max_decisions=2)
    base_config = GPT2Config(
        vocab_size=len(tokenizer), n_layer=1, n_head=1, n_embd=8, n_positions=1024,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(base_config)
    lora = LoraConfig(r=4, lora_alpha=8, target_modules=["c_attn"], lora_dropout=0.0, bias="none",
                      task_type="CAUSAL_LM", fan_in_fan_out=True)

    class Probe(TrainerCallback):
        def __init__(self):
            self.finite = False
            self.nonzero = False

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            del args, state, control
            total = 0.0
            for parameter in kwargs["model"].parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    value = float(parameter.grad.detach().abs().sum())
                    if value == value and value != float("inf"):
                        self.finite = True
                    total += value
            self.nonzero = total > 0

    probe = Probe()
    torch.manual_seed(config.seed)
    wrapper = create_trainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=lambda completions, **kwargs: [1.0 if index % 2 == 0 else 0.0 for index, _ in enumerate(completions)],
        train_dataset=Dataset.from_dict({"prompt": ["calculate 1+1", "calculate 1+1"]}),
        rollout_func=adapter,
        peft_config=lora,
        callbacks=[probe],
        output_dir=str(output_dir),
        num_generations=config.num_generations,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_steps=max_steps,
        warmup_steps=0,
        logging_steps=1,
        save_steps=1,
        save_strategy="steps",
        max_completion_length=SMOKE_COMPLETION_LENGTH,
        temperature=config.temperature,
        top_p=config.top_p,
        seed=config.seed,
        lr_scheduler_type="constant",
        bf16=False,
        fp16=False,
        report_to="none",
        use_cpu=True,
        use_vllm=False,
        beta=0.0,
        gradient_checkpointing=False,
        disable_tqdm=True,
    )
    wrapper.setup_data()
    wrapper.setup_trainer()
    trainer = wrapper.trainer
    if trainer is None:
        raise GrpoConfigError("backend did not construct a trainer")
    trained = trainer.model
    if getattr(trainer, "ref_model", None) is not None or float(trainer.beta) != 0.0:
        raise GrpoConfigError("beta_zero smoke constructed a reference model")
    before = {name: parameter.detach().cpu().clone() for name, parameter in trained.named_parameters()}
    trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
    named = {id(parameter): name for name, parameter in trained.named_parameters()}
    optimizer_names = [
        named.get(id(parameter), "")
        for group in trainer.optimizer.param_groups for parameter in group["params"]
    ]
    if not probe.finite or not probe.nonzero:
        raise GrpoConfigError("smoke update did not produce a finite nonzero gradient")
    if not optimizer_names or any("lora_" not in name for name in optimizer_names):
        raise GrpoConfigError("optimizer contains a parameter outside the LoRA adapters")
    frozen_changed = [
        name for name, parameter in trained.named_parameters()
        if not parameter.requires_grad and not torch.equal(before[name], parameter.detach().cpu())
    ]
    trained_changed = [
        name for name, parameter in trained.named_parameters()
        if parameter.requires_grad and not torch.equal(before[name], parameter.detach().cpu())
    ]
    if frozen_changed or not trained_changed:
        raise GrpoConfigError("smoke changed frozen weights or left every trainable weight unchanged")
    trainer.save_model(str(output_dir))
    reloaded = PeftModel.from_pretrained(GPT2LMHeadModel(base_config), str(output_dir))
    current = {name: parameter.detach().cpu() for name, parameter in trained.named_parameters() if "lora_" in name}
    restored = {name: parameter.detach().cpu() for name, parameter in reloaded.named_parameters() if "lora_" in name}
    if current.keys() != restored.keys() or any(not torch.equal(current[name], restored[name]) for name in current):
        raise GrpoConfigError("reloaded adapter weights do not match the trained adapter")
    state_path = _latest_trainer_state(output_dir)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if int(state.get("global_step", 0)) < max_steps:
        raise GrpoConfigError("trainer state did not record the requested updates")
    payload = {
        "mode": "resume" if resume_from else "smoke",
        "optimizes": True,
        "reference_policy": "beta_zero",
        "beta": 0.0,
        "global_step": int(state["global_step"]),
        "trainable_changed": len(trained_changed),
        "frozen_changed": 0,
        "optimizer_parameters": len(optimizer_names),
        "fixture_max_completion_length": SMOKE_COMPLETION_LENGTH,
        "study_max_completion_length": config.max_completion_length,
        "baseline_loaded": False,
    }
    _write_json(output_dir / "effective_config.json", config.public_dict())
    _write_json(completed_marker(output_dir), payload)
    return payload


def _latest_trainer_state(output_dir: Path) -> Path:
    checkpoints = sorted(output_dir.glob("checkpoint-*/trainer_state.json"))
    if not checkpoints:
        raise GrpoConfigError("training finished without a trainer state checkpoint")
    return checkpoints[-1]


def _require_fresh(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise GrpoConfigError("smoke output already exists; refusing to replace it")


def guard_gpu_output(output_dir: Path, project_root: Path) -> Path:
    """Keep a GPU smoke inside results/rl_v2 and away from the CPU smoke and adapters."""
    resolved = output_dir.resolve()
    allowed = (project_root.resolve() / "results" / "rl_v2").resolve()
    if allowed not in resolved.parents:
        raise GrpoConfigError("GPU smoke must write inside results/rl_v2")
    if resolved.name == "grpo_cpu_smoke" or "grpo_cpu_smoke" in resolved.parts:
        raise GrpoConfigError("GPU smoke must not overlap grpo_cpu_smoke")
    if "adapters" in resolved.parts:
        raise GrpoConfigError("GPU smoke must not replace an adapter")
    if resolved.exists():
        raise GrpoConfigError("GPU smoke output already exists; refusing to replace it")
    return resolved


def run_gpu_smoke(config: GrpoV2Config, project_root: Path, output_dir: Path, *, max_steps: int = 1) -> dict:
    """One CUDA LoRA update on the 1-layer GPT-2 fixture. This does not enter _fit."""
    if config.schema_version != GRPO_GPU_PROFILE:
        raise GrpoConfigError("run_gpu_smoke requires the grpo_v2_gpu profile")
    if os.environ.get("LEXSI_RL_GPU_CHILD") != "1":
        raise GrpoConfigError("live CUDA smoke runs only in the masked child process")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise GrpoConfigError("masked child is missing CUDA_VISIBLE_DEVICES")
    if max_steps < 1:
        raise GrpoConfigError("smoke requires at least one update")
    output_dir = guard_gpu_output(output_dir, project_root)
    from actproto.training.rl_device import DeviceSelectionError
    try:
        payload = _gpu_attempt(config, output_dir, max_steps=max_steps, bf16=True)
        payload["update_dtype"] = "bf16"
        payload["bf16_failure"] = None
    except (DeviceSelectionError, GrpoConfigError):
        raise
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        _clear_gpu_partial(output_dir, project_root)
        payload = _gpu_attempt(config, output_dir, max_steps=max_steps, bf16=False)
        payload["update_dtype"] = "fp32"
        payload["bf16_failure"] = failure
        payload["tiny_aya_profile_bf16"] = "unproven"
    payload["mode"] = "gpu-smoke"
    payload["schema_version"] = GRPO_GPU_PROFILE
    payload["reference_policy"] = "beta_zero"
    payload["study_claimed"] = False
    payload["baseline_loaded"] = False
    payload["profile_bf16"] = True
    _write_json(output_dir / "effective_config.json", config.public_dict())
    _write_json(completed_marker(output_dir), payload)
    return payload


def _clear_gpu_partial(output_dir: Path, project_root: Path) -> None:
    resolved = output_dir.resolve()
    allowed = (project_root.resolve() / "results" / "rl_v2" / "grpo_gpu_smoke").resolve()
    if resolved != allowed:
        raise GrpoConfigError("refusing to clear a GPU smoke directory that is not grpo_gpu_smoke")
    if resolved.exists():
        shutil.rmtree(resolved)


def _gpu_attempt(config: GrpoV2Config, output_dir: Path, *, max_steps: int, bf16: bool) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, TrainerCallback

    from actproto.integrations.agenttune_backend import AgenticRolloutAdapter, create_trainer
    from actproto.training.rl_device import DeviceRecord, normalize_uuid, select_5060_ti

    records = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        records.append(DeviceRecord(index, props.name, normalize_uuid(props.uuid), int(props.total_memory)))
    chosen_index = select_5060_ti(records)
    chosen = next(record for record in records if record.index == chosen_index)
    torch.cuda.set_device(chosen.index)

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("CohereLabs/tiny-aya-global", local_files_only=True)
    actions = [
        '{"name":"calculator","arguments":{"expression":"1+1"}}',
        '{"name":"finish","arguments":{"answer":"2"}}',
    ]

    def policy(prompt_ids, trainer):
        del trainer
        text = tokenizer.decode(prompt_ids)
        action = actions[1] if "Tool result:" in text else actions[0]
        sampled = tokenizer.encode(action, add_special_tokens=False)
        return action, sampled, [-0.4] * len(sampled)

    def execute(name, arguments):
        if name == "calculator" and arguments.get("expression") == "1+1":
            return "2"
        if name == "finish":
            return arguments.get("answer", "")
        raise ValueError("smoke tool received an unexpected call")

    adapter = AgenticRolloutAdapter(tokenizer, policy, execute, max_decisions=2)
    base_config = GPT2Config(
        vocab_size=len(tokenizer), n_layer=1, n_head=1, n_embd=8, n_positions=1024,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(base_config).to(torch.device("cuda", chosen.index))
    lora = LoraConfig(r=4, lora_alpha=8, target_modules=["c_attn"], lora_dropout=0.0, bias="none",
                      task_type="CAUSAL_LM", fan_in_fan_out=True)

    class Probe(TrainerCallback):
        def __init__(self):
            self.finite = False
            self.nonzero = False

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            del args, state, control
            total = 0.0
            for parameter in kwargs["model"].parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    value = float(parameter.grad.detach().float().abs().sum())
                    if value == value and value != float("inf"):
                        self.finite = True
                    total += value
            self.nonzero = total > 0

    probe = Probe()
    torch.manual_seed(config.seed)
    wrapper = create_trainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=lambda completions, **kwargs: [1.0 if index % 2 == 0 else 0.0 for index, _ in enumerate(completions)],
        train_dataset=Dataset.from_dict({"prompt": ["calculate 1+1", "calculate 1+1"]}),
        rollout_func=adapter,
        peft_config=lora,
        callbacks=[probe],
        output_dir=str(output_dir),
        num_generations=config.num_generations,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_steps=max_steps,
        warmup_steps=0,
        logging_steps=1,
        save_steps=1,
        save_strategy="steps",
        max_completion_length=SMOKE_COMPLETION_LENGTH,
        temperature=config.temperature,
        top_p=config.top_p,
        seed=config.seed,
        lr_scheduler_type="constant",
        bf16=bf16,
        fp16=False,
        report_to="none",
        use_cpu=False,
        gpu_profile=GRPO_GPU_PROFILE,
        use_vllm=False,
        beta=0.0,
        gradient_checkpointing=config.gradient_checkpointing,
        disable_tqdm=True,
    )
    wrapper.setup_data()
    wrapper.setup_trainer()
    trainer = wrapper.trainer
    if trainer is None:
        raise GrpoConfigError("backend did not construct a trainer")
    trained = trainer.model
    if getattr(trainer, "ref_model", None) is not None or float(trainer.beta) != 0.0:
        raise GrpoConfigError("beta_zero smoke constructed a reference model")
    before = {name: parameter.detach().cpu().clone() for name, parameter in trained.named_parameters()}
    trainer.train()
    named = {id(parameter): name for name, parameter in trained.named_parameters()}
    optimizer_names = [
        named.get(id(parameter), "")
        for group in trainer.optimizer.param_groups for parameter in group["params"]
    ]
    if not probe.finite or not probe.nonzero:
        raise GrpoConfigError("smoke update did not produce a finite nonzero gradient")
    if not optimizer_names or any("lora_" not in name for name in optimizer_names):
        raise GrpoConfigError("optimizer contains a parameter outside the LoRA adapters")
    frozen_changed = [
        name for name, parameter in trained.named_parameters()
        if not parameter.requires_grad and not torch.equal(before[name], parameter.detach().cpu())
    ]
    trained_changed = [
        name for name, parameter in trained.named_parameters()
        if parameter.requires_grad and not torch.equal(before[name], parameter.detach().cpu())
    ]
    if frozen_changed or not trained_changed:
        raise GrpoConfigError("smoke changed frozen weights or left every trainable weight unchanged")
    trainer.save_model(str(output_dir))
    reloaded = PeftModel.from_pretrained(GPT2LMHeadModel(base_config), str(output_dir))
    current = {name: parameter.detach().cpu() for name, parameter in trained.named_parameters() if "lora_" in name}
    restored = {name: parameter.detach().cpu() for name, parameter in reloaded.named_parameters() if "lora_" in name}
    if current.keys() != restored.keys() or any(not torch.equal(current[name], restored[name]) for name in current):
        raise GrpoConfigError("reloaded adapter weights do not match the trained adapter")
    state_path = _latest_trainer_state(output_dir)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if int(state.get("global_step", 0)) < max_steps:
        raise GrpoConfigError("trainer state did not record the requested updates")
    return {
        "optimizes": True,
        "beta": 0.0,
        "global_step": int(state["global_step"]),
        "trainable_changed": len(trained_changed),
        "frozen_changed": 0,
        "optimizer_parameters": len(optimizer_names),
        "fixture_max_completion_length": SMOKE_COMPLETION_LENGTH,
        "study_max_completion_length": config.max_completion_length,
        "device_uuid": chosen.uuid,
        "reference_model_built": False,
    }


def _require_baseline(project_root: Path, config: GrpoV2Config) -> dict:
    if not config.baseline_manifest:
        raise GrpoConfigError("frozen SFT baseline manifest is required")
    path = project_root / config.baseline_manifest
    if not path.is_file():
        raise GrpoConfigError("frozen SFT baseline manifest is missing; Plan 9 S7 has not selected a checkpoint")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("frozen") is not True or not manifest.get("adapter_path") or not manifest.get("adapter_sha256"):
        raise GrpoConfigError("baseline manifest must be frozen and name an adapter path and hash")
    adapter = Path(manifest["adapter_path"])
    if not adapter.is_dir():
        raise GrpoConfigError("baseline adapter path does not exist")
    digest = sha256()
    files = sorted(path for path in adapter.rglob("*") if path.is_file())
    if not files:
        raise GrpoConfigError("baseline adapter is empty")
    for file in files:
        digest.update(file.relative_to(adapter).as_posix().encode())
        digest.update(file.read_bytes())
    if digest.hexdigest() != manifest["adapter_sha256"]:
        raise GrpoConfigError("baseline adapter hash does not match the manifest")
    return manifest


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
