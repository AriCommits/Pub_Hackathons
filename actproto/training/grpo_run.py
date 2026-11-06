"""GRPO lifecycle. Plan and collect modes do not optimize. Named GPU training is fail-closed; CPU training stays closed."""
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


PINNED_MODEL_ID = "CohereLabs/tiny-aya-global"
PINNED_REVISION = "af89d219b53ed9b13b8a4645f9c8028973510324"


def assert_loaded_e2_policy(model, config) -> None:
    """The trainable policy is the frozen e2 LoRA. Beta 0 builds no reference."""
    from actproto.training.rl_config import E2_LORA_TARGETS, TOOL_REWARD_LEARNING_RATE

    if config.reference_policy != "beta_zero" or float(config.beta) != 0.0 or float(config.learning_rate) != TOOL_REWARD_LEARNING_RATE:
        raise GrpoConfigError("the tool-reward run keeps the e2 LoRA, learning rate 5e-6, and beta 0")
    peft_config = getattr(model, "peft_config", None)
    if not isinstance(peft_config, dict) or "default" not in peft_config:
        raise GrpoConfigError("training must load the existing adapter, not a new LoRA on the base model")
    if "ref" in peft_config:
        raise GrpoConfigError("beta 0 must not add a ref adapter; TRL 1.7.1 would KL-regularize toward the base model")
    targets = set(getattr(peft_config["default"], "target_modules", ()))
    if targets != set(E2_LORA_TARGETS):
        raise GrpoConfigError("trainable LoRA targets do not match the frozen e2 adapter")
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable):
        raise GrpoConfigError("only the loaded LoRA matrices are trainable")


def run_train(
    config: GrpoV2Config,
    project_root: Path,
    *,
    output_dir: Path | None = None,
    max_steps: int | None = None,
) -> dict:
    """Train Tiny Aya from the frozen adapter on the named GPU profile.

    CPU profiles still stop after the baseline and dataset checks. Beta 0 does
    not construct an SFT reference model.
    """
    manifest = _require_baseline(project_root, config)
    if not config.train_manifest:
        raise GrpoConfigError("real training requires a nonempty train manifest")
    train_path = project_root / config.train_manifest
    if not train_path.is_file():
        raise GrpoConfigError("train manifest is missing")
    from actproto.training.rl_dataset import load_scoped_train_manifest

    _scope, tasks = load_scoped_train_manifest(train_path)
    rows = public_prompt_rows(tasks)
    if any(set(row) != {"prompt"} for row in rows):
        raise GrpoConfigError("training prompts leaked private oracle data")
    adapter = Path(manifest["adapter_path"])
    output = output_dir if output_dir is not None else resolve_output(config.output_dir, project_root)
    if output.resolve() == adapter.resolve() or adapter.resolve() in output.resolve().parents:
        raise GrpoConfigError("training output must not overwrite the frozen SFT adapter")
    if config.schema_version != GRPO_GPU_PROFILE or config.use_cpu:
        raise GrpoConfigError(
            "real GRPO optimization is blocked until the frozen SFT baseline is loaded on a verified GPU configuration; "
            "beta_zero is not an SFT reference"
        )
    if os.environ.get("LEXSI_RL_GPU_CHILD") != "1":
        raise GrpoConfigError("live CUDA training runs only in the masked child process")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise GrpoConfigError("masked child is missing CUDA_VISIBLE_DEVICES")
    steps = config.max_steps if max_steps is None else max_steps
    if steps < 1:
        raise GrpoConfigError("training requires at least one update")
    output = guard_gpu_output(output, project_root)
    return _train_tiny_aya(
        config,
        project_root,
        output,
        rows,
        adapter,
        max_steps=steps,
        adapter_sha256=str(manifest["adapter_sha256"]),
        tasks=tasks,
        train_manifest=train_path,
    )


def _train_payload(*, global_step: int, baseline_loaded: bool, reference_model_built: bool) -> dict:
    """Record beta 0 as the absence of a KL reference, and only after the adapter loaded."""
    if baseline_loaded is not True:
        raise GrpoConfigError("beta_zero training must load the frozen adapter")
    if reference_model_built:
        raise GrpoConfigError("beta_zero training must not build a reference model")
    if global_step < 1:
        raise GrpoConfigError("training finished without an optimizer step")
    return {
        "mode": "train",
        "optimizes": True,
        "reference_policy": "beta_zero",
        "beta": 0.0,
        "baseline_loaded": True,
        "reference_model_built": False,
        "global_step": global_step,
    }


HISTORICAL_RUNS = {"grpo_gpu_e2", "grpo_gpu_smoke", "grpo_cpu_smoke"}
TRAIN_EPISODE_DEADLINE_SECONDS = 120.0


def refuse_historical_output(output_dir: Path) -> None:
    """Keep a tool-reward run from replacing the frozen e2 run or the smokes."""
    if output_dir.name in HISTORICAL_RUNS:
        raise GrpoConfigError("tool-reward training must use a new results/rl_v2 directory")


def _sample_peft_action(model, tokenizer, prompt_ids, *, temperature, top_p, max_new_tokens):
    """Sample one decision from the loaded policy and return its sampling logprobs.

    ``output.scores`` are the post-warper logits from ``generate``, so the log
    probability is a log-softmax of those scores. Temperature and top-p are
    applied by ``generate`` and are not applied a second time here.
    """
    import torch

    from actproto.agent.model import _parameter_device, _tool_call_stopping_criteria

    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 256:
        raise GrpoConfigError("max_new_tokens must be between 1 and 256")
    if temperature <= 0 or top_p <= 0 or top_p > 1:
        raise GrpoConfigError("sampling temperature and top_p are outside the study range")
    device = _parameter_device(model)
    input_ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    attention = torch.ones_like(input_ids)
    with torch.inference_mode():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=_tool_call_stopping_criteria(tokenizer, int(input_ids.shape[-1])),
        )
    new_ids = output.sequences[0, input_ids.shape[-1]:].tolist()
    scores = tuple(output.scores)
    kept: list[int] = []
    logprobs: list[float] = []
    for index, token_id in enumerate(new_ids):
        if token_id == tokenizer.eos_token_id or token_id == tokenizer.pad_token_id or index >= len(scores):
            break
        distribution = torch.log_softmax(scores[index][0].float(), dim=-1)
        logprobs.append(float(distribution[token_id]))
        kept.append(int(token_id))
    if not kept and new_ids and scores:
        token_id = int(new_ids[0])
        distribution = torch.log_softmax(scores[0][0].float(), dim=-1)
        value = float(distribution[token_id])
        if value == value and abs(value) != float("inf"):
            kept = [token_id]
            logprobs = [value]
    if not kept or not all(value == value and abs(value) != float("inf") for value in logprobs):
        raise GrpoConfigError("policy sample produced no finite tokens")
    text = tokenizer.decode(kept, skip_special_tokens=False if len(kept) == 1 and kept[0] == tokenizer.eos_token_id else True)
    return text, kept, logprobs, len(kept) >= max_new_tokens and kept[-1] != tokenizer.eos_token_id


STUDY_TURN_NEW_TOKENS = 256


def _study_rollout(model, tokenizer, tasks, scope, config, run_id: str):
    """Build the in-process tool rollout. The sampler is the loaded policy.

    ``max_completion_length`` stays the TRL loss denominator. Each sampled turn
    receives ``STUDY_TURN_NEW_TOKENS`` new tokens and the named ``default``
    chat template. The episode does not share one token pool across turns.
    """
    from actproto.training.rl_rollouts import BehaviorToolRollout, PolicySample
    from actproto.training.sft_masks import selected_template

    if config.max_completion_length > STUDY_TURN_NEW_TOKENS:
        raise GrpoConfigError("max_completion_length must stay at most 256")
    template = selected_template(tokenizer, "default")

    def active_model(trainer):
        return model if trainer is None or getattr(trainer, "model", None) is None else trainer.model

    def context_limit_reader(trainer):
        active = active_model(trainer)
        limit = getattr(getattr(active, "config", None), "max_position_embeddings", None)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise GrpoConfigError("model context limit unavailable")
        return limit

    def policy_factory(local_seed, trainer):
        del local_seed
        active = active_model(trainer)

        def policy(prompt_ids, remaining):
            # ``remaining`` is the per-turn cap from the episode, not a shrinking pool.
            text, token_ids, logprobs, truncated = _sample_peft_action(
                active, tokenizer, prompt_ids,
                temperature=config.temperature, top_p=config.top_p,
                max_new_tokens=int(remaining),
            )
            return PolicySample(text, token_ids, logprobs, truncated)

        return policy

    return BehaviorToolRollout(
        tokenizer, tasks, policy_factory, lambda: scope,
        deadline_seconds=TRAIN_EPISODE_DEADLINE_SECONDS,
        max_generated_tokens=STUDY_TURN_NEW_TOKENS,
        seed=config.seed,
        run_id=run_id,
        chat_template=template,
        context_limit_reader=context_limit_reader,
    )


def _executed_trace_reward(completions, reward_sessions=None, completion_reasons=None, group_size=2, **kwargs):
    """Score captured tool sessions with ``bounded_prefix_v1``.

    A decoded completion that merely contains "finish" is not success. The
    completion text is ignored. Serialized rollout JSON is not evidence, and
    no token-length bonus is added. A session that already has a ledger is
    read once and is not scored again.
    """
    del kwargs
    from actproto.training.rewards_v2 import PREFIX_VERSION, TrustedRewardSession

    if not reward_sessions:
        rewards = [0.0 for _ in completions]
        _remember_step(rewards, group_size)
        return rewards
    if any(isinstance(session, (str, bytes, dict, list)) for session in reward_sessions):
        raise GrpoConfigError("serialized rollout JSON is not execution evidence")
    if completion_reasons is None or len(completion_reasons) != len(reward_sessions) or len(reward_sessions) != len(completions):
        raise GrpoConfigError("each executed trace needs one completion reason")
    rewards = []
    for session, reason in zip(reward_sessions, completion_reasons):
        if session is None:
            rewards.append(0.0)
            continue
        if not isinstance(session, TrustedRewardSession):
            raise GrpoConfigError("training reward requires a live trusted session")
        existing = session.finalized_ledger()
        if existing is not None:
            rewards.append(existing.reward)
            continue
        rewards.append(session.score(completion_reason=str(reason), version=PREFIX_VERSION).reward)
    _remember_step(rewards, group_size)
    return rewards


_LAST_STEP: dict = {}


def reward_step_record(rewards: list[float], group_size: int, grad_norm: float | None = None) -> dict:
    """Record a GRPO step. A tied group has advantage 0 and is not given a fake gradient."""
    from actproto.training.rewards_v2 import group_advantages

    if group_size < 2 or len(rewards) < group_size or len(rewards) % group_size:
        raise GrpoConfigError("rewards must form complete groups of at least 2")
    advantages: list[float] = []
    zero = 0
    groups = [rewards[start:start + group_size] for start in range(0, len(rewards), group_size)]
    for group in groups:
        values, info = group_advantages(group)
        advantages.extend(values)
        zero += int(info["zero_variance_groups"])
    center = sum(rewards) / len(rewards)
    variance = sum((item - center) ** 2 for item in rewards) / len(rewards)
    all_tied = zero == len(groups)
    if all_tied and any(value != 0.0 for value in advantages):
        raise GrpoConfigError("a tied group produced a nonzero advantage")
    return {
        "reward_mean": center,
        "reward_std": variance ** 0.5,
        "grad_norm": grad_norm,
        "zero_variance_groups": zero,
        "groups": len(groups),
        "advantages": advantages,
        "all_groups_tied": all_tied,
    }


def tied_step_leaves_weights(weights: dict, rewards: list[float], group_size: int) -> tuple[dict, dict]:
    """A fully tied step records zero variance and does not change frozen weights."""
    record = reward_step_record(rewards, group_size, grad_norm=0.0)
    if not record["all_groups_tied"]:
        raise GrpoConfigError("this helper only covers a step whose groups are all tied")
    return dict(weights), record


def _remember_step(rewards: list[float], group_size: int) -> None:
    if group_size < 2 or len(rewards) < group_size or len(rewards) % group_size:
        return
    _LAST_STEP.clear()
    _LAST_STEP.update(reward_step_record(rewards, group_size))


def format_step_log(logs: dict, diagnostics: dict) -> str:
    """One step line: reward mean, reward std, gradient norm, and zero-variance groups."""
    reward_mean = logs.get("reward", diagnostics.get("reward_mean"))
    reward_std = logs.get("reward_std", diagnostics.get("reward_std"))
    grad_norm = logs.get("grad_norm", diagnostics.get("grad_norm"))
    zero = diagnostics.get("zero_variance_groups", 0)
    groups = diagnostics.get("groups", 0)
    text = f"reward_mean={reward_mean} reward_std={reward_std} grad_norm={grad_norm} zero_variance_groups={zero}"
    if groups and zero == groups:
        text += " advantages=0 all_groups_tied=true"
    return text


def _is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text


def _record_memory_error(output_dir: Path, exc: BaseException, *, baseline_loaded: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "memory_error.json", {
        "error": f"{type(exc).__name__}: {exc}",
        "baseline_loaded": baseline_loaded,
        "reference_policy": "beta_zero",
        "reference_model_built": False,
        "retried": False,
    })
    print(f"CUDA_OOM baseline_loaded={str(baseline_loaded).lower()} {type(exc).__name__}: {exc}", flush=True)


def _train_tiny_aya(
    config: GrpoV2Config,
    project_root: Path,
    output_dir: Path,
    rows: list[dict],
    adapter_path: Path,
    *,
    max_steps: int,
    adapter_sha256: str,
    tasks: list,
    train_manifest: Path,
) -> dict:
    """One bounded bf16 GRPO run from the frozen adapter. An OOM is recorded once."""
    from actproto.training.rl_config import guard_tool_sampling

    refuse_historical_output(output_dir)
    guard_tool_sampling(config)
    import yaml

    sft = yaml.safe_load((project_root / "actproto" / "configs" / "sft_v2.yaml").read_text(encoding="utf-8"))
    if sft.get("model_id") != PINNED_MODEL_ID or sft.get("revision") != PINNED_REVISION:
        raise GrpoConfigError("Tiny Aya revision does not match the pinned study model")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

    from actproto.integrations.agenttune_backend import create_trainer
    from actproto.training.rl_device import DeviceRecord, normalize_uuid, select_5060_ti

    records = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        records.append(DeviceRecord(index, props.name, normalize_uuid(props.uuid), int(props.total_memory)))
    chosen_index = select_5060_ti(records)
    chosen = next(record for record in records if record.index == chosen_index)
    device = torch.device("cuda", chosen.index)
    torch.cuda.set_device(device)
    print(f"device_uuid={chosen.uuid} device_name={chosen.name}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_loaded = False
    try:
        tokenizer = AutoTokenizer.from_pretrained(PINNED_MODEL_ID, revision=PINNED_REVISION, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            PINNED_MODEL_ID, revision=PINNED_REVISION, dtype=torch.bfloat16, local_files_only=True,
        )
        base.config.use_cache = False
        base = base.to(device)
        model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=True)
        model = model.to(device)
        for name, parameter in model.named_parameters():
            parameter.requires_grad = "lora_" in name
        baseline_loaded = True
    except RuntimeError as exc:
        if _is_cuda_oom(exc):
            _record_memory_error(output_dir, exc, baseline_loaded=baseline_loaded)
            raise GrpoConfigError(f"CUDA out of memory: {exc}") from exc
        raise
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad and "lora_" in name]
    print("baseline_loaded=true reference_policy=beta_zero reference_model_built=false", flush=True)
    if not trainable:
        raise GrpoConfigError("frozen adapter loaded without trainable LoRA weights")
    assert_loaded_e2_policy(model, config)
    print(f"train_rows={len(rows)} trainable_lora={len(trainable)}", flush=True)

    class StepLog(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            del args, control, kwargs
            if not logs:
                return
            line = format_step_log(logs, _LAST_STEP)
            print(f"global_step={state.global_step} {line}", flush=True)
            with (output_dir / "step_diagnostics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({
                    "global_step": state.global_step,
                    "line": line,
                    "reward_mean": _LAST_STEP.get("reward_mean"),
                    "reward_std": _LAST_STEP.get("reward_std"),
                    "grad_norm": logs.get("grad_norm", _LAST_STEP.get("grad_norm")),
                    "zero_variance_groups": _LAST_STEP.get("zero_variance_groups"),
                    "all_groups_tied": _LAST_STEP.get("all_groups_tied"),
                }, sort_keys=True) + "\n")

    from actproto.training.rl_dataset import VerifiedTrainFixtures

    scope = VerifiedTrainFixtures(train_manifest)
    rollout = _study_rollout(model, tokenizer, tasks, scope, config, output_dir.name)
    torch.manual_seed(config.seed)
    wrapper = create_trainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=lambda completions, **kwargs: _executed_trace_reward(
            completions, group_size=config.num_generations, **kwargs,
        ),
        train_dataset=Dataset.from_list(rows),
        rollout_func=rollout,
        callbacks=[StepLog()],
        output_dir=str(output_dir),
        num_generations=config.num_generations,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_steps=max_steps,
        warmup_steps=0,
        logging_steps=1,
        save_steps=max_steps,
        save_strategy="steps",
        max_completion_length=config.max_completion_length,
        temperature=config.temperature,
        top_p=config.top_p,
        seed=config.seed,
        lr_scheduler_type="constant",
        bf16=True,
        fp16=False,
        report_to="none",
        use_cpu=False,
        gpu_profile=GRPO_GPU_PROFILE,
        use_vllm=False,
        beta=0.0,
        gradient_checkpointing=True,
        disable_tqdm=True,
    )
    wrapper.setup_data()
    wrapper.setup_trainer()
    trainer = wrapper.trainer
    if trainer is None:
        raise GrpoConfigError("backend did not construct a trainer")
    reference_built = getattr(trainer, "ref_model", None) is not None or float(trainer.beta) != 0.0
    if reference_built:
        raise GrpoConfigError("beta_zero training constructed a reference model")
    _write_json(output_dir / "started.json", {
        "reference_policy": "beta_zero",
        "beta": 0.0,
        "baseline_loaded": True,
        "reference_model_built": False,
        "model_id": PINNED_MODEL_ID,
        "revision": PINNED_REVISION,
        "adapter_sha256": adapter_sha256,
        "max_steps": max_steps,
        "rows": len(rows),
        "device_uuid": chosen.uuid,
    })
    _write_json(output_dir / "effective_config.json", config.public_dict())
    try:
        trainer.train()
    except RuntimeError as exc:
        if _is_cuda_oom(exc):
            _record_memory_error(output_dir, exc, baseline_loaded=True)
            raise GrpoConfigError(f"CUDA out of memory: {exc}") from exc
        raise
    state_path = _latest_trainer_state(output_dir)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    payload = _train_payload(
        global_step=int(state.get("global_step", 0)),
        baseline_loaded=True,
        reference_model_built=False,
    )
    payload["device_uuid"] = chosen.uuid
    payload["model_id"] = PINNED_MODEL_ID
    payload["revision"] = PINNED_REVISION
    payload["adapter_sha256"] = adapter_sha256
    _write_json(completed_marker(output_dir), payload)
    return payload


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
    files = sorted(path for path in adapter.rglob("*") if path.is_file())
    if not files:
        raise GrpoConfigError("baseline adapter is empty")
    from scripts.run_experiment_v2 import adapter_digest

    try:
        digest = adapter_digest(adapter)
    except ValueError as exc:
        raise GrpoConfigError(str(exc)) from exc
    if digest != manifest["adapter_sha256"]:
        raise GrpoConfigError("baseline adapter hash does not match the manifest")
    return manifest


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
