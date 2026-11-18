#!/usr/bin/env python3
"""Dump GRPO rollout diagnostics without a full train (Colab / laptop friendly).

CPU modes:
  --prompt-diff          train vs eval public prompt surfaces
  --fixture-rollouts     3–4 synthetic multi-turn dumps + ledger (no GPU)
  --explain-reward       document why -0.1 means one invalid attempt / k=0

GPU modes (Tiny Aya + optional e2 adapter; requires --allow-any-gpu on Colab):
  --live-rollouts        sample real policy rollouts and dump ledger fields
  --manifest PATH        use sealed tooluse_v2 train tasks (GRPO train set)
  --task-ids id1,id2     optional explicit subset; else stratified sample
  --truncation-stats     summarize token-cap / completion_reason from those dumps

The RTX 5060 Ti UUID pin fails closed on Colab. Pass --allow-any-gpu (or
LEXSI_ALLOW_ANY_GPU=1). That override is DIAGNOSTIC-ONLY.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from actproto.experiments.contracts import TaskSpec  # noqa: E402
from actproto.training.rollout_diagnostics import (  # noqa: E402
    demo_task,
    diff_train_vs_eval_prompts,
    dump_behavior_rollout,
    explain_minus_point_one,
    fixture_parse_error_policy,
    fixture_success_policy,
    prompt_diff_as_dict,
    rollout_dump_as_dict,
    select_manifest_tasks,
    summarize_rollout_reasons,
    truncation_stat_row,
)
from actproto.training.rl_device import (  # noqa: E402
    ALLOW_ANY_GPU_ENV,
    DIAGNOSTIC_GPU_WARNING,
    DeviceRecord,
    allow_any_gpu_enabled,
    normalize_uuid,
    select_cuda_device,
)


def _fake_scope(task: TaskSpec):
    """Manifest-shaped scope so run_trusted_behavior_episode accepts the factory."""
    from actproto.training.rl_dataset import VerifiedTrainFixtures

    class Env:
        identifier = task.environment_id
        digest = task.fixture_hash

        def call(self, environment_id, name, arguments):
            del environment_id
            if name == "calculator":
                expr = str(arguments.get("expression", ""))
                if expr.replace(" ", "") == "1+1":
                    return {"ok": True, "data": {"value": 2}}
                return {"ok": False, "error": {"type": "bad_expr", "message": expr}}
            if name in {"finish", "no_tool"}:
                answer = str(arguments.get("answer", ""))
                return {"ok": True, "data": {"answer": answer, "done": True}}
            return {"ok": False, "error": {"type": "unknown_tool", "message": name}}

    scope = object.__new__(VerifiedTrainFixtures)
    scope._tasks = {task.task_id: task}

    def for_task(bound):
        if scope._tasks.get(bound.task_id) != bound:
            raise ValueError("task is not bound to the diagnostic scope")
        return Env()

    scope.for_task = for_task
    return scope


class _CharTokenizer:
    """CPU tokenizer stand-in for fixture dumps (no HF download)."""

    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, chat_template=None):
        del chat_template
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return list(map(ord, text)) if tokenize else text

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(map(ord, text)) or [ord("x")]

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return "".join(chr(i) for i in token_ids if isinstance(i, int) and i >= 0)


def run_prompt_diff(task: TaskSpec) -> dict:
    return prompt_diff_as_dict(diff_train_vs_eval_prompts(task))


def run_fixture_rollouts(count: int = 4) -> dict:
    task = demo_task()
    tokenizer = _CharTokenizer()
    scope = _fake_scope(task)
    from actproto.training.rollout_diagnostics import RolloutDump

    policies = [
        fixture_success_policy(tokenizer),
        fixture_success_policy(tokenizer),
        fixture_parse_error_policy(tokenizer),
        fixture_success_policy(tokenizer),
    ]
    dumps: list[RolloutDump] = []
    for policy in policies[: max(3, min(count, 4))]:
        dumps.append(
            dump_behavior_rollout(
                task,
                tokenizer=tokenizer,
                policy=policy,
                environment_factory=lambda: scope,
                max_generated_tokens=256,
                deadline_seconds=30.0,
            )
        )
    return {
        "mode": "fixture_rollouts",
        "gpu_required": False,
        "episode_kind": "multi_turn_behavior_tool_rollout",
        "note": (
            "Training GRPO uses the same multi-turn BehaviorToolRollout path. "
            "The parse_error fixture shows invalid_attempts=1 → reward -0.1."
        ),
        "rollouts": [rollout_dump_as_dict(d) for d in dumps],
        "truncation_stats": [truncation_stat_row(d) for d in dumps],
        "summary": summarize_rollout_reasons(dumps),
        "reward_legend": explain_minus_point_one(),
    }


def _resolve_device(*, allow_any_gpu: bool):
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for --live-rollouts; use --fixture-rollouts on CPU")
    records = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        records.append(
            DeviceRecord(index, props.name, normalize_uuid(props.uuid), int(props.total_memory))
        )
    index = select_cuda_device(records, allow_any_gpu=allow_any_gpu)
    chosen = next(r for r in records if r.index == index)
    if allow_any_gpu_enabled(allow_any_gpu):
        print(DIAGNOSTIC_GPU_WARNING, file=sys.stderr)
    print(f"device_uuid={chosen.uuid} device_name={chosen.name}", file=sys.stderr)
    import torch as _torch

    device = _torch.device("cuda", chosen.index)
    _torch.cuda.set_device(device)
    return device


def _load_manifest_tasks(
    manifest: Path,
    *,
    count: int,
    seed: int,
    task_ids: list[str] | None,
):
    from actproto.training.rl_dataset import load_scoped_train_manifest

    if not manifest.is_file():
        raise SystemExit(f"train manifest not found: {manifest}")
    scope, tasks = load_scoped_train_manifest(manifest)
    selected = select_manifest_tasks(tasks, count=count, seed=seed, task_ids=task_ids)
    return scope, selected


def run_live_rollouts(
    *,
    adapter: Path | None,
    count: int,
    allow_any_gpu: bool,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    output: Path | None,
    manifest: Path | None,
    seed: int,
    task_ids: list[str] | None,
) -> dict:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from actproto.training.grpo_run import PINNED_MODEL_ID, PINNED_REVISION, _sample_peft_action
    from actproto.training.rl_rollouts import PolicySample
    from actproto.training.sft_masks import selected_template

    device = _resolve_device(allow_any_gpu=allow_any_gpu)
    tokenizer = AutoTokenizer.from_pretrained(PINNED_MODEL_ID, revision=PINNED_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        PINNED_MODEL_ID, revision=PINNED_REVISION, dtype=torch.bfloat16,
    )
    model = model.to(device)
    if adapter is not None:
        if not adapter.is_dir():
            raise SystemExit(f"adapter path does not exist: {adapter}")
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
        model = model.to(device)
    model.eval()
    template = selected_template(tokenizer, "default")

    if manifest is not None:
        scope, tasks = _load_manifest_tasks(
            manifest, count=count, seed=seed, task_ids=task_ids,
        )
        env_factory = lambda: scope
        task_source = str(manifest)
    else:
        task = demo_task()
        tasks = [task]
        scope = _fake_scope(task)
        env_factory = lambda: scope
        task_source = "demo_task"

    def policy_factory(seed_value: int):
        seeded = False

        def policy(prompt_ids, remaining):
            nonlocal seeded
            if not seeded:
                torch.manual_seed(int(seed_value) % (2 ** 31))
                seeded = True
            text, token_ids, logprobs, truncated = _sample_peft_action(
                model, tokenizer, prompt_ids,
                temperature=temperature, top_p=top_p, max_new_tokens=int(remaining),
            )
            return PolicySample(text, token_ids, logprobs, truncated)

        return policy

    dumps = []
    # One rollout per selected task (manifest), or count repeats of the demo task.
    if manifest is not None:
        schedule = list(enumerate(tasks))
    else:
        schedule = [(index, tasks[0]) for index in range(max(3, min(count, 4)))]

    for index, task in schedule:
        print(
            f"rollout {index + 1}/{len(schedule)} task_id={task.task_id}",
            file=sys.stderr,
            flush=True,
        )
        dump = dump_behavior_rollout(
            task,
            tokenizer=tokenizer,
            policy=policy_factory(seed + index),
            environment_factory=env_factory,
            max_generated_tokens=max_new_tokens,
            deadline_seconds=180.0,
            chat_template=template,
        )
        dumps.append(dump)
        print(
            f"  reason={dump.completion_reason} reward={dump.ledger.get('reward')} "
            f"invalid={dump.ledger.get('invalid_attempts')} tokens={dump.completion_token_count}",
            file=sys.stderr,
            flush=True,
        )

    summary = summarize_rollout_reasons(dumps)
    payload = {
        "mode": "live_rollouts",
        "gpu_required": True,
        "allow_any_gpu": bool(allow_any_gpu or allow_any_gpu_enabled()),
        "adapter": str(adapter) if adapter else None,
        "manifest": task_source,
        "seed": seed,
        "task_ids": [task.task_id for task in (tasks if manifest is not None else [tasks[0]] * len(dumps))],
        "model_id": PINNED_MODEL_ID,
        "revision": PINNED_REVISION,
        "temperature": temperature,
        "top_p": top_p,
        "max_generated_tokens": max_new_tokens,
        "episode_kind": "multi_turn_behavior_tool_rollout",
        "rollouts": [rollout_dump_as_dict(d) for d in dumps],
        "truncation_stats": [truncation_stat_row(d) for d in dumps],
        "summary": summary,
        "reward_legend": explain_minus_point_one(),
        "prompt_diff": run_prompt_diff(tasks[0]),
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        payload["wrote"] = str(output)
        summary_path = output.with_name(output.stem + "_summary.json")
        summary_path.write_text(
            json.dumps(
                {
                    "summary": summary,
                    "task_ids": payload["task_ids"],
                    "manifest": task_source,
                    "adapter": payload["adapter"],
                    "temperature": temperature,
                    "top_p": top_p,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        payload["wrote_summary"] = str(summary_path)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prompt-diff", action="store_true")
    mode.add_argument("--fixture-rollouts", action="store_true")
    mode.add_argument("--explain-reward", action="store_true")
    mode.add_argument("--live-rollouts", action="store_true")
    parser.add_argument("--adapter", type=Path, help="local e2 checkpoint dir (PEFT adapter)")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="tooluse_v2 manifest.json (GRPO train_manifest); samples train tasks",
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        default="",
        help="comma-separated task ids (optional; otherwise stratified sample)",
    )
    parser.add_argument("--seed", type=int, default=42, help="sampling seed for task selection / generate")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    parser.add_argument(
        "--allow-any-gpu",
        action="store_true",
        help="DIAGNOSTIC-ONLY: bypass RTX 5060 Ti UUID pin for Colab/T4/L4",
    )
    args = parser.parse_args(argv)
    if args.allow_any_gpu:
        os.environ[ALLOW_ANY_GPU_ENV] = "1"
        print(DIAGNOSTIC_GPU_WARNING, file=sys.stderr)

    task_ids = [part.strip() for part in args.task_ids.split(",") if part.strip()] or None

    if args.prompt_diff:
        if args.manifest is not None:
            _scope, tasks = _load_manifest_tasks(
                args.manifest, count=1, seed=args.seed, task_ids=task_ids,
            )
            payload = run_prompt_diff(tasks[0])
        else:
            payload = run_prompt_diff(demo_task())
    elif args.explain_reward:
        payload = explain_minus_point_one()
    elif args.fixture_rollouts:
        payload = run_fixture_rollouts(args.count)
    else:
        payload = run_live_rollouts(
            adapter=args.adapter,
            count=args.count,
            allow_any_gpu=args.allow_any_gpu,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            output=args.output,
            manifest=args.manifest,
            seed=args.seed,
            task_ids=task_ids,
        )
    if args.output and not args.live_rollouts:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        payload = {**payload, "wrote": str(args.output)}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
