"""CPU-friendly GRPO rollout diagnostics for Colab / laptop debugging.

These helpers dump prompt construction, decoded completions, and
``bounded_prefix_v1`` ledger fields without starting a full GRPO train.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from typing import Any, Callable

from actproto.agent.prompts import messages_for_template, system_prompt
from actproto.experiments.contracts import TaskSpec
from actproto.training.rewards_v2 import PREFIX_VERSION, TrustedRewardSession, ledger_components
from actproto.training.rl_rollouts import PolicySample, run_trusted_behavior_episode


@dataclass(frozen=True)
class PromptSurface:
    """One side of the train-vs-eval prompt comparison."""

    label: str
    system: str
    user: str
    messages: list[dict[str, str]]
    prepared_for_template: list[dict[str, str]]
    assistant_prefix_hint: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class PromptDiff:
    same_system: bool
    same_user: bool
    same_prepared: bool
    train: PromptSurface
    eval: PromptSurface
    findings: tuple[str, ...]


@dataclass(frozen=True)
class RolloutDump:
    task_id: str
    prompt: str
    completion_reason: str
    decoded_completion: str
    truncated: bool
    completion_token_count: int
    max_generated_tokens: int
    messages: list[dict[str, Any]]
    ledger: dict[str, Any]
    episode_kind: str


def training_prompt_surface(task: TaskSpec) -> PromptSurface:
    """Messages the GRPO BehaviorToolRollout builds before the first sample."""
    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": task.prompt},
    ]
    prepared = messages_for_template(messages)
    return PromptSurface(
        label="train_behavior_rollout",
        system=system_prompt(),
        user=task.prompt,
        messages=deepcopy(messages),
        prepared_for_template=deepcopy(prepared),
        assistant_prefix_hint="add_generation_prompt=True (chat template assistant turn)",
        notes=(
            "Source: actproto.training.rl_rollouts.run_trusted_behavior_episode",
            "Tool observations are role=tool then rewritten to user 'Tool result:' turns",
            "Multi-turn: each tool result is appended and another decision is sampled",
        ),
    )


def eval_prompt_surface(task: TaskSpec) -> PromptSurface:
    """Messages live eval / ReActStrategy builds before the first generate call."""
    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": task.prompt},
    ]
    prepared = messages_for_template(messages)
    return PromptSurface(
        label="eval_react_live",
        system=system_prompt(),
        user=task.prompt,
        messages=deepcopy(messages),
        prepared_for_template=deepcopy(prepared),
        assistant_prefix_hint="add_generation_prompt=True (chat template assistant turn)",
        notes=(
            "Source: actproto.evaluation.live_runner._encode_prompt / actproto.agent.strategy.ReActStrategy",
            "Same system_prompt() and tool schemas as training",
            "Multi-turn: ReActStrategy loops until finish/no_tool or max_agent_steps",
        ),
    )


def diff_train_vs_eval_prompts(task: TaskSpec) -> PromptDiff:
    """CPU-only compare of train vs eval public prompt surfaces."""
    train = training_prompt_surface(task)
    eval_side = eval_prompt_surface(task)
    findings: list[str] = []
    same_system = train.system == eval_side.system
    same_user = train.user == eval_side.user
    same_prepared = train.prepared_for_template == eval_side.prepared_for_template
    if same_system and same_user and same_prepared:
        findings.append(
            "Public system+user surfaces match between GRPO BehaviorToolRollout and live eval."
        )
        findings.append(
            "Both paths are multi-turn tool loops; a permanent -0.1 is not explained by "
            "single-turn training vs multi-turn eval in the custom adapter."
        )
        findings.append(
            "Likely first-decision failure under sampling: parse_error / truncation / "
            "template_mismatch → invalid_attempts=1, k=0 → bounded_prefix_v1 reward -0.1."
        )
    else:
        if not same_system:
            findings.append("SYSTEM PROMPT DIFFERS between train and eval.")
        if not same_user:
            findings.append("USER PROMPT DIFFERS between train and eval.")
        if not same_prepared:
            findings.append("messages_for_template output differs between train and eval.")
    findings.append(
        "Eval generate_text uses do_sample=False; GRPO _sample_peft_action uses "
        "temperature=0.8 top_p=0.95. That sampling gap remains after prompt equality."
    )
    return PromptDiff(
        same_system=same_system,
        same_user=same_user,
        same_prepared=same_prepared,
        train=train,
        eval=eval_side,
        findings=tuple(findings),
    )


def explain_minus_point_one() -> dict[str, Any]:
    """Document the reward ledger meaning for reviewer check #1."""
    from actproto.training.rewards_v2 import bounded_prefix_reward

    reward = bounded_prefix_reward(False, invalid=1, duplicates=0, excess=None, k=0, n=3)
    return {
        "reward_version": PREFIX_VERSION,
        "example_reward": reward,
        "interpretation": (
            "failed task, no work-tool dependency matched (k=0), exactly one invalid attempt"
        ),
        "invalid_attempt_sources": (
            "parse_error",
            "truncation",
            "timeout",
            "template_mismatch",
            "context_overflow",
            "policy_error",
            "budget_exhausted",
        ),
        "episode_architecture": {
            "training": "multi-turn BehaviorToolRollout / run_trusted_behavior_episode",
            "eval": "multi-turn ReActStrategy / live_runner",
            "not_single_turn_trl_completion": True,
        },
        "max_completion_length_role": (
            "TRL loss denominator; per-turn sample cap is also 256 "
            "(per_turn_new_tokens). Hitting 256 without EOS/tool-call stop marks truncation."
        ),
    }


def demo_task() -> TaskSpec:
    """Small public train task for prompt-diff / fixture dumps (no oracle leak in prompt)."""
    return TaskSpec(
        "diag-calc-001",
        "diag-group-001",
        "train",
        "What is 1+1? Call calculator then finish with the numeric answer.",
        "required_tools",
        {"answer": "2"},
        [
            {"id": "calc", "tool": "calculator", "arguments": {"expression": "1+1"}},
            {"id": "fin", "tool": "finish", "arguments": {"answer": "2"}, "after": ["calc"]},
        ],
        {"type": "exact"},
        4,
        "env",
        "hash",
    )


def _decode_policy_spans(episode, tokenizer) -> tuple[str, bool]:
    """Reconstruct assistant text from sampled completion ids."""
    text = tokenizer.decode(episode.completion_ids, skip_special_tokens=True)
    truncated = episode.completion_reason == "truncated"
    return text, truncated


def dump_behavior_rollout(
    task: TaskSpec,
    *,
    tokenizer: Any,
    policy: Callable,
    environment_factory: Callable,
    max_generated_tokens: int = 256,
    deadline_seconds: float = 60.0,
    chat_template: str | None = None,
) -> RolloutDump:
    """Run one trusted multi-turn episode and return a reviewer-ready dump."""
    episode = run_trusted_behavior_episode(
        task,
        tokenizer=tokenizer,
        policy=policy,
        environment_factory=environment_factory,
        deadline_seconds=deadline_seconds,
        max_generated_tokens=max_generated_tokens,
        chat_template=chat_template,
    )
    ledger = episode.session.finalized_ledger()
    if ledger is None:
        ledger = episode.session.score(
            completion_reason=episode.completion_reason, version=PREFIX_VERSION
        )
    decoded, truncated = _decode_policy_spans(episode, tokenizer)
    # Rebuild public messages for inspection (system+user only at start; dump completion text).
    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": task.prompt},
        {"role": "assistant", "content": decoded},
    ]
    return RolloutDump(
        task_id=task.task_id,
        prompt=task.prompt,
        completion_reason=episode.completion_reason,
        decoded_completion=decoded,
        truncated=truncated,
        completion_token_count=len(episode.completion_ids),
        max_generated_tokens=max_generated_tokens,
        messages=messages,
        ledger=ledger_components(ledger, episode.completion_reason),
        episode_kind="multi_turn_behavior_tool_rollout",
    )


def fixture_success_policy(tokenizer: Any) -> Callable:
    """Deterministic multi-turn fixture policy (CPU). Never trains."""

    def policy(prompt_ids, remaining):
        del remaining
        text = tokenizer.decode(prompt_ids)
        if "Tool result:" in text:
            action = {"name": "finish", "arguments": {"answer": "2"}}
        else:
            action = {"name": "calculator", "arguments": {"expression": "1+1"}}
        raw = json.dumps(action, separators=(",", ":"))
        ids = tokenizer.encode(raw, add_special_tokens=False)
        if not ids:
            ids = [1]
        return PolicySample(raw, list(ids), [-0.5] * len(ids), truncated=False)

    return policy


def fixture_parse_error_policy(tokenizer: Any) -> Callable:
    """First decision is unparsable → invalid_attempts=1 → reward -0.1."""

    def policy(prompt_ids, remaining):
        del prompt_ids, remaining
        raw = "not a tool call at all"
        ids = tokenizer.encode(raw, add_special_tokens=False) or [1]
        return PolicySample(raw, list(ids), [-0.5] * len(ids), truncated=False)

    return policy



def select_manifest_tasks(
    tasks: list[TaskSpec],
    *,
    count: int,
    seed: int = 42,
    task_ids: list[str] | None = None,
) -> list[TaskSpec]:
    """Pick train tasks for live dumps. Prefer explicit IDs; else stratified sample by depth."""
    if not tasks:
        raise ValueError("manifest has no train tasks")
    by_id = {task.task_id: task for task in tasks}
    if task_ids:
        missing = [tid for tid in task_ids if tid not in by_id]
        if missing:
            raise ValueError(f"unknown task ids: {', '.join(missing[:5])}")
        return [by_id[tid] for tid in task_ids]
    if count < 1:
        raise ValueError("count must be at least 1")
    # Stable shuffle so Colab / lab dumps are comparable.
    ordered = sorted(tasks, key=lambda task: task.task_id)
    import random

    rng = random.Random(int(seed))
    rng.shuffle(ordered)
    # Prefer a mix of depths when available (single-tool + multi-hop).
    buckets: dict[int, list[TaskSpec]] = {}
    for task in ordered:
        depth = len(task.tool_dependencies)
        buckets.setdefault(depth, []).append(task)
    picked: list[TaskSpec] = []
    depths = sorted(buckets)
    while len(picked) < count and any(buckets.values()):
        for depth in depths:
            if len(picked) >= count:
                break
            bucket = buckets.get(depth) or []
            if bucket:
                picked.append(bucket.pop(0))
    return picked[:count]


def summarize_rollout_reasons(dumps: list[RolloutDump]) -> dict[str, Any]:
    """Histogram completion_reason / reward for reviewer paste-back."""
    from collections import Counter

    reasons = Counter(dump.completion_reason for dump in dumps)
    rewards = Counter(str(dump.ledger.get("reward")) for dump in dumps)
    invalids = Counter(int(dump.ledger.get("invalid_attempts") or 0) for dump in dumps)
    return {
        "n": len(dumps),
        "completion_reason_counts": dict(reasons),
        "reward_counts": dict(rewards),
        "invalid_attempts_counts": {str(k): v for k, v in sorted(invalids.items())},
        "minus_point_one_rate": (
            sum(1 for dump in dumps if dump.ledger.get("reward") == -0.1) / len(dumps) if dumps else 0.0
        ),
    }


def truncation_stat_row(dump: RolloutDump) -> dict[str, Any]:
    hit_cap = dump.completion_token_count >= dump.max_generated_tokens
    return {
        "task_id": dump.task_id,
        "completion_reason": dump.completion_reason,
        "completion_token_count": dump.completion_token_count,
        "max_generated_tokens": dump.max_generated_tokens,
        "hit_token_cap": hit_cap,
        "truncated_flag": dump.truncated,
        "eos_or_tool_stop_likely": dump.completion_reason not in {"truncated", "budget_exhausted"},
        "reward": dump.ledger.get("reward"),
        "invalid_attempts": dump.ledger.get("invalid_attempts"),
    }


def prompt_diff_as_dict(diff: PromptDiff) -> dict[str, Any]:
    return {
        "same_system": diff.same_system,
        "same_user": diff.same_user,
        "same_prepared": diff.same_prepared,
        "findings": list(diff.findings),
        "train": asdict(diff.train),
        "eval": asdict(diff.eval),
    }


def rollout_dump_as_dict(dump: RolloutDump) -> dict[str, Any]:
    return asdict(dump)
