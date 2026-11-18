"""CPU tests for Colab / laptop GRPO rollout diagnostics."""
from __future__ import annotations

import json
import os

import pytest

from actproto.training.rl_device import (
    ALLOW_ANY_GPU_ENV,
    DeviceRecord,
    DeviceSelectionError,
    allow_any_gpu_enabled,
    select_5060_ti,
    select_cuda_device,
)
from actproto.training.rollout_diagnostics import (
    demo_task,
    diff_train_vs_eval_prompts,
    dump_behavior_rollout,
    explain_minus_point_one,
    fixture_parse_error_policy,
    fixture_success_policy,
    truncation_stat_row,
)


TI = "134c34ef-c155-b001-4e3a-b9134b072c21"
TI_NAME = "NVIDIA GeForce RTX 5060 Ti"
OTHER = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
OTHER_NAME = "Tesla T4"


class _Tok:
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


def _scope(task):
    from actproto.training.rl_dataset import VerifiedTrainFixtures

    class Env:
        identifier = task.environment_id
        digest = task.fixture_hash

        def call(self, environment_id, name, arguments):
            del environment_id
            if name == "calculator":
                return {"ok": True, "data": {"value": 2}}
            if name in {"finish", "no_tool"}:
                return {"ok": True, "data": {"answer": arguments.get("answer", ""), "done": True}}
            return {"ok": False, "error": {"type": "unknown", "message": name}}

    scope = object.__new__(VerifiedTrainFixtures)
    scope._tasks = {task.task_id: task}
    scope.for_task = lambda bound: Env()
    return scope


def test_prompt_diff_train_matches_eval_on_cpu():
    diff = diff_train_vs_eval_prompts(demo_task())
    assert diff.same_system and diff.same_user and diff.same_prepared
    assert any("multi-turn" in finding.lower() for finding in diff.findings)
    assert "system_prompt()" in diff.train.system or "tool-using agent" in diff.train.system


def test_explain_minus_point_one_matches_reviewer_ledger():
    payload = explain_minus_point_one()
    assert payload["example_reward"] == pytest.approx(-0.1)
    assert payload["episode_architecture"]["not_single_turn_trl_completion"] is True
    assert payload["episode_architecture"]["training"].startswith("multi-turn")


def test_fixture_parse_error_dump_is_minus_point_one():
    task = demo_task()
    tok = _Tok()
    dump = dump_behavior_rollout(
        task,
        tokenizer=tok,
        policy=fixture_parse_error_policy(tok),
        environment_factory=lambda: _scope(task),
        max_generated_tokens=256,
        deadline_seconds=15.0,
    )
    assert dump.completion_reason == "parse_error"
    assert dump.ledger["invalid_attempts"] == 1
    assert dump.ledger["reward"] == pytest.approx(-0.1)
    assert dump.episode_kind.startswith("multi_turn")


def test_fixture_success_is_multi_turn_and_positive():
    task = demo_task()
    tok = _Tok()
    dump = dump_behavior_rollout(
        task,
        tokenizer=tok,
        policy=fixture_success_policy(tok),
        environment_factory=lambda: _scope(task),
        max_generated_tokens=256,
        deadline_seconds=15.0,
    )
    assert dump.completion_reason == "finish"
    assert dump.ledger["reward"] == pytest.approx(1.0)
    stats = truncation_stat_row(dump)
    assert stats["truncated_flag"] is False
    assert stats["hit_token_cap"] is False


def test_select_cuda_device_fails_closed_without_override():
    devices = [DeviceRecord(0, OTHER_NAME, OTHER, 16_000_000_000)]
    with pytest.raises(DeviceSelectionError, match="allow-any-gpu"):
        select_cuda_device(devices, allow_any_gpu=False)


def test_select_cuda_device_allow_any_gpu_uses_first_card(monkeypatch):
    monkeypatch.delenv(ALLOW_ANY_GPU_ENV, raising=False)
    devices = [DeviceRecord(0, OTHER_NAME, OTHER, 16_000_000_000)]
    assert select_cuda_device(devices, allow_any_gpu=True) == 0
    pinned = [
        DeviceRecord(0, OTHER_NAME, OTHER, 8_000_000_000),
        DeviceRecord(1, TI_NAME, TI, 16_000_000_000),
    ]
    assert select_5060_ti(pinned) == 1
    assert select_cuda_device(pinned, allow_any_gpu=True) == 1


def test_allow_any_gpu_env_flag(monkeypatch):
    monkeypatch.setenv(ALLOW_ANY_GPU_ENV, "1")
    assert allow_any_gpu_enabled() is True
    monkeypatch.setenv(ALLOW_ANY_GPU_ENV, "0")
    assert allow_any_gpu_enabled() is False


def test_diagnose_cli_prompt_diff_and_fixtures(tmp_path):
    from lexsi.scripts.diagnose_grpo_rollouts import main

    out = tmp_path / "prompt.json"
    assert main(["--prompt-diff", "--output", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["same_system"] is True

    out2 = tmp_path / "fixtures.json"
    assert main(["--fixture-rollouts", "--count", "3", "--output", str(out2)]) == 0
    fixtures = json.loads(out2.read_text(encoding="utf-8"))
    assert fixtures["gpu_required"] is False
    assert len(fixtures["rollouts"]) == 3
    assert any(row["ledger"]["reward"] == pytest.approx(-0.1) for row in fixtures["rollouts"])


def test_select_manifest_tasks_stratifies_and_honors_ids():
    from actproto.training.rollout_diagnostics import select_manifest_tasks

    def make(task_id: str, n_deps: int):
        deps = [
            {"id": f"d{i}", "tool": "calculator", "arguments": {"expression": "1+1"}, "after": []}
            for i in range(n_deps)
        ]
        return demo_task().__class__(
            task_id,
            f"g-{task_id}",
            "train",
            f"prompt {task_id}",
            "required_tools",
            {"answer": "2"},
            deps or [{"id": "d0", "tool": "finish", "arguments": {"answer": "2"}, "after": []}],
            {"type": "exact"},
            4,
            "env",
            "hash",
        )

    tasks = [make("a", 1), make("b", 2), make("c", 1), make("d", 3), make("e", 2)]
    picked = select_manifest_tasks(tasks, count=3, seed=0)
    assert len(picked) == 3
    assert len({t.task_id for t in picked}) == 3
    # One from each depth bucket when possible.
    depths = sorted(len(t.tool_dependencies) for t in picked)
    assert depths == [1, 2, 3]
    subset = select_manifest_tasks(tasks, count=2, task_ids=["d", "a"])
    assert [t.task_id for t in subset] == ["d", "a"]
