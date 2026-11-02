"""Tool-call and trajectory metrics. Verified success comes from outcome assertions."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from actproto.evaluation.cases import AgentCase
from actproto.evaluation.episodes import Episode
from actproto.tools.schemas import validate_args


def _predicted_calls(episode: Episode) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for msg in episode.messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append({"name": fn.get("name"), "arguments": args})
    return calls


def _gold_calls(case: AgentCase) -> list[dict[str, Any]]:
    if not case.reference_turns:
        return []
    return list(case.reference_turns[0])


def tool_call_f1(pred: list[str], gold: list[str]) -> float:
    if not pred and not gold:
        return 1.0
    pc, gc = Counter(pred), Counter(gold)
    overlap = sum((pc & gc).values())
    if overlap == 0:
        return 0.0
    precision = overlap / max(len(pred), 1)
    recall = overlap / max(len(gold), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def argument_validity(calls: list[dict[str, Any]]) -> float:
    if not calls:
        return 1.0
    ok = 0
    for call in calls:
        name = call.get("name") or ""
        valid, _ = validate_args(name, call.get("arguments") or {})
        ok += int(valid)
    return ok / len(calls)


def argument_correctness(pred: list[dict[str, Any]], gold: list[dict[str, Any]]) -> float:
    n = max(len(gold), 1)
    matched = 0
    for i, g in enumerate(gold):
        if i >= len(pred):
            break
        if pred[i].get("name") == g.get("name") and pred[i].get("arguments") == g.get("arguments"):
            matched += 1
    return matched / n


def prefix_tool_accuracy(pred: list[str], gold: list[str]) -> float:
    if not gold:
        return 1.0
    same = 0
    for i, name in enumerate(gold):
        if i < len(pred) and pred[i] == name:
            same += 1
        else:
            break
    return same / len(gold)


def repeated_calls(pred: list[str]) -> int:
    return sum(1 for i in range(1, len(pred)) if pred[i] == pred[i - 1])


def unnecessary_calls(pred: list[str], gold: list[str]) -> int:
    gold_set = set(gold)
    return sum(1 for name in pred if name not in gold_set)


def score_episode(case: AgentCase, episode: Episode) -> dict[str, Any]:
    pred_calls = _predicted_calls(episode)
    gold_calls = _gold_calls(case)
    pred_names = [c.get("name") or "" for c in pred_calls]
    gold_names = [c.get("name") or "" for c in gold_calls]
    outcome = case.outcome.check(final_state=episode.final_state, final_answer=episode.final_answer)
    return {
        "case_id": case.id,
        "status": "ok",
        "outcome": outcome,
        "coverage_label": "full",
        "ineligible": None,
        "task_success": outcome["verdict"] == "success",
        "tool_call_f1": tool_call_f1(pred_names, gold_names),
        "tool_call_validity": argument_validity(pred_calls),
        "tool_choice_accuracy": prefix_tool_accuracy(pred_names, gold_names),
        "argument_validity": argument_validity(pred_calls),
        "argument_correctness": argument_correctness(pred_calls, gold_calls),
        "step_count": len(pred_calls),
        "repeated_calls": repeated_calls(pred_names),
        "unnecessary_calls": unnecessary_calls(pred_names, gold_names),
        "latency_s": episode.latency_s,
        "predicted_tools": pred_names,
        "gold_tools": gold_names,
    }
