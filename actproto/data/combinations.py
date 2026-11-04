"""Held-out tool-order tasks. These are not part of generate_tasks()."""

from __future__ import annotations

import json
from collections import Counter
from itertools import permutations

from actproto.data.types import Oracle, Task
from actproto.tools.registry import call_tool

WORK_TOOLS = ("calculator", "search_docs", "query_database", "lookup_record")

_ARGS = {
    "calculator": (
        {"expression": "7 * 11"},
        {"expression": "3 + 4"},
        {"expression": "10 - 1"},
    ),
    "search_docs": (
        {"query": "widget"},
        {"query": "returns"},
        {"query": "growth"},
    ),
    "query_database": (
        {"product": "washer", "year": 2024},
        {"product": "widget", "year": 2023},
        {"product": "bolt", "year": 2022},
    ),
    "lookup_record": (
        {"record_id": "REC-001"},
        {"record_id": "REC-008"},
        {"record_id": "REC-003"},
    ),
}

# 11 work calls plus finish. Three tools occur three times; five names cannot
# fill 12 calls with each name used at most twice.
TWELVE_CALLS = (
    "calculator",
    "search_docs",
    "query_database",
    "lookup_record",
    "calculator",
    "search_docs",
    "query_database",
    "lookup_record",
    "calculator",
    "search_docs",
    "query_database",
    "finish",
)

TWICE_CALLS = (
    "calculator",
    "search_docs",
    "query_database",
    "lookup_record",
    "calculator",
    "search_docs",
    "query_database",
    "lookup_record",
    "finish",
)


def _args_for(name: str, seen: Counter[str]) -> dict:
    bank = _ARGS[name]
    args = dict(bank[seen[name] % len(bank)])
    seen[name] += 1
    return args


def _task(task_id: str, names: tuple[str, ...] | list[str]) -> Task:
    seen: Counter[str] = Counter()
    gold_tools: list[str] = []
    gold_args: list[dict] = []
    for name in names:
        if name == "finish":
            continue
        args = _args_for(name, seen)
        result = call_tool(name, args)
        if not result.get("ok"):
            raise RuntimeError(f"{task_id} {name} {args} -> {result}")
        gold_tools.append(name)
        gold_args.append(args)
    answer = task_id
    gold_tools.append("finish")
    gold_args.append({"answer": answer})
    lines = ["Follow this tool sequence exactly, one call per turn."]
    for index, (name, args) in enumerate(zip(gold_tools, gold_args, strict=True), start=1):
        lines.append(f"{index}. {name} {json.dumps(args, sort_keys=True)}")
    lines.append("Do not add, skip, or reorder calls.")
    return Task(
        id=task_id,
        bucket="multi_step",
        template_family="tool_combination",
        entity_ids=list(names),
        prompt="\n".join(lines),
        gold_tools=gold_tools,
        gold_args=gold_args,
        oracle=Oracle(final_state={"done": True, "answer": answer}, answer=answer),
        split="test",
        metadata={"n_calls": len(gold_tools)},
    )


def combination_tasks() -> list[Task]:
    """Every 2-tool and 3-tool order, the each-tool-twice chain, and one 12-call chain."""
    tasks: list[Task] = []
    for length in (2, 3):
        for order in permutations(WORK_TOOLS, length):
            tasks.append(_task("combo-" + "-".join(order), order))
    tasks.append(_task("combo-twice", TWICE_CALLS))
    tasks.append(_task("combo-12", TWELVE_CALLS))
    return tasks
