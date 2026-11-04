"""Extra SFT trajectories for tool orders the original train split never shows.

These tasks are not part of generate_tasks(). The finish answer is the
tool values joined by '|', computed by running the tools. The prompt does
not contain that string.
"""

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


def result_token(name: str, result: dict) -> str:
    data = result.get("data") or {}
    if name == "calculator":
        value = float(data["value"])
        return str(int(value)) if value.is_integer() else str(value)
    if name == "search_docs":
        hits = data.get("hits") or []
        return str(hits[0]["id"]) if hits else "none"
    if name == "query_database":
        rows = data.get("rows") or []
        if len(rows) == 1:
            return str(rows[0]["sales"])
        return str(data.get("count"))
    if name == "lookup_record":
        return str(data.get("price"))
    raise ValueError(name)


def _task(task_id: str, names: tuple[str, ...] | list[str], shift: int) -> Task:
    seen: Counter[str] = Counter()
    gold_tools: list[str] = []
    gold_args: list[dict] = []
    tokens: list[str] = []
    for name in names:
        if name == "finish":
            continue
        bank = _ARGS[name]
        args = dict(bank[(seen[name] + shift) % len(bank)])
        seen[name] += 1
        result = call_tool(name, args)
        if not result.get("ok"):
            raise RuntimeError(f"{task_id} {name} {args} -> {result}")
        gold_tools.append(name)
        gold_args.append(args)
        tokens.append(result_token(name, result))
    answer = "|".join(tokens)
    gold_tools.append("finish")
    gold_args.append({"answer": answer})
    lines = ["Follow this tool sequence exactly, one call per turn."]
    for index, (name, args) in enumerate(zip(gold_tools, gold_args, strict=True), start=1):
        if name == "finish":
            lines.append(f"{index}. finish with the tool values joined by | in call order.")
            continue
        lines.append(f"{index}. {name} {json.dumps(args, sort_keys=True)}")
    return Task(
        id=task_id,
        bucket="multi_step",
        template_family="order_curriculum",
        entity_ids=[str(shift)],
        prompt="\n".join(lines),
        gold_tools=gold_tools,
        gold_args=gold_args,
        oracle=Oracle(final_state={"done": True, "answer": answer}, answer=answer),
        split="train",
        metadata={"n_calls": len(gold_tools), "shift": shift},
    )


def order_tasks(repeats: int = 4) -> list[Task]:
    """Each short order, the twice-each chain, and the 12-call chain, repeated."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    tasks: list[Task] = []
    for length in (2, 3):
        for order in permutations(WORK_TOOLS, length):
            stem = "order-" + "-".join(order)
            for shift in range(repeats):
                tasks.append(_task(f"{stem}-{shift}", order, shift))
    for shift in range(repeats):
        tasks.append(_task(f"order-twice-{shift}", TWICE_CALLS, shift))
        tasks.append(_task(f"order-12-{shift}", TWELVE_CALLS, shift))
    return tasks
