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


SEQUENCE_FORMAT = "sequence"
HEADER_FIRST = "header_first"
HEADER_LAST = "header_last"
ALL_FORMATS = (SEQUENCE_FORMAT, HEADER_FIRST, HEADER_LAST)
_INSTRUCTIONS = ("only_first", "full", "last_then_first")


def _work_names(names: tuple[str, ...] | list[str]) -> list[str]:
    return [name for name in names if name != "finish"]


def _executed_indexes(n_steps: int, instruction: str) -> list[int]:
    if instruction == "only_first":
        return [1]
    if instruction == "full":
        return list(range(1, n_steps + 1))
    if instruction == "last_then_first":
        if n_steps < 2:
            return [1]
        return [n_steps, 1]
    raise ValueError(instruction)


def _execution_line(indexes: list[int], tools: list[str]) -> str:
    labels = [str(index) for index in indexes]
    named = ", then ".join(tools)
    if indexes == [1] and len(indexes) == 1:
        return f"Only execute step 1: {named}."
    if len(labels) == 2:
        return f"Execute steps {labels[0]} and {labels[1]}: {named}."
    body = ", ".join(labels[:-1]) + f", and {labels[-1]}"
    return f"Execute steps {body}: {named}."


def _catalog_lines(steps: list[tuple[str, dict, str]]) -> list[str]:
    lines = []
    for index, (name, args, _token) in enumerate(steps, start=1):
        lines.append(f"Step {index}: {name} {json.dumps(args, sort_keys=True)}")
    return lines


def _sequence_prompt(gold_tools: list[str], gold_args: list[dict]) -> str:
    lines = ["Follow this tool sequence exactly, one call per turn."]
    for index, (name, args) in enumerate(zip(gold_tools, gold_args, strict=True), start=1):
        if name == "finish":
            lines.append(f"{index}. finish with the tool values joined by | in call order.")
            continue
        lines.append(f"{index}. {name} {json.dumps(args, sort_keys=True)}")
    return "\n".join(lines)


def _header_prompt(fmt: str, steps: list[tuple[str, dict, str]], indexes: list[int]) -> str:
    tools = [steps[index - 1][0] for index in indexes]
    header = _execution_line(indexes, tools)
    catalog = _catalog_lines(steps)
    if fmt == HEADER_FIRST:
        return "\n".join([header, *catalog])
    if fmt == HEADER_LAST:
        return "\n".join([*catalog, header])
    raise ValueError(fmt)


def _task(
    task_id: str,
    names: tuple[str, ...] | list[str],
    shift: int,
    fmt: str = SEQUENCE_FORMAT,
    instruction: str = "full",
) -> Task:
    steps: list[tuple[str, dict, str]] = []
    seen: Counter[str] = Counter()
    for name in _work_names(names):
        bank = _ARGS[name]
        args = dict(bank[(seen[name] + shift) % len(bank)])
        seen[name] += 1
        result = call_tool(name, args)
        if not result.get("ok"):
            raise RuntimeError(f"{task_id} {name} {args} -> {result}")
        steps.append((name, args, result_token(name, result)))
    indexes = _executed_indexes(len(steps), "full" if fmt == SEQUENCE_FORMAT else instruction)
    chosen = [steps[index - 1] for index in indexes]
    answer = "|".join(token for _name, _args, token in chosen)
    gold_tools = [name for name, _args, _token in chosen] + ["finish"]
    gold_args = [args for _name, args, _token in chosen] + [{"answer": answer}]
    if fmt == SEQUENCE_FORMAT:
        prompt = _sequence_prompt(gold_tools, gold_args)
    else:
        prompt = _header_prompt(fmt, steps, indexes)
    return Task(
        id=task_id,
        bucket="multi_step",
        template_family="order_curriculum",
        entity_ids=[str(shift)],
        prompt=prompt,
        gold_tools=gold_tools,
        gold_args=gold_args,
        oracle=Oracle(final_state={"done": True, "answer": answer}, answer=answer),
        split="train",
        metadata={"n_calls": len(gold_tools), "shift": shift, "format": fmt, "instruction": instruction},
    )


def _iter_jobs(formats: tuple[str, ...]):
    catalogs: list[tuple[str, tuple[str, ...] | list[str]]] = []
    for length in (2, 3):
        for order in permutations(WORK_TOOLS, length):
            catalogs.append(("order-" + "-".join(order), order))
    catalogs.append(("order-twice", TWICE_CALLS))
    catalogs.append(("order-12", TWELVE_CALLS))
    for stem, names in catalogs:
        if SEQUENCE_FORMAT in formats:
            yield stem, names, SEQUENCE_FORMAT, "full"
        for fmt in formats:
            if fmt == SEQUENCE_FORMAT:
                continue
            for instruction in _INSTRUCTIONS:
                yield stem, names, fmt, instruction


def order_tasks(repeats: int = 4, formats: tuple[str, ...] | None = None) -> list[Task]:
    """Each short order, the twice-each chain, and the 12-call chain, repeated.

    The default format is the original sequence wording. Pass ALL_FORMATS to
    also emit header-first and header-last instructions.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    chosen = formats or (SEQUENCE_FORMAT,)
    tasks: list[Task] = []
    for stem, names, fmt, instruction in _iter_jobs(chosen):
        for shift in range(repeats):
            if fmt == SEQUENCE_FORMAT:
                task_id = f"{stem}-{shift}"
            else:
                task_id = f"{stem}-{fmt}-{instruction}-{shift}"
            tasks.append(_task(task_id, names, shift, fmt, instruction))
    return tasks
