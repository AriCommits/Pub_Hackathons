"""Generate the synthetic benchmark and write JSONL splits."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from actproto.data.splits import assign_split_by_index, assert_no_leakage
from actproto.data.templates import (
    FAMILIES,
    NO_TOOL_FACTS,
    confusable_record_vs_search,
    confusable_sales_number,
    easy_calc,
    easy_lookup,
    easy_query,
    easy_search,
    error_missing_product,
    error_missing_record,
    multi_avg_growth,
    no_tool_capital,
)
from actproto.data.types import Task
from actproto.tools.fixtures import (
    DOCUMENTS,
    KNOWN_MISSING_IDS,
    PRODUCTS,
    RECORDS,
    SALES_ROWS,
)


def _years_for(product: str) -> list[int]:
    return sorted({int(r["year"]) for r in SALES_ROWS if str(r["product"]) == product})


def _build_family_entities() -> dict[str, list[dict[str, Any]]]:
    calc_entities = []
    for a in range(7, 47):
        for b in (11, 13, 17, 19):
            calc_entities.append({"id": f"calc-{a}x{b}", "a": a, "b": b})

    lookup_entities = [{"record": rec} for rec in RECORDS]
    search_entities = [{"doc": doc} for doc in DOCUMENTS]
    query_entities = []
    for product in PRODUCTS:
        for year in _years_for(product):
            query_entities.append({"product": product, "year": year})
    growth_entities = [{"product": product} for product in PRODUCTS]
    capital_entities = [{"country": c, "capital": cap} for c, cap in NO_TOOL_FACTS]
    missing_record_entities = [{"missing_id": mid} for mid in KNOWN_MISSING_IDS + [f"REC-{n}" for n in range(800, 812)]]
    missing_product_entities = [{"product": name} for name in ("phantom", "ghost-sku", "no-such-part", "voidware", "unobtainium")]

    return {
        "easy_calc": calc_entities,
        "easy_lookup": lookup_entities,
        "easy_search": search_entities,
        "easy_query": query_entities,
        "confusable_sales_number": list(query_entities),
        "confusable_record_vs_search": list(lookup_entities),
        "multi_avg_growth": growth_entities,
        "no_tool_capital": capital_entities,
        "error_missing_record": missing_record_entities,
        "error_missing_product": missing_product_entities,
    }


_BUILDERS = {
    "easy_calc": easy_calc,
    "easy_lookup": easy_lookup,
    "easy_search": easy_search,
    "easy_query": easy_query,
    "confusable_sales_number": confusable_sales_number,
    "confusable_record_vs_search": confusable_record_vs_search,
    "multi_avg_growth": multi_avg_growth,
    "no_tool_capital": no_tool_capital,
    "error_missing_record": error_missing_record,
    "error_missing_product": error_missing_product,
}


def generate_tasks() -> list[Task]:
    family_entities = _build_family_entities()
    tasks: list[Task] = []
    for family, entities in family_entities.items():
        builder = _BUILDERS[family]
        n = len(entities)
        for i, entity in enumerate(entities):
            task = builder(entity, i)
            task.split = assign_split_by_index(n, i)
            tasks.append(task)
    # Repeat no-tool and error templates with paraphrases to pad smaller families
    extra = _paraphrase_pad(tasks)
    tasks.extend(extra)
    return tasks


def _paraphrase_pad(existing: list[Task]) -> list[Task]:
    """Duplicate small families with a prompt prefix that does not change entities."""
    by_family: dict[str, list[Task]] = defaultdict(list)
    for t in existing:
        by_family[t.template_family].append(t)
    extras: list[Task] = []
    prefixes = (
        "Please complete this request. ",
        "Follow the tool policy. ",
    )
    for family, members in by_family.items():
        if len(members) >= 40:
            continue
        for p_i, prefix in enumerate(prefixes):
            for t in members:
                clone = Task.from_dict(t.to_dict())
                clone.id = f"{t.id}-p{p_i}"
                clone.prompt = prefix + t.prompt
                extras.append(clone)
    return extras


def bucket_counts(tasks: list[Task]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for t in tasks:
        counts[t.bucket] += 1
    return dict(counts)


def write_splits(tasks: list[Task], out_dir: str | Path) -> dict[str, int]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[Task]] = {"train": [], "val": [], "test": []}
    for t in tasks:
        grouped[t.split or "train"].append(t)
    assert_no_leakage(grouped["train"], grouped["val"], grouped["test"])
    manifest = {
        "n": {k: len(v) for k, v in grouped.items()},
        "buckets": bucket_counts(tasks),
        "families": sorted({t.template_family for t in tasks}),
    }
    for split, rows in grouped.items():
        path = out / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for task in rows:
                fh.write(json.dumps(task.to_dict()) + "\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest["n"]


def load_split(path: str | Path) -> list[Task]:
    tasks = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                tasks.append(Task.from_dict(json.loads(line)))
    return tasks


def load_tasks_by_split(processed_dir: str | Path) -> dict[str, list[Task]]:
    root = Path(processed_dir)
    return {split: load_split(root / f"{split}.jsonl") for split in ("train", "val", "test")}
