"""Construct and replay a frozen, fixture-backed supervised benchmark."""
from __future__ import annotations

from collections import Counter, defaultdict
import random

from actproto.data.fixtures_v2 import build_environment
from actproto.data.splits_v2 import fingerprint, normalize_prompt, validate_splits
from actproto.experiments.contracts import ExecutionTrace, TaskSpec
from actproto.experiments.manifest import hash_object

STRATA = ("single_tool", "dependent", "ordered_subset", "recovery_no_tool")
QUOTAS = {"train": (300, 420, 300, 180), "val": (50, 70, 50, 30),
          "iid": (50, 70, 50, 30), "stress": (50, 70, 50, 30)}


def _number(value):
    return format(value, "g")


def _make_example(i, axis, stratum, env, fixture_hash):
    row, sale = env.records[i], env.sales[i]
    product, record, year = row["product"], row["id"], row["year"]
    calls, dependencies = [], []

    def call(tool, args, after=(), expected_error=None):
        output = env.call(env.identifier, tool, args)
        if bool(output.get("ok")) == bool(expected_error):
            raise ValueError("unexpected tool result")
        if expected_error and output["error"]["type"] != expected_error:
            raise ValueError("unexpected error type")
        dep = {"id": f"d{len(dependencies)}", "tool": tool, "arguments": args, "after": list(after), "output": output}
        if expected_error:
            dep["expected_error"] = expected_error
        dependencies.append(dep)
        calls.append({"tool": tool, "arguments": args, "output": output})
        return output

    variant = i % 3
    family = f"{axis == 'stress' and 'stress' or 'regular'}_{stratum}_{variant}"
    convention = "Return only the requested value as the final answer."
    if stratum == "single_tool":
        if variant == 0:
            prompt = f"Which product is listed in record {record}? {convention}"
            answer = call("lookup_record", {"record_id": record})["data"]["product"]
        elif variant == 1:
            prompt = f"How many sales were recorded for {product} in {year}? {convention}"
            answer = str(call("query_database", {"product": product, "year": year})["data"]["rows"][0]["sales"])
        else:
            prompt = f"Find the service guide for {product}; return its document ID. {convention}"
            answer = call("search_docs", {"query": product})["data"]["hits"][0]["id"]
    elif stratum == "dependent":
        if variant == 0:
            prompt = f"Using record {record}, find its product and annual sales, then double that sales value. {convention}"
            rec = call("lookup_record", {"record_id": record})["data"]
            sales = call("query_database", {"product": rec["product"], "year": rec["year"]}, ["d0"])["data"]["rows"][0]["sales"]
            answer = _number(call("calculator", {"expression": f"{sales} * 2"}, ["d1"])["data"]["value"])
        elif variant == 1:
            prompt = f"What is the annual sales figure for {product} in {year} after adding 17? {convention}"
            sales = call("query_database", {"product": product, "year": year})["data"]["rows"][0]["sales"]
            answer = _number(call("calculator", {"expression": f"{sales} + 17"}, ["d0"])["data"]["value"])
        else:
            prompt = f"Find the guide for {product} and triple its service interval in days. {convention}"
            hit = call("search_docs", {"query": product})["data"]["hits"][0]
            days = int(hit["text"].split()[-2])
            answer = _number(call("calculator", {"expression": f"{days} * 3"}, ["d0"])["data"]["value"])
    elif stratum == "ordered_subset":
        if variant == 0:
            prompt = (f"{convention} First look up {record}, then query sales for that record's product and year; "
                      "finally return sales less 5. Do not use the document catalog.")
            rec = call("lookup_record", {"record_id": record})["data"]
            sales = call("query_database", {"product": rec["product"], "year": rec["year"]}, ["d0"])["data"]["rows"][0]["sales"]
            answer = _number(call("calculator", {"expression": f"{sales} - 5"}, ["d1"])["data"]["value"])
        elif variant == 1:
            prompt = (f"{convention} First look up {record}, then locate its product's service guide; "
                      "return the guide ID. Do not query sales or calculate.")
            rec = call("lookup_record", {"record_id": record})["data"]
            answer = call("search_docs", {"query": rec["product"]}, ["d0"])["data"]["hits"][0]["id"]
        else:
            prompt = (f"{convention} First query {product} sales in {year}, then look up record {record}; "
                      "return the product field from the record. Do not search or calculate.")
            call("query_database", {"product": product, "year": year})
            answer = call("lookup_record", {"record_id": record}, ["d0"])["data"]["product"]
    else:
        if variant == 0:
            prompt = f"The reference code is CODE-{i:04d}. Repeat the code exactly without looking anything up. {convention}"
            answer = f"CODE-{i:04d}"
        else:
            missing = f"MISSING-{i:04d}"
            prompt = (f"Look up {missing}; if it is missing, use record {record} instead and return its product. "
                      f"{convention}")
            call("lookup_record", {"record_id": missing}, expected_error="not_found")
            answer = call("lookup_record", {"record_id": record}, ["d0"])["data"]["product"]
    if axis == "stress":
        if stratum == "single_tool":
            prompts = (f"Audit record {record}: provide the product field. {convention}",
                       f"Audit sales for year {year} and product {product}: provide the sales field. {convention}",
                       f"For {product}, report the ID of the matching service guide. {convention}")
            prompt = prompts[variant]
        elif stratum == "dependent":
            prompts = (f"Compute twice the sales associated with record {record}; use the record's product and year. {convention}",
                       f"Calculate 17 more than the {year} sales for {product}. {convention}",
                       f"For {product}, calculate three times the number of days in its service guide. {convention}")
            prompt = prompts[variant]
        elif stratum == "ordered_subset":
            prompts = (f"{convention} Use exactly this sequence: fetch record {record}, query its product/year sales, then subtract five. Omit document search.",
                       f"{convention} In order, fetch record {record} and search its product's guide; return its ID. Omit database and calculator.",
                       f"{convention} In order, query {year} sales for {product}, then fetch {record}; return its product. Omit search and calculator.")
            prompt = prompts[variant]
        elif variant == 0:
            prompt = f"Transcribe CODE-{i:04d} exactly; no tools are needed. {convention}"
        else:
            prompt = (f"Try record MISSING-{i:04d}. On not found, retrieve {record}; provide the product. "
                      f"{convention}")
    if stratum == "ordered_subset" and i % 2:
        # The execution line remains last; put the output convention before it.
        prompt = f"Output convention: {convention}\nExecution: {prompt.replace(convention, '').strip()}"
    call_budget = 8 if len(calls) > 2 else 4
    mode = "required_tools" if dependencies else "outcome"
    task = TaskSpec(f"v2-{i:04d}", f"group-{i:04d}", "train", prompt, mode,
                    {"answer": answer}, dependencies, {"type": "exact"}, call_budget,
                    env.identifier, fixture_hash, stratum)
    calls.append({"tool": "finish", "arguments": {"answer": answer},
                  "output": env.call(env.identifier, "finish", {"answer": answer})})
    if len(calls) > call_budget:
        raise ValueError("gold exceeds call budget")
    event_fingerprint = [{"tool": c["tool"], "arguments": c["arguments"]} for c in calls]
    # Fingerprints are formed before assigning the split, not from split labels.
    fps = {"group": fingerprint(task.group_id), "prompt": fingerprint(prompt),
           "normalized_prompt": fingerprint(normalize_prompt(prompt)),
           "calls": fingerprint(event_fingerprint), "entity": fingerprint(record)}
    return {"task": task.to_dict(), "axis": axis, "family": family,
            "entity": record, "gold_calls": calls, "fingerprints": fps}


def replay_gold(example, env):
    task = TaskSpec.from_dict(example["task"])
    if task.environment_id != env.identifier or task.fixture_hash != env.digest:
        raise ValueError("fixture mismatch")
    state, events = {}, []
    for call in example["gold_calls"]:
        before = dict(state)
        result = env.call(task.environment_id, call["tool"], call["arguments"])
        if result != call["output"]:
            raise ValueError("gold replay mismatch")
        if call["tool"] == "finish" and result.get("ok"):
            state = {"answer": result["data"]["answer"], "done": True}
        events.append({"kind": "tool_call", "tool": call["tool"], "arguments": call["arguments"],
                       "output": result, "source": "harness", "state_before": before, "state_after": dict(state)})
    if not state.get("done") or state["answer"] != task.oracle["answer"]:
        raise ValueError("gold final answer mismatch")
    return ExecutionTrace(task.task_id, events, "finish", "harness", final_answer=state["answer"], final_state=state)


def build_benchmark(seed=42):
    env = build_environment(sum(sum(q) for q in QUOTAS.values()))
    fixture_hash = env.digest
    slots = [(axis, name) for axis, quota in QUOTAS.items()
             for name, n in zip(STRATA, quota) for _ in range(n)]
    random.Random(seed).shuffle(slots)
    examples, rejects = [], []
    for i, (axis, stratum) in enumerate(slots):
        try:
            examples.append(_make_example(i, axis, stratum, env, fixture_hash))
        except (ValueError, IndexError, KeyError) as exc:
            rejects.append({"index": i, "axis": axis, "stratum": stratum, "reason": str(exc)})
    # Assign split labels only after all logical, prompt and call fingerprints exist.
    for example in examples:
        example["task"]["split"] = "test" if example["axis"] in {"iid", "stress"} else example["axis"]
    validate_splits(examples)
    for example in examples:
        replay_gold(example, env)
    counts, depth = Counter(), defaultdict(Counter)
    for example in examples:
        key = example["axis"]
        counts[key] += 1
        depth[f"{key}/{example['task']['stratum']}"][str(len(example["gold_calls"]) - 1)] += 1
    manifest = {"schema_version": "tooluse_v2", "seed": seed, "environment_id": env.identifier,
                "fixture_hash": fixture_hash, "split_strategy": "unique entities and logical groups; held-out family stress",
                "target_counts": {a: sum(q) for a, q in QUOTAS.items()}, "counts": dict(counts),
                "group_counts": dict(counts), "reject_count": len(rejects),
                "depth_by_axis_and_stratum": {k: dict(v) for k, v in depth.items()},
                "examples_hash": hash_object(examples)}
    return env, examples, rejects, manifest
