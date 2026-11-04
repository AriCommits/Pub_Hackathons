#!/usr/bin/env python3
"""Classify mismatched query_database arguments. Does not train or edit the router."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from actproto.agent.schema_regen import heuristic_args

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL = ROOT / "results" / "b0_b2" / "B2d1log" / "eval.json"
DEFAULT_OUT = ROOT / "docs" / "plan_4" / "query_arg_report.md"


def classify_query_call(prompt: str, predicted: dict[str, Any] | None, gold: dict[str, Any] | None) -> str:
    """Label one query call as wrong_value, missing_field, unfilled, or match."""
    heuristic = heuristic_args("query_database", prompt)
    pred = predicted or {}
    wrong: list[str] = []
    missing: list[str] = []
    unfilled: list[str] = []
    for key, gold_value in (gold or {}).items():
        if key not in pred or pred[key] in ("", None):
            if heuristic.get(key) == gold_value:
                missing.append(key)
            else:
                unfilled.append(key)
        elif pred[key] != gold_value:
            wrong.append(key)
    for key, value in pred.items():
        if key in (gold or {}):
            continue
        if value in ("", None):
            missing.append(key)
        else:
            wrong.append(key)
    if wrong:
        return "wrong_value"
    if missing:
        return "missing_field"
    if unfilled:
        return "unfilled"
    return "match"


def decision_sentence(n_wrong: int, n_missing: int) -> str:
    if n_wrong > n_missing:
        return "The next experiment is an assistant-only adapter. This plan does not train it."
    return "The next experiment fills blank query fields from the regex. This plan does not fill them."


def _tool_brief(tool: Any) -> str:
    if not isinstance(tool, dict):
        return json.dumps(tool, sort_keys=True)
    data = tool.get("data") if isinstance(tool.get("data"), dict) else {}
    brief = {"ok": tool.get("ok"), "count": data.get("count")}
    if "error" in tool:
        brief["error"] = tool["error"]
    return json.dumps(brief, sort_keys=True)


def mismatched_query_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in report["rows"]:
        gold_calls = row.get("gold_calls") or []
        if not gold_calls or gold_calls[0].get("name") != "query_database":
            continue
        if float(row.get("first_argument_correctness") or 0) != 0:
            continue
        rows.append(row)
    return rows


def render_report(report: dict[str, Any]) -> str:
    scoped = mismatched_query_rows(report)
    counts = {"wrong_value": 0, "missing_field": 0, "unfilled": 0, "match": 0}
    lines = []
    for row in scoped:
        predicted = (row.get("predicted_calls") or [{}])[0].get("arguments") or {}
        gold = (row.get("gold_calls") or [{}])[0].get("arguments") or {}
        label = classify_query_call(str(row.get("prompt") or ""), predicted, gold)
        counts[label] = counts.get(label, 0) + 1
        tool = (row.get("tool_results") or [None])[0]
        lines.append(
            f"- `{row.get('case_id')}` {label} predicted `{json.dumps(predicted, sort_keys=True)}` "
            f"gold `{json.dumps(gold, sort_keys=True)}` tool `{_tool_brief(tool)}`"
        )
    sentence = decision_sentence(counts["wrong_value"], counts["missing_field"])
    header = [
        "# Query argument log",
        "",
        f"Mismatched query_database calls: {len(scoped)}",
        "",
        "A blank extra argument counts as missing_field. A non-blank extra argument counts as wrong_value.",
        "",
        f"- wrong_value: {counts['wrong_value']}",
        f"- missing_field: {counts['missing_field']}",
        f"- unfilled: {counts['unfilled']}",
        "",
        sentence,
        "",
    ]
    return "\n".join(header + lines) + ("\n" if lines else "")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    report = json.loads(args.eval.read_text(encoding="utf-8"))
    text = render_report(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
