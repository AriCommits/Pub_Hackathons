#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUN_IDS = ("B0", "B1", "B2")


def _rates(rows: list[dict[str, Any]]) -> dict[str, int | float | None]:
    tool_correct = 0
    argument_correct = 0
    for row in rows:
        predicted = row["predicted_tools"]
        gold = row["gold_tools"]
        name_matches = bool(predicted and gold and predicted[0] == gold[0])
        tool_correct += int(name_matches)
        argument_correct += int(name_matches and round(float(row["argument_correctness"]) * len(gold)) >= 1)

    n = len(rows)
    return {
        "n": n,
        "first_tool_accuracy": tool_correct / n if n else None,
        "first_argument_accuracy": argument_correct / n if n else None,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, int | float | None]]:
    summary = {"overall": _rates(rows)}
    for gold_length in (1, 2, 3):
        summary[str(gold_length)] = _rates([row for row in rows if len(row["gold_tools"]) == gold_length])
    return summary


def render_markdown(reports: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Historical depth-1 accuracy",
        "",
        "| Run | Gold length | n | First-tool accuracy | First-argument accuracy |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for run_id in RUN_IDS:
        summary = summarize_rows(reports[run_id]["rows"])
        for bucket in ("overall", "1", "2", "3"):
            metrics = summary[bucket]
            tool_rate = metrics["first_tool_accuracy"]
            argument_rate = metrics["first_argument_accuracy"]
            lines.append(
                f"| {run_id} | {bucket} | {metrics['n']} | {_format_rate(tool_rate)} | {_format_rate(argument_rate)} |"
            )
    return "\n".join(lines) + "\n"


def _format_rate(rate: int | float | None) -> str:
    return "—" if rate is None else f"{rate:.3f}"


def main() -> int:
    results_dir = ROOT / "results" / "b0_b2"
    reports = {
        run_id: json.loads((results_dir / run_id / "eval.json").read_text(encoding="utf-8")) for run_id in RUN_IDS
    }
    markdown = render_markdown(reports)
    output = results_dir / "depth1.md"
    output.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
