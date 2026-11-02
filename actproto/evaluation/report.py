"""JSON + markdown eval reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from actproto.evaluation.runner import EvalResult


def write_report(result: EvalResult, out_dir: str | Path) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = result.as_dict()
    (out / "eval.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    avg = payload["averages"]
    lines = ["# Eval report", "", result.summary(), "", "| metric | value |", "| --- | --- |"]
    for key, value in avg.items():
        lines.append(f"| {key} | {value} |")
    (out / "eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload
