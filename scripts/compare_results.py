#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from actproto.evaluation.comparison import comparison_table, render_markdown  # noqa: E402


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Join B0/B1/B2 reports into a comparison table.")
    parser.add_argument("--b0", default=None)
    parser.add_argument("--b1", default=None)
    parser.add_argument("--b2", default=None)
    parser.add_argument("--fixtures", default=None, help="JSON object {B0: report, B1: report, B2: report}")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", default="results/b0_b2")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load((ROOT / "actproto" / "configs" / "experiments.yaml").read_text(encoding="utf-8"))
    reports = {}
    if args.fixtures:
        reports = _load(Path(args.fixtures))
    else:
        for key, flag in (("B0", args.b0), ("B1", args.b1), ("B2", args.b2)):
            if flag:
                reports[key] = _load(Path(flag))
    if args.dry_run and not reports:
        reports = {
            "B0": {"averages": {"task_success": 0.2, "unnecessary_calls": 1.5, "argument_correctness": 0.4, "latency_p50": 1.0}},
            "B1": {"averages": {"task_success": 0.5, "unnecessary_calls": 0.8, "argument_correctness": 0.7, "latency_p50": 1.2}},
            "B2": {"averages": {"task_success": 0.6, "unnecessary_calls": 0.4, "argument_correctness": 0.69, "latency_p50": 1.3}},
        }
    table = comparison_table(
        reports,
        correctness_drop_pp=float(cfg.get("correctness_drop_pp", 3.0)),
        latency_ratio=float(cfg.get("latency_ratio", 1.5)),
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = render_markdown(table)
    (out / "comparison.md").write_text(md, encoding="utf-8")
    (out / "comparison.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
