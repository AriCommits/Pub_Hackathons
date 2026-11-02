#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from actproto.evaluation.comparison import comparison_table, render_markdown  # noqa: E402
from scripts._cli import add_trial_argument, trial_from_args  # noqa: E402


def _load_report(path: Path) -> dict:
    eval_json = path / "eval.json"
    if eval_json.exists():
        return json.loads(eval_json.read_text(encoding="utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_json(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Join B0/B1/B2 eval reports into a comparison table (trial.yaml paths).")
    add_trial_argument(parser)
    parser.add_argument("--fixtures", default=None, help="JSON object {B0: report, B1: report, B2: report} for tests")
    args = parser.parse_args(argv)
    trial = trial_from_args(args)

    if args.fixtures:
        reports = json.loads(Path(args.fixtures).read_text(encoding="utf-8"))
    else:
        reports = {}
        for run_id in ("B0", "B1", "B2"):
            if run_id not in trial.runs:
                continue
            report_path = trial.report_dir(run_id)
            if not report_path.exists():
                raise SystemExit(f"missing report directory for {run_id}: {report_path} (run scripts/run_trial.py first)")
            reports[run_id] = _load_report(report_path)

    if len(reports) < 2:
        raise SystemExit("need at least two run reports (B0/B1/B2) under trial.paths.reports_dir")

    probe_path = trial.resolve(trial.paths.probe_metrics())
    proto_path = trial.resolve(trial.paths.prototype_metrics())
    table = comparison_table(
        reports,
        probe_metrics=_load_optional_json(probe_path),
        prototype_metrics=_load_optional_json(proto_path),
        correctness_drop_pp=trial.hypothesis.correctness_drop_pp,
        latency_ratio=trial.hypothesis.latency_ratio,
    )
    out = trial.resolve(trial.paths.reports_dir)
    assert out is not None
    out.mkdir(parents=True, exist_ok=True)
    md = render_markdown(table)
    (out / "comparison.md").write_text(md, encoding="utf-8")
    (out / "comparison.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
