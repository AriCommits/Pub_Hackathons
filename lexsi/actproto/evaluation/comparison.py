"""Join B0/B1/B2 reports; score the routing hypothesis without failing the process."""

from __future__ import annotations

from typing import Any


def _avg(report: dict[str, Any]) -> dict[str, Any]:
    if "averages" in report:
        return report["averages"]
    return report


def deltas(baseline: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    out = {}
    keys = set(_avg(baseline)) | set(_avg(other))
    for key in keys:
        b = _avg(baseline).get(key)
        o = _avg(other).get(key)
        if isinstance(b, (int, float)) and isinstance(o, (int, float)):
            out[key] = o - b
        else:
            out[key] = None
    return out


def evaluate_hypothesis(
    b1: dict[str, Any],
    b2: dict[str, Any],
    *,
    correctness_drop_pp: float = 3.0,
    latency_ratio: float = 1.5,
) -> dict[str, Any]:
    a1, a2 = _avg(b1), _avg(b2)
    success_improved = float(a2.get("task_success") or 0) > float(a1.get("task_success") or 0)
    waste_reduced = float(a2.get("unnecessary_calls") or 0) < float(a1.get("unnecessary_calls") or 0)
    drop = float(a1.get("argument_correctness") or 0) - float(a2.get("argument_correctness") or 0)
    correctness_ok = drop <= (correctness_drop_pp / 100.0)
    p50_1 = a1.get("latency_p50")
    p50_2 = a2.get("latency_p50")
    if p50_1 and p50_2:
        latency_ok = float(p50_2) <= float(p50_1) * latency_ratio
    else:
        latency_ok = True
    passed = (success_improved or waste_reduced) and correctness_ok and latency_ok
    return {
        "passed": passed,
        "success_improved": success_improved,
        "waste_reduced": waste_reduced,
        "correctness_ok": correctness_ok,
        "latency_ok": latency_ok,
        "correctness_drop": drop,
        "note": "A false passed flag is a valid negative result, not a process failure.",
    }


def comparison_table(
    reports: dict[str, dict[str, Any]],
    *,
    probe_metrics: dict[str, Any] | None = None,
    prototype_metrics: dict[str, Any] | None = None,
    correctness_drop_pp: float = 3.0,
    latency_ratio: float = 1.5,
) -> dict[str, Any]:
    metrics = [
        "task_success",
        "tool_choice_accuracy",
        "first_tool_accuracy",
        "argument_validity",
        "argument_correctness",
        "first_argument_correctness",
        "step_count",
        "repeated_calls",
        "unnecessary_calls",
        "abstention_coverage",
        "latency_p50",
    ]
    rows = []
    for run_id in ("B0", "B1", "B2"):
        avg = _avg(reports.get(run_id, {}))
        rows.append({"run": run_id, **{m: avg.get(m) for m in metrics}})
    hypothesis = None
    if "B1" in reports and "B2" in reports:
        hypothesis = evaluate_hypothesis(
            reports["B1"],
            reports["B2"],
            correctness_drop_pp=correctness_drop_pp,
            latency_ratio=latency_ratio,
        )
    return {
        "rows": rows,
        "delta_vs_b1": {k: deltas(reports.get("B1", {}), v) for k, v in reports.items() if k != "B1"},
        "delta_vs_b0": {k: deltas(reports.get("B0", {}), v) for k, v in reports.items() if k != "B0"},
        "hypothesis": hypothesis,
        "probe": probe_metrics,
        "prototypes": prototype_metrics,
    }


def render_markdown(table: dict[str, Any]) -> str:
    rows = table["rows"]
    if not rows:
        return "# B0–B2 comparison\n\n(no rows)\n"
    keys = [k for k in rows[0] if k != "run"]
    lines = ["# B0–B2 comparison", "", "| run | " + " | ".join(keys) + " |", "| --- | " + " | ".join("---" for _ in keys) + " |"]
    for row in rows:
        cells = [str(row.get("run"))] + ["" if row.get(k) is None else f"{row.get(k):.4f}" if isinstance(row.get(k), float) else str(row.get(k)) for k in keys]
        lines.append("| " + " | ".join(cells) + " |")
    hyp = table.get("hypothesis") or {}
    lines += ["", f"Hypothesis passed: {hyp.get('passed')}", f"Details: {hyp}"]
    if table.get("probe"):
        lines += ["", f"Linear probe: {table['probe']}"]
    if table.get("prototypes"):
        lines += ["", f"Prototypes: {table['prototypes']}"]
    return "\n".join(lines) + "\n"
