"""Recorded-mode eval runner (AuditKIT AgentEvalRunner stand-in + optional import)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from actproto.evaluation.cases import AgentCase
from actproto.evaluation.episodes import Episode
from actproto.evaluation.metrics import score_episode


@dataclass
class AgentEvalSpec:
    cases: list[AgentCase]
    mode: str = "recorded"
    episodes: list[Episode] | None = None
    scorers: list[str] = field(default_factory=lambda: ["tool_call_f1", "tool_call_validity"])
    trials: int = 1


@dataclass
class EvalRow:
    case_id: str
    status: str
    outcome: dict[str, str]
    coverage_label: str
    ineligible: Any
    metrics: dict[str, Any]


@dataclass
class EvalResult:
    rows: list[EvalRow]
    spec: AgentEvalSpec

    def summary(self) -> str:
        n = len(self.rows) or 1
        successes = sum(1 for r in self.rows if r.outcome.get("verdict") == "success")
        mean_f1 = sum(r.metrics.get("tool_call_f1", 0.0) for r in self.rows) / n
        return f"n={len(self.rows)} success={successes / n:.3f} tool_call_f1={mean_f1:.3f}"

    def as_dict(self) -> dict[str, Any]:
        n = len(self.rows) or 1
        numeric_keys = [
            "task_success",
            "tool_call_f1",
            "tool_call_validity",
            "tool_choice_accuracy",
            "first_tool_accuracy",
            "argument_validity",
            "argument_correctness",
            "first_argument_correctness",
            "step_count",
            "repeated_calls",
            "unnecessary_calls",
        ]
        averages = {}
        for key in numeric_keys:
            values = [r.metrics.get(key) for r in self.rows if r.metrics.get(key) is not None]
            if key == "task_success":
                averages[key] = sum(1.0 if v else 0.0 for v in values) / n
            elif values:
                averages[key] = sum(float(v) for v in values) / n
            else:
                averages[key] = None
        latencies = [r.metrics.get("latency_s") for r in self.rows if r.metrics.get("latency_s") is not None]
        if latencies:
            sorted_lat = sorted(float(x) for x in latencies)
            mid = len(sorted_lat) // 2
            if len(sorted_lat) % 2 == 1:
                averages["latency_p50"] = sorted_lat[mid]
            else:
                averages["latency_p50"] = (sorted_lat[mid - 1] + sorted_lat[mid]) / 2
        return {"summary": self.summary(), "averages": averages, "rows": [r.metrics for r in self.rows]}


class AgentEvalRunner:
    def run(self, spec: AgentEvalSpec) -> EvalResult:
        if spec.mode != "recorded":
            raise ValueError("only recorded mode is implemented in the local runner")
        episodes = {ep.case_id: ep for ep in (spec.episodes or [])}
        rows: list[EvalRow] = []
        for case in spec.cases:
            ep = episodes.get(case.id)
            if ep is None:
                rows.append(
                    EvalRow(
                        case_id=case.id,
                        status="missing",
                        outcome={"verdict": "failure", "reason": "no episode"},
                        coverage_label="none",
                        ineligible="missing_episode",
                        metrics={"case_id": case.id, "task_success": False},
                    )
                )
                continue
            scored = score_episode(case, ep)
            rows.append(
                EvalRow(
                    case_id=case.id,
                    status=scored["status"],
                    outcome=scored["outcome"],
                    coverage_label=scored["coverage_label"],
                    ineligible=scored["ineligible"],
                    metrics=scored,
                )
            )
        return EvalResult(rows=rows, spec=spec)
