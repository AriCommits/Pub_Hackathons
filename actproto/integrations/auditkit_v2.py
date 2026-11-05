"""Recorded AuditKIT evaluation using the shared execution verifier.

Only feed harness-recorded traces; a provenance string cannot authenticate a
model-supplied or externally forged trace. The recorded episode is the
predicate input, rather than a precomputed local verdict.
"""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

from actproto.evaluation.verification_v2 import verify
from actproto.experiments.contracts import ExecutionTrace
from actproto.experiments.manifest import hash_file, hash_object

REVISION = "71316da345784ff220d5ed59337441a59ab31526"
VERSION = "1.1.3"


def toolkit_info():
    try:
        dist = importlib.metadata.distribution("auditkit")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("AuditKIT submission backend is unavailable") from exc
    direct = dist.read_text("direct_url.json")
    try:
        commit = json.loads(direct or "{}").get("vcs_info", {}).get("commit_id")
    except (TypeError, ValueError) as exc:
        raise RuntimeError("AuditKIT install provenance invalid") from exc
    if dist.version != VERSION or commit != REVISION:
        raise RuntimeError("AuditKIT version or revision does not match the tested lock")
    return {"name": "auditkit", "version": dist.version, "revision": commit}


def evaluate_episode(task, trace, *, backend="auditkit"):
    task_hash, trace_hash = hash_object(task.to_dict()), hash_object(trace.to_dict())
    if backend == "local":
        result = verify(task, trace)
        return {"backend": "local", "task_sha256": task_hash, "trace_sha256": trace_hash,
                "verified_success_v2": result.verified_success_v2,
                "verification": result.to_dict(), "invocation": "actproto.evaluation.verification_v2.verify"}
    if backend != "auditkit":
        raise ValueError("unknown evaluation backend")
    info = toolkit_info()
    from auditkit.agent_eval import AgentCase, AgentEpisode, AgentEvent, AgentEvalRunner, AgentEvalSpec, CustomPredicate
    from actproto.evaluation import verification_v2
    verifier_hash = hash_file(Path(verification_v2.__file__))
    events = []
    for index, event in enumerate(trace.events):
        if event.get("kind") == "tool_call":
            events.append(AgentEvent(index=index, role="assistant", type="tool_call",
                payload={"name": event.get("tool"), "arguments": event.get("arguments")},
                turn_id=index, call_id=str(index), source="harness"))
    episode = AgentEpisode(events=events, mode="recorded", source_format="actproto_execution_trace_v2",
                           case_id=task.task_id, final_answer=trace.final_answer, final_state=trace.final_state,
                           stop_reason=trace.completion_reason, metadata={"execution_trace": trace.to_dict()})

    def predicate(recorded, case):
        # Read the stored episode at evaluation time; the verifier does not rely
        # on AuditKIT diagnostic tool-call metrics or a precomputed verdict.
        actual = ExecutionTrace.from_dict(recorded.metadata["execution_trace"])
        return verify(task, actual).verified_success_v2

    oracle = CustomPredicate(predicate, version=f"{verifier_hash}:{task_hash}", name="actproto.verify.tooluse_v2")
    case = AgentCase(id=task.task_id, task=task.prompt, outcome=oracle)
    spec = AgentEvalSpec(cases=[case], mode="recorded", episodes=[episode], scorers=[], trials=1,
                         name="actproto-tooluse-v2")
    result = AgentEvalRunner().run(spec)
    if len(result.rows) != 1:
        raise RuntimeError("AuditKIT did not produce exactly one case result")
    row = result.rows[0]
    verdict = row.outcome.get("verdict")
    if row.status != "completed" or verdict not in {"success", "failure"}:
        raise RuntimeError(f"AuditKIT evaluation failed: {row.status}: {row.outcome}")
    return {"backend": "auditkit", "toolkit": info, "task_sha256": task_hash, "trace_sha256": trace_hash,
            "verified_success_v2": verdict == "success",
            "verdict": verdict, "verifier_sha256": verifier_hash,
            "invocation": "AgentEvalRunner.run(AgentEvalSpec(mode=recorded, CustomPredicate=actproto.verify.tooluse_v2))"}


def write_evaluation_manifest(path, results):
    """Seal backend/version/invocation and verdicts in a new local run manifest."""
    if not results:
        raise ValueError("cannot record an empty evaluation")
    data = {"schema_version": "tooluse_v2", "evaluations": results}
    with Path(path).open("x", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
