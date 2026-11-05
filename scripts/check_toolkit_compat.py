"""Check the installed recorded-episode AuditKIT bridge without loading a model."""
import argparse
import json
from pathlib import Path

from actproto.agent.harness import DictToolHarness
from actproto.evaluation.trace_adapter import build_trace, execute_event
from actproto.experiments.contracts import ExecutionTrace, TaskSpec
from actproto.evaluation.experiment_runner import load_data
from actproto.integrations.auditkit_v2 import evaluate_episode, write_evaluation_manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("local", "auditkit"), default="auditkit")
    parser.add_argument("--submission", action="store_true")
    parser.add_argument("--run-manifest", required=True)
    parser.add_argument("--dataset-manifest", help="v2 dataset manifest for recorded run rescore")
    parser.add_argument("--traces", help="harness-produced S4 traces.jsonl")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    args = parser.parse_args(argv)
    if args.submission and args.backend != "auditkit":
        raise ValueError("submission requires AuditKIT backend")
    if bool(args.dataset_manifest) != bool(args.traces):
        raise ValueError("recorded rescore requires both --dataset-manifest and --traces")
    if args.dataset_manifest:
        _, tasks = load_data(args.dataset_manifest, args.split)
        task_map = {task.task_id: task for task in tasks}
        if len(task_map) != len(tasks):
            raise ValueError("duplicate task IDs in dataset")
        records = [json.loads(line) for line in Path(args.traces).read_text(encoding="utf-8").splitlines() if line.strip()]
        ids = [record.get("task_id") for record in records]
        if len(ids) != len(set(ids)) or set(ids) != set(task_map):
            raise ValueError("trace task IDs do not exactly match selected split")
        results = [evaluate_episode(task_map[record["task_id"]], ExecutionTrace.from_dict(record["trace"]),
                                    backend=args.backend) for record in records]
        write_evaluation_manifest(Path(args.run_manifest), results)
        print(f"{args.backend} rescored {len(results)} recorded episodes into {args.run_manifest}")
        return
    deps = [{"id": "calculator", "tool": "calculator", "arguments": {"expression": "2+2"}}]
    task = TaskSpec("compat", "compat", "train", "Calculate 2+2", "required_tools",
                    {"answer": "4"}, deps, {"type": "exact"}, 4, "compat", "compat")
    harness = DictToolHarness()
    events = [execute_event(harness, "calculator", {"expression": "2+2"}),
              execute_event(harness, "finish", {"answer": "4"})]
    result = evaluate_episode(task, build_trace("compat", events, "finish"), backend=args.backend)
    if not result["verified_success_v2"]:
        raise RuntimeError("toolkit compatibility smoke failed")
    write_evaluation_manifest(Path(args.run_manifest), [result])
    print(f"{args.backend} compatibility result recorded in {args.run_manifest}")


if __name__ == "__main__":
    main()
