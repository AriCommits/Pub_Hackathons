#!/usr/bin/env python3
"""Run the bounded v2 SFT study and freeze the validation-selected baseline.

The checked-in two-epoch configuration stays unchanged. This command writes a
study config only under results/sft_v2 and records every candidate it did not run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from actproto.experiments.manifest import hash_bytes, hash_file, hash_object  # noqa: E402
from scripts.compare_experiments_v2 import compare_candidates  # noqa: E402

DATASET = ROOT / "data" / "benchmarks" / "tooluse_v2"
RESULTS = ROOT / "results" / "sft_v2"
MANIFEST = ROOT / "docs" / "SFT_BASELINE_MANIFEST.json"
PYTHON = Path(sys.executable)


def _run(args, env):
    completed = subprocess.run(args, cwd=ROOT, env=env, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(args[1:3])}")


def build_dataset():
    if (DATASET / "manifest.json").is_file():
        return DATASET / "manifest.json"
    from scripts.build_benchmark_v2 import main
    if main(["--output", str(DATASET)]) != 0:
        raise RuntimeError("benchmark build did not meet its quotas")
    return DATASET / "manifest.json"


def _gpu_env():
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    env["ACTPROTO_V2_MANIFEST"] = str(DATASET / "manifest.json")
    env.pop("ACTPROTO_CUDA_DEVICES", None)
    return env


def _uuid(env):
    code = "import torch; print(str(torch.cuda.get_device_properties(0).uuid).lower())"
    return subprocess.check_output([str(PYTHON), "-c", code], cwd=ROOT, env=env, text=True).strip()


def _candidate(run, candidate_id, groups):
    summary = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    rows = []
    for line in (run / "traces.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        rows.append({"task_id": record["task_id"], "group_id": groups[record["task_id"]],
                     "stratum": record["stratum"], "verified_success_v2": record["metrics"]["verified_success_v2"]})
    tool_trace = any(event.get("kind") == "tool_call" for line in (run / "traces.jsonl").read_text(encoding="utf-8").splitlines()
                     if line.strip() for event in json.loads(line)["trace"]["events"])
    return {"id": candidate_id, "split": "val", "status": "completed", "metrics": summary,
            "rows": rows, "run": str(run), "tool_trace": tool_trace,
            "adapter_sha256": json.loads((run / "inference.json").read_text(encoding="utf-8"))["adapter_sha256"]}


def _groups():
    groups = {}
    for line in (DATASET / "val.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            task = json.loads(line)["task"]
            groups[task["task_id"]] = task["group_id"]
    return groups


def _eval(candidate, env, uuid, adapter=None):
    run = RESULTS / candidate
    if (run / "metrics.json").is_file():
        return run
    args = [str(PYTHON), "scripts/run_experiment_v2.py", "--manifest", str(DATASET / "manifest.json"),
            "--split", "val", "--run-id", candidate, "--device-uuid", uuid, "--output-root", str(RESULTS)]
    if adapter:
        args.extend(["--adapter", adapter])
    _run(args, env)
    return run


def _train(env, uuid, epochs):
    run = RESULTS / "e2"
    if (run / "complete.json").is_file():
        return json.loads((run / "complete.json").read_text(encoding="utf-8"))["selected"]["checkpoint"]
    import yaml
    config = yaml.safe_load((ROOT / "actproto/configs/sft_v2.yaml").read_text(encoding="utf-8"))
    config["epochs"] = epochs
    config_path = RESULTS / "e2-config.yaml"
    RESULTS.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    from actproto.training.sft_masks import selected_template
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["revision"])
    template = selected_template(tokenizer, config["template_name"])
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {
        "run_id": "e2", "schema_version": "tooluse_v2",
        "config_hash": hash_file(config_path), "data_hash": hash_file(DATASET / "train_messages.jsonl"),
        "model_hash": hash_object({"model_id": config["model_id"], "revision": config["revision"]}),
        "fixture_hash": json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))["fixture_hash"],
        "template_hash": hash_bytes(template.encode()),
        "verifier_hash": hash_file(ROOT / "actproto/evaluation/verification_v2.py"),
        "code_revision": revision,
    }
    manifest_path = RESULTS / "e2-run-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _run([str(PYTHON), "scripts/train_sft_v2.py", "--data", str(DATASET / "train_messages.jsonl"),
          "--config", str(config_path), "--manifest", str(manifest_path), "--output-root", str(RESULTS),
          "--device-uuid", uuid, "--evaluator", "actproto.evaluation.experiment_runner:validation_evaluator"], env)
    return json.loads((run / "complete.json").read_text(encoding="utf-8"))["selected"]["checkpoint"]


def execute(epochs):
    if MANIFEST.exists():
        raise FileExistsError("baseline manifest already exists")
    build_dataset()
    env = _gpu_env()
    uuid = _uuid(env)
    groups = _groups()
    candidates = []
    base_run = _eval("ebase", env, uuid)
    candidates.append(_candidate(base_run, "ebase", groups))
    legacy = Path(r"C:\Users\arian\Github\Hackathons\lexsi\results\adapters\b1_assistant_only")
    if (legacy / "adapter_config.json").is_file():
        candidates.append(_candidate(_eval("e0", env, uuid, str(legacy)), "e0", groups))
    else:
        candidates.append({"id": "e0", "split": "val", "status": "failed", "reason": "legacy adapter missing"})
    trained = _train(env, uuid, epochs)
    candidates.append(_candidate(_eval("e2", env, uuid, trained), "e2", groups))
    for name, reason in (("e1", "original-trajectory retrain cut before launch"), ("e3", "optional repair not reached")):
        candidates.append({"id": name, "split": "val", "status": "failed", "reason": reason})
    baseline = "e0" if any(item.get("id") == "e0" and item.get("status") == "completed" for item in candidates) else "ebase"
    decision = compare_candidates(candidates, baseline)
    selected = next(item for item in candidates if item.get("id") == decision["selected_id"])
    if not selected.get("tool_trace"):
        raise RuntimeError("selected candidate did not record a tool call")
    if selected["id"] == "e2":
        adapter_path = trained
    elif selected["id"] == "e0":
        adapter_path = str(legacy)
    else:
        adapter_path = ""
    template = hash_bytes(subprocess.check_output([str(PYTHON), "-c",
        "import yaml; from pathlib import Path; from transformers import AutoTokenizer; from actproto.training.sft_masks import selected_template; "
        "c=yaml.safe_load(Path('actproto/configs/sft_v2.yaml').read_text(encoding='utf-8')); "
        "t=AutoTokenizer.from_pretrained(c['model_id'], revision=c['revision']); print(selected_template(t, c['template_name']), end='')"],
        cwd=ROOT, text=True).encode())
    payload = {
        "status": "frozen", "frozen": True, "schema_version": "tooluse_v2",
        "selected_id": selected["id"], "adapter_path": adapter_path,
        "adapter_sha256": selected["adapter_sha256"], "selected_adapter_sha256": selected["adapter_sha256"],
        "dataset_manifest_sha256": hash_file(DATASET / "manifest.json"),
        "template_sha256": template, "verifier_sha256": hash_file(ROOT / "actproto/evaluation/verification_v2.py"),
        "selection_split": "val", "epochs_executed": epochs, "full_config_epochs": 2,
        "target_met": decision["target_met"], "improvement": decision["improvement"],
        "omitted": decision["failed"], "reference_policy": "selected_sft",
    }
    MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (RESULTS / "decision.json").write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-data", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args(argv)
    if args.build_data:
        print(build_dataset())
    if args.execute:
        if not 1 <= args.epochs <= 2:
            raise ValueError("epochs must be 1 or 2")
        print(json.dumps(execute(args.epochs), indent=2))
    if not args.build_data and not args.execute:
        print(json.dumps({"dataset": str(DATASET), "manifest_present": MANIFEST.exists(), "optimizes": False}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
