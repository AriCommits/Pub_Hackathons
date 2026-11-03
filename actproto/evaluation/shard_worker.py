"""Eval shard. Set CUDA_VISIBLE_DEVICES before any torch import."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


def eval_shard(job: dict[str, Any]) -> list[dict[str, Any]]:
    import os

    visible = str(job["cuda_visible"])
    if visible != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
    from actproto.agent.model import load_model
    from actproto.configs.schema import DEFAULT_CONFIG_PATH, MODEL_CONFIG_PATH, load_config
    from actproto.configs.trial_schema import TrialConfig, merge_run_paths
    from actproto.data.types import Task
    from actproto.evaluation.episodes import eventlog_to_episode
    from actproto.evaluation.live_runner import build_live_project

    trial = TrialConfig.from_dict(job["trial"], root=Path(job["root"]))
    run_id = job["run_id"]
    spec = trial.run_spec(run_id)
    adapter_rel, _, _ = merge_run_paths(trial, spec)
    config = load_config(DEFAULT_CONFIG_PATH, MODEL_CONFIG_PATH)
    if adapter_rel:
        config.adapter_path = adapter_rel
    device = "cpu" if visible == "cpu" else "cuda:0"
    loaded = load_model(config, device=device)
    project = build_live_project(run_id, trial, loaded)
    rows = []
    items = job["items"]
    for n, item in enumerate(items, start=1):
        task = Task.from_dict(item["task"])
        t0 = time.perf_counter()
        log = project.infer(task.prompt)
        elapsed = time.perf_counter() - t0
        ep = eventlog_to_episode(log, task.id)
        print(f"{run_id} gpu={visible} {n}/{len(items)} {task.id} {elapsed:.1f}s", flush=True)
        rows.append(
            {
                "index": item["index"],
                "case_id": ep.case_id,
                "messages": ep.messages,
                "final_answer": ep.final_answer,
                "final_state": ep.final_state,
                "latency_s": elapsed,
            }
        )
    return rows
