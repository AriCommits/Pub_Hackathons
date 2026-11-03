"""Live Tiny Aya eval: B0 base, B1 LoRA, B2 routed (activation + prototypes)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from actproto.activations.hooks import hooked
from actproto.activations.prototypes import load_prototypes_npz
from actproto.agent.abstention import Threshold
from actproto.agent.model import LoadedModel, generate_text, load_model
from actproto.agent.project import Project, build_project
from actproto.agent.prompts import messages_for_template
from actproto.agent.routed_strategy import build_routed_project
from actproto.agent.strategy import model_policy
from actproto.configs.schema import DEFAULT_CONFIG_PATH, MODEL_CONFIG_PATH, ExperimentConfig, load_config, load_default
from actproto.configs.trial_schema import TrialConfig, merge_run_paths
from actproto.evaluation.cases import task_to_case
from actproto.evaluation.episodes import Episode, eventlog_to_episode
from actproto.evaluation.report import write_report
from actproto.evaluation.runner import AgentEvalRunner, AgentEvalSpec
from actproto.evaluation.task_loader import load_tasks


def _encode_prompt(
    loaded: LoadedModel,
    prompt: str,
    messages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    from actproto.agent.prompts import system_prompt

    if messages is None:
        messages = [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": prompt},
        ]
    tokenizer = loaded.tokenizer
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    from actproto.agent.model import _parameter_device

    model = loaded.model
    device = _parameter_device(model)
    return {k: v.to(device) for k, v in encoded.items() if hasattr(v, "to")}


def make_vector_fn(loaded: LoadedModel, config: ExperimentConfig) -> Callable[[Any], np.ndarray]:
    def _vector(state) -> np.ndarray:
        messages = messages_for_template(state.log.messages) if state.step > 0 else None
        batch = _encode_prompt(loaded, state.task, messages=messages)
        seq_len = int(batch["input_ids"].shape[-1])
        with hooked(
            loaded.model,
            config.activation.layer,
            detach=config.activation.detach,
            token=config.activation.token,
        ) as hook:
            loaded.model(**batch)
            return hook.numpy_vector(seq_len)

    return _vector


def build_live_project(run_id: str, trial: TrialConfig, loaded: LoadedModel) -> Project:
    spec = trial.run_spec(run_id)
    adapter_rel, prototypes_rel, threshold_rel = merge_run_paths(trial, spec)
    config = load_default()
    if adapter_rel:
        config.adapter_path = adapter_rel
    if spec.max_agent_steps is not None:
        config.max_agent_steps = spec.max_agent_steps

    def generate_fn(messages: list[dict[str, Any]]) -> str:
        return generate_text(loaded, messages, max_new_tokens=config.max_new_tokens)

    inner = model_policy(generate_fn)

    if not spec.router:
        return build_project(config, policy=inner)

    if prototypes_rel is None:
        raise ValueError(f"{run_id} has router=true but no prototypes path")
    proto_path = trial.resolve(prototypes_rel)
    if not proto_path or not proto_path.exists():
        raise FileNotFoundError(f"prototypes not found: {proto_path}")
    prototypes = load_prototypes_npz(proto_path)
    threshold = None
    if threshold_rel:
        tpath = trial.resolve(threshold_rel)
        if tpath and tpath.exists():
            threshold = Threshold.load(tpath)

    return build_routed_project(
        config,
        prototypes=prototypes,
        threshold=threshold,
        inner_policy=inner,
        vector_fn=make_vector_fn(loaded, config),
        generate_fn=generate_fn,
    )


def _replica_indices(max_replicas: int | None = None) -> list[str]:
    """How many bf16 copies fit. Subprocess so this process does not init CUDA."""
    import json
    import subprocess
    import sys

    lines = [
        "import json, os, torch",
        "bytes_per = int(6.35 * 1024**3)",
        "reserve = int(0.30 * 1024**3)",
        "cap = 2",
        "if not torch.cuda.is_available():",
        "    print(json.dumps({'plan': ['cpu'], 'gpus': []}))",
        "    raise SystemExit",
        "allowed = os.environ.get('ACTPROTO_CUDA_DEVICES')",
        "indices = list(range(torch.cuda.device_count()))",
        "if allowed:",
        "    indices = [int(x) for x in allowed.split(',') if x.strip()]",
        "    indices = [i for i in indices if 0 <= i < torch.cuda.device_count()]",
        "plan = []",
        "gpus = []",
        "for i in indices:",
        "    free, total = torch.cuda.mem_get_info(i)",
        "    n = max(0, min(cap, int((free - reserve) // bytes_per)))",
        "    plan.extend([str(i)] * n)",
        "    gpus.append({'index': i, 'name': torch.cuda.get_device_name(i), 'free_gb': round(free/1024**3, 2), 'replicas': n})",
        "if not plan and indices:",
        "    best = max(indices, key=lambda i: torch.cuda.get_device_properties(i).total_memory)",
        "    plan = [str(best)]",
        "print(json.dumps({'plan': plan, 'gpus': gpus}))",
    ]
    proc = subprocess.run([sys.executable, "-c", "\n".join(lines)], capture_output=True, text=True, check=True)
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")][-1]
    payload = json.loads(line)
    plan = list(payload["plan"])
    if max_replicas is not None:
        plan = plan[:max_replicas]
        payload["plan"] = plan
    print(json.dumps(payload), flush=True)
    return plan


def run_live_trial(run_id: str, trial: TrialConfig) -> dict[str, Any]:
    import multiprocessing as mp

    from actproto.evaluation.shard_worker import eval_shard

    tasks = load_tasks(trial.eval.split, trial.eval.limit)
    spec = trial.run_spec(run_id)
    plan = _replica_indices(spec.replicas)
    buckets: list[list[dict[str, Any]]] = [[] for _ in plan]
    for index, task in enumerate(tasks):
        buckets[index % len(plan)].append({"index": index, "task": task.to_dict()})
    jobs = []
    for visible, items in zip(plan, buckets):
        if not items:
            continue
        jobs.append(
            {
                "cuda_visible": visible,
                "items": items,
                "run_id": run_id,
                "trial": trial.to_dict(),
                "root": str(trial.root),
            }
        )
    print(f"{run_id}: {len(tasks)} tasks on {len(jobs)} replicas", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(len(jobs)) as pool:
        parts = pool.map(eval_shard, jobs)
    packed = [row for part in parts for row in part]
    packed.sort(key=lambda row: row["index"])
    episodes = []
    for row in packed:
        ep = Episode(
            case_id=row["case_id"],
            messages=row["messages"],
            final_answer=row["final_answer"],
            final_state=row["final_state"] or {},
            latency_s=row["latency_s"],
        )
        episodes.append(ep)

    cases = [task_to_case(t) for t in tasks]
    eval_spec = AgentEvalSpec(cases=cases, mode="recorded", episodes=episodes)
    result = AgentEvalRunner().run(eval_spec)
    out_dir = trial.report_dir(run_id)
    payload = write_report(result, out_dir)
    payload["run_id"] = run_id
    payload["n_tasks"] = len(tasks)
    payload["description"] = spec.description
    payload["replicas"] = plan
    (out_dir / "eval.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
