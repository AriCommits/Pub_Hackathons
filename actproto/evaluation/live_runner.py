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
from actproto.agent.routed_strategy import build_routed_project
from actproto.agent.strategy import model_policy
from actproto.configs.schema import DEFAULT_CONFIG_PATH, MODEL_CONFIG_PATH, ExperimentConfig, load_config, load_default
from actproto.configs.trial_schema import TrialConfig, merge_run_paths
from actproto.evaluation.cases import task_to_case
from actproto.evaluation.episodes import eventlog_to_episode
from actproto.evaluation.report import write_report
from actproto.evaluation.runner import AgentEvalRunner, AgentEvalSpec
from actproto.evaluation.task_loader import load_tasks


def _encode_prompt(loaded: LoadedModel, prompt: str) -> dict[str, Any]:
    from actproto.agent.prompts import system_prompt

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
    model = loaded.model
    return {k: v.to(model.device) for k, v in encoded.items()}


def make_vector_fn(loaded: LoadedModel, config: ExperimentConfig) -> Callable[[Any], np.ndarray]:
    def _vector(state) -> np.ndarray:
        batch = _encode_prompt(loaded, state.task)
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
    )


def run_live_trial(run_id: str, trial: TrialConfig) -> dict[str, Any]:
    tasks = load_tasks(trial.eval.split, trial.eval.limit)
    spec = trial.run_spec(run_id)
    adapter_rel, _, _ = merge_run_paths(trial, spec)
    config = load_config(DEFAULT_CONFIG_PATH, MODEL_CONFIG_PATH)
    if adapter_rel:
        config.adapter_path = adapter_rel

    loaded = load_model(config)
    project = build_live_project(run_id, trial, loaded)

    cases = [task_to_case(t) for t in tasks]
    episodes = []
    for task in tasks:
        t0 = time.perf_counter()
        log = project.infer(task.prompt)
        elapsed = time.perf_counter() - t0
        ep = eventlog_to_episode(log, task.id)
        ep.latency_s = elapsed
        episodes.append(ep)

    eval_spec = AgentEvalSpec(cases=cases, mode="recorded", episodes=episodes)
    result = AgentEvalRunner().run(eval_spec)
    out_dir = trial.report_dir(run_id)
    payload = write_report(result, out_dir)
    payload["run_id"] = run_id
    payload["n_tasks"] = len(tasks)
    payload["description"] = spec.description
    (out_dir / "eval.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
