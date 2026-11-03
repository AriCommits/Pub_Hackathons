#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from actproto.activations.extract import extract_and_write  # noqa: E402
from actproto.agent.model import load_model  # noqa: E402
from actproto.configs.schema import DEFAULT_CONFIG_PATH, MODEL_CONFIG_PATH, load_config  # noqa: E402
from actproto.data.types import Task  # noqa: E402
from actproto.evaluation.live_runner import _encode_prompt  # noqa: E402
from actproto.evaluation.task_loader import load_tasks  # noqa: E402
from scripts._cli import add_trial_argument, trial_from_args  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract activation vectors into trial.paths.activations_dump.")
    add_trial_argument(parser)
    parser.add_argument("--split", default=None, help="train, val, test, or all (default: train)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--plan-only", action="store_true", help="Print planned paths and task count without loading weights.")
    args = parser.parse_args(argv)
    trial = trial_from_args(args)
    split = args.split or "train"
    if split == "all":
        from actproto.data.generator import generate_tasks
        tasks = generate_tasks()
        if args.limit is not None:
            tasks = tasks[: args.limit]
    else:
        tasks = load_tasks(split, args.limit)
    dump = trial.resolve(trial.paths.activations_dump)
    if args.plan_only:
        print(
            json.dumps(
                {
                    "split": split,
                    "n_tasks": len(tasks),
                    "out": str(dump),
                    "adapter": trial.paths.adapter_b1,
                },
                indent=2,
            )
        )
        return 0

    config = load_config(DEFAULT_CONFIG_PATH, MODEL_CONFIG_PATH)
    config.adapter_path = trial.paths.adapter_b1
    loaded = load_model(config)

    def encode_fn(task: Task):
        return _encode_prompt(loaded, task.prompt)

    extract_and_write(
        loaded.model,
        tasks,
        str(dump),
        encode_fn=encode_fn,
        layer=config.activation.layer,
        detach=config.activation.detach,
        token=config.activation.token,
        model_id=config.model_id,
        revision=config.revision,
        seed=config.seed,
    )
    print(json.dumps({"split": split, "n_tasks": len(tasks), "out": str(dump)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
