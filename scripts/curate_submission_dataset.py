#!/usr/bin/env python3
"""Write the public submission slice of tooluse_v2.

Train messages and validation prompts are kept. Test tasks, stress tasks,
oracles, fixtures, and traces are not.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from actproto.experiments.manifest import hash_file

FORBIDDEN_KEYS = frozenset({"oracle", "gold_calls", "fingerprints", "dependencies"})
WITHHELD = ["test", "stress", "oracles", "fixtures", "traces"]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _keys(child)


def _forbidden(value) -> bool:
    return any(key in FORBIDDEN_KEYS for key in _keys(value))


def public_train_row(row: dict) -> dict | None:
    """Keep a public training message. Drop anything that still carries an oracle."""
    if row.get("split") not in (None, "train"):
        return None
    if row.get("axis") in {"iid", "stress", "test"}:
        return None
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    public = {"split": "train", "messages": messages}
    if _forbidden(public):
        return None
    return public


def public_val_prompt(row: dict) -> dict | None:
    """Keep the validation prompt and task type. Drop gold, oracles, and test axes."""
    if row.get("axis") != "val":
        return None
    task = row.get("task")
    if not isinstance(task, dict):
        return None
    if task.get("split") == "test":
        return None
    prompt = task.get("prompt")
    task_type = task.get("stratum")
    if not isinstance(prompt, str) or not isinstance(task_type, str):
        return None
    public = {"task_type": task_type, "prompt": prompt}
    if _forbidden(public):
        return None
    return public


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def curate(build_dir: Path, output_dir: Path) -> dict:
    build_dir = Path(build_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = {}
    manifest_path = build_dir / "manifest.json"
    if manifest_path.is_file():
        source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    train_rows = [public for row in _read_jsonl(build_dir / "train_messages.jsonl") if (public := public_train_row(row))]
    if not train_rows:
        from scripts.build_benchmark_v2 import training_row

        train_rows = []
        for row in _read_jsonl(build_dir / "train.jsonl"):
            if row.get("axis") != "train":
                continue
            public = public_train_row(training_row(row))
            if public:
                train_rows.append(public)
    val_rows = [public for row in _read_jsonl(build_dir / "val.jsonl") if (public := public_val_prompt(row))]

    train_path = output_dir / "train_messages.jsonl"
    val_path = output_dir / "val_prompts.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, val_rows)
    manifest = {
        "seed": source_manifest.get("seed"),
        "fixture_environment_id": source_manifest.get("environment_id", "tooluse_v2.fixtures.1"),
        "split_sizes": source_manifest.get("target_counts") or source_manifest.get("counts") or {},
        "train_messages": len(train_rows),
        "val_prompts": len(val_rows),
        "withheld": list(WITHHELD),
        "files": {
            train_path.name: hash_file(train_path),
            val_path.name: hash_file(val_path),
        },
    }
    manifest_out = output_dir / "manifest.json"
    manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["files"][manifest_out.name] = hash_file(manifest_out)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True, help="tooluse_v2 build directory")
    parser.add_argument("--output", type=Path, required=True, help="public slice directory")
    args = parser.parse_args(argv)
    manifest = curate(args.build, args.output)
    print(json.dumps({"train_messages": manifest["train_messages"], "val_prompts": manifest["val_prompts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
