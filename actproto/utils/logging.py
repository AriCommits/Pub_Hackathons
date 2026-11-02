"""Run directories and config snapshots."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml


def git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def init_run(config: Any, results_root: str | Path = "results") -> Path:
    """Create results/<run_id>/ with a copied config snapshot. Returns the run dir."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}-{uuid4().hex[:8]}"
    run_dir = Path(results_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any]
    if hasattr(config, "to_dict"):
        payload = config.to_dict()
    elif isinstance(config, dict):
        payload = dict(config)
    else:
        payload = {"repr": repr(config)}
    payload["_meta"] = {
        "run_id": run_id,
        "git_sha": git_sha(),
        "seed": payload.get("seed"),
        "cwd": os.getcwd(),
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    (run_dir / "meta.json").write_text(json.dumps(payload["_meta"], indent=2), encoding="utf-8")
    return run_dir
