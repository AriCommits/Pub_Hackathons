"""Choose an SFT candidate from validation metrics only."""
from __future__ import annotations

import json
from pathlib import Path

from actproto.evaluation.statistics_v2 import paired_group_bootstrap
from actproto.training.sft_v2 import select_checkpoint

TARGET_GAIN = 0.05


def load_candidates(paths: list[str | Path]) -> list[dict]:
    candidates = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("split") != "val":
            raise ValueError("candidate comparison accepts validation results only")
        if payload.get("status") not in {"completed", "failed"}:
            raise ValueError("candidate status must be completed or failed")
        candidates.append(payload)
    return candidates


def compare_candidates(candidates: list[dict], baseline_id: str) -> dict:
    for item in candidates:
        if item.get("split") != "val":
            raise ValueError("candidate comparison accepts validation results only")
    completed = [item for item in candidates if item.get("status") == "completed"]
    if not completed:
        raise ValueError("no completed validation candidate")
    selected = select_checkpoint(completed)
    baseline = next((item for item in completed if item.get("id") == baseline_id), None)
    if baseline is None:
        raise ValueError("baseline candidate is missing from the completed validation set")
    gain = float(selected["metrics"]["macro_verified_success_v2"]) - float(baseline["metrics"]["macro_verified_success_v2"])
    interval = None
    if selected.get("rows") and baseline.get("rows") and selected["id"] != baseline_id:
        interval = paired_group_bootstrap(baseline["rows"], selected["rows"])
    return {
        "selected_id": selected["id"],
        "baseline_id": baseline_id,
        "improvement": gain,
        "target_met": gain >= TARGET_GAIN,
        "interval": interval,
        "failed": [item["id"] for item in candidates if item.get("status") == "failed"],
        "selected": selected,
    }
