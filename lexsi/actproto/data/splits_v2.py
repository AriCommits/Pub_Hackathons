"""Split guards and stable, pre-split fingerprints for v2 examples."""
import hashlib
import json
import re


def normalize_prompt(prompt):
    return re.sub(r"\s+", " ", prompt.casefold()).strip()


def fingerprint(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def validate_splits(examples):
    seen = {"group": {}, "prompt": {}, "normalized_prompt": {}, "calls": {}, "entity": {}}
    regular_families, stress_families = set(), set()
    for item in examples:
        split = item["task"]["split"]
        for field in seen:
            value = item["fingerprints"][field]
            old = seen[field].setdefault(value, split)
            if old != split:
                raise ValueError(f"{field} crosses splits")
        families = stress_families if item["axis"] == "stress" else regular_families
        families.add(item["family"])
    if stress_families & regular_families:
        raise ValueError("stress family overlaps regular families")
