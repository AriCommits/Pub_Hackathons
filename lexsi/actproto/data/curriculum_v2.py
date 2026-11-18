"""Train-only, bounded failure repair selection."""
from collections import Counter


def select_probe(examples, seed=42):
    import random
    train = [e for e in examples if e["task"]["split"] == "train"]
    if len(train) < 100:
        raise ValueError("at least 100 train examples required")
    rng = random.Random(seed)
    return rng.sample(sorted(train, key=lambda e: e["task"]["task_id"]), 100)


def repair_mix(original, variants, verify):
    if len(variants) > 120 or len(variants) > len(original):
        raise ValueError("repair variants exceed bounded budget or original share")
    groups = {e["task"]["group_id"] for e in original if e["task"]["split"] == "train"}
    if any(e["task"]["split"] != "train" for e in original) or any(
        e["task"]["split"] != "train" or e["task"]["group_id"] not in groups or not verify(e)
        for e in variants):
        raise ValueError("repair examples must be verified and from train groups")
    return list(original) + list(variants)


def classify_failures(failures):
    allowed = {"tool", "argument", "termination"}
    if any(x not in allowed for x in failures):
        raise ValueError("unknown failure category")
    return dict(Counter(failures))
