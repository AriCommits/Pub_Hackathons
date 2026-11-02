from actproto.data.generator import generate_tasks, write_splits
from actproto.data.splits import assert_no_leakage, leakage_keys


def test_no_template_entity_leakage():
    tasks = generate_tasks()
    grouped = {"train": [], "val": [], "test": []}
    for t in tasks:
        grouped[t.split].append(t)
    assert_no_leakage(grouped["train"], grouped["val"], grouped["test"])
    train_keys = set()
    for t in grouped["train"]:
        train_keys |= leakage_keys(t)
    test_keys = set()
    for t in grouped["test"]:
        test_keys |= leakage_keys(t)
    assert train_keys.isdisjoint(test_keys)


def test_leakage_detector_fails_on_overlap():
    tasks = generate_tasks()
    t0 = tasks[0]
    clone = type(t0).from_dict(t0.to_dict())
    clone.split = "test"
    try:
        assert_no_leakage([t0], [], [clone])
        raised = False
    except AssertionError:
        raised = True
    assert raised


def test_write_splits_roundtrip(tmp_path):
    tasks = generate_tasks()
    counts = write_splits(tasks, tmp_path)
    assert counts["train"] > 0 and counts["val"] > 0 and counts["test"] > 0
    assert (tmp_path / "manifest.json").exists()


def test_buckets_present():
    buckets = {t.bucket for t in generate_tasks()}
    assert buckets == {"easy", "confusable", "multi_step", "no_tool", "error_recovery"}
