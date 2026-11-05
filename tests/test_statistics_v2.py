import json

import pytest

from actproto.evaluation.statistics_v2 import macro_verified_success, paired_group_bootstrap
from scripts.compare_experiments_v2 import compare_candidates


def _row(task, group, stratum, success):
    return {"task_id": task, "group_id": group, "stratum": stratum, "verified_success_v2": success}


def test_macro_success_weights_strata_equally():
    rows = [_row("a", "g", "single_tool", 1), _row("b", "g", "dependent", 0), _row("c", "h", "dependent", 0)]
    assert macro_verified_success(rows) == 0.5


def test_bootstrap_is_zero_for_identical_groups_and_rejects_nonfinite_values():
    rows = [_row("a", "g1", "single_tool", 1), _row("b", "g1", "single_tool", 0), _row("c", "g2", "dependent", 1)]
    interval = paired_group_bootstrap(rows, rows, draws=100)
    assert interval["difference"] == 0
    assert interval["low"] == interval["high"] == 0
    broken = [dict(rows[0], verified_success_v2=float("nan")), *rows[1:]]
    with pytest.raises(ValueError, match="finite"):
        paired_group_bootstrap(broken, rows, draws=100)


def test_comparison_uses_validation_and_keeps_failed_runs_visible(tmp_path):
    def write(name, success, status="completed"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({
            "id": name, "split": "val", "status": status,
            "metrics": {"macro_verified_success_v2": success, "argument_correctness": 1,
                        "invalid_calls": 0, "generated_tokens": 10},
            "rows": [_row(f"{name}-1", "g", "single_tool", success)],
        }), encoding="utf-8")
        return path
    chosen = compare_candidates([
        json.loads(write("ebase", 0.2).read_text(encoding="utf-8")),
        json.loads(write("e0", 0.4).read_text(encoding="utf-8")),
        json.loads(write("e3", 0.9, status="failed").read_text(encoding="utf-8")),
    ], "e0")
    assert chosen["selected_id"] == "e0"
    assert chosen["target_met"] is False
    assert chosen["failed"] == ["e3"]
    with pytest.raises(ValueError, match="validation"):
        compare_candidates([{"id": "secret", "split": "test", "status": "completed", "metrics": {}}], "secret")
