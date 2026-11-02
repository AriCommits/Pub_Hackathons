from actproto.evaluation.comparison import comparison_table, evaluate_hypothesis, render_markdown


def test_hypothesis_flag_from_fixture_reports():
    b1 = {"averages": {"task_success": 0.5, "unnecessary_calls": 0.8, "argument_correctness": 0.7, "latency_p50": 1.0}}
    b2 = {"averages": {"task_success": 0.6, "unnecessary_calls": 0.4, "argument_correctness": 0.69, "latency_p50": 1.1}}
    hyp = evaluate_hypothesis(b1, b2)
    assert hyp["passed"] is True
    table = comparison_table({"B0": b1, "B1": b1, "B2": b2})
    md = render_markdown(table)
    assert "B2" in md
    assert table["hypothesis"]["passed"] is True


def test_negative_result_does_not_raise():
    b1 = {"averages": {"task_success": 0.8, "unnecessary_calls": 0.1, "argument_correctness": 0.9, "latency_p50": 1.0}}
    b2 = {"averages": {"task_success": 0.5, "unnecessary_calls": 0.4, "argument_correctness": 0.5, "latency_p50": 3.0}}
    hyp = evaluate_hypothesis(b1, b2)
    assert hyp["passed"] is False
