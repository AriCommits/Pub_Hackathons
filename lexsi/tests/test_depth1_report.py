from lexsi.scripts.depth1_report import render_markdown, summarize_rows


ROWS = [
    {
        "predicted_tools": ["finish"],
        "gold_tools": ["finish"],
        "argument_correctness": 1.0,
    },
    {
        "predicted_tools": ["calculator", "finish"],
        "gold_tools": ["calculator", "finish"],
        "argument_correctness": 0.5,
    },
    {
        "predicted_tools": ["search_docs"],
        "gold_tools": ["lookup_record", "calculator", "finish"],
        "argument_correctness": 1.0,
    },
]


def test_summarize_rows_reports_overall_and_gold_lengths():
    summary = summarize_rows(ROWS)

    assert summary["overall"] == {
        "n": 3,
        "first_tool_accuracy": 2 / 3,
        "first_argument_accuracy": 2 / 3,
    }
    assert summary["1"] == {"n": 1, "first_tool_accuracy": 1.0, "first_argument_accuracy": 1.0}
    assert summary["2"] == {"n": 1, "first_tool_accuracy": 1.0, "first_argument_accuracy": 1.0}
    assert summary["3"] == {"n": 1, "first_tool_accuracy": 0.0, "first_argument_accuracy": 0.0}


def test_render_markdown_includes_each_run_and_bucket():
    markdown = render_markdown({run_id: {"rows": ROWS} for run_id in ("B0", "B1", "B2")})

    assert "| B2 | overall | 3 | 0.667 | 0.667 |" in markdown
    assert "| B2 | 1 | 1 | 1.000 | 1.000 |" in markdown
    assert "| B2 | 2 | 1 | 1.000 | 1.000 |" in markdown
    assert "| B2 | 3 | 1 | 0.000 | 0.000 |" in markdown
