from scripts.query_arg_report import classify_query_call, decision_sentence, render_report


def test_missing_year_that_the_regex_recovers_is_missing_field():
    label = classify_query_call(
        "Query the sales database for Widget in 2023 and finish with the sales number.",
        {"product": "Widget"},
        {"product": "Widget", "year": 2023},
    )
    assert label == "missing_field"


def test_wrong_year_is_wrong_value():
    label = classify_query_call(
        "Query the sales database for Widget in 2023 and finish with the sales number.",
        {"product": "Widget", "year": 1999},
        {"product": "Widget", "year": 2023},
    )
    assert label == "wrong_value"


def test_blank_extra_region_is_a_missing_field_and_a_real_region_is_wrong():
    prompt = "Query the sales database for Widget in 2023 and finish with the sales number."
    gold = {"product": "Widget", "year": 2023}
    assert classify_query_call(prompt, {"product": "Widget", "region": "", "year": 2023}, gold) == "missing_field"
    assert classify_query_call(prompt, {"product": "Widget", "region": "europe", "year": 2023}, gold) == "wrong_value"


def test_decision_prefers_training_only_when_wrong_values_are_the_majority():
    assert decision_sentence(3, 1).startswith("The next experiment is an assistant-only adapter")
    assert decision_sentence(1, 1).startswith("The next experiment fills blank query fields")
    text = render_report(
        {
            "rows": [
                {
                    "case_id": "easy_query-1",
                    "prompt": "Query the sales database for Widget in 2023 and finish with the sales number.",
                    "first_argument_correctness": 0,
                    "gold_calls": [{"name": "query_database", "arguments": {"product": "Widget", "year": 2023}}],
                    "predicted_calls": [{"name": "query_database", "arguments": {"product": "Widget", "year": 1999}}],
                    "tool_results": [{"ok": True, "data": {"rows": [], "count": 0}}],
                }
            ]
        }
    )
    assert "wrong_value: 1" in text
    assert "easy_query-1" in text
