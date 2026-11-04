import json

from scripts import train_lora


def test_output_dir_flag_overrides_the_b1_path(capsys):
    assert train_lora.main(["--plan-only", "--output-dir", "results/adapters/b1_assistant_only"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["output_dir"].endswith("b1_assistant_only")


def test_orders_flag_appends_the_curriculum(capsys):
    assert train_lora.main(["--plan-only", "--orders", "--output-dir", "results/adapters/b1_orders"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_rows"] == 440
    assert payload["output_dir"].endswith("b1_orders")


def test_formats_flag_appends_every_prompt_layout(capsys):
    assert train_lora.main(["--plan-only", "--formats", "--output-dir", "results/adapters/b1_formats"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_rows"] == 820
    assert payload["output_dir"].endswith("b1_formats")


def test_plan_only_default_stays_on_b1(capsys):
    assert train_lora.main(["--plan-only"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["output_dir"].endswith("b1")
    assert not payload["output_dir"].endswith("b1_assistant_only")
