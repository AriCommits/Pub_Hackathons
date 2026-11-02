from pathlib import Path

from scripts.export_hf_artifacts import assert_no_secrets, collect, main


def test_export_list_only():
    assert main(["--list-only"]) == 0


def test_refuses_secrets(tmp_path):
    secret = tmp_path / ".env"
    secret.write_text("HF_TOKEN=nope\n", encoding="utf-8")
    try:
        assert_no_secrets([secret])
        raised = False
    except SystemExit:
        raised = True
    assert raised
    assert collect([tmp_path / "missing"]) == []
