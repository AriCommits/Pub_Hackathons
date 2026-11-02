from pathlib import Path

from scripts.export_hf_artifacts import assert_no_secrets, collect, main


def test_export_dry_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # script paths are relative to lexsi ROOT, not cwd — just exercise main --dry-run
    assert main(["--dry-run"]) == 0


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
