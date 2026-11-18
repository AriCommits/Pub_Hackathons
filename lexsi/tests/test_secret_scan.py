from pathlib import Path

import pytest

from actproto.utils.paths import PathEscapeError, contain_under
from actproto.utils.secret_scan import assert_paths_clean, scan_file_for_secrets, write_secret_file
from lexsi.scripts.export_hf_artifacts import assert_no_secrets


def test_content_scan_catches_api_key_json(tmp_path):
    metrics = tmp_path / "metrics.json"
    metrics.write_text('{"api_key": "secret-value"}', encoding="utf-8")
    assert scan_file_for_secrets(metrics)
    with pytest.raises(SystemExit):
        assert_no_secrets([metrics])


def test_resume_token_name_is_forbidden(tmp_path):
    token = tmp_path / "resume-token.txt"
    write_secret_file(token, "abc")
    assert scan_file_for_secrets(token)
    with pytest.raises(ValueError, match="secret"):
        assert_paths_clean([token])


def test_contain_under_rejects_escape(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    with pytest.raises(PathEscapeError):
        contain_under(root, "../outside")
    inside = contain_under(root, "results/out")
    assert inside.is_relative_to(root.resolve())
