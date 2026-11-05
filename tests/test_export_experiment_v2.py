from copy import deepcopy
import json

import pytest

from actproto.experiments.manifest import hash_file
from scripts.export_experiment_v2 import export


def bundle_fixture(tmp_path):
    names = {"adapter": ["adapter/adapter_config.json", "adapter/adapter_model.safetensors"],
             "tokenizer": ["tokenizer/tokenizer_config.json", "tokenizer/tokenizer.json"],
             "template": ["template/default.jinja"], "data_manifest": ["data/manifest.json"],
             "fixture_manifest": ["data/fixtures.json"], "curriculum": ["data/train_messages.jsonl"],
             "traces": ["run/traces.jsonl"], "metrics": ["run/metrics.json"],
             "environment_lock": ["env/requirements.lock"], "model_card": ["model/README.md"]}
    entries = []
    root = tmp_path / "inputs"
    for role, paths in names.items():
        for name in paths:
            file = root / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("{}", encoding="utf-8")
            entries.append({"role": role, "path": name, "sha256": hash_file(file)})
    return root, {"schema_version": "tooluse_v2", "artifacts": entries}


def test_list_copy_verify_and_no_overwrite(tmp_path):
    root, manifest = bundle_fixture(tmp_path)
    listing = export(root, manifest)
    assert len(listing) == 12
    target = tmp_path / "bundle"
    assert export(root, manifest, target) == listing
    assert json.loads((target / "export-manifest.json").read_text()) == manifest
    for item in manifest["artifacts"]:
        assert hash_file(target / item["path"]) == item["sha256"]
    with pytest.raises(FileExistsError):
        export(root, manifest, target)


def test_rejects_missing_modified_private_and_credentials(tmp_path):
    root, manifest = bundle_fixture(tmp_path)
    missing = deepcopy(manifest)
    missing["artifacts"] = [a for a in missing["artifacts"] if a["role"] != "model_card"]
    with pytest.raises(ValueError, match="missing"):
        export(root, missing)
    modified = deepcopy(manifest)
    modified["artifacts"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        export(root, modified)
    private = deepcopy(manifest)
    private["artifacts"][0]["split"] = "test"
    with pytest.raises(ValueError, match="held-out"):
        export(root, private)
    file = root / "run/metrics.json"
    file.write_text('{"api_key": "secret-value"}', encoding="utf-8")
    record = next(a for a in manifest["artifacts"] if a["path"] == "run/metrics.json")
    record["sha256"] = hash_file(file)
    with pytest.raises(ValueError, match="credential"):
        export(root, manifest)


def test_rejects_traversal_and_unpaired_router(tmp_path):
    root, manifest = bundle_fixture(tmp_path)
    malformed = deepcopy(manifest)
    malformed["artifacts"][0]["path"] = "../other"
    with pytest.raises(ValueError, match="unsafe"):
        export(root, malformed)
    manifest["artifacts"].append({"role": "prototype", "path": "x", "sha256": "0" * 64})
    with pytest.raises(ValueError, match="together"):
        export(root, manifest)
