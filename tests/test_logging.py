from actproto.configs.schema import load_default
from actproto.utils.logging import init_run
from actproto.utils.seed import seed_everything


def test_init_run_writes_snapshot(tmp_results_dir):
    seed_everything(0)
    run_dir = init_run(load_default(), results_root=tmp_results_dir)
    assert (run_dir / "config.yaml").exists()
    text = (run_dir / "config.yaml").read_text(encoding="utf-8")
    assert "CohereLabs/tiny-aya-global" in text
    assert (run_dir / "meta.json").exists()
