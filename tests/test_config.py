from actproto.configs.schema import load_default


def test_default_model_is_tiny_aya():
    cfg = load_default()
    assert cfg.model_id == "CohereLabs/tiny-aya-global"
    assert cfg.seed == 42
    assert cfg.activation.token == "prompt_last"
    assert cfg.activation.layer == "auto"
