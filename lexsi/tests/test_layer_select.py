from actproto.activations.layer_select import candidate_layers, pick_layer, write_layer_config


def test_pick_layer_writes_argmax(tmp_path):
    scores = {2: 0.4, 4: 0.9, 5: 0.1}
    layer, score = pick_layer(scores)
    assert layer == 4
    assert score == 0.9
    dest = tmp_path / "layer.yaml"
    write_layer_config(dest, layer, "sweep", score)
    text = dest.read_text(encoding="utf-8")
    assert "layer: 4" in text
    assert "method: sweep" in text


def test_candidate_layers():
    assert 4 in candidate_layers(6)
    assert candidate_layers(4)
