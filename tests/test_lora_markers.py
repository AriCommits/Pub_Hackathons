from actproto.training.lora_run import _template_text, _with_generation_markers

NEEDLE = "<|START_RESPONSE|>{{ message['content'] }}<|END_RESPONSE|>"


def test_named_list_template_keeps_the_generation_needle():
    listed = [{"name": "default", "template": f"before {NEEDLE} after"}]
    text = _template_text(listed)
    marked = _with_generation_markers(text)

    assert marked is not None
    assert "{% generation %}{{ message['content'] }}{% endgeneration %}" in marked


def test_missing_needle_does_not_invent_markers():
    assert _with_generation_markers("no tool call span here") is None
