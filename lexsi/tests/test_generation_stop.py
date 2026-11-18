import torch

from actproto.agent.model import LoadedModel, generate_text
from actproto.agent.stop import completion_should_stop
from actproto.configs.schema import ExperimentConfig


def test_completion_should_stop_on_one_closed_tool_call():
    closed = '<tool_call>{"name":"finish","arguments":{"answer":"429"}}</tool_call>'
    assert completion_should_stop(closed) is True


def test_completion_should_stop_on_end_of_turn_token():
    assert completion_should_stop("partial <|END_OF_TURN_TOKEN|>") is True


def test_completion_should_not_stop_on_empty_or_half_open_tag():
    assert completion_should_stop("") is False
    assert completion_should_stop('<tool_call>{"name":"finish"') is False


def test_generate_text_passes_stopping_criteria_and_keeps_the_token_cap():
    captured = {}

    class Tokenizer:
        pad_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": torch.tensor([[1, 2, 3]])}

        def decode(self, token_ids, skip_special_tokens=True):
            return '<tool_call>{"name":"finish","arguments":{"answer":"1"}}</tool_call>'

    class Model:
        device = "cpu"

        def generate(self, **kwargs):
            captured.update(kwargs)
            return torch.tensor([[1, 2, 3, 4]])

        def parameters(self):
            yield torch.zeros(1)

    loaded = LoadedModel(tokenizer=Tokenizer(), model=Model(), config=ExperimentConfig(max_new_tokens=17))
    text = generate_text(loaded, [{"role": "user", "content": "hi"}], max_new_tokens=17)

    assert text
    assert captured["max_new_tokens"] == 17
    assert captured["do_sample"] is False
    criteria = captured["stopping_criteria"]
    result = criteria(torch.tensor([[1, 2, 3, 9]]), torch.zeros(1, 1))
    stopped = bool(result.item()) if hasattr(result, "item") else bool(result)
    assert stopped is True
