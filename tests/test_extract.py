import torch

from actproto.activations.extract import extract_tasks
from actproto.data.types import Oracle, Task


def test_extract_records_from_tiny_lm(tiny_lm):
    task = Task(
        id="toy-1",
        bucket="easy",
        template_family="easy_calc",
        entity_ids=["e"],
        prompt="hi",
        gold_tools=["calculator", "finish"],
        gold_args=[{"expression": "1+1"}, {"answer": "2"}],
        oracle=Oracle(final_state={"done": True}, answer="2"),
        split="train",
    )

    def encode(_task):
        return {"input_ids": torch.randint(0, 32, (1, 6))}

    records = extract_tasks(tiny_lm, [task], encode_fn=encode, model_id="toy")
    assert len(records) == 1
    assert records[0]["gold_tool"] == "calculator"
    assert records[0]["vector"].shape[0] == 8
