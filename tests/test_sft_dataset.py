from actproto.data.generator import generate_tasks
from actproto.training.lora_config import LoraSftConfig
from actproto.training.sft import build_train_payload
from actproto.training.sft_dataset import gold_to_sft_row


def test_sft_row_shape_and_loss_mask():
    task = next(t for t in generate_tasks() if t.template_family == "easy_calc")
    row = gold_to_sft_row(task)
    assert set(row) == {"messages", "segment_weights", "loss_mask"}
    assert len(row["messages"]) == len(row["segment_weights"]) == len(row["loss_mask"])
    assert all((w == 0.0) == m for w, m in zip(row["segment_weights"], row["loss_mask"]))
    assert row["messages"][0]["role"] == "user"
    assert any(m["role"] == "assistant" for m in row["messages"])


def test_dry_run_payload_no_model():
    tasks = [t for t in generate_tasks() if t.split == "train"][:5]
    payload = build_train_payload(tasks, LoraSftConfig.load())
    assert payload["n_rows"] == 5
    assert payload["output_dir"] == "results/adapters/b1"
    assert payload["peft"]["r"] == 16
