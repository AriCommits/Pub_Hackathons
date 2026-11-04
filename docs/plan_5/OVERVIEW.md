# Plan 5: Assistant-only adapter for query arguments

## Summary

`B2d1log` showed the router already picks `query_database`, and 21 of 39 bad calls put a wrong value in the arguments (`washing machine` instead of `washer`, or a region the gold call does not use). The current adapter was trained with loss on the whole sequence, so the prompt tokens dominated. This plan trains a second adapter, `results/adapters/b1_assistant_only`, with loss only on the assistant tool-call text, then reruns the depth-1 episode as `B2d1a`. The original `results/adapters/b1` stays in place.

## Goals

- The chat-template marker is read from the default template string, including when the tokenizer stores that template as a named list. `{% generation %}` wraps the assistant content.
- Training aborts if that marker is missing. It does not fall back to full-sequence loss.
- The new adapter is written to `results/adapters/b1_assistant_only`. Rank, alpha, dropout, learning rate, and epoch count stay at the `b1` values (16, 32, 0.05, 2e-4, 2 epochs).
- `B2d1a` uses that adapter with the `B2d1r` episode shape (route hop 1, model on hop 2, two steps, one replica on the 5060 Ti) and writes `results/b0_b2/B2d1a/`.
- `docs/plan_5/query_arg_report.md` and one pointer in `docs/EXPERIMENT_STATUS.md` record verified success and the new `wrong_value` / `missing_field` counts.

## Non-Goals

- Overwriting `results/adapters/b1` or any of `results/b0_b2/{B0,B1,B2,B2d1,B2d1r,B2d1log,B2chain}`.
- Focal loss, class-balanced loss, a third epoch, a higher rank, or a learning-rate change.
- Changing `query_database` so a blank `region` is ignored. That is a separate tool change. This experiment asks whether the new adapter stops emitting the bad arguments.
- Hop-2 prototypes or another chained-router run.

## Background / Context

Project root is `C:\Users\arian\Github\Hackathons\lexsi`. Interpreter is `.venv\Scripts\python`. Training pins the largest GPU before torch is imported. The eval uses `ACTPROTO_CUDA_DEVICES=1`.

`scripts/train_lora.py` currently replaces `sft.yaml`'s `output_dir` with `trial.paths.adapter_b1`, so a new yaml path alone would still overwrite `b1`. `run_lora_sft` stringifies a non-string `chat_template`. Tiny Aya stores the template as a named list, `str(...)` escapes the quotes, the needle `<|START_RESPONSE|>{{ message['content'] }}<|END_RESPONSE|>` misses, and the `except` path trains the full sequence. That is the adapter in `results/adapters/b1` (token accuracy 0.9945, 72 steps).

`docs/plan_4/query_arg_report.md` is the reason to retrain. 15 of the mismatches are a blank `region: ""` on an otherwise right product and year. 21 are a real wrong value. The decision rule there said the next experiment is assistant-only loss.

## Features / Tasks

### T1: Train only the assistant tool-call span

**Files:** `actproto/training/lora_run.py` (modify), `scripts/train_lora.py` (modify), `tests/test_lora_markers.py` (new), `tests/test_train_lora.py` (new)
**Complexity:** M
**Depends on:** none

Add `_template_text` so a string is used as itself and a list or dict of named templates yields the string that contains `<|START_RESPONSE|>`. `_with_generation_markers` then inserts `{% generation %}` and `{% endgeneration %}` around `{{ message['content'] }}` inside `<|START_RESPONSE|>` and `<|END_RESPONSE|>`. If the needle is absent, `run_lora_sft` raises before `trainer.train()`. Delete the handler that prints "retrying with loss on the full sequence" and calls `_fit(False)`.

`train_lora.py` gains `--output-dir`. When it is set, that path wins over `trial.paths.adapter_b1`. The default with no flag remains `results/adapters/b1`. The unit test calls `main(["--plan-only", "--output-dir", "results/adapters/b1_assistant_only"])` and checks the printed `output_dir`. The marker test uses a one-element named list and does not load Tiny Aya.

The real train, after the tests pass, is:

```text
python scripts/train_lora.py --output-dir results/adapters/b1_assistant_only
```

Do not pass `--plan-only` for that run. Do not set `CUDA_VISIBLE_DEVICES` in the parent shell. The script pins the largest GPU itself.

### E1: Depth-1 eval of the new adapter

**Files:** `actproto/configs/trial.yaml` (modify), `scripts/run_trial.py` (modify), `scripts/query_arg_report.py` (modify), `tests/test_run_trial.py` (modify), `tests/test_trial_config.py` (modify), `docs/EXPERIMENT_STATUS.md` (append one pointer), `docs/plan_5/query_arg_report.md` (new, written by the script)
**Complexity:** M
**Depends on:** T1

Add `B2d1a`: adapter `results/adapters/b1_assistant_only`, same prototypes and threshold as `B2d1r`, `route_first_hop_only: true`, `max_agent_steps: 2`, `replicas: 1`. `--run all` stays B0, B1, B2. `query_arg_report.py` accepts `--eval` and `--out` so the B2d1log default is unchanged.

After the adapter file exists:

```text
$env:ACTPROTO_CUDA_DEVICES = "1"
python scripts/run_trial.py --run B2d1a
python scripts/query_arg_report.py --eval results/b0_b2/B2d1a/eval.json --out docs/plan_5/query_arg_report.md
```

Unset `ACTPROTO_CUDA_DEVICES` afterward. Append one pointer with verified task success, first-tool accuracy, first-argument correctness, and the `wrong_value` / `missing_field` counts. Compare first-tool accuracy to 1.000. A drop means the new adapter changed the first hop and is a failed run, not a reason to overwrite `b1`.

## New Dependencies

| Package | Feature | Optional? |
| --- | --- | --- |
| none | TRL assistant-only loss is already installed. | |

## File Change Summary

| File | Action (new/modify/delete) |
| --- | --- |
| `actproto/training/lora_run.py` | modify |
| `scripts/train_lora.py` | modify |
| `tests/test_lora_markers.py` | new |
| `tests/test_train_lora.py` | new |
| `actproto/configs/trial.yaml` | modify |
| `scripts/run_trial.py` | modify |
| `scripts/query_arg_report.py` | modify |
| `tests/test_run_trial.py` | modify |
| `tests/test_trial_config.py` | modify |
| `docs/EXPERIMENT_STATUS.md` | modify |
| `docs/plan_5/query_arg_report.md` | new |

## Open Questions

- If `B2d1a` still emits `region: ""` on the easy query tasks, the blank filter is a tool behavior, not something this adapter run is allowed to patch. Record the count and leave `query_database` unchanged.
- If assistant-only loss raises because the mask is empty, stop. Do not retry on the full sequence.
