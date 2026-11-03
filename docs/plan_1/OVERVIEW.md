# Plan 1: One-pass accuracy fixes for ActProto-LoRA

## Summary

Verified task success is about zero because the base model copies the wrong answer string and the LoRA emits more than one tool call in a single turn. B0 (Tiny Aya, no adapter) has task success 0.000, tool_call_f1 0.880, and argument correctness 0.269 on the 104-task test. B1 (LoRA, 2 epochs, 288 train rows, train token accuracy 0.9945, train_loss 0.305) has task success 0.010, tool_call_f1 0.696, and argument correctness 0.024. The linear probe and prototypes are perfect on frozen first-tool activations (test accuracy 1.0 for the five tools) while the router threshold is min_score -0.2 and min_margin 0.0, so coverage is 1.0 and it never abstains. This plan is one implementation pass: seven file-disjoint workstreams land together, then a single gated GPU run writes a new adapter and new reports. Epochs, rank, and learning rate stay where they are. The scored benchmark stays the synthetic entity-split test.

## Goals

- Record a bucket and family slice of `results/b0_b2/B0/eval.json` and `results/b0_b2/B1/eval.json`, including parse-error finishes, before any new training.
- Train the next adapter with loss only on assistant tool-call and finish spans, including the end-of-turn token. Cap remains 2 epochs. Keep the epoch with higher validation argument correctness. Write it to `results/adapters/b1_assistant_only/`.
- Stop generation at the first complete `<tool_call>` so one turn cannot dump the rest of the trajectory.
- Repair finish arguments from the last tool result when the call is already schema-valid (integral calculator values, prose wrapped around a field that is already present).
- Encode the post-tool transcript for the second hop, and fit prototypes on each gold step, including `finish`.
- Choose the strictest validation threshold that preserves peak accuracy and still meets `min_coverage`, so a weak second hop can abstain and fall through to the model.
- Add a small number of harness-checked local traces only for families the slice shows are thin. Keep the `(template_family, entity_id)` split.

## Non-Goals

- More than 2 epochs, a higher LoRA rank, a higher alpha, or a higher learning rate.
- Focal loss, class-balanced loss, or `class_weight` on the probe.
- Replacing or mixing the scored test with ToolBench, API-Bank, Gorilla, ToolACE, xLAM, Glaive, Hermes, or BFCL.
- A large hand-written corpus, train-only noise, or a second tool catalog.
- Overwriting `results/adapters/b1` or `results/b0_b2/{B0,B1,B2}`.
- Starting `train_lora.py`, `extract_activations.py`, `build_prototypes.py`, `train_probe.py`, or `run_trial.py` while a B2 trial is still using the GPUs.
- Killing any running eval, or inventing B2 numbers before `results/b0_b2/B2/eval.json` exists.
- Changing `parse_tool_call` so that two tool calls in one string are accepted. `tests/test_agent_parse.py` keeps that refusal.
- Using the test oracle to accept or reject a sample.

## Background / Context

Project root is `C:\Users\arian\Github\Hackathons\lexsi`. Interpreter is `.venv\Scripts\python` (CPython 3.12). Model is `CohereLabs/tiny-aya-global`, revision `af89d219b53ed9b13b8a4645f9c8028973510324`. Trust PyTorch GPU order: `cuda:0` is the RTX 4060 (8 GB) and is too small for a bf16 replica; `cuda:1` is the RTX 5060 Ti. New GPU work uses `ACTPROTO_CUDA_DEVICES=1` and one replica.

`docs/plan_1.md` is an earlier sequential note. This directory is the parallel plan. The slice below is the measurement that note was waiting on. Code for the loss mask, the decode stop, argument repair, hop-2 encoding, the threshold, and the small local traces can land in one pass because they touch different files. The GPU run stays gated.

Tools, and only these: `calculator`, `search_docs`, `query_database`, `lookup_record`, `finish`. `no_tool` aliases `finish`. Schemas live in `actproto/tools/schemas.py`. The chat template is Cohere. `messages_for_template` in `actproto/agent/prompts.py` rewrites tool results as user turns. SFT rows from `actproto/training/sft_dataset.py` stay `{messages, segment_weights, loss_mask}`: user and tool weight 0, assistant weight 2. TRL does not read those fields. They are the mask specification.

Assistant-only loss did not engage. `lora_run.py` stringifies `tokenizer.chat_template`. Tiny Aya stores that template as a named list, so `str(...)` escapes the quotes, the needle `<|START_RESPONSE|>{{ message['content'] }}<|END_RESPONSE|>` misses, `marked` is `None`, and `_fit(False)` trains the full sequence. Token accuracy 0.9945 is mostly user, system, and tool-result text.

`make_vector_fn` in `actproto/evaluation/live_runner.py` encodes only `state.task` via `_encode_prompt`. `extract_tasks` stores one vector per task, labeled with `gold_tools[0]`. The second hop therefore routes as if the tool result did not exist. `choose_threshold` keeps the first grid point when accuracy and coverage tie, which is min_score -0.2 and min_margin 0.0 (`results/prototype_metrics.json`).

`model_policy` turns every `ParseError` into `finish` with answer `PARSE_ERROR: ...`. Those rows count as a predicted `finish`. On `no_tool`, gold is only `finish`, so tool_call_f1 is 1.0 even when the answer is the parse error.

### Slice of the existing evals (n=104 test)

Case ids map to families by stripping a `-pN` paraphrase suffix and the trailing index. Buckets follow the family prefix. This is a CPU read of the two `eval.json` files. B2 was not sliced.

| Run | Bucket | n | task_success | tool_call_f1 | tool_choice | arg_correct | steps | What the rows actually did |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| B0 | easy | 56 | 0.000 | 0.929 | 0.812 | 0.366 | 2.21 | `easy_calc` 32/32 is calculator then finish, answer `429.0` vs gold `429`. Lookup tools are right; the finish string is prose around the price. Query and search often fail to parse or add an extra lookup. |
| B0 | confusable | 21 | 0.000 | 0.775 | 0.333 | 0.143 | 1.48 | Record lookup tools are right; finish is prose or a parse error. Sales often finishes immediately (13/15) instead of `query_database`. |
| B0 | error_recovery | 12 | 0.000 | 0.917 | 0.750 | 0.375 | 1.75 | Missing-record tools are right; finish is not the exact token `NOT_FOUND`. Missing-product is an immediate parse-error finish. |
| B0 | multi_step | 6 | 0.000 | 0.550 | 0.056 | 0.000 | 1.17 | Five finishes immediately; the rest emits two calls in one turn. |
| B0 | no_tool | 9 | 0.000 | 1.000 | 1.000 | 0.000 | 1.00 | Predicted tool is `finish` because the parser failed. The answer is `PARSE_ERROR`, not the capital. |
| B1 | easy | 56 | 0.000 | 0.685 | 0.054 | 0.027 | 1.05 | Almost every family is one `finish` whose answer is `PARSE_ERROR: multiple tool calls in one turn`. |
| B1 | confusable | 21 | 0.000 | 0.667 | 0.000 | 0.000 | 1.00 | Same multi-call parse error, stored as `finish`. |
| B1 | error_recovery | 12 | 0.000 | 0.667 | 0.000 | 0.000 | 1.00 | Same. |
| B1 | multi_step | 6 | 0.000 | 0.500 | 0.000 | 0.000 | 1.00 | Same. |
| B1 | no_tool | 9 | 0.111 | 1.000 | 1.000 | 0.111 | 1.00 | One real success. The other eight are multi-call parse errors scored as `finish`. |

B1 `easy_search` (3 tasks) is the exception inside easy: the first turn is a real `search_docs`, and the second turn is the multi-call parse error. Argument correctness 0.5 is the first call matching and `finish` missing.

`argument_validity` is 1.0 on B1 and about 0.995 on B0. The calls that parse are schema-valid. The failures are the answer string, a second call in the same generation, or a skipped tool. Another epoch on the full sequence would memorize those patterns harder.

### Data recommendation

Keep the synthetic generator as the scored benchmark. `generate_tasks()` plus `assign_split_by_index` is the only test that measures these five tools on these fixtures. Do not replace it, and do not mix in an external corpus.

Hugging Face and open tool-use sets are the wrong training source for this score. Live dataset-card fetches were blocked in this session; re-read the card before any download. The sets that people mean by “open tool-calling data” are:

| Dataset | License as published on the card / paper | Why it does not fit |
| --- | --- | --- |
| `Salesforce/xlam-function-calling-60k` (APIGen) | CC-BY-4.0 | Thousands of public APIs. Not these five schemas or fixtures. |
| `NousResearch/hermes-function-calling-v1` | Apache-2.0 | Generic function-call JSON. Different tools. |
| `glaiveai/glaive-function-calling-v2` | Apache-2.0 | Generic functions in ShareGPT shape. |
| `Team-ACE/ToolACE` | Apache-2.0 | Large synthetic multi-API set. |
| Berkeley Function-Calling Leaderboard (Gorilla) | Apache-2.0 repo; it is an eval suite | Different functions. Must not become the scored test. |
| ToolBench (RapidAPI) | Research corpus of thousands of APIs | No `calculator` / `search_docs` / `query_database` / `lookup_record` / `finish` fixture world. |

APIGen’s useful idea is the filter, not the files: a new trace has to parse, execute in this harness, and pass the task oracle. That filter already exists as `call_tool` plus `replay_gold`.

Local generation is appropriate only for the thin failing families, and only as a small add-on. `multi_step` has 6 test rows, `error_missing_product` has 3, and `easy_search` has 3. `easy_calc` is 32 of 104 test rows and already calls both tools; more calculator rows would thicken the majority family. Build the new rows by replaying this harness on new entity ids inside the existing families (`multi_avg_growth`, `error_missing_product`, `confusable_sales_number`). Each new entity goes through `assign_split_by_index` once. Paraphrase clones stay on the source split. Do not hand-write a second corpus, and do not point eval at anything other than `generate_tasks()` filtered to `split == "test"`.

## Features / Tasks

### S1: Slice eval.json by bucket, family, and parse-error

**Files:** `scripts/slice_eval_buckets.py` (new), `tests/test_slice_eval_buckets.py` (new), `results/b0_b2/bucket_slice.md` (new)
**Complexity:** S
**Depends on:** none

CPU only. Join each row of `results/b0_b2/{B0,B1}/eval.json` to `generate_tasks()` on `case_id`. If a B2 file appears, include it; if it does not, skip it. For each run, bucket, and `template_family`, write n, task_success, tool_call_f1, tool_choice_accuracy, argument_validity, argument_correctness, and step_count. Split predicted `finish` into a real finish versus a `PARSE_ERROR` answer. Also count first predicted tool against first gold tool. The unit test uses a tiny inline JSON fixture, not the 104-row files. Do not call the model.

### L1: Assistant-only loss and early stop, same two epochs

**Files:** `actproto/training/lora_run.py` (modify), `actproto/configs/sft.yaml` (modify `output_dir` only), `tests/test_lora_assistant_loss.py` (new)
**Complexity:** M
**Depends on:** none

Leave `num_train_epochs: 2`, rank 16, alpha 32, dropout 0.05, lr 2.0e-4, batch 1, grad accumulation 8, and the target modules. Change `output_dir` to `results/adapters/b1_assistant_only` so a later train cannot overwrite `results/adapters/b1`.

In `lora_run.py`:

- Read the default variant string from the tokenizer chat template. Do not search `str(list)`.
- Insert `{% generation %}` the way TRL’s Cohere training template does: the span covers assistant `message['content']`, `<|END_RESPONSE|>`, and `<|END_OF_TURN_TOKEN|>`. Assistant content is the `<tool_call>` string for every gold step, including `finish`.
- Set `assistant_only_loss=True`.
- Delete the fallback that prints “retrying with loss on the full sequence” and calls `_fit(False)`. If the mask is empty, abort.
- Before `trainer.train()`, render a few rows and assert assistant turns (segment weight 2) are inside the mask and user and tool turns (weight 0) are outside it. The rendered user and system text must match the unpatched template.
- Save the epoch-1 and epoch-2 adapters. After each epoch, score the 104 validation tasks for `argument_correctness` (record `task_success` beside it) with one replica on the 5060 Ti. Keep the better epoch. Do not run a third epoch.
- Do not swap in TRL’s bundled `cohere2_training.jinja` unless it renders the same turns as Tiny Aya’s default variant.

`sft_dataset.py` stays as it is. The new test checks the marker insertion and the mask assertion on a toy conversation, without loading weights. The function that starts `trainer.train()` must not be invoked by the test.

### P1: Stop generation after the first tool call

**Files:** `actproto/agent/stop.py` (new), `actproto/agent/model.py` (modify), `tests/test_generation_stop.py` (new)
**Complexity:** S
**Depends on:** none

`generate_text` currently decodes until `max_new_tokens` (256). B1’s dominant outcome is `PARSE_ERROR: multiple tool calls in one turn`. Add a pure helper that reports when a completion contains one complete `<tool_call>...</tool_call>` or a Cohere end-of-turn token. Use it as a stopping criterion inside `generate_text`. Leave `parse_tool_call` fail-closed: two calls in one string are still a `ParseError` (`test_parse_multiple_json_refuses`). The test for the helper does not load Tiny Aya.

### R1: Finish-argument repair and one schema retry

**Files:** `actproto/agent/answer_repair.py` (new), `actproto/agent/strategy.py` (modify `model_policy` only), `tests/test_answer_repair.py` (new)
**Complexity:** M
**Depends on:** none

Repair runs only after `parse_tool_call` returns one `ToolCall`. It must not read the oracle or the gold arguments.

- If the call is `finish` and the last tool result is a successful `calculator` whose `data.value` is an integral float, and the model answer is that number with a trailing `.0`, rewrite the answer with the generator rule: `str(int(value))` when `value.is_integer()`, otherwise `str(value)`. `429.0` becomes `429`. Do not rewrite a non-integral string such as `0.1000` into a shorter decimal.
- If the call is `finish` and the last tool result’s canonical field (`price`, `id`, sales number, `NOT_FOUND`, `NO_ROWS`) is a substring of a longer prose answer, snap the answer to that exact field string.
- If `parse_tool_call` fails, or `validate_args` fails, call `generate_fn` at most once more with the same transcript plus a user line that asks for exactly one tool call. A second failure still returns the existing parse-error `finish`. Do not loop.
- Do not reuse arguments from a different tool. That rule in `schema_regen.py` stays.

`scripted_policy` stays unchanged. Tests cover `429.0` → `429`, prose price → `3.1`, and a non-integral growth string left alone.

### H1: Re-encode activations after the tool result

**Files:** `actproto/evaluation/live_runner.py` (modify), `actproto/activations/extract.py` (modify), `scripts/extract_activations.py` (modify), `tests/test_extract.py` (modify)
**Complexity:** M
**Depends on:** none

`make_vector_fn` must encode the current transcript (`state.log.messages`, including tool results) through `messages_for_template`, then take `prompt_last` on that sequence. `_encode_prompt` can stay for the first turn, where the transcript is system plus the user task.

Add a new extractor that replays gold tool results and emits one record per gold step. Label step k with `gold_tools[k]`, including `finish`. Keep `extract_tasks` behavior so the existing one-record test still passes. `extract_activations.py` writes the step-wise records. The new test uses `tiny_lm` and a fake `encode_fn`; it does not load Tiny Aya.

Hop-2 vectors are a different distribution from the first-tool probe. The published probe accuracy of 1.0 is the first call only. Do not claim hop-2 accuracy until a new dump exists, and do not build that dump in this workstream.

### T1: Strictest validation threshold that still abstains

**Files:** `actproto/activations/threshold.py` (modify), `tests/test_prototypes.py` (modify)
**Complexity:** S
**Depends on:** none

`choose_threshold` walks `min_score` from -0.2 upward and keeps a point only when accuracy is strictly higher or coverage is strictly higher. Tied perfect points therefore stay at -0.2 / 0.0, and the router never abstains. Among points with accuracy equal to the best accuracy and coverage at least `min_coverage`, keep the highest `min_margin`, then the highest `min_score`. `should_route` and the fail-open behavior of a missing threshold file stay as they are. The existing test (`coverage >= 0.5`, `n` equals the val count) must still pass. Add a tie case where a stricter grid point has the same accuracy and a lower-or-equal coverage and assert that the stricter point wins.

A perfect first-hop margin will still route the first hop. Abstention matters on hop-2 vectors from H1. This task does not retune on the test split.

### D1: Local traces for the thin failing families only

**Files:** `actproto/data/templates.py` (modify), `actproto/data/generator.py` (modify), `tests/test_data_splits.py` (modify)
**Complexity:** S
**Depends on:** none

Add a small number of new entity ids inside `multi_avg_growth`, `error_missing_product`, and `confusable_sales_number`. Reuse the existing builders and `call_tool`. New entities must pass `assert_no_leakage`. Do not add `easy_calc` entities. Do not add a Hugging Face loader. Do not change `assign_split_by_index`. Paraphrase padding stays on the source split. The official eval continues to regenerate `generate_tasks()` and filter by split.

### G1: Gated GPU run of the new adapter and the new router

**Files:** `actproto/configs/trial.yaml` (modify), `docs/EXPERIMENT_STATUS.md` (append a short pointer only)
**Complexity:** M
**Depends on:** S1, L1, P1, R1, H1, T1, D1

This is the only GPU work, and it waits until any live `run_trial.py` has exited on its own. One replica, `ACTPROTO_CUDA_DEVICES=1`.

- Train with the L1 entry into `results/adapters/b1_assistant_only/`. Do not pass `--plan-only` for that run. Do not overwrite `results/adapters/b1`.
- Extract step-wise activations, rebuild prototypes and the threshold, and refit the probe from the new dump. Write the new artifacts beside the old ones (`results/activations_span/`, `results/prototypes_span.npz`, `results/prototype_metrics_span.json`) so `results/prototypes.npz` and `results/prototype_metrics.json` stay the B2 evidence.
- Add trial run ids that point at the new adapter and the new prototype files. Leave the B0, B1, and B2 entries and their output directories unchanged.
- Score the new adapter with and without the router on the same 104-task test. Compare argument correctness and task success to B0 (0.269 and 0.000) and B1 (0.024 and 0.010). Slice the new `eval.json` with S1.
- If tool choice collapses again, the mask is still on the wrong span. Fix L1. Do not raise epochs, rank, or the learning rate.
- Append one pointer to `docs/EXPERIMENT_STATUS.md`. Do not rewrite that file.

## New Dependencies

| Package | Feature | Optional? |
| --- | --- | --- |
| none | The pass uses the installed torch, transformers, peft, and trl. | |

## File Change Summary

| File | Action (new/modify/delete) |
| --- | --- |
| `scripts/slice_eval_buckets.py` | new |
| `tests/test_slice_eval_buckets.py` | new |
| `results/b0_b2/bucket_slice.md` | new |
| `actproto/training/lora_run.py` | modify |
| `actproto/configs/sft.yaml` | modify |
| `tests/test_lora_assistant_loss.py` | new |
| `actproto/agent/stop.py` | new |
| `actproto/agent/model.py` | modify |
| `tests/test_generation_stop.py` | new |
| `actproto/agent/answer_repair.py` | new |
| `actproto/agent/strategy.py` | modify |
| `tests/test_answer_repair.py` | new |
| `actproto/evaluation/live_runner.py` | modify |
| `actproto/activations/extract.py` | modify |
| `scripts/extract_activations.py` | modify |
| `tests/test_extract.py` | modify |
| `actproto/activations/threshold.py` | modify |
| `tests/test_prototypes.py` | modify |
| `actproto/data/templates.py` | modify |
| `actproto/data/generator.py` | modify |
| `tests/test_data_splits.py` | modify |
| `actproto/configs/trial.yaml` | modify |
| `docs/EXPERIMENT_STATUS.md` | modify |

## Open Questions

- B2 averages are unknown until the live trial writes `results/b0_b2/B2/eval.json`. S1 includes that file when it exists and does not start a replacement run.
- If hop-2 vectors remain as separated as the first hop, the stricter threshold will still cover them. In that case G1 reports coverage and the next lever is the decode stop plus argument repair, not another threshold search.
- Validation argument correctness costs one 104-task generation pass per epoch. That is the early-stop metric. It runs only inside G1, on the 5060 Ti, after the current GPU work is idle.
