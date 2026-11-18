# Sprint 1 — One implementation pass

**Plan:** docs/plan_1/OVERVIEW.md
**Wave:** 1 of 2
**Can run in parallel with:** none — this wave is the whole code pass, and every agent in it runs together
**Must complete before:** Sprint 2

No agent in this wave loads Tiny Aya, starts `train_lora.py`, `extract_activations.py`, `build_prototypes.py`, `train_probe.py`, or `run_trial.py`, or writes under `results/adapters/` or `results/b0_b2/B0|B1|B2`. Interpreter: `C:\Users\arian\Github\Hackathons\lexsi\.venv\Scripts\python`. Working directory: `C:\Users\arian\Github\Hackathons\lexsi`.

The seven agents touch disjoint files. A machine with fewer people still does this as one pass: split the agents across people, land all seven, then stop. Sprint 2 is the gated GPU run.

---

## Agents in This Wave

### Agent A: S1 — Slice eval.json by bucket, family, and parse-error

**Complexity:** S
**Estimated time:** 1 hour
**Files to modify:**
- `scripts/slice_eval_buckets.py` (NEW) — join eval rows to `generate_tasks()` and print the tables
- `tests/test_slice_eval_buckets.py` (NEW) — fixture-only test
- `results/b0_b2/bucket_slice.md` (NEW) — tables for B0 and B1

**Depends on:** none
**Blocks:** G1

**Instructions:**
Read `results/b0_b2/B0/eval.json` and `results/b0_b2/B1/eval.json`. Join `case_id` to `generate_tasks()` for bucket and `template_family`. Strip a `-pN` suffix before matching. For each run, bucket, and family, report n, task_success, tool_call_f1, tool_choice_accuracy, argument_validity, argument_correctness, and step_count. Count a predicted `finish` whose answer starts with `PARSE_ERROR` separately from a real finish. Also tabulate the first predicted tool against the first gold tool. Include B2 only when `results/b0_b2/B2/eval.json` is already on disk. The test builds a few fake rows and does not read the real eval files. Do not call the model.

Known shape of the real files, so the script can be checked against them: B0 `easy_calc` is 32 rows, both tools correct, finish answer like `429.0` against gold `429`. B1’s dominant answer is `PARSE_ERROR: multiple tool calls in one turn`, and `model_policy` stores that as `finish`.

**Definition of Done:**
- [ ] `bucket_slice.md` has a bucket table and a family table for B0 and B1
- [ ] Parse-error finishes are a separate count from real finishes
- [ ] `tests/test_slice_eval_buckets.py` passes
- [ ] No model load and no change to the eval JSON files

### Agent B: L1 — Assistant-only loss and early stop, same two epochs

**Complexity:** M
**Estimated time:** 3 hours
**Files to modify:**
- `actproto/training/lora_run.py` — mask the real chat-template variant, refuse the full-sequence fallback, keep both epochs
- `actproto/configs/sft.yaml` — `output_dir: results/adapters/b1_assistant_only` only
- `tests/test_lora_assistant_loss.py` (NEW) — marker and mask checks without weights

**Depends on:** none
**Blocks:** G1

**Instructions:**
Leave epochs at 2, rank 16, alpha 32, dropout 0.05, lr 2.0e-4, batch 1, grad accumulation 8, and the target modules. The current bug is in `_with_generation_markers`: `tokenizer.chat_template` is a named list, `str(...)` escapes quotes, the needle misses, and `_fit(False)` trains every token. Read the default variant string. Insert `{% generation %}` so the span covers assistant content, `<|END_RESPONSE|>`, and `<|END_OF_TURN_TOKEN|>`. Set `assistant_only_loss=True`. Delete the `except` path that retries with full-sequence loss. Abort if the assistant mask is empty.

Before training, render a few `tasks_to_sft_rows` conversations (system prompt prepended, `messages_for_template` applied) and assert segment weight 2 is inside the loss and weight 0 is outside it. User and system text must match the unpatched template. Save epoch 1 and epoch 2 under `results/adapters/b1_assistant_only/`. The early-stop hook scores validation `argument_correctness` after each epoch and keeps the better epoch. It must not run a third epoch, and the unit test must not call it.

Do not edit `sft_dataset.py`. Do not load Tiny Aya in the test. Do not start `trainer.train()`.

**Definition of Done:**
- [ ] A unit test shows the generation markers land on the default variant string, not on `str(list)`
- [ ] A unit test shows assistant turns unmasked and user/tool turns masked
- [ ] The full-sequence fallback is gone
- [ ] `sft.yaml` still says `num_train_epochs: 2` and the output dir is `results/adapters/b1_assistant_only`
- [ ] Existing `tests/test_sft_dataset.py` still passes

### Agent C: P1 — Stop generation after the first tool call

**Complexity:** S
**Estimated time:** 1.5 hours
**Files to modify:**
- `actproto/agent/stop.py` (NEW) — pure “one complete tool call” detector
- `actproto/agent/model.py` — stopping criterion inside `generate_text`
- `tests/test_generation_stop.py` (NEW) — detector tests, no weights

**Depends on:** none
**Blocks:** G1

**Instructions:**
B1 fails closed because `parse_tool_call` sees two `<tool_call>` blocks in one `generate()` result, and `model_policy` turns that into `finish`. Add `completion_should_stop(text) -> bool` that is true once the decoded new tokens contain one closed `<tool_call>...</tool_call>` or `<|END_OF_TURN_TOKEN|>`. Call it from a `generate` stopping criterion in `generate_text`. Leave `max_new_tokens` as the hard cap. Do not change `parse_tool_call`. Two calls in one string remain a `ParseError`; `tests/test_agent_parse.py::test_parse_multiple_json_refuses` must stay green. The new test covers an empty string, a half-open tag, one closed tag, and an end-of-turn token. It does not import `load_model`.

**Definition of Done:**
- [ ] `completion_should_stop` is true only after one complete tool call or an end-of-turn token
- [ ] `generate_text` passes that criterion into `model.generate`
- [ ] `tests/test_agent_parse.py` still passes
- [ ] No weight load in the new test

### Agent D: R1 — Finish-argument repair and one schema retry

**Complexity:** M
**Estimated time:** 2.5 hours
**Files to modify:**
- `actproto/agent/answer_repair.py` (NEW) — deterministic finish rewrite from the last tool result
- `actproto/agent/strategy.py` — `model_policy` only
- `tests/test_answer_repair.py` (NEW)

**Depends on:** none
**Blocks:** G1

**Instructions:**
B0 `easy_calc` already emits `calculator` then `finish`, and the oracle fails because the answer is `429.0` instead of `429`. The calculator returns `data.value` as a float. Repair only after a single successful `ToolCall`.

- Integral calculator value whose model answer is the same number with a trailing `.0` becomes `str(int(value))`. Leave a non-integral answer such as `0.1000` unchanged.
- If the last tool result’s canonical field is a substring of a longer finish answer, snap to that field (`price`, document id, sales number, `NOT_FOUND`, `NO_ROWS`).
- If parsing fails or `validate_args` fails, call `generate_fn` once more with one extra user line asking for exactly one tool call. A second failure keeps today’s parse-error `finish`.
- Do not read gold arguments or the oracle. Do not reuse another tool’s arguments. Do not change `scripted_policy` or `schema_regen.py`.

**Definition of Done:**
- [ ] Test: calculator value `429.0` and finish answer `"429.0"` becomes `"429"`
- [ ] Test: a prose price that contains `3.1` becomes `"3.1"`
- [ ] Test: `"0.1000"` is unchanged
- [ ] `model_policy` performs at most one extra generation
- [ ] `tests/test_schema_regen.py` and `tests/test_agent_harness.py` still pass

### Agent E: H1 — Re-encode activations after the tool result

**Complexity:** M
**Estimated time:** 3 hours
**Files to modify:**
- `actproto/evaluation/live_runner.py` — `make_vector_fn` encodes the current transcript
- `actproto/activations/extract.py` — new step-wise extractor; `extract_tasks` stays
- `scripts/extract_activations.py` — write step-wise records
- `tests/test_extract.py` — new tiny-LM test for two gold steps

**Depends on:** none
**Blocks:** G1

**Instructions:**
`make_vector_fn` today calls `_encode_prompt(loaded, state.task)`, so hop 2 routes on the original user string. Encode `state.log.messages` with `messages_for_template`, apply the chat template with `add_generation_prompt=True`, and keep `prompt_last`. The first turn is system plus the user task; later turns include tool results rewritten as user text.

Add `extract_step_records` (name it clearly) that replays gold tool results and emits one vector per gold step, labeled with that step’s tool, including `finish`. Leave `extract_tasks` returning one first-tool record so `test_extract_records_from_tiny_lm` stays valid. Point `extract_activations.py` at the step-wise function. The new test uses the `tiny_lm` fixture and a fake encoder. Do not run the real extractor and do not load Tiny Aya.

The published probe accuracy of 1.0 is first-tool only. Do not write a new accuracy claim.

**Definition of Done:**
- [ ] `make_vector_fn` uses the transcript, not only `state.task`
- [ ] A two-step toy task yields two records, labeled with the first tool and then `finish`
- [ ] The old `extract_tasks` test still passes
- [ ] No GPU extraction in this wave

### Agent F: T1 — Strictest validation threshold that still abstains

**Complexity:** S
**Estimated time:** 1.5 hours
**Files to modify:**
- `actproto/activations/threshold.py` — tie-break toward a stricter cutoff
- `tests/test_prototypes.py` — tie-break assertion

**Depends on:** none
**Blocks:** G1

**Instructions:**
`choose_threshold` keeps the first grid point when accuracy and coverage tie, which is why `results/prototype_metrics.json` says min_score -0.2, min_margin 0.0, coverage 1.0. Among points whose accuracy equals the best accuracy and whose coverage is at least `min_coverage`, select the highest `min_margin`, then the highest `min_score`. Leave `should_route` and the missing-file fail-open path in `abstention.py` alone. Do not fit on test vectors. `test_threshold_uses_only_val_indices` must still pass (`coverage >= 0.5` and `n` equal to the val count). Add a small tie fixture.

A perfect first-hop score will still route. The stricter cutoff is for hop-2 vectors from H1, which this wave does not compute.

**Definition of Done:**
- [ ] On an accuracy tie, the returned `min_margin` / `min_score` is the stricter point
- [ ] Coverage still respects `min_coverage`
- [ ] `tests/test_prototypes.py` passes
- [ ] No edit to `results/prototype_metrics.json`

### Agent G: D1 — Local traces for the thin failing families only

**Complexity:** S
**Estimated time:** 2 hours
**Files to modify:**
- `actproto/data/templates.py` — only if a builder needs a new entity shape
- `actproto/data/generator.py` — a few new entity ids in three families
- `tests/test_data_splits.py` — leakage still holds, and `easy_calc` did not grow

**Depends on:** none
**Blocks:** G1

**Instructions:**
The scored test stays `generate_tasks()` filtered by split. Do not add a Hugging Face dataset, a loader, or new tool names. Add a handful of new entity ids only in `multi_avg_growth`, `error_missing_product`, and `confusable_sales_number`. Reuse the existing builders and `call_tool`. `assign_split_by_index` is unchanged, and paraphrase clones stay on the source entity’s split. Do not add `easy_calc` entities: those 32 test failures are the `429.0` format bug, which R1 fixes. Every new task must survive `assert_no_leakage`.

**Definition of Done:**
- [ ] `test_no_template_entity_leakage` and `test_buckets_present` pass
- [ ] New rows exist only in the three families above
- [ ] `easy_calc` entity count is unchanged
- [ ] No network fetch and no new dependency
