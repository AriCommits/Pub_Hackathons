# Plan 1: Raise verified task success without more SFT capacity

This file is the `create_plan` artifact for the next ActProto-LoRA step (Lexsi Track 3). It supersedes any unfinished parallel plan for this next step, including the planner that `docs/EXPERIMENT_STATUS.md` notes did not finish (no `OVERVIEW.md` is in the repo). If an `OVERVIEW.md` appears later, leave it; do not treat it as the plan to execute.

Do not implement this plan in the same session that writes it. Do not start training or eval. Do not kill processes. Do not commit. Do not rerun B0, B1, or B2.

## Summary

Tool identity on this 5-tool set is already solved. Verified task success is near zero because argument values are wrong, and the one LoRA run trained the wrong tokens. B1 (full-sequence loss, token accuracy 0.9945) is worse than the base model. The next step is a bucket slice of the existing evals, then one training change: assistant-only loss on the tool-call and finish turns, early-stopped on validation argument correctness, with rank, learning rate, and epoch cap unchanged. Skip class-balanced and focal loss. Do not add data, noise, a new tool catalog, or a second copy of the model until that comparison exists.

## Goals

- Measure where B0 and B1 fail (bucket, template family, wrong tool vs wrong arguments) before any retrain.
- Train one new adapter whose loss is only on assistant tool-call and finish tokens, including the end-of-turn token.
- Keep the checkpoint with higher validation argument correctness, at most 2 epochs.
- Score that adapter on the same clean test split and compare it to B0 and B1.
- Leave the live B2 process alone until `results/b0_b2/B2/eval.json` exists.

## Non-goals

- Raising epochs, LoRA rank, LoRA alpha, or learning rate.
- Class-balanced loss, focal loss, or class-weighted token loss on this run.
- Replacing the 5-tool generator with ToolBench, API-Bank, Gorilla, ToolACE, xLAM, BFCL, or any other Hugging Face tool-calling corpus.
- Injecting noise into train only.
- A multi-sprint roadmap, a parallel implementation DAG, or a GRPO run.
- Overwriting `results/adapters/b1` or `results/b0_b2/{B0,B1,B2}`.
- Using `cuda:0` (RTX 4060, 8 GB).

## Background

Model: `CohereLabs/tiny-aya-global`, revision `af89d219b53ed9b13b8a4645f9c8028973510324`, 3.35B, bf16. Environment: `lexsi/.venv`, Python 3.12, uv, torch 2.11+cu128, transformers 5.22, peft, trl 1.17. Trust PyTorch GPU indices. `cuda:0` is the RTX 4060 (8 GB) and already stalled a bf16 replica. `cuda:1` is the RTX 5060 Ti (~16 GB).

Tools, and only these: `calculator`, `search_docs`, `query_database`, `lookup_record`, `finish`. `no_tool` aliases `finish`. Train and eval use the same schemas. There is no 15-tool catalog in this repo.

Data comes from `generate_tasks()` in `actproto/data/generator.py`. Splits are 288/104/104 by `(template_family, entity_id)` via `assign_split_by_index` 60/20/20 inside each family. Templates are shared across splits. Paraphrase copies (prompt prefixes) stay on the source split. `task_loader` / `run_eval` regenerate `generate_tasks()` and filter by split. The test set is not a wild set. No noise is injected.

Bucket counts: easy 271 (train 159), confusable 96 (54), multi_step 24 (12), no_tool 45 (27), error_recovery 60 (36). `easy_calc` is 96 of 288 train tasks.

Gold call counts: every task ends in `finish`. Train calls: calculator 108, query_database 105, lookup_record 51, finish 288, search_docs 9. Train first call: calculator 96, query_database 105, lookup_record 51, finish 27, search_docs 9. Test first call: 32 / 39 / 21 / 9 / 3.

B1 training that already ran: LoRA r=16, alpha=32, dropout 0.05, targets `q/k/v/o/gate/up/down`, lr 2e-4, 2 epochs, batch 1, grad accum 8, 72 steps, max_length 960, sequence max 957. `assistant_only_loss` was false. `train_loss` 0.3053, `mean_token_accuracy` 0.9945. Adapter: `results/adapters/b1/`. SFT row schema stays `{messages, segment_weights, loss_mask}`. `loss_mask` is true iff segment weight is 0. User and tool weight 0, assistant weight 2. TRL did not consume those fields; it trained the rendered full sequence.

Routing: the vector encodes only the original user task, not the post-tool conversation (`make_vector_fn` in `actproto/evaluation/live_runner.py` calls `_encode_prompt` on `state.task`). Validation threshold: `min_score` -0.2, `min_margin` 0.0, coverage 1.0, so the router never abstains. Prototype and linear-probe test accuracy are 1.0 on the 5 first-tool classes. The probe is `LogisticRegression` with no `class_weight`. That accuracy is not verified task success.

Eval n=104, already on disk:

| Run | task_success | tool_call_f1 | tool_choice_accuracy | argument_validity | argument_correctness | step_count | latency_p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 base, no adapter, no router | 0.0 | 0.880 | 0.681 | 0.995 | 0.269 | 1.85 | 4.33s |
| B1 LoRA, no router | 0.0096 | 0.696 | 0.115 | (do not invent) | 0.024 | 1.03 | 21.1s |

`argument_correctness` in `actproto/evaluation/metrics.py` is exact name-and-arguments equality per gold step. B0 already emits schema-valid calls (`argument_validity` 0.995) whose values are usually wrong (0.269). B1 then collapsed tool choice (0.681 to 0.115) and step count (1.85 to 1.03).

`results/b0_b2/B2/eval.json` does not exist yet. B2 is live: terminal `236925`, pid 9252, `ACTPROTO_CUDA_DEVICES=1`, `scripts/run_trial.py --run B2`, two replicas on the 5060 Ti, started 2026-11-02 19:59 America/Chicago. Do not start another B2, and do not rerun B0 or B1, while that process is alive. Do not kill it.

## What the literature says about this failure

Tool selection and argument fidelity are different errors. API-Bank (Li et al., EMNLP 2023) splits failures into unparsed calls, wrong or hallucinated API names, and parameters that are missing, placeholder, or semantically wrong. Gorilla (Patil et al., NeurIPS 2024) scores an unknown API as a hallucination and a known API invoked with the wrong arguments as an error, using AST subtree matching. BFCL (Yan et al., 2024) and later function-calling writeups keep the same split: format validity stays high while selection or argument match is what fails. DualTune (arxiv 2510.00229) states the same split explicitly: selection is classification; arguments are structured values plus syntax. In their on-device setup, selection was the bottleneck. Here the split goes the other way. The probe is perfect, B0 argument validity is 0.995, and B0 argument correctness is 0.269. Copying DualTune's fix (a better tool classifier) would not move verified success.

Assistant-only loss is the standard SFT mask, not a new objective. TRL's SFT docs (`assistant_only_loss=True`) compute loss only on assistant spans, and only if the chat template contains `{% generation %}` / `{% endgeneration %}`. `completion_only_loss` is the prompt-completion equivalent and is the wrong switch for these `{messages}` rows. TRL 1.17 will try `get_training_chat_template` when the flag is on and the template has no markers. That helper swaps in a bundled template only on exact string equality, including `cohere2_training.jinja`. Otherwise it raises. Tiny Aya ships the template as a named list. The default variant does contain `<|START_RESPONSE|>{{ message['content'] }}<|END_RESPONSE|>`. `_with_generation_markers` in `actproto/training/lora_run.py` searches `str(chat_template)`. Stringifying the list escapes the quotes, the needle misses, and training silently uses full-sequence loss. Full-sequence loss fits user text, the long system preamble, and tool results that `messages_for_template` rewrites as user turns. Token accuracy can sit at 0.99 while the assistant span that eval parses gets worse. B1 is that outcome.

More epochs or a higher rank will not fix a saturated token accuracy on shared templates. Chu et al. (ICML 2025, "SFT Memorizes, RL Generalizes") find SFT memorizes the training rule and drops on variants; extra SFT is not the generalization lever. "LoRA Users Beware" (arxiv 2506.11402) finds that under light spurious correlations, higher LoRA rank increases reliance on the shortcut. Shared `template_family` strings are that shortcut. Entity holdout is not distribution shift: `assign_split_by_index` keeps a `(family, entity_id)` in one split, and paraphrase copies stay on the source split, but every family is present in train and test. A later paper on long chain-of-thought SFT (arxiv 2604.06628) shows some SFT "failures" are under-training. That does not apply here. Token accuracy is already 0.9945, and the adapted model is worse than the base model on the metrics that matter.

Class-balanced loss (Cui et al., CVPR 2019) and focal loss (Lin et al., 2017) reweight a classification loss. Focal loss down-weights examples the model is already confident on. The NLP imbalance survey (arxiv 2210.04675) treats them as classification tools; on sequence labeling, Dice beat focal. They help when a rare class is never selected. They do not supply the right argument string. `search_docs` is rare (9 train calls, 3 test first calls) and `multi_step` is small (12 train tasks), but the 5-way probe is already 1.0 with no `class_weight`. At token accuracy 0.9945, a focal term `(1-p)^γ` shrinks the gradient on almost every token, including the argument tokens inside majority tools (`calculator` 96 train first calls, `query_database` 105). Up-weighting the `search_docs` name token cannot move exact-match argument correctness on the bulk of the 104-task test. Skip focal loss and class-balanced loss. Do not spend a GPU run on them. Revisit only if the bucket slice shows `search_docs` or `multi_step` failing by tool name while `easy_calc` and `easy_query` are argument-correct. Even then, the experiment is a classifier-side check, not the SFT loss.

Schema-constrained decoding and argument repair are inference constraints, not a substitute for the loss mask, and they are the wrong first lever. ToolDec (arxiv 2310.07075) masks tokens so names exist and argument types parse. Constrained JSON decoding has the same limit: valid is not correct. B0 is already valid. APIGen (Liu et al., NeurIPS 2024, the xLAM data pipeline) rejects candidates with a format check, an execution check, and a semantic check. That is how to filter new local traces. It is not a reason to import an external corpus, and it is not a reason to reweight the current loss. A test-time retry may resample a schema-invalid call. It must not call the test oracle to accept or reject a sample.

Train-only noise would make the train distribution differ from the clean test on purpose. If a stress set is built later, the same noise has to be in the training copies and in a separate eval slice. The official test split stays clean.

## Decision on focal and class-balanced loss

Skip them for the next run. The bottleneck is argument values, not rare-class tool identity. The evidence is local (probe accuracy 1.0, B0 validity 0.995, B0 argument correctness 0.269, B1 tool-choice collapse) and matches the classification-vs-structured-output split in the papers above. A small focal-loss ablation is not worth the 5060 Ti time until the bucket slice contradicts that split.

## Next step, in order

### 0. Leave the live B2 process alone

While terminal `236925` (pid 9252, `run_trial.py --run B2`, `ACTPROTO_CUDA_DEVICES=1`, two replicas) is running:

- Do not start `train_lora.py`, `extract_activations.py`, `run_trial.py`, or a second B2.
- Do not rerun B0 or B1.
- Do not kill the process.
- CPU-only reads of `results/b0_b2/B0/eval.json` and `results/b0_b2/B1/eval.json` are fine.

When `results/b0_b2/B2/eval.json` exists, run `scripts/compare_results.py` and add a short pointer plus the B2 averages to `docs/EXPERIMENT_STATUS.md`. Do not invent B2 numbers before that file exists. Slice B2 with the same script as B0 and B1. Do not restart B2 to produce the file.

### 1. Measure first: bucket slice, no GPU

Join each row in `results/b0_b2/B0/eval.json` and `results/b0_b2/B1/eval.json` to `generate_tasks()` on `case_id`. Keep test rows only. `score_episode` does not store `bucket`; `task_to_case` puts it on `AgentCase.category`, but the written metrics do not. The join is the source of bucket and `template_family`.

For each run, each bucket, and each `template_family`, report n, `task_success`, `tool_choice_accuracy`, `argument_validity`, `argument_correctness`, and `step_count`. Also tabulate first predicted tool against first gold tool.

Label every row as one of:

- wrong tool name
- right tool, wrong arguments
- arguments match gold, oracle still fails
- too few steps (B1 mean step count is 1.03; gold always includes `finish`)

`easy_calc` is a third of train. An average can move because that family moved and still hide `search_docs` (3 test first calls) and `multi_step` (test share of 24). Read the slice before deciding the retrain worked.

Write the tables to `results/b0_b2/bucket_slice.md`. This step does not train and does not call the model.

### 2. The one training change

Wait until the 5060 Ti is free (B2 exited on its own). Then change only the loss mask.

Keep all of the following: r=16, alpha=32, dropout 0.05, the same target modules, lr 2e-4, batch 1, grad accumulation 8, max 2 epochs, max_length 960, seed 42, bf16, the same 288 train tasks, and the same system prompt through `messages_for_template`. Row schema stays `{messages, segment_weights, loss_mask}`. Do not ask TRL to read `segment_weights`; it will not. Use those fields as the assertion target for the mask.

Patch `actproto/training/lora_run.py`:

- Read the default variant string from the tokenizer chat template. Do not search `str(list)`.
- Insert markers the way TRL's `cohere2_training.jinja` does: `{% generation %}` covers `{{ message['content'] }}`, `<|END_RESPONSE|>`, and `<|END_OF_TURN_TOKEN|>`, then `{% endgeneration %}`. Assistant content is only the `<tool_call>...</tool_call>` string and the final `finish` call. There is no separate thinking channel to mask in.
- Set `assistant_only_loss=True`.
- Delete the fallback that retries with full-sequence loss. If the mask is empty, or `is_chat_template_stop_token_trained` is false, abort.
- Before `trainer.train()`, render a few train rows and assert: assistant turns (segment weight 2) have a non-empty mask; user and tool turns (weight 0) are masked out; the rendered user and system text matches the unpatched template, so eval formatting does not move.
- Do not swap in TRL's bundled `cohere2_training.jinja` unless a string compare shows it renders the same turns as Tiny Aya's default variant. Exact equality already fails because of the named-list wrapper. A mismatched template would train a different string than `generate_text` parses.

Save checkpoints under `results/adapters/b1_assistant_only/`, not over `results/adapters/b1/`.

Early stop on validation argument correctness, not on train loss or token accuracy:

- Save the epoch-1 and epoch-2 adapters.
- After each epoch, score the 104 validation tasks with one replica on the 5060 Ti. Metric: `argument_correctness`, with `task_success` recorded beside it.
- Keep the epoch with the higher validation argument correctness.
- Do not run a third epoch. If epoch 2 is worse, the epoch-1 adapter is the one to test.

### 3. Evaluate that one adapter

Same test split the official eval already uses: regenerate `generate_tasks()`, filter `split == "test"`, n=104. One replica. `ACTPROTO_CUDA_DEVICES=1`. New run id and new output directory. Do not overwrite B0, B1, or B2.

Report the same averages as B0/B1, plus the bucket and family slice from step 1.

Read the result in this order:

- If test argument correctness and task success both beat B0 (0.269 and 0.0) and B1 (0.024 and 0.0096), and tool choice does not collapse the way B1 did, keep this adapter and stop SFT changes.
- If the new adapter only recovers B0, the mask fix removed the regression. Do not add epochs, rank, or focal loss. Go to step 4.
- If tool choice falls again, the mask is still on the wrong span. Fix the mask. Do not retune hyperparameters.

### 4. Only after step 3, and only for the failing slice

Do these one at a time, each with the same clean test split. Do not bundle them into the training run above.

- Hop-2 routing. Re-encode the activation after the tool result. The current vector is the original user task only, so the second call cannot depend on the tool output. Prototypes and the threshold have to be rebuilt from those states. The current threshold never abstains; retune `min_score` and `min_margin` on validation so low margin can abstain. Missing threshold files fail open today (`actproto/agent/abstention.py`).
- Schema retry. If the span is not valid JSON, or `validate_args` fails, resample once. Do not use the test oracle as the rejector. Constrained decoding is optional and will not fix values that already validate.
- Extra data only for families the slice shows are failing. New entity ids. Same `assign_split_by_index` rule so a new entity does not land in two splits. Traces must be produced and checked by this harness (tool execution plus the task oracle), in the spirit of APIGen's format and execution filters. Do not download a replacement corpus.
- A matched stress slice is optional and separate. Same noise in the added train rows and in that stress eval. The official test file stays clean. Train-only noise is not a robustness test.

GRPO (`scripts/train_grpo.py`) is out of scope until assistant-only SFT has been scored. SFT-then-RL is a later paper result, not this step.

## What not to change

- Epoch cap 2, rank 16, alpha 32, dropout 0.05, lr 2e-4, batch 1, grad accum 8, max_length 960.
- The five tools and their schemas.
- The scored generator, the 60/20/20 family split, and the clean test set.
- B0 and B1 adapters and reports.
- Focal loss, class-balanced loss, and `class_weight` on the probe.
- GPU: 5060 Ti only. One replica for the new val and test evals. Two copies are what made B1 latency sit at 21s and what stalled the 4060 earlier.

## Files

| Path | Change |
| --- | --- |
| `actproto/training/lora_run.py` | Default-variant marker patch, `assistant_only_loss=True`, no full-sequence fallback, new output dir, epoch checkpoints |
| `tests/test_sft_dataset.py` or a neighbor | Mask assertion on a toy conversation: assistant on, user and tool off, end-of-turn inside the span |
| `scripts/slice_eval_buckets.py` (new) | CPU join of eval rows to `generate_tasks()` |
| `results/b0_b2/bucket_slice.md` | Slice output |
| `results/adapters/b1_assistant_only/` | New adapter only |
| `docs/EXPERIMENT_STATUS.md` | Short pointer when B2 `eval.json` exists, plus a one-line pointer at this plan. Do not rewrite the status doc |

`actproto/training/sft_dataset.py` stays as it is. `segment_weights` remain the spec the mask must match.

## Open questions

- B2 averages are unknown until `results/b0_b2/B2/eval.json` is written by the process that is already running.
- The bucket slice has to say whether B1's step count of 1.03 is an immediate `finish` or one wrong call. That changes nothing about the loss-mask fix. It decides whether step 4 starts with hop-2 routing or with argument repair.
- Validation n=104 is the right early-stop set. Do not early-stop on a handful of `search_docs` rows.
