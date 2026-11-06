# Sprint 3 — Reward breakdown and tied-batch gate

**Plan:** docs/plan_13/OVERVIEW.md
**Wave:** 3 of 5
**Can run in parallel with:** none — serial
**Must complete before:** Sprint 4, Sprint 5

---

## Agents in This Wave

Sprints 1 and 2 are merged. The study rollout uses the named `default` template and 256 new tokens per turn. The trainer scores live sessions with `bounded_prefix_v1`. This wave adds the component log and the tied-batch gate. It is alone because it edits `actproto/training/grpo_run.py`, `actproto/training/rl_rollouts.py`, and `tests/test_train_grpo.py`, which Sprint 2 just changed, and `actproto/training/rl_config.py` plus `tests/test_grpo_config.py`, which Sprint 4 needs next.

Zero GRPO loss is tied rewards, not a solved val set. The 200/200 macro `1.0` is frozen SFT e2 (`results/sft_v2/e2-scored`). A larger group of identical rewards is still advantage 0. Beta stays `0.0`. Do not open the sealed test split. Do not put Tiny Aya on the 8 GB RTX 4060 (`5b885e12-623f-268c-f07c-ee33db379969`). The 5060 Ti UUID is `134c34ef-c155-b001-4e3a-b9134b072c21`. Do not train and do not edit `docs/plan_13/OVERVIEW.md`.

Run tests with `C:\Users\arian\Github\Hackathons\lexsi\.venv\Scripts\python.exe` from the `lexsi` directory. Never use `C:\Python314\python.exe`. If you commit, `git add` only the files listed below. Do not run `git add .`. Do not push.

### Agent A: G4 — Log the reward breakdown and gate a fully tied batch

**Complexity:** L
**Estimated time:** 6–8 hours
**Files to modify:**
- `actproto/training/grpo_run.py` — step diagnostics, tied-batch resample-or-skip, gradient zeroing
- `actproto/training/rl_rollouts.py` — expose reward components, termination reason, and a second seeded draw
- `actproto/training/rl_config.py` — `guard_tool_sampling` rejects a larger group as a tie fix
- `tests/test_train_grpo.py` — redraw, skip, frozen weights, `frac_reward_zero_std`
- `tests/test_rl_rollouts.py` — component and termination fields
- `tests/test_grpo_config.py` — group-size error text

**Depends on:** G1, G2
**Blocks:** G5, G6

**Instructions:**
Read `docs/plan_13/OVERVIEW.md` for context. This task is fix (3): log reward components and the termination reason, and resample or skip an all-tied batch. Do not manufacture an advantage and do not add jitter.

Keep Sprint 1's sampler: `selected_template(tokenizer, "default")`, 256 new tokens on every sampled turn, and `context_overflow` when `prompt_len + 256` exceeds `max_position_embeddings`. Keep Sprint 2's `_executed_trace_reward` on `bounded_prefix_v1`. Do not switch it back to `bounded_shaped_v1`. When a resample finalizes the ledger, the reward function reads that ledger and does not score the session a second time.

Extend the step record written to `step_diagnostics.jsonl`. For each completion in the step, store the raw reward, `verified_success`, `invalid_attempts`, `duplicate_calls`, `excess_calls`, `excess_omitted_reason`, and `completion_reason`. Also store the loss, the gradient norm, reward mean, reward standard deviation, the zero-variance group count, and TRL `frac_reward_zero_std`. Read `frac_reward_zero_std` from the trainer log when it is present, and from `trainer._metrics["train"]["frac_reward_zero_std"]` when the log line omits it. Keep logging `completions/clipped_ratio` and `completions/mean_terminated_length` when TRL provides them, labeled as the EOS-based clip, so a later reader does not confuse that clip with the per-turn budget. Do not write `task.oracle` into the diagnostic file.

Termination reasons that must stay distinct are `finish`, `parse_error`, `truncated`, `template_mismatch`, `timeout`, `budget_exhausted`, `policy_error`, and `context_overflow` from Sprint 1. A constant `-0.1` with `parse_error` is not the same event as a constant `-0.1` with `truncated` or a finished episode that failed verification.

An all-tied group is not given a manufactured advantage. Score the pair before the trainer consumes it. If the two rewards are equal, redraw that prompt's pair once, with a new seed, the same temperature `0.8`, the same template, and the same 256-token per-turn cap. If the redraw is still tied, skip the optimizer update: zero the gradients in a trainer callback before the step and log `skipped_tied_batch=true` with `grad_norm=0`. Do not add jitter to the rewards. A skipped step may still increment TRL's `global_step`; the diagnostic has to say the update was skipped so the step count is not read as learning. Two draws is the cap. A systematic failure will still be tied after the redraw, and the component log is the result.

Update `guard_tool_sampling` and its tests so this plan does not treat `num_generations: 4` as the next step. The loader may keep rejecting a group size other than 2. Do not set `LEXSI_RL_GROUP4_FIT`. Do not change temperature or top-p. State in the config error text that a larger group of identical rewards is still advantage 0.

Do not add `grpo_gpu_tools` to `HISTORICAL_RUNS` and do not create `actproto/configs/grpo_v2_gpu_prefix.yaml`. That is Sprint 4. Do not edit `actproto/configs/grpo_v2_gpu_tools.yaml`. Beta stays `0.0`. Do not set `scale_rewards` to `none`. Do not train, do not open the test split, and do not load Tiny Aya onto the 4060.

**Definition of Done:**
- [ ] A fake policy that ties once and then returns two different prefix rewards produces a nonzero advantage on the redraw and does not skip.
- [ ] A fake policy that always returns `-0.1` skips, leaves a frozen-weight assertion unchanged, and writes both termination reasons and the four components (`verified_success`, `invalid_attempts`, `duplicate_calls`, `excess_calls`).
- [ ] That skip records `frac_reward_zero_std` and `skipped_tied_batch=true` with `grad_norm=0`.
- [ ] EOS clip fields stay labeled as the EOS-based clip. `task.oracle` is absent from the diagnostic file.
- [ ] `context_overflow` stays distinct from `parse_error` and `truncated`.
- [ ] A group size other than 2 is still rejected, and the error text says a larger group of identical rewards is still advantage 0. `LEXSI_RL_GROUP4_FIT` is not required and is not set.
- [ ] `_executed_trace_reward` still uses `bounded_prefix_v1` and does not score a finalized ledger twice.
- [ ] Tests written and passing for the modified files, using `C:\Users\arian\Github\Hackathons\lexsi\.venv\Scripts\python.exe -m pytest tests/test_train_grpo.py tests/test_rl_rollouts.py tests/test_grpo_config.py`
- [ ] No regressions in existing tests
