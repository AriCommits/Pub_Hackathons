# Sprint 1 — Train rollout budget and failure-only prefix credit

**Plan:** docs/plan_13/OVERVIEW.md
**Wave:** 1 of 5
**Can run in parallel with:** none — serial. Agent A and Agent B run in parallel inside this wave.
**Must complete before:** Sprint 2, Sprint 3, Sprint 4, Sprint 5

---

## Agents in This Wave

Zero GRPO loss on the tool-reward runs is tied rewards (advantage 0), not a solved validation set. The 200/200 macro `1.0` is frozen SFT e2 in `results/sft_v2/e2-scored`, not either GRPO run. Beta stays `0.0`. Do not open the sealed test split, pass `--final-eval`, or select a checkpoint on test. Do not put Tiny Aya on the 8 GB RTX 4060 (`5b885e12-623f-268c-f07c-ee33db379969`). The 16 GB RTX 5060 Ti UUID is `134c34ef-c155-b001-4e3a-b9134b072c21`. Do not train, do not launch a GPU job, and do not edit `docs/plan_13/OVERVIEW.md`.

Run tests with `C:\Users\arian\Github\Hackathons\lexsi\.venv\Scripts\python.exe` from the `lexsi` directory. Never use `C:\Python314\python.exe`. If you commit, `git add` only the files listed in your brief. Do not run `git add .`. Do not push.

### Agent A: G1 — Match the train rollout template and per-turn budget to greedy eval

**Complexity:** L
**Estimated time:** 6–8 hours
**Files to modify:**
- `actproto/integrations/agenttune_backend.py` — `render_tokens` takes an explicit chat template
- `actproto/training/rl_rollouts.py` — each sampled study turn gets 256 new tokens; context overflow is its own stop
- `actproto/training/grpo_run.py` — study sampler passes the named template and stops feeding `min(remaining, max_completion_length)`
- `tests/test_agenttune_backend.py` — named-template selection
- `tests/test_rl_rollouts.py` — per-turn budget, truncation, context overflow
- `tests/test_train_grpo.py` — study rollout returns an assembled completion longer than 256 tokens

**Depends on:** none
**Blocks:** G4, G5, G6

**Instructions:**
Read `docs/plan_13/OVERVIEW.md` for context. This task is fix (1) only: an explicit chat template, and 256 new tokens on every sampled turn. It is not one 256-token budget for the whole episode.

`ModelPolicy` in `actproto/agent/inference_v2.py` already calls `selected_template(tokenizer, "default")` from `actproto/training/sft_masks.py`, passes that string as `chat_template`, and allows 256 new tokens per turn with `do_sample=False`. Do not change `ModelPolicy`. Training temperature stays `0.8`. Eval stays greedy.

`render_tokens` currently calls `apply_chat_template` with no `chat_template` argument. It must accept the template explicitly. The study path passes `selected_template(tokenizer, "default")` on every render of the system prompt, the user prompt, an assistant turn, and a tool observation. Keep `messages_for_template` in front of that call so tool observations are still rewritten as user turns. If `tokenizer.chat_template` is a dict and the caller does not pass a template, fail before sampling. A string template may keep today's call so existing fixture tokenizers still run.

`run_trusted_behavior_episode` must stop using one shrinking token pool for the study episode. Each decision that is actually sampled gets `max_new_tokens=256`. `_study_rollout` must stop passing `min(remaining, config.max_completion_length)` and must stop using `max_completion_length` as `max_generated_tokens` for that pool. A turn that hits 256 new tokens without the tool-call stop is still an invalid truncation and still ends the episode. It does not leave a budget of 0 for a later turn, because that later turn is not sampled. Before each turn, apply the same context check as `ModelPolicy`: if `prompt_len + 256` exceeds `max_position_embeddings`, do not sample a shorter hidden budget. Record that stop as `context_overflow`, distinct from `parse_error` and from a turn that hit the 256-token cap. Sprint 3 logs that reason; do not rename it.

`max_completion_length` remains the TRL loss denominator. It is not the per-turn sampler cap. Do not raise the per-turn cap above 256. A unit test builds a two-turn episode whose completion is longer than 256 tokens and asserts the study rollout returns those completion ids intact. That locks the TRL 1.7.1 fact that a custom `rollout_func` is not sliced to `max_completion_length`. If a future pin slices them, the test fails and the trainer cap has to grow to the assembled episode. That larger number must not be copied back into the per-turn sampler.

Leave reward scoring on `bounded_shaped_v1`. Do not add prefix credit, diagnostics, a tied-batch redraw, or `HISTORICAL_RUNS` entries. Those belong to later sprints that also edit `grpo_run.py` and `tests/test_train_grpo.py`. Touch only the template and budget wiring in those files.

CPU and GPU mechanic smokes keep their injected policies and their own short fixture bound. They must not load Tiny Aya. `collect_episode` stays fixture-only. If it still shares one token pool, leave that pool alone unless a test shows the study path accidentally calls it.

Do not edit `actproto/configs/grpo_v2_gpu_tools.yaml`, `actproto/evaluation/verification_v2.py`, `actproto/tools/schemas.py`, or the train/eval scripts. Do not schedule Tiny Aya on the 4060. Do not open the test split.

**Definition of Done:**
- [ ] A tokenizer with two named templates shows the study renderer and `ModelPolicy` both selecting `default`.
- [ ] A three-turn fake episode is allowed 256 new tokens on each sampled turn.
- [ ] A first turn of 256 truncated tokens ends the episode as truncation and does not grant a second turn a leftover of 0.
- [ ] `prompt_len + 256` above `max_position_embeddings` stops as `context_overflow` and does not sample a shorter budget.
- [ ] A two-turn study completion longer than 256 tokens is returned intact.
- [ ] `ModelPolicy` is unchanged: greedy, template name `default`, 256 new tokens per turn. Training temperature is still `0.8`.
- [ ] Mechanic smokes and `collect_episode` do not load Tiny Aya.
- [ ] Tests written and passing for the modified files, using `C:\Users\arian\Github\Hackathons\lexsi\.venv\Scripts\python.exe -m pytest tests/test_agenttune_backend.py tests/test_rl_rollouts.py tests/test_train_grpo.py`
- [ ] No regressions in existing tests

### Agent B: G2 — Add failure-only prefix credit that cannot pass a success

**Complexity:** M
**Estimated time:** 4–6 hours
**Files to modify:**
- `actproto/training/rewards_v2.py` — new `bounded_prefix_v1` version; old versions stay numerically the same
- `actproto/configs/reward_v2.yaml` — document the new constants; not a live override
- `docs/REWARD_SPEC.md` — document the formula and the success ceiling
- `tests/test_rewards_v2.py` — lock old and new scores
- `tests/test_reward_attacks.py` — forged events still raise

**Depends on:** none
**Blocks:** G3, G4

**Instructions:**
Read `docs/plan_13/OVERVIEW.md` for context. This task is fix (2) only: failure-only prefix credit. Every successful episode scores strictly above every failed episode. There is no length bonus.

Add version `bounded_prefix_v1`. Leave `verified_success_v2` and `bounded_shaped_v1` numerically unchanged, including `score_executed_session`, which stays on `bounded_shaped_v1` until Sprint 2 switches the trainer. The YAML file documents the new constants the same way it documents the old ones. It is not a live override.

The prefix count `k` is the number of `required_tools` dependencies matched by the ordered walker already in `TrustedRewardSession.score`. A dependency counts only when every parent in `after` is already matched and `_call_match` accepts the call: the tool, the exact arguments, and any declared state or expected error. `n` is the number of those dependencies. `finish` and `no_tool` do not add prefix credit. A later dependency whose parents are unmet does not count. Unmatched calls, repeated calls, and extra tokens do not increase `k`. Do not read `dependency_argument_coverage` into the reward. Credit comes from the live session events only. A forged trace, a model-written observation, and a serialized rollout JSON still cannot create it.

Let `S` be verified success. The penalty is the existing shaped penalty `0.10*min(I, 3) + 0.05*min(D, 3) + 0.05*S*min(E, 4)`. The excess term stays success-conditioned and stays omitted when `valid_call_bound` is unset. `FAILURE_CEILING` is `0.30`, strictly below the lowest shaped success `0.35` (`1 - 0.65`).

- If `S` is true, the reward is `1 - penalty`. Clean success is `1.0`. Prefix credit is not added on top.
- If `S` is false and `n == 0` or `k == 0`, the reward is `-penalty`, at most `0`.
- If `S` is false and `k >= 1`, `progress = 0.30 * (k / n)` and `previous = 0.30 * ((k - 1) / n)`. The reward is `min(progress, max(previous + 0.01 * (0.30 / n), progress - penalty))`.

That floor keeps a short correct prefix strictly above a long chain with a shorter prefix, even when the short correct episode also takes the maximum invalid and duplicate penalties. The same clamp keeps extra penalties from raising the score. A full correct prefix that still fails the terminal answer tops out at `0.30`, strictly below `0.35`. No term reads token length or raw call count. Do not add a verbosity bonus.

`group_advantages([1, 1])` and `group_advantages([-0.1, -0.1])` stay exact zeros. Do not set `scale_rewards` to `none` and do not add noise. Beta stays `0.0`. Do not switch `_executed_trace_reward`; that is Sprint 2. Do not edit `grpo_run.py`. Do not train, do not open the test split, and do not load Tiny Aya.

**Definition of Done:**
- [ ] `bounded_shaped_v1` tests still match current values, including a partial dependency whose old reward is `0`.
- [ ] `bounded_prefix_v1` locks clean success at `1.0`.
- [ ] Every penalized success is strictly above `0.30`.
- [ ] A prefix of `k` scores strictly above a prefix of `k - 1`.
- [ ] A zero-prefix chain of any length scores at or below `0`.
- [ ] A correct prefix plus extra unmatched calls scores at or below that prefix's `progress`.
- [ ] `group_advantages([1, 1])` and `group_advantages([-0.1, -0.1])` are exact zeros.
- [ ] A forged event raises instead of scoring.
- [ ] `score_executed_session` still returns `bounded_shaped_v1`.
- [ ] Tests written and passing for the modified files, using `C:\Users\arian\Github\Hackathons\lexsi\.venv\Scripts\python.exe -m pytest tests/test_rewards_v2.py tests/test_reward_attacks.py`
- [ ] No regressions in existing tests
