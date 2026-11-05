# Plan 12: Tool-executed GRPO from the frozen Tiny Aya adapter

## Summary

The live Tiny Aya GRPO path optimizes a string check, not tool use. `_train_tiny_aya` in `lexsi/actproto/training/grpo_run.py` calls `create_trainer` with `_public_completion_reward`, which returns 1.0 when the decoded text contains "finish" and 0.0 otherwise, and it does not pass a rollout function. The 20-step run at `results/rl_v2/grpo_gpu_e2` finished with reward 0, reward_std 0, and grad_norm 0, and `checkpoint-20` is byte-identical to the frozen e2 adapter (sha256 `2da2899c0fd2f5ca0bffc32c2e9201e0b7bab8b1758d2fde429ff19cf50be8bd`). This plan attaches in-process tool execution to that live path, scores episodes with the existing `bounded_shaped_v1` verifier reward, keeps the frozen e2 LoRA as the initial policy, leaves beta at 0, and requires a validation comparison against e2 before anyone claims the policy improved.

## Goals

- `_train_tiny_aya` samples from the loaded e2 LoRA, executes each parsed tool call on the train-task harness, and scores the captured trace with `TrustedRewardSession.score(version="bounded_shaped_v1")`. A completion that merely contains the word "finish" is not a success.
- The shaped reward stays the checked-in formula: verified success 0 or 1, minus `0.10 * min(invalid, 3)`, minus `0.05 * min(duplicates, 3)`, and after success minus `0.05 * min(excess calls, 4)`. No token-length bonus is added.
- Equal rewards in a group still produce advantage 0. The run log records reward mean, reward standard deviation, gradient norm, and the zero-variance group count. A step whose groups are all tied does not get a manufactured gradient.
- The trainable weights are the existing LoRA matrices on `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj` (rank 16, alpha 32). Learning rate stays `5e-6`. Beta stays `0.0`, and the trainer still builds no reference model.
- A new output directory under `results/rl_v2` is scored on the val split with the router off and compared with `scripts/compare_experiments_v2.py` to `results/sft_v2/e2/checkpoint-150`. The sealed test split is not opened. The write-up states that `grpo_gpu_e2` did not change the adapter.

## Non-Goals

- Opening the sealed test split, passing `--final-eval`, or regenerating tasks from test failures.
- Replacing the frozen SFT baseline, deleting historical adapters, or treating `results/rl_v2/grpo_gpu_e2` as an improved policy.
- Claiming the previous 20-step run improved anything. Its checkpoint matches e2 and its gradient norm was 0.
- A token-length bonus, a new untuned length penalty, or any reward term that is not already in `bounded_shaped_v1`.
- Full finetuning of Tiny Aya. Adam states on the full bf16 weights do not fit beside rollouts on the 16 GB card.
- Setting beta above 0 against the base model. TRL 1.7.1 with PEFT and no adapter named `ref` disables the adapter and would KL-regularize toward base Tiny Aya, which is not e2. A second full reference model does not fit.
- vLLM, quantization, bitsandbytes, and any rollout path other than the explicit `AgenticRolloutAdapter`.
- Putting Tiny Aya GRPO on the RTX 4060, running two model-bearing GPU jobs at once, or leaving `CUDA_VISIBLE_DEVICES` set in the parent shell.
- Editing AgentTune or TRL. The pins stay AgentTune 1.1.3 at `d4db4c12d52342227c4113a70183d0f64aa2f345` and TRL 1.7.1.

## Background / Context

Plans 9 through 11 are on `master` at `c3d570e`. Plan 9 is the S0–S7 runner and the frozen SFT manifest `lexsi/docs/SFT_BASELINE_MANIFEST.json` (adapter `results/sft_v2/e2/checkpoint-150`, manifest sha256 `6125195bc713f8c4d9e89070233b65667a8068f8fc684a1e14a2522a26b8d670`). Plan 10 R0–R3 is the GRPO stack. Plan 11 is the CUDA env at `C:\Users\arian\Github\Hackathons-worktrees\plan-10-rl-gpu-env` (torch `2.11.0+cu128`, AgentTune 1.1.3, TRL 1.7.1). The CPU mechanics env stays `C:\Users\arian\Github\Hackathons-worktrees\plan-10-rl-env`.

`_public_completion_reward` is the study reward today. The agentic rollout is wired only into the smoke fits: `_fit` and `_gpu_attempt` pass an `AgenticRolloutAdapter` whose policy is an injected calculator-then-finish script, not Tiny Aya. `TrustedRolloutAdapter` still rejects `sampling_mode="behavior"` until a hard-deadline executor exists. `collect_scored_fixture_episode` already runs a scoped `ToolSession` and a `TrustedRewardSession` in one process, and `docs/REWARD_SPEC.md` says that handoff does not wire trainer rewards. `public_prompt_rows` puts only the public user prompt in the trainer dataset and rejects rows that contain the oracle JSON. The trusted scorer has to look the task up from that prompt. It must not put the oracle into the prompt.

`group_advantages` in `lexsi/actproto/training/rewards_v2.py` returns exact zeros when the group variance is 0. TRL 1.7.1 does the same for a tied group under the default `scale_rewards="group"`: advantage is `(reward - group_mean) / (group_std + 1e-4)`, so a numerator of 0 is an advantage of 0. The epsilon does not create a gradient from a tie. It only damps a near-tie. With beta 0, the KL term is also absent (`grpo_trainer.py` sets `ref_model = None` and skips `per_token_kl`). A tied group and beta 0 together are a zero gradient, which is what `grpo_gpu_e2` logged.

A group of 2 from a peaked SFT policy ties often. GRPO's signal exists only when the samples for one prompt disagree in reward (DeepSeekMath; DAPO calls the all-equal case a zero-variance group and drops it). Temperature 0.8 and `num_generations: 2` are already in `actproto/configs/grpo_v2_gpu_e2.yaml`. Temperature 0.8 is not greedy, but two draws from a sharp tool-use policy still often follow the same calls. `num_generations` of 2 is the minimum the config loader accepts. Raising it to 4 raises the chance that one sample differs, and it also raises activation memory, so 4 is allowed only after a measured fit on the 5060 Ti. Setting `scale_rewards` to `none` does not fix a tie: the advantage is still reward minus the group mean. Dividing by a tiny standard deviation would amplify a near-tie (the difficulty bias in Liu et al.), which is the wrong way to invent a signal. The right way is a reward that can differ, then more samples or a higher temperature only if the first tool-reward run is still all ties.

LoRA versus a full update: SFT in `actproto/training/sft_v2.py` trains `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj` at rank 16, alpha 32, dropout 0.05, learning rate `2e-4`. The GRPO smoke and the e2 GPU config use `5e-6`, the same order as the Hugging Face GRPO exercise and well below the SFT rate. TRL's own GRPO default is `1e-6`. Keep `5e-6`. Full finetuning is out of scope: Tiny Aya bf16 weights are about 6.5 GB before rollouts (`allocated_bytes` 6820222464 in `results/rl_v2/gpu_memory/tiny_aya_bf16.json`). The RTX 4060 is 8 GB (`cuda:0` when `CUDA_VISIBLE_DEVICES` is unset, UUID `5b885e12-623f-268c-f07c-ee33db379969`) and must not load this model. The RTX 5060 Ti is 16 GB (`cuda:1`, UUID `134c34ef-c155-b001-4e3a-b9134b072c21`). One model-bearing job at a time. The child may mask devices and must re-check the UUID. The parent shell must not keep `CUDA_VISIBLE_DEVICES`.

Beta and the reference: DeepSeekMath adds a KL term against a reference policy. DeepSeek-R1 uses a small coefficient, 0.001. DAPO removes that term and relies on mixed groups. TRL pull request 2806 skips the reference when beta is 0, which is what this tree already requires. In the installed TRL 1.7.1, beta other than 0 with a PEFT model and no adapter named `ref` runs the forward with the adapter disabled, so the reference is the base model. That is not the frozen SFT policy, and Plan 10 already forbids labeling it as one. Copying e2 onto an adapter named `ref` would be a small extra weight set on the same base, and the KL forward is sequential, but that probe is not part of the first corrected run. A nonzero KL would also move weights when every advantage is 0, which would hide whether the tool reward itself produced a gradient. This plan keeps beta at 0 until a later probe, and that probe is a non-goal here.

Reward: `verified_success_v2` is 1 only when `verification_v2.verify` reports strict verified success. `bounded_shaped_v1` subtracts the capped invalid, duplicate, and success-conditioned excess terms. The maximum penalty is 0.65, so every success outranks every failure (`docs/REWARD_SPEC.md`). Invalid calls, forbidden tools, schema failures, and unexpected tool errors increment the invalid count. Repeats of the same tool, arguments, and pre-call state increment duplicates. Excess calls count only after success and only against `oracle.valid_call_bound`. That is the penalty this plan uses. A length bonus is the wrong objective. GRPO's response-level average gives a short correct answer a larger per-token update than a long correct one, and it penalizes a long wrong answer less per token than a short wrong one (DAPO; Liu et al.). Paying extra for more tokens would train verbosity. DAPO's soft overlong term is a truncation guard, not a bonus, and this plan does not add a new coefficient for it. Tool-use RL that executes calls scores validity and extra calls against a budget (the multi-step tool reward in "Synthesize and Reward"), and final-answer groups that are all right or all wrong teach nothing until they are filtered (Tool-DAPO). The 0.10 invalid weight is already enough to separate a clean failure from a failure with one invalid call, which is what a peaked policy needs before it ever succeeds. Do not retune it in this plan.

Measurement stays on the val split. `actproto/configs/sft_matrix.yaml` `primary` sets `router: false`, `semantic_fallback: false`, and `do_sample: false`. `scripts/run_experiment_v2.py --mode primary --split val` is that contract. `scripts/compare_experiments_v2.py` accepts val metrics only and reports the macro `verified_success_v2` gain against a named baseline. The test split path in `run_experiment_v2.py` requires `--final-eval` and a frozen selection. This plan must not pass those flags.

Research this plan relies on:

- Zhihong Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." https://arxiv.org/abs/2402.03300
- DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." https://arxiv.org/abs/2501.12948
- Qiying Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System at Scale." https://arxiv.org/abs/2503.14476
- Zichen Liu et al., "Understanding R1-Zero-Like Training: A Critical Perspective." https://arxiv.org/abs/2503.20783
- Hugging Face, "GRPO - Do not load reference model when beta == 0," TRL pull request 2806. https://github.com/huggingface/trl/pull/2806
- "Synthesize and Reward — Reinforcement Learning for Multi-Step Tool Use in Live Environments." https://arxiv.org/abs/2606.03892

## Features / Tasks

### T1: Remove the finish-string reward from the live train path
**Files:** `actproto/training/grpo_run.py` (modify), `tests/test_train_grpo.py` (modify), `docs/REWARD_SPEC.md` (modify)
**Complexity:** M
**Depends on:** none

`_train_tiny_aya` must stop passing `_public_completion_reward`. A unit test feeds a decoded completion that contains "finish" and no tool trace and asserts that this string is not a positive training reward. Keep the CPU `_fit` and GPU `_gpu_attempt` smokes on their injected calculator policy. They are mechanics checks, and they must not load Tiny Aya or e2. Update `docs/REWARD_SPEC.md` so the live-handoff section no longer says the trainer is unwired once T2 and T3 land, and so it still says a stored JSON row is not authenticated execution evidence.

**Acceptance:** `tests/test_train_grpo.py` fails the live path if the reward function is `_public_completion_reward`. The smoke tests still expect the injected adapter and beta 0.

### T2: Score executed episodes with the existing shaped verifier reward
**Files:** `actproto/training/grpo_run.py` (modify), `actproto/training/rewards_v2.py` (modify only if the caller needs a thin wrapper), `tests/test_rewards_v2.py` (modify), `tests/test_train_grpo.py` (modify)
**Complexity:** M
**Depends on:** T3

The live reward function calls `TrustedRewardSession.score` with `bounded_shaped_v1` on the trace that T3 captured in-process. Do not score serialized rollout JSON. Do not add a length term. Keep the coefficients in `actproto/configs/reward_v2.yaml`: invalid weight 0.10 capped at 3, duplicate weight 0.05 capped at 3, excess weight 0.05 capped at 4 and only after verified success. Those constants are the decision from the research above: the outcome bit stays larger than the whole penalty (maximum 0.65), and the invalid term is what separates two failures when neither sample verifies. A success with invalid calls must still outrank a clean failure. Tests cover a verified success, a failure with one invalid call, a duplicate, a success-conditioned excess, and a group of two equal rewards whose `group_advantages` are exactly 0.

**Acceptance:** The shaped formula and the 0.65 success-over-failure gap stay as specified. No test introduces a length bonus.

### T3: Roll out real tool calls on the Tiny Aya train path
**Files:** `actproto/training/grpo_run.py` (modify), `actproto/training/rl_rollouts.py` (modify), `tests/test_rl_rollouts.py` (modify), `tests/test_train_grpo.py` (modify)
**Complexity:** L
**Depends on:** none

`_train_tiny_aya` passes an `AgenticRolloutAdapter` into `create_trainer`. The sampler is the loaded `PeftModel`, not the smoke's injected policy. Each decision is parsed, executed on a fresh scoped train-task harness, and returned as an observation before the next sample. `finish` ends the episode. The decision budget is a hard cap, and a timeout or a parse failure is an invalid attempt recorded by the trusted session, not a synthetic successful finish. Prompts stay the public rows from `public_prompt_rows`. The trusted side resolves the `TaskSpec` from that prompt and never writes `task.oracle` into the trainer dataset.

`TrustedRolloutAdapter` may accept an in-process behavior mode only for this trainer callback, and only with a deadline. It still rejects deserialized JSON as execution evidence, caller-supplied sessions, and unscoped fixtures. `collect_scored_fixture_episode` remains the fixture proof that one live session can both execute and score. The study path uses the same session rules on train tasks.

The job runs in the Plan 11 CUDA env, in a child that selects UUID `134c34ef-c155-b001-4e3a-b9134b072c21`. It refuses the 4060 UUID. `use_vllm` stays false. The parent process must not finish with `CUDA_VISIBLE_DEVICES` set. One model-bearing GPU job at a time. The output directory is new, inside `results/rl_v2`, and must not replace `grpo_gpu_e2`, `grpo_gpu_smoke`, `grpo_cpu_smoke`, or any adapter directory. Default pytest does not launch the live job.

**Acceptance:** A fake-model test shows a tool observation changing the next prompt and a ledger built from executed events. The live command is gated and writes only under the new directory. The smoke fits are unchanged.

### T4: Log ties, and do not invent a gradient when the group agrees
**Files:** `actproto/training/grpo_run.py` (modify), `actproto/configs/grpo_v2_gpu_e2.yaml` (modify only to point the next run at a new output directory, or add `actproto/configs/grpo_v2_gpu_tools.yaml`), `actproto/training/rl_config.py` (modify), `tests/test_grpo_config.py` (modify), `tests/test_train_grpo.py` (modify)
**Complexity:** M
**Depends on:** T2, T3

Keep temperature 0.8, top-p 0.95, and `num_generations` 2 for the first tool-reward run. The loader still rejects `num_generations` below 2 and still requires the batch product to be divisible by the group size. Each logged step records reward mean, reward standard deviation, gradient norm, and how many groups had zero variance. If every group in a step is tied, the advantages are 0 and the log says so. Do not flip `scale_rewards` to `none`, do not add noise, and do not add a length bonus to break the tie.

If that first run still logs reward_std 0 on a nonzero number of steps, the only follow-up sampling change allowed inside this plan is a second run with `num_generations` 4, and only after a memory note from the 5060 Ti says four completions fit. A higher temperature is not authorized until that group-size check has been tried or has been shown not to fit. Do not start either run on the 4060.

**Acceptance:** A tied pair of rewards produces a recorded zero-variance group and does not change a frozen-weight assertion in the unit test. The config still has beta 0, learning rate `5.0e-6`, and `use_vllm: false`.

### T5: Keep e2 LoRA as the initial policy and keep beta at 0
**Files:** `actproto/training/grpo_run.py` (modify), `actproto/training/rl_config.py` (modify), `tests/test_grpo_config.py` (modify), `tests/test_train_grpo.py` (modify), `docs/RL_BACKEND_COMPATIBILITY.md` (modify)
**Complexity:** M
**Depends on:** none

Continue to load the manifest adapter with `PeftModel.from_pretrained(..., is_trainable=True)` and set `requires_grad` only on names containing `lora_`. The target modules are the ones already stored on that adapter, which match `sft_v2.py`: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. Do not build a new LoRA on the base model and do not merge the adapter into the base weights. Learning rate stays `5.0e-6`. Output must not overwrite the frozen adapter path.

Beta stays 0.0. `load_grpo_config` keeps rejecting any other beta. The train payload keeps `reference_policy: beta_zero` and `reference_model_built: false`, and it still raises if `trainer.ref_model` is not None. Do not create a PEFT adapter named `ref` in this plan. Document in `docs/RL_BACKEND_COMPATIBILITY.md` that TRL 1.7.1 would treat a missing `ref` adapter as the base model, that a second full model does not fit in 16 GB, and that beta stays 0 so the first nonzero gradient can be attributed to the tool reward.

**Acceptance:** Config tests reject beta other than 0 and reject a reference-policy rename. The train path still refuses to run outside the masked child and still refuses the 4060.

### T6: Compare the new checkpoint to e2 on val before any improvement claim
**Files:** `docs/RL_BACKEND_COMPATIBILITY.md` (modify), `tests/test_train_grpo.py` (modify)
**Complexity:** S
**Depends on:** T3, T4, T5

After a live tool-reward run that records at least one optimizer step, score the new checkpoint with `scripts/run_experiment_v2.py` on the val split, `--mode primary`, matrix `actproto/configs/sft_matrix.yaml` (router off, no sampling). Pass `--device-uuid 134c34ef-c155-b001-4e3a-b9134b072c21` and do not pass `--final-eval` or `--split test`. Compare that val JSON to the existing e2 val metrics for `results/sft_v2/e2/checkpoint-150` with `scripts/compare_experiments_v2.py`. The comparison accepts val only. A gain below 0.05, a failed run, or a checkpoint whose weights match e2 is a negative result and is written that way.

Record the `grpo_gpu_e2` negative control in the same doc section: reward 0, reward_std 0, grad_norm 0, and `checkpoint-20` sha256 `2da2899c0fd2f5ca0bffc32c2e9201e0b7bab8b1758d2fde429ff19cf50be8bd`, identical to e2. Do not describe that run as training progress. The eval that loads Tiny Aya is a model-bearing job and waits until training has released the 5060 Ti. The parent shell does not keep `CUDA_VISIBLE_DEVICES`.

**Acceptance:** The doc names the command, the val-only rule, the e2 baseline path, and the identical-checkpoint hash. A test rejects a comparison payload whose split is `test`.

## New Dependencies

| Package | Feature | Optional? |
| --- | --- | --- |
| none | Tool rollouts, the shaped reward, and the val comparison use the existing Plan 11 CUDA env (TRL 1.7.1, AgentTune 1.1.3, torch 2.11.0+cu128) | No new package |

## File Change Summary

| File | Action |
| --- | --- |
| `actproto/training/grpo_run.py` | modify |
| `actproto/training/rl_rollouts.py` | modify |
| `actproto/training/rewards_v2.py` | modify |
| `actproto/training/rl_config.py` | modify |
| `actproto/configs/grpo_v2_gpu_e2.yaml` | modify |
| `tests/test_train_grpo.py` | modify |
| `tests/test_rewards_v2.py` | modify |
| `tests/test_rl_rollouts.py` | modify |
| `tests/test_grpo_config.py` | modify |
| `docs/REWARD_SPEC.md` | modify |
| `docs/RL_BACKEND_COMPATIBILITY.md` | modify |

If T4 adds `actproto/configs/grpo_v2_gpu_tools.yaml` instead of editing `grpo_v2_gpu_e2.yaml`, that new file replaces the config row and the e2 GPU config stays the record of the 20-step run. Do not delete `grpo_v2_gpu_e2.yaml`.

## Open Questions

- The first tool-reward run may still log reward_std 0 because two samples from e2 agree. T4 allows `num_generations` 4 only after the 5060 Ti memory note says it fits. If it does not fit, stop. Do not move the job to the 4060 and do not add a length bonus.
- A later plan can try beta 0.001 with a frozen copy of e2 loaded as a PEFT adapter named `ref`, and only after this plan has shown a nonzero gradient from the tool reward and a memory log for the extra forward. This plan does not run that probe. Disabling the adapter is not an acceptable reference.
- The hard decision cap and the per-episode deadline are not measured yet. Start from the smoke's two-decision fixture only as a lower bound. The train cap has to cover the train tasks' real call depth and must count truncation as an invalid attempt. Do not raise `max_completion_length` past 256 to chase a length effect.
