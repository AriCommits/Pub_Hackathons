# Plan 10: Real tool-rollout RL baseline, correctness tests, and matched evaluation

## Summary

Implement an actual AgentTune-backed GRPO training path initialized from Plan 9's frozen SFT checkpoint. Collect model-generated multi-turn tool rollouts, score trusted executions, optimize only policy tokens, and save/reload a changed adapter. Validate the learning mechanics and reward defenses before a bounded GPU study. Compare RL against the same SFT checkpoint and a continued-SFT compute control using the same frozen tools, tasks, verifier, decoding, and budgets; report held-out performance and compute costs without assuming RL wins.

## Goals

- Replace the successful no-op GRPO CLI with explicit dry-run, collect-only, smoke, train and resume modes that fail loudly when their required actions do not happen.
- Use a real AgentTune training factory and actual AuditKIT evaluation, with tested revision locks and artifact provenance.
- Sample independent tool-executing trajectories from the current policy, with no reference answers in model context and no prototype/heuristic actions in the primary RL path.
- Test reward correctness, tool provenance, policy-token masks, reference-policy identity, adapter updates, save/reload, and resume before interpreting reward curves.
- Measure validation-selected RL versus frozen SFT and continued SFT on the jointly sealed test, with success, tool compliance, efficiency, latency, and uncertainty.
- Produce reproducible local artifacts and a truthful submission narrative for Track 3 reliable tool calling.

## Non-Goals

- No PPO/DPO/RLOO algorithm sweep, new teacher model, model-size change, full-model finetuning, all-hop learned router, or safety claims beyond tested reward attacks.
- No reward from model-declared success, oracle-shaped answer prompts, fake tool outputs, or imported but uncalled trainers.
- No backpropagation through tool outputs, dataset regeneration after seeing test failures, or changing verifier thresholds to improve the headline score.
- No distributed/multi-GPU or vLLM dependency in the initial Windows/16-GB baseline. No automatic HF publishing.

## Background / Context

Depends on Plan 9 (`docs/plan_9/OVERVIEW.md`) and its immutable `docs/SFT_BASELINE_MANIFEST.json`. Source tree's current `scripts/train_grpo.py` accepts empty data, prints an import check, and never trains. `actproto/training/rewards.py` supplies disconnected reward helpers. Existing SFT is a valuable initialization/control, not RL. Existing 104/104 system result is not a standalone SFT baseline.

The current [AgentTune repository](https://github.com/Lexsi-Labs/AgentTune) exposes a GRPO factory with tool-rollout support and a Tiny Aya example. Use it as the intended backend, then prove that the pinned version performs the required behavior. Do not copy the example's superficial text-based reward. [TRL GRPO documentation](https://huggingface.co/docs/trl/grpo_trainer) describes online grouped generation, token accounting and template constraints. These are references, not proof that today's installed stack supports our format. [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT) supplies recorded-episode evaluation through Plan 9's tested bridge.

Use a separate `.venv-rl` or documented isolated Linux/WSL environment if compatibility requires it; keep the existing SFT environment working. A Linux move is a contingency after a concrete compatibility failure, not a silent assumption. Record all backend choices and platform tests. Observed training card: RTX 5060 Ti ~16 GB; discover UUID/device mapping again. Never launch concurrent model-bearing jobs. Read the long-running-background-tasks skill before launching or supervising any run. This plan authoring does not install packages or start training.

### Mandatory cross-plan gates

Interface exploration R0 can begin after Plan 9 S0/S6 contracts and locks exist. Rollout/reward implementation can use toy fixtures after those interfaces stabilize. No real optimization until Plan 9 S7 has frozen SFT selection, test hashes, model/template/verifier hashes and baseline provenance. Plan 9 S8 documentation must be integrated before the final RL report. Any change to shared scoring/data semantics requires a versioned amendment and rerunning both controls, not a favorable one-sided comparison.

Primary evaluation uses Plan 9's deterministic router-off, semantic-fallback-off mode and identical task budgets. RL begins from the selected SFT checkpoint; the KL reference, if used, is that same SFT policy, not the unadapted base. Optional routing is a post-training system ablation only. Native-model tools, format conversion and observation serialization must be equivalent across SFT and RL evaluation.

### Initial optimization and budget choices

GRPO only. Smoke: two prompts, two stochastic rollouts each, one or two optimizer updates. Pilot: 32 train tasks stratified by family, 20 updates, validation on a fixed 40-task validation subset. Candidate full run: up to 200 updates, evaluate full validation every 25 updates, save resumable checkpoints at each evaluation, early-stop after three non-improving evaluations. Start learning rate 5e-6, LoRA capacity inherited from SFT, group size 2 (4 only if measured memory allows), temperature .8, top-p .95, gradient checkpointing, 256 generated tokens per call. Validate installed-trainer batch/group divisibility; do not assume `batch=1, group=2` is accepted. Use sequential generation if supported and document its batch semantics. Bound total tokens per episode and label truncation; do not silently reduce evaluation budgets to fit training.

These are initial config hypotheses, adjusted only on train/validation pilots before final run freeze. Pilot may compare 2e-6 and 5e-6; at most two full seed-42 candidates including the selected pilot settings. A nominal KL coefficient .02 is allowed only with a proven frozen SFT reference implementation. If no such reference fits memory, explicitly run/report beta=0 and use a validation drift gate; never label base-only reference as SFT reference. Record sampling logprobs, likelihood recomputation and behavior-policy age; no unrestricted stale rollout replay.

Budget: at most 36 GPU-hours for RL study plus controls, selected after a ten-minute throughput/memory pilot; do not claim a duration before measuring. Reserve final evaluation time before spending the training budget. Minimum experiment is one real seed-42 RL run and both controls; seeds 43/44 and shaped-reward ablation are optional. Three seed runs are required for seed-stability claims; otherwise label findings exploratory. Cut optional experiments rather than verifier checks, save/reload validation, or the fair comparison.

## Features / Tasks

### R0: Pin real AgentTune integration and prove API feasibility
**Files:** `actproto/integrations/agenttune_backend.py` (new), `requirements-rl.lock` (new), `scripts/check_rl_backend.py` (new), `tests/test_agenttune_backend.py` (new), `docs/RL_BACKEND_COMPATIBILITY.md` (new)
**Complexity:** L
**Depends on:** external S0, external S6
**Estimate:** 4–7 engineering hours

Inspect pinned AgentTune source and signatures for trainer construction, model/PEFT injection, tool callbacks, rollout output, reward arguments, token masking, logging, checkpoint resume and reference-policy handling. Record supported versions and actual import paths, then pin the resolved environment. Build a thin adapter with explicit capability checks; do not reimplement GRPO under the guise of integration. Reject unsupported options rather than silently dropping kwargs. AuditKIT reuse comes from Plan 9.

Tiny Aya uses textual tool tags and tool results rewritten as user turns in the current harness. Establish one lossless serialization path: either preserve that format through AgentTune's parser hooks or explicitly migrate both training and evaluation to a tested canonical template. Verify prefix-preserving token rendering when observations append, including a one-tool-followed-by-finish episode. Tool responses must be marked non-policy tokens. Confirm how `finish` terminates and that observations—not the full reference trajectory—are fed back.

**Acceptance:** Installed package, not a mock, creates the intended trainer on a tiny supported model/fixture; signatures and effective kwargs are recorded. Fake tool loop demonstrates two decisions and observation insertion. Tiny Aya real-tokenizer probe verifies serialization/masks without a full train. Unsupported platform/version/template returns actionable nonzero failure. No import-only success.

### R1: Trusted grouped rollout collection
**Files:** `actproto/training/rl_rollouts.py` (new), `actproto/training/rl_dataset.py` (new), `actproto/training/rl_tools.py` (new), `scripts/collect_rl_rollouts.py` (new), `tests/test_rl_rollouts.py` (new), `tests/test_rl_dataset.py` (new)
**Complexity:** L
**Depends on:** R0
**Estimate:** 5–8 engineering hours

Adapt the backend rollout to Plan 9 TaskSpec/PolicyInput/ExecutionTrace and versioned tool environment. Each group shares a public task but has isolated harness state and its own stochastic seed. Execute actual generated calls, append actual observations and continue from the current policy. Persist run/task/group/rollout IDs, policy/checkpoint hash, seeds, public messages, token IDs/spans, attention/completion masks, sampled-action logprobs where supported, trusted outputs, errors and termination. Reward-side private metadata stays outside generation inputs. Enforce turn/token/time limits, finish handling and expected tool failures.

Router, heuristic argument filling, answer repair and gold-call substitution remain disabled in primary RL. Trace malformed attempts explicitly; do not discard low-reward failures. Stable task IDs are collector metadata only. Collect-only is useful diagnostics but never counts as training. Do not refresh group samples during likelihood recomputation; preserve exact sampled tokens and masks. Avoid stale cached rollouts across optimizer updates unless the backend's correction explicitly supports them.

**Acceptance:** Deterministic fake-policy tests prove independent group state, feedback-driven second action, no oracle leakage, no cross-task reward association, budget termination and failure retention. Injected fake tool messages are not trusted. Actual backend collect-only smoke produces executed tool records in the v2 schema. Resume IDs do not duplicate completed rollouts.

### R2: Verifiable rewards and reward-hacking tests
**Files:** `actproto/training/rewards_v2.py` (new), `actproto/configs/reward_v2.yaml` (new), `tests/test_rewards_v2.py` (new), `tests/test_reward_attacks.py` (new), `docs/REWARD_SPEC.md` (new)
**Complexity:** L
**Depends on:** R0
**Estimate:** 4–7 engineering hours

Primary RL baseline uses sparse `R = 1.0 * verified_success_v2`, computed from the shared verifier and trusted execution. No reward for emitting schema-shaped text or declaring success. Keep reward calculation separate from token/logprob mechanics and emit a component ledger per rollout. Expected recoverable errors are scored according to task contract, not blanket punished.

Optional secondary shaped reward: `R = S - .10*min(I,3) - .05*min(D,3) - .05*S*min(E,4)`, where S is strict verified success, I counts invalid/noncompliant attempts, D counts identical state-aware duplicate calls, E is verified excess calls beyond the declared valid bound. Maximum combined penalty is .65, so every valid success outranks a failure under this bounded formula. If E is undefined, omit that component and log why. This reward does not give dense credit for unverified partial work. No format-only reward that could swamp task success. Keep completion/step penalties success-conditioned so early failure cannot win by being short. Add partial progress only in a separately versioned experiment with its own adversarial tests, not in the initial baseline.

Test exploit matrix: known-answer finish-only, copied task ID, hallucinated observation, wrong tool/right answer, legal alternate solution, empty finish, invalid JSON, repeated changed-argument calls, true identical loops, tool error, truncated output, duplicated finish, no-tool task, same answer across tasks, task-state tampering. Oracles never query model-written logs for proof of execution.

**Acceptance:** Reward ledger equals verifier outputs; bounded-ranking property holds under generated extreme traces; failure shortcuts cannot receive success credit; legitimate no-tool/recovery tasks are not penalized unfairly. Tests verify success is not inferred from arbitrary nonempty strings or aggregate average metrics. Identical-reward groups yield finite zero advantage and diagnostic counters, not NaNs.

### R3: Real GRPO optimizer, reference policy, and lifecycle
**Files:** `actproto/training/grpo.py` (modify), `actproto/training/grpo_run.py` (new), `actproto/training/rl_config.py` (new), `actproto/configs/grpo_v2.yaml` (new), `scripts/train_grpo.py` (modify), `tests/test_grpo_config.py` (new), `tests/test_train_grpo.py` (new)
**Complexity:** L
**Depends on:** R1, R2, external S7
**Estimate:** 6–10 engineering hours

Replace metadata-only execution with actual backend factory construction, train_dataset, callable rewards, registered tools and `trainer.train()`. Support `--plan-only`, `--collect-only`, `--smoke`, `--resume`, explicit config, frozen baseline manifest and immutable output directory. Real mode must run updates, save trainer/optimizer/scheduler/RNG state, trained adapter and effective config, then verify artifacts. Empty data, missing backend, absent trainable parameters or no completed update is failure. Preserve old stubs' documented limitations in migration notes, not silent success behavior.

Load the selected SFT adapter trainably; verify initial logits against the frozen SFT control. If the backend requires a merged base to preserve a frozen SFT reference while training a new LoRA, create a separately hashed derivative and show pre-update parity—do not overwrite original weights. Reference stays frozen and excluded from optimizer; establish what is reference for PEFT, not merely disabling the current adapter. Start with no vLLM and one GPU. Do not claim LoRA implies full GRPO fits 16 GB: measure generation/likelihood/reference peaks. Quantization, CPU reference/offload or beta=0 are explicit named configurations, not automatic silent fallback.

Backpropagate through only newly generated assistant action/finish tokens. No prompt, gold, padding or tool-result gradients; all turns in an episode receive correctly associated trajectory advantage. Validate stochastic sampling versus likelihood logprob accounting and deterministic checkpoint resume. Log policy loss, optional KL, entropy proxy, reward variance, zero-variance-group fraction, clipping, gradient norm, learning rate, generated tokens, tool failures, memory and wall time. Validate group/batch divisibility against pinned TRL. Validation uses shared runner, no training updates and no test access.

**Acceptance:** Tests prove real invocation, nonempty train set, frozen-reference identity, optimizer parameter membership, bounded config and nonzero failure exit paths. Tiny backend smoke changes only permitted weights with finite gradients and reloads. Resume continues update counters without replacing earlier outputs. `--plan-only` does no optimization and cannot create a completed-run marker.

### R4: Learning-mechanics and end-to-end GPU gates
**Files:** `scripts/verify_rl_training.py` (new), `tests/test_rl_learning.py` (new), `tests/test_rl_end_to_end.py` (new), `docs/RL_SMOKE_REPORT.md` (new)
**Complexity:** L
**Depends on:** R3
**Estimate:** 4–6 engineering hours plus bounded smoke runtime

Run staged gates: CPU contract/reward/rollout tests, differentiable tiny-policy test, installed-trainer integration, Tiny Aya two-update smoke, save/reload and resume. In a controlled two-action bandit with fixed distinct rewards and seeded sampling, verify GRPO's computed gradient increases the higher-reward action probability over bounded updates. This isolates optimizer direction; it is not a claim that every stochastic Tiny Aya run must improve after two updates.

Capture tensor hashes, nonzero finite gradients and actual parameter deltas; a changed file timestamp is insufficient. Prove tool-observation tokens carry zero policy loss, training consumes genuine executed rollouts, reference weights stay fixed, and group-to-task reward mapping remains correct after batching. Check initial policy matches SFT, reload matches the saved policy, and resume preserves RNG/step state within documented precision. Compare RL verifier versus AuditKIT on the same episodes.

Tiny Aya smoke must execute at least one actual work-tool call and a later observation-conditioned action. If two chosen train prompts yield no tool execution, diagnose formatting/collector behavior before a long run; don't inject gold calls to pass. Measure memory by stage and throughput. For all-equal rewards, log zero advantage; never force random reward differences. Overfit a tiny mixed-success train subset as a diagnostic only when time permits.

**Acceptance:** All mechanics assertions and actual tool execution pass; memory headroom is measured; smoke report records exact commands, versions, deltas, masked spans and limitations. Failure blocks R5. No large training run is launched to discover an already-failing basic invariant.

### R5: Bounded RL pilot and validation-only selection
**Files:** `scripts/run_rl_study.py` (new), `actproto/configs/rl_study.yaml` (new), `docs/RL_TRAINING_RESULTS.md` (new), `docs/RL_SELECTED_MANIFEST.json` (new)
**Complexity:** L
**Depends on:** R4
**Estimate:** 3–5 engineering hours plus measured GPU runtime

Execute the initial optimization schedule and time cap above with run manifests and one GPU lease. Primary reward is sparse verified success. Pilot logs distribution of valid successes, reward variance and truncation by family. If groups have zero variance, diagnose task difficulty using train only and use fixed train-stratified sampling/temperature pilot; do not secretly add shaping or mine final-test errors. Freeze chosen settings after at most two pilot learning rates. Allow at most two full candidates, select checkpoints by validation macro verified success with Plan 9 tie-breaks and regression gate. Do not select by training reward.

Evaluate full validation every 25 updates, save/stop on finite-state failures or the configured patience, preserve every failed candidate. Monitor PID/log heartbeat, loss/reward components, NaNs, KL drift, repeated calls, memory and token budget until terminal status. An interrupted run is resumable or failed, never completed. Repeat selected config on independent seeds only within budget; optional shaped reward gets a separate run ID and same budgets. Record what was omitted.

**Acceptance:** At least one real trained and reloadable RL candidate with provenance and full validation traces exists, even if worse than SFT. Selection manifest freezes checkpoint, reward version, config, final-test hashes and remaining final-eval budget. SFT baseline and verifier unchanged. No final-test performance used to choose settings or checkpoint.

### R6: Matched controls and sealed final evaluation
**Files:** `actproto/configs/rl_eval_matrix.yaml` (new), `scripts/evaluate_rl_study.py` (new), `tests/test_rl_comparison.py` (new), `docs/RL_EVALUATION.md` (new)
**Complexity:** L
**Depends on:** R5, external S8
**Estimate:** 4–6 engineering hours plus measured control/evaluation runtime

Before unlocking final test, freeze all candidates and the matrix: frozen SFT router-off; continued SFT on the same train task pool; selected RL router-off; optional SFT+router and RL+router with respective train-fitted/validation-calibrated banks. The primary RL effect is RL minus frozen SFT with all other settings matched. Continued SFT controls for extra training exposure/compute: target the RL run's wall-clock GPU budget within 10%, stop by budget not test results, and disclose token/step/sample differences. Select its checkpoint on validation using the same criterion. If hardware prevents a credible match, report it as an approximate control with measured mismatch, not compute-equivalent.

Run the 400-task sealed set once per frozen candidate, plus separately labeled old-development and long-chain diagnostics. Never train or calibrate using final trajectories. If SFT final test was previously exposed under the Plan 9 standalone exception, enforce its documented reuse restrictions and clearly state which set remained unseen. Report micro/macro verified success, answer success, tool compliance, first-tool and semantic argument correctness, invalid executions, loops, success-conditioned and unconditional steps, all model invocations/tokens, p50/p95 latency and GPU training cost. Failed episodes remain in denominators; do not drop timeouts.

Use paired group-clustered bootstrap 95% confidence intervals for differences, per-family slices, and per-seed results when available. Distinguish independent training seeds from bootstrap sampling uncertainty. Preregister claim gates: improvement if strict-success delta is positive with lower CI above zero; efficiency if 2-point success noninferiority is supported and steps/tokens fall. If CIs are wide or regressions exceed the predeclared gate, state inconclusive/negative result. Broadly training RL is complete even without a win.

**Acceptance:** Every compared row has identical manifest/verifier/decoding/task budget and known intervention settings; comparison code rejects mismatches and missing cases. Control compute and final access are logged. Final test remains untouched by optimization. No generalization claim from the old 38/266 seen curricula. Complete report includes negative and uncertain findings.

### R7: Reproducible RL artifacts and submission narrative
**Files:** `docs/RL_REPRODUCTION.md` (new), `docs/RL_MODEL_CARD.md` (new), `README.md` (modify), `docs/METHOD.md` (modify), `docs/SUBMISSION.md` (modify), `docs/EXPERIMENT_STATUS.md` (modify)
**Complexity:** M
**Depends on:** R6
**Estimate:** 2–4 engineering hours

Document actual AgentTune→executed-rollouts→reward→optimizer→AuditKIT pipeline and tested commands. Distinguish base, legacy SFT, refined SFT, continued SFT, RL, and optional routed systems. State synthetic-tool scope, exact holdouts, task coverage, reward limitations, seed counts and compute. Export via Plan 9's manifest bundle, including initialized SFT checkpoint identity, RL adapter, tokenizer/template, data, fixtures, reward config, sampled traces, trainer state or reproducible resume instructions, all evaluation results and environment locks. Include a small verifiable before/after tool trace and a failed example.

Prepare public model/data collection and article checklist with local bundle validation. Publishing requires the user's explicit upload instruction; planning and local packaging need no separate approval. Never present a successful stub, smoke-only run or reward curve as measured task improvement. Update current status while retaining historical records.

**Acceptance:** Clean-environment import/CLI checks pass, trained adapter reloads, local bundle hashes validate, reproduction describes hardware/runtime, and every headline metric links to a saved report with its baseline. Submission text accurately states whether RL improved strict held-out performance.

## New Dependencies

| Package | Feature | Optional? |
| --- | --- | --- |
| AgentTune at a verified commit | Actual GRPO tool-rollout training | Required for this intended backend; no silent shim fallback |
| Tested torch/transformers/TRL/PEFT/datasets set | Training and adapter lifecycle | Required; resolve and pin together in isolated environment |
| AuditKIT from Plan 9 lock | Real recorded evaluation | Required for submission mode |
| bitsandbytes | Optional memory-reduction experiment | Optional, capability-tested |
| vLLM / distributed training | None in primary baseline | Not required |

Do not upgrade the working SFT environment in place. Backend runtime configuration must be included in comparisons when dtype or generation engine differs.

## File Change Summary

| File | Action | Owner |
| --- | --- | --- |
| `actproto/integrations/agenttune_backend.py` | new | R0 |
| `requirements-rl.lock` | new | R0 |
| `scripts/check_rl_backend.py` | new | R0 |
| `tests/test_agenttune_backend.py` | new | R0 |
| `docs/RL_BACKEND_COMPATIBILITY.md` | new | R0 |
| `actproto/training/rl_rollouts.py` | new | R1 |
| `actproto/training/rl_dataset.py` | new | R1 |
| `actproto/training/rl_tools.py` | new | R1 |
| `scripts/collect_rl_rollouts.py` | new | R1 |
| `tests/test_rl_rollouts.py` | new | R1 |
| `tests/test_rl_dataset.py` | new | R1 |
| `actproto/training/rewards_v2.py` | new | R2 |
| `actproto/configs/reward_v2.yaml` | new | R2 |
| `tests/test_rewards_v2.py` | new | R2 |
| `tests/test_reward_attacks.py` | new | R2 |
| `docs/REWARD_SPEC.md` | new | R2 |
| `actproto/training/grpo.py` | modify | R3 |
| `actproto/training/grpo_run.py` | new | R3 |
| `actproto/training/rl_config.py` | new | R3 |
| `actproto/configs/grpo_v2.yaml` | new | R3 |
| `scripts/train_grpo.py` | modify | R3 |
| `tests/test_grpo_config.py` | new | R3 |
| `tests/test_train_grpo.py` | new | R3 |
| `scripts/verify_rl_training.py` | new | R4 |
| `tests/test_rl_learning.py` | new | R4 |
| `tests/test_rl_end_to_end.py` | new | R4 |
| `docs/RL_SMOKE_REPORT.md` | new | R4 |
| `scripts/run_rl_study.py` | new | R5 |
| `actproto/configs/rl_study.yaml` | new | R5 |
| `docs/RL_TRAINING_RESULTS.md` | new | R5 |
| `docs/RL_SELECTED_MANIFEST.json` | new | R5 |
| `actproto/configs/rl_eval_matrix.yaml` | new | R6 |
| `scripts/evaluate_rl_study.py` | new | R6 |
| `tests/test_rl_comparison.py` | new | R6 |
| `docs/RL_EVALUATION.md` | new | R6 |
| `docs/RL_REPRODUCTION.md` | new | R7 |
| `docs/RL_MODEL_CARD.md` | new | R7 |
| `README.md` | modify | R7 |
| `docs/METHOD.md` | modify | R7 |
| `docs/SUBMISSION.md` | modify | R7 |
| `docs/EXPERIMENT_STATUS.md` | modify | R7 |

## Open Questions

- Whether AgentTune exposes lossless Tiny Aya trajectories with correct per-turn masks is a hard R0/R4 gate, not an assumption. Fix a small tested adapter or pin a supported version; do not claim RL completion if unavailable.
- Whether a frozen SFT reference fits beside trainable Tiny Aya on 16 GB is unknown. Profile reference adapter switching/merged SFT derivative first; document beta=0 if needed. A second fully resident 3.35B model is not the default.
- Sparse GRPO may have zero-variance groups when tasks are too easy or too hard. Diagnose on train; a reward or curriculum change is a new versioned experiment. Increasing epochs blindly is not the response.
- Full multi-seed runs may exceed the hackathon budget. Minimum honest result is a tested single-seed run with uncertainty and disclosed limits; prioritize final comparison over an algorithm sweep.
- If Plan 9 yields no SFT improvement, use its best validated checkpoint anyway. If RL performs worse, retain SFT for deployment and report the measured negative result.

## Execution priority and completion

Critical path is R0 → (R1 and R2) → R3 → R4 → R5 → R6 → R7, gated by Plan 9's frozen checkpoint. No optimization without mechanics tests. Optional items to cut first are extra seeds, shaped reward and router ablations. Completion requires trained weights, actual tool-rollout evidence, strict held-out comparison, reloadable artifacts and documentation—not merely code that can theoretically launch training.
