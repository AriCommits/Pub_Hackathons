# Sprint 2 — Rollouts and rewards in parallel

**Plan:** docs/plan_10/OVERVIEW.md
**Wave:** 2 of 7
**Can run in parallel with:** none across waves; agents inside this wave may work in parallel.
**Must complete before:** Sprint 3

---

## Preconditions and shared execution contract

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

Integrate and verify all prior waves of Plan 10 before starting this wave. External dependencies below must exist. Work in an isolated branch/worktree; modify only owned files. All new outputs go to `results/rl_v2/`. Preserve old artifacts. One GPU-owning job across both plans. Implementation-only tasks do not launch long experiments.

## Agents in This Wave

### Agent A: R1 — Trusted grouped rollout collection

**Complexity:** L
**Estimated time:** 5–8 engineering hours (rough estimate).
**Files to modify:**

- `actproto/training/rl_rollouts.py` — NEW; owned by R1.
- `actproto/training/rl_dataset.py` — NEW; owned by R1.
- `actproto/training/rl_tools.py` — NEW; owned by R1.
- `scripts/collect_rl_rollouts.py` — NEW; owned by R1.
- `tests/test_rl_rollouts.py` — NEW; owned by R1.
- `tests/test_rl_dataset.py` — NEW; owned by R1.

**Depends on:** R0
**Blocks:** R3

**Instructions:**

Adapt the backend rollout to Plan 9 TaskSpec/PolicyInput/ExecutionTrace and versioned tool environment. Each group shares a public task but has isolated harness state and its own stochastic seed. Execute actual generated calls, append actual observations and continue from the current policy. Persist run/task/group/rollout IDs, policy/checkpoint hash, seeds, public messages, token IDs/spans, attention/completion masks, sampled-action logprobs where supported, trusted outputs, errors and termination. Reward-side private metadata stays outside generation inputs. Enforce turn/token/time limits, finish handling and expected tool failures.

Router, heuristic argument filling, answer repair and gold-call substitution remain disabled in primary RL. Trace malformed attempts explicitly; do not discard low-reward failures. Stable task IDs are collector metadata only. Collect-only is useful diagnostics but never counts as training. Do not refresh group samples during likelihood recomputation; preserve exact sampled tokens and masks. Avoid stale cached rollouts across optimizer updates unless the backend's correction explicitly supports them.

**Definition of Done:**

- [ ] Deterministic fake-policy tests prove independent group state, feedback-driven second action, no oracle leakage, no cross-task reward association, budget termination and failure retention. Injected fake tool messages are not trusted. Actual backend collect-only smoke produces executed tool records in the v2 schema. Resume IDs do not duplicate completed rollouts.
- [ ] Focused behavior/invariant tests pass for code changes; document-only work uses file/link/command validation.
- [ ] Targeted tests and the existing full suite pass from the project root; exact commands/failures recorded.
- [ ] Executed experiments have actual logs, artifact hashes and terminal completion; planned commands do not count.
- [ ] Commit only owned files; keep weights, private artifacts, caches and `.agents/` out of Git.

---

### Agent B: R2 — Verifiable rewards and reward-hacking tests

**Complexity:** L
**Estimated time:** 4–7 engineering hours (rough estimate).
**Files to modify:**

- `actproto/training/rewards_v2.py` — NEW; owned by R2.
- `actproto/configs/reward_v2.yaml` — NEW; owned by R2.
- `tests/test_rewards_v2.py` — NEW; owned by R2.
- `tests/test_reward_attacks.py` — NEW; owned by R2.
- `docs/REWARD_SPEC.md` — NEW; owned by R2.

**Depends on:** R0
**Blocks:** R3

**Instructions:**

Primary RL baseline uses sparse `R = 1.0 * verified_success_v2`, computed from the shared verifier and trusted execution. No reward for emitting schema-shaped text or declaring success. Keep reward calculation separate from token/logprob mechanics and emit a component ledger per rollout. Expected recoverable errors are scored according to task contract, not blanket punished.

Optional secondary shaped reward: `R = S - .10*min(I,3) - .05*min(D,3) - .05*S*min(E,4)`, where S is strict verified success, I counts invalid/noncompliant attempts, D counts identical state-aware duplicate calls, E is verified excess calls beyond the declared valid bound. Maximum combined penalty is .65, so every valid success outranks a failure under this bounded formula. If E is undefined, omit that component and log why. This reward does not give dense credit for unverified partial work. No format-only reward that could swamp task success. Keep completion/step penalties success-conditioned so early failure cannot win by being short. Add partial progress only in a separately versioned experiment with its own adversarial tests, not in the initial baseline.

Test exploit matrix: known-answer finish-only, copied task ID, hallucinated observation, wrong tool/right answer, legal alternate solution, empty finish, invalid JSON, repeated changed-argument calls, true identical loops, tool error, truncated output, duplicated finish, no-tool task, same answer across tasks, task-state tampering. Oracles never query model-written logs for proof of execution.

**Definition of Done:**

- [ ] Reward ledger equals verifier outputs; bounded-ranking property holds under generated extreme traces; failure shortcuts cannot receive success credit; legitimate no-tool/recovery tasks are not penalized unfairly. Tests verify success is not inferred from arbitrary nonempty strings or aggregate average metrics. Identical-reward groups yield finite zero advantage and diagnostic counters, not NaNs.
- [ ] Focused behavior/invariant tests pass for code changes; document-only work uses file/link/command validation.
- [ ] Targeted tests and the existing full suite pass from the project root; exact commands/failures recorded.
- [ ] Executed experiments have actual logs, artifact hashes and terminal completion; planned commands do not count.
- [ ] Commit only owned files; keep weights, private artifacts, caches and `.agents/` out of Git.

---

## Integration and what this wave unblocks

Follow `docs/plan_10/PARALLEL-WORK.md`. The integrator reviews API compatibility and file ownership, merges completed task commits, and runs the full suite. This unblocks Sprint 3.
