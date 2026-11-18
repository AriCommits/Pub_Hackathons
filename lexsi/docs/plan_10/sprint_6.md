# Sprint 6 — Matched controls and sealed evaluation

**Plan:** docs/plan_10/OVERVIEW.md
**Wave:** 6 of 7
**Can run in parallel with:** none across waves; agents inside this wave may work in parallel.
**Must complete before:** Sprint 7

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

### Agent A: R6 — Matched controls and sealed final evaluation

**Complexity:** L
**Estimated time:** 4–6 engineering hours plus measured control/evaluation runtime (rough estimate).
**Files to modify:**

- `actproto/configs/rl_eval_matrix.yaml` — NEW; owned by R6.
- `scripts/evaluate_rl_study.py` — NEW; owned by R6.
- `tests/test_rl_comparison.py` — NEW; owned by R6.
- `docs/RL_EVALUATION.md` — NEW; owned by R6.

**Depends on:** R5, external S8
**Blocks:** R7

**Instructions:**

Before unlocking final test, freeze all candidates and the matrix: frozen SFT router-off; continued SFT on the same train task pool; selected RL router-off; optional SFT+router and RL+router with respective train-fitted/validation-calibrated banks. The primary RL effect is RL minus frozen SFT with all other settings matched. Continued SFT controls for extra training exposure/compute: target the RL run's wall-clock GPU budget within 10%, stop by budget not test results, and disclose token/step/sample differences. Select its checkpoint on validation using the same criterion. If hardware prevents a credible match, report it as an approximate control with measured mismatch, not compute-equivalent.

Run the 400-task sealed set once per frozen candidate, plus separately labeled old-development and long-chain diagnostics. Never train or calibrate using final trajectories. If SFT final test was previously exposed under the Plan 9 standalone exception, enforce its documented reuse restrictions and clearly state which set remained unseen. Report micro/macro verified success, answer success, tool compliance, first-tool and semantic argument correctness, invalid executions, loops, success-conditioned and unconditional steps, all model invocations/tokens, p50/p95 latency and GPU training cost. Failed episodes remain in denominators; do not drop timeouts.

Use paired group-clustered bootstrap 95% confidence intervals for differences, per-family slices, and per-seed results when available. Distinguish independent training seeds from bootstrap sampling uncertainty. Preregister claim gates: improvement if strict-success delta is positive with lower CI above zero; efficiency if 2-point success noninferiority is supported and steps/tokens fall. If CIs are wide or regressions exceed the predeclared gate, state inconclusive/negative result. Broadly training RL is complete even without a win.

**Definition of Done:**

- [ ] Every compared row has identical manifest/verifier/decoding/task budget and known intervention settings; comparison code rejects mismatches and missing cases. Control compute and final access are logged. Final test remains untouched by optimization. No generalization claim from the old 38/266 seen curricula. Complete report includes negative and uncertain findings.
- [ ] Focused behavior/invariant tests pass for code changes; document-only work uses file/link/command validation.
- [ ] Targeted tests and the existing full suite pass from the project root; exact commands/failures recorded.
- [ ] Executed experiments have actual logs, artifact hashes and terminal completion; planned commands do not count.
- [ ] Commit only owned files; keep weights, private artifacts, caches and `.agents/` out of Git.

---

## Integration and what this wave unblocks

Follow `docs/plan_10/PARALLEL-WORK.md`. The integrator reviews API compatibility and file ownership, merges completed task commits, and runs the full suite. This unblocks Sprint 7.
