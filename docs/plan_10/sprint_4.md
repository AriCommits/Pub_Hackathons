# Sprint 4 — Learning-mechanics and GPU smoke gates

**Plan:** docs/plan_10/OVERVIEW.md
**Wave:** 4 of 7
**Can run in parallel with:** none across waves; agents inside this wave may work in parallel.
**Must complete before:** Sprint 5

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

### Agent A: R4 — Learning-mechanics and end-to-end GPU gates

**Complexity:** L
**Estimated time:** 4–6 engineering hours plus bounded smoke runtime (rough estimate).
**Files to modify:**

- `scripts/verify_rl_training.py` — NEW; owned by R4.
- `tests/test_rl_learning.py` — NEW; owned by R4.
- `tests/test_rl_end_to_end.py` — NEW; owned by R4.
- `docs/RL_SMOKE_REPORT.md` — NEW; owned by R4.

**Depends on:** R3
**Blocks:** R5

**Instructions:**

Run staged gates: CPU contract/reward/rollout tests, differentiable tiny-policy test, installed-trainer integration, Tiny Aya two-update smoke, save/reload and resume. In a controlled two-action bandit with fixed distinct rewards and seeded sampling, verify GRPO's computed gradient increases the higher-reward action probability over bounded updates. This isolates optimizer direction; it is not a claim that every stochastic Tiny Aya run must improve after two updates.

Capture tensor hashes, nonzero finite gradients and actual parameter deltas; a changed file timestamp is insufficient. Prove tool-observation tokens carry zero policy loss, training consumes genuine executed rollouts, reference weights stay fixed, and group-to-task reward mapping remains correct after batching. Check initial policy matches SFT, reload matches the saved policy, and resume preserves RNG/step state within documented precision. Compare RL verifier versus AuditKIT on the same episodes.

Tiny Aya smoke must execute at least one actual work-tool call and a later observation-conditioned action. If two chosen train prompts yield no tool execution, diagnose formatting/collector behavior before a long run; don't inject gold calls to pass. Measure memory by stage and throughput. For all-equal rewards, log zero advantage; never force random reward differences. Overfit a tiny mixed-success train subset as a diagnostic only when time permits.

**Definition of Done:**

- [ ] All mechanics assertions and actual tool execution pass; memory headroom is measured; smoke report records exact commands, versions, deltas, masked spans and limitations. Failure blocks R5. No large training run is launched to discover an already-failing basic invariant.
- [ ] Focused behavior/invariant tests pass for code changes; document-only work uses file/link/command validation.
- [ ] Targeted tests and the existing full suite pass from the project root; exact commands/failures recorded.
- [ ] Executed experiments have actual logs, artifact hashes and terminal completion; planned commands do not count.
- [ ] Commit only owned files; keep weights, private artifacts, caches and `.agents/` out of Git.

---

## Integration and what this wave unblocks

Follow `docs/plan_10/PARALLEL-WORK.md`. The integrator reviews API compatibility and file ownership, merges completed task commits, and runs the full suite. This unblocks Sprint 5.
