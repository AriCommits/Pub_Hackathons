# Supervised refinement and RL comparison roadmap

Prepared 2026-11-04 from the project audit and the user's decision to preserve supervised learning as a baseline. These are implementation plans, not claims of completed experiments. No training, dependency installation, or publishing was performed while writing them.

| Plan | Design | Coordination | Starting brief |
| --- | --- | --- | --- |
| 9: Refine and freeze SFT | [OVERVIEW](plan_9/OVERVIEW.md) | [PARALLEL-WORK](plan_9/PARALLEL-WORK.md) | [Sprint 1](plan_9/sprint_1.md) |
| 10: Implement and evaluate RL | [OVERVIEW](plan_10/OVERVIEW.md) | [PARALLEL-WORK](plan_10/PARALLEL-WORK.md) | [Sprint 1](plan_10/sprint_1.md) |

Plan 9 has nine tasks over six waves. Wave 2 parallelizes verification, data construction, and SFT implementation; wave 4 parallelizes router calibration and actual AuditKIT integration/export. Plan 10 has eight tasks over seven waves; wave 2 parallelizes rollout and reward implementation. GPU execution remains serial across both plans.

## What each plan establishes

Plan 9 tests whether corrected supervision, a balanced leakage-resistant curriculum, and bounded train-only failure repair improve the supervised policy. It separately measures whether prototype routing helps. The old 104-task benchmark remains a development/regression set, and historical adapters remain preserved. A base-model control and old assistant-only adapter are measured with the same new harness as the refined candidates.

Plan 10 initializes genuine rollout-based GRPO from the selected SFT checkpoint. It proves that real tools execute, observations feed subsequent decisions, rewards come from trusted execution, policy weights change, tool tokens are excluded from loss, and saved weights reload. Its final comparison includes frozen SFT, continued SFT as an approximate compute control, and RL under identical evaluation conditions.

## Gates between the plans

1. **Shared contract:** S0 fixes public/private data boundaries, tool environment, verifier versioning, budgets and immutable manifests.
2. **Trustworthy benchmark:** S1/S2/S4 reject tool-use shortcuts, prevent split leakage, and provide the common evaluation runner.
3. **Actual toolkit use:** S6 exercises real AuditKIT; R0 establishes actual AgentTune capability with pinned dependencies.
4. **Frozen SFT initialization:** S7 supplies `docs/SFT_BASELINE_MANIFEST.json` and reloadable adapter. No real RL optimization precedes this gate.
5. **Learning mechanics:** R4 proves masks, gradients, optimizer direction, tool feedback and checkpoint lifecycle before full RL training.
6. **Matched final evaluation:** R6 consumes the immutable candidate choices and sealed test. Final test failures cannot feed training or selection.

The final SFT test normally runs alongside RL in Plan 10. Plan 9 can ship alone using its documented one-time disclosure procedure, but then future work must distinguish exposed test data from genuinely unseen evaluation.

## Audit finding coverage

| Audit gap | Work that addresses it | Evidence required |
| --- | --- | --- |
| GRPO command never trains | R0, R3, R4 | Actual trainer invocation, generated rollouts, parameter deltas, reload |
| Seen curriculum evaluated as generalization | S0, S2, S4, R6 | Logical-group split audit, zero cross-split prompt overlap, sealed manifest |
| Finish-only reward shortcut | S1, R2 | Adversarial execution/provenance tests |
| Toolkit compatibility shims | S6, R0 | Installed-package execution and version locks |
| Missing standalone SFT baseline | S4, S7, R6 | Current-harness router-off results and matched ablations |
| Assistant/EOT masking and checkpoint selection | S3, S7 | Render/token-mask assertions and validation selection evidence |
| Threshold tie behavior and prototype mismatch | S5 | Strict-tie tests, adapter-bound bank hashes and risk/coverage report |
| Ambiguous header-format answers | S2 | Every layout specifies output convention and oracle replay passes |
| Misleading efficiency counts | S1, S4, R6 | State-aware duplicates, failures, attempts, tokens and valid bounds |
| Stale docs and incomplete exports | S6, S8, R7 | Manifest-driven bundle, complete reproduction commands and honest status |

## Scope discipline

The new modules named in the plans are proposed ownership boundaries, not a mandate to duplicate the whole codebase. Reuse existing parsing, deterministic tools, model loading, serialization and reporting where behavior is correct. Isolate versioned experiment semantics behind adapters so old results remain reproducible. Do not introduce parallel copies of a working generic function merely because a new file has a `v2` suffix. Shared fixes discovered during implementation must be coordinated by the integrator and assigned to one task before parallel edits.

Must-have work is benchmark integrity, correct SFT, real RL mechanics, primary controls, and final comparison. Optional scope is failure-repair candidate E3, repeated training seeds beyond the primary run, shaped rewards, and extra router ablations. Defer these before cutting measurement validity. Nominal ceilings are 24 SFT GPU-hours and 36 RL/control GPU-hours; these are study caps, not estimates. Throughput pilots determine feasibility and reserve evaluation time.

A scientifically valid negative result completes an experiment. Missing trained weights, missing provenance, contaminated evaluation, or a trainer stub do not. Neither plan promises that SFT refinements or RL will win.

## Suggested first action

Begin Plan 9 Sprint 1 (S0). Commit reviewed planning artifacts before branching, then use the coordination guide's worktree and integration commands. Do not execute Plan 10's optimization until the frozen SFT handoff exists. Current authoring changed documentation only.
