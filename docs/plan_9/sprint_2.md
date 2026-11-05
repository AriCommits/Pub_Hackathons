# Sprint 2 — Verification, data and SFT implementation in parallel

**Plan:** docs/plan_9/OVERVIEW.md
**Wave:** 2 of 6
**Can run in parallel with:** none across waves; agents inside this wave may work in parallel.
**Must complete before:** Sprint 3

---

## Preconditions and shared execution contract

Read `docs/AUDIT_2026-11-04.md` for evidence. Existing tests: 88 passing. Best historical system `B2apchain` scored 104/104 but includes first-hop routing and fallbacks; router-off performance for its assistant-only adapter is unknown. `B2orders` scores 103/104 and `B2fmt` 88/104 on the old development set. Curriculum evaluations reuse all 38/266 evaluation prompts in training. The existing verifier accepts a correct finish-only answer on a task requiring tools.

This plan supersedes unfinished Plan 1 improvements for future work. Retain historical plans as records. Adopt useful requirements (mask tests, validation checkpointing, strict thresholds), replace answer-snapping with declared semantic scoring, and defer step-wise routing until a matched first-hop ablation justifies it. Plans 3–8 remain provenance, not cumulative implementation instructions. Plan 2 is not present in this checkout.

Project root is `C:/Users/arian/Github/Hackathons/lexsi`; Git root is its parent `Hackathons`. Use project `.venv/Scripts/python.exe` for existing tests. GPU identities must be detected by UUID/name at execution time: observed PyTorch cuda:1 is RTX 5060 Ti (~16 GB), cuda:0 is RTX 4060 (8 GB). One model-bearing GPU job at a time; never assume memory will accommodate concurrent workers. Read the long-running-background-tasks skill before launching or monitoring experiments. Existing environment remains untouched while toolkit compatibility is tested in a separate environment. No training is launched by creating this plan.

### Shared experiment contract

Version `tooluse_v2` separates `TaskSpec` (private oracle, split, group, tool dependencies), `PolicyInput` (public prompt, tool schemas, observations only), and `ExecutionTrace` (harness-recorded calls/results, provenance, model spans, stopping reason). Gold answers, expected arguments, split labels, and case identifiers are never passed in policy messages. A hash-addressed manifest ties together data, tool fixtures, verifier, prompt template, decoding, model revision, adapter, and code revision.

Use two task modes: `outcome` allows semantically equivalent solutions; `required_tools` checks declared calls/dependencies from trusted execution in addition to the result. An explicit sequence instruction can require an exact order; an ordinary business question should not be constrained to one arbitrary reference sequence. Record `answer_success`, `execution_compliance`, and `verified_success_v2` separately. Preserve `task_success_v1` only for historical comparison. Parse failure is an event, never a synthetic successful finish.

Target data budget: 1,200 train, 200 validation, 400 final-test tasks, expandable only before freeze if unique executable groups cannot meet quotas. Use 4 semantic strata: single-tool 25%, dependent 2–4 calls 35%, ordered/subset compositions 25%, recoverable failures and no-tool tasks 15%; provide per-stratum counts. Allocate 200 final tasks to new-entity in-distribution evaluation and 200 to held-out composition/prompt-family stress. Keep all layouts/paraphrases/variants of a logical task together. Split entities and argument tuples; additionally hold out composition families for stress. Do not claim long-chain generalization if its examples occur in train.

The 400-task final test is sealed from model selection for both plans. Freeze SFT choice before any final evaluation. Default: perform the single matched final SFT+RL evaluation in Plan 10. If Plan 9 must ship alone, final-test SFT once, record that disclosure, and do not use its failures to tune RL; reserve a second immutable test version before further tuning. Developer/agent access cannot literally be prevented by a local hash: enforce split guards, access logging, and workflow discipline.

Default policy evaluation is deterministic, router off, semantic fallback off, at most 256 generated tokens per call, with task-class budgets fixed in the manifest: 4 total calls for short tasks, 8 for dependent medium tasks, 12 for long diagnostics; finish counts as a call. Reject gold tasks that cannot fit. Training samples only train tasks. All candidates see the same schema/prompt contract. Record emitted tokens, model invocations, executed tool calls, failures, latency, and truncation separately.

Performance decision: primary validation metric is macro-averaged `verified_success_v2` across strata, with micro success also reported. Target improvement is at least 5 percentage points over the legacy assistant-only checkpoint under the same harness, with no more than 2 points loss on single-tool/no-tool strata and no more than 1.5x median latency. These are engineering targets, not promised results. If unmet, retain the stronger candidate and publish a negative result. Use family/group-clustered bootstrap paired confidence intervals; the point-estimate gate alone is not statistical significance. For efficiency claims, require a predeclared 2-point noninferiority margin and report uncertainty, including success-conditioned and unconditional steps.

Integrate and verify all prior waves of Plan 9 before starting this wave. External dependencies below must exist. Work in an isolated branch/worktree; modify only owned files. All new outputs go to `results/sft_v2/`. Preserve old artifacts. One GPU-owning job across both plans. Implementation-only tasks do not launch long experiments.

## Agents in This Wave

### Agent A: S1 — Execution-grounded verification and trustworthy metrics

**Complexity:** L
**Estimated time:** 5–8 engineering hours (rough estimate).
**Files to modify:**

- `actproto/evaluation/verification_v2.py` — NEW; owned by S1.
- `actproto/evaluation/metrics_v2.py` — NEW; owned by S1.
- `actproto/evaluation/trace_adapter.py` — NEW; owned by S1.
- `tests/test_verification_v2.py` — NEW; owned by S1.
- `tests/test_metrics_v2.py` — NEW; owned by S1.

**Depends on:** S0
**Blocks:** S4

**Instructions:**

Implement a pure verifier over TaskSpec and trusted harness trace. Do not accept model-supplied tool JSON as executed observations. Verify tool names, schema and semantic values, actual tool success or task-declared expected error, finish timing, dependencies, and final outcome. Respect legitimate no-tool tasks and alternative valid trajectories. Numeric tolerance and output formatting belong to the task contract; IDs and sentinel strings remain exact where required. Do not use arbitrary substring acceptance. Keep historical scoring untouched.

Measure exact duplicate `(tool, canonical args, relevant state)` calls and cycles separately from useful repeated use. Count invalid parsing, failed execution, noncompliant calls, model invocations and generated tokens, not just tool names. Excess-call metrics require a valid minimum/reference bound and must be labeled unavailable when no sound bound exists. Capture task compliance, answer accuracy, first-tool and argument metrics without conflating them.

**Definition of Done:**

- [ ] Finish-only known answer fails required-tools tasks; forged output, wrong order, incorrect arguments, stale state, wrong numeric units, and truncation fail. Legal repeated database calls, alternative valid ordering, expected errors, and no-tool answers pass. Run adversarial verifier fixtures and cross-check oracle-generated traces; no historical result changes.
- [ ] Focused behavior/invariant tests pass for code changes; document-only work uses file/link/command validation.
- [ ] Targeted tests and the existing full suite pass from the project root; exact commands/failures recorded.
- [ ] Executed experiments have actual logs, artifact hashes and terminal completion; planned commands do not count.
- [ ] Commit only owned files; keep weights, private artifacts, caches and `.agents/` out of Git.

---

### Agent B: S2 — Leakage-resistant, balanced, fully specified curriculum

**Complexity:** L
**Estimated time:** 6–9 engineering hours (rough estimate).
**Files to modify:**

- `actproto/data/benchmark_v2.py` — NEW; owned by S2.
- `actproto/data/splits_v2.py` — NEW; owned by S2.
- `actproto/data/curriculum_v2.py` — NEW; owned by S2.
- `actproto/data/fixtures_v2.py` — NEW; owned by S2.
- `scripts/build_benchmark_v2.py` — NEW; owned by S2.
- `tests/test_benchmark_v2.py` — NEW; owned by S2.
- `docs/DATASET_V2.md` — NEW; owned by S2.

**Depends on:** S0
**Blocks:** S4

**Instructions:**

Generate the contract's target counts with fixture-backed tools, varied products/years/records/arithmetic, meaningful data dependencies, distractor catalogs, subset execution, and bounded recoverable failures. Include explicit output convention in every format, placing it before the execution line when header-last must end on that line. Do not train a parser by merely listing tools in execution order for all tasks; include natural task descriptions requiring selection and observation-dependent arguments. Expand fixture banks in a separate versioned environment rather than changing old global fixtures. Pass environment identifiers into execution.

Compute logical group and normalized prompt/call fingerprints before splitting. Keep semantic duplicates and paraphrases together. Hold out argument tuples/entities for IID-new-entity evaluation, and entire composition or language-template families for stress evaluation; report these axes separately rather than claiming both from random seeds. Construct explicit manifests and private oracles; never rely on `split='train'` from the old curriculum factory. Build a reject log for invalid/ambiguous tasks. Replay every gold trace through the real deterministic tool functions before accepting the dataset. Full S1 verifier validation happens at integration.

Train-only failure repair: later collect 100 fixed train-probe tasks, classify tool/argument/termination errors, add at most 120 verified variants within their train groups, and preserve at least 50% original balanced samples in the repair mix. Validation/test errors may inform reports but never supply repair examples. Avoid oversampling easy calculator/finish tokens. Preserve legacy 288/104/104 exactly.

**Definition of Done:**

- [ ] Exact and normalized prompt overlap is zero across splits; logical groups do not cross splits; held-out composition/family constraints pass; all gold tool results replay; header variants state output rules; quota and unique-group counts are reported honestly. Immutable manifest contains seed, split strategy, fixture hash, rejected counts, and per-stratum depth distribution.
- [ ] Focused behavior/invariant tests pass for code changes; document-only work uses file/link/command validation.
- [ ] Targeted tests and the existing full suite pass from the project root; exact commands/failures recorded.
- [ ] Executed experiments have actual logs, artifact hashes and terminal completion; planned commands do not count.
- [ ] Commit only owned files; keep weights, private artifacts, caches and `.agents/` out of Git.

---

### Agent C: S3 — Correct masking and validation-selected SFT checkpoints

**Complexity:** L
**Estimated time:** 5–8 engineering hours (rough estimate).
**Files to modify:**

- `actproto/training/sft_v2.py` — NEW; owned by S3.
- `actproto/training/sft_masks.py` — NEW; owned by S3.
- `actproto/configs/sft_v2.yaml` — NEW; owned by S3.
- `scripts/train_sft_v2.py` — NEW; owned by S3.
- `tests/test_sft_v2.py` — NEW; owned by S3.
- `tests/test_sft_masks.py` — NEW; owned by S3.

**Depends on:** S0
**Blocks:** S4

**Instructions:**

Add a new SFT entry point; leave historical train_lora defaults untouched until documentation marks them historical. Validate the selected named Tiny Aya template and insert generation spans covering assistant call content plus its actual response/end-of-turn delimiters. Compare rendered bytes before and after adding Jinja markers. Verify real token masks, not the existence of marker strings: user/system/tool and padding labels are -100; assistant calls/finish and intended termination labels are trained. Detect empty masks, wrong alignment, and truncated assistant turns before model training. Exclude policy-invisible private fields.

Start from the pinned base model for newly corrected SFT candidates; preserve the old assistant-only adapter as a control. Default rank 16, alpha 32, dropout .05, lr 2e-4, batch 1, grad accumulation 8, maximum two epochs. Save epoch checkpoints plus training state. Inject a validation callback using the shared evaluation runner after it lands; choose by macro verified success, then argument correctness, then fewer invalid calls, then fewer tokens. Do not select on token accuracy alone. Respect configured sequence length; detect overflow rather than silently dropping finish. Bucket long examples and report excluded counts. Use a real quantization config only when selected and supported; otherwise record bf16, never claim QLoRA from an ignored flag.

**Definition of Done:**

- [ ] Tiny mock-model tests establish masking/gradient ownership, callback selection and checkpoint reload; real-tokenizer preflight establishes template equality and token labels before a GPU run. Missing data, marker failure, empty masks, and occupied output paths produce nonzero exit codes. Dry run never loads model weights; real mode must save a trained adapter and completion manifest.
- [ ] Focused behavior/invariant tests pass for code changes; document-only work uses file/link/command validation.
- [ ] Targeted tests and the existing full suite pass from the project root; exact commands/failures recorded.
- [ ] Executed experiments have actual logs, artifact hashes and terminal completion; planned commands do not count.
- [ ] Commit only owned files; keep weights, private artifacts, caches and `.agents/` out of Git.

---

## Integration and what this wave unblocks

Follow `docs/plan_9/PARALLEL-WORK.md`. The integrator reviews API compatibility and file ownership, merges completed task commits, and runs the full suite. This unblocks Sprint 3.
