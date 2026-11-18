# Plan 10: Parallel work and integration guide

## Reading order and ownership

Start at `sprint_1.md`; OVERVIEW is authoritative. 8 tasks across 7 waves. One integrator owns shared interfaces, merges and GPU leases. Estimates exclude external gates and GPU runtime. No automatic push or merge to master.

## Dependency matrix

| Task | Depends on | Unlocks | Wave |
| --- | --- | --- | --- |
| R0 | external S0, external S6 | R1, R2 | 1 |
| R1 | R0 | R3 | 2 |
| R2 | R0 | R3 | 2 |
| R3 | R1, R2, external S7 | R4 | 3 |
| R4 | R3 | R5 | 4 |
| R5 | R4 | R6 | 5 |
| R6 | R5, external S8 | R7 | 6 |
| R7 | R6 | delivery | 7 |

## File dependency matrix

`██` marks file ownership. Same-wave tasks have no file overlap. Shared package initializer files have a single explicit owner.

```text
File                                       |  R0  R1  R2  R3  R4  R5  R6  R7
-----------------------------------------------------------------------------
README.md                                  |                              ██
actproto/configs/grpo_v2.yaml              |              ██
actproto/configs/reward_v2.yaml            |          ██
actproto/configs/rl_eval_matrix.yaml       |                          ██
actproto/configs/rl_study.yaml             |                      ██
actproto/integrations/agenttune_backend.py |  ██
actproto/training/grpo.py                  |              ██
actproto/training/grpo_run.py              |              ██
actproto/training/rewards_v2.py            |          ██
actproto/training/rl_config.py             |              ██
actproto/training/rl_dataset.py            |      ██
actproto/training/rl_rollouts.py           |      ██
actproto/training/rl_tools.py              |      ██
docs/EXPERIMENT_STATUS.md                  |                              ██
docs/METHOD.md                             |                              ██
docs/REWARD_SPEC.md                        |          ██
docs/RL_BACKEND_COMPATIBILITY.md           |  ██
docs/RL_EVALUATION.md                      |                          ██
docs/RL_MODEL_CARD.md                      |                              ██
docs/RL_REPRODUCTION.md                    |                              ██
docs/RL_SELECTED_MANIFEST.json             |                      ██
docs/RL_SMOKE_REPORT.md                    |                  ██
docs/RL_TRAINING_RESULTS.md                |                      ██
docs/SUBMISSION.md                         |                              ██
requirements-rl.lock                       |  ██
scripts/check_rl_backend.py                |  ██
scripts/collect_rl_rollouts.py             |      ██
scripts/evaluate_rl_study.py               |                          ██
scripts/run_rl_study.py                    |                      ██
scripts/train_grpo.py                      |              ██
scripts/verify_rl_training.py              |                  ██
tests/test_agenttune_backend.py            |  ██
tests/test_grpo_config.py                  |              ██
tests/test_reward_attacks.py               |          ██
tests/test_rewards_v2.py                   |          ██
tests/test_rl_comparison.py                |                          ██
tests/test_rl_dataset.py                   |      ██
tests/test_rl_end_to_end.py                |                  ██
tests/test_rl_learning.py                  |                  ██
tests/test_rl_rollouts.py                  |      ██
tests/test_train_grpo.py                   |              ██
```

## Wave execution diagram

```text
Wave 1: AgentTune capability and template gate
  +-- Agent A: R0 -- 4–7 engineering hours --+
                         | integration + verification
                         v
Wave 2: Rollouts and rewards in parallel
  +-- Agent A: R1 -- 5–8 engineering hours --+
  +-- Agent B: R2 -- 4–7 engineering hours --+
                         | integration + verification
                         v
Wave 3: Actual GRPO training lifecycle
  +-- Agent A: R3 -- 6–10 engineering hours --+
                         | integration + verification
                         v
Wave 4: Learning-mechanics and GPU smoke gates
  +-- Agent A: R4 -- 4–6 engineering hours plus bounded smoke runtime --+
                         | integration + verification
                         v
Wave 5: Bounded RL study and checkpoint selection
  +-- Agent A: R5 -- 3–5 engineering hours plus measured GPU runtime --+
                         | integration + verification
                         v
Wave 6: Matched controls and sealed evaluation
  +-- Agent A: R6 -- 4–6 engineering hours plus measured control/evaluation runtime --+
                         | integration + verification
                         v
Wave 7: RL reproduction and submission artifacts
  +-- Agent A: R7 -- 2–4 engineering hours --+
```

## File and resource conflicts

| Task | File conflicts | Safe same-wave peers |
| --- | --- | --- |
| R0 | None within plan | None; dependency barrier |
| R1 | None within plan | R2 |
| R2 | None within plan | R1 |
| R3 | None within plan | None; dependency barrier |
| R4 | None within plan | None; dependency barrier |
| R5 | None within plan | None; dependency barrier |
| R6 | None within plan | None; dependency barrier |
| R7 | None within plan | None; dependency barrier |

Cross-plan conflicts: S8 and R7 both edit README, METHOD, SUBMISSION and EXPERIMENT_STATUS; R7 follows S8. R0 consumes S0/S6; R3 consumes frozen S7; R6 consumes S8. S7, R4, R5 and R6 require exclusive GPU access. R0 CPU/tokenizer exploration can overlap later SFT documentation after its prerequisites exist, but any GPU capability probe also requires the lease. Independent files never waive resource/dependency gates.

## Integration git workflow

Future execution recipe only. Commit the reviewed plans before branching; preserve unrelated `.agents/` records. Git root is parent Hackathons; project path within each worktree is lexsi. Plan 10 starts from completed Plan 9 integration HEAD. Do not run these commands while merely authoring the plan.

```powershell
$repoRoot = (git rev-parse --show-toplevel).Trim()
$worktreeRoot = Join-Path (Split-Path $repoRoot -Parent) 'Hackathons-worktrees'
New-Item -ItemType Directory -Path $worktreeRoot -Force | Out-Null
git -C $repoRoot switch -c plan-10-integration
```

### Wave 1 setup and integration

After the previous wave is integrated, create branches below. Each task agent commits scoped changes and reports tests before the integrator runs the merge lines. These are separate phases; do not merge empty branches immediately after creating them.

```powershell
git -C $repoRoot worktree add (Join-Path $worktreeRoot "plan-10-r0") -b plan-10-r0 plan-10-integration
```

After agent completion:

```powershell
git -C $repoRoot switch plan-10-integration
git -C $repoRoot merge --no-ff --no-edit plan-10-r0
Push-Location (Join-Path $repoRoot "lexsi")
try {
    & ./.venv/Scripts/python.exe -m pytest
    if ($LASTEXITCODE -ne 0) { throw "Integration tests failed; do not start the next wave." }
    git diff --check
    if ($LASTEXITCODE -ne 0) { throw "Whitespace check failed." }
} finally { Pop-Location }
```

Also run installed-toolkit integration tests in the isolated environment chosen by S6/R0; mocks and the lightweight suite are not substitutes. Stop on merge failure, resolve and verify before proceeding. Record completion SHA and evidence; no automatic worktree deletion or push.

### Wave 2 setup and integration

After the previous wave is integrated, create branches below. Each task agent commits scoped changes and reports tests before the integrator runs the merge lines. These are separate phases; do not merge empty branches immediately after creating them.

```powershell
git -C $repoRoot worktree add (Join-Path $worktreeRoot "plan-10-r1") -b plan-10-r1 plan-10-integration
git -C $repoRoot worktree add (Join-Path $worktreeRoot "plan-10-r2") -b plan-10-r2 plan-10-integration
```

After agent completion:

```powershell
git -C $repoRoot switch plan-10-integration
git -C $repoRoot merge --no-ff --no-edit plan-10-r1
git -C $repoRoot merge --no-ff --no-edit plan-10-r2
Push-Location (Join-Path $repoRoot "lexsi")
try {
    & ./.venv/Scripts/python.exe -m pytest
    if ($LASTEXITCODE -ne 0) { throw "Integration tests failed; do not start the next wave." }
    git diff --check
    if ($LASTEXITCODE -ne 0) { throw "Whitespace check failed." }
} finally { Pop-Location }
```

Also run installed-toolkit integration tests in the isolated environment chosen by S6/R0; mocks and the lightweight suite are not substitutes. Stop on merge failure, resolve and verify before proceeding. Record completion SHA and evidence; no automatic worktree deletion or push.

### Wave 3 setup and integration

After the previous wave is integrated, create branches below. Each task agent commits scoped changes and reports tests before the integrator runs the merge lines. These are separate phases; do not merge empty branches immediately after creating them.

```powershell
git -C $repoRoot worktree add (Join-Path $worktreeRoot "plan-10-r3") -b plan-10-r3 plan-10-integration
```

After agent completion:

```powershell
git -C $repoRoot switch plan-10-integration
git -C $repoRoot merge --no-ff --no-edit plan-10-r3
Push-Location (Join-Path $repoRoot "lexsi")
try {
    & ./.venv/Scripts/python.exe -m pytest
    if ($LASTEXITCODE -ne 0) { throw "Integration tests failed; do not start the next wave." }
    git diff --check
    if ($LASTEXITCODE -ne 0) { throw "Whitespace check failed." }
} finally { Pop-Location }
```

Also run installed-toolkit integration tests in the isolated environment chosen by S6/R0; mocks and the lightweight suite are not substitutes. Stop on merge failure, resolve and verify before proceeding. Record completion SHA and evidence; no automatic worktree deletion or push.

### Wave 4 setup and integration

After the previous wave is integrated, create branches below. Each task agent commits scoped changes and reports tests before the integrator runs the merge lines. These are separate phases; do not merge empty branches immediately after creating them.

```powershell
git -C $repoRoot worktree add (Join-Path $worktreeRoot "plan-10-r4") -b plan-10-r4 plan-10-integration
```

After agent completion:

```powershell
git -C $repoRoot switch plan-10-integration
git -C $repoRoot merge --no-ff --no-edit plan-10-r4
Push-Location (Join-Path $repoRoot "lexsi")
try {
    & ./.venv/Scripts/python.exe -m pytest
    if ($LASTEXITCODE -ne 0) { throw "Integration tests failed; do not start the next wave." }
    git diff --check
    if ($LASTEXITCODE -ne 0) { throw "Whitespace check failed." }
} finally { Pop-Location }
```

Also run installed-toolkit integration tests in the isolated environment chosen by S6/R0; mocks and the lightweight suite are not substitutes. Stop on merge failure, resolve and verify before proceeding. Record completion SHA and evidence; no automatic worktree deletion or push.

### Wave 5 setup and integration

After the previous wave is integrated, create branches below. Each task agent commits scoped changes and reports tests before the integrator runs the merge lines. These are separate phases; do not merge empty branches immediately after creating them.

```powershell
git -C $repoRoot worktree add (Join-Path $worktreeRoot "plan-10-r5") -b plan-10-r5 plan-10-integration
```

After agent completion:

```powershell
git -C $repoRoot switch plan-10-integration
git -C $repoRoot merge --no-ff --no-edit plan-10-r5
Push-Location (Join-Path $repoRoot "lexsi")
try {
    & ./.venv/Scripts/python.exe -m pytest
    if ($LASTEXITCODE -ne 0) { throw "Integration tests failed; do not start the next wave." }
    git diff --check
    if ($LASTEXITCODE -ne 0) { throw "Whitespace check failed." }
} finally { Pop-Location }
```

Also run installed-toolkit integration tests in the isolated environment chosen by S6/R0; mocks and the lightweight suite are not substitutes. Stop on merge failure, resolve and verify before proceeding. Record completion SHA and evidence; no automatic worktree deletion or push.

### Wave 6 setup and integration

After the previous wave is integrated, create branches below. Each task agent commits scoped changes and reports tests before the integrator runs the merge lines. These are separate phases; do not merge empty branches immediately after creating them.

```powershell
git -C $repoRoot worktree add (Join-Path $worktreeRoot "plan-10-r6") -b plan-10-r6 plan-10-integration
```

After agent completion:

```powershell
git -C $repoRoot switch plan-10-integration
git -C $repoRoot merge --no-ff --no-edit plan-10-r6
Push-Location (Join-Path $repoRoot "lexsi")
try {
    & ./.venv/Scripts/python.exe -m pytest
    if ($LASTEXITCODE -ne 0) { throw "Integration tests failed; do not start the next wave." }
    git diff --check
    if ($LASTEXITCODE -ne 0) { throw "Whitespace check failed." }
} finally { Pop-Location }
```

Also run installed-toolkit integration tests in the isolated environment chosen by S6/R0; mocks and the lightweight suite are not substitutes. Stop on merge failure, resolve and verify before proceeding. Record completion SHA and evidence; no automatic worktree deletion or push.

### Wave 7 setup and integration

After the previous wave is integrated, create branches below. Each task agent commits scoped changes and reports tests before the integrator runs the merge lines. These are separate phases; do not merge empty branches immediately after creating them.

```powershell
git -C $repoRoot worktree add (Join-Path $worktreeRoot "plan-10-r7") -b plan-10-r7 plan-10-integration
```

After agent completion:

```powershell
git -C $repoRoot switch plan-10-integration
git -C $repoRoot merge --no-ff --no-edit plan-10-r7
Push-Location (Join-Path $repoRoot "lexsi")
try {
    & ./.venv/Scripts/python.exe -m pytest
    if ($LASTEXITCODE -ne 0) { throw "Integration tests failed; do not start the next wave." }
    git diff --check
    if ($LASTEXITCODE -ne 0) { throw "Whitespace check failed." }
} finally { Pop-Location }
```

Also run installed-toolkit integration tests in the isolated environment chosen by S6/R0; mocks and the lightweight suite are not substitutes. Stop on merge failure, resolve and verify before proceeding. Record completion SHA and evidence; no automatic worktree deletion or push.

## Recommended agent assignments

An integrator can be one of the workers. With two workers, queue surplus tasks within the wave. Never overlap dependent waves.

| Wave | Two-agent schedule | Three-agent schedule |
| --- | --- | --- |
| 1 | A: R0; B: review/idle | A: R0; B: review/idle; C: review/idle |
| 2 | A: R1; B: R2 | A: R1; B: R2; C: review/idle |
| 3 | A: R3; B: review/idle | A: R3; B: review/idle; C: review/idle |
| 4 | A: R4; B: review/idle | A: R4; B: review/idle; C: review/idle |
| 5 | A: R5; B: review/idle | A: R5; B: review/idle; C: review/idle |
| 6 | A: R6; B: review/idle | A: R6; B: review/idle; C: review/idle |
| 7 | A: R7; B: review/idle | A: R7; B: review/idle; C: review/idle |

2-agent engineering critical path: roughly 28–46 hours plus GPU time, review/integration overhead and external gates. Planning estimate, not a deadline commitment.

3-agent engineering critical path: roughly 28–46 hours plus GPU time, review/integration overhead and external gates. Planning estimate, not a deadline commitment.

## Completion gate

Every task has evidence and a reviewed commit; required tests pass; training tasks save changed reloadable weights with complete manifests; final comparisons include uncertainty and failures. If budget expires, preserve artifacts and report incomplete work. Planned or aborted experiments never count as completed.
