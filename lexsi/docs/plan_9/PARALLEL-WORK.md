# Plan 9: Parallel work and integration guide

## Reading order and ownership

Start at `sprint_1.md`; OVERVIEW is authoritative. 9 tasks across 6 waves. One integrator owns shared interfaces, merges and GPU leases. Estimates exclude external gates and GPU runtime. No automatic push or merge to master.

## Dependency matrix

| Task | Depends on | Unlocks | Wave |
| --- | --- | --- | --- |
| S0 | none | S1, S2, S3 | 1 |
| S1 | S0 | S4 | 2 |
| S2 | S0 | S4 | 2 |
| S3 | S0 | S4 | 2 |
| S4 | S1, S2, S3 | S5, S6 | 3 |
| S5 | S4 | S7 | 4 |
| S6 | S4 | S7 | 4 |
| S7 | S5, S6 | S8 | 5 |
| S8 | S7 | delivery | 6 |

## File dependency matrix

`██` marks file ownership. Same-wave tasks have no file overlap. Shared package initializer files have a single explicit owner.

```text
File                                       |  S0  S1  S2  S3  S4  S5  S6  S7  S8
---------------------------------------------------------------------------------
README.md                                  |                                  ██
actproto/activations/router_calibration.py |                      ██
actproto/activations/threshold_v2.py       |                      ██
actproto/agent/inference_v2.py             |                  ██
actproto/agent/trace_v2.py                 |                  ██
actproto/configs/experiment_v2.yaml        |  ██
actproto/configs/router_v2.yaml            |                      ██
actproto/configs/sft_matrix.yaml           |                  ██
actproto/configs/sft_v2.yaml               |              ██
actproto/data/benchmark_v2.py              |          ██
actproto/data/curriculum_v2.py             |          ██
actproto/data/fixtures_v2.py               |          ██
actproto/data/splits_v2.py                 |          ██
actproto/evaluation/experiment_runner.py   |                  ██
actproto/evaluation/metrics_v2.py          |      ██
actproto/evaluation/statistics_v2.py       |                              ██
actproto/evaluation/trace_adapter.py       |      ██
actproto/evaluation/verification_v2.py     |      ██
actproto/experiments/__init__.py           |  ██
actproto/experiments/contracts.py          |  ██
actproto/experiments/manifest.py           |  ██
actproto/integrations/__init__.py          |                          ██
actproto/integrations/auditkit_v2.py       |                          ██
actproto/training/sft_masks.py             |              ██
actproto/training/sft_v2.py                |              ██
docs/ACTIVE_PLANS.md                       |  ██
docs/DATASET_V2.md                         |          ██
docs/EXPERIMENT_CONTRACT.md                |  ██
docs/EXPERIMENT_STATUS.md                  |                                  ██
docs/METHOD.md                             |                                  ██
docs/SFT_BASELINE_MANIFEST.json            |                              ██
docs/SFT_REPRODUCTION.md                   |                                  ██
docs/SFT_V2_RESULTS.md                     |                              ██
docs/SUBMISSION.md                         |                                  ██
docs/TOOLKIT_COMPATIBILITY.md              |                          ██
requirements-auditkit.lock                 |                          ██
scripts/build_benchmark_v2.py              |          ██
scripts/calibrate_router_v2.py             |                      ██
scripts/check_toolkit_compat.py            |                          ██
scripts/compare_experiments_v2.py          |                              ██
scripts/export_experiment_v2.py            |                          ██
scripts/run_experiment_v2.py               |                  ██
scripts/run_sft_study.py                   |                              ██
scripts/train_sft_v2.py                    |              ██
tests/test_auditkit_v2.py                  |                          ██
tests/test_benchmark_v2.py                 |          ██
tests/test_experiment_contracts.py         |  ██
tests/test_experiment_runner.py            |                  ██
tests/test_export_experiment_v2.py         |                          ██
tests/test_metrics_v2.py                   |      ██
tests/test_router_calibration.py           |                      ██
tests/test_sft_masks.py                    |              ██
tests/test_sft_v2.py                       |              ██
tests/test_statistics_v2.py                |                              ██
tests/test_trace_v2.py                     |                  ██
tests/test_verification_v2.py              |      ██
```

## Wave execution diagram

```text
Wave 1: Contracts and immutable experiment definitions
  +-- Agent A: S0 -- 3–5 engineering hours --+
                         | integration + verification
                         v
Wave 2: Verification, data and SFT implementation in parallel
  +-- Agent A: S1 -- 5–8 engineering hours --+
  +-- Agent B: S2 -- 6–9 engineering hours --+
  +-- Agent C: S3 -- 5–8 engineering hours --+
                         | integration + verification
                         v
Wave 3: Integrated inference and evaluation
  +-- Agent A: S4 -- 5–8 engineering hours --+
                         | integration + verification
                         v
Wave 4: Router calibration and toolkit/export integration
  +-- Agent A: S5 -- 3–5 engineering hours --+
  +-- Agent B: S6 -- 4–6 engineering hours --+
                         | integration + verification
                         v
Wave 5: Bounded SFT study and baseline selection
  +-- Agent A: S7 -- 4–6 engineering hours plus measured GPU runtime --+
                         | integration + verification
                         v
Wave 6: SFT documentation and RL handoff
  +-- Agent A: S8 -- 2–3 engineering hours --+
```

## File and resource conflicts

| Task | File conflicts | Safe same-wave peers |
| --- | --- | --- |
| S0 | None within plan | None; dependency barrier |
| S1 | None within plan | S2, S3 |
| S2 | None within plan | S1, S3 |
| S3 | None within plan | S1, S2 |
| S4 | None within plan | None; dependency barrier |
| S5 | None within plan | S6 |
| S6 | None within plan | S5 |
| S7 | None within plan | None; dependency barrier |
| S8 | None within plan | None; dependency barrier |

Cross-plan conflicts: S8 and R7 both edit README, METHOD, SUBMISSION and EXPERIMENT_STATUS; R7 follows S8. R0 consumes S0/S6; R3 consumes frozen S7; R6 consumes S8. S7, R4, R5 and R6 require exclusive GPU access. R0 CPU/tokenizer exploration can overlap later SFT documentation after its prerequisites exist, but any GPU capability probe also requires the lease. Independent files never waive resource/dependency gates.

## Integration git workflow

Future execution recipe only. Commit the reviewed plans before branching; preserve unrelated `.agents/` records. Git root is parent Hackathons; project path within each worktree is lexsi. Plan 10 starts from completed Plan 9 integration HEAD. Do not run these commands while merely authoring the plan.

```powershell
$repoRoot = (git rev-parse --show-toplevel).Trim()
$worktreeRoot = Join-Path (Split-Path $repoRoot -Parent) 'Hackathons-worktrees'
New-Item -ItemType Directory -Path $worktreeRoot -Force | Out-Null
git -C $repoRoot switch -c plan-9-integration
```

### Wave 1 setup and integration

After the previous wave is integrated, create branches below. Each task agent commits scoped changes and reports tests before the integrator runs the merge lines. These are separate phases; do not merge empty branches immediately after creating them.

```powershell
git -C $repoRoot worktree add (Join-Path $worktreeRoot "plan-9-s0") -b plan-9-s0 plan-9-integration
```

After agent completion:

```powershell
git -C $repoRoot switch plan-9-integration
git -C $repoRoot merge --no-ff --no-edit plan-9-s0
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
git -C $repoRoot worktree add (Join-Path $worktreeRoot "plan-9-s1") -b plan-9-s1 plan-9-integration
git -C $repoRoot worktree add (Join-Path $worktreeRoot "plan-9-s2") -b plan-9-s2 plan-9-integration
git -C $repoRoot worktree add (Join-Path $worktreeRoot "plan-9-s3") -b plan-9-s3 plan-9-integration
```

After agent completion:

```powershell
git -C $repoRoot switch plan-9-integration
git -C $repoRoot merge --no-ff --no-edit plan-9-s1
git -C $repoRoot merge --no-ff --no-edit plan-9-s2
git -C $repoRoot merge --no-ff --no-edit plan-9-s3
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
git -C $repoRoot worktree add (Join-Path $worktreeRoot "plan-9-s4") -b plan-9-s4 plan-9-integration
```

After agent completion:

```powershell
git -C $repoRoot switch plan-9-integration
git -C $repoRoot merge --no-ff --no-edit plan-9-s4
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
git -C $repoRoot worktree add (Join-Path $worktreeRoot "plan-9-s5") -b plan-9-s5 plan-9-integration
git -C $repoRoot worktree add (Join-Path $worktreeRoot "plan-9-s6") -b plan-9-s6 plan-9-integration
```

After agent completion:

```powershell
git -C $repoRoot switch plan-9-integration
git -C $repoRoot merge --no-ff --no-edit plan-9-s5
git -C $repoRoot merge --no-ff --no-edit plan-9-s6
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
git -C $repoRoot worktree add (Join-Path $worktreeRoot "plan-9-s7") -b plan-9-s7 plan-9-integration
```

After agent completion:

```powershell
git -C $repoRoot switch plan-9-integration
git -C $repoRoot merge --no-ff --no-edit plan-9-s7
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
git -C $repoRoot worktree add (Join-Path $worktreeRoot "plan-9-s8") -b plan-9-s8 plan-9-integration
```

After agent completion:

```powershell
git -C $repoRoot switch plan-9-integration
git -C $repoRoot merge --no-ff --no-edit plan-9-s8
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
| 1 | A: S0; B: review/idle | A: S0; B: review/idle; C: review/idle |
| 2 | A: S1 → S3; B: S2 | A: S1; B: S2; C: S3 |
| 3 | A: S4; B: review/idle | A: S4; B: review/idle; C: review/idle |
| 4 | A: S5; B: S6 | A: S5; B: S6; C: review/idle |
| 5 | A: S7; B: review/idle | A: S7; B: review/idle; C: review/idle |
| 6 | A: S8; B: review/idle | A: S8; B: review/idle; C: review/idle |

2-agent engineering critical path: roughly 28–44 hours plus GPU time, review/integration overhead and external gates. Planning estimate, not a deadline commitment.

3-agent engineering critical path: roughly 24–37 hours plus GPU time, review/integration overhead and external gates. Planning estimate, not a deadline commitment.

## Completion gate

Every task has evidence and a reviewed commit; required tests pass; training tasks save changed reloadable weights with complete manifests; final comparisons include uncertainty and failures. If budget expires, preserve artifacts and report incomplete work. Planned or aborted experiments never count as completed.
