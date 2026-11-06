# Plan 13: Parallel work and integration guide

Coordination guide for `docs/plan_13/OVERVIEW.md`. Six tasks, five waves. Only Wave 1 has two agents. Waves 2–5 are one agent each because they share files. This document does not implement code, train, or commit.

The repository default branch is `master`, not `main`. The integration checkout is `C:\Users\arian\Github\Hackathons-worktrees\plan-9-complete`, and that checkout is already on `master`. Agent work happens on other branches. Do not push. Do not run `git add .`. Tests use `C:\Users\arian\Github\Hackathons\lexsi\.venv\Scripts\python.exe` from the `lexsi` directory. Never use `C:\Python314\python.exe`.

Zero GRPO loss is tied rewards, not a solved val set. The 200/200 macro `1.0` is frozen SFT e2 (`results/sft_v2/e2-scored`). Beta stays `0.0`. The sealed test split stays closed. Tiny Aya does not go on the 8 GB RTX 4060 (`5b885e12-623f-268c-f07c-ee33db379969`). The RTX 5060 Ti UUID is `134c34ef-c155-b001-4e3a-b9134b072c21`. No wave launches training.

## Dependency graph

| Task | Depends on | Blocks | Wave |
| --- | --- | --- | --- |
| G1 | none | G4, G5, G6 | 1 |
| G2 | none | G3, G4 | 1 |
| G3 | G2 | G6 | 2 |
| G4 | G1, G2 | G5, G6 | 3 |
| G5 | G1, G4 | G6 | 4 |
| G6 | G1, G3, G4, G5 | none | 5 |

G3 and G4 are both ready after Wave 1. They both modify `actproto/training/grpo_run.py` and `tests/test_train_grpo.py`, so they cannot share a wave. Ready tasks are packed in task-id order. G3 takes Wave 2. G4 moves to Wave 3, the earliest later wave with its dependencies done and no file conflict. Putting G4 in Wave 2 and G3 later does not remove a wave: G3 also conflicts with G5 and G6 on `tests/test_train_grpo.py`, and G6 cannot start until G3 and G5 are both done.

## 4a — File dependency matrix

`██` marks a file that task modifies. Empty cells are untouched. Same-wave tasks have no `██` in the same row.

```text
File                                       │ G1  G2  G3  G4  G5  G6
───────────────────────────────────────────┼───────────────────────
actproto/integrations/agenttune_backend.py │ ██
actproto/training/rl_rollouts.py           │ ██          ██
actproto/training/grpo_run.py              │ ██      ██  ██  ██
actproto/training/rewards_v2.py            │     ██
actproto/training/rl_config.py             │             ██  ██
actproto/configs/reward_v2.yaml            │     ██
actproto/configs/grpo_v2_gpu_prefix.yaml   │                 ██
tests/test_agenttune_backend.py            │ ██
tests/test_rl_rollouts.py                  │ ██          ██
tests/test_train_grpo.py                   │ ██      ██  ██  ██  ██
tests/test_rewards_v2.py                   │     ██
tests/test_reward_attacks.py               │     ██
tests/test_grpo_config.py                  │             ██  ██
docs/REWARD_SPEC.md                        │     ██
docs/RL_BACKEND_COMPATIBILITY.md           │                 ██  ██
```

Files the plan leaves untouched, in every wave: `actproto/configs/grpo_v2_gpu_tools.yaml`, `actproto/agent/inference_v2.py`, `actproto/evaluation/verification_v2.py`, `actproto/tools/schemas.py`, `scripts/train_grpo.py`, `scripts/run_experiment_v2.py`, `scripts/compare_experiments_v2.py`, and `docs/plan_13/OVERVIEW.md`.

## 4b — Wave execution diagram

```text
Wave 1: Train rollout budget and failure-only prefix credit
  +-- Agent A: G1 -- 6–8 hours --+
  +-- Agent B: G2 -- 4–6 hours --+
                 | merge Wave 1, then pytest
                 v
Wave 2: Score live sessions with the prefix version
  +-- Agent A: G3 -- 2–3 hours --+
                 | merge Wave 2, then pytest
                 v
Wave 3: Reward breakdown and tied-batch gate
  +-- Agent A: G4 -- 6–8 hours --+
                 | merge Wave 3, then pytest
                 v
Wave 4: Historical tie record and prefix run config
  +-- Agent A: G5 -- 4–6 hours --+
                 | merge Wave 4, then pytest
                 v
Wave 5: Bounded smoke procedure and val comparison
  +-- Agent A: G6 -- 2–3 hours --+
                 | merge Wave 5, then pytest
                 v
               done (no training launched)
```

Wall clock is the Wave 1 max plus Waves 2–5 in order: about 20–28 hours. A third agent does not shorten it.

## 4c — Conflict table

| Task | Conflicts with | Safe to run with |
| --- | --- | --- |
| G1 | G3 (`grpo_run.py`, `tests/test_train_grpo.py`); G4 (`rl_rollouts.py`, `grpo_run.py`, `tests/test_rl_rollouts.py`, `tests/test_train_grpo.py`); G5 (`grpo_run.py`, `tests/test_train_grpo.py`); G6 (`tests/test_train_grpo.py`) | G2 |
| G2 | none on files. G3 and G4 still wait, because they depend on G2 | G1 |
| G3 | G1 (`grpo_run.py`, `tests/test_train_grpo.py`); G4 (same two files); G5 (same two files); G6 (`tests/test_train_grpo.py`) | none in its wave |
| G4 | G1 (rollout, trainer, and both test modules above); G3 (`grpo_run.py`, `tests/test_train_grpo.py`); G5 (`grpo_run.py`, `rl_config.py`, `tests/test_grpo_config.py`, `tests/test_train_grpo.py`); G6 (`tests/test_train_grpo.py`) | none in its wave |
| G5 | G1 (`grpo_run.py`, `tests/test_train_grpo.py`); G3 (same two); G4 (`grpo_run.py`, `rl_config.py`, `tests/test_grpo_config.py`, `tests/test_train_grpo.py`); G6 (`docs/RL_BACKEND_COMPATIBILITY.md`, `tests/test_train_grpo.py`) | none in its wave |
| G6 | G1, G3, G4, G5 on `tests/test_train_grpo.py`; G5 also on `docs/RL_BACKEND_COMPATIBILITY.md` | none in its wave |

G2 shares no file with G5 or G6, but both of those waves are after G4, and G4 depends on G2, so G2 is already merged before they start.

## 4d — Integration workflow

Future execution only. Do not run these commands while authoring the plan. The default branch is `master`. Do not check out or name a branch `main`. Integrate by merging agent branches into the `master` checkout at `C:\Users\arian\Github\Hackathons-worktrees\plan-9-complete`. Do not push. Do not run `git add .`. A clean merge is already staged; `git commit` records it. On a conflict, `git add` only the resolved paths listed for that wave, then commit.

```powershell
$repo = 'C:\Users\arian\Github\Hackathons-worktrees\plan-9-complete'
$py = 'C:\Users\arian\Github\Hackathons\lexsi\.venv\Scripts\python.exe'
# Never C:\Python314\python.exe
```

### After Wave 1 completes

Agents commit on their own branches, adding only their sprint files by path. G1: `lexsi/actproto/integrations/agenttune_backend.py`, `lexsi/actproto/training/rl_rollouts.py`, `lexsi/actproto/training/grpo_run.py`, `lexsi/tests/test_agenttune_backend.py`, `lexsi/tests/test_rl_rollouts.py`, `lexsi/tests/test_train_grpo.py`. G2: `lexsi/actproto/training/rewards_v2.py`, `lexsi/actproto/configs/reward_v2.yaml`, `lexsi/docs/REWARD_SPEC.md`, `lexsi/tests/test_rewards_v2.py`, `lexsi/tests/test_reward_attacks.py`.

```powershell
git -C $repo branch agent-a-sprint1 HEAD
git -C $repo branch agent-b-sprint1 HEAD
git -C $repo worktree add 'C:\Users\arian\Github\Hackathons-worktrees\plan-13-g1' agent-a-sprint1
git -C $repo worktree add 'C:\Users\arian\Github\Hackathons-worktrees\plan-13-g2' agent-b-sprint1
```

Create those branches before the agents edit. Do not merge empty branches. After both agents have committed:

```powershell
git -C $repo checkout master
git -C $repo merge agent-a-sprint1 --no-commit
git -C $repo merge agent-b-sprint1 --no-commit
# resolve conflicts, then git add only the resolved Wave 1 paths
Set-Location (Join-Path $repo 'lexsi')
& $py -m pytest
if ($LASTEXITCODE -ne 0) { throw 'Wave 1 pytest failed. Do not start Sprint 2.' }
git -C $repo commit -m "Merge Wave 1: G1, G2"
```

Wave 1 should not conflict: G1 and G2 touch disjoint files. Stop if pytest fails.

### After Wave 2 completes

G3 paths: `lexsi/actproto/training/grpo_run.py`, `lexsi/tests/test_train_grpo.py`.

```powershell
git -C $repo branch agent-a-sprint2 HEAD
git -C $repo worktree add 'C:\Users\arian\Github\Hackathons-worktrees\plan-13-g3' agent-a-sprint2
```

After the agent has committed:

```powershell
git -C $repo checkout master
git -C $repo merge agent-a-sprint2 --no-commit
Set-Location (Join-Path $repo 'lexsi')
& $py -m pytest
if ($LASTEXITCODE -ne 0) { throw 'Wave 2 pytest failed. Do not start Sprint 3.' }
git -C $repo commit -m "Merge Wave 2: G3"
```

### After Wave 3 completes

G4 paths: `lexsi/actproto/training/grpo_run.py`, `lexsi/actproto/training/rl_rollouts.py`, `lexsi/actproto/training/rl_config.py`, `lexsi/tests/test_train_grpo.py`, `lexsi/tests/test_rl_rollouts.py`, `lexsi/tests/test_grpo_config.py`.

```powershell
git -C $repo branch agent-a-sprint3 HEAD
git -C $repo worktree add 'C:\Users\arian\Github\Hackathons-worktrees\plan-13-g4' agent-a-sprint3
```

After the agent has committed:

```powershell
git -C $repo checkout master
git -C $repo merge agent-a-sprint3 --no-commit
Set-Location (Join-Path $repo 'lexsi')
& $py -m pytest
if ($LASTEXITCODE -ne 0) { throw 'Wave 3 pytest failed. Do not start Sprint 4.' }
git -C $repo commit -m "Merge Wave 3: G4"
```

### After Wave 4 completes

G5 paths: `lexsi/actproto/configs/grpo_v2_gpu_prefix.yaml`, `lexsi/actproto/training/grpo_run.py`, `lexsi/actproto/training/rl_config.py`, `lexsi/tests/test_grpo_config.py`, `lexsi/tests/test_train_grpo.py`, `lexsi/docs/RL_BACKEND_COMPATIBILITY.md`.

```powershell
git -C $repo branch agent-a-sprint4 HEAD
git -C $repo worktree add 'C:\Users\arian\Github\Hackathons-worktrees\plan-13-g5' agent-a-sprint4
```

After the agent has committed:

```powershell
git -C $repo checkout master
git -C $repo merge agent-a-sprint4 --no-commit
Set-Location (Join-Path $repo 'lexsi')
& $py -m pytest
if ($LASTEXITCODE -ne 0) { throw 'Wave 4 pytest failed. Do not start Sprint 5.' }
git -C $repo commit -m "Merge Wave 4: G5"
```

### After Wave 5 completes

G6 paths: `lexsi/docs/RL_BACKEND_COMPATIBILITY.md`, `lexsi/tests/test_train_grpo.py`.

```powershell
git -C $repo branch agent-a-sprint5 HEAD
git -C $repo worktree add 'C:\Users\arian\Github\Hackathons-worktrees\plan-13-g6' agent-a-sprint5
```

After the agent has committed:

```powershell
git -C $repo checkout master
git -C $repo merge agent-a-sprint5 --no-commit
Set-Location (Join-Path $repo 'lexsi')
& $py -m pytest
if ($LASTEXITCODE -ne 0) { throw 'Wave 5 pytest failed.' }
git -C $repo commit -m "Merge Wave 5: G6"
```

Do not follow Wave 5 with a training launch. The smoke procedure in Sprint 5 is documentation. Default pytest must not start it. Leave extra worktrees in place until a human asks to remove them.

## 4e — Recommended agent assignments

Estimates are engineering time for the implementation sprints. They exclude a later GPU smoke. No schedule trains.

### Two agents

| Agent | Sequence | Busy time |
| --- | --- | --- |
| A | G1 (Wave 1), then G3, G4, G5, G6 | 20–28 hours |
| B | G2 (Wave 1), then idle | 4–6 hours |

Wave 1 wall clock is G1, 6–8 hours, because it is longer than G2. Waves 2–5 add 2–3 + 6–8 + 4–6 + 2–3 hours. Total about 20–28 hours. Agent B cannot pick up G3 during Wave 1: G3 depends on G2. After Wave 1, every remaining task conflicts with the one Agent A is finishing, so a second agent does not overlap Waves 2–5.

### Three agents

| Agent | Sequence | Busy time |
| --- | --- | --- |
| A | G1, then G3, G4, G5, G6 | 20–28 hours |
| B | G2, then idle | 4–6 hours |
| C | idle in every wave | 0 hours |

The third agent has no file-disjoint task left. Wall clock stays 20–28 hours, the same as two agents.

Start at `docs/plan_13/sprint_1.md`. Agent A takes G1. Agent B takes G2. Merge Wave 1 and run pytest before Sprint 2.
