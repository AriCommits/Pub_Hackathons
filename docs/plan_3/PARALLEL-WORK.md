# Plan 3 parallel work

Depth-1 retry (`B2d1r`) and chained routing (`B2chain`) share one GPU, so the evals are serial. The three code tasks share no files.

## File dependency matrix

```
                              │ S1  F1  T1  E1  E2
──────────────────────────────┼─────────────────────
actproto/agent/stop.py        │ ██
actproto/agent/model.py       │ ██
tests/test_generation_stop.py │ ██
actproto/agent/routed_strategy.py │  ██
tests/test_finish_retry.py    │     ██
actproto/configs/trial_schema.py │     ██
actproto/configs/trial.yaml   │         ██
actproto/evaluation/live_runner.py │     ██
tests/test_trial_config.py    │         ██
tests/test_run_trial.py       │         ██
docs/EXPERIMENT_STATUS.md     │                 ██  ██
```

E1 and E2 both append to `EXPERIMENT_STATUS.md`. They do not run in the same wave.

## Wave execution plan

```
Wave 1 (parallel, CPU, ~1.5h)
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│ Agent A             │  │ Agent B             │  │ Agent C             │
│ S1 stop criterion   │  │ F1 finish retry     │  │ T1 B2d1r + B2chain  │
│ ~1h                 │  │ ~1.5h               │  │ ~1h                 │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘
            \                      |                      /
             \                     |                     /
              v                    v                    v
Wave 2 (serial, GPU, ~1.5h)
┌──────────────────────────────────────────────────────────┐
│ Agent A                                                  │
│ E1  run_trial.py --run B2d1r   on cuda:1, one replica    │
└──────────────────────────────────────────────────────────┘
                                |
                                v
Wave 3 (serial, GPU, ~2.5h)
┌──────────────────────────────────────────────────────────┐
│ Agent A                                                  │
│ E2  run_trial.py --run B2chain on cuda:1, one replica    │
└──────────────────────────────────────────────────────────┘
```

## Conflict table

| Task | Conflicts with | Safe to run with |
| --- | --- | --- |
| S1 | none in Wave 1 | F1, T1 |
| F1 | none in Wave 1 | S1, T1 |
| T1 | none in Wave 1 | S1, F1 |
| E1 | E2 (same GPU, same status file) | none |
| E2 | E1 (same GPU, same status file) | none |

## Integration workflow

Branch names below are suggestions. The repo's default branch is `master`. Do not commit result JSON. `results/` is gitignored except the small tracked README.

```bash
# After Wave 1 completes
git checkout master
git merge agent-a-s1 --no-commit
git merge agent-b-f1 --no-commit
git merge agent-c-t1 --no-commit
# resolve conflicts if a merge touched the same file anyway
.venv/Scripts/python -m pytest tests/test_generation_stop.py tests/test_agent_parse.py tests/test_finish_retry.py tests/test_arg_prompt.py tests/test_routed_strategy.py tests/test_trial_config.py tests/test_run_trial.py
git commit -m "Merge Wave 1: one-call stop, finish retry, B2d1r and B2chain ids"

# After Wave 2: commit the status pointer only
git add lexsi/docs/EXPERIMENT_STATUS.md
git commit -m "Record the B2d1r depth-1 retry"

# After Wave 3
git add lexsi/docs/EXPERIMENT_STATUS.md
git commit -m "Record the B2chain transcript-routing eval"
```

Run those commits only when the user has asked for them. The commands are the integration sequence, not a license to commit mid-sprint.

## Recommended assignments

### 2 agents

| Agent | Sequence | Time |
| --- | --- | --- |
| A | S1, then T1, then E1, then E2 | ~6h |
| B | F1 during Wave 1, then idle until the status pointers need a read | ~1.5h |

Wave 1 wall clock is about 1.5h (A's S1+T1 is the long pole if A does both, or 1h if a third person takes T1). GPU waves add about 4h and cannot overlap.

### 3 agents

| Agent | Sequence | Time |
| --- | --- | --- |
| A | S1, then E1, then E2 | ~5h |
| B | F1 | ~1.5h |
| C | T1 | ~1h |

Wave 1 wall clock is about 1.5h. The same person should run E1 and E2 so the second eval starts only after `B2d1r/eval.json` is on disk.
