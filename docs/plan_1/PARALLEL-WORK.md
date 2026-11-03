# Parallel work — Plan 1

One code pass, then one gated GPU run. Wave 1 has no file conflicts. Headcount is the only limit on how many of those seven agents run at once.

## File dependency matrix

```
                                      │ S1  L1  P1  R1  H1  T1  D1  G1
──────────────────────────────────────┼────────────────────────────────
scripts/slice_eval_buckets.py         │ ██
tests/test_slice_eval_buckets.py      │ ██
results/b0_b2/bucket_slice.md         │ ██
actproto/training/lora_run.py         │     ██
actproto/configs/sft.yaml             │     ██
tests/test_lora_assistant_loss.py     │     ██
actproto/agent/stop.py                │         ██
actproto/agent/model.py               │         ██
tests/test_generation_stop.py         │         ██
actproto/agent/answer_repair.py       │             ██
actproto/agent/strategy.py            │             ██
tests/test_answer_repair.py           │             ██
actproto/evaluation/live_runner.py    │                 ██
actproto/activations/extract.py       │                 ██
scripts/extract_activations.py        │                 ██
tests/test_extract.py                 │                 ██
actproto/activations/threshold.py     │                     ██
tests/test_prototypes.py              │                     ██
actproto/data/templates.py            │                         ██
actproto/data/generator.py            │                         ██
tests/test_data_splits.py             │                         ██
actproto/configs/trial.yaml           │                             ██
docs/EXPERIMENT_STATUS.md             │                             ██
```

## Wave execution plan

```
Wave 1 (one pass, file-disjoint, no GPU)
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ A  S1  ~1h  │ │ B  L1  ~3h  │ │ C  P1 ~1.5h │ │ D  R1 ~2.5h │
│ slice eval  │ │ assistant   │ │ stop after  │ │ answer      │
│             │ │ loss mask   │ │ one call    │ │ repair      │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ E  H1  ~3h  │ │ F  T1 ~1.5h │ │ G  D1  ~2h  │
│ hop-2       │ │ strict      │ │ thin-family │
│ encodings   │ │ threshold   │ │ local traces│
└─────────────┘ └─────────────┘ └─────────────┘
        │
        │  all seven landed, pytest green, 5060 Ti idle
        ▼
Wave 2 (serial, one agent, GPU)
┌──────────────────────────────────────────────┐
│ A  G1  ~4h                                    │
│ train b1_assistant_only, extract, score      │
│ do not overwrite B0 / B1 / B2                 │
└──────────────────────────────────────────────┘
```

Wave 1 wall time is about 3 hours if all seven run together, about 6 hours with three people, about 8 hours with two. Wave 2 does not overlap Wave 1.

## Conflict table

| Task | Conflicts with | Safe to run with |
| --- | --- | --- |
| S1 | none in Wave 1 | L1, P1, R1, H1, T1, D1 |
| L1 | none in Wave 1 | S1, P1, R1, H1, T1, D1 |
| P1 | none in Wave 1 | S1, L1, R1, H1, T1, D1 |
| R1 | none in Wave 1 | S1, L1, P1, H1, T1, D1 |
| H1 | none in Wave 1 | S1, L1, P1, R1, T1, D1 |
| T1 | none in Wave 1 | S1, L1, P1, R1, H1, D1 |
| D1 | none in Wave 1 | S1, L1, P1, R1, H1, T1 |
| G1 | none, and it still waits | nothing; it consumes the Wave 1 tree and the GPU |

G1 waits because it trains and scores the code from Wave 1, and because the 5060 Ti may still be running B2. It does not edit a Wave 1 file.

## Integration workflow

Each Wave 1 agent commits on its own branch. The integrator merges after that agent’s tests pass. Do not push unless asked.

```bash
# After each Wave 1 agent finishes
git checkout main
git merge agent-s1 --no-commit
git merge agent-l1 --no-commit
git merge agent-p1 --no-commit
git merge agent-r1 --no-commit
git merge agent-h1 --no-commit
git merge agent-t1 --no-commit
git merge agent-d1 --no-commit
.venv\Scripts\python -m pytest
git commit -m "Merge Wave 1: S1 L1 P1 R1 H1 T1 D1"
```

There is nothing to conflict on. If a merge does conflict, the branches touched a file outside their list; put that hunk back on the owning agent.

```bash
# Wave 2, only after the 5060 Ti is idle
git checkout main
git merge agent-g1 --no-commit
git commit -m "Merge Wave 2: G1 masked adapter eval"
```

Wave 2 does not start `train_lora.py` from this document. The agent in `sprint_2.md` starts it, once, on `ACTPROTO_CUDA_DEVICES=1`.

## Recommended agent assignments

Seven Wave 1 tasks are independent. With seven people, each takes one task and the pass ends when the slowest of L1 and H1 ends (~3h), then one person runs G1 (~4h).

Three people, one pass (tasks grouped so nobody shares a file):

| Person | Wave 1 sequence | Then |
| --- | --- | --- |
| A | S1, P1, T1 (~4h) | idle until G1 |
| B | L1, D1 (~5h) | idle until G1 |
| C | R1, H1 (~5.5h) | G1 after the GPU is free (~4h) |

Two people:

| Person | Wave 1 sequence | Then |
| --- | --- | --- |
| A | S1, P1, T1, D1 (~6h) | idle until G1 |
| B | L1, R1, H1 (~8.5h) | G1 after the GPU is free (~4h) |

## Data recommendation (same decision as the overview)

Keep the synthetic generator as the scored benchmark. Do not train on xLAM (CC-BY-4.0), Hermes function-calling (Apache-2.0), Glaive function-calling (Apache-2.0), ToolACE (Apache-2.0), BFCL, or ToolBench. None of them use `calculator`, `search_docs`, `query_database`, `lookup_record`, and `finish` on these fixtures.

Local traces are D1 only: a few new entity ids in `multi_avg_growth`, `error_missing_product`, and `confusable_sales_number`, checked by this harness, split by entity. `easy_calc` does not get more rows. Re-read a dataset card before any download; the card fetch was blocked while this plan was written.
