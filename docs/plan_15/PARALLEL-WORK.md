# Plan 15 parallel work

Evidence lock first, four writers second, three publishers of prose third, one closer last. No wave trains a model or opens the sealed test.

## File dependency matrix

| Task | Reads | Writes | Must wait for |
| --- | --- | --- | --- |
| F1 | `docs/STATUS.md`, `RESULTS_2026-11-14.md`, `DATASET_V2.md`, `METHOD.md`, `SFT_BASELINE_MANIFEST.json`, toolkit and RL docs, `model.yaml`, `sft_v2.yaml`, `trial.yaml`, `e2-scored` if present | `submission/FACTS.md` | nothing |
| D1 | FACTS, `docs/DATASET_V2.md`, benchmark builder | curator script, test, `submission/data/*`, `DATACARD.md` | F1 |
| W1 | FACTS | `submission/ARTICLE.md` | F1 |
| M1 | FACTS | `submission/MODEL_CARD.md` | F1 |
| R1 | FACTS, `README.md`, `STATUS.md` | `submission/REPRODUCE.md` | F1 |
| P1 | ARTICLE | `submission/FORM.md` | W1 |
| H1 | ARTICLE | `submission/hf_article.md` | W1 |
| S2 | FACTS, ARTICLE, MODEL_CARD | `submission/space/**` | W1, M1 |
| G1 | FACTS, the four public docs | `README.md`, `docs/SUBMISSION.md` | F1, D1, W1, M1, R1 |
| C1 | every submission file | `submission/README.md`, `COVERAGE.md`, `PUBLISH.md` | all of the above |

`submission/FACTS.md` is read-only after wave 1. Later tasks that need a missing number write "unknown" and cite FACTS. They do not edit FACTS.

## Wave diagram

```text
Wave 1 (serial)
  [ F1 facts ]
        |
        v
Wave 2 (parallel, disjoint files)
  [ D1 data + card ] [ W1 article ] [ M1 model card ] [ R1 reproduce ]
        |                 |               |                |
        +--------+--------+---------------+----------------+
                 v
Wave 3 (parallel, disjoint files)
  [ P1 form ] [ H1 hf article ] [ S2 space ] [ G1 README ]
        |            |               |            |
        +------------+---------------+------------+
                          v
Wave 4 (serial)
                     [ C1 coverage + runbook ]
```

## Conflict table

| Pair | Conflict | Resolution |
| --- | --- | --- |
| W1 and H1 | Both would be the article | W1 owns the science. H1 copies it in wave 3 and only adds the paste note and link TODOs |
| W1 and P1 | Two abstracts | P1 copies the abstract after W1 and checks the 500-word cap |
| D1 and W1 | Data claims | Both quote FACTS. D1 owns counts in the card; W1 points at the card |
| G1 and W1 | README vs article | G1 links the article and does not restate results |
| S2 and M1 | Numbers in the Space | S2 starts after M1 and copies FACTS only |
| C1 and everyone | Index would rot if written early | C1 is the only writer of `submission/README.md`, `COVERAGE.md`, and `PUBLISH.md` |
| Any task and `results/**` | Adapter stub READMEs | Left alone until a human runs the publish stop-gate |
| Any task and `data/benchmarks/` | Sealed split | Regenerated locally if needed, never committed |

Shared files that must not be edited in parallel: none inside a wave. `FACTS.md` is single-writer. `ARTICLE.md` is single-writer.

## Integration git workflow

Work from `lexsi/`. One branch. Commit at the end of each wave, and do not start the next wave on a dirty tree.

```text
git add lexsi/submission/FACTS.md
git commit -m "Lock Lexsi submission facts"

git add lexsi/scripts/curate_submission_dataset.py lexsi/tests/test_submission_dataset.py lexsi/submission/data lexsi/submission/DATACARD.md lexsi/submission/ARTICLE.md lexsi/submission/MODEL_CARD.md lexsi/submission/REPRODUCE.md
git commit -m "Add Lexsi article, cards, and public data slice"

git add lexsi/submission/FORM.md lexsi/submission/hf_article.md lexsi/submission/space lexsi/README.md lexsi/docs/SUBMISSION.md
git commit -m "Add form, article mirror, Space, and README pointers"

git add lexsi/submission/README.md lexsi/submission/COVERAGE.md lexsi/submission/PUBLISH.md
git commit -m "Add submission coverage matrix and publish runbook"
```

`git add` those paths only. Do not `git add .`. Do not commit `data/benchmarks/`, `.env`, or tokenizer dumps created while checking the license.

Tests for wave 2: from `lexsi/`, `python -m pytest tests/test_submission_dataset.py`. No GPU.

## Recommended assignments

### 2 agents

| Wave | Agent A | Agent B |
| --- | --- | --- |
| 1 | F1 | waits |
| 2 | D1, then R1 | W1, then M1 |
| 3 | P1, then G1 | H1, then S2 |
| 4 | C1 | waits |

Wave 2 agent A must not open `ARTICLE.md`. Agent B must not open `submission/data/`.

### 3 agents

| Wave | Agent A | Agent B | Agent C |
| --- | --- | --- | --- |
| 1 | F1 | waits | waits |
| 2 | D1 | W1 | M1, then R1 |
| 3 | P1 and G1 | H1 | S2 |
| 4 | C1 | waits | waits |

Agent C in wave 2 finishes M1 before R1 so wave 3's Space can start. G1 waits until D1, W1, M1, and R1 are committed.

## Stop rules

- A number that FACTS marks unknown stays unknown in every public file.
- No task publishes to Hugging Face or to `AriCommits/Pub_Hackathons`. C1 only writes `PUBLISH.md`.
- No task copies another project's directory into `lexsi/submission/`.
