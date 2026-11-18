---
label: datacard
track: T3
audience: public-dataset
status: draft
license: cc-by-nc-4.0
language:
  - en
pretty_name: ActProto tooluse_v2 public slice
---

# ActProto tooluse_v2 public slice

Public training messages and validation prompts for the Lexsi Track 3 tool-use project. The sealed test, the stress split, oracles, fixtures, and traces are not in this folder.

## Summary

The generator builds a 1,200 / 200 / 200 / 200 train, validation, new-entity test, and stress set at seed 42. This release keeps the 1,200 training conversations and the 200 validation prompts. Validation prompts have no gold answer.

| file | rows | sha256 |
| --- | --- | --- |
| `train_messages.jsonl` | 1,200 | `c01647d7e4a0d36eefd6cfe1dabf2a11ca095a0508ce3e253d7ff02612338a5c` |
| `val_prompts.jsonl` | 200 | `59069d522d1086a85ed31f11f765d3b86bf5c3ffe0acdba1a4ec2bbec632e30a` |

The fixture environment id is `tooluse_v2.fixtures.1`. Split sizes in the full build are train 1,200, val 200, iid 200, stress 200. The iid and stress axes are the sealed 400-task test and are withheld.

## Source

The rows are synthetic English templates in this repository (`actproto/data/benchmark_v2.py`). They are not scraped, not taken from an existing public corpus, and not distilled from a teacher model. Assistant tool calls in `train_messages.jsonl` are the generator's gold trajectory, including the tool observation. They are not samples drawn from Tiny Aya.

Task types are single-tool lookup, dependent calls, ordered subsets, and recovery or no-tool. Tools are `lookup_record`, `query_database`, `search_docs`, `calculator`, and `finish`.

## Languages

English only. The stress axis is a held-out template family on unseen entities. It is not a held-out human language.

## Splits and leak check

The builder checks group, entity, exact prompt, normalized prompt, and call fingerprint, and it assigns the split label only after those fingerprints exist. Train and validation entities do not overlap the test entities. This public folder does not include the test rows, so a reader cannot re-run that check from the release alone. The check lives in the generator.

## What was removed, and why

| withheld | why |
| --- | --- |
| test (`iid`) and stress | sealed final set |
| oracles and gold answers on validation | so the prompts can be published without the key |
| fixtures | the table would let a reader solve the sealed items |
| traces | eval logs are not part of this slice |

No CuratorKIT filter was applied. Nothing was dropped for safety classification. The curator drops a row that still carries an oracle, a gold-call object, a test split, or a stress axis.

## License

CC-BY-NC-4.0 for this synthetic slice. The base model `CohereLabs/tiny-aya-global` is separately CC-BY-NC plus Cohere Labs' Acceptable Use Policy, and it is gated. These JSONL files do not contain base-model weights.

The empty dataset repo on the Hub was created with the license name `fair-noncommercial-research-license`. Replace that stub with `cc-by-nc-4.0` when this card is uploaded.

## How to regenerate

From `lexsi/`:

```bash
python -m scripts.build_benchmark_v2 --output data/benchmarks/tooluse_v2 --seed 42
python -m scripts.curate_submission_dataset --build data/benchmarks/tooluse_v2 --output submission/data
```

The builder refuses to overwrite an existing output directory. `data/benchmarks/` is gitignored on purpose. Only `submission/data/` is the public slice.
