# ActProto-LoRA

Verified tool use on **Tiny Aya Global**, for the Lexsi Track 3 hackathon (agentic fine-tuning).

A LoRA adapter calls local tools to answer synthetic business questions. A task counts only when the harness runs those calls. The selected policy is frozen supervised adapter **e2**. A prototype router and a GRPO trainer sit beside it and do not replace it.

## Start here

| | |
| --- | --- |
| Write-up | [`submission/ARTICLE.md`](submission/ARTICLE.md) |
| Data card | [`submission/DATACARD.md`](submission/DATACARD.md) |
| Public data | [`submission/data/`](submission/data/) — 1,200 training messages, 200 validation prompts |
| Model card | [`submission/MODEL_CARD.md`](submission/MODEL_CARD.md) |
| Reproduce | [`submission/REPRODUCE.md`](submission/REPRODUCE.md) |
| Selected adapter | [`results/sft_v2/e2/checkpoint-150`](results/sft_v2/e2/checkpoint-150) |

The base model is [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global), revision `af89d219b53ed9b13b8a4645f9c8028973510324`, bf16. It is gated. This repo does not include the base weights or the base tokenizer.

## Setup

Python 3.10 or newer.

```bash
cd lexsi
python -m pip install -e ".[dev]"
python -m pytest
```

GPU training and live inference also need `pip install -e ".[train]"` and `huggingface-cli login`. Use a 16 GB card. The notes use an RTX 5060 Ti. An 8 GB card is the wrong device for bf16 Tiny Aya.

## What is in the tree

```text
actproto/     library: tools, data, training, evaluation
scripts/      command-line entry points
tests/        pytest suite
submission/   write-up, cards, and the public data slice
results/      LoRA adapters (weights via Git LFS)
docs/         status and method notes
```

The sealed test is not in `submission/data/`. `data/benchmarks/` is gitignored and is where a local full build goes.

## License

The adapter and the synthetic data slice are released for noncommercial use under CC-BY-NC-4.0. The base model is CC-BY-NC and also requires Cohere Labs' Acceptable Use Policy.
