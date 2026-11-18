---
base_model: CohereLabs/tiny-aya-global
library_name: peft
pipeline_tag: text-generation
license: cc-by-nc-4.0
tags:
- base_model:adapter:CohereLabs/tiny-aya-global
- lora
- transformers
---

# ActProto-LoRA e2

Selected frozen SFT adapter for verified tool use on Tiny Aya Global.

- Revision of the base: `af89d219b53ed9b13b8a4645f9c8028973510324`
- Directory digest: `6125195bc713f8c4d9e89070233b65667a8068f8fc684a1e14a2522a26b8d670`
- Rank 16, alpha 32, dropout 0.05, bf16, not 4-bit
- Card: [`submission/MODEL_CARD.md`](../../../../submission/MODEL_CARD.md)

The base weights and the base tokenizer are not in this folder. Load this directory as a PEFT adapter on the gated base model.
