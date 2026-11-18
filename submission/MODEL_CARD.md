---
label: model-card
track: T3
audience: public-model
status: draft
license: cc-by-nc-4.0
base_model: CohereLabs/tiny-aya-global
---

# ActProto-LoRA e2

LoRA adapter for verified tool use on Tiny Aya Global. This card is for `results/sft_v2/e2/checkpoint-150` only.

## Model details

- Developed by: Arian Moridani
- Base model: `CohereLabs/tiny-aya-global`
- Base revision: `af89d219b53ed9b13b8a4645f9c8028973510324`
- Adapter id: `e2`, frozen, selected on the validation split
- Directory digest: `6125195bc713f8c4d9e89070233b65667a8068f8fc684a1e14a2522a26b8d670`
- Type: PEFT LoRA on a causal language model
- Language: English tool-calling data. The base model is multilingual. This adapter was not trained or scored as a multilingual system.
- License: the base model card says CC-BY-NC and Cohere Labs' Acceptable Use Policy. The base repository is gated. This adapter is released for noncommercial use under CC-BY-NC-4.0. Base weights are not included.

## Uses

Direct use is local verified tool calls on the fixture tools: `lookup_record`, `query_database`, `search_docs`, `calculator`, and `finish`.

Out of scope: live web access, security or red-team claims, and business decisions that skip the verifier. A fluent answer that did not come from the tools is not a success under the project metric.

## Training

Hyperparameters from `actproto/configs/sft_v2.yaml` and `actproto/training/sft_v2.py`: rank 16, alpha 32, dropout 0.05, learning rate `2e-4`, batch size 1, gradient accumulation 8, seed 42, bf16, 4-bit off. Targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. The frozen manifest records one epoch executed. The config lists two.

Training data is the synthetic English tooluse_v2 train split. See `DATACARD.md`.

## Evaluation

Project notes record macro verified task success of 1.0 on 200 validation tasks. The raw file `results/sft_v2/e2-scored` is not in this checkout.

GRPO studies on this adapter are not a better policy. A pre-fix run stayed at reward −0.1 because sampling ran in train mode. A post-fix run reached reward 1.0 and the groups were often tied, so the weights did not have to move.

## Hardware

Training and the GRPO studies used an NVIDIA RTX 5060 Ti (16 GB). An RTX 4060 (8 GB) is the wrong device for this model in bf16. Hours and dollar cost were not logged.

## How to start

Accept the Tiny Aya license, log in to Hugging Face, and load this directory as a PEFT adapter on the base revision above. Reproduction commands are in `REPRODUCE.md`.
