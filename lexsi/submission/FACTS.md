---
label: internal-facts
track: T3
audience: internal
status: draft
---

# Submission facts

This sheet is the numeric source for the public packet. A value marked `unknown` must stay unknown in every other file. The scored directory `results/sft_v2/e2-scored` is not in this checkout.

| claim | status | value | source |
| --- | --- | --- | --- |
| Base model id | measured | `CohereLabs/tiny-aya-global` | `actproto/configs/model.yaml` |
| Base revision | measured | `af89d219b53ed9b13b8a4645f9c8028973510324` | `actproto/configs/model.yaml` |
| dtype | measured | bfloat16 | `actproto/configs/model.yaml` |
| quantization | measured | off (`load_in_4bit: false`, `quantization: none`) | `actproto/configs/model.yaml`, `actproto/configs/sft_v2.yaml` |
| gated | measured | yes | `actproto/configs/model.yaml` |
| Base license | measured | CC-BY-NC, and Cohere Labs' Acceptable Use Policy. Card text: "License: CC-BY-NC, requires also adhering to Cohere Lab's Acceptable Use Policy." Read from the model card on 18 November 2026. | https://huggingface.co/CohereLabs/tiny-aya-global |
| Base size and context | measured | 3.35B parameters, 8K context, tensor type BF16, about 70 languages in the base model | same model card |
| Selected adapter path | measured | `results/sft_v2/e2/checkpoint-150` | `docs/SFT_BASELINE_MANIFEST.json` |
| Adapter digest | measured | `6125195bc713f8c4d9e89070233b65667a8068f8fc684a1e14a2522a26b8d670` | `docs/SFT_BASELINE_MANIFEST.json` |
| Selected id | measured | `e2`, frozen, selection split `val`, `target_met: true` | `docs/SFT_BASELINE_MANIFEST.json` |
| Epochs | measured | config asks for 2; manifest says `epochs_executed: 1` | `actproto/configs/sft_v2.yaml`, `docs/SFT_BASELINE_MANIFEST.json` |
| LoRA hyperparameters | measured | rank 16, alpha 32, dropout 0.05, learning rate 0.0002, batch size 1, gradient accumulation 8, seed 42, max length 8192, targets `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | `actproto/configs/sft_v2.yaml`, `actproto/training/sft_v2.py` |
| Improvement field | measured | `0.7466666666666666`, stored as a field. The manifest does not define the subtraction. | `docs/SFT_BASELINE_MANIFEST.json` |
| Before-score reconstructed from improvement | unknown | Do not invert the improvement field into a baseline score. | `docs/SFT_BASELINE_MANIFEST.json` |
| Val headline | measured | Project notes say macro `verified_success_v2` is 1.0 on 200/200 validation tasks in `results/sft_v2/e2-scored`. Raw scored JSON absent in this checkout. `docs/RESULTS_2026-11-14.md` calls e2 the primary verified claim and does not restate 200/200. | `docs/plan_13/OVERVIEW.md` |
| Tool-call F1, unnecessary calls, latency on that val file | unknown | Not in a file in this checkout. | `results/sft_v2/e2-scored` missing |
| Dataset contract | measured | seed 42; 1,200 train / 200 val / 200 new-entity test (`iid`) / 200 stress. Environment `tooluse_v2.fixtures.1`. | `docs/DATASET_V2.md`, `actproto/data/benchmark_v2.py`, `actproto/data/fixtures_v2.py` |
| Public slice built here | measured | 1,200 `train_messages`, 200 `val_prompts`. Train sha256 `c01647d7e4a0d36eefd6cfe1dabf2a11ca095a0508ce3e253d7ff02612338a5c`. Val sha256 `59069d522d1086a85ed31f11f765d3b86bf5c3ffe0acdba1a4ec2bbec632e30a`. | `submission/data/manifest.json` |
| Languages in the training text | measured | English templates. Stress is a held-out template family on unseen entities, not a human language. | `docs/DATASET_V2.md`, `actproto/data/benchmark_v2.py` |
| Leak checks | measured | group, entity, exact prompt, normalized prompt, call fingerprint. Checked before the split label is assigned. | `docs/DATASET_V2.md`, `actproto/data/benchmark_v2.py` |
| Data source | measured | Synthetic templates in this repo. Assistant calls in the public train file are generator gold, not Tiny Aya samples and not a teacher distillation. | `scripts/build_benchmark_v2.py` |
| Public data license | measured | CC-BY-NC-4.0 for the synthetic slice. It stays noncommercial, which does not contradict the base-model card. The empty Hub dataset README currently says `fair-noncommercial-research-license`; replace that stub with CC-BY-NC-4.0 on upload. | this sheet; Hub README observed 18 November 2026 |
| Sealed test | measured | withheld, with stress, oracles, fixtures, and traces | `docs/DATASET_V2.md`, `submission/data/manifest.json` |
| AgentTune | measured | 1.1.3 at commit `d4db4c12d52342227c4113a70183d0f64aa2f345`, used as the GRPO and rollout stack when the CUDA env is installed. The selected policy is SFT e2, not a GRPO checkpoint. | `docs/plan_12/OVERVIEW.md`, `docs/RL_GPU_ENV.md` |
| AuditKIT | measured | 1.1.3, commit `71316da345784ff220d5ed59337441a59ab31526`, license LSAL-1.2. Recorded-episode API exists. The 200/200 file is absent, so this checkout cannot show an AuditKIT verdict for that number. The notes attribute the figure to the local val runner. | `docs/TOOLKIT_COMPATIBILITY.md`, `docs/plan_13/OVERVIEW.md` |
| CuratorKIT, AlignTune, SafeTune | measured | not used | no call sites in the submission notes |
| CircuitKIT | measured | named only as an optional layer sweep that must not rewrite `model.yaml`. No committed result shows that sweep changed a reported number. | `docs/METHOD.md` |
| GRPO pre-fix | measured | `results/rl_v2/grpo_gpu_study`: 20 steps, reward −0.1 every step, advantages 0, `template_mismatch` × 42, adapter unchanged. Cause: sampling under `model.train()` plus gradient checkpointing. | `docs/RESULTS_2026-11-14.md` |
| GRPO post-fix | measured | `results/rl_v2/grpo_gpu_study_fix`: 8 steps, `finish` × 18, reward 1.0 × 18, groups often tied at 1.0. Not a validation gain. | `docs/RESULTS_2026-11-14.md` |
| GRPO study hyperparameters | measured | beta 0, learning rate `5e-6`, 2 generations, temperature 0.8, top_p 0.95, 256 new tokens per turn, bf16, vLLM off, seed 42, RTX 5060 Ti UUID `134c34ef-c155-b001-4e3a-b9134b072c21` | `actproto/configs/grpo_v2_gpu_prefix.yaml` |
| Weight-file sha256 cited for "unchanged adapter" | measured | `2da2899c0fd2f5ca0bffc32c2e9201e0b7bab8b1758d2fde429ff19cf50be8bd` is the Git LFS oid of `adapter_model.safetensors`, not the directory digest above. | `docs/plan_13/OVERVIEW.md`, LFS pointer in `results/sft_v2/e2/checkpoint-150/adapter_model.safetensors` |
| Hardware | measured | Model jobs on an RTX 5060 Ti 16 GB. An RTX 4060 8 GB is the wrong device. | `docs/EXPERIMENT_STATUS.md`, `actproto/configs/grpo_v2_gpu_prefix.yaml` |
| GPU hours and dollars | unknown | No committed ledger of hours or cost. | none |
| Problem 2 efficiency number | unknown | Do not claim Track 3 problem 2. | none |
| Closed-model baseline | unknown | none recorded | none |
| Author | measured | Arian Moridani, from the Hugging Face article byline | https://huggingface.co/blog/ArianMoridani/actproto-lora-verified-tool-use-on-tiny-aya-and-a |
| Teammates | measured | none found in the repo | `pyproject.toml` author string is "ActProto-LoRA", not a second person |
| Email | unknown | use the registered address; do not guess | none |
| Article URL | measured | https://huggingface.co/blog/ArianMoridani/actproto-lora-verified-tool-use-on-tiny-aya-and-a | live page, 18 November 2026; body still says links will be added |
| Dataset URL | measured | https://huggingface.co/datasets/ArianMoridani/LexsiHackathon is an empty card | live page, 18 November 2026 |
| Space URL | measured | https://huggingface.co/spaces/ArianMoridani/LexsiHackathon is the default Gradio template | live page, 18 November 2026 |
| GitHub URL | measured | https://github.com/AriCommits/Pub_Hackathons has a stale `lexsi/` tree and no root README | live page, 18 November 2026 |
| Model collection | unknown | none found | https://huggingface.co/collections/ArianMoridani returned 404 |
| Dataset collection | unknown | none found | same |

## Do not claim

- Agentic distillation, or any teacher model.
- A safety-benchmark gain, an attack success rate, or an over-refusal result.
- Multilingual coverage of this adapter. The base model is multilingual. The data is English.
- A live web agent.
- CuratorKIT filtering.
- GRPO as the run that scored 200/200.
- A before-score computed by subtracting `improvement` from 1.0.
- Track 3 problem 2, until an efficiency number exists.
