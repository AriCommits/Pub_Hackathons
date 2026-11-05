# Plan 11: GPU environment for the ActProto GRPO stack

## Summary

Move Lexsi ActProto GRPO off the CPU-only AgentTune environment and onto a separate CUDA environment aimed at the RTX 5060 Ti. Keep the already-verified CPU smoke as a mechanics fallback, add a fail-closed GPU config that is distinct from `actproto/configs/grpo_v2.yaml`, and allow `use_cpu: false` only for that named profile. Prove the new stack with a smallest-safe CUDA LoRA update, then measure Tiny Aya bf16 memory on the 5060 Ti while that card is free. Record what those runs actually verified. This plan does not run the Plan 10 study, does not treat beta 0 as an SFT reference, and does not start vLLM.

## Goals

- A new RL virtualenv, separate from both `C:\Users\arian\Github\Hackathons\lexsi\.venv` and `C:\Users\arian\Github\Hackathons-worktrees\plan-10-rl-env`, reports `torch 2.11.0+cu128`, `torch.cuda.is_available() is True`, AgentTune 1.1.3 at commit `d4db4c12d52342227c4113a70183d0f64aa2f345`, and TRL 1.7.1.
- `actproto/configs/grpo_v2.yaml` still loads only as the beta-zero CPU profile. A new GPU config is rejected if it is rewritten into that CPU profile, and the CPU file is rejected if `use_cpu` is flipped.
- `create_trainer` still raises "Only CPU trainer construction is currently verified" for a bare `use_cpu: false`. The same call succeeds only when the named GPU profile is passed and `use_vllm` is false.
- A one-step CUDA smoke on the 5060 Ti, using the existing 1-layer GPT-2 LoRA fixture, produces a finite nonzero gradient, changes only LoRA weights, builds no reference model, and writes under a new `results/rl_v2` directory.
- A short Tiny Aya bf16 memory probe on the same card records allocated, reserved, and peak memory before any claim that Tiny Aya GRPO fits. The probe waits until the SFT scoring job is no longer using the card.
- `docs/RL_BACKEND_COMPATIBILITY.md` states the interpreter, the device UUID that was used, the smoke result, and the memory numbers from the run that produced them.

## Non-Goals

- Plan 10 R4 through R7: the learning-mechanics bandit, the bounded pilot, sealed final-test comparison, and the submission narrative. Those stay in `docs/plan_10/OVERVIEW.md`.
- vLLM, multi-GPU training, bitsandbytes, and any other quantization path. vLLM stays closed until this plan's HF/TRL GPU smoke and memory probe have both been measured.
- Replacing or upgrading the supervised environment at `C:\Users\arian\Github\Hackathons\lexsi\.venv`, and replacing the CPU RL environment at `C:\Users\arian\Github\Hackathons-worktrees\plan-10-rl-env`.
- Loading the frozen SFT adapter as the GRPO reference. beta 0 still means no reference model. `run_train` stays blocked for a full Tiny Aya optimization.
- Opening the sealed final test split, regenerating datasets from test failures, or writing historical adapters and prototype banks.
- Overwriting `results/rl_v2/grpo_cpu_smoke` or any existing adapter directory.
- Stopping, pausing, or sharing the card with the running SFT scoring job. One model-bearing GPU process at a time.
- Editing AgentTune or TRL. The CPU wheel was our install choice; this plan changes the environment and the fail-closed gates around it.

## Background / Context

Plan 10 (`docs/plan_10/OVERVIEW.md`) is the RL study. Its R0–R3 path is what this tree implements today: AgentTune wraps TRL 1.7.1 GRPO, the custom `AgenticRolloutAdapter` is the only rollout, and real optimization is refused until a frozen SFT manifest exists. After that check, `run_train` still raises because real GRPO is blocked and `beta_zero` is not an SFT reference. Plan 10 R4 is the later learning-mechanics and Tiny Aya GPU gate. This plan only builds the GPU environment those later tasks need.

The CPU environment is `C:\Users\arian\Github\Hackathons-worktrees\plan-10-rl-env`, Python 3.12.13, torch `2.11.0+cpu`. `docs/RL_BACKEND_COMPATIBILITY.md` and `requirements-rl.lock` record that stack. AgentTune source is `C:\Users\arian\Github\Hackathons-worktrees\plan-10-agenttune-source` at `d4db4c12d52342227c4113a70183d0f64aa2f345`. `validate_agenttune_source` accepts that git commit or an unmodified local checkout whose installed `agenttune` Python files match the checkout. The supervised env is a different interpreter, `C:\Users\arian\Github\Hackathons\lexsi\.venv`, CPython 3.12, torch `2.11.0+cu128`. Never use `C:\Python314\python.exe`.

The CPU gate is in three places. `actproto/integrations/agenttune_backend.py` `create_trainer` raises unless `use_cpu` is True and raises if `use_vllm` is set. `actproto/training/rl_config.py` requires `use_cpu: true`, `bf16: false`, `fp16: false`, `gradient_checkpointing: false`, `reference_policy: beta_zero`, and beta exactly 0. `actproto/training/grpo_run.py` `_fit` is the only `trainer.train()` path: a 1-layer GPT-2 LoRA (`c_attn`, `fan_in_fan_out=True`, constant scheduler, beta 0, `use_cpu=True`). A 2-step run of that smoke already succeeded at `results/rl_v2/grpo_cpu_smoke` (gitignored). The CLI is `lexsi/scripts/train_grpo.py`. The checked-in CPU config is `lexsi/actproto/configs/grpo_v2.yaml`. `actproto/training/grpo.py` remains a rejection stub. Its helpers must stay unsuccessful.

Device identity comes from PyTorch, not from `nvidia-smi` order. With `CUDA_VISIBLE_DEVICES` unset, PyTorch `cuda:0` is the RTX 4060 8 GB, UUID `5b885e12-623f-268c-f07c-ee33db379969`. PyTorch `cuda:1` is the RTX 5060 Ti, about 16 GB, UUID `134c34ef-c155-b001-4e3a-b9134b072c21`. `nvidia-smi` index 0 is the 5060 Ti. Those ordinals are different numbering systems. On this machine, `CUDA_VISIBLE_DEVICES=0` exposed the RTX 4060 and `CUDA_VISIBLE_DEVICES=1` exposed the RTX 5060 Ti, so the nvidia-smi index is not the mask value. A child sets the variable before importing torch and aborts unless that import still shows UUID `134c34ef-c155-b001-4e3a-b9134b072c21`. Never leave `CUDA_VISIBLE_DEVICES` set in a parent shell. `scripts/train_lora.py` must not be copied: it pins the largest PyTorch index.

Tiny Aya (`CohereLabs/tiny-aya-global`, Cohere2, revision `af89d219b53ed9b13b8a4645f9c8028973510324` in `actproto/configs/sft_v2.yaml`) is about 6.6 GB of bf16 weights before cache, with hidden size 2048, 36 layers, and vocab 262144. SFT training filled the 5060 Ti to about 15.8 of 16 GB. The 4060 cannot hold Tiny Aya GRPO. SFT memory usage is evidence that the card was full during SFT, and it is not a measurement of GRPO. The toy smoke and the Tiny Aya load are separate model-bearing jobs. Neither starts while the SFT scoring process is using the 5060 Ti. Read the long-running-background-tasks skill before launching or watching either job.

Hackathon scope: the code changes are small and fail-closed. The expensive part is two short, exclusive GPU runs after the card is idle. Critical path for code is G2 → G3, with G1 and G4 beside them, then G5 and G6. G5 and G6 may be written together and must be executed one at a time. G7 records the runs. `run_train` stays unavailable for the study after this plan.

## Features / Tasks

### G1: Isolated CUDA RL environment
**Files:** `requirements-rl-gpu.txt` (new), `requirements-rl-gpu.lock` (new), `docs/RL_GPU_ENV.md` (new)
**Complexity:** M
**Depends on:** none

Create `C:\Users\arian\Github\Hackathons-worktrees\plan-10-rl-gpu-env` with the CPU environment's Python 3.12.13 (`C:\Users\arian\Github\Hackathons-worktrees\plan-10-rl-env\Scripts\python.exe -m venv`). Do not create it inside the supervised tree, and do not install into either existing environment. `requirements-rl-gpu.txt` is the install recipe. Install `torch==2.11.0` from the cu128 index (`https://download.pytorch.org/whl/cu128`) before the rest of the AgentTune stack. Then install the same AgentTune commit and TRL 1.7.1 already pinned in `requirements-rl.lock`, plus the other pins that import needs, without letting pip replace the CUDA wheel. `requirements-rl.lock` line `torch==2.11.0` does not encode `+cu128`. After every install command, require `torch.__version__ == "2.11.0+cu128"` and `torch.cuda.is_available() is True`. If a later install leaves `2.11.0+cpu`, reinstall the cu128 wheel and check again. Install AgentTune from the pinned commit so `validate_agenttune_source` passes. Write `requirements-rl-gpu.lock` from that verified env. Leave `requirements-rl.lock` as the CPU lock.

`docs/RL_GPU_ENV.md` records both interpreter paths, the cu128 index, the AgentTune commit, and the command that prints the torch version and each device name plus UUID. It states that the CPU env remains the interpreter for the already-verified `results/rl_v2/grpo_cpu_smoke` mechanics run, and that new GRPO training uses the CUDA env. Verification results from G5 and G6 are recorded in `docs/RL_BACKEND_COMPATIBILITY.md`, not invented here.

**Acceptance:** The new interpreter imports AgentTune 1.1.3, TRL 1.7.1, and torch `2.11.0+cu128` with CUDA available. The supervised venv and the CPU RL venv still import as they did before this task. The recipe never calls `C:\Python314\python.exe`.

### G2: Fail-closed GPU config distinct from the CPU smoke
**Files:** `actproto/configs/grpo_v2_gpu.yaml` (new), `actproto/training/rl_config.py` (modify), `tests/test_grpo_config.py` (modify)
**Complexity:** M
**Depends on:** none

Add `actproto/configs/grpo_v2_gpu.yaml` with `schema_version: grpo_v2_gpu`. Required values: `reference_policy: beta_zero`, `beta: 0.0`, `use_cpu: false`, `use_vllm: false`, `bf16: true`, `fp16: false`, `gradient_checkpointing: true`, `report_to: none`, `device_name` containing `RTX 5060 Ti`, and `device_uuid: 134c34ef-c155-b001-4e3a-b9134b072c21`. Keep the CPU divisibility rule: `num_generations` at least 2, and `per_device_train_batch_size * gradient_accumulation_steps` divisible by `num_generations`. Start from the CPU file's learning rate `5.0e-6`, group size 2, batch 2, accumulation 1, temperature 0.8, top-p 0.95, and `max_completion_length` 256, with `max_steps: 1` and `seed: 42`. `output_dir` is `results/rl_v2/grpo_gpu_smoke`. `baseline_manifest` may name `docs/SFT_BASELINE_MANIFEST.json` and `train_manifest` stays empty. The loader must not treat a present manifest path as permission to load that adapter.

Extend `load_grpo_config` so schema `grpo_v2` keeps today's CPU requirements, including `use_cpu: true` and `gradient_checkpointing: false`. Schema `grpo_v2_gpu` accepts only the GPU requirements above. Flipping `use_cpu` in either file, setting beta away from 0, renaming `reference_policy`, setting `use_vllm: true`, targeting the 4060 UUID, or adding an unknown key raises `GrpoConfigError`. `public_dict` reports the schema's real `use_cpu` value. The CPU schema's public dict stays `use_cpu: true` and `reference_policy: beta_zero`. Output paths still have to resolve inside `results/rl_v2` and still cannot escape into adapter directories.

**Acceptance:** `tests/test_grpo_config.py` proves the checked-in CPU file is unchanged in meaning, the GPU file loads only as `grpo_v2_gpu`, and each cross-rewrite fails. No test starts a trainer.

### G3: Lift the CPU trainer gate only for the named GPU profile
**Files:** `actproto/integrations/agenttune_backend.py` (modify), `tests/test_agenttune_backend.py` (modify)
**Complexity:** S
**Depends on:** G2

`create_trainer` keeps the current rejection when `use_cpu` is omitted or false and `gpu_profile` is absent: "Only CPU trainer construction is currently verified". Accept `use_cpu: false` only when `gpu_profile` is exactly `grpo_v2_gpu`. `use_vllm` true still raises "vLLM is outside the verified baseline" for both profiles. Version, source-provenance, unknown-kwarg, dataset, reward, and `AgenticRolloutAdapter` checks stay as they are. The profile name is the G2 schema string. Importing it from `rl_config` is fine; `rl_config` must not import the backend.

Add tests that call the validator behavior without a GPU. A bare `use_cpu: false` still raises. `use_cpu: false` with `gpu_profile="grpo_v2_gpu"` passes the gate and still rejects `use_vllm: true`. `use_cpu: true` with the GPU profile raises, so the CPU smoke cannot be silently relabeled. Tests that need AgentTune stay skipped when the package is absent, matching the existing isolated-env skip.

**Acceptance:** The CPU construction path used by `_fit` and `scripts/check_rl_backend.py` still passes `use_cpu=True` and does not need `gpu_profile`. A GPU profile cannot enable vLLM.

### G4: Select the 5060 Ti by UUID
**Files:** `actproto/training/rl_device.py` (new), `tests/test_rl_device.py` (new), `scripts/check_rl_gpu_env.py` (new)
**Complexity:** S
**Depends on:** G1

Add a pure selector that takes device records of name, UUID, and total memory and returns the index whose UUID is `134c34ef-c155-b001-4e3a-b9134b072c21`. It raises if that UUID is missing, if the chosen name does not identify an RTX 5060 Ti, or if the 4060 UUID `5b885e12-623f-268c-f07c-ee33db379969` is the selected device. It must not choose "largest memory" or `nvidia-smi` index 0. Unit tests cover the known two-card map, a swapped-index map, a 4060-only map, and a UUID whose name is wrong.

`scripts/check_rl_gpu_env.py` runs in the G1 interpreter. It refuses to continue when the parent process already has `CUDA_VISIBLE_DEVICES` set. It prints torch version, CUDA availability, and each device's PyTorch index, name, and UUID, then exits nonzero unless version `2.11.0+cu128` is active and the selector returns the 5060 Ti. This script loads no model weights. It is not a license to start training. The live check is part of G1's env being finished; the unit tests do not need a GPU.

**Acceptance:** `tests/test_rl_device.py` passes on the supervised interpreter without touching a GPU. The live script, run only in the CUDA env with the parent variable unset, prints the 5060 Ti UUID and exits 0. An already-set parent `CUDA_VISIBLE_DEVICES` exits nonzero and leaves that parent value unchanged.

### G5: Smallest-safe CUDA LoRA smoke
**Files:** `actproto/training/grpo_run.py` (modify), `scripts/train_grpo.py` (modify), `tests/test_train_grpo.py` (modify)
**Complexity:** L
**Depends on:** G1, G2, G3, G4

Dispatch `--smoke` by schema. `grpo_v2` keeps `run_smoke` / `_fit` on CPU: `use_cpu=True`, fp32 flags, `fan_in_fan_out` on `c_attn`, constant scheduler, beta 0, no reference model. `grpo_v2_gpu` calls a new `run_gpu_smoke` and must not enter `_fit`. `--resume` with the GPU schema fails closed. `--train` fails closed for both schemas; a GPU config does not open Tiny Aya optimization. `--plan-only` still builds nothing and writes no completed marker.

`run_gpu_smoke` uses the G1 interpreter's CUDA torch, the G4 selector, and `create_trainer(..., use_cpu=False, gpu_profile="grpo_v2_gpu", use_vllm=False, beta=0.0)`. The model is the same 1-layer GPT-2 fixture `_fit` already trains, with the same Tiny Aya tokenizer via `local_files_only` and the same injected calculator-then-finish policy. It loads no Tiny Aya weights and no SFT adapter. `baseline_loaded` stays false, and `ref_model` stays absent. One optimizer step is enough. Require a finite nonzero gradient, LoRA-only optimizer parameters, frozen weights unchanged, and a reload that matches the saved LoRA. Pass `bf16=True` and `gradient_checkpointing=True` from the GPU config. If the toy GPT-2 rejects bf16, stop and record that error in the smoke payload's failure path. An fp32 CUDA retry is allowed only inside `run_gpu_smoke`, only after that recorded failure, and the completed payload must say the toy update dtype was fp32 while the named Tiny Aya profile is still bf16 and unproven. Never retry on CPU.

Write only under the resolved `results/rl_v2/grpo_gpu_smoke` directory. Refuse an output that already exists, that overlaps `grpo_cpu_smoke`, or that could replace an adapter. The completed payload names `mode: gpu-smoke`, `schema_version: grpo_v2_gpu`, `reference_policy: beta_zero`, the device UUID, and `study_claimed: false`.

The live command runs only after the SFT scoring job is off the 5060 Ti, and only after G6 is not already loading a model. Read the long-running-background-tasks skill before launching it. Do not export `CUDA_VISIBLE_DEVICES` in the parent. If the child masks devices, it tries candidate ordinals in a fresh process and keeps the one whose reimported torch device still has the 5060 Ti UUID. Default pytest must not launch this job. Cover dispatch and refusal in `tests/test_train_grpo.py` without CUDA. Gate the live smoke on an explicit environment flag or on the CLI, and skip it when AgentTune or the 5060 Ti is absent.

**Acceptance:** CPU `--smoke` behavior and the existing resume test still pass in the CPU env. GPU `--train` exits 2. One live CLI smoke on the idle 5060 Ti finishes one finite LoRA update, reloads, and leaves `grpo_cpu_smoke` untouched. The payload does not say the frozen SFT checkpoint was the reference.

### G6: Measure Tiny Aya memory on a free 5060 Ti
**Files:** `scripts/measure_rl_gpu_memory.py` (new), `tests/test_rl_gpu_memory.py` (new)
**Complexity:** M
**Depends on:** G1, G4

Add a measurement script that loads `CohereLabs/tiny-aya-global` at the SFT revision in bf16 on the G4-selected 5060 Ti and writes `results/rl_v2/gpu_memory/tiny_aya_bf16.json`. Record device name, UUID, PyTorch index, torch version, weight dtype, allocated bytes, reserved bytes, and peak allocated bytes after the load. Optionally run one forward or a single generated token if the load itself leaves headroom, and record that peak separately. Attach no historical adapter and start no optimizer. Read LoRA rank and alpha from `actproto/configs/sft_v2.yaml` only as a note in the JSON. Do not guess Tiny Aya target modules and do not write an adapter.

Before the load, exit nonzero if the parent has `CUDA_VISIBLE_DEVICES` set, if the selector would choose the 4060, or if the 5060 Ti already has another compute process. Check occupancy with a read-only query. Do not kill the SFT scorer. This script is model-bearing, so it runs only while G5's smoke is not running and only after the SFT job has released the card. Read the long-running-background-tasks skill first. The JSON is gitignored under `results/` and is the source for G7. Missing headroom is a successful measurement with `grpo_fit_claimed: false`. It is not a failure to paper over with the 4060, with vLLM, or with quantization.

`tests/test_rl_gpu_memory.py` checks argument refusal and the output schema with a fake device table and a fake memory snapshot. The test does not download weights or import a CUDA context.

**Acceptance:** Unit tests pass without a GPU. One live run on the idle 5060 Ti writes the JSON with the expected UUID and real byte counts. No adapter directory and no sealed split file changes. The script has no path that calls `trainer.train`.

### G7: Record the verified GPU path
**Files:** `docs/RL_BACKEND_COMPATIBILITY.md` (modify)
**Complexity:** S
**Depends on:** G1, G5, G6

Add a GPU section to `docs/RL_BACKEND_COMPATIBILITY.md`. Include the CUDA interpreter path, torch `2.11.0+cu128`, the unchanged AgentTune commit and TRL pin, the PyTorch-versus-`nvidia-smi` map, and the rule that the parent shell does not keep `CUDA_VISIBLE_DEVICES`. Quote the G5 payload fields that were actually written: step count, dtype, device UUID, `baseline_loaded: false`, and `reference_policy: beta_zero`. Quote the G6 byte counts from `results/rl_v2/gpu_memory/tiny_aya_bf16.json`. State that the toy smoke is a CUDA LoRA mechanics check on a 1-layer GPT-2 fixture, and that the memory JSON is a load measurement. State that Tiny Aya GRPO, a frozen SFT reference, vLLM, and Plan 10 R4–R7 remain unverified. If the card was busy and a live run did not happen, write that the run is not done and leave the numbers out. Do not reuse the SFT 15.8 GB figure as the GRPO measurement.

Keep the existing CPU fixture section. Point operators who need the old smoke at the CPU env and `results/rl_v2/grpo_cpu_smoke`.

**Acceptance:** A reader can tell which interpreter to use, which card UUID was required, what the smoke updated, and which claims this plan refuses to make. Every number in the new section appears in the G5 payload or the G6 JSON.

## New Dependencies

| Package | Feature | Optional? |
| --- | --- | --- |
| torch 2.11.0+cu128, installed from the cu128 wheel index into `plan-10-rl-gpu-env` | CUDA GRPO smoke and Tiny Aya memory probe | Required for the GPU env. Forbidden in the supervised venv and in the CPU RL venv |
| torchvision / torchaudio wheels from the same cu128 index | Only if another pinned package imports them | Optional. Install the matching cu128 build if an import needs them, then re-check that torch is still `2.11.0+cu128` |
| AgentTune 1.1.3 at `d4db4c12d52342227c4113a70183d0f64aa2f345` | Same trainer factory as the CPU env | Required. Same pin, new env |
| trl 1.7.1 | Same GRPO trainer as the CPU env | Required. Same pin, new env |
| vLLM | None in this plan | Not required |

`requirements-rl.lock` stays the CPU lock. `requirements-rl-gpu.lock` is the new env's freeze after the CUDA wheel has been verified.

## File Change Summary

| File | Action |
| --- | --- |
| `requirements-rl-gpu.txt` | new |
| `requirements-rl-gpu.lock` | new |
| `docs/RL_GPU_ENV.md` | new |
| `actproto/configs/grpo_v2_gpu.yaml` | new |
| `actproto/training/rl_config.py` | modify |
| `tests/test_grpo_config.py` | modify |
| `actproto/integrations/agenttune_backend.py` | modify |
| `tests/test_agenttune_backend.py` | modify |
| `actproto/training/rl_device.py` | new |
| `tests/test_rl_device.py` | new |
| `scripts/check_rl_gpu_env.py` | new |
| `actproto/training/grpo_run.py` | modify |
| `scripts/train_grpo.py` | modify |
| `tests/test_train_grpo.py` | modify |
| `scripts/measure_rl_gpu_memory.py` | new |
| `tests/test_rl_gpu_memory.py` | new |
| `docs/RL_BACKEND_COMPATIBILITY.md` | modify |

## Open Questions

- The toy GPT-2 smoke may reject `bf16=True`. G5 allows one recorded fp32 CUDA retry inside the smoke function. If that also fails, stop. Do not move the named GPU profile back to CPU and do not treat a CPU retry as the GPU result.
- G6 can show that Tiny Aya bf16 weights leave too little room for GRPO generation and a reference model. That result blocks a Tiny Aya GRPO claim. Quantization, a CPU reference, or beta 0 for the study are Plan 10 decisions made after this measurement, and this plan does not choose them.
- The SFT scoring job may still own the 5060 Ti when implementation starts. G1–G4 and the unit tests can land while it runs. G5 and G6 wait. Do not stop that job to finish the plan.
- SFT LoRA target-module names for Tiny Aya are not in `actproto/configs/sft_v2.yaml`. G6 therefore loads the base weights only. Choosing modules and attaching the frozen adapter is later study work, and it is the point at which a task could first prove an SFT reference. This plan does not do that.
- A child-process mask value is known only at runtime. nvidia-smi index 0 is the 5060 Ti, but on this machine `CUDA_VISIBLE_DEVICES=0` exposed the 4060 and `CUDA_VISIBLE_DEVICES=1` kept the 5060 Ti UUID. Hard-coding either value in a parent shell is outside the plan. The child rechecks the UUID after the mask.
