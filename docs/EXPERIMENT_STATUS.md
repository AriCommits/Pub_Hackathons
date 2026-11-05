# Experiment status (resume here)

Last updated: 2026-11-02, ~19:56 America/Chicago. B2 live eval is still running (started 18:59). Do not start another train or eval, and do not kill the B2 processes.

This file is the handoff for the first trial set (B0, B1, B2). Another agent should read it before touching the machine. Update the "Last updated" line and the checklist at the bottom whenever a step finishes or fails.

## Goal

Run the documented B0–B2 pipeline in `lexsi/` and write reports under `results/b0_b2/`.

```text
python scripts/generate_dataset.py
python scripts/train_lora.py            # real GPU train; --plan-only only prints the plan
python scripts/extract_activations.py   # --plan-only skips weights
python scripts/build_prototypes.py
python scripts/train_probe.py
python scripts/run_trial.py --run all   # B0, B1, B2 live eval
python scripts/compare_results.py
```

Paths and run definitions: `actproto/configs/trial.yaml`. Narrative: `docs/METHOD.md`, `docs/TRIAL_CONFIG.md`.

## Repo and shell

- Project root: `C:\Users\arian\Github\Hackathons\lexsi`
- Parent checkout: `C:\Users\arian\Github\Hackathons` (README there points at `lexsi/`). The Cursor workspace path `...\Hackathons\hackathons` does not exist. Set the shell working directory to `lexsi` explicitly.
- Use the project venv. Do not use `C:\Python314\python.exe`. That interpreter has a broken `torch` install (`import torch` succeeds, `torch.__version__` and `torch.manual_seed` do not). `actproto/utils/seed.py` imports torch when it is present, so system Python crashes in `generate_dataset.py`.

## Environment already created

```text
uv 0.13.5
.venv  ->  CPython 3.12.13  (matches .python-version)
torch 2.11.0+cu128
transformers 5.22.1
peft 0.21.2
trl 1.17.1
datasets 5.1.0
accelerate 1.15.0
huggingface_hub 2.4.0
pytest 9.2.0
```

Install command that was used (do not reinstall unless the venv is gone):

```text
uv venv --python 3.12 .venv
uv pip install --python .venv -e ".[dev,train]" --torch-backend cu128
```

`bitsandbytes` is **not** installed. It lives in the optional `qlora` extra. `actproto/configs/sft.yaml` sets `load_in_4bit: true`, but nothing calls that flag yet.

`.venv/` is gitignored.

## GPUs

`nvidia-smi` order and PyTorch order disagree. Trust PyTorch:

| PyTorch index | Card | Capability | Memory |
| --- | --- | --- | --- |
| `cuda:0` | NVIDIA GeForce RTX 4060 | 8.9 | 8 GB |
| `cuda:1` | NVIDIA GeForce RTX 5060 Ti | 12.0 | 15.9 GB |

Put the model on `cuda:1` (the 16 GB card). `cuda:0` is too small for bf16 weights plus activations.

## Hugging Face access

- Account that owns the local token: **ArianMoridani** (`~/.cache/huggingface/token`). Do not print or copy the token.
- `HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN` are unset in the shell. The hub still finds the token file.
- Repo: `CohereLabs/tiny-aya-global` (gated).
- Access was denied (403, not on the authorized list) until the user accepted the license.
- After acceptance, `config.json` downloaded successfully.
- Snapshot: `af89d219b53ed9b13b8a4645f9c8028973510324`
- Cache dir: `C:\Users\arian\.cache\huggingface\hub\models--CohereLabs--tiny-aya-global`
- Weights downloaded (snapshot_download exit 0, ~2 min). Do not download again.
  - `model-00001-of-00002.safetensors` 4761 MB
  - `model-00002-of-00002.safetensors` 1627 MB
  - `tokenizer.json` 20 MB
  - plus `model.safetensors.index.json`, `tokenizer_config.json`, `generation_config.json`
- Windows cache cannot use symlinks here (no Developer Mode). Hub warned that caching is degraded and uses more disk. Harmless.
- Record this revision in `actproto/configs/model.yaml` (`revision` is still `null`) once a run actually loads it.

Model shape from `config.json`:

- `Cohere2ForCausalLM`, `model_type: cohere2`
- 36 layers, hidden 2048, intermediate 11008, 16 heads / 4 KV heads, vocab 262144
- `torch_dtype: bfloat16`, `use_embedding_sharing: true`
- Roughly a 3B-class model. Confirmed load: **3.349B** parameters, **6.24 GB** allocated in bf16 on the 5060 Ti. Leaves ~10 GB headroom on that card. Will not fit on the 8 GB 4060.

## Data

`python scripts/generate_dataset.py` already ran with the venv (seed 42).

| Split | Tasks |
| --- | --- |
| train | 288 |
| val | 104 |
| test | 104 |

Buckets: easy 271, confusable 96, multi_step 24, no_tool 45, error_recovery 60.

Files (gitignored): `lexsi/data/processed/{train,val,test}.jsonl` and `manifest.json`. Eval does **not** read those files. `actproto/evaluation/task_loader.py` calls `generate_tasks()` again and filters by split. Regenerating with the same seed is deterministic.

## What already ran

- Offline tests: `lexsi\.venv\Scripts\python -m pytest` → **51 passed** (19.6s).
- Plan-only training payload: 288 rows, LoRA r=16, alpha=32, dropout=0.05, targets `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj`, output `results/adapters/b1`. AgentTune is not installed, so the plan falls back to `peft_trl`.
- Plan-only activation extract: split `train`, 288 tasks, out `results/activations`, adapter `results/adapters/b1`.
- No live eval, no adapter, no activation dump, no prototypes, no probe, no `results/b0_b2/{B0,B1,B2}` reports.

## Code gaps that block a real run

1. `scripts/train_lora.py` without `--plan-only` prints "wire TRL/PEFT trainer here" and **returns exit code 2**. It does not train. B1 and B2 need `results/adapters/b1`. Tests do not cover this script. `tests/test_sft_dataset.py` only checks row shape and `build_train_payload`.
2. `actproto/agent/model.py` `load_model` does not set `device_map` and does not move the model to CUDA. `generate_text` sends inputs to `model.device`, which will be CPU. Live eval of 104 test tasks × up to 6 steps on CPU is not a practical run. Load on the 5060 Ti.
   - Smoke test that worked (2026-11-02): set `CUDA_VISIBLE_DEVICES=1` so the 5060 Ti is PyTorch `cuda:0`, then `AutoModelForCausalLM.from_pretrained(..., dtype=torch.bfloat16, device_map="cuda")`. `model.device` was `cuda:0`, card name RTX 5060 Ti.
   - Without that env var, pass `device_map={"": "cuda:1"}`.
   - This transformers build (5.22.1) accepted `dtype=`. `load_model` still passes `torch_dtype=`. Confirm that keyword still works before a long run; switch it to `dtype` if `from_pretrained` raises `TypeError`.
3. `extract_activations.py` uses the same `load_model`. Same GPU placement issue.
4. SFT rows from `actproto/training/sft_dataset.py` start at the user turn. Live eval prepends `system_prompt()` from `actproto/agent/prompts.py`. A trainer should train with that same system prompt or B1 will not match the eval format. Do not change the tested row schema (`messages`, `segment_weights`, `loss_mask`). Prepend the system message inside the trainer only.
5. Loss is assistant-only: `loss_mask` is true (masked out) exactly when `segment_weights` is 0. Assistant turns have weight 2.0 and `loss_mask` false.
6. `sft.yaml` asks for 4-bit, but bitsandbytes is absent. bf16 LoRA on `cuda:1` with gradient checkpointing is the path that does not need a new package. Batch size 1 and grad accumulation 8 are already in the yaml. 288 rows × 2 epochs is a small run.
7. TRL is 1.17.1. Use this API, not the old 0.9 one:
   - `SFTTrainer(model=<hub id or module>, args=SFTConfig(...), train_dataset=..., peft_config=LoraConfig(...))`
   - `SFTConfig` subclasses the training args. Relevant fields: `output_dir`, `learning_rate`, `num_train_epochs`, `per_device_train_batch_size`, `gradient_accumulation_steps`, `max_length` (default 1024; yaml cap is 2048), `assistant_only_loss=True` for a conversational dataset whose rows are `{messages: [{role, content}, ...]}`, `bf16=True`, `gradient_checkpointing=True` (already the SFTConfig default).
   - `completion_only_loss` is for prompt-completion datasets, not this chat format.
   - Pass `model_init_kwargs={"dtype": "bfloat16", "device_map": ...}` when `model` is the hub id string.
   - Reference script in the installed package: `.venv/Lib/site-packages/trl/scripts/sft.py`.
   - After `trainer.train()`, `trainer.save_model(output_dir)` writes a PEFT adapter.

## Resume checklist

Work from `C:\Users\arian\Github\Hackathons\lexsi`. Interpreter: `.venv\Scripts\python`.

- [x] uv venv + train extras + CUDA torch
- [x] Generate `data/processed`
- [x] pytest (51 passed)
- [x] HF license accepted; `config.json` cached
- [x] Download the rest of `CohereLabs/tiny-aya-global` (both safetensors shards + tokenizer)
- [x] Confirm the model loads on the 5060 Ti (bf16, 3.349B params, 6.24 GB). This was a direct `from_pretrained`, not `actproto.agent.model.load_model`.
- [ ] Place the loaded model on GPU inside `load_model` (or an equivalent call site)
- [ ] Wire real LoRA SFT in `scripts/train_lora.py` / `actproto/training/sft.py` for the non-`--plan-only` path
- [ ] Train adapter to `results/adapters/b1`
- [ ] `extract_activations.py` (no `--plan-only`) → `results/activations`
- [ ] `build_prototypes.py` → `results/prototypes.npz` and `results/prototype_metrics.json`
- [ ] `train_probe.py` → `results/linear_probe.joblib` and probe metrics
- [ ] `run_trial.py --run all` → `results/b0_b2/{B0,B1,B2}/eval.json`
- [ ] `compare_results.py` → `results/b0_b2/comparison.md`
- [ ] Write the revision hash into `actproto/configs/model.yaml` if a real load happened

Weights are cached and a direct bf16 load on the 5060 Ti succeeded. Next code change is GPU placement inside `load_model`, then wire LoRA SFT. Smoke command that already worked:

```text
$env:CUDA_VISIBLE_DEVICES = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
.venv\Scripts\python -c "import torch; from transformers import AutoModelForCausalLM; m=AutoModelForCausalLM.from_pretrained('CohereLabs/tiny-aya-global', dtype=torch.bfloat16, device_map='cuda'); print(m.device, torch.cuda.get_device_name(0), round(sum(p.numel() for p in m.parameters())/1e9, 3))"
```

With `CUDA_VISIBLE_DEVICES=1`, PyTorch's `cuda:0` is the 5060 Ti. Without that variable, pass `device_map={'': 'cuda:1'}`.



## 2026-11-02 evening session

Code now in the tree:

- ctproto/agent/model.py loads with dtype and device_map on the largest GPU unless a device is passed. generate_text uses messages_for_template.
- ctproto/agent/prompts.py messages_for_template turns tool results into user turns so the Cohere chat template does not reject them.
- ctproto/training/lora_run.py is the real TRL/PEFT LoRA trainer. scripts/train_lora.py calls it (still --plan-only for a dry run). It pins CUDA_VISIBLE_DEVICES to the GPU with the most memory before importing torch.
- ctproto/evaluation/live_runner.py shards eval across resident bf16 copies. ctproto/evaluation/shard_worker.py sets CUDA_VISIBLE_DEVICES inside each process before torch import.
- scripts/extract_activations.py --split all dumps train+val+test. Use ll. Train-only would make the probe's test score meaningless.
- ctproto/configs/model.yaml revision is f89d219b53ed9b13b8a4645f9c8028973510324.
- Activation hooks walk PEFT wrappers (ase_model / model) so B2 and extraction can see Cohere layers.
- pytest: 51 passed after these edits.

Replica budget (bf16, about 6.24 GB weights, planner uses 6.35 GB plus 0.30 GB reserve, max 2 per GPU):

- PyTorch cuda:0 RTX 4060, 6.94 GB free -> 1 replica
- PyTorch cuda:1 RTX 5060 Ti, 14.82 GB free -> 2 replicas
- Plan: ['0', '1', '1'] (3 copies). The 4060 is tight. If that worker OOMs, rerun eval with ACTPROTO_CUDA_DEVICES=1 (two copies on the 5060 Ti only).

Do not leave CUDA_VISIBLE_DEVICES set in the shell from an earlier command. Training sets it only inside its own process. A leftover value hides the 4060.

Training command already launched:

`	ext
.venv\Scripts\python scripts/train_lora.py
`

It writes 
esults/adapters/b1. After it exits 0:

`	ext
.venv\Scripts\python scripts/extract_activations.py --split all
.venv\Scripts\python scripts/build_prototypes.py --eval-test
.venv\Scripts\python scripts/train_probe.py
.venv\Scripts\python scripts/run_trial.py --run all
.venv\Scripts\python scripts/compare_results.py
`


un_trial.py --run all runs B0, then B1, then B2, each with the 3-replica pool (104 test tasks). B1/B2 need the adapter. B2 also needs 
esults/prototypes.npz.


Training progress (do not launch another one): pinned to CUDA index 1 (RTX 5060 Ti). 288 rows, seq max 957, max_length 960, 72 optimizer steps, about 14 s/step after the first. assistant_only was False because the chat template string did not contain the exact marker needle, so the loss is on the full sequence. Adapter target remains results/adapters/b1.


B1 training finished exit 0 in about 9 minutes. train_loss 0.305, mean_token_accuracy 0.9945. Adapter: results/adapters/b1/adapter_model.safetensors (121 MB). Next step that was started after that: extract_activations.py --split all.


Activation extract finished exit 0 in about 2.5 minutes: 496 tasks (train+val+test) at results/activations.
Prototypes: 5 tools, test accuracy 1.0, threshold min_score -0.2, min_margin 0.0, coverage 1.0 (val n=104). Probe val and test accuracy 1.0. Files: results/prototypes.npz, results/prototype_metrics.json, results/linear_probe.joblib, results/probe_metrics.json.
Live eval started next: scripts/run_trial.py --run all (B0, B1, B2), 104 test tasks, replica plan 1x RTX 4060 + 2x RTX 5060 Ti. Do not start a second run_trial.py.


Live eval relaunched after fixing scripts/run_trial.py (it did not put the lexsi root on sys.path, so the first launch died with ModuleNotFoundError: scripts). Second launch is running: B0, 104 tasks, replicas [0, 1, 1] (4060 x1, 5060 Ti x2). Then B1, then B2, then compare_results.py still needs to be run.


B0 and B1 live eval finished. B0: verified success 0.000, tool_call_f1 0.880 (n=104). B1: verified success 0.010, tool_call_f1 0.696 (n=104). Reports: results/b0_b2/B0 and results/b0_b2/B1.
B2 died on load_prototypes_npz: Path was not imported. Fixed in actproto/activations/prototypes.py. B2 rerun is in progress (3 replicas). Then compare_results.py.

Status check ~19:56: B2 rerun is still running (parent pid 3140, started 18:59:34, plus three shard workers). No results/b0_b2/B2 yet. Terminal file 236924.txt is gone, so B2 gpu= lines cannot be counted. RTX 5060 Ti was at about 54% util and 15.7 GB (two workers still using CPU). The third worker (pid 22068) used 0 CPU over an 8s sample, and the RTX 4060 sat at 0% util with 7.9 GB still held, so that replica looks stalled while the other two continue. Do not kill or restart. Parallel-plan files (OVERVIEW.md) are not in the repo. Planner transcript b9932af6 last wrote at 19:47 and has no finished summary.


B2 rerun stalled: the RTX 4060 worker (pid 22068) held 7.9 GB at 0% GPU and 0 CPU while the two 5060 Ti workers kept going. Pool.map would wait forever. Killed that process tree and restarted B2 with ACTPROTO_CUDA_DEVICES=1 (5060 Ti only, up to 2 replicas).

## Do not

- Do not commit `.env`, tokens, `data/processed/`, or `results/**` (gitignored except `results/.gitkeep` and `results/b0_b2/README.txt`).
- Do not use system Python 3.14.
- Do not start a full 104-task live eval on CPU.
- Do not treat `--plan-only` as a finished training run.

## B2 depth-1 follow-up

- `B2d1` completed on CUDA device 1 with one replica (`n=104`): first-tool accuracy **1.000** (matches saved B2 **1.000**), first-argument correctness **0.885**, verified task success **0.000**, and median latency **31.802 s**. Reports: `results/b0_b2/B2d1/eval.json` and `eval.md`.

- `B2d1r` completed on CUDA device 1 with one replica (`n=104`): verified task success **68/104 (0.654)**, first-tool accuracy **1.000** (matches saved B2 **1.000**), first-argument correctness **0.625**, `multiple tool calls` parse errors **0**, median latency **3.883 s**. The other 36 failures are 35 answer mismatches and 1 `done: expected True got None`. Reports: `results/b0_b2/B2d1r/eval.json` and `eval.md`.

- `B2chain` completed on CUDA device 1 with one replica (`n=104`): verified task success **36/104 (0.346)**, prefix tool choice **0.639**, first-tool accuracy **1.000**, repeated calls **0.019**, mean steps **2.66**, median latency **7.803 s**. Of the 6 length-3 tasks, hop 2 differs from hop 1 on **6/6** and matches gold on **0/6** (gold hop 2 is `calculator`; predicted hop 2 is `finish` or `lookup_record`). Reports: `results/b0_b2/B2chain/eval.json` and `eval.md`.

- `B2d1log` completed on CUDA device 1 with one replica (`n=104`). Of 39 mismatched `query_database` calls, **21** are `wrong_value`, **15** are `missing_field` (a blank `region: ""` that the tool treats as a filter), and **3** are `unfilled`. The next experiment is an assistant-only adapter. This plan does not train it. Report: `docs/plan_4/query_arg_report.md`.

- `B2d1a` trained `results/adapters/b1_assistant_only` (assistant-only loss, 2 epochs, train loss 0.01047, mean token accuracy 1.0 on the masked tokens) and evaluated it on the depth-1 episode. Verified success **36/104**, first-tool accuracy **0.317** (saved B2 and B2d1r are **1.000**), first-argument correctness **0.317**, `wrong_value` **39**, `missing_field` **0**. The router now emits `lookup_record` for the database tasks. The prototypes still come from the old adapter, so this run does not replace `results/adapters/b1`. Report: `docs/plan_5/query_arg_report.md`.

- `B2d1ap` refit prototypes on `results/activations_b1a` (assistant-only adapter, train fit, val threshold) into `results/prototypes_b1a.npz`. Prototype test accuracy **0.981**. Live eval on CUDA device 1, one replica (`n=104`): verified success **98/104 (0.942)**, first-tool accuracy **0.981** (102/104; saved B2 is **1.000**, the two misses are `error_missing_product-4` and `-p1`), first-argument correctness **0.981**, median latency **3.736 s**. The 6 failures are the length-3 `multi_growth` tasks: hop 1 `query_database` and hop 2 `calculator` match gold, then the step cap stops before `finish`. `results/prototypes.npz` and `results/adapters/b1` were not replaced. Reports: `results/b0_b2/B2d1ap/eval.json` and `eval.md`. Plan: `docs/plan_6/OVERVIEW.md`.

- `B2apchain` completed on CUDA device 1 with one replica (`n=104`): the same adapter and prototypes as `B2d1ap`, router on hop 1 only, `max_agent_steps` 4. Verified success **104/104**, first-tool accuracy **0.981**, first-argument correctness **0.981**, tool-call F1 **0.990**, mean steps **1.97**, repeated calls **0**, median latency **3.770 s**. All 6 length-3 `multi_growth` tasks are `query_database` then `calculator` then `finish`, and each predicted call matches gold arguments. The two first-tool misses (`error_missing_product-4` and `-p1`) still succeed via `lookup_record` then `finish`. Reports: `results/b0_b2/B2apchain/eval.json` and `eval.md`.

- Simplification assumption, 2026-11-04: the organizer does not publish a chain longer than the local gold maximum of 3 calls. Plan later episodes for a 12-step budget in which each tool is used at most twice in one session. Experimentation stops here.

- `B2combo` scored 38 held-out orders on the paired adapter, router on hop 1 only, `max_agent_steps` 12. The set is every 2-tool and 3-tool permutation, one chain that uses each work tool twice (9 calls), and one 12-call chain. Five tool names cannot fill 12 calls at two uses each, so `combo-12` uses calculator, search_docs, and query_database a third time. Tool-name sequences matched gold on **11/38**. One of those also copied the scripted finish string (`combo-query_database-calculator-search_docs`). The other 10 did the right tools and arguments, then finished with the tool's real result (`77`, `8470`, `widget`) instead of the task id. Verified success **4/38** counts three early `search_docs` calls that copied the task id into the answer. `combo-twice` and `combo-12` both stopped after 2 calls. Reports: `results/b0_b2/B2combo/eval.json`.

- `B2orders` trained `results/adapters/b1_orders` on the original 288 train rows plus 152 curriculum orders (assistant-only, 110 steps, train loss 0.00806). Prototypes fit on the original prompts: test accuracy **0.990**, files `results/prototypes_b1_orders.npz`. Original test, CUDA device 1, one replica: verified success **103/104 (0.990)**, first-tool accuracy **0.990**, first-argument correctness **0.971**, median latency **3.756 s**. The one failure is `error_missing_product-4-p0`. `B2ordereval` on one copy of each curriculum order: verified success **11/38**, exact tool-name sequences **9/38** (3 of 12 length-3, 6 of 24 length-4, 0 of the 9-call and 12-call chains). Those two long tasks used the full step budget and did not match the scripted order. `results/adapters/b1` and `results/prototypes.npz` were not replaced. Reports: `results/b0_b2/B2orders/` and `results/b0_b2/B2ordereval/`. Plan: `docs/plan_7/OVERVIEW.md`.

- `B2fmt` trained `results/adapters/b1_formats` on 820 rows (assistant-only, 206 steps, train loss 0.007327). Prototypes were fit on the original prompts plus one copy of every format, so prototype test accuracy fell to **0.856**. Original test: verified success **88/104 (0.846)**, first-tool accuracy **0.856**, first-argument correctness **0.798**, median latency **3.836 s**. `B2fmteval` on 266 format prompts: verified success **90/266 (0.338)**, first-tool accuracy **0.293**. By layout, first-tool accuracy is sequence **9/38**, header-first **36/114**, header-last **33/114**. Putting the instruction on the last line did not beat putting it on the first line. `b1`, `b1_orders`, and `results/prototypes.npz` were not replaced. Reports: `results/b0_b2/B2fmt/` and `results/b0_b2/B2fmteval/`. Plan: `docs/plan_8/OVERVIEW.md`.

- SFT v2 validation study, 2026-11-05: one epoch, router off, 200 validation tasks. `ebase` macro **0.000**. Legacy `e0` (`b1_assistant_only`) macro **0.253**. New adapter `e2` (`results/sft_v2/e2/checkpoint-150`, scored in `results/sft_v2/e2-scored`) macro **1.000** on every stratum, argument correctness **1.0**, invalid calls **0**, 570 tool calls. Selected `e2`, gain **+0.747** versus `e0`, paired group interval **0.72 to 0.835**. `e1` and `e3` were not run. The sealed test split was not opened. Frozen record: `docs/SFT_BASELINE_MANIFEST.json`.

- CUDA GRPO fixture smoke, 2026-11-05: one bf16 LoRA step on the 1-layer GPT-2 fixture, device UUID `134c34ef-c155-b001-4e3a-b9134b072c21`, global_step 1, baseline_loaded false, reference_policy beta_zero. Payload: `results/rl_v2/grpo_gpu_smoke/completed.json`. This is not a Tiny Aya GRPO measurement.

- Tiny Aya bf16 load, 2026-11-05: RTX 5060 Ti UUID `134c34ef-c155-b001-4e3a-b9134b072c21`, allocated_bytes 6820222464, reserved_bytes 6834618368, peak_allocated_bytes 6821362688, forward_peak_allocated_bytes 6821362688. `grpo_fit_claimed` is false. Report: `results/rl_v2/gpu_memory/tiny_aya_bf16.json`.
