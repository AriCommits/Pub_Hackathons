# Track 3 submission checklist

- Track: Agentic Fine-tuning (Lexsi × Cohere Labs × Hugging Face).
- Repo chain: AgentTune-compatible `Project` / `DictToolHarness` / `EventLog` plus AuditKIT-compatible `AgentCase` / `FinalStateAssertion` / recorded eval. Local stand-ins run when those packages are not installed; the APIs match the public examples.
- Model: `CohereLabs/tiny-aya-global` (gated). Record revision in `actproto/configs/model.yaml`.
- Before/after metrics (≥2): verified task success, tool-call F1 / validity, unnecessary calls, latency. See `scripts/compare_results.py`.
- Public code: this repository (`lexsi/`).
- Hugging Face artifacts: adapters under `results/adapters/b1`, dataset JSONL under `data/processed`, logs under `results/b0_b2`. Upload with `python scripts/export_hf_artifacts.py --repo <id>` after a real train.
- Write-up: `docs/METHOD.md` (metrics, rationale, negative-result language).
- Deadline: 14 November 2026. Stretch (B3 steering, B4 alignment loss) remains optional.
- GRPO: implemented via `scripts/train_grpo.py` (see `STATUS.md`). Historical `grpo_gpu_tools` / `grpo_gpu_study` were false negatives — every step `template_mismatch` / reward −0.1 because sampling ran under `model.train()` + gradient checkpointing. Fix: eval-mode generate in `_sample_peft_action`. Post-fix study `results/rl_v2/grpo_gpu_study_fix` shows `completion_reason=finish` and reward **1.0** (groups still often tied because e2 already solves many train prompts).
- Submission science claim: frozen SFT e2 remains the primary verified baseline; GRPO stack is now a working multi-turn tool trainer (no longer stuck at −0.1).
