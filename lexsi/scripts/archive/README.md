# Archived scripts

Not on the Live LoRA / SFT-v2 / GRPO submission path. Kept for reproducibility of older experiments.

| Script | Why archived |
| --- | --- |
| `train_aligned.py` | B4 alignment-loss stub; never trains. Use `train_lora.py` / `train_sft_v2.py`. |
| `run_order_eval.py` | `B2ordereval` train-echo diagnostic (prompts overlap curriculum). |
| `run_format_eval.py` | `B2fmteval` train-echo diagnostic (prompts overlap curriculum). |

Run from repo root with `python scripts/archive/<name>.py` if needed. Live eval: `run_trial.py` / `run_experiment_v2.py`.
