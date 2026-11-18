# CUDA RL environment

Lexsi keeps three interpreters apart.

| Role | Interpreter | Torch |
| --- | --- | --- |
| Supervised SFT | `C:\Users\arian\Github\Hackathons\lexsi\.venv\Scripts\python.exe` | `2.11.0+cu128` (do not reinstall) |
| CPU GRPO mechanics | `C:\Users\arian\Github\Hackathons-worktrees\plan-10-rl-env\Scripts\python.exe` | `2.11.0+cpu` |
| CUDA GRPO | `C:\Users\arian\Github\Hackathons-worktrees\plan-10-rl-gpu-env\Scripts\python.exe` | `2.11.0+cu128` |

The CPU environment remains the interpreter for the already-verified `results/rl_v2/grpo_cpu_smoke` mechanics run. New GRPO training uses the CUDA environment. Neither path uses `C:\Python314\python.exe`.

The CUDA environment was created from the CPU environment's Python 3.12.13:

```text
C:\Users\arian\Github\Hackathons-worktrees\plan-10-rl-env\Scripts\python.exe -m venv C:\Users\arian\Github\Hackathons-worktrees\plan-10-rl-gpu-env
```

`requirements-rl-gpu.txt` is the install recipe. Install `torch==2.11.0` and `torchvision==0.26.0` from `https://download.pytorch.org/whl/cu128` before the rest of the AgentTune stack. Then install that requirements file. AgentTune is 1.1.3 at commit `d4db4c12d52342227c4113a70183d0f64aa2f345`. TRL is 1.7.1. `requirements-rl.lock` stays the CPU freeze. `requirements-rl-gpu.lock` is the freeze of this environment after the CUDA wheel was verified (`torch==2.11.0+cu128`).

After every install, require `torch.__version__ == "2.11.0+cu128"` and `torch.cuda.is_available() is True`. If a later install leaves `2.11.0+cpu`, reinstall the cu128 wheel and check again.

With `CUDA_VISIBLE_DEVICES` unset in the parent shell, this command prints the torch version and each device name plus UUID. It loads no model weights.

```text
C:\Users\arian\Github\Hackathons-worktrees\plan-10-rl-gpu-env\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available());
[print(i, torch.cuda.get_device_properties(i).name, torch.cuda.get_device_properties(i).uuid) for i in range(torch.cuda.device_count())]"
```

The import check for this environment reports AgentTune 1.1.3, TRL 1.7.1, torch `2.11.0+cu128`, and CUDA available. `validate_agenttune_source` accepts the pinned commit. Smoke-step counts and Tiny Aya memory bytes are recorded in `docs/RL_BACKEND_COMPATIBILITY.md` only after those runs, not in this file.
