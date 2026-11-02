Reproduce B0–B2 (after installing Tiny Aya weights):

  python scripts/generate_dataset.py
  python scripts/train_lora.py --dry-run   # drop --dry-run on a GPU box
  python scripts/extract_activations.py --dry-run
  python scripts/build_prototypes.py --dump results/activations
  python scripts/run_b0.py --limit 10
  python scripts/run_b1.py --limit 10
  python scripts/run_b2.py --limit 10
  python scripts/compare_results.py

Gold-replay smoke (no weights):

  python scripts/run_b0.py --dry-run --limit 10
  python scripts/compare_results.py --dry-run
