#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FORBIDDEN_NAMES = {".env", "HF_TOKEN", ".hf_token", "secrets.yaml", "credentials.json"}


def collect(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        if path.is_file():
            files.append(path)
        else:
            files.extend(p for p in path.rglob("*") if p.is_file())
    return files


def assert_no_secrets(files: list[Path]) -> None:
    bad = [p for p in files if p.name in FORBIDDEN_NAMES or p.suffix == ".env"]
    if bad:
        raise SystemExit(f"refusing to upload secrets: {bad}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List (or upload) HF artifacts. Refuses secrets.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--repo", default=None, help="HF repo id; required unless --dry-run")
    parser.add_argument("--adapter", default="results/adapters/b1")
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--logs", default="results/b0_b2")
    args = parser.parse_args(argv)
    files = collect([ROOT / args.adapter, ROOT / args.data, ROOT / args.logs])
    assert_no_secrets(files)
    rel = [str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p) for p in files]
    print("files:")
    for item in rel:
        print(f"  {item}")
    if args.dry_run or not args.repo:
        return 0
    from huggingface_hub import HfApi  # pragma: no cover

    api = HfApi()
    for path in files:
        api.upload_file(path_or_fileobj=str(path), path_in_repo=path.name, repo_id=args.repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
