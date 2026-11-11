#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from actproto.utils.paths import PathEscapeError, contain_under  # noqa: E402
from actproto.utils.secret_scan import assert_paths_clean  # noqa: E402


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
    assert_paths_clean(files, raise_system_exit=True)


def main(argv: list[str] | None = None) -> int:
    from scripts._cli import add_trial_argument, trial_from_args

    parser = argparse.ArgumentParser(description="List (or upload) HF artifacts. Refuses secrets.")
    add_trial_argument(parser)
    parser.add_argument("--list-only", action="store_true", help="List files that would be uploaded (default without --repo).")
    parser.add_argument("--repo", default=None, help="HF repo id; required to upload")
    args = parser.parse_args(argv)
    trial = trial_from_args(args)
    try:
        paths = [
            trial.resolve(trial.export.adapter),
            trial.resolve(trial.export.data),
            trial.resolve(trial.export.logs),
        ]
    except PathEscapeError as exc:
        raise SystemExit(str(exc)) from exc
    paths = [p for p in paths if p is not None]
    files = collect(paths)
    assert_no_secrets(files)
    root = trial.root
    rel = []
    for path in files:
        try:
            contain_under(root, path, label="export artifact")
        except PathEscapeError as exc:
            raise SystemExit(str(exc)) from exc
        rel.append(str(path.relative_to(root)))
    print("files:")
    for item in rel:
        print(f"  {item}")
    if args.list_only or not args.repo:
        return 0
    from huggingface_hub import HfApi  # pragma: no cover

    api = HfApi()
    for path in files:
        path_in_repo = str(path.relative_to(root))
        api.upload_file(path_or_fileobj=str(path), path_in_repo=path_in_repo, repo_id=args.repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
