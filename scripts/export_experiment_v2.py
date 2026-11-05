"""List or copy only explicitly selected, hash-verified public experiment artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import shutil

from actproto.experiments.manifest import hash_file

REQUIRED = {"adapter", "tokenizer", "template", "data_manifest", "fixture_manifest",
            "curriculum", "traces", "metrics", "environment_lock", "model_card"}
OPTIONAL = {"prototype", "threshold", "private_final"}
BANNED_COMPONENTS = {".git", ".env", ".aws", ".ssh", "__pycache__", ".cache", "cache", "caches", "logs"}
SECRET = re.compile(r"(?:-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16})\b)")
SECRET_FIELD = re.compile(r"(?i)[\"'](?:api[_-]?key|secret|password|access[_-]?token|authorization|credential)[\"']\s*:\s*[\"'][^\"']+[\"']")
TEXT_SUFFIXES = {".json", ".jsonl", ".txt", ".yaml", ".yml", ".md", ".jinja", ".lock", ".py", ".toml"}


def validate_allowlist(root, manifest, *, include_private=False):
    root = Path(root).resolve(strict=True)
    entries = manifest.get("artifacts")
    if manifest.get("schema_version") != "tooluse_v2" or not isinstance(entries, list):
        raise ValueError("invalid export manifest")
    roles = {entry.get("role") for entry in entries if isinstance(entry, dict)}
    if not REQUIRED <= roles or roles - REQUIRED - OPTIONAL:
        raise ValueError(f"missing or unsupported artifact roles: {sorted(REQUIRED - roles)}")
    if ("prototype" in roles) != ("threshold" in roles):
        raise ValueError("prototype and threshold must be exported together")
    if "private_final" in roles and not (include_private and manifest.get("joint_evaluation_complete") is True):
        raise ValueError("final answers are sealed until joint evaluation completes")
    seen, resolved = set(), []
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("invalid artifact entry")
        name, role, digest = item.get("path"), item.get("role"), item.get("sha256")
        if not isinstance(name, str) or not name or "\\" in name or ":" in name:
            raise ValueError("artifact paths must be portable relative paths")
        path = PurePosixPath(name)
        if path.is_absolute() or any(p in (".", "..") or p.lower() in BANNED_COMPONENTS or p.startswith(".") for p in path.parts):
            raise ValueError("unsafe or disallowed artifact path")
        canonical = name.casefold()
        if canonical in seen:
            raise ValueError("duplicate artifact path")
        seen.add(canonical)
        if path.suffix.lower() in {".pem", ".key", ".p12", ".pfx"} or path.name.lower() in {"credentials", "credentials.json", "secrets.json", "resume-token.txt"}:
            raise ValueError("credential file not permitted")
        if path.name.lower() in {"iid.jsonl", "stress.jsonl", "test.jsonl", "final-access.jsonl"} and role != "private_final":
            raise ValueError("held-out answers must be private")
        if role != "private_final" and item.get("split") in {"test", "iid", "stress"}:
            raise ValueError("held-out artifacts must be private")
        candidate = root.joinpath(*path.parts)
        if any(part.is_symlink() for part in [candidate, *candidate.parents] if part != root.parent):
            raise ValueError("symlinks are not permitted")
        if not candidate.is_file() or not candidate.resolve().is_relative_to(root):
            raise ValueError(f"missing or unsafe artifact: {name}")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or hash_file(candidate) != digest:
            raise ValueError(f"artifact hash mismatch: {name}")
        # Scan even extensionless or binary artifacts for recognizable secrets.
        # Chunks overlap so a token crossing a block boundary is still detected.
        with candidate.open("rb") as file:
            tail = b""
            for block in iter(lambda: file.read(1024 * 1024), b""):
                chunk = (tail + block).decode("latin-1")
                if SECRET.search(chunk) or SECRET_FIELD.search(chunk):
                    raise ValueError(f"credential-like material in {name}")
                tail = (tail + block)[-512:]
        resolved.append((item, candidate))
    by_role = {role: {PurePosixPath(item["path"]).name for item, _ in resolved if item["role"] == role}
               for role in roles}
    if "adapter_config.json" not in by_role["adapter"] or not any(
            name.endswith((".safetensors", ".bin")) for name in by_role["adapter"]):
        raise ValueError("adapter config and weights required for reload")
    if "tokenizer_config.json" not in by_role["tokenizer"] or not any(
            name in {"tokenizer.json", "tokenizer.model", "spiece.model"} for name in by_role["tokenizer"]):
        raise ValueError("tokenizer config and model required for reload")
    return resolved


def export(root, manifest, destination=None, *, include_private=False):
    resolved = validate_allowlist(root, manifest, include_private=include_private)
    if destination is None:
        return [item["path"] for item, _ in resolved]
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=False)
    for item, source in resolved:
        target = dest.joinpath(*PurePosixPath(item["path"]).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if hash_file(target) != item["sha256"]:
            raise ValueError(f"copy hash mismatch: {item['path']}")
    (dest / "export-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return [item["path"] for item, _ in resolved]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--destination", help="copy a local bundle; omit for list-only")
    parser.add_argument("--include-private", action="store_true")
    args = parser.parse_args(argv)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    for name in export(args.root, manifest, args.destination, include_private=args.include_private):
        print(name)


if __name__ == "__main__":
    main()
