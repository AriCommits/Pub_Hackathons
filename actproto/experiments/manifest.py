"""Hash-bound manifests, exclusive run directories, and operating-system GPU leases."""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .contracts import VERSION, _version


def hash_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_object(value: Any) -> str:
    return hash_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode())


def snapshot_files(paths: list[str | Path]) -> dict[str, str]:
    """Stream protected artifact hashes without duplicating their contents."""
    return {str(path): hash_file(path) for path in paths}


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    config_hash: str
    data_hash: str
    model_hash: str
    fixture_hash: str
    template_hash: str
    verifier_hash: str
    code_revision: str
    schema_version: str = VERSION

    def __post_init__(self):
        _version(self.schema_version)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.run_id):
            raise ValueError("invalid run ID")
        for field in (self.config_hash, self.data_hash, self.model_hash, self.fixture_hash, self.template_hash, self.verifier_hash):
            if not re.fullmatch(r"[0-9a-f]{64}", field):
                raise ValueError("artifact hashes must be lowercase SHA-256")
        if not self.code_revision:
            raise ValueError("missing code revision")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        _version(data.get("schema_version"))
        return cls(**data)

    @property
    def digest(self):
        return hash_object(self.to_dict())


def prepare_run(root: str | Path, manifest: RunManifest, resume_token: str | None = None) -> tuple[Path, str]:
    """Create once, or resume only with the original token and identical manifest."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    run = root / manifest.run_id
    try:
        run.mkdir()
    except FileExistsError:
        if resume_token is None:
            raise FileExistsError(f"run already exists: {run}")
        try:
            existing = json.loads((run / "run-owner.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError("run ownership record missing or invalid") from exc
        if not secrets.compare_digest(existing.get("token_hash", ""), hash_bytes(resume_token.encode())):
            raise ValueError("invalid resume token")
        if existing.get("manifest_digest") != manifest.digest:
            raise ValueError("resume manifest does not match")
        return run, resume_token
    token = secrets.token_urlsafe(32)
    owner = {"manifest": manifest.to_dict(), "manifest_digest": manifest.digest,
             "token_hash": hash_bytes(token.encode())}
    with (run / "run-owner.json").open("x", encoding="utf-8") as output:
        json.dump(owner, output, sort_keys=True, indent=2)
    return run, token


def _process_start() -> str:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        values = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(kernel.GetCurrentProcess(), *(ctypes.byref(x) for x in values)):
            raise OSError(ctypes.get_last_error(), "GetProcessTimes failed")
        return str((values[0].dwHighDateTime << 32) | values[0].dwLowDateTime)
    return Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19]


class GpuLease:
    """Exclusive OS lock is authoritative; an old heartbeat never steals a live lock.

    Use the same lease directory on the same host for every GPU-owning process.
    Metadata is informational; process exit releases the kernel lock even after a crash.
    """
    def __init__(self, root: str | Path, device_uuid: str):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", device_uuid):
            raise ValueError("invalid GPU UUID")
        self.root, self.device_uuid = Path(root), device_uuid
        self._file = None

    def acquire(self):
        if self._file is not None:
            raise RuntimeError("lease already held")
        self.root.mkdir(parents=True, exist_ok=True)
        handle = os.fdopen(os.open(self.root / f"{self.device_uuid}.lease", os.O_RDWR | os.O_CREAT, 0o600), "r+b")
        try:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RuntimeError(f"GPU lease held: {self.device_uuid}") from exc
        self._file = handle
        try:
            self.owner = {"pid": os.getpid(), "process_start": _process_start(), "device_uuid": self.device_uuid}
            self.heartbeat()
        except Exception:
            self.release()
            raise
        return self

    def heartbeat(self):
        if self._file is None:
            raise RuntimeError("lease not held")
        payload = dict(self.owner, heartbeat_unix=time.time())
        metadata = self.root / f"{self.device_uuid}.json"
        temporary = self.root / f"{self.device_uuid}.{os.getpid()}.tmp"
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(temporary, metadata)

    def release(self):
        if self._file is not None:
            handle, self._file = self._file, None
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *args):
        self.release()
