"""Trial paths, run matrix, and hypothesis gates — load from trial.yaml."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from actproto.utils.paths import PathEscapeError, contain_under

CONFIG_DIR = Path(__file__).resolve().parent
DEFAULT_TRIAL_PATH = CONFIG_DIR / "trial.yaml"


def project_root() -> Path:
    """Lexsi package root (parent of actproto/)."""
    return CONFIG_DIR.parent.parent


@dataclass
class PathsConfig:
    data_processed: str = "data/processed"
    activations_dump: str = "results/activations"
    adapter_b1: str = "results/adapters/b1"  # historical B1 artifact path (do not train into this by default)
    results_root: str = "results"
    prototypes_dir: str = "results"
    reports_dir: str = "results/b0_b2"

    def prototypes_npz(self) -> str:
        return f"{self.prototypes_dir.rstrip('/')}/prototypes.npz"

    def prototype_metrics(self) -> str:
        return f"{self.prototypes_dir.rstrip('/')}/prototype_metrics.json"

    def probe_metrics(self) -> str:
        return f"{self.results_root.rstrip('/')}/probe_metrics.json"


@dataclass
class EvalConfig:
    split: str = "test"
    limit: int | None = None


@dataclass
class HypothesisConfig:
    correctness_drop_pp: float = 3.0
    latency_ratio: float = 1.5
    min_coverage: float = 0.5


@dataclass
class PrototypeFitConfig:
    min_coverage: float = 0.5


@dataclass
class RunSpec:
    description: str = ""
    adapter: str | None = None
    router: bool = False
    prototypes: str | None = None
    threshold: str | None = None
    max_agent_steps: int | None = None
    replicas: int | None = None
    route_first_hop_only: bool = False


@dataclass
class ExportConfig:
    adapter: str = "results/sft_v2/e2"
    data: str = "data/processed"
    logs: str = "results/b0_b2"


@dataclass
class TrialConfig:
    """All filesystem layout and B0–B2 run definitions for one experiment trial."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    hypothesis: HypothesisConfig = field(default_factory=HypothesisConfig)
    prototype_fit: PrototypeFitConfig = field(default_factory=PrototypeFitConfig)
    runs: dict[str, RunSpec] = field(default_factory=dict)
    export: ExportConfig = field(default_factory=ExportConfig)
    _root: Path = field(default_factory=project_root, repr=False)

    @property
    def root(self) -> Path:
        return self._root

    def resolve(self, rel: str | None) -> Path | None:
        if rel is None:
            return None
        try:
            return contain_under(self._root, rel, label="trial path")
        except PathEscapeError:
            raise

    def report_dir(self, run_id: str) -> Path:
        return self.resolve(self.paths.reports_dir) / run_id  # type: ignore[operator]

    def run_spec(self, run_id: str) -> RunSpec:
        if run_id not in self.runs:
            raise KeyError(f"unknown run {run_id!r}; configured: {sorted(self.runs)}")
        return self.runs[run_id]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, root: Path | None = None) -> TrialConfig:
        root = root or CONFIG_DIR.parent.parent
        paths = PathsConfig(**_filter(data.get("paths") or {}, PathsConfig))
        ev = EvalConfig(**_filter(data.get("eval") or {}, EvalConfig))
        hyp = HypothesisConfig(**_filter(data.get("hypothesis") or {}, HypothesisConfig))
        pfit = PrototypeFitConfig(**_filter(data.get("prototype_fit") or {}, PrototypeFitConfig))
        export = ExportConfig(**_filter(data.get("export") or {}, ExportConfig))
        runs_raw = data.get("runs") or {}
        runs = {k: RunSpec(**_filter(v or {}, RunSpec)) for k, v in runs_raw.items()}
        return cls(
            paths=paths,
            eval=ev,
            hypothesis=hyp,
            prototype_fit=pfit,
            runs=runs,
            export=export,
            _root=root,
        )


def _filter(data: dict[str, Any], cls: type) -> dict[str, Any]:
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in known}


def _read_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config {path} must be a mapping")
    return loaded


def load_trial(path: str | Path | None = None, *, root: Path | None = None) -> TrialConfig:
    p = Path(path) if path else DEFAULT_TRIAL_PATH
    if not p.is_absolute():
        candidate = CONFIG_DIR / p
        p = candidate.resolve() if candidate.exists() else (Path.cwd() / p).resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    trial_root = root or project_root()
    return TrialConfig.from_dict(_read_yaml(p), root=trial_root)


def merge_run_paths(trial: TrialConfig, spec: RunSpec) -> tuple[str | None, str | None, str | None]:
    """Return (adapter_rel, prototypes_rel, threshold_rel) with trial defaults for B2."""
    adapter = spec.adapter
    prototypes = spec.prototypes or (trial.paths.prototypes_npz() if spec.router else None)
    threshold = spec.threshold or (trial.paths.prototype_metrics() if spec.router else None)
    return adapter, prototypes, threshold
