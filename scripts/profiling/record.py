"""ProbeRecord: a stage- and scale-stamped profiling measurement.

No prolix imports, no JAX import at module scope (grep-enforced in
tests/profiling/test_claim_contract.py) -- provenance capture that needs jax
(jax_version/jaxlib_version/x64_enabled) is done lazily, inside a
default_factory, only when a ProbeRecord is actually constructed. See
.praxia/docs/specs/260817_jax-profiling-optimization-workflow.md section P1.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.profiling.claims import CONTRACT_VERSION, ClaimValidityError

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _capture_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _capture_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _capture_x64_enabled() -> bool:
    import jax

    return bool(jax.config.x64_enabled)


def _capture_jax_version() -> str:
    import jax

    return jax.__version__


def _capture_jaxlib_version() -> str:
    import jaxlib

    return jaxlib.__version__


def _capture_xla_flags() -> str:
    return os.environ.get("XLA_FLAGS", "")


@dataclasses.dataclass
class ProbeRecord:
    """A single profiling measurement, stamped with what it may be cited for.

    Caller-supplied (the semantic content of the measurement): probe_id,
    stage, n_atoms, platform, device_kind, metrics, scopes, config.

    Auto-captured at construction, never passed by the caller -- a
    provenance field a caller can forget is a provenance field that will be
    forgotten: git_sha, timestamp, x64_enabled, jax_version, jaxlib_version,
    xla_flags. (The spec's P1 text names x64_enabled/jax_version/
    jaxlib_version/xla_flags explicitly as auto-captured; git_sha/timestamp
    are the same kind of provenance a caller would otherwise forget, so this
    implementation auto-captures them too rather than leaving them as an
    unenforced convention.)
    """

    probe_id: str
    stage: int
    n_atoms: int
    platform: str  # "cpu" | "gpu"

    device_kind: str | None = None
    git_sha: str = dataclasses.field(default_factory=_capture_git_sha)
    timestamp: str = dataclasses.field(default_factory=_capture_timestamp)
    x64_enabled: bool = dataclasses.field(default_factory=_capture_x64_enabled)
    jax_version: str = dataclasses.field(default_factory=_capture_jax_version)
    jaxlib_version: str = dataclasses.field(default_factory=_capture_jaxlib_version)
    xla_flags: str = dataclasses.field(default_factory=_capture_xla_flags)

    # metrics is float-only; config carries non-numeric identity (protein
    # name, mode, on/off flags) and is kept separate deliberately.
    metrics: dict[str, float] = dataclasses.field(default_factory=dict)
    # Per-scope exclusive seconds paired with occurrence count, matching
    # trace.py's parser return type exactly. A value of None means the label
    # was expected but absent from the trace (never 0.0). The field itself
    # being None means the probe captured no trace at all.
    scopes: dict[str, tuple[float, int] | None] | None = None
    config: dict[str, str] = dataclasses.field(default_factory=dict)

    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.stage not in (0, 1, 2, 3):
            raise ClaimValidityError(f"stage must be in {{0,1,2,3}}, got {self.stage!r}")
        if self.stage >= 2 and self.platform != "gpu":
            raise ClaimValidityError(
                f"stage={self.stage} requires platform='gpu' (Stage-2+ "
                f"records are GPU-measured by definition), got "
                f"platform={self.platform!r}"
            )
        if self.stage >= 2 and self.device_kind is None:
            raise ClaimValidityError(
                f"stage={self.stage} requires device_kind, got None"
            )
        if self.n_atoms <= 0:
            raise ClaimValidityError(f"n_atoms must be > 0, got {self.n_atoms}")

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "ProbeRecord":
        return cls(**_decode_fields(json.loads(text)))

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @classmethod
    def read(cls, path: str | Path) -> "ProbeRecord":
        return cls.from_json(Path(path).read_text())


def _decode_fields(d: dict[str, Any]) -> dict[str, Any]:
    """Restore tuple-vs-None in `scopes` after a JSON round-trip.

    json.loads turns a serialized 2-element array back into a Python list
    and `null` back into None natively -- the only thing that needs
    restoring is list -> tuple for non-null scope entries, since a bare list
    is not what ProbeRecord.scopes is typed to hold.
    """
    scopes = d.get("scopes")
    if scopes is not None:
        d["scopes"] = {
            k: (tuple(v) if v is not None else None) for k, v in scopes.items()
        }
    return d
