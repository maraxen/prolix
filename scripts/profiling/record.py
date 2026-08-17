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
    """HEAD sha, suffixed "-dirty" if the working tree has uncommitted changes.

    Returns "unknown" on any failure (e.g. no .git present, as on some
    cluster checkouts) -- claims.py rejects both "unknown" and a "-dirty"
    suffix outright for TERM_RANKING/END_TO_END sources, since unanimity
    alone would not catch two records that independently failed to resolve
    a sha (both report "unknown", which trivially "agrees").
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"
    try:
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=_REPO_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except Exception:
        dirty = False
    return f"{sha}-dirty" if dirty else sha


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


_DEVICE_KIND_ALIASES = ("h200", "h100", "a100", "l40s", "l40", "l4", "v100", "a10")


def _normalize_device_kind(raw: str) -> str:
    lowered = raw.lower()
    for canon in _DEVICE_KIND_ALIASES:
        if canon in lowered:
            return canon
    return lowered.replace(" ", "_")


def _capture_device_kind() -> str | None:
    """Auto-detect device_kind from the running JAX environment.

    Returns None on CPU (or if no devices are visible) -- callers on Stage
    0/1 (CPU-only) never need a device_kind; Stage 2+ construction requires
    one, so an unset value there fails __post_init__ rather than silently
    stamping a wrong platform's hardware string.
    """
    import jax

    devices = jax.devices()
    if not devices or getattr(devices[0], "platform", None) != "gpu":
        return None
    return _normalize_device_kind(getattr(devices[0], "device_kind", "unknown"))


@dataclasses.dataclass
class ProbeRecord:
    """A single profiling measurement, stamped with what it may be cited for.

    Caller-supplied (the semantic content of the measurement): probe_id,
    stage, n_atoms, platform, metrics, scopes, config, attribution_method.

    Auto-captured at construction (default_factory, overridable by an
    explicit kwarg -- e.g. tests construct synthetic multi-device-kind
    fixtures on CPU-only hardware): git_sha, timestamp, x64_enabled,
    jax_version, jaxlib_version, xla_flags, device_kind. A provenance field
    a caller can forget is a provenance field that will be forgotten --
    device_kind auto-capture in particular closes a laundering hole: a
    freeform caller-set string could otherwise falsely agree (or disagree)
    with another source's value under the unanimity guard.
    """

    probe_id: str
    stage: int
    n_atoms: int
    platform: str  # "cpu" | "gpu"

    git_sha: str = dataclasses.field(default_factory=_capture_git_sha)
    timestamp: str = dataclasses.field(default_factory=_capture_timestamp)
    x64_enabled: bool = dataclasses.field(default_factory=_capture_x64_enabled)
    jax_version: str = dataclasses.field(default_factory=_capture_jax_version)
    jaxlib_version: str = dataclasses.field(default_factory=_capture_jaxlib_version)
    xla_flags: str = dataclasses.field(default_factory=_capture_xla_flags)
    device_kind: str | None = dataclasses.field(default_factory=_capture_device_kind)

    # metrics is float-only (coerced in __post_init__); config carries
    # non-numeric identity (protein name, mode, on/off flags) and is kept
    # separate deliberately.
    metrics: dict[str, float] = dataclasses.field(default_factory=dict)
    # Per-scope exclusive seconds paired with occurrence count, matching
    # trace.py's parser return type exactly. A value of None means the label
    # was expected but absent from the trace (never 0.0). The field itself
    # being None means the probe captured no trace at all.
    scopes: dict[str, tuple[float, int] | None] | None = None
    config: dict[str, str] = dataclasses.field(default_factory=dict)
    # Optional per-scope note on how that scope's time was attributed (e.g.
    # {"bonded": "trace"} vs {"bonded": "manual_split"}) -- distinct from
    # scopes itself, which is the measurement; this is metadata about how
    # trustworthy that measurement's attribution is.
    attribution_method: dict[str, str] | None = None

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
        coerced_metrics: dict[str, float] = {}
        for key, value in self.metrics.items():
            try:
                coerced_metrics[key] = float(value)
            except (TypeError, ValueError) as exc:
                raise ClaimValidityError(
                    f"metrics[{key!r}]={value!r} is not coercible to float -- "
                    "ProbeRecord.metrics must be float-valued"
                ) from exc
        self.metrics = coerced_metrics

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "ProbeRecord":
        raw = json.loads(text)
        # ty 0.0.29 mis-resolves `cls`/the concrete class here as not
        # satisfying DataclassInstance in this classmethod context (matches
        # the pre-existing ty-panic class already seen on this package's
        # test file); dataclasses.fields() works correctly on ProbeRecord
        # at runtime -- see the passing round-trip tests.
        field_names = {f.name for f in dataclasses.fields(cls)}  # ty: ignore[invalid-argument-type]
        unknown = set(raw.keys()) - field_names
        if unknown:
            raise ClaimValidityError(
                f"ProbeRecord JSON has unknown field(s) {sorted(unknown)} -- "
                "this record was written by a newer contract version than "
                "this code understands; do not deserialize across an "
                "unrecognized schema change"
            )
        missing = field_names - set(raw.keys())
        if missing:
            raise ClaimValidityError(
                f"ProbeRecord JSON is missing field(s) {sorted(missing)} -- "
                "this record predates the field and its provenance cannot "
                "be reconstructed by defaulting to the READING machine's "
                "environment (that would silently corrupt provenance)"
            )
        record_major = str(raw.get("contract_version", "0.0")).split(".")[0]
        current_major = CONTRACT_VERSION.split(".")[0]
        if record_major != current_major:
            raise ClaimValidityError(
                f"ProbeRecord contract_version {raw.get('contract_version')!r} "
                f"has a different major version than the running contract "
                f"{CONTRACT_VERSION!r} -- a claim-validity verdict computed "
                "under one major version is not guaranteed valid under another"
            )
        return cls(**_decode_fields(raw))

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
