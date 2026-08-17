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
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.profiling.claims import CONTRACT_VERSION, ClaimValidityError

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _capture_git_sha() -> str:
    """HEAD sha, suffixed "-dirty"/"-unverified" per working-tree state.

    Returns "unknown" on any failure to resolve HEAD (e.g. no .git present,
    as on some cluster checkouts). If HEAD resolves but the dirty check
    itself fails, returns "<sha>-unverified" rather than silently reporting
    clean -- a failed `git status` is not evidence of a clean tree, and
    treating it as such would let two records that both failed the dirty
    check trivially "agree" under the unanimity guard. claims.py rejects
    "unknown" and both suffixes outright for TERM_RANKING/END_TO_END
    sources.
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
        return f"{sha}-unverified"
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


_VENDOR_PREFIXES = ("nvidia ", "amd ")


def _normalize_device_kind(raw: str) -> str:
    """Lowercase, strip a leading vendor token, collapse whitespace to "_".

    A normalisation, not an allow-list: "NVIDIA H200" -> "h200",
    "NVIDIA RTX PRO 6000 Blackwell" -> "rtx_pro_6000_blackwell". An
    allow-list (matching known device substrings) was tried first and
    rejected -- "NVIDIA GH200 480GB" contains the substring "h200", so a
    Grace-Hopper GH200 would silently collapse to the same device_kind as a
    real H200, defeating the point of auto-capturing this field at all.
    """
    lowered = raw.strip().lower()
    for prefix in _VENDOR_PREFIXES:
        if lowered.startswith(prefix):
            lowered = lowered[len(prefix) :]
            break
    return "_".join(lowered.split())


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


@dataclasses.dataclass(frozen=True)
class ProbeRecord:
    """A single profiling measurement, stamped with what it may be cited for.

    Frozen: every guard below runs once, at construction, and stays true for
    the record's lifetime -- a mutable record would let a downstream report
    or aggregation script (accidentally, not adversarially) rewrite a field
    after validation and silently launder a value the contract never
    actually checked.

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
    # Per-scope attribution method: each value must be "named_scope" (a
    # jax.named_scope label captured directly) or "op_name" (attributed by
    # HLO op name, a weaker guarantee). Distinct from scopes itself, which
    # is the measurement; this is metadata about how trustworthy that
    # measurement's attribution is. Required whenever scopes is not None --
    # enforced in __post_init__.
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
                coerced = float(value)
            except (TypeError, ValueError) as exc:
                raise ClaimValidityError(
                    f"metrics[{key!r}]={value!r} is not coercible to float -- "
                    "ProbeRecord.metrics must be float-valued"
                ) from exc
            if not math.isfinite(coerced):
                raise ClaimValidityError(
                    f"metrics[{key!r}]={coerced!r} is not finite -- a failed "
                    "or diverged measurement (nan/inf) is not a citable metric"
                )
            coerced_metrics[key] = coerced
        object.__setattr__(self, "metrics", coerced_metrics)

        if self.scopes is not None and self.attribution_method is None:
            raise ClaimValidityError(
                "attribution_method is required whenever scopes is not None"
            )
        if self.attribution_method is not None:
            bad = {
                v
                for v in self.attribution_method.values()
                if v not in ("named_scope", "op_name")
            }
            if bad:
                raise ClaimValidityError(
                    f"attribution_method values must be 'named_scope' or "
                    f"'op_name', got {sorted(bad)}"
                )

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "ProbeRecord":
        """Deserialize, fail-closed on version drift AND on malformed input.

        Rule order matters: the contract_version check runs BEFORE the
        missing-field check, so a version-skewed record is always diagnosed
        as version skew (naming both versions) rather than misdiagnosed as
        field-level corruption -- a genuine 1.0 record missing
        attribution_method would otherwise raise the "provenance cannot be
        reconstructed" error instead of the version-mismatch error that
        actually explains it.

        The missing-field check only fires for fields with a default_factory
        (git_sha, timestamp, x64_enabled, ..., metrics, config): those are
        the fields a MAJOR bump introduces (spec P1's bump rule). A field
        with a plain default (e.g. attribution_method) is a MINOR-bump
        field by the same rule -- an old record missing it is still
        readable, and simply gets that field's literal default.
        """
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ClaimValidityError(f"ProbeRecord JSON is not valid JSON: {exc}") from exc
        if not isinstance(raw, dict):
            raise ClaimValidityError(
                "ProbeRecord JSON must decode to an object, got "
                f"{type(raw).__name__}"
            )

        # ty 0.0.29 mis-resolves `cls`/the concrete class here as not
        # satisfying DataclassInstance in this classmethod context (matches
        # the pre-existing ty-panic class already seen on this package's
        # test file); dataclasses.fields() works correctly on ProbeRecord
        # at runtime -- see the passing round-trip tests.
        fields = {f.name: f for f in dataclasses.fields(cls)}  # ty: ignore[invalid-argument-type]
        field_names = set(fields)
        unknown = set(raw.keys()) - field_names
        if unknown:
            raise ClaimValidityError(
                f"ProbeRecord JSON has unknown field(s) {sorted(unknown)} -- "
                "this record was written by a newer contract version than "
                "this code understands; do not deserialize across an "
                "unrecognized schema change"
            )

        record_version = raw.get("contract_version", "0.0")
        record_major = str(record_version).split(".")[0]
        current_major = CONTRACT_VERSION.split(".")[0]
        if record_major != current_major:
            raise ClaimValidityError(
                f"ProbeRecord contract_version {record_version!r} has a "
                f"different major version than the running contract "
                f"{CONTRACT_VERSION!r} -- a claim-validity verdict computed "
                "under one major version is not guaranteed valid under another"
            )

        missing = field_names - set(raw.keys())
        hard_missing = sorted(
            name for name in missing if fields[name].default_factory is not dataclasses.MISSING
        )
        if hard_missing:
            raise ClaimValidityError(
                f"ProbeRecord JSON is missing field(s) {hard_missing} -- each "
                "has a default_factory (not a static default), so silently "
                "applying it here would fabricate a value (this reading "
                "machine's own environment, or an empty container standing "
                "in for lost data) rather than reconstructing what was "
                "actually recorded"
            )

        try:
            return cls(**_decode_fields(raw))
        except ClaimValidityError:
            raise
        except (TypeError, AttributeError, ValueError) as exc:
            raise ClaimValidityError(
                f"ProbeRecord JSON has malformed field value(s): {exc}"
            ) from exc

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
