"""Claim-validity contract: what a ProbeRecord may be cited to support.

No prolix imports (grep-enforced in tests/profiling/test_claim_contract.py).
See .praxia/docs/specs/260817_jax-profiling-optimization-workflow.md section
P1 for the full design rationale, including why the guards below are the
specific ones that exist (the roofline argument for TERM_RANKING's stage>=2
floor, the memory-regime argument for SCALE_EXTRAPOLATION_LIMIT, and the
1170x XLA_FLAGS finding on this project's Blackwell nodes for the unanimity
checks).
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.profiling.record import ProbeRecord

CONTRACT_VERSION = "2.0"

# A deliberate cheap linear proxy, not a derived bound -- see the spec's P1
# note on SCALE_EXTRAPOLATION_LIMIT for the known false-pass case (the real
# regime change is a *memory* transition, not linear in n_atoms).
SCALE_EXTRAPOLATION_LIMIT = 10.0

# Fields the unanimity check (assert_claim_supported guard (d)) requires
# TERM_RANKING/END_TO_END source records to agree on. `platform` is included
# even though it is already structurally implied for TERM_RANKING (stage>=2
# construction check forces platform=="gpu"); checking it uniformly avoids
# special-casing per claim class and is the only thing that actually catches
# a mixed CPU/GPU source set for END_TO_END, which has no stage floor.
# `git_sha` catches a claim silently mixing records measured on different
# commits -- unanimity alone is why it must pair with the separate
# _reject_unverifiable_git_sha check below: two records that both failed to
# resolve a sha ("unknown") would otherwise trivially "agree".
_UNANIMITY_FIELDS = ("x64_enabled", "xla_flags", "device_kind", "platform", "git_sha")


def _reject_unverifiable_git_sha(sources: list["ProbeRecord"], claim: ClaimClass) -> None:
    for r in sources:
        if (
            r.git_sha == "unknown"
            or r.git_sha.endswith("-dirty")
            or r.git_sha.endswith("-unverified")
        ):
            raise ClaimValidityError(
                f"{claim.name} source {r.probe_id!r} has unverifiable "
                f"git_sha ({r.git_sha!r}) -- measured outside a clean, "
                "resolvable git checkout (or the dirty-tree check itself "
                "failed); provenance cannot be trusted for a citable claim"
            )


class ClaimValidityError(ValueError):
    """A claim is not supported by the given ProbeRecord set."""


class ClaimClass(enum.Enum):
    STRUCTURAL = "structural"
    DISPATCH_COUNT = "dispatch_count"
    TERM_RANKING = "term_ranking"
    END_TO_END = "end_to_end"


REQUIRED_METRICS: dict[ClaimClass, frozenset[str]] = {
    ClaimClass.STRUCTURAL: frozenset(),
    ClaimClass.DISPATCH_COUNT: frozenset(
        {"n_executions", "n_compilations", "n_host_syncs", "n_jit_traces"}
    ),
    ClaimClass.TERM_RANKING: frozenset({"total_step_seconds"}),
    ClaimClass.END_TO_END: frozenset({"total_step_seconds"}),
}


def permitted_claims(record: "ProbeRecord") -> set[ClaimClass]:
    """Claim classes `record`'s stage floor permits in isolation.

    TERM_RANKING and END_TO_END are never granted here: both depend on
    properties of the *set* of records supporting a claim (unanimity,
    pairing), not on a single record considered alone. Use
    assert_claim_supported for those.
    """
    claims = {ClaimClass.STRUCTURAL}
    if record.stage >= 1:
        claims.add(ClaimClass.DISPATCH_COUNT)
    return claims


def select_sources(
    records: list["ProbeRecord"], claim: ClaimClass
) -> list["ProbeRecord"]:
    """Return the records that back `claim`: metric-keyed, then stage-maximal.

    Fail-closed: raises ClaimValidityError (never returns an empty list),
    distinguishing two different empty-result causes so the message tells
    the caller which one actually fired -- "add this metric" is the wrong
    fix when the real problem is missing scope attribution, and vice versa.
    """
    required = REQUIRED_METRICS[claim]
    candidates = [r for r in records if required.issubset(r.metrics.keys())]
    if not candidates:
        raise ClaimValidityError(
            f"no record carries the metrics {claim.name} ranges over "
            f"(required metrics: {sorted(required)})"
        )
    if claim is ClaimClass.TERM_RANKING:
        # A ranking needs at least two terms to rank.
        candidates = [
            r
            for r in candidates
            if r.scopes is not None
            and sum(1 for v in r.scopes.values() if v is not None) >= 2
        ]
        if not candidates:
            raise ClaimValidityError(
                f"{claim.name} requires >=2 attributed scopes, but no "
                "record carrying the required metrics also carries scope "
                "attribution (scopes is None, or has fewer than 2 non-None "
                "entries) -- capture a trace, not just aggregate metrics"
            )
    max_stage = max(r.stage for r in candidates)
    sources = [r for r in candidates if r.stage == max_stage]

    # Read-side rule (iii): sources must agree on contract_version's MAJOR
    # component (a minor disagreement is permitted). from_json already
    # refuses to deserialize a non-current-major record on its own, but
    # contract_version is a plain settable field -- an in-process or
    # programmatically constructed source set can still mix majors
    # undetected without this check.
    majors = {r.contract_version.split(".")[0] for r in sources}
    if len(majors) > 1:
        raise ClaimValidityError(
            f"{claim.name} sources disagree on contract_version's MAJOR "
            f"component: {sorted(majors)} -- records at different major "
            "contract versions cannot be mixed into one claim"
        )
    return sources


def paired_configs(
    records: list["ProbeRecord"],
    claim: ClaimClass,
    *,
    axis: str,
    hold_fixed: tuple[str, ...],
) -> list[tuple["ProbeRecord", "ProbeRecord"]]:
    """Pairs of select_sources(records, claim) differing only on `axis`.

    Groups by the `hold_fixed` config keys; within each group, returns one
    pair per distinct pair of `axis` values (not just adjacent ones, if a
    group has more than two). select_sources does the metric/stage
    filtering; this only groups and pairs what survives that.

    Raises ClaimValidityError if any selected source is missing the axis
    key or any hold_fixed key, naming the record and the key -- defaulting
    a missing key to "" would merge distinct cells and report a confounded
    pair as a pass.
    """
    sources = select_sources(records, claim)
    groups: dict[tuple[str, ...], dict[str, list["ProbeRecord"]]] = {}
    for r in sources:
        missing_hold = [k for k in hold_fixed if k not in r.config]
        if missing_hold:
            raise ClaimValidityError(
                f"{claim.name} source {r.probe_id!r} is missing hold_fixed "
                f"config key(s) {missing_hold} -- defaulting would merge "
                "distinct cells and report a confounded pair as a pass"
            )
        if axis not in r.config:
            raise ClaimValidityError(
                f"{claim.name} source {r.probe_id!r} is missing axis config "
                f"key {axis!r} -- defaulting would merge distinct cells and "
                "report a confounded pair as a pass"
            )
        key = tuple(r.config[k] for k in hold_fixed)
        by_axis = groups.setdefault(key, {})
        by_axis.setdefault(r.config[axis], []).append(r)

    pairs: list[tuple["ProbeRecord", "ProbeRecord"]] = []
    for by_axis in groups.values():
        axis_values = sorted(by_axis.keys())
        if len(axis_values) < 2:
            continue
        for i in range(len(axis_values)):
            for j in range(i + 1, len(axis_values)):
                for a in by_axis[axis_values[i]]:
                    for b in by_axis[axis_values[j]]:
                        pairs.append((a, b))
    return pairs


def assert_claim_supported(
    records: list["ProbeRecord"],
    claim: ClaimClass,
    *,
    target_n_atoms: int | None = None,
    allow_no_target: bool = False,
) -> None:
    """Raise ClaimValidityError if `claim` is not supported by `records`.

    Does not return a value -- the call itself is the assertion. See P1's
    guard list (a)-(d) in the spec for the exact rules; summarized:
      (a) TERM_RANKING requires every selected source at stage>=2 (a
          CPU-derived term ranking is not valid on GPU by the roofline
          argument in spec section 1).
      (b) END_TO_END must declare target_n_atoms, or explicitly opt out via
          allow_no_target=True.
      (c) END_TO_END's target_n_atoms / min(source n_atoms) must not exceed
          SCALE_EXTRAPOLATION_LIMIT. Uses min (not max) because the claim is
          backed by *all* selected sources -- deliberately means a further
          valid same-stage record with smaller n_atoms can tighten the guard
          and flip a passing claim to raising. That is fail-closed, not a
          bug.
      (d) TERM_RANKING/END_TO_END sources must be unanimous on
          x64_enabled/xla_flags/device_kind/platform/git_sha, and no source
          may carry an unverifiable git_sha ("unknown", dirty-tree, or a
          failed dirty-check). select_sources additionally enforces
          contract_version MAJOR-component unanimity (read-side rule iii).
    """
    sources = select_sources(records, claim)

    if claim is ClaimClass.TERM_RANKING:
        sub_stage = sorted({r.stage for r in sources if r.stage < 2})
        if sub_stage:
            raise ClaimValidityError(
                "TERM_RANKING requires stage>=2 sources -- a CPU-derived term "
                "ranking is not valid on GPU (see spec section 1's roofline "
                f"argument); got sources at stage(s) {sub_stage}"
            )

    if claim is ClaimClass.END_TO_END:
        if target_n_atoms is None and not allow_no_target:
            raise ClaimValidityError(
                "END_TO_END claim must declare target_n_atoms, or pass "
                "allow_no_target=True to explicitly claim no extrapolation"
            )
        if target_n_atoms is not None:
            min_n = min(r.n_atoms for r in sources)
            ratio = target_n_atoms / min_n
            if ratio > SCALE_EXTRAPOLATION_LIMIT:
                raise ClaimValidityError(
                    f"END_TO_END extrapolation ratio {ratio:.2f}x exceeds "
                    f"SCALE_EXTRAPOLATION_LIMIT={SCALE_EXTRAPOLATION_LIMIT}x "
                    f"(target_n_atoms={target_n_atoms}, min source "
                    f"n_atoms={min_n}). Narrow the claim to a smaller "
                    "target_n_atoms, or obtain a larger-n_atoms source -- "
                    "never raise SCALE_EXTRAPOLATION_LIMIT to make this pass."
                )

    if claim in (ClaimClass.TERM_RANKING, ClaimClass.END_TO_END):
        _reject_unverifiable_git_sha(sources, claim)
        for field in _UNANIMITY_FIELDS:
            values = {getattr(r, field) for r in sources}
            if len(values) > 1:
                raise ClaimValidityError(
                    f"{claim.name} sources are not unanimous on {field!r}: "
                    f"{sorted(map(repr, values))} -- a claim mixing these is "
                    "not citable (see spec section P1: measured XLA_FLAGS "
                    "differences alone produced a 1170x throughput change on "
                    "this project's Blackwell nodes)"
                )
