"""scripts.profiling -- stage/scale-stamped JAX profiling measurement primitives.

No prolix imports anywhere in this package (grep-enforced in
tests/profiling/test_claim_contract.py) -- this is the seam intended for a
possible later upstream to xtrax.profiling. Whether that upstream ever
happens is governed by a separate trigger (see
.praxia/docs/specs/260817_jax-profiling-optimization-workflow.md section 4);
this package's only job is to not foreclose it.

See that spec's section P1 for the full design rationale behind ProbeRecord
and the claim-validity contract.
"""

from scripts.profiling.claims import (
    CONTRACT_VERSION,
    SCALE_EXTRAPOLATION_LIMIT,
    ClaimClass,
    ClaimValidityError,
    assert_claim_supported,
    paired_configs,
    permitted_claims,
    select_sources,
)
from scripts.profiling.record import ProbeRecord

__all__ = [
    "CONTRACT_VERSION",
    "SCALE_EXTRAPOLATION_LIMIT",
    "ClaimClass",
    "ClaimValidityError",
    "ProbeRecord",
    "assert_claim_supported",
    "paired_configs",
    "permitted_claims",
    "select_sources",
]
