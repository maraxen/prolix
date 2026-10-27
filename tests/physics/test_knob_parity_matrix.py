"""CI test suite for Tier 3 (#6194 / FR-3): Bathos Knob-Parity Matrix & Union Gate Claim."""

from __future__ import annotations

from pathlib import Path
import sys
import tomllib

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from scripts.experiments.knob_parity_matrix import (  # noqa: E402
    evaluate_knob_parity_matrix,
)
from prolix.governance.pending_knobs import PENDING_EPIC_KNOBS  # noqa: E402


def test_pending_epic_knobs_empty_at_tier_3() -> None:
    """Assert PENDING_EPIC_KNOBS is completely empty now that #6185-#6193 are complete."""
    assert PENDING_EPIC_KNOBS == frozenset(), (
        f"Expected PENDING_EPIC_KNOBS == frozenset(), got {sorted(PENDING_EPIC_KNOBS)}"
    )
    assert len(PENDING_EPIC_KNOBS) == 0


def test_knob_parity_matrix_passes_with_negative_control() -> None:
    """Verify 0 silent drops, 0 parity failures, 0 untracked gaps, and negative control detection."""
    res = evaluate_knob_parity_matrix(inject_drop=False)
    assert res["n_silent_drops"] == 0, f"Unexpected silent drops: {res}"
    assert res["n_parity_failures"] == 0, f"Unexpected parity failures: {res}"
    assert res["n_untracked_gaps"] == 0, f"Unexpected untracked gaps: {res}"
    assert res["max_rel_energy_err"] < 1e-5, f"Relative error too high: {res}"
    assert res["negative_control_detected"] is True
    assert res["n_total_knobs"] == 72
    assert res["n_pending_epic_knobs"] == 0

    # Verify differential injection increases n_silent_drops by >= 1
    res_injected = evaluate_knob_parity_matrix(inject_drop=True)
    assert res_injected["n_silent_drops"] >= 1
    assert "reaction_field_dielectric" in res_injected["runtime_unread_knobs"]


def test_bathos_sidecar_and_claim_toml_contracts() -> None:
    """Verify .bth.toml sidecars and .bth/claims/knob-governance-matrix.claim.toml contracts."""
    for sidecar_rel in (
        "scripts/experiments/knob_parity_matrix.bth.toml",
        "experiments/governance/knob_parity_matrix.bth.toml",
    ):
        sidecar_path = _REPO_ROOT / sidecar_rel
        assert sidecar_path.exists(), f"Missing sidecar: {sidecar_path}"
        sidecar = tomllib.loads(sidecar_path.read_text(encoding="utf-8"))
        assert sidecar["experiment"]["stage_name"] == "validation"
        assert sidecar["experiment"]["novel"] is True
        schema = sidecar["result_schema"]
        for req_key in (
            "n_silent_drops",
            "n_parity_failures",
            "n_untracked_gaps",
            "max_rel_energy_err",
            "negative_control_detected",
        ):
            assert req_key in schema
        outcomes = sidecar["outcomes"]
        residual_keys = [k for k, v in outcomes.items() if v.get("is_residual") is True]
        assert residual_keys == ["fail"]

    claim_path = _REPO_ROOT / ".bth" / "claims" / "knob-governance-matrix.claim.toml"
    assert claim_path.exists()
    claim_doc = tomllib.loads(claim_path.read_text(encoding="utf-8"))
    assert claim_doc["claim"]["kill_condition_satisfiable_by_null"] is True
    assert len(claim_doc["hypotheses"]) >= 2
    clauses = claim_doc["claim"]["union_gate"]["clauses"]
    assert any(c.get("positive_control") is True for c in clauses)


if __name__ == "__main__":
    test_pending_epic_knobs_empty_at_tier_3()
    test_knob_parity_matrix_passes_with_negative_control()
    test_bathos_sidecar_and_claim_toml_contracts()
    print("All Tier 3 (#6194) Bathos knob-parity matrix tests passed.")
