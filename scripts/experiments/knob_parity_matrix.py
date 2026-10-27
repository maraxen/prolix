r"""Tier 3 (#6194 / FR-3): Bathos Knob-Parity Coverage Matrix & Synthetic Negative Control.

Evaluates:
1. Consumer-only static AST reachability (U1) across all 72 canonical ``KnobSurface``
   leaf fields, ``NonbondedConfig``, and ``NeighborListConfig``, asserting
   ``set(unread_fields) - DOCUMENTED_INERT_KNOBS == set()`` and
   ``PENDING_EPIC_KNOBS == frozenset()``.
2. Vendor/reference schema superset parity (U5) across all 16 reference config
   schemas (OpenMM, kUPS, ``naurmalade``, and Prolix).
3. Runtime ``FieldAuditProxy`` consumption across real Prolix consumer lowering
   (``prolix.api.bundle_md.lower_knob_surface``) and real Prolix physics kernels
   (``solve_ewald_alpha``, ``single_padded_energy``, ``compute_beutler_softcore_lj_energy``,
   ``compute_positional_restraint_energy``, ``cv_plumed_kd_distance``, and
   ``pbc.invert_box_lower_triangular``).
4. Synthetic instrument-sensitivity negative control: deliberately drops a
   mutated knob (`reaction_field_dielectric`) in a shadow consumer proxy and
   verifies that ``FieldAuditProxy`` + energy-delta detection catches the drop
   (`negative_control_detected = True`). Also honours ``PROLIX_NEG_CONTROL_DROP=1``
   for Bathos ``[differential]`` pre-flight.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
if "/tmp/prolix_py312_site" not in sys.path and Path("/tmp/prolix_py312_site").exists():
    sys.path.insert(0, "/tmp/prolix_py312_site")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax_md import space  # noqa: E402

from prolix.api.bundle_md import lower_knob_surface  # noqa: E402
from prolix.batched_energy import single_padded_energy, solve_ewald_alpha  # noqa: E402
from prolix.governance import (  # noqa: E402
    DOCUMENTED_INERT_KNOBS,
    PENDING_EPIC_KNOBS,
    SUPERSET_ALIAS_MAP,
    check_prolix_reachability,
    check_superset,
    get_reference_config_schemas,
)
from prolix.knob_surface import KnobSurface  # noqa: E402
from prolix.physics import pbc  # noqa: E402
from prolix.physics.collective_variables import cv_plumed_kd_distance  # noqa: E402
from prolix.physics.field_audit import FieldAuditProxy  # noqa: E402
from prolix.physics.softcore_minimizer import (  # noqa: E402
    compute_beutler_softcore_lj_energy,
    compute_positional_restraint_energy,
)
from prolix.typing import PhysicsSystem  # noqa: E402


class _DropKnobShadowProxy:
    """Shadow proxy that intercepts ``drop_knob`` and returns default without touching wrapped proxy."""

    def __init__(self, target: Any, drop_knob: str, default_val: Any) -> None:
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_drop_knob", drop_knob)
        object.__setattr__(self, "_default_val", default_val)

    def __getattr__(self, name: str) -> Any:
        target = object.__getattribute__(self, "_target")
        drop_knob = object.__getattribute__(self, "_drop_knob")
        default_val = object.__getattribute__(self, "_default_val")
        if name == drop_knob:
            return default_val
        val = getattr(target, name)
        if isinstance(val, FieldAuditProxy) or dataclasses.is_dataclass(val):
            return _DropKnobShadowProxy(val, drop_knob, default_val)
        return val


def _lower_all_knobs_runtime(surface_or_proxy: Any, *, drop_knob: str | None = None) -> dict[str, Any]:
    """Lower ``KnobSurface`` through the real ``prolix.api.bundle_md.lower_knob_surface`` consumer."""
    if drop_knob is not None:
        default_val = getattr(KnobSurface(), drop_knob)
        consumer_input = _DropKnobShadowProxy(surface_or_proxy, drop_knob, default_val)
    else:
        consumer_input = surface_or_proxy
    return dict(lower_knob_surface(consumer_input))


def _compute_reference_energy_from_lowered(cfg: dict[str, Any]) -> float:
    """Evaluate real Prolix physics kernels parameterized by lowered KnobSurface kwargs."""
    r = 4.2
    pos = jnp.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=jnp.float32)
    charges = jnp.array([0.5, -0.5], dtype=jnp.float32)
    sigmas = jnp.array([1.0, 1.0], dtype=jnp.float32)
    epsilons = jnp.array([0.0, 0.0], dtype=jnp.float32)
    mask = jnp.array([True, True], dtype=jnp.bool_)
    box = np.array([25.0, 25.0, 25.0], dtype=np.float32)
    disp_fn, _ = space.periodic(jnp.asarray(box))

    rc = float(cfg.get("cutoff_angstrom", 9.0))
    eps_rf = float(cfg.get("reaction_field_dielectric", 78.3))
    coulomb = float(cfg.get("coulomb_constant", 332.0637))
    sys_rf = PhysicsSystem(
        positions=pos,
        charges=charges,
        sigmas=sigmas,
        epsilons=epsilons,
        radii=jnp.ones(2, dtype=jnp.float32) * 1.5,
        scaled_radii=jnp.ones(2, dtype=jnp.float32) * 1.2,
        masses=jnp.ones(2, dtype=jnp.float32) * 12.0,
        element_ids=jnp.ones(2, dtype=jnp.int32) * 6,
        atom_mask=mask,
        is_hydrogen=jnp.zeros(2, dtype=bool),
        is_backbone=jnp.zeros(2, dtype=bool),
        is_heavy=mask,
        protein_atom_mask=mask,
        water_atom_mask=jnp.zeros(2, dtype=bool),
        bonds=jnp.zeros((1, 2), dtype=jnp.int32),
        bond_params=jnp.zeros((1, 2), dtype=jnp.float32),
        bond_mask=jnp.zeros(1, dtype=bool),
        angles=jnp.zeros((1, 3), dtype=jnp.int32),
        angle_params=jnp.zeros((1, 2), dtype=jnp.float32),
        angle_mask=jnp.zeros(1, dtype=bool),
        dihedrals=jnp.zeros((1, 4), dtype=jnp.int32),
        dihedral_params=jnp.zeros((1, 3), dtype=jnp.float32),
        dihedral_mask=jnp.zeros(1, dtype=bool),
        impropers=jnp.zeros((1, 4), dtype=jnp.int32),
        improper_params=jnp.zeros((1, 3), dtype=jnp.float32),
        improper_mask=jnp.zeros(1, dtype=bool),
        urey_bradley_bonds=jnp.zeros((1, 2), dtype=jnp.int32),
        urey_bradley_params=jnp.zeros((1, 2), dtype=jnp.float32),
        urey_bradley_mask=jnp.zeros(1, dtype=bool),
        box_size=box,
        nonbonded_cutoff=rc,
        electrostatic_method="reaction_field",
        reaction_field_dielectric=eps_rf,
        use_dispersion_correction=False,
        dense_excl_scale_vdw=jnp.ones((2, 2), dtype=jnp.float32),
        dense_excl_scale_elec=jnp.ones((2, 2), dtype=jnp.float32),
    )
    e_rf = float(single_padded_energy(sys_rf, disp_fn, implicit_solvent=False)) * (coulomb / 332.0637)

    e_sc_lj = float(
        compute_beutler_softcore_lj_energy(
            jnp.array([r], dtype=jnp.float32),
            None,
            jnp.array([3.5], dtype=jnp.float32),
            jnp.array([0.15], dtype=jnp.float32),
            jnp.array([True], dtype=jnp.bool_),
            lambda_lj=0.5,
            softcore_alpha=float(cfg.get("softcore_alpha", 0.5)),
            softcore_exponent=int(cfg.get("softcore_exponent", 2)),
        )
    )

    ref_pos = jnp.array([[0.0, 0.0, 0.0], [r - 0.5, 0.0, 0.0]], dtype=jnp.float32)
    e_rest = float(
        compute_positional_restraint_energy(
            pos,
            ref_pos,
            jnp.array([False, True], dtype=jnp.bool_),
            disp_fn,
            k_kcal_mol_a2=float(cfg.get("restraint_k_kcal_mol_a2", 1.0)),
            flat_bottom_radius_angstrom=float(cfg.get("flat_bottom_radius_angstrom", 0.0)),
        )
    )

    pos_kd = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]], dtype=jnp.float32)
    kd_dist = float(
        cv_plumed_kd_distance(
            pos_kd,
            jnp.array([[0, 1], [0, 2]], dtype=jnp.int32),
            jnp.array([True, True], dtype=jnp.bool_),
            disp_fn,
            beta_nm=float(cfg.get("plumed_min_beta_nm", 20.0)),
        )
    )
    return float(e_rf + e_sc_lj + e_rest + 0.01 * kd_dist)


def _run_numerical_parity_probes() -> tuple[int, float, dict[str, float]]:
    """Verify real Prolix kernel implementations against exact analytical reference formulas."""
    errors: dict[str, float] = {}

    # 1. Ewald alpha solver: erfc(alpha * r_cut) / r_cut == tol
    rc = 9.0
    tol = 5e-4
    alpha = solve_ewald_alpha(rc, tol)
    residual = abs(math.erfc(alpha * rc) / rc - tol) / tol
    errors["ewald_alpha_rel_err"] = float(residual)

    # 2. Beutler squared softcore (1 - lambda)^2 at lambda=0.5, r=sigma=3.0, alpha=0.5, eps=1.0
    e_sc_actual = float(
        compute_beutler_softcore_lj_energy(
            jnp.array([3.0], dtype=jnp.float32),
            None,
            jnp.array([3.0], dtype=jnp.float32),
            jnp.array([1.0], dtype=jnp.float32),
            jnp.array([True], dtype=jnp.bool_),
            lambda_lj=0.5,
            softcore_alpha=0.5,
            softcore_exponent=2,
        )
    )
    s_expected = 1.125
    e_sc_expected = 4.0 * 1.0 * 0.5 * (1.0 / (s_expected**2) - 1.0 / s_expected)
    errors["softcore_denom_rel_err"] = abs(e_sc_actual - e_sc_expected) / abs(e_sc_expected)

    # 3. PLUMED MIN smooth minimum distance (Spike 7a: beta_nm=20 -> beta_A=200, d1=d2=3.0 A)
    pos_kd = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]], dtype=jnp.float32)
    d_kd_actual = float(
        cv_plumed_kd_distance(
            pos_kd,
            jnp.array([[0, 1], [0, 2]], dtype=jnp.int32),
            jnp.array([True, True], dtype=jnp.bool_),
            beta_nm=20.0,
        )
    )
    d_kd_expected = 200.0 / (200.0 / 3.0 + math.log(2.0))
    errors["plumed_kd_rel_err"] = abs(d_kd_actual - d_kd_expected) / d_kd_expected

    # 4. Triclinic closed-form reciprocal matrix B = A^-1 via pbc.invert_box_lower_triangular
    box = jnp.array(
        [[40.0, 0.0, 0.0], [12.0, 38.0, 0.0], [-8.0, 15.0, 35.0]],
        dtype=jnp.float32,
    )
    b_mat = pbc.invert_box_lower_triangular(box)
    ident_err = float(jnp.max(jnp.abs((box @ b_mat) - jnp.eye(3, dtype=jnp.float32))))
    errors["triclinic_inv_err"] = ident_err

    # 5. OpenMM unhalved restraint convention: k_kcal_a2 = 1.0, E(dr=2.0) = 4.0
    pos_r = jnp.array([[2.0, 0.0, 0.0]], dtype=jnp.float32)
    ref_r = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
    free_disp, _ = space.free()
    e_rest = float(
        compute_positional_restraint_energy(
            pos_r,
            ref_r,
            jnp.array([True], dtype=jnp.bool_),
            free_disp,
            k_kcal_mol_a2=1.0,
        )
    )
    errors["restraint_rel_err"] = abs(e_rest - 4.0) / 4.0

    max_err = max(errors.values()) if errors else 0.0
    n_failures = sum(1 for v in errors.values() if v >= 1e-5)
    return n_failures, float(max_err), errors


def evaluate_knob_parity_matrix(*, inject_drop: bool = False) -> dict[str, Any]:
    """Run the full static U1/U5 + runtime FieldAuditProxy + negative-control matrix."""
    # 1. Static U1 consumer-only reachability
    u1_res = check_prolix_reachability(_REPO_ROOT)
    unaccounted_u1 = (
        set(u1_res.unread_fields) - PENDING_EPIC_KNOBS - DOCUMENTED_INERT_KNOBS
    )

    # 2. Static U5 vendor/schema superset parity across all 16 reference schemas
    schemas = get_reference_config_schemas()
    unmapped_u5: set[str] = set()
    per_schema_u5: dict[str, list[str]] = {}
    for name, cls in schemas.items():
        sup_res = check_superset(KnobSurface, cls, SUPERSET_ALIAS_MAP)
        if not sup_res.is_superset:
            unmapped_u5 |= set(sup_res.missing_fields)
            per_schema_u5[name] = sorted(sup_res.missing_fields)

    # 3. Runtime FieldAuditProxy over real lower_knob_surface consumer
    surface = KnobSurface.from_flat_dict(
        {
            "reaction_field_dielectric": 80.0,
            "softcore_alpha": 0.5,
            "restraint_k_kcal_mol_a2": 2.5,
            "plumed_min_beta_nm": 20.0,
        }
    )
    proxy = FieldAuditProxy(surface)
    dropped_in_main = "reaction_field_dielectric" if inject_drop else None
    lowered_main = _lower_all_knobs_runtime(proxy, drop_knob=dropped_in_main)
    runtime_unread = KnobSurface.leaf_field_names() - proxy.accessed_fields

    # 4. Synthetic negative control: deliberately drop a mutated knob in a shadow consumer proxy
    mutated_surface = KnobSurface.from_flat_dict(
        {
            "reaction_field_dielectric": 2.0,  # mutated from 78.3 -> 2.0
            "softcore_alpha": 0.25,
            "restraint_k_kcal_mol_a2": 5.0,
            "plumed_min_beta_nm": 20.0,
        }
    )
    baseline_surface = KnobSurface()
    shadow_proxy = FieldAuditProxy(mutated_surface)
    shadow_lowered = _lower_all_knobs_runtime(
        shadow_proxy, drop_knob="reaction_field_dielectric"
    )
    full_lowered = _lower_all_knobs_runtime(mutated_surface, drop_knob=None)
    baseline_lowered = _lower_all_knobs_runtime(baseline_surface, drop_knob=None)

    shadow_missing = KnobSurface.leaf_field_names() - shadow_proxy.accessed_fields
    e_shadow = _compute_reference_energy_from_lowered(shadow_lowered)
    e_full = _compute_reference_energy_from_lowered(full_lowered)
    e_base = _compute_reference_energy_from_lowered(baseline_lowered)
    negative_control_detected = (
        ("reaction_field_dielectric" in shadow_missing)
        and (abs(e_full - e_shadow) > 1e-3)
        and (abs(e_full - e_base) > 1e-3)
    )

    # 5. Numerical parity probes
    n_parity_failures, max_rel_energy_err, probe_errors = _run_numerical_parity_probes()

    n_silent_drops = len(unaccounted_u1) + len(runtime_unread)
    n_untracked_gaps = len(unmapped_u5) + len(PENDING_EPIC_KNOBS)

    return {
        "n_silent_drops": int(n_silent_drops),
        "n_parity_failures": int(n_parity_failures),
        "n_untracked_gaps": int(n_untracked_gaps),
        "max_rel_energy_err": float(max_rel_energy_err),
        "negative_control_detected": bool(negative_control_detected),
        "n_total_knobs": int(len(KnobSurface.leaf_field_names())),
        "n_pending_epic_knobs": int(len(PENDING_EPIC_KNOBS)),
        "u1_unaccounted_knobs": sorted(unaccounted_u1),
        "runtime_unread_knobs": sorted(runtime_unread),
        "u5_unmapped_by_schema": per_schema_u5,
        "probe_errors": probe_errors,
        "sample_energy_kcal_mol": _compute_reference_energy_from_lowered(lowered_main),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Prolix Knob-Parity Coverage Matrix and Synthetic Negative Control."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "experiments" / "governance" / "knob_parity_matrix_results.json",
        help="Output JSON path (default: experiments/governance/knob_parity_matrix_results.json)",
    )
    parser.add_argument(
        "--inject-drop",
        action="store_true",
        help="Deliberately inject a silent knob drop (used for differential pre-flight testing)",
    )
    args = parser.parse_args(argv)

    env_inject = os.environ.get("PROLIX_NEG_CONTROL_DROP", "0") == "1"
    result = evaluate_knob_parity_matrix(inject_drop=bool(args.inject_drop or env_inject))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    bth_results_path = os.environ.get("BTH_RESULTS_PATH")
    if bth_results_path:
        bth_path = Path(bth_results_path)
        bth_path.parent.mkdir(parents=True, exist_ok=True)
        bth_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2))
    if (
        result["n_silent_drops"] > 0
        or result["n_parity_failures"] > 0
        or result["n_untracked_gaps"] > 0
        or not result["negative_control_detected"]
    ):
        return 1 if not (args.inject_drop or env_inject) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
