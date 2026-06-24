"""HP2 (#162): ``make_bundle_from_system`` parity vs PhysicsSystem energy path.

Gate: bonded total energy from ``bonded_energy_fn_from_bundle`` matches the
direct PhysicsSystem bonded factories on alanine dipeptide (OpenMM reference).
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.openmm

from prolix.api.bundle_md import bonded_energy_fn_from_bundle
from prolix.physics.system import _dense_excl_to_pair_list, make_bundle_from_system

from .fixtures_openmm_parity import (
    build_ala_dip_openmm_system,
    build_prolix_bonded_system,
    extract_bonded_params,
    get_prolix_per_term_energies,
)


def test_hp2_bonded_total_energy_matches_physics_system_path():
    """Bundle factory bonded energy matches direct PhysicsSystem evaluation."""
    omm_system, positions_ang, _ = build_ala_dip_openmm_system()
    bonded_params = extract_bonded_params(omm_system)
    sys, disp = build_prolix_bonded_system(bonded_params, positions_ang)

    terms = get_prolix_per_term_energies(sys, disp, positions_ang)
    ps_total = sum(terms[k] for k in ("bonds", "angles", "dihedrals", "impropers"))

    bundle = make_bundle_from_system(sys, boundary_condition="free")
    positions = jnp.asarray(positions_ang, dtype=jnp.float32)
    bundle_total = float(bonded_energy_fn_from_bundle(bundle)(positions))

    delta = abs(ps_total - bundle_total)
    assert delta < 1e-4, (
        f"HP2 bonded parity failed: PS={ps_total:.6f}, bundle={bundle_total:.6f}, "
        f"delta={delta:.3e} kcal/mol"
    )


def test_hp2_exclusion_spec_maps_to_bundle_pairs():
    """ExclusionSpec populates bundle excl/exception pair fields."""
    from prolix.physics.neighbor_list import map_exclusions_to_dense_padded

    from .fixtures_openmm_parity import build_exclusion_spec

    omm_system, positions_ang, _ = build_ala_dip_openmm_system()
    bonded_params = extract_bonded_params(omm_system)
    sys, _ = build_prolix_bonded_system(bonded_params, positions_ang)

    excl_spec = build_exclusion_spec(omm_system, positions_ang.shape[0])
    bundle = make_bundle_from_system(sys, boundary_condition="free", exclusion_spec=excl_spec)

    assert int(bundle.n_excl) > 0
    assert int(bundle.n_exception_pairs) > 0
    assert bundle.excl_indices.shape[1] == 2
    assert int(bundle.excl_mask.sum()) == int(bundle.n_excl)

    dense_i, dense_sv, dense_se = map_exclusions_to_dense_padded(excl_spec)
    pairs, _, _, n = _dense_excl_to_pair_list(dense_i, dense_sv, dense_se)
    assert pairs is not None
    assert n == int(bundle.n_excl)
