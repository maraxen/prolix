"""OpenMM-shaped LJ switch polynomial (unit, no OpenMM import)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from prolix.physics.lj_switching import lj_switch_weight


def test_switch_off_is_ones():
    r = jnp.array([1.0, 8.5, 9.0, 10.0])
    w = lj_switch_weight(r, cutoff=9.0, switch_width=None)
    np.testing.assert_allclose(np.asarray(w), 1.0)


def test_switch_plateau_and_zero_past_cutoff():
    cutoff = 9.0
    width = 1.0
    r = jnp.array([7.0, 8.0, 8.5, 9.0, 9.5])
    w = np.asarray(lj_switch_weight(r, cutoff=cutoff, switch_width=width))
    np.testing.assert_allclose(w[0], 1.0)
    np.testing.assert_allclose(w[1], 1.0)
    assert 0.0 < w[2] < 1.0
    np.testing.assert_allclose(w[3], 0.0)
    np.testing.assert_allclose(w[4], 0.0)


def test_mid_switch_openmm_polynomial():
    cutoff = 9.0
    width = 1.0
    r = 8.5
    z = (r - 8.0) / 1.0
    expected = 1.0 - 10.0 * z**3 + 15.0 * z**4 - 6.0 * z**5
    w = float(lj_switch_weight(jnp.array([r]), cutoff=cutoff, switch_width=width)[0])
    assert abs(w - expected) < 1e-6


def test_energy_fn_from_bundle_closes_over_lj_switch_width():
    """NL pair in the OpenMM switch window must see switch_width via the bundle energy fn."""
    import jax.numpy as jnp

    from prolix.api.bundle_md import energy_fn_from_bundle
    from prolix.physics.neighbor_list import ExclusionSpec
    from prolix.physics.system import make_bundle_from_system
    from prolix.typing import PhysicsSystem

    n = 2
    ones_b = jnp.ones(n, dtype=bool)
    empty2 = jnp.zeros((0, 2), dtype=jnp.int32)
    empty3 = jnp.zeros((0, 3), dtype=jnp.int32)
    sys = PhysicsSystem(
        positions=jnp.array([[0.0, 0.0, 0.0], [8.5, 0.0, 0.0]], dtype=jnp.float64),
        charges=jnp.zeros(n, dtype=jnp.float64),
        sigmas=jnp.array([3.0, 3.0], dtype=jnp.float64),
        epsilons=jnp.array([0.2, 0.2], dtype=jnp.float64),
        radii=jnp.ones(n),
        scaled_radii=jnp.ones(n),
        masses=jnp.ones(n),
        element_ids=jnp.zeros(n, dtype=jnp.int32),
        atom_mask=ones_b,
        is_hydrogen=jnp.zeros(n, dtype=bool),
        is_backbone=jnp.zeros(n, dtype=bool),
        is_heavy=ones_b,
        protein_atom_mask=ones_b,
        water_atom_mask=jnp.zeros(n, dtype=bool),
        bonds=empty2,
        bond_params=jnp.zeros((0, 2)),
        bond_mask=jnp.zeros(0, dtype=bool),
        angles=empty3,
        angle_params=jnp.zeros((0, 2)),
        angle_mask=jnp.zeros(0, dtype=bool),
        dihedrals=jnp.zeros((0, 4), dtype=jnp.int32),
        dihedral_params=jnp.zeros((0, 3)),
        dihedral_mask=jnp.zeros(0, dtype=bool),
        impropers=jnp.zeros((0, 4), dtype=jnp.int32),
        improper_params=jnp.zeros((0, 3)),
        improper_mask=jnp.zeros(0, dtype=bool),
        excl_indices=jnp.full((n, 2), -1, dtype=jnp.int32),
        excl_scales_vdw=jnp.ones((n, 2)),
        excl_scales_elec=jnp.ones((n, 2)),
        n_real_atoms=jnp.array(n, dtype=jnp.int32),
        n_padded_atoms=n,
        pme_alpha=0.0,
        box_size=jnp.array([20.0, 20.0, 20.0]),
        nonbonded_cutoff=9.0,
    )
    excl_spec = ExclusionSpec(
        n_atoms=n,
        idx_12_13=jnp.zeros((0, 2), dtype=jnp.int32),
        idx_14=jnp.zeros((0, 2), dtype=jnp.int32),
        scale_14_elec=1.0,
        scale_14_vdw=1.0,
        exception_pairs=jnp.zeros((0, 2), dtype=jnp.int32),
        exception_sigmas=jnp.zeros(0),
        exception_epsilons=jnp.zeros(0),
        exception_chargeprods=jnp.zeros(0),
    )
    bundle = make_bundle_from_system(
        sys,
        boundary_condition="periodic",
        exclusion_spec=excl_spec,
        use_size_buckets=False,
    )
    n_b = int(bundle.positions.shape[0])
    nb = jnp.full((n_b, 1), -1, dtype=jnp.int32)
    nb = nb.at[0, 0].set(1)
    nb = nb.at[1, 0].set(0)
    r = bundle.positions
    e_off = float(energy_fn_from_bundle(bundle, lj_switch_width=0.0)(r, neighbor=nb))
    e_on = float(energy_fn_from_bundle(bundle, lj_switch_width=1.0)(r, neighbor=nb))
    assert abs(e_off - e_on) > 1e-4
