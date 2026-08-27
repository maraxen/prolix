"""CPU match: remat on/off and T=128 vs T=256 on a tiny lattice."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from prolix.padding import PaddedSystem
from prolix.physics.flash_explicit import flash_explicit_energy, flash_explicit_forces


def _tiny_system() -> PaddedSystem:
    n = 8
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [5.0, 5.0, 0.0],
            [2.5, 2.5, 0.0],
            [7.5, 2.5, 0.0],
            [2.5, 7.5, 0.0],
            [7.5, 7.5, 0.0],
        ],
        dtype=jnp.float32,
    )
    atom_mask = jnp.ones(n, dtype=jnp.bool_)
    return PaddedSystem(
        positions=positions,
        charges=jnp.ones(n, dtype=jnp.float32) * 0.5,
        sigmas=jnp.ones(n, dtype=jnp.float32) * 3.0,
        epsilons=jnp.ones(n, dtype=jnp.float32) * 0.1,
        radii=jnp.ones(n, dtype=jnp.float32) * 1.5,
        scaled_radii=jnp.ones(n, dtype=jnp.float32) * 0.8,
        masses=jnp.ones(n, dtype=jnp.float32) * 12.0,
        element_ids=jnp.ones(n, dtype=jnp.int32) * 6,
        atom_mask=atom_mask,
        is_hydrogen=jnp.zeros(n, dtype=jnp.bool_),
        is_backbone=jnp.ones(n, dtype=jnp.bool_),
        is_heavy=jnp.ones(n, dtype=jnp.bool_),
        protein_atom_mask=jnp.ones(n, dtype=jnp.bool_),
        water_atom_mask=jnp.zeros(n, dtype=jnp.bool_),
        bonds=jnp.zeros((0, 2), dtype=jnp.int32),
        bond_params=jnp.zeros((0, 2), dtype=jnp.float32),
        bond_mask=jnp.zeros((0,), dtype=jnp.bool_),
        angles=jnp.zeros((0, 3), dtype=jnp.int32),
        angle_params=jnp.zeros((0, 2), dtype=jnp.float32),
        angle_mask=jnp.zeros((0,), dtype=jnp.bool_),
        dihedrals=jnp.zeros((0, 4), dtype=jnp.int32),
        dihedral_params=jnp.zeros((0, 3), dtype=jnp.float32),
        dihedral_mask=jnp.zeros((0,), dtype=jnp.bool_),
        impropers=jnp.zeros((0, 4), dtype=jnp.int32),
        improper_params=jnp.zeros((0, 3), dtype=jnp.float32),
        improper_mask=jnp.zeros((0,), dtype=jnp.bool_),
        urey_bradley_bonds=None,
        urey_bradley_params=None,
        urey_bradley_mask=None,
        excl_indices=jnp.full((n, 4), -1, dtype=jnp.int32),
        excl_scales_vdw=jnp.ones((n, 4), dtype=jnp.float32),
        excl_scales_elec=jnp.ones((n, 4), dtype=jnp.float32),
        n_padded_atoms=n,
        pme_alpha=0.0,
        box_size=None,
    )


@pytest.fixture
def sys():
    return _tiny_system()


def test_remat_off_matches_on_energy_and_forces(sys):
    e_on = flash_explicit_energy(sys, T=8, remat=True)
    e_off = flash_explicit_energy(sys, T=8, remat=False)
    f_on = flash_explicit_forces(sys, T=8, remat=True)
    f_off = flash_explicit_forces(sys, T=8, remat=False)
    mask = sys.atom_mask
    np.testing.assert_allclose(e_on, e_off, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(f_on[mask]), np.asarray(f_off[mask]), rtol=1e-5, atol=1e-5)


def test_tile_128_matches_256_on_real_atoms(sys):
    e_128 = flash_explicit_energy(sys, T=128, remat=True)
    e_256 = flash_explicit_energy(sys, T=256, remat=True)
    f_128 = flash_explicit_forces(sys, T=128, remat=True)
    f_256 = flash_explicit_forces(sys, T=256, remat=True)
    mask = sys.atom_mask
    np.testing.assert_allclose(e_128, e_256, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(f_128[mask]), np.asarray(f_256[mask]), rtol=1e-4, atol=1e-4)
