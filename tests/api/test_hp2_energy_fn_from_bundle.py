"""HP2 (#162): ``energy_fn_from_bundle`` parity vs ``single_padded_energy``."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from prolix.api.bundle_md import (
    active_positions,
    displacement_fn_for_bundle,
    energy_fn_from_bundle,
)
from prolix.batched_energy import single_padded_energy
from prolix.physics.system import make_bundle_from_system
from prolix.typing import PhysicsSystem


def _two_atom_lj_system():
    """Minimal nonbonded pair in free space."""
    n = 2
    positions = jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=jnp.float64)
    ones_b = jnp.ones(n, dtype=bool)
    empty2 = jnp.zeros((0, 2), dtype=jnp.int32)
    empty3 = jnp.zeros((0, 3), dtype=jnp.int32)
    empty_p2 = jnp.zeros((0, 2))
    empty_m = jnp.zeros(0, dtype=bool)
    empty_dih_p = jnp.zeros((0, 1, 3))

    return PhysicsSystem(
        positions=positions,
        charges=jnp.array([0.5, -0.5], dtype=jnp.float64),
        sigmas=jnp.array([3.5, 3.5], dtype=jnp.float64),
        epsilons=jnp.array([0.1, 0.1], dtype=jnp.float64),
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
        bond_params=empty_p2,
        bond_mask=empty_m,
        angles=empty3,
        angle_params=empty_p2,
        angle_mask=empty_m,
        dihedrals=jnp.zeros((0, 4), dtype=jnp.int32),
        dihedral_params=jnp.zeros((0, 3), dtype=jnp.float64),
        dihedral_mask=empty_m,
        impropers=jnp.zeros((0, 4), dtype=jnp.int32),
        improper_params=jnp.zeros((0, 3), dtype=jnp.float64),
        improper_mask=empty_m,
        urey_bradley_bonds=jnp.zeros((0, 3), dtype=jnp.int32),
        urey_bradley_params=jnp.zeros((0, 2), dtype=jnp.float64),
        urey_bradley_mask=empty_m,
        n_real_atoms=jnp.array(n, dtype=jnp.int32),
        n_padded_atoms=n,
        pme_alpha=0.0,
    )


def test_energy_fn_from_bundle_matches_single_padded_energy():
    """Bundle-path total energy matches direct PhysicsSystem evaluation."""
    jax.config.update("jax_enable_x64", True)
    sys = _two_atom_lj_system()
    bundle = make_bundle_from_system(sys, boundary_condition="free")
    r = active_positions(bundle)
    n = int(r.shape[0])
    disp_fn, _ = displacement_fn_for_bundle(bundle)

    e_ref = float(
        single_padded_energy(
            sys.replace(
                positions=r,
                dense_excl_scale_vdw=jnp.ones((n, n), dtype=jnp.float32),
                dense_excl_scale_elec=jnp.ones((n, n), dtype=jnp.float32),
            ),
            disp_fn,
            implicit_solvent=False,
        )
    )
    e_bun = float(energy_fn_from_bundle(bundle)(r))
    delta = abs(e_ref - e_bun)
    assert delta < 1e-4, f"NB bundle parity failed: ref={e_ref:.6f}, bundle={e_bun:.6f}, Δ={delta:.3e}"
