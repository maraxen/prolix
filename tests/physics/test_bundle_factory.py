"""Tests for make_bundle_from_system factory (T6: Phase 1 bridge).

Verifies that PhysicsSystem -> MolecularBundle conversion:
- Returns the correct type
- Pads arrays to the smallest containing bucket
- Preserves the atom mask for real atoms
- Threads boundary_condition through to shape_spec
- Flattens multi-term dihedral/improper params
- Converts dense exclusion layouts to pair lists
"""

import jax.numpy as jnp
import numpy as np
import pytest

from prolix.physics.system import _dense_excl_to_pair_list, make_bundle_from_system
from prolix.types.bundles import ATOM_BUCKETS, MolecularBundle


def _make_minimal_physics_system():
    """Build a minimal PhysicsSystem with 10 real atoms and empty topology."""
    from prolix.typing import PhysicsSystem

    n = 10
    pos = jnp.zeros((n, 3))
    ones_f = jnp.zeros(n)
    ones_b = jnp.ones(n, dtype=bool)
    zeros_b = jnp.zeros(n, dtype=bool)
    elem = jnp.ones(n, dtype=jnp.int32)

    # Empty topology arrays (size == 0 in first dim)
    empty_bonds = jnp.zeros((0, 2), dtype=jnp.int32)
    empty_bond_params = jnp.zeros((0, 2))
    empty_bond_mask = jnp.zeros(0, dtype=bool)

    empty_angles = jnp.zeros((0, 3), dtype=jnp.int32)
    empty_angle_params = jnp.zeros((0, 2))
    empty_angle_mask = jnp.zeros(0, dtype=bool)

    empty_dihedrals = jnp.zeros((0, 4), dtype=jnp.int32)
    # PhysicsSystem stores dihedral_params as 3D: (N, N_terms, 3)
    empty_dihedral_params = jnp.zeros((0, 1, 3))
    empty_dihedral_mask = jnp.zeros(0, dtype=bool)

    empty_impropers = jnp.zeros((0, 4), dtype=jnp.int32)
    empty_improper_params = jnp.zeros((0, 1, 3))
    empty_improper_mask = jnp.zeros(0, dtype=bool)

    return PhysicsSystem(
        positions=pos,
        charges=ones_f,
        sigmas=ones_f,
        epsilons=ones_f,
        radii=ones_f,
        scaled_radii=ones_f,
        masses=ones_f,
        element_ids=elem,
        atom_mask=ones_b,
        is_hydrogen=zeros_b,
        is_backbone=zeros_b,
        is_heavy=zeros_b,
        protein_atom_mask=zeros_b,
        water_atom_mask=zeros_b,
        bonds=empty_bonds,
        bond_params=empty_bond_params,
        bond_mask=empty_bond_mask,
        angles=empty_angles,
        angle_params=empty_angle_params,
        angle_mask=empty_angle_mask,
        dihedrals=empty_dihedrals,
        dihedral_params=empty_dihedral_params,
        dihedral_mask=empty_dihedral_mask,
        impropers=empty_impropers,
        improper_params=empty_improper_params,
        improper_mask=empty_improper_mask,
    )


def test_factory_returns_molecular_bundle():
    sys = _make_minimal_physics_system()
    bundle = make_bundle_from_system(sys, boundary_condition="free")
    assert isinstance(bundle, MolecularBundle)


def test_factory_pads_to_bucket():
    sys = _make_minimal_physics_system()
    bundle = make_bundle_from_system(sys, boundary_condition="free")
    # 10 atoms should land in the first bucket (64)
    assert bundle.positions.shape[0] == ATOM_BUCKETS[0]


def test_factory_preserves_atom_mask():
    sys = _make_minimal_physics_system()
    bundle = make_bundle_from_system(sys, boundary_condition="free")
    assert int(bundle.atom_mask.sum()) == 10  # 10 real atoms, rest padded False


def test_factory_sets_boundary_condition():
    sys = _make_minimal_physics_system()
    bundle = make_bundle_from_system(sys, boundary_condition="periodic")
    assert bundle.shape_spec.boundary_condition == "periodic"
    bundle_free = make_bundle_from_system(sys, boundary_condition="free")
    assert bundle_free.shape_spec.boundary_condition == "free"


def test_factory_flattens_multi_term_dihedrals():
    """(N, T, 3) dihedral_params expand to N*T rows in the bundle."""
    from prolix.typing import PhysicsSystem

    n = 4
    pos = jnp.zeros((n, 3))
    ones_f = jnp.zeros(n)
    ones_b = jnp.ones(n, dtype=bool)
    zeros_b = jnp.zeros(n, dtype=bool)
    elem = jnp.ones(n, dtype=jnp.int32)
    empty = jnp.zeros((0, 2), dtype=jnp.int32)
    empty_p = jnp.zeros((0, 2))
    empty_m = jnp.zeros(0, dtype=bool)
    empty_ang = jnp.zeros((0, 3), dtype=jnp.int32)

    dihs = jnp.array([[0, 1, 2, 3], [0, 1, 2, 4]], dtype=jnp.int32)
    # 2 dihedrals × 2 terms each
    dih_p = jnp.array(
        [
            [[1.0, 0.0, 2.0], [2.0, jnp.pi, 3.0]],
            [[1.0, 0.0, 1.5], [1.0, 0.0, 1.5]],
        ],
        dtype=jnp.float64,
    )

    sys = PhysicsSystem(
        positions=pos,
        charges=ones_f,
        sigmas=ones_f,
        epsilons=ones_f,
        radii=ones_f,
        scaled_radii=ones_f,
        masses=ones_f,
        element_ids=elem,
        atom_mask=ones_b,
        is_hydrogen=zeros_b,
        is_backbone=zeros_b,
        is_heavy=zeros_b,
        protein_atom_mask=zeros_b,
        water_atom_mask=zeros_b,
        bonds=empty,
        bond_params=empty_p,
        bond_mask=empty_m,
        angles=empty_ang,
        angle_params=empty_p,
        angle_mask=empty_m,
        dihedrals=dihs,
        dihedral_params=dih_p,
        dihedral_mask=jnp.ones(2, dtype=bool),
        impropers=jnp.zeros((0, 4), dtype=jnp.int32),
        improper_params=jnp.zeros((0, 1, 3)),
        improper_mask=jnp.zeros(0, dtype=bool),
    )
    bundle = make_bundle_from_system(sys, boundary_condition="free")
    assert int(bundle.n_dihedrals) == 4
    assert bundle.dihedral_params.shape[1] == 3
    assert int(bundle.dihedral_mask.sum()) == 4


def test_dense_excl_to_pair_list_deduplicates():
    """Dense (N, M) excl layout converts to unique (E, 2) pairs."""
    excl = np.array([[1, -1], [0, -1]], dtype=np.int32)
    vdw = np.array([[0.0, 1.0], [0.5, 1.0]], dtype=np.float32)
    elec = np.ones_like(vdw)
    pairs, sv, se, n = _dense_excl_to_pair_list(excl, vdw, elec)
    assert n == 1
    assert pairs is not None
    assert pairs.shape == (1, 2)
    assert list(pairs[0]) == [0, 1]
    assert float(sv[0]) == 0.0
    assert float(se[0]) == 1.0
