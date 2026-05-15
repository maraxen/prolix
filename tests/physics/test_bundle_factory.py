"""Tests for make_bundle_from_system factory (T6: Phase 1 bridge).

Verifies that PhysicsSystem -> MolecularBundle conversion:
- Returns the correct type
- Pads arrays to the smallest containing bucket
- Preserves the atom mask for real atoms
- Threads boundary_condition through to shape_spec
"""

import jax.numpy as jnp
import pytest

from prolix.physics.system import make_bundle_from_system
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
    # 10 atoms should land in the first bucket (256)
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
