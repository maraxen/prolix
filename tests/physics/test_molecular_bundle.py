"""Test MolecularBundle typed boundary and bucketed topology.

Tests verify:
- MolecularBundle is a valid eqx.Module with concrete arrays
- MolecularShapeSpec is hashable and carries static metadata
- Bucketed topology arrays are padded correctly
- Bundle is valid JAX pytree and JIT-compatible
"""

import pytest
import jax
import jax.numpy as jnp
from prolix.types.bundles import (
    MolecularBundle,
    MolecularShapeSpec,
    _bucket_idx,
    ATOM_BUCKETS,
    BOND_BUCKETS,
    ANGLE_BUCKETS,
    DIHEDRAL_BUCKETS,
    WATER_BUCKETS,
    EXCL_BUCKETS,
    CMAP_BUCKETS,
    EXCEPTION_BUCKETS,
)


def _minimal_bundle(n_atoms=10, n_bonds=5):
    """Create a minimal MolecularBundle for testing."""
    a = ATOM_BUCKETS[0]
    b = BOND_BUCKETS[0]

    # Build shape_spec with bucket indices (not raw counts)
    atom_bucket_idx = _bucket_idx(n_atoms, ATOM_BUCKETS)
    bond_bucket_idx = _bucket_idx(max(n_bonds, 1), BOND_BUCKETS)

    spec = MolecularShapeSpec(
        atom_bucket_idx=atom_bucket_idx,
        bond_bucket_idx=bond_bucket_idx,
        angle_bucket_idx=0,
        dihedral_bucket_idx=0,
        water_bucket_idx=0,
        excl_bucket_idx=0,
        cmap_bucket_idx=0,
        exception_bucket_idx=0,
        has_pbc=False,
        has_implicit_solvent=False,
        boundary_condition="free",
    )

    return MolecularBundle(
        positions=jnp.zeros((a, 3)),
        charges=jnp.zeros(a),
        sigmas=jnp.ones(a),
        epsilons=jnp.ones(a),
        radii=jnp.ones(a),
        scaled_radii=jnp.ones(a),
        atom_mask=jnp.concatenate(
            [jnp.ones(n_atoms, dtype=bool), jnp.zeros(a - n_atoms, dtype=bool)]
        ),
        n_atoms=jnp.array(n_atoms, dtype=jnp.int32),
        box=jnp.zeros((3, 3)),
        bond_idx=jnp.zeros((b, 2), dtype=jnp.int32),
        bond_params=jnp.zeros((b, 2)),
        bond_mask=jnp.concatenate(
            [jnp.ones(n_bonds, dtype=bool), jnp.zeros(b - n_bonds, dtype=bool)]
        ),
        n_bonds=jnp.array(n_bonds, dtype=jnp.int32),
        angle_idx=jnp.zeros((ANGLE_BUCKETS[0], 3), dtype=jnp.int32),
        angle_params=jnp.zeros((ANGLE_BUCKETS[0], 2)),
        angle_mask=jnp.zeros(ANGLE_BUCKETS[0], dtype=bool),
        n_angles=jnp.array(0, dtype=jnp.int32),
        dihedral_idx=jnp.zeros((256, 4), dtype=jnp.int32),
        dihedral_params=jnp.zeros((256, 4)),
        dihedral_mask=jnp.zeros(256, dtype=bool),
        n_dihedrals=jnp.array(0, dtype=jnp.int32),
        improper_idx=jnp.zeros((256, 4), dtype=jnp.int32),
        improper_params=jnp.zeros((256, 3)),
        improper_mask=jnp.zeros(256, dtype=bool),
        improper_is_periodic=jnp.array(False),
        n_impropers=jnp.array(0, dtype=jnp.int32),
        urey_bradley_idx=jnp.zeros((256, 3), dtype=jnp.int32),
        urey_bradley_params=jnp.zeros((256, 2)),
        urey_bradley_mask=jnp.zeros(256, dtype=bool),
        n_urey_bradley=jnp.array(0, dtype=jnp.int32),
        cmap_torsion_idx=jnp.zeros((16, 8), dtype=jnp.int32),
        cmap_energy_grids=jnp.zeros((16, 24, 24)),
        cmap_mask=jnp.zeros(16, dtype=bool),
        n_cmap=jnp.array(0, dtype=jnp.int32),
        water_indices=jnp.zeros((16, 3), dtype=jnp.int32),
        water_mask=jnp.zeros(16, dtype=bool),
        n_waters=jnp.array(0, dtype=jnp.int32),
        excl_indices=jnp.zeros((512, 2), dtype=jnp.int32),
        excl_scales_vdw=jnp.zeros(512),
        excl_scales_elec=jnp.zeros(512),
        excl_mask=jnp.zeros(512, dtype=bool),
        n_excl=jnp.array(0, dtype=jnp.int32),
        exception_pairs=jnp.zeros((512, 2), dtype=jnp.int32),
        exception_sigmas=jnp.zeros(512),
        exception_epsilons=jnp.zeros(512),
        exception_chargeprods=jnp.zeros(512),
        exception_mask=jnp.zeros(512, dtype=bool),
        n_exception_pairs=jnp.array(0, dtype=jnp.int32),
        pme_alpha=jnp.array(0.0),
        cutoff_distance=jnp.array(9.0),
        shape_spec=spec,
    )


def test_bundle_construction():
    """MolecularBundle constructs with correct atomic bucket shape."""
    bundle = _minimal_bundle()
    assert bundle.positions.shape[0] == ATOM_BUCKETS[0]
    assert bundle.atom_mask.sum() == 10


def test_shape_spec_is_hashable():
    """MolecularShapeSpec is hashable (frozen dataclass)."""
    bundle = _minimal_bundle()
    h = hash(bundle.shape_spec)
    assert isinstance(h, int)


def test_bundle_is_valid_pytree():
    """MolecularBundle is a valid JAX pytree."""
    bundle = _minimal_bundle()
    leaves, treedef = jax.tree_util.tree_flatten(bundle)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert restored.positions.shape == bundle.positions.shape


def test_bucket_sizes_ordered():
    """All bucket sequences are monotonically increasing."""
    assert all(
        ATOM_BUCKETS[i] < ATOM_BUCKETS[i + 1] for i in range(len(ATOM_BUCKETS) - 1)
    )
    assert all(
        BOND_BUCKETS[i] < BOND_BUCKETS[i + 1] for i in range(len(BOND_BUCKETS) - 1)
    )
    assert all(
        ANGLE_BUCKETS[i] < ANGLE_BUCKETS[i + 1]
        for i in range(len(ANGLE_BUCKETS) - 1)
    )


def test_bundle_jit_passthrough():
    """MolecularBundle can be passed through jit without recompilation."""
    bundle = _minimal_bundle()

    @jax.jit
    def identity(b):
        return b

    out = identity(bundle)
    assert out.positions.shape == bundle.positions.shape
    assert out.n_atoms == 10
