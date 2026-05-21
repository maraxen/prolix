"""Tests for build_fitting_bundle factory (Phase 4 §5).

Validates optional field resolution, shape normalization, invariant checking,
and round-trip behavior of the bundle builder.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prolix.fitting import BondedParams, BondedTopology
from prolix.fitting.bundle_builder import build_fitting_bundle
from prolix.fitting.bundles import ConformerBundle, FittingBundle


# ===== FIXTURES =====


@pytest.fixture
def minimal_topology():
    """Minimal bonded topology: single bond between atoms 0-1, no angles/torsions."""
    bond_idx = np.array([[0, 1]], dtype=np.int32)
    angle_idx = np.zeros((0, 3), dtype=np.int32)
    torsion_idx = np.zeros((0, 4), dtype=np.int32)
    torsion_periodicity = np.zeros((0, 1), dtype=np.int32)
    torsion_phase_rad = np.zeros((0, 1), dtype=np.float32)

    return BondedTopology(
        bond_idx=bond_idx,
        angle_idx=angle_idx,
        torsion_idx=torsion_idx,
        torsion_periodicity=torsion_periodicity,
        torsion_phase_rad=torsion_phase_rad,
    )


@pytest.fixture
def minimal_params():
    """Minimal bonded parameters: single bond, no angles/torsions."""
    k_bond = jnp.array([500.0], dtype=jnp.float32)
    r0 = jnp.array([1.5], dtype=jnp.float32)
    k_theta = jnp.array([], dtype=jnp.float32)
    theta0_rad = jnp.array([], dtype=jnp.float32)
    k_phi = jnp.zeros((0, 1), dtype=jnp.float32)

    return BondedParams(
        k_bond=k_bond,
        r0=r0,
        k_theta=k_theta,
        theta0_rad=theta0_rad,
        k_phi=k_phi,
    )


# ===== SHAPE NORMALIZATION TESTS =====


def test_single_conformer_batch_axis_added(minimal_params, minimal_topology):
    """2D positions (n_atoms, 3) should be expanded to 3D (1, n_atoms, 3)."""
    positions = jnp.zeros((5, 3), dtype=jnp.float32)
    forces = jnp.zeros((5, 3), dtype=jnp.float32)
    energies = jnp.array(1.0, dtype=jnp.float32)

    bundle = build_fitting_bundle(
        positions, forces, energies, minimal_params, minimal_topology
    )

    assert bundle.conformers.positions.shape == (1, 5, 3)
    assert bundle.conformers.forces_ref.shape == (1, 5, 3)
    assert bundle.conformers.energies_ref.shape == (1,)


def test_scalar_energy_expanded_to_1d(minimal_params, minimal_topology):
    """Scalar energy should be expanded to (1,) array when single conformer input."""
    positions = jnp.zeros((5, 3), dtype=jnp.float32)  # Single conformer (2D)
    forces = jnp.zeros((5, 3), dtype=jnp.float32)
    energies = jnp.array(1.5, dtype=jnp.float32)  # Scalar

    bundle = build_fitting_bundle(
        positions, forces, energies, minimal_params, minimal_topology
    )

    assert bundle.conformers.energies_ref.shape == (1,)
    assert float(bundle.conformers.energies_ref[0]) == pytest.approx(1.5)


def test_3d_shapes_accepted_as_is(minimal_params, minimal_topology):
    """3D positions and forces should be accepted without modification."""
    positions = jnp.ones((3, 4, 3), dtype=jnp.float32)
    forces = jnp.ones((3, 4, 3), dtype=jnp.float32)
    energies = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)

    bundle = build_fitting_bundle(
        positions, forces, energies, minimal_params, minimal_topology
    )

    assert bundle.conformers.positions.shape == (3, 4, 3)
    assert bundle.conformers.energies_ref.shape == (3,)


# ===== OPTIONAL FIELD RESOLUTION TESTS =====


def test_optional_fields_resolve_to_zeros(minimal_params, minimal_topology):
    """atom_mask=None → all-ones; box=None → jnp.zeros((3,3))."""
    positions = jnp.zeros((2, 5, 3), dtype=jnp.float32)
    forces = jnp.zeros((2, 5, 3), dtype=jnp.float32)
    energies = jnp.zeros((2,), dtype=jnp.float32)

    bundle = build_fitting_bundle(
        positions,
        forces,
        energies,
        minimal_params,
        minimal_topology,
    )

    # atom_mask should be all-ones
    assert jnp.all(bundle.conformers.atom_mask)
    assert bundle.conformers.atom_mask.shape == (5,)

    # box should be zero matrix
    assert jnp.all(bundle.box == 0.0)
    assert bundle.box.shape == (3, 3)


def test_box_none_resolves_to_zero_matrix(minimal_params, minimal_topology):
    """Explicit test for box vacuum sentinel."""
    positions = jnp.zeros((2, 5, 3), dtype=jnp.float32)
    forces = jnp.zeros((2, 5, 3), dtype=jnp.float32)
    energies = jnp.zeros((2,), dtype=jnp.float32)

    bundle = build_fitting_bundle(
        positions,
        forces,
        energies,
        minimal_params,
        minimal_topology,
        box=None,
    )

    assert bundle.box.shape == (3, 3)
    assert float(jnp.sum(jnp.abs(bundle.box))) == 0.0


def test_n_conf_real_stored_as_static(minimal_params, minimal_topology):
    """n_conf_real argument stored as static int field on ConformerBundle."""
    positions = jnp.zeros((10, 5, 3), dtype=jnp.float32)
    forces = jnp.zeros((10, 5, 3), dtype=jnp.float32)
    energies = jnp.zeros((10,), dtype=jnp.float32)

    bundle = build_fitting_bundle(
        positions,
        forces,
        energies,
        minimal_params,
        minimal_topology,
        n_conf_real=7,  # buffer is 10, but only 7 real
    )

    assert bundle.conformers.n_conf == 7
    assert bundle.conformers.positions.shape[0] == 10  # buffer unchanged
    # Verify n_conf is a Python int (not a JAX array)
    assert isinstance(bundle.conformers.n_conf, int)
    assert isinstance(bundle.conformers.n_atoms, int)


# ===== EXPLICIT FIELD PRESERVATION TESTS =====


def test_explicit_atom_mask_preserved(minimal_params, minimal_topology):
    """atom_mask argument passed through verbatim."""
    mask = jnp.array([True, True, False, False, False], dtype=jnp.bool_)
    positions = jnp.zeros((2, 5, 3), dtype=jnp.float32)
    forces = jnp.zeros((2, 5, 3), dtype=jnp.float32)
    energies = jnp.zeros((2,), dtype=jnp.float32)

    bundle = build_fitting_bundle(
        positions,
        forces,
        energies,
        minimal_params,
        minimal_topology,
        atom_mask=mask,
    )

    assert jnp.array_equal(bundle.conformers.atom_mask, mask)


def test_explicit_box_preserved(minimal_params, minimal_topology):
    """Custom box passed through (e.g., cubic 10Å box)."""
    box = 10.0 * jnp.eye(3, dtype=jnp.float32)
    positions = jnp.zeros((2, 5, 3), dtype=jnp.float32)
    forces = jnp.zeros((2, 5, 3), dtype=jnp.float32)
    energies = jnp.zeros((2,), dtype=jnp.float32)

    bundle = build_fitting_bundle(
        positions,
        forces,
        energies,
        minimal_params,
        minimal_topology,
        box=box,
    )

    assert jnp.array_equal(bundle.box, box)


# ===== VALIDATION INVARIANT TESTS =====


def test_shape_mismatch_raises_positions_forces(minimal_params, minimal_topology):
    """ValueError if positions and forces have different shapes."""
    positions = jnp.zeros((2, 5, 3), dtype=jnp.float32)
    forces = jnp.zeros((2, 4, 3), dtype=jnp.float32)  # wrong n_atoms
    energies = jnp.zeros((2,), dtype=jnp.float32)

    with pytest.raises(ValueError, match="identical shapes"):
        build_fitting_bundle(
            positions,
            forces,
            energies,
            minimal_params,
            minimal_topology,
        )


def test_energies_length_mismatch_raises(minimal_params, minimal_topology):
    """ValueError if energies.shape[0] != N_conf."""
    positions = jnp.zeros((2, 5, 3), dtype=jnp.float32)
    forces = jnp.zeros((2, 5, 3), dtype=jnp.float32)
    energies = jnp.zeros((3,), dtype=jnp.float32)  # wrong N_conf

    with pytest.raises(ValueError, match="energies_all.shape"):
        build_fitting_bundle(
            positions,
            forces,
            energies,
            minimal_params,
            minimal_topology,
        )


def test_atom_mask_length_mismatch_raises(minimal_params, minimal_topology):
    """ValueError if atom_mask.shape[0] != n_atoms."""
    mask = jnp.array([True, True, False], dtype=jnp.bool_)
    positions = jnp.zeros((2, 5, 3), dtype=jnp.float32)
    forces = jnp.zeros((2, 5, 3), dtype=jnp.float32)
    energies = jnp.zeros((2,), dtype=jnp.float32)

    with pytest.raises(ValueError, match="atom_mask.shape"):
        build_fitting_bundle(
            positions,
            forces,
            energies,
            minimal_params,
            minimal_topology,
            atom_mask=mask,
        )


def test_box_shape_mismatch_raises(minimal_params, minimal_topology):
    """ValueError if box has wrong shape."""
    box = jnp.zeros((3, 4), dtype=jnp.float32)  # wrong shape
    positions = jnp.zeros((2, 5, 3), dtype=jnp.float32)
    forces = jnp.zeros((2, 5, 3), dtype=jnp.float32)
    energies = jnp.zeros((2,), dtype=jnp.float32)

    with pytest.raises(ValueError, match="box must have shape"):
        build_fitting_bundle(
            positions,
            forces,
            energies,
            minimal_params,
            minimal_topology,
            box=box,
        )


def test_n_conf_real_exceeds_buffer_raises(minimal_params, minimal_topology):
    """ValueError if n_conf_real > N_conf."""
    positions = jnp.zeros((5, 3, 3), dtype=jnp.float32)
    forces = jnp.zeros((5, 3, 3), dtype=jnp.float32)
    energies = jnp.zeros((5,), dtype=jnp.float32)

    with pytest.raises(ValueError, match="n_conf_real.*cannot exceed"):
        build_fitting_bundle(
            positions,
            forces,
            energies,
            minimal_params,
            minimal_topology,
            n_conf_real=10,  # buffer is 5
        )


def test_invalid_positions_ndim_raises(minimal_params, minimal_topology):
    """ValueError if positions has wrong ndim."""
    positions = jnp.zeros((2, 3, 4, 3), dtype=jnp.float32)  # 4D
    forces = jnp.zeros((2, 3, 4, 3), dtype=jnp.float32)
    energies = jnp.zeros((2,), dtype=jnp.float32)

    with pytest.raises(ValueError, match="positions_all must be 2D.*or 3D"):
        build_fitting_bundle(
            positions,
            forces,
            energies,
            minimal_params,
            minimal_topology,
        )


# ===== JIT PASSTHROUGH TEST =====


def test_bundle_jit_passthrough_post_build(minimal_params, minimal_topology):
    """Built bundle can pass through jax.jit identity without trace error."""
    positions = jnp.zeros((2, 5, 3), dtype=jnp.float32)
    forces = jnp.zeros((2, 5, 3), dtype=jnp.float32)
    energies = jnp.zeros((2,), dtype=jnp.float32)

    bundle = build_fitting_bundle(
        positions,
        forces,
        energies,
        minimal_params,
        minimal_topology,
    )

    # This should compile without trace errors
    identity = jax.jit(lambda b: b)
    out = identity(bundle)

    # Roundtrip preserves shapes
    assert out.conformers.positions.shape == bundle.conformers.positions.shape
    assert out.conformers.forces_ref.shape == bundle.conformers.forces_ref.shape
    assert out.conformers.energies_ref.shape == bundle.conformers.energies_ref.shape
    assert out.box.shape == bundle.box.shape


# ===== BUNDLE TYPE TESTS =====


def test_returned_type_is_fitting_bundle(minimal_params, minimal_topology):
    """build_fitting_bundle returns a FittingBundle instance."""
    positions = jnp.zeros((2, 5, 3), dtype=jnp.float32)
    forces = jnp.zeros((2, 5, 3), dtype=jnp.float32)
    energies = jnp.zeros((2,), dtype=jnp.float32)

    bundle = build_fitting_bundle(
        positions,
        forces,
        energies,
        minimal_params,
        minimal_topology,
    )

    assert isinstance(bundle, FittingBundle)
    assert isinstance(bundle.conformers, ConformerBundle)
    assert isinstance(bundle.params, BondedParams)
    assert isinstance(bundle.topology, BondedTopology)


def test_conformer_bundle_structure(minimal_params, minimal_topology):
    """ConformerBundle inside FittingBundle has correct structure."""
    positions = jnp.zeros((3, 4, 3), dtype=jnp.float32)
    forces = jnp.zeros((3, 4, 3), dtype=jnp.float32)
    energies = jnp.zeros((3,), dtype=jnp.float32)

    bundle = build_fitting_bundle(
        positions,
        forces,
        energies,
        minimal_params,
        minimal_topology,
    )

    conf = bundle.conformers
    assert conf.positions.shape == (3, 4, 3)
    assert conf.forces_ref.shape == (3, 4, 3)
    assert conf.energies_ref.shape == (3,)
    assert conf.atom_mask.shape == (4,)
    assert conf.n_conf == 3
    assert conf.n_atoms == 4


# ===== EDGE CASE TESTS =====


def test_single_atom_single_conformer(minimal_params, minimal_topology):
    """Minimal valid system: 1 atom, 1 conformer."""
    positions = jnp.zeros((1, 1, 3), dtype=jnp.float32)
    forces = jnp.zeros((1, 1, 3), dtype=jnp.float32)
    energies = jnp.zeros((1,), dtype=jnp.float32)

    bundle = build_fitting_bundle(
        positions,
        forces,
        energies,
        minimal_params,
        minimal_topology,
    )

    assert bundle.conformers.positions.shape == (1, 1, 3)
    assert bundle.conformers.n_conf == 1
    assert bundle.conformers.n_atoms == 1


def test_many_conformers_large_system(minimal_params, minimal_topology):
    """Large system: 100 conformers, 500 atoms."""
    positions = jnp.zeros((100, 500, 3), dtype=jnp.float32)
    forces = jnp.zeros((100, 500, 3), dtype=jnp.float32)
    energies = jnp.zeros((100,), dtype=jnp.float32)

    bundle = build_fitting_bundle(
        positions,
        forces,
        energies,
        minimal_params,
        minimal_topology,
    )

    assert bundle.conformers.positions.shape == (100, 500, 3)
    assert bundle.conformers.n_conf == 100
    assert bundle.conformers.n_atoms == 500


def test_partial_atom_mask(minimal_params, minimal_topology):
    """Mixed real/padded atoms with partial mask."""
    mask = jnp.array([True, True, True, False, False], dtype=jnp.bool_)
    positions = jnp.zeros((2, 5, 3), dtype=jnp.float32)
    forces = jnp.zeros((2, 5, 3), dtype=jnp.float32)
    energies = jnp.zeros((2,), dtype=jnp.float32)

    bundle = build_fitting_bundle(
        positions,
        forces,
        energies,
        minimal_params,
        minimal_topology,
        atom_mask=mask,
    )

    assert jnp.array_equal(bundle.conformers.atom_mask, mask)
    assert jnp.sum(bundle.conformers.atom_mask) == 3  # 3 real atoms
