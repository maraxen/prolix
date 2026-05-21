"""Tests for bundle invariants (Phase 3 §5).

Validates ConformerBundle, FittingBundle, BatchedFittingBundle, and TrainState
types for pytree structure, JIT compatibility, and filter_diff behavior.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from prolix.fitting import BondedParams, BondedTopology
from prolix.fitting.bundles import (
    BatchedConformerBundle,
    BatchedFittingBundle,
    ConformerBundle,
    FittingBundle,
    TrainState,
)


# ===== FIXTURES =====


@pytest.fixture
def water_geometry():
    """3-atom water at standard geometry."""
    positions = np.array([
        [0.0, 0.0, 0.0],  # O
        [0.96, 0.0, 0.0],  # H1
        [-0.24, 0.93, 0.0],  # H2
    ], dtype=np.float32)
    return jnp.array(positions)


@pytest.fixture
def water_topology():
    """Bonded topology for water (2 O-H bonds, 1 H-O-H angle)."""
    bond_idx = np.array([[0, 1], [0, 2]], dtype=np.int32)
    angle_idx = np.array([[1, 0, 2]], dtype=np.int32)
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
def water_params():
    """Bonded parameters for water."""
    k_bond = jnp.array([500.0, 500.0], dtype=jnp.float32)
    r0 = jnp.array([0.95, 0.95], dtype=jnp.float32)
    k_theta = jnp.array([45.0], dtype=jnp.float32)
    theta0_rad = jnp.array([103.5 * jnp.pi / 180.0], dtype=jnp.float32)
    k_phi = jnp.zeros((0, 1), dtype=jnp.float32)

    return BondedParams(
        k_bond=k_bond,
        r0=r0,
        k_theta=k_theta,
        theta0_rad=theta0_rad,
        k_phi=k_phi,
    )


@pytest.fixture
def conformer_bundle(water_geometry):
    """ConformerBundle with 2 conformers, 3 atoms."""
    n_conf = 2
    n_atoms = 3

    positions = jnp.stack([water_geometry, water_geometry + 0.01], axis=0)
    forces_ref = jnp.ones((n_conf, n_atoms, 3), dtype=jnp.float32)
    energies_ref = jnp.array([1.0, 1.5], dtype=jnp.float32)
    atom_mask = jnp.array([True, True, True], dtype=jnp.bool_)

    return ConformerBundle(
        positions=positions,
        forces_ref=forces_ref,
        energies_ref=energies_ref,
        atom_mask=atom_mask,
        n_conf=n_conf,
        n_atoms=n_atoms,
    )


@pytest.fixture
def fitting_bundle(conformer_bundle, water_params, water_topology):
    """FittingBundle for a single molecule."""
    box = jnp.zeros((3, 3), dtype=jnp.float32)

    return FittingBundle(
        conformers=conformer_bundle,
        params=water_params,
        topology=water_topology,
        box=box,
    )


@pytest.fixture
def train_state(water_params):
    """TrainState with default optax.sgd optimizer."""
    optimizer = optax.sgd(learning_rate=0.01)
    opt_state = optimizer.init(water_params)
    key = jax.random.PRNGKey(42)
    step_count = jnp.array(0, dtype=jnp.int32)

    return TrainState(
        params=water_params,
        opt_state=opt_state,
        key=key,
        step_count=step_count,
    )


# ===== CONFORMER BUNDLE TESTS =====


def test_conformer_bundle_pytree_leaves_count(conformer_bundle):
    """ConformerBundle should have exactly 4 pytree leaves: positions, forces_ref, energies_ref, atom_mask."""
    leaves = jax.tree_util.tree_leaves(conformer_bundle)
    # Expect 4 leaves (the 4 array fields; n_conf and n_atoms are static)
    assert len(leaves) == 4, f"Expected 4 leaves, got {len(leaves)}"


def test_conformer_bundle_static_fields_not_leaves(conformer_bundle):
    """n_conf and n_atoms should be static fields (not in pytree leaves)."""
    # Static fields should not appear in tree_leaves
    leaves = jax.tree_util.tree_leaves(conformer_bundle)
    leaf_values = [str(leaf) for leaf in leaves]

    # Check that neither n_conf nor n_atoms scalars appear as standalone leaves
    # (they are stored as eqx.field(static=True), so they won't show up)
    assert conformer_bundle.n_conf == 2
    assert conformer_bundle.n_atoms == 3

    # Verify the bundle structure itself is intact
    assert len(leaves) == 4


def test_conformer_bundle_round_trip(conformer_bundle):
    """ConformerBundle should round-trip through tree_flatten/tree_unflatten."""
    flat, treedef = jax.tree_util.tree_flatten(conformer_bundle)
    reconstructed = jax.tree_util.tree_unflatten(treedef, flat)

    # Check shapes are preserved
    assert reconstructed.positions.shape == conformer_bundle.positions.shape
    assert reconstructed.forces_ref.shape == conformer_bundle.forces_ref.shape
    assert reconstructed.energies_ref.shape == conformer_bundle.energies_ref.shape
    assert reconstructed.atom_mask.shape == conformer_bundle.atom_mask.shape

    # Check static fields are preserved
    assert reconstructed.n_conf == conformer_bundle.n_conf
    assert reconstructed.n_atoms == conformer_bundle.n_atoms


# ===== FITTING BUNDLE TESTS =====


def test_fitting_bundle_jit_passthrough(fitting_bundle):
    """FittingBundle should round-trip through jax.jit(lambda b: b) without trace errors."""
    # This tests that the bundle is JIT-compatible and static fields are handled correctly
    jitted_identity = jax.jit(lambda b: b)
    result = jitted_identity(fitting_bundle)

    # Verify the result is structurally identical
    assert result.conformers.n_conf == fitting_bundle.conformers.n_conf
    assert result.conformers.n_atoms == fitting_bundle.conformers.n_atoms
    assert result.box.shape == fitting_bundle.box.shape


def test_fitting_bundle_filter_diff(fitting_bundle):
    """eqx.filter(bundle, eqx.is_array) should isolate array leaves."""
    import equinox as eqx

    array_filter = eqx.filter(fitting_bundle, eqx.is_array)

    # Check that array fields are present
    assert array_filter.conformers.positions is not None
    assert array_filter.conformers.forces_ref is not None
    assert array_filter.conformers.energies_ref is not None
    assert array_filter.conformers.atom_mask is not None
    assert array_filter.box is not None

    # Check that params array fields are present
    assert array_filter.params.k_bond is not None
    assert array_filter.params.r0 is not None

    # topology should be None (not array pytree leaves) since it's a dataclass with numpy arrays
    # but those arrays are stored as fields, not pytree leaves


def test_fitting_bundle_pytree_round_trip(fitting_bundle):
    """FittingBundle should round-trip through tree_flatten/tree_unflatten."""
    flat, treedef = jax.tree_util.tree_flatten(fitting_bundle)
    reconstructed = jax.tree_util.tree_unflatten(treedef, flat)

    # Verify key structures are preserved
    assert reconstructed.conformers.n_conf == fitting_bundle.conformers.n_conf
    assert reconstructed.conformers.n_atoms == fitting_bundle.conformers.n_atoms
    assert reconstructed.box.shape == fitting_bundle.box.shape


# ===== BATCHED FITTING BUNDLE TESTS =====


def test_batched_fitting_bundle_stack_is_phase5(fitting_bundle):
    """BatchedFittingBundle.stack should raise NotImplementedError with Phase 5 message."""
    with pytest.raises(NotImplementedError, match="Phase 5"):
        BatchedFittingBundle.stack([fitting_bundle])


def test_batched_fitting_bundle_step_is_phase5(fitting_bundle, train_state):
    """BatchedFittingBundle.step should raise NotImplementedError with Phase 5 message."""
    # First we need to construct a minimal BatchedFittingBundle manually
    # (since .stack is not implemented)
    batched_bundle = _make_minimal_batched_fitting_bundle()

    with pytest.raises(NotImplementedError, match="Phase 5"):
        batched_bundle.step(train_state, conformer_idx=0)


def test_batched_fitting_bundle_evaluate_is_phase5(fitting_bundle, train_state):
    """BatchedFittingBundle.evaluate should raise NotImplementedError with Phase 5 message."""
    batched_bundle = _make_minimal_batched_fitting_bundle()

    with pytest.raises(NotImplementedError, match="Phase 5"):
        batched_bundle.evaluate(train_state)


def _make_minimal_batched_fitting_bundle() -> BatchedFittingBundle:
    """Helper to construct a minimal BatchedFittingBundle for testing."""
    from prolix.fitting.batched import BondedParamsBundle, BondedTopologyBundle

    n_mols = 1
    max_n_atoms = 3
    max_n_conf = 2
    max_n_bonds = 2
    max_n_angles = 1
    max_n_torsions = 0
    n_torsion_terms = 1

    conf_bundle = BatchedConformerBundle(
        positions=jnp.zeros((n_mols, max_n_conf, max_n_atoms, 3), dtype=jnp.float32),
        forces_ref=jnp.zeros((n_mols, max_n_conf, max_n_atoms, 3), dtype=jnp.float32),
        energies_ref=jnp.zeros((n_mols, max_n_conf), dtype=jnp.float32),
        atom_mask=jnp.ones((n_mols, max_n_atoms), dtype=jnp.bool_),
        n_conf_real=jnp.array([max_n_conf], dtype=jnp.int32),
        n_atoms_real=jnp.array([max_n_atoms], dtype=jnp.int32),
        n_mols=n_mols,
        max_n_atoms=max_n_atoms,
        max_n_conf=max_n_conf,
    )

    params_bundle = BondedParamsBundle(
        k_bond=jnp.zeros((n_mols, max_n_bonds), dtype=jnp.float32),
        r0=jnp.zeros((n_mols, max_n_bonds), dtype=jnp.float32),
        k_theta=jnp.zeros((n_mols, max_n_angles), dtype=jnp.float32),
        theta0_rad=jnp.zeros((n_mols, max_n_angles), dtype=jnp.float32),
        k_phi=jnp.zeros((n_mols, max_n_torsions, n_torsion_terms), dtype=jnp.float32),
    )

    topology_bundle = BondedTopologyBundle(
        bond_idx=jnp.zeros((n_mols, max_n_bonds, 2), dtype=jnp.int32),
        angle_idx=jnp.zeros((n_mols, max_n_angles, 3), dtype=jnp.int32),
        torsion_idx=jnp.zeros((n_mols, max_n_torsions, 4), dtype=jnp.int32),
        torsion_periodicity=jnp.zeros((n_mols, max_n_torsions, n_torsion_terms), dtype=jnp.int32),
        torsion_phase_rad=jnp.zeros((n_mols, max_n_torsions, n_torsion_terms), dtype=jnp.float32),
        bond_mask=jnp.ones((n_mols, max_n_bonds), dtype=jnp.bool_),
        angle_mask=jnp.ones((n_mols, max_n_angles), dtype=jnp.bool_),
        torsion_mask=jnp.ones((n_mols, max_n_torsions), dtype=jnp.bool_),
    )

    box_batched = jnp.zeros((n_mols, 3, 3), dtype=jnp.float32)

    return BatchedFittingBundle(
        conformers_batched=conf_bundle,
        params_batched=params_bundle,
        topology_batched=topology_bundle,
        box_batched=box_batched,
        n_mols_real=n_mols,
    )


# ===== TRAIN STATE TESTS =====


def test_train_state_pytree_round_trip(train_state):
    """TrainState should round-trip through tree_flatten/tree_unflatten."""
    flat, treedef = jax.tree_util.tree_flatten(train_state)
    reconstructed = jax.tree_util.tree_unflatten(treedef, flat)

    # Verify key fields are preserved
    assert reconstructed.step_count == train_state.step_count
    assert reconstructed.params.k_bond.shape == train_state.params.k_bond.shape


def test_train_state_jit_passthrough(train_state):
    """TrainState should round-trip through jax.jit(lambda s: s) without errors."""
    jitted_identity = jax.jit(lambda s: s)
    result = jitted_identity(train_state)

    # Verify the result is structurally identical
    assert result.step_count == train_state.step_count


def test_train_state_has_required_fields(water_params):
    """TrainState must have params, opt_state, key, and step_count fields."""
    optimizer = optax.sgd(learning_rate=0.01)
    opt_state = optimizer.init(water_params)
    key = jax.random.PRNGKey(0)
    step_count = jnp.array(0, dtype=jnp.int32)

    state = TrainState(
        params=water_params,
        opt_state=opt_state,
        key=key,
        step_count=step_count,
    )

    assert state.params is water_params
    assert state.opt_state is opt_state
    assert state.key is key
    assert state.step_count == step_count
