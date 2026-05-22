"""Tests for BatchPlan dispatch in FittingPlan.

Tests for the newly added plan-aware dispatch in FittingPlan.step() and
FittingPlan.evaluate(). Validates that:
- plan=None uses full vmap (current behavior)
- plan with batch_size == 0 uses full vmap (vmap mode)
- plan with batch_size > 0 and < B uses chunked vmap (safe_map mode)
- chunked vmap produces equivalent results to full vmap
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from prolix.fitting import BondedParams, BondedTopology
from prolix.fitting.bundle_builder import build_fitting_bundle
from prolix.fitting.bundles import BatchedFittingBundle, FittingBundle, TrainState
from prolix.fitting.config import FittingConfig, FittingPlan, TrainMetrics, make_fitting_plan
from prolix.run.spec import BatchingConfig, FittingAxisNames, make_fitting_planner
from prolix.tiling.axes import N_MOLS
from prolix.tiling.planner import AxisDecision, AxisSpec, BatchPlan


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
def fitting_bundle_3atom(water_geometry, water_params, water_topology):
    """FittingBundle for 3-atom water with 2 conformers."""
    positions = jnp.stack([water_geometry, water_geometry + 0.01], axis=0)
    forces = jnp.ones((2, 3, 3), dtype=jnp.float32)
    energies = jnp.array([1.0, 1.5], dtype=jnp.float32)

    return build_fitting_bundle(
        positions, forces, energies, water_params, water_topology
    )


@pytest.fixture
def batched_bundle_4mols(fitting_bundle_3atom):
    """BatchedFittingBundle with 4 molecules (for chunked vmap testing)."""
    bundles = [fitting_bundle_3atom] * 4
    return BatchedFittingBundle.stack(bundles)


@pytest.fixture
def batched_bundle_single(fitting_bundle_3atom):
    """BatchedFittingBundle with single molecule."""
    return BatchedFittingBundle.stack([fitting_bundle_3atom])


@pytest.fixture
def train_state_init_4mols(batched_bundle_4mols):
    """Initial TrainState for 4 molecules."""
    config = FittingConfig(lr=1e-3, n_steps=1)
    plan = make_fitting_plan(config)
    opt_state = plan.optimizer.init(batched_bundle_4mols.params_batched)
    return TrainState(
        params=batched_bundle_4mols.params_batched,
        opt_state=opt_state,
        key=jax.random.PRNGKey(0),
        step_count=jnp.array(0, dtype=jnp.int32),
    )


@pytest.fixture
def train_state_init_single(batched_bundle_single):
    """Initial TrainState for single molecule."""
    config = FittingConfig(lr=1e-3, n_steps=1)
    plan = make_fitting_plan(config)
    opt_state = plan.optimizer.init(batched_bundle_single.params_batched)
    return TrainState(
        params=batched_bundle_single.params_batched,
        opt_state=opt_state,
        key=jax.random.PRNGKey(0),
        step_count=jnp.array(0, dtype=jnp.int32),
    )


def _make_batch_plan_with_n_mols_batch_size(batch_size: int) -> BatchPlan:
    """Create a BatchPlan with N_MOLS axis decision."""
    n_mols_axis = AxisSpec(
        name="n_mols",
        axis_index=5,
        cardinality=4,
        default_batch_size=batch_size,
        tile_granularity=1,
        heterogeneous=True,
        doc="test axis",
    )
    decision = AxisDecision(
        axis=n_mols_axis,
        batch_size=batch_size,
        reasoning="test decision",
    )
    return BatchPlan(
        decisions=[decision],
        total_memory_estimate=0.0,
        axes_by_index={5: n_mols_axis},
        budget_exceeded=False,
    )


# ===== TESTS =====


def test_no_plan_uses_full_vmap(batched_bundle_single, train_state_init_single):
    """FittingPlan with plan=None uses full vmap (current behavior)."""
    config = FittingConfig(lr=1e-3, n_steps=1)
    plan_none = make_fitting_plan(config, plan=None)

    # plan=None should work and produce a loss
    new_state, metrics = plan_none.step(batched_bundle_single, train_state_init_single, conformer_idx=0)
    assert jnp.isfinite(metrics.loss)


def test_plan_with_batch_size_zero_uses_full_vmap(batched_bundle_4mols, train_state_init_4mols):
    """BatchPlan with batch_size=0 (vmap sentinel) uses full vmap."""
    config = FittingConfig(lr=1e-3, n_steps=1)
    batch_plan = _make_batch_plan_with_n_mols_batch_size(batch_size=0)
    plan_vmap = make_fitting_plan(config, plan=batch_plan)

    # Should run and produce finite loss
    new_state, metrics = plan_vmap.step(batched_bundle_4mols, train_state_init_4mols, conformer_idx=0)
    assert jnp.isfinite(metrics.loss)


def test_plan_with_batch_size_ge_B_uses_full_vmap(batched_bundle_4mols, train_state_init_4mols):
    """BatchPlan with batch_size >= B uses full vmap (no chunking)."""
    config = FittingConfig(lr=1e-3, n_steps=1)
    B = 4
    batch_plan = _make_batch_plan_with_n_mols_batch_size(batch_size=B)
    plan = make_fitting_plan(config, plan=batch_plan)

    # Should run and produce finite loss
    new_state, metrics = plan.step(batched_bundle_4mols, train_state_init_4mols, conformer_idx=0)
    assert jnp.isfinite(metrics.loss)


def test_chunked_vmap_matches_full_vmap(batched_bundle_4mols, train_state_init_4mols):
    """B=4 mols with plan.batch_size=2 → 2 chunks → same loss as full vmap."""
    config = FittingConfig(lr=1e-3, n_steps=1)

    # Plan with full vmap
    plan_full = make_fitting_plan(config, plan=_make_batch_plan_with_n_mols_batch_size(batch_size=0))

    # Plan with chunked vmap (batch_size=2, so 2 chunks of 2)
    plan_chunked = make_fitting_plan(config, plan=_make_batch_plan_with_n_mols_batch_size(batch_size=2))

    # Run both
    _, metrics_full = plan_full.step(batched_bundle_4mols, train_state_init_4mols, conformer_idx=0)
    _, metrics_chunked = plan_chunked.step(batched_bundle_4mols, train_state_init_4mols, conformer_idx=0)

    # Losses should match within float tolerance
    # Note: float reordering in scan may cause small differences
    assert jnp.allclose(metrics_full.loss, metrics_chunked.loss, atol=1e-6, rtol=1e-5)


def test_chunked_vmap_evaluate_matches_full(batched_bundle_4mols, train_state_init_4mols):
    """Evaluate path: chunked vmap matches full vmap."""
    config = FittingConfig(lr=1e-3, n_steps=1)

    plan_full = make_fitting_plan(config, plan=_make_batch_plan_with_n_mols_batch_size(batch_size=0))
    plan_chunked = make_fitting_plan(config, plan=_make_batch_plan_with_n_mols_batch_size(batch_size=2))

    metrics_full = plan_full.evaluate(batched_bundle_4mols, train_state_init_4mols)
    metrics_chunked = plan_chunked.evaluate(batched_bundle_4mols, train_state_init_4mols)

    assert jnp.allclose(metrics_full.loss, metrics_chunked.loss, atol=1e-6, rtol=1e-5)


def test_plan_with_batch_size_1_works(batched_bundle_4mols, train_state_init_4mols):
    """B=4 with batch_size=1 should produce same loss as full vmap."""
    config = FittingConfig(lr=1e-3, n_steps=1)

    plan_full = make_fitting_plan(config, plan=_make_batch_plan_with_n_mols_batch_size(batch_size=0))
    plan_bs1 = make_fitting_plan(config, plan=_make_batch_plan_with_n_mols_batch_size(batch_size=1))

    _, metrics_full = plan_full.step(batched_bundle_4mols, train_state_init_4mols, conformer_idx=0)
    _, metrics_bs1 = plan_bs1.step(batched_bundle_4mols, train_state_init_4mols, conformer_idx=0)

    assert jnp.allclose(metrics_full.loss, metrics_bs1.loss, atol=1e-6, rtol=1e-5)


def test_plan_not_divisible_raises(batched_bundle_4mols, train_state_init_4mols):
    """B=4 with batch_size=3 (not divisible) should raise ValueError."""
    config = FittingConfig(lr=1e-3, n_steps=1)

    # Create a plan with batch_size=3 (4 % 3 != 0)
    batch_plan = _make_batch_plan_with_n_mols_batch_size(batch_size=3)
    plan = make_fitting_plan(config, plan=batch_plan)

    # Should raise ValueError about divisibility
    with pytest.raises(ValueError, match="not divisible"):
        plan.step(batched_bundle_4mols, train_state_init_4mols, conformer_idx=0)


def test_plan_without_n_mols_falls_back_to_vmap(batched_bundle_4mols, train_state_init_4mols):
    """BatchPlan without N_MOLS axis falls back to full vmap (no error)."""
    config = FittingConfig(lr=1e-3, n_steps=1)

    # Create a plan with some other axis, NOT N_MOLS
    other_axis = AxisSpec(
        name="n_atoms",
        axis_index=0,
        cardinality=3,
        default_batch_size=0,
        tile_granularity=1,
        heterogeneous=False,
        doc="test other axis",
    )
    decision = AxisDecision(
        axis=other_axis,
        batch_size=0,
        reasoning="test",
    )
    batch_plan = BatchPlan(
        decisions=[decision],
        total_memory_estimate=0.0,
        axes_by_index={0: other_axis},
        budget_exceeded=False,
    )
    plan = make_fitting_plan(config, plan=batch_plan)

    # Should NOT raise; should fall back to full vmap
    new_state, metrics = plan.step(batched_bundle_4mols, train_state_init_4mols, conformer_idx=0)
    assert jnp.isfinite(metrics.loss)


def test_make_fitting_plan_accepts_plan_kwarg():
    """make_fitting_plan should accept plan kwarg and plumb it into FittingPlan.plan."""
    config = FittingConfig(lr=1e-3, n_steps=1)
    batch_plan = _make_batch_plan_with_n_mols_batch_size(batch_size=2)

    plan = make_fitting_plan(config, plan=batch_plan)

    # plan.plan is wrapped in _BatchPlanWrapper for hashing, check the unwrapped version
    assert hasattr(plan.plan, '_plan')
    assert plan.plan._plan is batch_plan


def test_make_fitting_plan_default_plan_is_none():
    """make_fitting_plan without plan kwarg should default to None."""
    config = FittingConfig(lr=1e-3, n_steps=1)

    plan = make_fitting_plan(config)

    assert plan.plan is None


def test_plan_deterministic_on_single_call(batched_bundle_4mols, train_state_init_4mols):
    """Two calls to the same plan should give same loss (deterministic)."""
    config = FittingConfig(lr=1e-3, n_steps=1)
    plan = make_fitting_plan(config, plan=_make_batch_plan_with_n_mols_batch_size(batch_size=2))

    # Call twice
    _, metrics1 = plan.step(batched_bundle_4mols, train_state_init_4mols, conformer_idx=0)
    _, metrics2 = plan.step(batched_bundle_4mols, train_state_init_4mols, conformer_idx=0)

    assert jnp.allclose(metrics1.loss, metrics2.loss)


def test_chunked_vmap_with_larger_chunk_size(batched_bundle_4mols, train_state_init_4mols):
    """B=4 with batch_size=4 (full chunk, no chunking) should match batch_size=0."""
    config = FittingConfig(lr=1e-3, n_steps=1)

    plan_full = make_fitting_plan(config, plan=_make_batch_plan_with_n_mols_batch_size(batch_size=0))
    plan_full_chunk = make_fitting_plan(config, plan=_make_batch_plan_with_n_mols_batch_size(batch_size=4))

    _, metrics_full = plan_full.step(batched_bundle_4mols, train_state_init_4mols, conformer_idx=0)
    _, metrics_full_chunk = plan_full_chunk.step(batched_bundle_4mols, train_state_init_4mols, conformer_idx=0)

    assert jnp.allclose(metrics_full.loss, metrics_full_chunk.loss, atol=1e-6, rtol=1e-5)


def test_multiple_step_calls_with_plan(batched_bundle_4mols, train_state_init_4mols):
    """Multiple .step() calls with plan should work and track state."""
    config = FittingConfig(lr=1e-3, n_steps=5)
    plan = make_fitting_plan(config, plan=_make_batch_plan_with_n_mols_batch_size(batch_size=2))

    state = train_state_init_4mols
    losses = []

    for i in range(5):
        state, metrics = plan.step(batched_bundle_4mols, state, conformer_idx=0)
        losses.append(float(metrics.loss))
        assert state.step_count == i + 1

    # All losses should be finite
    assert all(jnp.isfinite(l) for l in losses)
    # Step count should have incremented
    assert state.step_count == 5
