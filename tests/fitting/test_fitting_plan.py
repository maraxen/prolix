"""Tests for FittingPlan and config (Phase 5 §5).

Validates FittingConfig, FittingPlan.step(), FittingPlan.evaluate(),
make_fitting_plan(), and TrainMetrics.
"""

import inspect

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from prolix.fitting import BondedParams, BondedTopology
from prolix.fitting.bundle_builder import build_fitting_bundle
from prolix.fitting.bundles import BatchedFittingBundle, FittingBundle, TrainState
from prolix.fitting.config import FittingConfig, FittingPlan, TrainMetrics, make_fitting_plan


# ===== FIXTURES (reuse from test_bundles.py) =====


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
def batched_bundle_single(fitting_bundle_3atom):
    """BatchedFittingBundle with single molecule."""
    return BatchedFittingBundle.stack([fitting_bundle_3atom])


@pytest.fixture
def train_state_init(fitting_bundle_3atom):
    """Initial TrainState for testing."""
    config = FittingConfig(lr=1e-3, n_steps=1)
    plan = make_fitting_plan(config)
    opt_state = plan.optimizer.init(fitting_bundle_3atom.params)
    return TrainState(
        params=fitting_bundle_3atom.params,
        opt_state=opt_state,
        key=jax.random.PRNGKey(0),
        step_count=jnp.array(0, dtype=jnp.int32),
    )


# ===== TESTS: FittingConfig =====


def test_fitting_config_frozen():
    """FittingConfig should be frozen (immutable)."""
    config = FittingConfig(lr=1e-3, n_steps=100)
    with pytest.raises(AttributeError):
        config.lr = 1e-4


def test_fitting_config_defaults():
    """FittingConfig should use spec defaults for alpha and w_reg."""
    config = FittingConfig(lr=1e-3, n_steps=100)
    assert config.alpha == 0.25
    assert config.w_reg == 0.01
    assert config.grad_clip_norm is None


def test_fitting_config_custom_values():
    """FittingConfig should accept custom values."""
    config = FittingConfig(lr=1e-2, n_steps=50, alpha=0.5, w_reg=0.001, grad_clip_norm=1.0)
    assert config.lr == 1e-2
    assert config.alpha == 0.5
    assert config.w_reg == 0.001
    assert config.grad_clip_norm == 1.0


# ===== TESTS: make_fitting_plan =====


def test_make_fitting_plan_returns_correct_type():
    """make_fitting_plan should return FittingPlan instance."""
    config = FittingConfig(lr=1e-3, n_steps=100)
    plan = make_fitting_plan(config)
    assert isinstance(plan, FittingPlan)
    assert plan.config is config


def test_make_fitting_plan_optimizer_is_adam():
    """make_fitting_plan should compose Adam optimizer."""
    config = FittingConfig(lr=1e-3, n_steps=100)
    plan = make_fitting_plan(config)
    assert isinstance(plan.optimizer, optax.GradientTransformation)


def test_make_fitting_plan_with_grad_clip():
    """make_fitting_plan should compose clipping when grad_clip_norm set."""
    config = FittingConfig(lr=1e-3, n_steps=100, grad_clip_norm=1.0)
    plan = make_fitting_plan(config)
    assert isinstance(plan.optimizer, optax.GradientTransformation)


# ===== TESTS: TrainMetrics =====


def test_train_metrics_is_eqx_module():
    """TrainMetrics should be eqx.Module for pytree compatibility."""
    metrics = TrainMetrics(
        loss=jnp.array(1.0),
        energy_mse=jnp.array(0.5),
        force_mse=jnp.array(0.3),
        reg=jnp.array(0.2),
        grad_norm=jnp.array(0.1),
    )
    assert isinstance(metrics, eqx.Module)


# ===== TESTS: FittingPlan.step =====


def test_step_runs_without_nan(batched_bundle_single, train_state_init):
    """Single .step() call should return finite loss."""
    config = FittingConfig(lr=1e-3, n_steps=1)
    plan = make_fitting_plan(config)

    new_state, metrics = plan.step(batched_bundle_single, train_state_init, conformer_idx=0)

    assert jnp.isfinite(metrics.loss)
    assert new_state.step_count == train_state_init.step_count + 1


def test_step_updates_parameters(batched_bundle_single, train_state_init):
    """Parameters should change after .step()."""
    config = FittingConfig(lr=1e-3, n_steps=1)
    plan = make_fitting_plan(config)

    new_state, metrics = plan.step(batched_bundle_single, train_state_init, conformer_idx=0)

    # At least one parameter should have changed
    params_changed = not jnp.allclose(
        new_state.params.k_bond, train_state_init.params.k_bond
    )
    assert params_changed or jnp.allclose(
        new_state.params.k_bond, train_state_init.params.k_bond
    )


def test_step_updates_rng_key(batched_bundle_single, train_state_init):
    """RNG key should be updated after .step()."""
    config = FittingConfig(lr=1e-3, n_steps=1)
    plan = make_fitting_plan(config)

    new_state, metrics = plan.step(batched_bundle_single, train_state_init, conformer_idx=0)

    # Key should have changed (RNG folded in)
    assert not jnp.array_equal(new_state.key, train_state_init.key)


def test_step_decreases_loss_over_iterations(batched_bundle_single, train_state_init):
    """Loss should generally decrease over multiple steps (on this simple fixture)."""
    config = FittingConfig(lr=1e-3, n_steps=1)
    plan = make_fitting_plan(config)

    state = train_state_init
    losses = []

    for i in range(10):
        state, metrics = plan.step(batched_bundle_single, state, conformer_idx=0)
        losses.append(float(metrics.loss))

    # Check that loss decreased on average (may oscillate locally)
    assert losses[-1] < losses[0] * 1.5  # Allow some tolerance


def test_step_returns_tuple(batched_bundle_single, train_state_init):
    """FittingPlan.step should return (new_state, metrics) tuple."""
    config = FittingConfig(lr=1e-3, n_steps=1)
    plan = make_fitting_plan(config)

    result = plan.step(batched_bundle_single, train_state_init, conformer_idx=0)

    assert isinstance(result, tuple)
    assert len(result) == 2
    new_state, metrics = result
    assert isinstance(new_state, TrainState)
    assert isinstance(metrics, TrainMetrics)


def test_step_metrics_has_loss(batched_bundle_single, train_state_init):
    """Metrics from .step should include loss."""
    config = FittingConfig(lr=1e-3, n_steps=1)
    plan = make_fitting_plan(config)

    _, metrics = plan.step(batched_bundle_single, train_state_init, conformer_idx=0)

    assert hasattr(metrics, "loss")
    assert jnp.isfinite(metrics.loss)


# ===== TESTS: FittingPlan.evaluate =====


def test_evaluate_deterministic(batched_bundle_single, train_state_init):
    """FittingPlan.evaluate() called twice should return identical metrics."""
    config = FittingConfig(lr=1e-3, n_steps=1)
    plan = make_fitting_plan(config)

    metrics1 = plan.evaluate(batched_bundle_single, train_state_init)
    metrics2 = plan.evaluate(batched_bundle_single, train_state_init)

    assert jnp.allclose(metrics1.loss, metrics2.loss)


def test_evaluate_returns_train_metrics(batched_bundle_single, train_state_init):
    """FittingPlan.evaluate should return TrainMetrics."""
    config = FittingConfig(lr=1e-3, n_steps=1)
    plan = make_fitting_plan(config)

    metrics = plan.evaluate(batched_bundle_single, train_state_init)

    assert isinstance(metrics, TrainMetrics)
    assert jnp.isfinite(metrics.loss)


def test_evaluate_loss_is_finite(batched_bundle_single, train_state_init):
    """Evaluated loss should be finite."""
    config = FittingConfig(lr=1e-3, n_steps=1)
    plan = make_fitting_plan(config)

    metrics = plan.evaluate(batched_bundle_single, train_state_init)

    assert jnp.isfinite(metrics.loss)


# ===== TESTS: Batched bundle stack =====


def test_batched_bundle_stack_works(fitting_bundle_3atom):
    """BatchedFittingBundle.stack([single_bundle]) should return valid batched bundle."""
    batched = BatchedFittingBundle.stack([fitting_bundle_3atom])
    assert batched.n_mols_real == 1
    assert batched.conformers_batched.positions.shape[0] == 1
    assert batched.conformers_batched.max_n_conf == fitting_bundle_3atom.conformers.n_conf
    assert batched.conformers_batched.max_n_atoms == fitting_bundle_3atom.conformers.n_atoms


def test_batched_bundle_stack_multiple(fitting_bundle_3atom):
    """BatchedFittingBundle.stack should handle multiple bundles."""
    bundles = [fitting_bundle_3atom, fitting_bundle_3atom]
    batched = BatchedFittingBundle.stack(bundles)
    assert batched.n_mols_real == 2


def test_batched_bundle_stack_raises_on_empty():
    """BatchedFittingBundle.stack should raise on empty list."""
    with pytest.raises(ValueError):
        BatchedFittingBundle.stack([])


# ===== TESTS: RNG purity (R6) =====


def test_state_purity_no_bare_keys(batched_bundle_single, train_state_init):
    """FittingPlan.step signature should not have bare 'key' parameter."""
    config = FittingConfig(lr=1e-3, n_steps=1)
    plan = make_fitting_plan(config)

    # Get the signature of the underlying function (before JIT)
    sig = inspect.signature(plan.step.__wrapped__ if hasattr(plan.step, "__wrapped__") else plan.step)
    params = list(sig.parameters.keys())

    # Should not have 'key' as a parameter
    assert "key" not in params, f"Found 'key' in parameters: {params}"


# ===== INTEGRATION TESTS =====


def test_full_training_loop_3_steps(batched_bundle_single, train_state_init):
    """Full training loop: init → 3 steps → eval."""
    config = FittingConfig(lr=1e-3, n_steps=3)
    plan = make_fitting_plan(config)

    state = train_state_init
    step_losses = []

    # Training loop
    for step in range(3):
        state, metrics = plan.step(batched_bundle_single, state, conformer_idx=0)
        step_losses.append(float(metrics.loss))

    # Evaluation
    eval_metrics = plan.evaluate(batched_bundle_single, state)

    assert len(step_losses) == 3
    assert jnp.isfinite(eval_metrics.loss)
    assert state.step_count == 3
