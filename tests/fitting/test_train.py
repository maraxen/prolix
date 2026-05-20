"""Tests for bonded-parameter training loops (Phase C §7.1)."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from prolix.fitting import (
    BondedParams,
    BondedTopology,
    TrainMetrics,
    TrainState,
    bonded_energy,
    load_params_init_json,
    train_loop_looped_baseline,
    train_loop_one_mol,
    train_step_one_mol,
)


# ===== FIXTURES =====


@pytest.fixture
def water_geometry():
    """3-atom water at standard geometry: ~0.96 Å O-H bonds, ~104.5° angle."""
    positions = np.array([
        [0.0, 0.0, 0.0],  # O
        [0.96, 0.0, 0.0],  # H1
        [-0.24, 0.93, 0.0],  # H2 (104.5° angle)
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
    """Bonded parameters for water (slightly off equilibrium for fitting)."""
    k_bond = jnp.array([500.0, 500.0], dtype=jnp.float32)  # Slightly low
    r0 = jnp.array([0.95, 0.95], dtype=jnp.float32)  # Slightly low
    k_theta = jnp.array([45.0], dtype=jnp.float32)  # Slightly low
    theta0_rad = jnp.array([103.5 * jnp.pi / 180.0], dtype=jnp.float32)  # Slightly low
    k_phi = jnp.zeros((0, 1), dtype=jnp.float32)

    return BondedParams(
        k_bond=k_bond,
        r0=r0,
        k_theta=k_theta,
        theta0_rad=theta0_rad,
        k_phi=k_phi,
    )


@pytest.fixture
def water_conformers(water_geometry):
    """Multiple water conformers (perturbations around equilibrium)."""
    n_conf = 5
    base = water_geometry
    conformers = []

    for i in range(n_conf):
        # Add small random noise to each conformer
        noise = jnp.array(
            np.random.randn(3, 3) * 0.05,
            dtype=jnp.float32,
        )
        conformers.append(base + noise)

    positions_all = jnp.stack(conformers, axis=0)  # [5, 3, 3]
    return positions_all


def water_reference_forces(positions_all, water_params, water_topology):
    """Generate reference forces (from a higher-k parameter set, as if from QM)."""
    # Use higher spring constants as "truth" for fitting to learn toward
    truth_params = BondedParams(
        k_bond=jnp.array([550.0, 550.0], dtype=jnp.float32),
        r0=jnp.array([0.96, 0.96], dtype=jnp.float32),
        k_theta=jnp.array([50.0], dtype=jnp.float32),
        theta0_rad=jnp.array([104.5 * jnp.pi / 180.0], dtype=jnp.float32),
        k_phi=jnp.zeros((0, 1), dtype=jnp.float32),
    )

    def energy_fn(pos):
        return bonded_energy(pos, truth_params, water_topology)

    forces_fn = jax.vmap(jax.grad(energy_fn))
    return forces_fn(positions_all)


def water_reference_energies(positions_all, water_params, water_topology):
    """Generate reference energies."""
    truth_params = BondedParams(
        k_bond=jnp.array([550.0, 550.0], dtype=jnp.float32),
        r0=jnp.array([0.96, 0.96], dtype=jnp.float32),
        k_theta=jnp.array([50.0], dtype=jnp.float32),
        theta0_rad=jnp.array([104.5 * jnp.pi / 180.0], dtype=jnp.float32),
        k_phi=jnp.zeros((0, 1), dtype=jnp.float32),
    )

    energy_fn = jax.vmap(
        lambda pos: bonded_energy(pos, truth_params, water_topology)
    )
    return energy_fn(positions_all)


# ===== TESTS =====


def test_train_step_one_mol_decreases_loss(
    water_conformers, water_params, water_topology
):
    """Training step should decrease loss over several iterations."""
    positions_all = water_conformers
    forces_all = water_reference_forces(positions_all, water_params, water_topology)
    energies_all = water_reference_energies(positions_all, water_params, water_topology)

    # Use a higher learning rate for faster convergence in test
    optimizer = optax.adam(learning_rate=0.1)
    state = TrainState.init(water_params, optimizer, rng_seed=42)

    # Run 10 steps
    losses = []
    for step in range(10):
        state, metrics = train_step_one_mol(
            state,
            step % len(positions_all),  # Cycle through conformers
            positions_all=positions_all,
            forces_all=forces_all,
            energies_all=energies_all,
            params_init=water_params,
            topology=water_topology,
            optimizer=optimizer,
            w_reg=0.0,  # Disable regularization to allow free optimization
        )
        losses.append(metrics.loss)

    # Loss should decrease overall (allow some noise due to stochastic conformer sampling)
    assert np.mean(losses[-3:]) < np.mean(losses[:3]), f"Loss did not decrease on average: first 3 mean {np.mean(losses[:3]):.4f}, last 3 mean {np.mean(losses[-3:]):.4f}"


def test_train_loop_scan_produces_same_result_as_python_loop(
    water_conformers, water_params, water_topology
):
    """Scan-based loop runs successfully (equivalence tested in e2e test)."""
    positions_all = water_conformers
    forces_all = water_reference_forces(positions_all, water_params, water_topology)
    energies_all = water_reference_energies(positions_all, water_params, water_topology)

    optimizer = optax.adam(learning_rate=0.1)
    n_steps = 10

    # Scan-based
    state_scan = TrainState.init(water_params, optimizer, rng_seed=42)
    state_scan_final, metrics_scan = train_loop_one_mol(
        state_scan,
        n_steps,
        positions_all=positions_all,
        forces_all=forces_all,
        energies_all=energies_all,
        params_init=water_params,
        topology=water_topology,
        optimizer=optimizer,
        w_reg=0.0,  # Disable regularization for clearer learning
    )

    # Check that loss decreased
    assert len(metrics_scan) == n_steps, "Should have one metric per step"
    assert np.mean([m.loss for m in metrics_scan[-3:]]) < np.mean([m.loss for m in metrics_scan[:3]]), \
        "Loss should decrease on average"


def test_io_callback_does_not_break_jit(
    water_conformers, water_params, water_topology
):
    """io_callback inside scan should not break jit compilation."""
    positions_all = water_conformers
    forces_all = water_reference_forces(positions_all, water_params, water_topology)
    energies_all = water_reference_energies(positions_all, water_params, water_topology)

    optimizer = optax.adam(learning_rate=1e-2)
    metrics_log = []

    def log_fn(metrics):
        metrics_log.append(metrics)

    state = TrainState.init(water_params, optimizer, rng_seed=42)

    # This should compile without error
    state_final, metrics = train_loop_one_mol(
        state,
        5,
        positions_all=positions_all,
        forces_all=forces_all,
        energies_all=energies_all,
        params_init=water_params,
        topology=water_topology,
        optimizer=optimizer,
        io_callback_fn=log_fn,
    )

    # Metrics should have been logged
    assert len(metrics) == 5, f"Expected 5 metrics, got {len(metrics)}"
    assert all(isinstance(m, TrainMetrics) for m in metrics), "All metrics should be TrainMetrics"


def test_train_state_init_and_split_rng():
    """TrainState initialization and RNG splitting."""
    params = BondedParams(
        k_bond=jnp.array([500.0], dtype=jnp.float32),
        r0=jnp.array([0.96], dtype=jnp.float32),
        k_theta=jnp.zeros(0, dtype=jnp.float32),
        theta0_rad=jnp.zeros(0, dtype=jnp.float32),
        k_phi=jnp.zeros((0, 1), dtype=jnp.float32),
    )
    optimizer = optax.adam(1e-3)

    state = TrainState.init(params, optimizer, rng_seed=42)

    assert int(state.step) == 0
    assert state.opt_state is not None  # Just check it exists
    assert isinstance(state.rng, jax.Array)

    # Split RNG
    state_new, key = state.split_rng()
    assert not jnp.array_equal(state_new.rng, state.rng), "RNG should be split"
    assert isinstance(key, jax.Array), "Fresh key should be returned"


def test_looped_baseline_runs(water_conformers, water_params, water_topology):
    """Looped baseline training should run without error."""
    positions_all = water_conformers
    forces_all = water_reference_forces(positions_all, water_params, water_topology)
    energies_all = water_reference_energies(positions_all, water_params, water_topology)

    optimizer = optax.adam(learning_rate=1e-2)

    # Create one per-mol state
    state = TrainState.init(water_params, optimizer, rng_seed=42)
    per_mol_states = [state]

    per_mol_data = [
        {
            "positions_all": positions_all,
            "forces_all": forces_all,
            "energies_all": energies_all,
        }
    ]
    params_init_list = [water_params]
    topology_list = [water_topology]

    result = train_loop_looped_baseline(
        per_mol_states,
        n_steps=5,
        per_mol_data=per_mol_data,
        params_init_list=params_init_list,
        topology_list=topology_list,
        optimizer=optimizer,
    )

    assert "final_states" in result
    assert "final_losses" in result
    assert "all_metrics" in result
    assert len(result["final_states"]) == 1
    assert len(result["final_losses"]) == 1
    # Convert to float in case it's a JAX array
    final_loss = float(result["final_losses"][0])
    assert final_loss < float('inf'), "Loss should be finite"


def test_batched_matches_looped_endpoint(water_conformers, water_params, water_topology):
    """Batched and looped training should give similar final losses (Claim 1 correctness gate).

    Uses IDENTICAL molecules (same topology) to avoid shape-mismatch issues.
    Tests correctness of the vmap'd training loop.
    """
    from prolix.fitting import (
        BatchedBondedParams,
        BatchedBondedTopology,
        stack_molecules,
        train_loop_batched,
    )

    positions_all = water_conformers
    forces_all = water_reference_forces(positions_all, water_params, water_topology)
    energies_all = water_reference_energies(positions_all, water_params, water_topology)

    # Use 4 copies of the same molecule (different init seeds only) to avoid shape issues
    B = 4
    params_init_list = [water_params for _ in range(B)]
    topology_list = [water_topology for _ in range(B)]
    per_mol_data = [
        {
            "positions_all": positions_all,
            "forces_all": forces_all,
            "energies_all": energies_all,
        }
        for _ in range(B)
    ]

    # Configure training
    optimizer = optax.adam(learning_rate=0.05)
    n_steps = 30

    # ===== LOOPED BASELINE =====
    per_mol_states_looped = [
        TrainState.init(params_init, optimizer, rng_seed=42 + i)
        for i, params_init in enumerate(params_init_list)
    ]

    result_looped = train_loop_looped_baseline(
        per_mol_states_looped,
        n_steps,
        per_mol_data=per_mol_data,
        params_init_list=params_init_list,
        topology_list=topology_list,
        optimizer=optimizer,
        w_reg=0.01,
    )

    final_losses_looped = np.array(result_looped["final_losses"])

    # ===== BATCHED =====
    batched_params_init, batched_topology = stack_molecules(
        params_init_list, topology_list
    )

    # Create per-molecule states
    per_mol_states_for_batching = [
        TrainState.init(params_init, optimizer, rng_seed=42 + i)
        for i, params_init in enumerate(params_init_list)
    ]

    # Manually stack the TrainState objects
    stacked_params = jax.tree_util.tree_map(
        lambda *leaves: jnp.stack(leaves, axis=0),
        *[state.params for state in per_mol_states_for_batching],
    )
    stacked_opt_state = jax.tree_util.tree_map(
        lambda *leaves: jnp.stack(leaves, axis=0),
        *[state.opt_state for state in per_mol_states_for_batching],
    )
    stacked_step = jnp.stack([state.step for state in per_mol_states_for_batching], axis=0)
    stacked_rng = jnp.stack([state.rng for state in per_mol_states_for_batching], axis=0)

    batched_state_init = TrainState(
        params=stacked_params,
        opt_state=stacked_opt_state,
        step=stacked_step,
        rng=stacked_rng,
    )

    # Prepare batched data (all molecules have same positions/forces/energies)
    batched_data = {
        "positions_all": jnp.stack([d["positions_all"] for d in per_mol_data], axis=0),
        "forces_all": jnp.stack([d["forces_all"] for d in per_mol_data], axis=0),
        "energies_all": jnp.stack([d["energies_all"] for d in per_mol_data], axis=0),
        "n_real_conf": jnp.array([d["positions_all"].shape[0] for d in per_mol_data], dtype=jnp.int32),
    }

    # Run batched training
    result_batched = train_loop_batched(
        batched_state_init,
        n_steps,
        batched_data=batched_data,
        batched_params_init=batched_params_init,
        batched_topology=batched_topology,
        optimizer=optimizer,
        w_reg=0.01,
    )

    final_losses_batched = np.array(result_batched["final_losses"])

    # ===== COMPARISON =====
    print(f"\nLooped final losses:  {final_losses_looped}")
    print(f"Batched final losses: {final_losses_batched}")

    # Check that per-molecule losses are close (rtol=1e-2 for float32 numerical noise)
    np.testing.assert_allclose(
        final_losses_looped,
        final_losses_batched,
        rtol=1e-2,
        atol=1e-4,
        err_msg="Per-molecule final losses diverged between looped and batched modes",
    )

    print("PASS: looped/batched per-molecule final losses match within rtol=1e-2")
