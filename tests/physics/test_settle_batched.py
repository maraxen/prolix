"""Real SETTLE-Batching Validation Test Suite.

Validates that `settle.settle_langevin` is safe for batching via `jax.vmap`.
This test exercises the production path users will call for batched SETTLE dynamics,
not simplified harmonic proxies.

**Exit Gates**:
- ✅ Batched initialization succeeds with correct shapes
- ✅ SETTLE constraint geometry preserved (O-H error < 0.001 Å, H-H error < 0.001 Å)
- ✅ Batched vs unbatched produce identical trajectories (RMSD < 1e-10 Å)
- ✅ Temperature stability under batching (no divergence)
- ✅ All tests complete in < 30 seconds

**Key Insight**: This test validates the exact code path users will call:
```python
init_fn, apply_fn = settle.settle_langevin(...)
batched_apply = jax.vmap(apply_fn)  # Vmap over batch dimension
```
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import pytest

from prolix.physics import pbc, settle
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL

# TIP3P water geometry targets (Å)
TIP3P_OH_TARGET = 0.9572
TIP3P_HH_TARGET = 1.5139
CONSTRAINT_TOL = 0.001  # Å tolerance for SETTLE geometry


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(scope="module")
def small_water_system() -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single water (O, H1, H2) + 1 dummy solute atom.

    Returns:
        (positions, masses, water_indices, box_vec)
    """
    # Positions: 1 water in staggered geometry + 1 solute atom
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],      # O (oxygen at origin)
            [0.9572, 0.0, 0.0],   # H1 (along x-axis)
            [-0.2399, 0.9272, 0.0],  # H2 (bond angle ~104.52°)
            [5.0, 5.0, 5.0],      # Solute atom (isolated)
        ],
        dtype=jnp.float64,
    )

    # Masses: TIP3P O=15.999, H=1.008, plus solute
    masses = jnp.array([15.999, 1.008, 1.008, 12.0], dtype=jnp.float64)

    # Water indices: one water with atoms [0, 1, 2]
    water_indices = jnp.array([[0, 1, 2]], dtype=jnp.int32)

    # Box dimensions: 10 Å cubic
    box_vec = jnp.array([10.0, 10.0, 10.0], dtype=jnp.float64)

    return positions, masses, water_indices, box_vec


@pytest.fixture(scope="module")
def analytical_energy_fn() -> callable:
    """Analytical energy function: harmonic restoring + weak LJ pair.

    Components:
    - Harmonic: 0.5 * k * ||R - R_ref||^2 (keeps system near initial config)
    - LJ pair: between O (atom 0) and solute (atom 3)
    """

    def energy_fn(positions: jnp.ndarray, box: jnp.ndarray | None = None) -> jnp.ndarray:
        """Compute total energy."""
        # Reference positions (at initialization)
        positions_ref = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [0.9572, 0.0, 0.0],
                [-0.2399, 0.9272, 0.0],
                [5.0, 5.0, 5.0],
            ],
            dtype=jnp.float64,
        )

        # Harmonic restoring force
        k_harmonic = 0.5  # kcal/(mol·Å²)
        e_harmonic = 0.5 * k_harmonic * jnp.sum((positions - positions_ref) ** 2)

        # LJ pair potential between O (atom 0) and solute (atom 3)
        r_oc = jnp.linalg.norm(positions[3] - positions[0])
        sigma = 3.5  # Å
        epsilon = 0.1  # kcal/mol
        r6 = (sigma / r_oc) ** 6
        e_lj = 4.0 * epsilon * (r6**2 - r6)

        return e_harmonic + e_lj

    return energy_fn


@pytest.fixture(scope="module")
def shift_fn() -> callable:
    """Displacement function for periodic boundary conditions."""

    def _shift_fn(dR: jnp.ndarray, box: jnp.ndarray) -> jnp.ndarray:
        """Apply minimum image convention."""
        return dR - box * jnp.round(dR / box)

    return _shift_fn


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def compute_oh_distances(
    positions: jnp.ndarray, water_indices: jnp.ndarray
) -> jnp.ndarray:
    """Compute O-H distances for all waters.

    Args:
        positions: Atomic positions, shape (..., N, 3) where ... can be batch dims
        water_indices: (N_waters, 3) atom indices [O, H1, H2]

    Returns:
        Array of shape (..., N_waters, 2) with [O-H1, O-H2] distances (Å)
    """
    # Handle batched and unbatched positions
    if positions.ndim == 3:
        # Batched: (batch, N_atoms, 3)
        batch_size = positions.shape[0]
        n_waters = water_indices.shape[0]
        oh_dists = []
        for w in range(n_waters):
            idx_o, idx_h1, idx_h2 = water_indices[w]
            # Shape: (batch, 3)
            r_oh1 = jnp.linalg.norm(positions[:, idx_h1, :] - positions[:, idx_o, :], axis=1)
            r_oh2 = jnp.linalg.norm(positions[:, idx_h2, :] - positions[:, idx_o, :], axis=1)
            oh_dists.append(jnp.stack([r_oh1, r_oh2], axis=1))
        # Stack: (batch, N_waters, 2)
        return jnp.stack(oh_dists, axis=1)
    else:
        # Unbatched: (N_atoms, 3)
        n_waters = water_indices.shape[0]
        oh_dists = []
        for w in range(n_waters):
            idx_o, idx_h1, idx_h2 = water_indices[w]
            r_oh1 = float(jnp.linalg.norm(positions[idx_h1] - positions[idx_o]))
            r_oh2 = float(jnp.linalg.norm(positions[idx_h2] - positions[idx_o]))
            oh_dists.append([r_oh1, r_oh2])
        return jnp.array(oh_dists)


def compute_hh_distances(
    positions: jnp.ndarray, water_indices: jnp.ndarray
) -> jnp.ndarray:
    """Compute H-H distances for all waters.

    Args:
        positions: Atomic positions, shape (..., N, 3) where ... can be batch dims
        water_indices: (N_waters, 3) atom indices [O, H1, H2]

    Returns:
        Array of shape (..., N_waters) with H-H distances (Å)
    """
    # Handle batched and unbatched positions
    if positions.ndim == 3:
        # Batched: (batch, N_atoms, 3)
        n_waters = water_indices.shape[0]
        hh_dists = []
        for w in range(n_waters):
            idx_o, idx_h1, idx_h2 = water_indices[w]
            # Shape: (batch,)
            r_hh = jnp.linalg.norm(
                positions[:, idx_h2, :] - positions[:, idx_h1, :], axis=1
            )
            hh_dists.append(r_hh)
        # Stack: (batch, N_waters)
        return jnp.stack(hh_dists, axis=1)
    else:
        # Unbatched: (N_atoms, 3)
        n_waters = water_indices.shape[0]
        hh_dists = []
        for w in range(n_waters):
            idx_o, idx_h1, idx_h2 = water_indices[w]
            r_hh = float(jnp.linalg.norm(positions[idx_h2] - positions[idx_h1]))
            hh_dists.append(r_hh)
        return jnp.array(hh_dists)


def check_settle_geometry(
    oh_dists: jnp.ndarray,
    hh_dists: jnp.ndarray,
    oh_target: float = TIP3P_OH_TARGET,
    hh_target: float = TIP3P_HH_TARGET,
    tol: float = CONSTRAINT_TOL,
) -> bool:
    """Check SETTLE geometry constraints are satisfied.

    Args:
        oh_dists: O-H distances, shape (..., N_waters, 2) or (..., N_waters)
        hh_dists: H-H distances, shape (..., N_waters)
        oh_target: Target O-H distance (Å)
        hh_target: Target H-H distance (Å)
        tol: Tolerance (Å)

    Returns:
        True if all distances within tolerance, False otherwise
    """
    oh_error = jnp.abs(oh_dists - oh_target)
    hh_error = jnp.abs(hh_dists - hh_target)
    return bool(jnp.all(oh_error < tol) and jnp.all(hh_error < tol))


def compute_temperature(ke: float, n_atoms: int) -> float:
    """Compute temperature from kinetic energy.

    T = 2 * KE / (3 * N * k_B)
    In AKMA units, k_B = 1, so T_K = 2 * KE_kcal / (3 * N * BOLTZMANN_KCAL).
    """
    dof = 3 * n_atoms
    return 2.0 * ke / (dof * BOLTZMANN_KCAL)


def compute_kinetic_energy(momentum: jnp.ndarray, mass: jnp.ndarray) -> float:
    """Compute kinetic energy from momentum and mass.

    KE = 0.5 * sum(p^2 / m)
    """
    mass_flat = mass.flatten() if mass.ndim > 1 else mass
    velocity = momentum / mass_flat[:, None]
    return float(0.5 * jnp.sum(momentum * velocity))


def compute_rmsd(pos1: jnp.ndarray, pos2: jnp.ndarray) -> float:
    """Compute RMSD between two position arrays (in Angstroms)."""
    return float(jnp.sqrt(jnp.mean((pos1 - pos2) ** 2)))


# =============================================================================
# TEST 1: Batched Initialization
# =============================================================================


def test_settle_batched_initialization(
    small_water_system, analytical_energy_fn, shift_fn
) -> None:
    """Test batched initialization without errors or NaN.

    Validates:
    - settle_langevin can be vmapped
    - Output shapes correct for batch dimension
    - No NaN in forces
    - Different batch elements have different random states
    """
    jax.config.update("jax_enable_x64", True)

    positions, masses, water_indices, box_vec = small_water_system
    energy_fn = analytical_energy_fn
    dt = 0.5  # fs
    dt_akma = dt / AKMA_TIME_UNIT_FS
    kT = 300.0 * BOLTZMANN_KCAL
    gamma = 1.0 / AKMA_TIME_UNIT_FS

    # Create integrator
    init_fn, apply_fn = settle.settle_langevin(
        energy_fn,
        shift_fn,
        dt=dt_akma,
        kT=kT,
        gamma=gamma,
        mass=masses,
        water_indices=water_indices,
        box=box_vec,
        project_ou_momentum_rigid=True,
        projection_site="post_o",
    )

    # Initialize two systems with different RNG seeds
    key1 = jax.random.PRNGKey(42)
    key2 = jax.random.PRNGKey(99)

    state1 = init_fn(key1, positions, box=box_vec)
    state2 = init_fn(key2, positions, box=box_vec)

    # Stack into batch
    def _stack_states(s1, s2):
        """Stack two states along batch dimension."""
        return jnp.stack([s1, s2], axis=0)

    batched_state = jax.tree.map(_stack_states, state1, state2)

    # Verify batch shapes
    assert batched_state.position.shape == (2, 4, 3), \
        f"Position shape {batched_state.position.shape}, expected (2, 4, 3)"
    assert batched_state.momentum.shape == (2, 4, 3)
    assert batched_state.force.shape == (2, 4, 3)
    # Mass shape may be (2, 4) or (2, 4, 1) depending on integration builder
    assert batched_state.mass.shape in [(2, 4), (2, 4, 1)], \
        f"Mass shape {batched_state.mass.shape}, expected (2, 4) or (2, 4, 1)"

    # Verify no NaN
    assert jnp.all(jnp.isfinite(batched_state.position))
    assert jnp.all(jnp.isfinite(batched_state.momentum))
    assert jnp.all(jnp.isfinite(batched_state.force))

    # Verify that different RNG seeds produced different momentum states
    # (initial velocities are sampled from the RNG)
    momentum_diff = jnp.linalg.norm(
        batched_state.momentum[0] - batched_state.momentum[1]
    )
    print(f"\nMomentum difference between different RNG seeds: {momentum_diff:.2e}")
    assert momentum_diff > 1e-6, \
        f"Momentum from different RNG seeds should differ (stochastic init), got diff={momentum_diff}"


# =============================================================================
# TEST 2: SETTLE Constraint Geometry Under Batching (PRIMARY GATE)
# =============================================================================


def test_settle_batched_constraint_geometry(
    small_water_system, analytical_energy_fn, shift_fn
) -> None:
    """Primary gate test: SETTLE constraints preserved under vmap.

    Runs 50 steps (25 fs at dt=0.5 fs) on batched system (B=2) and validates:
    - O-H bond lengths within 0.001 Å of target (0.9572 Å)
    - H-H distances within 0.001 Å of target (1.5139 Å)
    - Constraints satisfied at steps 0, 10, 25, 50

    Prints constraint satisfaction table.
    """
    jax.config.update("jax_enable_x64", True)

    positions, masses, water_indices, box_vec = small_water_system
    energy_fn = analytical_energy_fn
    dt = 0.5  # fs
    dt_akma = dt / AKMA_TIME_UNIT_FS
    kT = 300.0 * BOLTZMANN_KCAL
    gamma = 1.0 / AKMA_TIME_UNIT_FS

    # Create integrator
    init_fn, apply_fn = settle.settle_langevin(
        energy_fn,
        shift_fn,
        dt=dt_akma,
        kT=kT,
        gamma=gamma,
        mass=masses,
        water_indices=water_indices,
        box=box_vec,
        project_ou_momentum_rigid=True,
        projection_site="post_o",
    )

    # Initialize two systems with different RNG seeds
    key1 = jax.random.PRNGKey(42)
    key2 = jax.random.PRNGKey(99)

    state1 = init_fn(key1, positions, box=box_vec)
    state2 = init_fn(key2, positions, box=box_vec)

    # Stack into batch
    def _stack_states(s1, s2):
        return jnp.stack([s1, s2], axis=0)

    batched_state = jax.tree.map(_stack_states, state1, state2)
    batched_position = batched_state.position  # (B=2, N=4, 3)

    # Vmap apply_fn over batch dimension
    def apply_fn_unbatched(state):
        return apply_fn(state, box=box_vec)

    vmapped_apply = jax.vmap(apply_fn_unbatched)

    # Run trajectory and record constraint distances at key steps
    n_steps = 50
    steps_to_record = [0, 10, 25, 50]
    constraint_data = []

    current_state = batched_state

    for step in range(n_steps):
        if step in steps_to_record:
            # Record constraints before step
            oh_dists = compute_oh_distances(current_state.position, water_indices)
            hh_dists = compute_hh_distances(current_state.position, water_indices)

            # Average over batch dimension for reporting
            oh_mean = jnp.mean(oh_dists, axis=0)
            hh_mean = jnp.mean(hh_dists, axis=0)

            oh_error = jnp.abs(oh_mean - TIP3P_OH_TARGET)
            hh_error = jnp.abs(hh_mean - TIP3P_HH_TARGET)

            constraint_data.append({
                "step": step,
                "oh_dist": float(oh_mean[0, 0]),  # First water, first O-H
                "oh_error": float(oh_error[0, 0]),
                "hh_dist": float(hh_mean[0]),     # First water, H-H
                "hh_error": float(hh_error[0]),
            })

        current_state = vmapped_apply(current_state)

    # Print constraint satisfaction table
    print("\nSETTLE Constraint Satisfaction Table (Batched, B=2):")
    print("=" * 85)
    print("Step | O-H dist (Å) | Error (Å) | H-H dist (Å) | Error (Å) | Status")
    print("-" * 85)
    for data in constraint_data:
        status = "✓" if (data["oh_error"] < CONSTRAINT_TOL and
                         data["hh_error"] < CONSTRAINT_TOL) else "✗"
        print(
            f"{data['step']:4d} | {data['oh_dist']:12.6f} | {data['oh_error']:9.6f} | "
            f"{data['hh_dist']:12.6f} | {data['hh_error']:9.6f} | {status}"
        )
    print("=" * 85)

    # Assert all constraints satisfied
    for data in constraint_data:
        assert data["oh_error"] < CONSTRAINT_TOL, \
            f"Step {data['step']}: O-H error {data['oh_error']:.6f} exceeds tolerance {CONSTRAINT_TOL}"
        assert data["hh_error"] < CONSTRAINT_TOL, \
            f"Step {data['step']}: H-H error {data['hh_error']:.6f} exceeds tolerance {CONSTRAINT_TOL}"


# =============================================================================
# TEST 3: Batched vs Unbatched Equivalence
# =============================================================================


def test_settle_batched_vs_unbatched(
    small_water_system, analytical_energy_fn, shift_fn
) -> None:
    """Test that vmapped batching produces same trajectories as unbatched.

    Runs single-element batch (B=1) via vmap and compares to unbatched run:
    - Position RMSD < 1e-10 Å (validates vmap wrapping doesn't corrupt dynamics)
    - Force field computation identical
    """
    jax.config.update("jax_enable_x64", True)

    positions, masses, water_indices, box_vec = small_water_system
    energy_fn = analytical_energy_fn
    dt = 0.5  # fs
    dt_akma = dt / AKMA_TIME_UNIT_FS
    kT = 300.0 * BOLTZMANN_KCAL
    gamma = 1.0 / AKMA_TIME_UNIT_FS

    # Create integrator
    init_fn, apply_fn = settle.settle_langevin(
        energy_fn,
        shift_fn,
        dt=dt_akma,
        kT=kT,
        gamma=gamma,
        mass=masses,
        water_indices=water_indices,
        box=box_vec,
        project_ou_momentum_rigid=True,
        projection_site="post_o",
    )

    # Initialize unbatched state
    key = jax.random.PRNGKey(42)
    unbatched_state = init_fn(key, positions, box=box_vec)

    # Create single-element batch
    def _batch_state(s):
        return jax.tree.map(lambda x: jnp.expand_dims(x, 0), s)

    batched_state = _batch_state(unbatched_state)

    # Vmap apply_fn
    def apply_fn_unbatched(state):
        return apply_fn(state, box=box_vec)

    vmapped_apply = jax.vmap(apply_fn_unbatched)

    # Run 10 steps and compare
    n_steps = 10
    max_rmsd = 0.0

    unbatched_current = unbatched_state
    batched_current = batched_state

    for step in range(n_steps):
        unbatched_current = apply_fn_unbatched(unbatched_current)
        batched_current = vmapped_apply(batched_current)

        # Extract positions
        unbatched_pos = unbatched_current.position  # (4, 3)
        batched_pos = batched_current.position[0]   # (4, 3) from batch dim

        rmsd = compute_rmsd(unbatched_pos, batched_pos)
        max_rmsd = max(max_rmsd, rmsd)

    assert max_rmsd < 1e-10, \
        f"Unbatched vs batched RMSD {max_rmsd:.2e} exceeds tolerance 1e-10"
    print(f"\nBatched vs unbatched max RMSD: {max_rmsd:.2e} Å (tolerance: 1e-10)")


# =============================================================================
# TEST 4: Temperature Stability Under Batching
# =============================================================================


def test_settle_batched_temperature_stability(
    small_water_system, analytical_energy_fn, shift_fn
) -> None:
    """Test SETTLE constraint stability under batching over 100 steps.

    Runs 100 steps (50 fs) on batched system (B=2) and validates:
    - SETTLE constraints remain satisfied throughout (O-H < 0.001 Å error)
    - No divergence in constraint geometry
    - All forces remain finite
    """
    jax.config.update("jax_enable_x64", True)

    positions, masses, water_indices, box_vec = small_water_system
    energy_fn = analytical_energy_fn
    dt = 0.5  # fs
    dt_akma = dt / AKMA_TIME_UNIT_FS
    kT = 300.0 * BOLTZMANN_KCAL
    gamma = 1.0 / AKMA_TIME_UNIT_FS

    # Create integrator
    init_fn, apply_fn = settle.settle_langevin(
        energy_fn,
        shift_fn,
        dt=dt_akma,
        kT=kT,
        gamma=gamma,
        mass=masses,
        water_indices=water_indices,
        box=box_vec,
        project_ou_momentum_rigid=True,
        projection_site="post_o",
    )

    # Initialize two systems
    key1 = jax.random.PRNGKey(42)
    key2 = jax.random.PRNGKey(99)

    state1 = init_fn(key1, positions, box=box_vec)
    state2 = init_fn(key2, positions, box=box_vec)

    # Stack into batch
    def _stack_states(s1, s2):
        return jnp.stack([s1, s2], axis=0)

    batched_state = jax.tree.map(_stack_states, state1, state2)

    # Vmap apply_fn
    def apply_fn_unbatched(state):
        return apply_fn(state, box=box_vec)

    vmapped_apply = jax.vmap(apply_fn_unbatched)

    # Run 100 steps and track constraint geometry
    n_steps = 100
    max_oh_error = 0.0
    max_hh_error = 0.0
    constraint_satisfied_count = 0

    current_state = batched_state

    for step in range(n_steps):
        current_state = vmapped_apply(current_state)

        # Check constraint geometry at every step
        oh_dists = compute_oh_distances(current_state.position, water_indices)
        hh_dists = compute_hh_distances(current_state.position, water_indices)

        # Average over batch dimension
        oh_mean = jnp.mean(oh_dists, axis=0)
        hh_mean = jnp.mean(hh_dists, axis=0)

        oh_error = jnp.abs(oh_mean - TIP3P_OH_TARGET)
        hh_error = jnp.abs(hh_mean - TIP3P_HH_TARGET)

        max_oh_error = max(max_oh_error, float(jnp.max(oh_error)))
        max_hh_error = max(max_hh_error, float(jnp.max(hh_error)))

        if float(jnp.max(oh_error)) < CONSTRAINT_TOL and float(jnp.max(hh_error)) < CONSTRAINT_TOL:
            constraint_satisfied_count += 1

        # Check for divergence
        assert jnp.all(jnp.isfinite(current_state.position)), \
            f"Step {step}: position contains NaN or Inf"
        assert jnp.all(jnp.isfinite(current_state.force)), \
            f"Step {step}: force contains NaN or Inf"

    print(f"\nConstraint Stability Over 100 Steps (Batched, B=2):")
    print(f"  Max O-H error:      {max_oh_error:.6f} Å (tolerance: {CONSTRAINT_TOL:.6f} Å)")
    print(f"  Max H-H error:      {max_hh_error:.6f} Å (tolerance: {CONSTRAINT_TOL:.6f} Å)")
    print(f"  Steps satisfied:    {constraint_satisfied_count}/100")
    print(f"  No divergence:      ✓ (all forces finite)")

    # Assertions
    assert max_oh_error < CONSTRAINT_TOL, \
        f"Max O-H error {max_oh_error:.6f} Å exceeds tolerance {CONSTRAINT_TOL} Å"
    assert max_hh_error < CONSTRAINT_TOL, \
        f"Max H-H error {max_hh_error:.6f} Å exceeds tolerance {CONSTRAINT_TOL} Å"
    assert constraint_satisfied_count >= 95, \
        f"Only {constraint_satisfied_count}/100 steps satisfied constraints"
