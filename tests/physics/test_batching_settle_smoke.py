"""Smoke test for batched SETTLE constraint integration.

This test validates that the SETTLE constraint algorithm works correctly
under vmap (batched) execution. This is a CRITICAL v1.0 blocker because:

1. Batching equivalence tests (test_batching_equivalence.py) only validate
   unconstrained systems (water_indices=None).
2. Production use of batching requires SETTLE constraint support.
3. This smoke test fills that gap with minimal coverage of the critical path.

**Test Strategy**:
- Test 1: Initialization with SETTLE constraints (shapes, no NaN)
- Test 2: 100-step smoke run (50 fs) with geometry validation
- Test 3: Equivalence between batched (batch_size=1) and unbatched

**Failure Escalation**:
- NaN/Inf divergence → potential vmap + SETTLE composition bug
- Temperature runaway → thermostat coupling issue
- Constraint geometry violation → SETTLE projection bug under vmap

**Author**: Fixer Agent (Phase 4 remediation)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prolix.physics.integrator_builder import make_integrator, make_integrator_batched
from prolix.typing import IntegratorState


# ========== FIXTURES ==========

@pytest.fixture
def water_system_batch_2():
    """4-water system in 8.0 Å cubic box, batch_size=2.

    - 4 TIP3P water molecules (12 atoms total)
    - Cubic box: 8.0 Å
    - Batch size: 2 independent trajectories
    - Water indices: [[0,1,2], [3,4,5], [6,7,8], [9,10,11]] (O, H1, H2)
    """
    # Create 4 water molecules in a grid layout
    # Each water: O at center, H1 and H2 offset by O-H bond length and angle
    tip3p_roh = 0.9572  # O-H bond length (Å)
    tip3p_theta = 104.52 * jnp.pi / 180.0  # H-O-H angle (rad)

    # Offset for H atoms from O (in local frame)
    # H1: distance roh along x-axis
    # H2: distance roh at angle theta from H1
    h_offset_x = tip3p_roh
    h_offset_y_h1 = 0.0
    h_offset_y_h2 = tip3p_roh * jnp.sin(tip3p_theta)
    h_offset_x_h2 = tip3p_roh * jnp.cos(tip3p_theta)

    # Build 4 waters in a 2x2 grid (spacing 3.0 Å)
    waters = []
    for i in range(2):
        for j in range(2):
            ox = 1.5 + 3.0 * i
            oy = 1.5 + 3.0 * j
            oz = 4.0

            o_pos = jnp.array([ox, oy, oz], dtype=jnp.float64)
            h1_pos = jnp.array([ox + h_offset_x, oy + h_offset_y_h1, oz], dtype=jnp.float64)
            h2_pos = jnp.array([ox + h_offset_x_h2, oy + h_offset_y_h2, oz], dtype=jnp.float64)

            waters.append([o_pos, h1_pos, h2_pos])

    # Flatten: [O1, H1_1, H2_1, O2, H1_2, H2_2, O3, H1_3, H2_3, O4, H1_4, H2_4]
    positions_unbatched = jnp.stack([w for water in waters for w in water])

    # Water indices: (4, 3) array of [O, H1, H2] indices
    water_indices = jnp.array([
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11],
    ], dtype=jnp.int32)

    # Masses: 4 oxygens (15.999 amu) + 8 hydrogens (1.008 amu)
    # Shape: (12, 1) for proper broadcasting with momentum (B, 12, 3) under vmap
    mass = jnp.array([[15.999], [1.008], [1.008], [15.999], [1.008], [1.008],
                      [15.999], [1.008], [1.008], [15.999], [1.008], [1.008]], dtype=jnp.float64)

    # Simple harmonic potential centered on each water
    def energy_fn(positions, box=None):
        """Harmonic potential to keep waters roughly in place."""
        # Compute O positions
        o_indices = jnp.array([0, 3, 6, 9], dtype=jnp.int32)
        o_positions = positions[o_indices]

        # Target O positions (grid centers)
        targets = jnp.array([
            [1.5, 1.5, 4.0],
            [4.5, 1.5, 4.0],
            [1.5, 4.5, 4.0],
            [4.5, 4.5, 4.0],
        ], dtype=jnp.float64)

        # Harmonic potential: 0.1 * sum((r - target)^2)
        energy = 0.1 * jnp.sum((o_positions - targets)**2)
        return energy

    def shift_fn(positions, box=None):
        """No-op shift function for this test."""
        return positions

    box = jnp.array([8.0, 8.0, 8.0], dtype=jnp.float64)

    return {
        "positions_unbatched": positions_unbatched,
        "water_indices": water_indices,
        "mass": mass,
        "energy_fn": energy_fn,
        "shift_fn": shift_fn,
        "box": box,
    }


@pytest.fixture
def rng_key():
    """JAX PRNG key for deterministic tests."""
    return jax.random.PRNGKey(999)


# ========== HELPER FUNCTIONS ==========

def compute_water_geometry(positions, water_indices):
    """Compute O-H distances and H-H distance for constraint validation.

    Args:
        positions: (N, 3) positions array (unbatched) or (B, N, 3) batched
        water_indices: (n_waters, 3) array of [O, H1, H2] indices

    Returns:
        (oh1_dists, oh2_dists, hh_dists): arrays of distances
    """
    o_indices = water_indices[:, 0]
    h1_indices = water_indices[:, 1]
    h2_indices = water_indices[:, 2]

    # Handle both unbatched and batched
    if positions.ndim == 3:  # Batched: (B, N, 3)
        pos_o = positions[:, o_indices, :]  # (B, n_waters, 3)
        pos_h1 = positions[:, h1_indices, :]
        pos_h2 = positions[:, h2_indices, :]
    else:  # Unbatched: (N, 3)
        pos_o = positions[o_indices, :]  # (n_waters, 3)
        pos_h1 = positions[h1_indices, :]
        pos_h2 = positions[h2_indices, :]

    # Compute distances
    oh1_dists = jnp.linalg.norm(pos_h1 - pos_o, axis=-1)  # (n_waters,) or (B, n_waters)
    oh2_dists = jnp.linalg.norm(pos_h2 - pos_o, axis=-1)
    hh_dists = jnp.linalg.norm(pos_h2 - pos_h1, axis=-1)

    return oh1_dists, oh2_dists, hh_dists


def compute_temperature(momentum, mass, n_dof=None):
    """Compute temperature from kinetic energy.

    Args:
        momentum: (N, 3) or (B, N, 3) momenta array
        mass: (N, 1) mass array
        n_dof: Number of degrees of freedom (optional; computed if None)

    Returns:
        Temperature in Kelvin (or thermal units, scalar or (B,) array)
    """
    # Ensure mass is (N, 1) for proper broadcasting
    if mass.ndim == 1:
        mass = mass[:, None]  # (N,) -> (N, 1)

    if momentum.ndim == 3:  # Batched: (B, N, 3)
        # Reshape mass to (N, 1) for broadcasting with (B, N, 3)
        v = momentum / mass.T.reshape(1, -1, 1)  # (1, N, 1) broadcasts to (B, N, 3)
        ke = 0.5 * jnp.sum(mass.T.reshape(1, -1, 1) * v**2, axis=(1, 2))  # (B,)
        if n_dof is None:
            n_dof = momentum.shape[1] * 3
        temperature = 2 * ke / n_dof
    else:  # Unbatched: (N, 3)
        v = momentum / mass  # (N, 1) broadcasts to (N, 3)
        ke = 0.5 * jnp.sum(mass * v**2)
        if n_dof is None:
            n_dof = momentum.shape[0] * 3
        temperature = 2 * ke / n_dof

    return temperature


# ========== TESTS ==========

def test_batching_settle_smoke_initialization(water_system_batch_2, rng_key):
    """Test that init_fn_batched returns correct shapes with SETTLE constraints.

    **Gate**: No NaN/Inf in initialized state; shapes correct.
    """
    batch_size = 2
    positions_unbatched = water_system_batch_2["positions_unbatched"]
    water_indices = water_system_batch_2["water_indices"]
    mass = water_system_batch_2["mass"]
    energy_fn = water_system_batch_2["energy_fn"]
    shift_fn = water_system_batch_2["shift_fn"]
    box = water_system_batch_2["box"]

    init_fn_batched, _ = make_integrator_batched(
        energy_fn=energy_fn,
        shift_fn=shift_fn,
        mass=mass,
        batch_size=batch_size,
        sequence_name="baoab_langevin",
        dt=0.5,
        kT=2.479,  # 300 K in thermal units
        gamma=1.0,
        water_indices=water_indices,
    )

    # Create batched positions: tile unbatched across batch dimension
    positions_batch = jnp.tile(positions_unbatched[None, :, :], (batch_size, 1, 1))

    # Initialize
    state_batch = init_fn_batched(rng_key, positions_batch, box=box)

    # Check shapes
    n_atoms = positions_unbatched.shape[0]
    assert state_batch.positions.shape == (batch_size, n_atoms, 3), \
        f"Expected position shape ({batch_size}, {n_atoms}, 3), got {state_batch.positions.shape}"
    assert state_batch.momentum.shape == (batch_size, n_atoms, 3)
    assert state_batch.force.shape == (batch_size, n_atoms, 3)
    assert state_batch.rng.shape == (batch_size, 2)

    # Check no NaN/Inf
    assert not jnp.isnan(state_batch.positions).any(), "NaN in position"
    assert not jnp.isinf(state_batch.positions).any(), "Inf in position"
    assert not jnp.isnan(state_batch.force).any(), "NaN in force"
    assert not jnp.isinf(state_batch.force).any(), "Inf in force"

    # Check that force field is computed (non-zero)
    assert jnp.any(jnp.abs(state_batch.force) > 1e-6), \
        "Force field appears zero; energy_fn may not be called"


@pytest.mark.slow
def test_batching_settle_smoke_100_steps(water_system_batch_2, rng_key):
    """100-step smoke run (50 fs at dt=0.5 fs) with geometry validation.

    **CRITICAL GATE**:
    - No NaN/Inf at any step
    - Temperature < 500 K (upper bound to catch runaway)
    - O-H distances within ±0.01 Å of target (0.9572 Å)
    - H-H distances within ±0.01 Å of target (1.5139 Å)
    """
    batch_size = 2
    n_steps = 100

    positions_unbatched = water_system_batch_2["positions_unbatched"]
    water_indices = water_system_batch_2["water_indices"]
    mass = water_system_batch_2["mass"]
    energy_fn = water_system_batch_2["energy_fn"]
    shift_fn = water_system_batch_2["shift_fn"]
    box = water_system_batch_2["box"]

    init_fn_batched, apply_fn_batched = make_integrator_batched(
        energy_fn=energy_fn,
        shift_fn=shift_fn,
        mass=mass,
        batch_size=batch_size,
        sequence_name="baoab_langevin",
        dt=0.5,
        kT=2.479,  # 300 K in thermal units
        gamma=1.0,
        water_indices=water_indices,
    )

    # Initialize
    positions_batch = jnp.tile(positions_unbatched[None, :, :], (batch_size, 1, 1))
    state_batch = init_fn_batched(rng_key, positions_batch, box=box)

    # Track statistics
    max_oh_error = 0.0
    max_hh_error = 0.0
    max_temperature = 0.0
    nan_step = None

    target_roh = 0.9572
    target_rhh = 1.5139

    # Run 100 steps
    for step in range(n_steps):
        state_batch = apply_fn_batched(state_batch)

        # Check for NaN/Inf
        if jnp.isnan(state_batch.positions).any() or jnp.isinf(state_batch.positions).any():
            nan_step = step
            break

        # Compute geometry
        oh1_dists, oh2_dists, hh_dists = compute_water_geometry(
            state_batch.positions, water_indices
        )

        # Track max error (across all batch elements and waters)
        oh_error = jnp.maximum(
            jnp.abs(oh1_dists - target_roh).max(),
            jnp.abs(oh2_dists - target_roh).max(),
        )
        hh_error = jnp.abs(hh_dists - target_rhh).max()
        max_oh_error = max(max_oh_error, float(oh_error))
        max_hh_error = max(max_hh_error, float(hh_error))

        # Compute temperature
        temperature = compute_temperature(state_batch.momentum, mass)
        max_temperature = max(max_temperature, float(temperature.max()))

    # Report results
    print(f"\nSETTLE Smoke Test Results (100 steps, 50 fs at dt=0.5 fs):")
    print(f"  Batch size: {batch_size}")
    print(f"  Max O-H distance error: {max_oh_error:.6f} Å (target {target_roh})")
    print(f"  Max H-H distance error: {max_hh_error:.6f} Å (target {target_rhh})")
    print(f"  Max temperature: {max_temperature:.1f} K (target 300, threshold 500)")

    # Assertions
    assert nan_step is None, \
        f"NaN detected at step {nan_step} — vmap + SETTLE composition issue"

    assert max_oh_error < 0.01, \
        f"O-H constraint violated: error {max_oh_error:.6f} Å exceeds 0.01 Å tolerance"

    assert max_hh_error < 0.01, \
        f"H-H constraint violated: error {max_hh_error:.6f} Å exceeds 0.01 Å tolerance"

    assert max_temperature < 500.0, \
        f"Temperature runaway: {max_temperature:.1f} K exceeds 500 K threshold"

    print(f"  Status: PASS ✓")


@pytest.mark.slow
def test_batching_settle_vs_unbatched(water_system_batch_2, rng_key):
    """Equivalence test: batched (batch_size=1) vs unbatched.

    Runs same 100-step trajectory in both modes and checks RMSD < 1e-10 Å.

    **CRITICAL GATE**: Trajectory equivalence validates correct vmap composition.
    """
    n_steps = 100

    positions_unbatched = water_system_batch_2["positions_unbatched"]
    water_indices = water_system_batch_2["water_indices"]
    mass = water_system_batch_2["mass"]
    energy_fn = water_system_batch_2["energy_fn"]
    shift_fn = water_system_batch_2["shift_fn"]
    box = water_system_batch_2["box"]

    # Create unbatched integrator
    init_fn_unbatched, apply_fn_unbatched = make_integrator(
        energy_fn=energy_fn,
        shift_fn=shift_fn,
        mass=mass,
        sequence_name="baoab_langevin",
        dt=0.5,
        kT=2.479,
        gamma=1.0,
        water_indices=water_indices,
    )

    # Create batched integrator with batch_size=1
    init_fn_batched, apply_fn_batched = make_integrator_batched(
        energy_fn=energy_fn,
        shift_fn=shift_fn,
        mass=mass,
        batch_size=1,
        sequence_name="baoab_langevin",
        dt=0.5,
        kT=2.479,
        gamma=1.0,
        water_indices=water_indices,
    )

    # Initialize unbatched
    state_unbatched = init_fn_unbatched(rng_key, positions_unbatched, box=box)

    # Initialize batched
    positions_batch = positions_unbatched[None, :, :]  # (1, N, 3)
    state_batched = init_fn_batched(rng_key, positions_batch, box=box)

    # Run 100 steps and track RMSD
    max_rmsd = 0.0
    max_constraint_drift = 0.0
    target_roh = 0.9572
    target_rhh = 1.5139

    for step in range(n_steps):
        state_unbatched = apply_fn_unbatched(state_unbatched)
        state_batched = apply_fn_batched(state_batched)

        # Sample at specific steps
        if step in [0, 10, 50, 99]:
            # RMSD
            pos_diff = state_unbatched.positions - state_batched.positions[0]
            rmsd = jnp.sqrt(jnp.mean(pos_diff**2))
            max_rmsd = max(max_rmsd, float(rmsd))

            # Check constraint geometry in both
            oh1_u, oh2_u, hh_u = compute_water_geometry(
                state_unbatched.positions, water_indices
            )
            oh1_b, oh2_b, hh_b = compute_water_geometry(
                state_batched.positions[0], water_indices
            )

            oh_error_u = jnp.maximum(
                jnp.abs(oh1_u - target_roh).max(),
                jnp.abs(oh2_u - target_roh).max(),
            )
            oh_error_b = jnp.maximum(
                jnp.abs(oh1_b - target_roh).max(),
                jnp.abs(oh2_b - target_roh).max(),
            )
            drift = jnp.abs(oh_error_u - oh_error_b)
            max_constraint_drift = max(max_constraint_drift, float(drift))

    print(f"\nBatched vs Unbatched Equivalence Test (100 steps):")
    print(f"  Max RMSD: {max_rmsd:.2e} Å (gate: < 1e-10)")
    print(f"  Max constraint drift: {max_constraint_drift:.2e} Å")

    assert max_rmsd < 1e-10, \
        f"Batched vs unbatched RMSD {max_rmsd:.2e} exceeds 1e-10 Å"

    assert max_constraint_drift < 1e-6, \
        f"Constraint enforcement differs: drift {max_constraint_drift:.2e}"

    print(f"  Status: PASS ✓")
