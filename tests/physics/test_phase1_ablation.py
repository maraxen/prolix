"""Phase 1 root-cause isolation for Langevin thermostat parity.

This test suite implements comprehensive ablation across three axes:
1. RATTLE iteration count (n_iters sweep)
2. Float32 vs Float64 precision
3. Projection site (post_o vs post_settle_vel vs both)

Target: Quantify hypothesis support for ΔT error (28.8 K @ dt=1.0 fs, 10.6 K @ dt=2.0 fs).
Goal: Identify root cause(s) blocking G4 bound (< 5 K acceptable ΔT).

Design:
- Use TIP3P water-only system (200 waters, ~45 Å periodic cell)
- Run 10-100 ps trajectories at dt ∈ {0.5, 1.0, 2.0} fs
- Measure per-step: T_inst, KE_rigid, constraint_residual_norm
- Support multiple replicas via vmap for ensemble statistics
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_md import partition, space, quantity as jax_md_quantity

from prolix.physics import settle
from prolix.types import WaterIndicesArray

jax.config.update("jax_enable_x64", True)

# Constants
BOLTZMANN_KCAL = 0.0019872041  # kB in kcal/(mol·K)
AKMA_TIME_UNIT_FS = 48.88821291839  # 1 AKMA time unit in fs


@dataclasses.dataclass
class DiagnosticMetrics:
    """Per-step diagnostic metrics for a single trajectory replica."""
    T_inst: jax.Array  # (n_steps,) instantaneous temperature
    KE_rigid: jax.Array  # (n_steps,) kinetic energy of rigid subsystem
    KE_total: jax.Array  # (n_steps,) total kinetic energy
    constraint_residual_norm: jax.Array  # (n_steps,) SETTLE position error
    positions: jax.Array  # (n_steps, N, 3) for debugging
    momentum: jax.Array  # (n_steps, N, 3) for debugging

    def to_dict(self):
        """Convert to dict for saving."""
        return {
            'T_inst': np.array(self.T_inst),
            'KE_rigid': np.array(self.KE_rigid),
            'KE_total': np.array(self.KE_total),
            'constraint_residual_norm': np.array(self.constraint_residual_norm),
        }


def make_tip3p_water_box(
    n_waters: int = 200,
    target_density: float = 1.0,  # g/cm^3, TIP3P at 298K ~ 1.0
    rng_key=None,
) -> tuple[jax.Array, jax.Array, jax.Array, WaterIndicesArray]:
    """Create TIP3P water box with random initial velocities.

    Args:
        n_waters: Number of water molecules.
        target_density: Target density in g/cm^3.
        rng_key: JAX random key for velocity initialization.

    Returns:
        (positions, velocities, mass, water_indices)
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    # TIP3P masses (amu)
    mass_O = 15.999
    mass_H = 1.008
    mass_water = mass_O + 2 * mass_H

    # Estimate box size from target density
    # density = (n_waters * mass_water) / (V * N_A)
    # V (Å^3) = (n_waters * mass_water) / (density * 1.66054e-24)
    # Note: 1 amu = 1.66054e-27 kg, 1 Å^3 = 1e-30 m^3
    # For density in g/cm^3 and mass in amu:
    # V_angstrom3 = (n_waters * mass_water) / (density * 0.602214076)
    total_mass_amu = n_waters * mass_water
    V_angstrom3 = total_mass_amu / (target_density * 0.602214076)
    L = V_angstrom3 ** (1/3)
    box = jnp.array([L, L, L])

    # Generate initial positions: random distribution in box
    key, split = jax.random.split(rng_key)
    positions = jax.random.uniform(split, (n_waters * 3, 3), minval=0.0, maxval=L)

    # Initialize as (O, H1, H2) triplets with TIP3P geometry
    # For now, use random geometry (will be corrected by SETTLE in first step)
    positions = jnp.mod(positions, L)

    # Masses: (O, H, H) repeated for each water
    masses = jnp.tile(jnp.array([mass_O, mass_H, mass_H]), n_waters)

    # Water indices: [(0, 1, 2), (3, 4, 5), ...]
    water_indices = jnp.array(
        [[i*3, i*3+1, i*3+2] for i in range(n_waters)],
        dtype=jnp.int32
    )

    # Initialize velocities: Boltzmann distribution at T=300K
    key, split = jax.random.split(key)
    T_target = 300.0  # K
    kT = BOLTZMANN_KCAL * T_target
    velocities = jnp.sqrt(masses[:, None] * kT) * jax.random.normal(split, (n_waters*3, 3))

    return positions, velocities, masses, water_indices, box


def compute_instantaneous_temperature(
    momentum: jax.Array,
    mass: jax.Array,
    water_indices: WaterIndicesArray,
    exclude_com: bool = True,
) -> float:
    """Compute instantaneous temperature from kinetic energy.

    For rigid water, only counts 6N_w - 3 degrees of freedom (3 translational + 3 rotational per water).

    Args:
        momentum: (N_atoms, 3) momentum array
        mass: (N_atoms,) mass array
        water_indices: (N_waters, 3) water atom indices
        exclude_com: If True, subtract COM momentum before computing T

    Returns:
        Instantaneous temperature in K
    """
    p = momentum
    m = mass[:, None]  # (N, 1)

    # Subtract COM momentum if requested
    if exclude_com:
        m_total = jnp.sum(mass)
        p_com = jnp.sum(p, axis=0)
        v_com = p_com / m_total
        p = p - mass[:, None] * v_com

    # Kinetic energy: 0.5 * sum(p_i^2 / m_i)
    v = p / m
    KE = 0.5 * jnp.sum(p * v)

    # For rigid water: N_dof = 6 * N_waters - 3 (remove COM translation)
    # But measure from full momentum degrees of freedom
    N_atoms = momentum.shape[0]
    N_waters = water_indices.shape[0]
    N_dof = 3 * N_atoms - 3  # Remove COM translation for full system

    # For RIGID water interpretation: 6*N_w - 3 DOF
    # (3 translational per water + 3 rotational per water - 3 global COM)
    N_dof_rigid = 6 * N_waters - 3

    # T = 2 * KE / (N_dof * k_B)
    T = 2 * KE / (N_dof_rigid * BOLTZMANN_KCAL)
    return T


def compute_rigid_kinetic_energy(
    momentum: jax.Array,
    position: jax.Array,
    mass: jax.Array,
    water_indices: WaterIndicesArray,
) -> float:
    """Compute kinetic energy in rigid-body subspace.

    Project atomic momenta onto rigid-body subspace (COM + rotation)
    for each water, then sum kinetic energies.

    Args:
        momentum: (N_atoms, 3)
        position: (N_atoms, 3)
        mass: (N_atoms,)
        water_indices: (N_waters, 3)

    Returns:
        KE_rigid (scalar)
    """
    if water_indices.shape[0] == 0:
        return 0.0

    # Project momenta onto rigid subspace for each water
    p_rigid = settle.project_tip3p_waters_momentum_rigid(
        momentum, position, mass, water_indices
    )

    # Compute KE from rigid momenta
    v_rigid = p_rigid / mass[:, None]
    KE_rigid = 0.5 * jnp.sum(p_rigid * v_rigid)
    return KE_rigid


def compute_settle_residual_norm(
    positions: jax.Array,
    water_indices: WaterIndicesArray,
    r_OH: float = settle.TIP3P_ROH,
    r_HH: float = settle.TIP3P_RHH,
) -> float:
    """Compute L2 norm of SETTLE constraint violations (bond length errors).

    Args:
        positions: (N_atoms, 3)
        water_indices: (N_waters, 3)
        r_OH, r_HH: Target bond lengths

    Returns:
        Residual norm (scalar)
    """
    if water_indices.shape[0] == 0:
        return 0.0

    indices = settle.WaterIndices.from_row(water_indices.T)

    # Extract positions
    r_O = positions[indices.oxygen]
    r_H1 = positions[indices.hydrogen1]
    r_H2 = positions[indices.hydrogen2]

    # Compute bond lengths
    d_OH1 = jnp.linalg.norm(r_H1 - r_O, axis=-1)
    d_OH2 = jnp.linalg.norm(r_H2 - r_O, axis=-1)
    d_H1H2 = jnp.linalg.norm(r_H2 - r_H1, axis=-1)

    # Residuals (should be zero)
    err_OH1 = d_OH1 - r_OH
    err_OH2 = d_OH2 - r_OH
    err_H1H2 = d_H1H2 - r_HH

    # L2 norm across all waters and bond types
    residual = jnp.sqrt(
        jnp.sum(err_OH1**2) + jnp.sum(err_OH2**2) + jnp.sum(err_H1H2**2)
    )
    return residual


def run_settle_langevin_trajectory(
    init_fn: Any,
    apply_fn: Any,
    positions: jax.Array,
    velocities: jax.Array,
    mass: jax.Array,
    n_steps: int,
    dt: float,
    water_indices: WaterIndicesArray,
    T_target: float = 300.0,
    key=None,
) -> DiagnosticMetrics:
    """Run single trajectory replica and collect diagnostics.

    Args:
        init_fn, apply_fn: SETTLE Langevin integrator pair
        positions, velocities, mass: Initial conditions
        n_steps: Number of steps
        dt: Timestep (AKMA units)
        water_indices: Water molecule indices
        T_target: Target temperature
        key: JAX random key

    Returns:
        DiagnosticMetrics with per-step arrays
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    # Initialize state
    kT = BOLTZMANN_KCAL * T_target
    momentum = velocities * mass[:, None]  # p = m * v
    state = init_fn(key, positions, mass=mass, kT=kT)

    # Pre-allocate diagnostic arrays
    T_inst_list = []
    KE_rigid_list = []
    KE_total_list = []
    residual_list = []
    pos_list = []
    mom_list = []

    # Run trajectory
    for step in range(n_steps):
        # Compute diagnostics before step
        T = compute_instantaneous_temperature(state.momentum, state.mass, water_indices)
        KE_rigid = compute_rigid_kinetic_energy(state.momentum, state.position, state.mass, water_indices)
        KE_total = jax_md_quantity.kinetic_energy(momentum=state.momentum, mass=state.mass)
        residual = compute_settle_residual_norm(state.position, water_indices)

        T_inst_list.append(T)
        KE_rigid_list.append(KE_rigid)
        KE_total_list.append(KE_total)
        residual_list.append(residual)
        pos_list.append(state.position)
        mom_list.append(state.momentum)

        # Apply integrator step
        state = apply_fn(state)

    # Stack into arrays
    metrics = DiagnosticMetrics(
        T_inst=jnp.array(T_inst_list),
        KE_rigid=jnp.array(KE_rigid_list),
        KE_total=jnp.array(KE_total_list),
        constraint_residual_norm=jnp.array(residual_list),
        positions=jnp.stack(pos_list),
        momentum=jnp.stack(mom_list),
    )

    return metrics


class TestPhase1AblationRATTLEIterations:
    """ABLATION AXIS 1: RATTLE iteration count sweep.

    Hypothesis: Fixed RATTLE saturation at low n_iters causes KE over-drive.

    Test: Sweep n_iters ∈ [1, 3, 5, 10, 20, 50] at each dt.
    Prediction (HIGH support):
    - Residual norm monotonically decreases with n_iters
    - Saturation visible ~n_iters=2-3 @ dt=1.0 fs
    - Saturation visible ~n_iters=8-10 @ dt=2.0 fs
    - ΔT decreases monotonically with n_iters
    """

    @pytest.mark.parametrize("dt_akma", [0.5, 1.0, 2.0])
    def test_rattle_convergence_vs_iterations(self, dt_akma):
        """Run trajectory sweep over n_iters for a single dt value."""
        # Create test system
        n_waters = 200
        n_steps = 100  # ~5 ps @ dt=1.0 fs equivalent

        positions, velocities, mass, water_indices, box = make_tip3p_water_box(n_waters)

        # SETTLE position correction on initial state
        positions = settle.settle_positions(
            positions, positions, water_indices, box=box
        )

        # Test multiple iteration counts
        n_iters_values = [1, 3, 5, 10, 20, 50]
        results = {}

        for n_iters in n_iters_values:
            # Create SETTLE Langevin integrator
            # Note: settle_langevin returns (init_fn, apply_fn)
            force_fn = lambda r, **kw: jnp.zeros_like(r)  # Dummy force (ideal gas)
            _, shift_fn = space.periodic(box)

            init_fn, apply_fn = settle.settle_langevin(
                force_fn,
                shift_fn,
                dt_akma,
                kT=BOLTZMANN_KCAL * 300.0,
                gamma=1.0,
                mass=mass,
                water_indices=water_indices,
                box=box,
                project_ou_momentum_rigid=True,
                projection_site="post_o",
                settle_velocity_iters=n_iters,
            )

            # Run trajectory
            metrics = run_settle_langevin_trajectory(
                init_fn, apply_fn,
                positions, velocities, mass,
                n_steps, dt_akma, water_indices
            )

            results[n_iters] = metrics

        # Extract summary statistics
        summary = {}
        for n_iters, metrics in results.items():
            T_mean = float(jnp.mean(metrics.T_inst))
            T_std = float(jnp.std(metrics.T_inst))
            residual_mean = float(jnp.mean(metrics.constraint_residual_norm))
            residual_final = float(metrics.constraint_residual_norm[-1])

            summary[n_iters] = {
                'T_mean': T_mean,
                'T_std': T_std,
                'residual_mean': residual_mean,
                'residual_final': residual_final,
            }

        # Basic assertion: residual should decrease with n_iters
        residuals = [summary[n]['residual_mean'] for n in n_iters_values]
        assert all(residuals[i] >= residuals[i+1] for i in range(len(residuals)-1)), \
            f"Residual should decrease with n_iters; got {residuals}"

        # Print summary for manual inspection
        print(f"\ndt={dt_akma} AKMA, n_iters sweep:")
        for n_iters in n_iters_values:
            s = summary[n_iters]
            print(f"  n_iters={n_iters:2d}: T={s['T_mean']:.1f}±{s['T_std']:.1f} K, "
                  f"residual={s['residual_final']:.2e}")

    def test_single_config(self):
        """Quick smoke test: single RATTLE iteration config."""
        n_waters, n_steps, dt_akma, n_iters = 10, 50, 1.0, 10

        positions, velocities, mass, water_indices, box = make_tip3p_water_box(n_waters)
        positions = settle.settle_positions(positions, positions, water_indices, box=box)

        force_fn = lambda r, **kw: jnp.zeros_like(r)
        _, shift_fn = space.periodic(box)

        init_fn, apply_fn = settle.settle_langevin(
            force_fn, shift_fn, dt_akma,
            kT=BOLTZMANN_KCAL * 300.0,
            gamma=1.0,
            mass=mass,
            water_indices=water_indices,
            box=box,
            project_ou_momentum_rigid=True,
            projection_site="post_o",
            settle_velocity_iters=n_iters,
        )

        metrics = run_settle_langevin_trajectory(
            init_fn, apply_fn, positions, velocities, mass,
            n_steps, dt_akma, water_indices
        )

        assert jnp.all(jnp.isfinite(metrics.positions)), "Position contains NaN"
        assert jnp.all(jnp.isfinite(metrics.momentum)), "Momentum contains NaN"
        assert jnp.all(jnp.isfinite(metrics.T_inst)), "Temperature contains NaN"
        assert jnp.all(metrics.KE_rigid >= 0), "Kinetic energy must be non-negative"
        assert float(metrics.T_inst[-1]) > 0, f"Temperature must be positive, got {metrics.T_inst[-1]}"


class TestPhase1AblationFloatPrecision:
    """ABLATION AXIS 2: Float32 vs Float64 precision.

    Hypothesis: Float32 inertia tensor conditioning causes KE divergence.

    Test: Run identical trajectory at float32 and float64.
    Prediction (MEDIUM support):
    - Condition number(inertia matrix, float32) > 1e8
    - Float32 shows ΔT > 20 K while float64 shows < 5 K
    - KE divergence visible by step ~50
    """

    def test_float32_vs_float64_precision(self):
        """Compare float32 and float64 kinetic energy tracking."""
        n_waters = 100  # Smaller system for precision comparison
        n_steps = 100
        dt_akma = 1.0

        positions, velocities, mass, water_indices, box = make_tip3p_water_box(n_waters)

        # Position correction
        positions = settle.settle_positions(
            positions, positions, water_indices, box=box
        )

        # Dummy force function
        force_fn = lambda r, **kw: jnp.zeros_like(r)
        _, shift_fn = space.periodic(box)

        results_by_dtype = {}

        for dtype_name, enable_x64 in [('float32', False), ('float64', True)]:
            jax.config.update('jax_enable_x64', enable_x64)

            # Create integrator
            pos = positions.astype(jnp.float32 if not enable_x64 else jnp.float64)
            vel = velocities.astype(jnp.float32 if not enable_x64 else jnp.float64)
            m = mass.astype(jnp.float32 if not enable_x64 else jnp.float64)

            init_fn, apply_fn = settle.settle_langevin(
                force_fn,
                shift_fn,
                dt_akma,
                kT=BOLTZMANN_KCAL * 300.0,
                gamma=1.0,
                mass=m,
                water_indices=water_indices,
                box=box.astype(jnp.float32 if not enable_x64 else jnp.float64),
                settle_velocity_iters=10,
            )

            # Run trajectory
            metrics = run_settle_langevin_trajectory(
                init_fn, apply_fn, pos, vel, m,
                n_steps, dt_akma, water_indices
            )

            results_by_dtype[dtype_name] = metrics

        # Re-enable x64
        jax.config.update('jax_enable_x64', True)

        # Compare results
        T32_mean = float(jnp.mean(results_by_dtype['float32'].T_inst))
        T64_mean = float(jnp.mean(results_by_dtype['float64'].T_inst))

        print(f"\nFloat precision comparison (dt={dt_akma} AKMA):")
        print(f"  Float32: T_mean = {T32_mean:.1f} K")
        print(f"  Float64: T_mean = {T64_mean:.1f} K")
        print(f"  Difference: {abs(T32_mean - T64_mean):.1f} K")

        # Assertion: float32 should show worse stability (placeholder)
        # Real test would check condition numbers and KE divergence


class TestPhase1AblationProjectionSite:
    """ABLATION AXIS 3: Momentum projection site.

    Hypothesis: Projection timing (post_o vs post_settle_vel vs both) biases KE.

    Test: Compare projection_site ∈ {'post_o', 'post_settle_vel', 'both'}.
    Prediction (MEDIUM support):
    - 'both' should be most stable (projects out OU noise and correction errors)
    - 'post_settle_vel' removes correction bias
    - 'post_o' (legacy) leaves settling errors in KE
    - ΔT difference between modes > 5 K suggests projection is critical
    """

    @pytest.mark.parametrize("dt_akma", [0.5, 1.0, 2.0])
    def test_projection_site_comparison(self, dt_akma):
        """Compare different projection sites at fixed dt."""
        n_waters = 200
        n_steps = 100

        positions, velocities, mass, water_indices, box = make_tip3p_water_box(n_waters)
        positions = settle.settle_positions(
            positions, positions, water_indices, box=box
        )

        force_fn = lambda r, **kw: jnp.zeros_like(r)
        _, shift_fn = space.periodic(box)

        projection_sites = ['post_o', 'post_settle_vel', 'both']
        results = {}

        for proj_site in projection_sites:
            init_fn, apply_fn = settle.settle_langevin(
                force_fn,
                shift_fn,
                dt_akma,
                kT=BOLTZMANN_KCAL * 300.0,
                gamma=1.0,
                mass=mass,
                water_indices=water_indices,
                box=box,
                project_ou_momentum_rigid=True,
                projection_site=proj_site,
                settle_velocity_iters=10,
            )

            metrics = run_settle_langevin_trajectory(
                init_fn, apply_fn,
                positions, velocities, mass,
                n_steps, dt_akma, water_indices
            )

            results[proj_site] = metrics

        # Summary
        print(f"\nProjection site comparison (dt={dt_akma} AKMA):")
        for proj_site in projection_sites:
            T_mean = float(jnp.mean(results[proj_site].T_inst))
            T_std = float(jnp.std(results[proj_site].T_inst))
            print(f"  {proj_site:20s}: T = {T_mean:.1f} ± {T_std:.1f} K")

        # Check for significant differences
        T_values = [float(jnp.mean(results[s].T_inst)) for s in projection_sites]
        max_diff = max(T_values) - min(T_values)
        print(f"  Max difference: {max_diff:.1f} K")


class TestPhase1VmapEnsemble:
    """Fast vmap-based ensemble runner for 10+ replicas.

    Purpose: Parallelize across replicas to get ensemble statistics efficiently.
    """

    def test_vmap_ensemble_runner(self):
        """Run ensemble of trajectories via vmap."""
        n_replicas = 5  # Use 5 for test, scale to 10+ in production
        n_waters = 100
        n_steps = 50
        dt_akma = 1.0

        positions, velocities, mass, water_indices, box = make_tip3p_water_box(n_waters)
        positions = settle.settle_positions(
            positions, positions, water_indices, box=box
        )

        # Create batch of initial conditions (different RNG seeds)
        def make_replica_state(seed):
            key = jax.random.PRNGKey(seed)
            key, split = jax.random.split(key)
            # Perturb positions slightly
            pert = jax.random.normal(split, positions.shape) * 0.01
            return positions + pert, velocities

        replica_positions = jnp.stack([make_replica_state(i)[0] for i in range(n_replicas)])
        replica_velocities = jnp.stack([make_replica_state(i)[1] for i in range(n_replicas)])

        # Dummy force
        force_fn = lambda r, **kw: jnp.zeros_like(r)
        _, shift_fn = space.periodic(box)

        # Create integrator
        init_fn, apply_fn = settle.settle_langevin(
            force_fn, shift_fn, dt_akma,
            kT=BOLTZMANN_KCAL * 300.0,
            gamma=1.0,
            mass=mass,
            water_indices=water_indices,
            box=box,
            settle_velocity_iters=10,
        )

        # VMAP single trajectory over replicas
        def run_one_replica(pos_init, vel_init):
            metrics = run_settle_langevin_trajectory(
                init_fn, apply_fn,
                pos_init, vel_init, mass,
                n_steps, dt_akma, water_indices
            )
            return metrics.T_inst, metrics.KE_rigid, metrics.constraint_residual_norm

        # Batch over replicas (should be much faster than serial)
        T_ensemble, KE_ensemble, residual_ensemble = jax.vmap(
            run_one_replica, in_axes=(0, 0)
        )(replica_positions, replica_velocities)

        # Print ensemble statistics
        print(f"\nEnsemble run ({n_replicas} replicas, {n_steps} steps, dt={dt_akma} AKMA):")
        T_mean_per_replica = jnp.mean(T_ensemble, axis=1)
        print(f"  T (replica means): {T_mean_per_replica}")
        print(f"  T (overall mean): {jnp.mean(T_mean_per_replica):.1f} K")
        print(f"  T (overall std):  {jnp.std(T_mean_per_replica):.1f} K")

        # Assertions
        assert T_ensemble.shape == (n_replicas, n_steps)
        assert all(jnp.isfinite(T_ensemble).all() for _ in [True])  # No NaNs


if __name__ == '__main__':
    # Run tests manually for debugging
    test1 = TestPhase1AblationRATTLEIterations()
    test1.test_rattle_convergence_vs_iterations(1.0)

    test3 = TestPhase1AblationProjectionSite()
    test3.test_projection_site_comparison(1.0)
