#!/usr/bin/env python3
"""Phase 1 diagnostic runner: Multi-axis ablation with ensemble parallelization.

Runs comprehensive diagnostics for Langevin thermostat parity root-cause isolation:
- Axis 1: RATTLE iteration sweep (n_iters ∈ [1, 3, 5, 10, 20, 50])
- Axis 2: Float32 vs Float64 precision
- Axis 3: Projection site comparison (post_o vs post_settle_vel vs both)

Uses vmap parallelization for 10+ trajectory replicas per configuration.
Outputs: CSV tables + matplotlib plots for hypothesis support scoring.

Usage:
    python scripts/phase1_diagnostic_runner.py --n-replicas 10 --output ./phase1_results
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import space, quantity as jax_md_quantity

# Import from prolix
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prolix.physics import settle
from prolix.types import WaterIndicesArray

jax.config.update("jax_enable_x64", True)

# Constants
BOLTZMANN_KCAL = 0.0019872041
AKMA_TIME_UNIT_FS = 48.88821291839


@dataclasses.dataclass
class AblationConfig:
    """Configuration for one ablation run."""
    dt_akma: float
    n_iters: int | None = None
    precision: str = "float64"  # float32 or float64
    projection_site: str = "post_o"
    n_replicas: int = 10
    n_steps: int = 100
    n_waters: int = 200
    T_target: float = 300.0


@dataclasses.dataclass
class TrajectoryMetrics:
    """Metrics from a single replica trajectory."""
    T_inst: np.ndarray  # (n_steps,)
    KE_rigid: np.ndarray
    KE_total: np.ndarray
    constraint_residual_norm: np.ndarray
    positions: np.ndarray | None = None
    momentum: np.ndarray | None = None

    def statistics(self) -> dict:
        """Compute summary statistics."""
        return {
            'T_mean': float(np.mean(self.T_inst)),
            'T_std': float(np.std(self.T_inst)),
            'T_min': float(np.min(self.T_inst)),
            'T_max': float(np.max(self.T_inst)),
            'KE_rigid_mean': float(np.mean(self.KE_rigid)),
            'KE_rigid_std': float(np.std(self.KE_rigid)),
            'residual_mean': float(np.mean(self.constraint_residual_norm)),
            'residual_final': float(self.constraint_residual_norm[-1]),
        }


def make_tip3p_water_box(
    n_waters: int = 200,
    target_density: float = 1.0,
    rng_key=None,
    dtype=jnp.float64,
) -> tuple[jax.Array, jax.Array, jax.Array, WaterIndicesArray, jax.Array]:
    """Create TIP3P water box."""
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    mass_O = 15.999
    mass_H = 1.008
    mass_water = mass_O + 2 * mass_H

    total_mass_amu = n_waters * mass_water
    V_angstrom3 = total_mass_amu / (target_density * 0.602214076)
    L = V_angstrom3 ** (1/3)
    box = jnp.array([L, L, L], dtype=dtype)

    key, split = jax.random.split(rng_key)
    positions = jax.random.uniform(split, (n_waters * 3, 3), minval=0.0, maxval=L)
    positions = positions.astype(dtype)

    masses = jnp.tile(jnp.array([mass_O, mass_H, mass_H], dtype=dtype), n_waters)

    water_indices = jnp.array(
        [[i*3, i*3+1, i*3+2] for i in range(n_waters)],
        dtype=jnp.int32
    )

    key, split = jax.random.split(key)
    T_target = 300.0
    kT = BOLTZMANN_KCAL * T_target
    velocities = jnp.sqrt(masses[:, None] * kT) * jax.random.normal(split, (n_waters*3, 3), dtype=dtype)

    return positions, velocities, masses, water_indices, box


def compute_instantaneous_temperature(
    momentum: jax.Array,
    mass: jax.Array,
    water_indices: WaterIndicesArray,
) -> float:
    """Compute instantaneous temperature (rigid water DOF)."""
    m = mass[:, None]
    m_total = jnp.sum(mass)
    p_com = jnp.sum(momentum, axis=0)
    p = momentum - (mass[:, None] / m_total) * p_com

    v = p / m
    KE = 0.5 * jnp.sum(p * v)

    N_waters = water_indices.shape[0]
    N_dof_rigid = 6 * N_waters - 3

    T = 2 * KE / (N_dof_rigid * BOLTZMANN_KCAL)
    return float(T)


def compute_rigid_kinetic_energy(
    momentum: jax.Array,
    position: jax.Array,
    mass: jax.Array,
    water_indices: WaterIndicesArray,
) -> float:
    """Compute KE in rigid-body subspace."""
    if water_indices.shape[0] == 0:
        return 0.0

    p_rigid = settle.project_tip3p_waters_momentum_rigid(
        momentum, position, mass, water_indices
    )
    v_rigid = p_rigid / mass[:, None]
    KE_rigid = 0.5 * jnp.sum(p_rigid * v_rigid)
    return float(KE_rigid)


def compute_settle_residual_norm(
    positions: jax.Array,
    water_indices: WaterIndicesArray,
    r_OH: float = settle.TIP3P_ROH,
    r_HH: float = settle.TIP3P_RHH,
) -> float:
    """Compute SETTLE constraint violation norm."""
    if water_indices.shape[0] == 0:
        return 0.0

    indices = settle.WaterIndices.from_row(water_indices.T)
    r_O = positions[indices.oxygen]
    r_H1 = positions[indices.hydrogen1]
    r_H2 = positions[indices.hydrogen2]

    d_OH1 = jnp.linalg.norm(r_H1 - r_O, axis=-1)
    d_OH2 = jnp.linalg.norm(r_H2 - r_O, axis=-1)
    d_H1H2 = jnp.linalg.norm(r_H2 - r_H1, axis=-1)

    err_OH1 = d_OH1 - r_OH
    err_OH2 = d_OH2 - r_OH
    err_H1H2 = d_H1H2 - r_HH

    residual = jnp.sqrt(
        jnp.sum(err_OH1**2) + jnp.sum(err_OH2**2) + jnp.sum(err_H1H2**2)
    )
    return float(residual)


def run_trajectory_replica(
    init_fn: Any,
    apply_fn: Any,
    positions: jax.Array,
    velocities: jax.Array,
    mass: jax.Array,
    n_steps: int,
    dt_akma: float,
    water_indices: WaterIndicesArray,
    key=None,
) -> TrajectoryMetrics:
    """Run single replica and collect per-step diagnostics."""
    if key is None:
        key = jax.random.PRNGKey(0)

    kT = BOLTZMANN_KCAL * 300.0
    state = init_fn(key, positions, mass=mass, kT=kT)

    T_list = []
    KE_rigid_list = []
    KE_total_list = []
    residual_list = []

    for step in range(n_steps):
        T = compute_instantaneous_temperature(state.momentum, mass, water_indices)
        KE_rigid = compute_rigid_kinetic_energy(state.momentum, state.position, mass, water_indices)
        KE_total = float(jax_md_quantity.kinetic_energy(momentum=state.momentum, mass=mass[:, None]))
        residual = compute_settle_residual_norm(state.position, water_indices)

        T_list.append(T)
        KE_rigid_list.append(KE_rigid)
        KE_total_list.append(KE_total)
        residual_list.append(residual)

        state = apply_fn(state)

    return TrajectoryMetrics(
        T_inst=np.array(T_list),
        KE_rigid=np.array(KE_rigid_list),
        KE_total=np.array(KE_total_list),
        constraint_residual_norm=np.array(residual_list),
    )


def run_ablation_configuration(
    config: AblationConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Run one complete ablation configuration (n_replicas trajectories)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Config: dt={config.dt_akma} AKMA, n_iters={config.n_iters}, "
          f"precision={config.precision}, proj={config.projection_site}")
    print(f"{'='*70}")

    # Create test system
    dtype = jnp.float32 if config.precision == 'float32' else jnp.float64
    positions, velocities, mass, water_indices, box = make_tip3p_water_box(
        config.n_waters, dtype=dtype
    )

    # SETTLE position correction
    positions = settle.settle_positions(
        positions, positions, water_indices, box=box
    )

    # Create integrator
    force_fn = lambda r, **kw: jnp.zeros_like(r)
    _, shift_fn = space.periodic(box)

    kwargs = {
        'gamma': 1.0,
        'mass': mass,
        'water_indices': water_indices,
        'box': box,
        'project_ou_momentum_rigid': True,
        'projection_site': config.projection_site,
    }
    if config.n_iters is not None:
        kwargs['settle_velocity_iters'] = config.n_iters

    init_fn, apply_fn = settle.settle_langevin(
        force_fn, shift_fn, config.dt_akma,
        kT=BOLTZMANN_KCAL * config.T_target,
        **kwargs
    )

    # Run ensemble via vmap
    def make_replica_initial_state(seed: int):
        key = jax.random.PRNGKey(seed)
        key, split = jax.random.split(key)
        pert = jax.random.normal(split, positions.shape) * 0.001
        return positions + pert, velocities

    replica_positions = jnp.stack([
        make_replica_initial_state(i)[0] for i in range(config.n_replicas)
    ])
    replica_velocities = jnp.stack([
        make_replica_initial_state(i)[1] for i in range(config.n_replicas)
    ])

    def run_one_replica(pos_init, vel_init):
        return run_trajectory_replica(
            init_fn, apply_fn,
            pos_init, vel_init, mass,
            config.n_steps, config.dt_akma, water_indices,
        )

    print(f"Running {config.n_replicas} replicas (serial)...")
    start = time.time()

    # Run replicas serially
    T_ensemble = []
    KE_rigid_ensemble = []
    residual_ensemble = []

    for i in range(config.n_replicas):
        metrics = run_one_replica(replica_positions[i], replica_velocities[i])
        T_ensemble.append(metrics.T_inst)
        KE_rigid_ensemble.append(metrics.KE_rigid)
        residual_ensemble.append(metrics.constraint_residual_norm)

    elapsed = time.time() - start
    print(f"Completed in {elapsed:.1f} s")

    # Compute ensemble statistics
    T_ensemble = np.array(T_ensemble)  # (n_replicas, n_steps)
    T_mean_per_replica = np.mean(T_ensemble, axis=1)
    T_overall_mean = np.mean(T_mean_per_replica)
    T_overall_std = np.std(T_mean_per_replica)

    print(f"\nResults:")
    print(f"  T (mean ± std over replicas): {T_overall_mean:.1f} ± {T_overall_std:.1f} K")
    print(f"  T (min-max over replica means): {np.min(T_mean_per_replica):.1f} - {np.max(T_mean_per_replica):.1f} K")

    # Save per-replica data
    config_label = f"dt{config.dt_akma}_n{config.n_iters}_p{config.precision}_proj{config.projection_site}"
    csv_path = output_dir / f"{config_label}_ensemble.csv"

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['replica', 'T_mean', 'T_std', 'T_min', 'T_max'])
        for i, T_traj in enumerate(T_ensemble):
            writer.writerow([
                i,
                np.mean(T_traj),
                np.std(T_traj),
                np.min(T_traj),
                np.max(T_traj),
            ])

    print(f"  Saved: {csv_path}")

    result = {
        'config': config,
        'T_ensemble': T_ensemble,
        'KE_rigid_ensemble': np.array(KE_rigid_ensemble),
        'residual_ensemble': np.array(residual_ensemble),
        'T_mean': T_overall_mean,
        'T_std': T_overall_std,
        'csv_path': str(csv_path),
    }

    return result


def run_full_phase1_diagnostic(
    output_dir: Path = Path("./phase1_results"),
    n_replicas: int = 10,
) -> None:
    """Run comprehensive Phase 1 diagnostic suite."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("PHASE 1: ROOT-CAUSE ISOLATION - COMPREHENSIVE ABLATION")
    print("="*70)

    results_summary = []

    # AXIS 1: RATTLE iteration sweep
    print("\n" + "-"*70)
    print("AXIS 1: RATTLE ITERATION SWEEP")
    print("-"*70)

    for dt_akma in [0.5, 1.0, 2.0]:
        for n_iters in [1, 3, 5, 10, 20, 50]:
            config = AblationConfig(
                dt_akma=dt_akma,
                n_iters=n_iters,
                precision='float64',
                projection_site='post_o',
                n_replicas=n_replicas,
                n_steps=100,
            )

            result = run_ablation_configuration(config, output_dir / "axis1_iterations")
            results_summary.append({
                'axis': 'iterations',
                'dt_akma': dt_akma,
                'n_iters': n_iters,
                'T_mean': result['T_mean'],
                'T_std': result['T_std'],
            })

    # AXIS 2: Float32 vs Float64
    print("\n" + "-"*70)
    print("AXIS 2: FLOAT PRECISION COMPARISON")
    print("-"*70)

    for dt_akma in [0.5, 1.0, 2.0]:
        for precision in ['float32', 'float64']:
            config = AblationConfig(
                dt_akma=dt_akma,
                n_iters=10,
                precision=precision,
                projection_site='post_o',
                n_replicas=n_replicas,
                n_steps=100,
                n_waters=100,  # Smaller for precision test
            )

            result = run_ablation_configuration(config, output_dir / "axis2_precision")
            results_summary.append({
                'axis': 'precision',
                'dt_akma': dt_akma,
                'precision': precision,
                'T_mean': result['T_mean'],
                'T_std': result['T_std'],
            })

    # AXIS 3: Projection site
    print("\n" + "-"*70)
    print("AXIS 3: PROJECTION SITE COMPARISON")
    print("-"*70)

    for dt_akma in [0.5, 1.0, 2.0]:
        for proj_site in ['post_o', 'post_settle_vel', 'both']:
            config = AblationConfig(
                dt_akma=dt_akma,
                n_iters=10,
                precision='float64',
                projection_site=proj_site,
                n_replicas=n_replicas,
                n_steps=100,
            )

            result = run_ablation_configuration(config, output_dir / "axis3_projection")
            results_summary.append({
                'axis': 'projection',
                'dt_akma': dt_akma,
                'projection_site': proj_site,
                'T_mean': result['T_mean'],
                'T_std': result['T_std'],
            })

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Summary saved to {summary_path}")
    print("="*70)

    # Print final summary table
    print("\nQUICK REFERENCE: Temperature by Configuration")
    print("-"*70)
    for r in results_summary:
        if r['axis'] == 'iterations':
            print(f"dt={r['dt_akma']:.1f}, n_iters={r['n_iters']:2d}: T={r['T_mean']:6.1f}±{r['T_std']:5.1f} K")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 1 diagnostic runner')
    parser.add_argument('--n-replicas', type=int, default=5, help='Number of trajectory replicas')
    parser.add_argument('--output', type=Path, default=Path('./phase1_results'), help='Output directory')
    args = parser.parse_args()

    run_full_phase1_diagnostic(output_dir=args.output, n_replicas=args.n_replicas)
