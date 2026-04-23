#!/usr/bin/env python3
"""P2: Validate Temperature Fix with Corrected DOF Count.

Objective: Determine if adaptive RATTLE (settle_velocity_tol) fixes the
inverse-timestep temperature error observed in Phase 1.

Runs two experiments:
  - dt=1.0 fs for 500 steps with SETTLE+Langevin
  - dt=2.0 fs for 500 steps with SETTLE+Langevin

Measures temperature using CORRECTED DOF count:
  DOF = 3*N_total - 3*N_waters - 3
  T_measured = 2 * KE / (k_B * DOF)

Expected results (prior baseline):
  - dt=1.0 fs → ΔT=28.8 K (BAD)
  - dt=2.0 fs → ΔT=10.6 K (BAD)

G4 gate threshold: ΔT < 5 K for both timesteps
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Add prolix to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prolix.physics import pbc, settle, system
from prolix.physics.rigid_water_ke import rigid_tip3p_box_ke_kcal
from prolix.physics.water_models import WaterModelType, get_water_params
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL

jax.config.update("jax_enable_x64", True)


def _tip3p_local_frame() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get TIP3P water geometry in local frame."""
    tip = get_water_params(WaterModelType.TIP3P)
    r = float(tip.r_OH)
    theta = 104.52 * math.pi / 180.0
    o = np.zeros(3, dtype=np.float64)
    h1 = np.array([r, 0.0, 0.0])
    h2 = np.array([r * math.cos(theta), r * math.sin(theta), 0.0])
    return o, h1, h2


def _grid_water_positions(
    n_waters: int, spacing_angstrom: float
) -> tuple[np.ndarray, float]:
    """Generate TIP3P water grid positions."""
    o0, h1l, h2l = _tip3p_local_frame()
    sites: list[tuple[int, int, int]] = []
    n = int(math.ceil(n_waters ** (1.0 / 3.0))) + 3
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                sites.append((ix, iy, iz))
                if len(sites) >= n_waters:
                    break
            if len(sites) >= n_waters:
                break
        if len(sites) >= n_waters:
            break
    sites = sites[:n_waters]

    base = np.array([3.0, 3.0, 3.0], dtype=np.float64)
    pos: list[np.ndarray] = []
    for ix, iy, iz in sites:
        o = base + np.array(
            [ix * spacing_angstrom, iy * spacing_angstrom, iz * spacing_angstrom],
            dtype=np.float64,
        )
        pos.append(o + o0)
        pos.append(o + h1l)
        pos.append(o + h2l)
    arr = np.vstack(pos)
    span = np.max(arr, axis=0) - np.min(arr, axis=0)
    box_edge = float(np.max(span) + 16.0)
    return arr, box_edge


def _prolix_params_pure_water(n_waters: int) -> dict:
    """Create Prolix system parameters for pure TIP3P water."""
    tip = get_water_params(WaterModelType.TIP3P)
    qo, qh = float(tip.charge_O), float(tip.charge_H)
    sig_o = float(tip.sigma_O)
    eps_o = float(tip.epsilon_O)
    n = n_waters * 3
    charges: list[float] = []
    sigmas: list[float] = []
    epsilons: list[float] = []
    for _ in range(n_waters):
        charges.extend([qo, qh, qh])
        sigmas.extend([sig_o, 1.0, 1.0])
        epsilons.extend([eps_o, 0.0, 0.0])
    mask = jnp.ones((n, n), dtype=jnp.float64) - jnp.eye(n, dtype=jnp.float64)
    for w in range(n_waters):
        b = w * 3
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            a, c = b + i, b + j
            mask = mask.at[a, c].set(0.0).at[c, a].set(0.0)
    return {
        "charges": jnp.array(charges, dtype=jnp.float64),
        "sigmas": jnp.array(sigmas, dtype=jnp.float64),
        "epsilons": jnp.array(epsilons, dtype=jnp.float64),
        "bonds": jnp.zeros((0, 2), dtype=jnp.int32),
        "bond_params": jnp.zeros((0, 2), dtype=jnp.float64),
        "angles": jnp.zeros((0, 3), dtype=jnp.int32),
        "angle_params": jnp.zeros((0, 2), dtype=jnp.float64),
        "dihedrals": jnp.zeros((0, 4), dtype=jnp.int32),
        "dihedral_params": jnp.zeros((0, 3), dtype=jnp.float64),
        "impropers": jnp.zeros((0, 4), dtype=jnp.int32),
        "improper_params": jnp.zeros((0, 3), dtype=jnp.float64),
        "exclusion_mask": mask,
    }


def run_temperature_validation_experiment(
    dt_fs: float,
    n_steps: int = 500,
    n_waters: int = 64,
    T_target: float = 300.0,
) -> dict:
    """Run one temperature validation experiment.

    Args:
        dt_fs: Timestep in femtoseconds
        n_steps: Number of integration steps
        n_waters: Number of water molecules
        T_target: Target temperature (K)

    Returns:
        Dictionary with results:
        - dt_fs: timestep
        - T_measured: measured temperature
        - T_error: |T_measured - T_target|
        - KE_mean: mean kinetic energy
        - KE_std: std of kinetic energy
    """
    print(f"\n=== Experiment: dt={dt_fs} fs, n_steps={n_steps} ===")

    # Create system
    positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
    box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)

    print(f"System: {n_waters} waters ({3*n_waters} atoms)")
    print(f"Box: {box_edge:.2f} Å")

    # Convert dt to AKMA units
    dt_akma = dt_fs / AKMA_TIME_UNIT_FS
    print(f"Timestep: {dt_fs} fs = {dt_akma:.6f} AKMA")

    # Setup energy function and integrator
    sys_dict = _prolix_params_pure_water(n_waters)
    displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)

    # Minimal PME parameters (not critical for temperature validation)
    pme_alpha = 0.34
    pme_grid = 32
    cutoff = 12.0

    energy_fn = system.make_energy_fn(
        displacement_fn,
        sys_dict,
        box=box_vec,
        use_pbc=True,
        implicit_solvent=False,
        pme_grid_points=pme_grid,
        pme_alpha=pme_alpha,
        cutoff_distance=cutoff,
        strict_parameterization=False,
    )

    # Physical parameters
    kT = T_target * BOLTZMANN_KCAL
    gamma_reduced = 1.0 * AKMA_TIME_UNIT_FS * 1e-3  # friction: 1 ps^-1
    n_atoms = n_waters * 3
    mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_atoms)
    water_indices = settle.get_water_indices(0, n_waters)
    dof = float(6 * n_waters - 3)

    print(f"DOF (rigid water): {dof}")
    print(f"kT: {kT:.6f} kcal/mol")

    # Create SETTLE+Langevin integrator
    init_s, apply_s = settle.settle_langevin(
        energy_fn,
        shift_fn,
        dt=dt_akma,
        kT=kT,
        gamma=gamma_reduced,
        mass=mass,
        water_indices=water_indices,
        box=box_vec,
    )

    # Initialize state
    state = init_s(jax.random.PRNGKey(42), jnp.array(positions_a), mass=mass)

    # Run trajectory
    print(f"Running {n_steps} steps...")
    ts: list[float] = []
    for step in range(n_steps):
        state = apply_s(state)
        assert jnp.all(jnp.isfinite(state.position)), f"NaN/Inf at step {step}"

        # Measure temperature
        ke = float(rigid_tip3p_box_ke_kcal(state.position, state.momentum, state.mass, n_waters))
        T_inst = 2.0 * ke / (dof * BOLTZMANN_KCAL)
        ts.append(T_inst)

    # Analyze last 300 steps (after equilibration)
    equilibration_steps = min(300, n_steps // 2)
    arr = np.array(ts[-equilibration_steps:], dtype=np.float64)

    T_mean = float(np.mean(arr))
    T_std = float(np.std(arr))
    T_error = abs(T_mean - T_target)

    # Kinetic energies for reference
    ke_array = np.array([rigid_tip3p_box_ke_kcal(state.position, state.momentum, state.mass, n_waters)
                         for _ in range(1)])  # Just final state for simplicity
    KE_mean = float(np.mean(ke_array)) if len(ke_array) > 0 else 0.0
    KE_std = float(np.std(ke_array)) if len(ke_array) > 0 else 0.0

    print(f"Results (last {equilibration_steps} steps):")
    print(f"  T_measured: {T_mean:.2f} ± {T_std:.2f} K")
    print(f"  T_target:  {T_target:.2f} K")
    print(f"  ΔT = |T_meas - T_target|: {T_error:.2f} K")

    return {
        "dt_fs": dt_fs,
        "T_measured": T_mean,
        "T_std": T_std,
        "T_error": T_error,
        "KE_mean": KE_mean,
        "KE_std": KE_std,
        "n_steps": n_steps,
        "n_waters": n_waters,
        "DOF": int(dof),
    }


def main():
    """Run full temperature validation suite."""
    print("=" * 70)
    print("P2: Validate Temperature Fix with Corrected DOF Count")
    print("=" * 70)
    print()
    print("Configuration:")
    print("  - System: 64 waters (192 atoms)")
    print("  - T_target: 300 K")
    print("  - settle_velocity_tol: 1e-6")
    print("  - Duration: 500 steps per experiment")
    print()
    print("DOF formula (corrected):")
    print("  DOF = 3*N_total - 3*N_waters - 3")
    print("      = 3*192 - 3*64 - 3 = 576 - 192 - 3 = 381")
    print()

    # Run experiments
    results = []
    for dt in [1.0, 2.0]:
        result = run_temperature_validation_experiment(dt_fs=dt, n_steps=500, n_waters=64)
        results.append(result)

    # Summary and gate status
    print("\n" + "=" * 70)
    print("SUMMARY AND G4 GATE STATUS")
    print("=" * 70)
    print()

    for result in results:
        dt = result["dt_fs"]
        dT = result["T_error"]
        status = "PASS" if dT < 5.0 else "FAIL"
        print(f"dt = {dt} fs: ΔT = {dT:.2f} K [{status}]")

    print()
    all_pass = all(r["T_error"] < 5.0 for r in results)
    gate_status = "PASS" if all_pass else "FAIL"
    print(f"G4 Gate Status: {gate_status}")
    print()

    if all_pass:
        print("INTERPRETATION:")
        print("  ✓ Root cause CONFIRMED: iteration count was the issue.")
        print("  ✓ Adaptive RATTLE with settle_velocity_tol=1e-6 FIXES the problem.")
        print("  ✓ Temperature now independent of timestep.")
        print()
        print("RECOMMENDATION:")
        print("  - Close G4 gate (temperature validation complete)")
        print("  - Proceed to Phase 2 implementation with confidence")
    else:
        print("INTERPRETATION:")
        print("  ✗ Root cause is NOT iteration count alone.")
        print("  ✗ Temperature error persists at ΔT >= 5 K.")
        print()
        print("ESCALATION:")
        print("  - Oracle needs to investigate:")
        print("    1. DOF denominator correctness")
        print("    2. Ornstein-Uhlenbeck (OU) noise projection")
        print("    3. Langevin thermostat implementation details")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
