#!/usr/bin/env python3
"""
v0.3.0 Phase 2: Validate constraint-aware Langevin thermostat.

Tests that constraint-aware Langevin (noise projected to 6D rigid-body subspace)
achieves stable temperature at dt=1.0fs with SETTLE constraints.

Configuration: 4 TIP3P water molecules, 100ps simulation
Gates: T=300±5K, energy drift <1%, equipartition KS p>0.05
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats
import sys
from pathlib import Path

# Add prolix to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prolix.physics.settle import settle_langevin
from prolix.physics.water_models import get_water_params, WaterModelType
from prolix.physics.rigid_water_ke import rigid_tip3p_box_ke_kcal


def setup_water_system(n_waters: int = 4, seed: int = 42):
    """
    Setup N water molecules in a box.

    Returns:
        positions: (N*3, 3) array of atomic positions
        velocities: (N*3, 3) array of velocities (Maxwell-Boltzmann at 300K)
        masses: (N*3,) array of atomic masses
        water_indices: (N, 3) array of atom indices for each water
    """
    rng = np.random.RandomState(seed)

    # Get TIP3P parameters
    water_params = get_water_params(WaterModelType.TIP3P)

    # Create N waters with random positions in 12 Å box
    box_size = 12.0
    positions_list = []
    waters_indices = []
    atom_masses = []

    for i_water in range(n_waters):
        # Random COM position
        com = rng.uniform(-box_size/2, box_size/2, 3)

        # TIP3P geometry: O-H1-H2 triangle
        # Bond lengths: O-H = 0.9572 Å
        # Angle H-O-H = 104.52°
        roh = 0.9572
        angle = 104.52 * np.pi / 180.0

        # Place atoms around COM
        r_o = com
        r_h1 = com + roh * np.array([np.sin(angle/2), np.cos(angle/2), 0])
        r_h2 = com + roh * np.array([-np.sin(angle/2), np.cos(angle/2), 0])

        n_atoms_so_far = len(positions_list)
        positions_list.extend([r_o, r_h1, r_h2])
        waters_indices.append([n_atoms_so_far, n_atoms_so_far + 1, n_atoms_so_far + 2])

        # Masses: O=15.999, H=1.008
        atom_masses.extend([15.999, 1.008, 1.008])

    positions = np.array(positions_list, dtype=np.float32)
    masses = np.array(atom_masses, dtype=np.float32)
    water_indices = np.array(waters_indices, dtype=np.int32)

    # Maxwell-Boltzmann velocities at 300K
    kT = 0.6  # kcal/mol at 300K
    velocities = rng.normal(0, np.sqrt(kT / masses[:, None]), positions.shape).astype(np.float32)

    return positions, velocities, masses, water_indices


def run_validation_test(dt_fs: float, duration_ps: float = 100.0, n_waters: int = 4):
    """
    Run SETTLE+Langevin validation test.

    Args:
        dt_fs: Timestep in fs
        duration_ps: Total simulation time in ps
        n_waters: Number of water molecules

    Returns:
        results: Dict with T_mean, T_std, energy_drift, ks_pvalue
    """
    print(f"\n{'='*70}")
    print(f"Validation: dt={dt_fs:.1f}fs, duration={duration_ps:.0f}ps, {n_waters} waters")
    print(f"{'='*70}")

    # Setup system
    positions, velocities, masses, water_indices = setup_water_system(n_waters)

    # Prolix uses AKMA units; convert dt
    # 1 fs = 1/48.88821 AKMA units
    dt_akma = dt_fs / 48.88821
    kT = 0.6  # kcal/mol for 300K
    gamma = 1.0  # ps^-1

    # Number of timesteps
    n_steps = int(duration_ps * 1000.0 / dt_fs)  # 1ps = 1000fs
    print(f"Timesteps: {n_steps} (dt={dt_akma:.6f} AKMA)")

    # Create momentum from velocities
    momentum = velocities * masses[:, None]

    # Initialize SETTLE+Langevin integrator
    def mock_energy_fn(r):
        """Dummy energy function (no interactions for this test)."""
        return 0.0

    def mock_shift_fn(r, displacement):
        """Dummy shift function - applies displacement to positions."""
        return r + displacement

    try:
        init_fn, apply_fn = settle_langevin(
            energy_or_force_fn=mock_energy_fn,  # Correct parameter name
            shift_fn=mock_shift_fn,
            dt=dt_akma,
            kT=kT,
            gamma=gamma,
            mass=masses,
            water_indices=water_indices,
            project_ou_momentum_rigid=True,  # <-- CRITICAL: Enable constraint-aware mode
            projection_site="post_o",
        )
    except Exception as e:
        print(f"ERROR: Failed to initialize settle_langevin: {e}")
        return {"error": str(e)}

    # Initialize state
    from prolix.physics.simulate import NVTLangevinState
    rng_key = jax.random.PRNGKey(42)

    state = NVTLangevinState(
        position=jnp.asarray(positions),
        momentum=jnp.asarray(momentum),
        force=jnp.zeros_like(positions),
        mass=jnp.asarray(masses)[:, None],  # mass is stored with extra dimension
        rng=rng_key,
    )

    # JIT-compile the integrator step (critical for performance)
    apply_fn_jit = jax.jit(apply_fn)

    # Storage for metrics
    temperatures = []
    energies = []
    checkpoint_every = max(1, n_steps // 100)  # 100 checkpoints over simulation

    print(f"\nRunning {n_steps} steps (JIT compiling first step)...")

    # Run simulation
    for step in range(n_steps):
        # Apply integrator
        state = apply_fn_jit(state)

        # Checkpoint every checkpoint_every steps
        if step % checkpoint_every == 0:
            # Compute rigid-body kinetic energy
            ke_rigid = rigid_tip3p_box_ke_kcal(
                state.position,
                state.momentum,
                masses,
                n_waters
            )

            # Temperature from rigid-body KE
            dof_rigid = 6 * n_waters - 3
            boltzmann_kcal = 1.987e-3  # kcal/(mol*K)
            t_instant = 2.0 * ke_rigid / (dof_rigid * boltzmann_kcal)

            temperatures.append(float(t_instant))

            # Total energy (KE only for this test, no PE)
            energies.append(float(ke_rigid))

            # Progress
            progress = (step + 1) / n_steps * 100
            if step % (checkpoint_every * 10) == 0:
                print(f"  Step {step:6d}/{n_steps} ({progress:5.1f}%) - T={t_instant:6.1f}K")

    # Analysis
    temperatures = np.array(temperatures)
    energies = np.array(energies)

    t_mean = np.mean(temperatures)
    t_std = np.std(temperatures)
    t_min = np.min(temperatures)
    t_max = np.max(temperatures)

    e_initial = energies[0]
    e_final = energies[-1]
    e_drift = abs(e_final - e_initial) / abs(e_initial) * 100 if e_initial != 0 else 0

    # Equipartition test: KS test on normalized velocities
    velocities_3d = np.asarray(state.momentum) / masses[:, None]
    sigma_v = np.sqrt(kT / masses)
    velocities_normalized = velocities_3d.flatten() / np.repeat(sigma_v, 3)

    ks_stat, ks_pvalue = stats.kstest(velocities_normalized, 'norm')

    # Results
    results = {
        "dt_fs": dt_fs,
        "duration_ps": duration_ps,
        "n_waters": n_waters,
        "n_steps": n_steps,
        "T_mean": t_mean,
        "T_std": t_std,
        "T_min": t_min,
        "T_max": t_max,
        "E_drift_percent": e_drift,
        "KS_pvalue": ks_pvalue,
    }

    # Print results
    print(f"\n{'─'*70}")
    print(f"Temperature:")
    print(f"  Mean = {t_mean:7.1f} K")
    print(f"  Std  = {t_std:7.1f} K")
    print(f"  Range: {t_min:7.1f} - {t_max:7.1f} K")

    print(f"\nEnergy Conservation:")
    print(f"  Drift = {e_drift:6.2f}%")

    print(f"\nEquipartition (KS test):")
    print(f"  p-value = {ks_pvalue:.4f} (threshold > 0.05)")

    # Gates
    print(f"\n{'─'*70}")
    t_gate = abs(t_mean - 300.0) < 5.0
    e_gate = e_drift < 1.0
    ks_gate = ks_pvalue > 0.05

    print(f"Temperature Gate  (300±5K): {'✓ PASS' if t_gate else '✗ FAIL'}")
    print(f"Energy Gate       (<1%):    {'✓ PASS' if e_gate else '✗ FAIL'}")
    print(f"Equipartition Gate (p>0.05): {'✓ PASS' if ks_gate else '✗ FAIL'}")

    all_pass = t_gate and e_gate and ks_gate
    print(f"\nOverall: {'✓ PASS' if all_pass else '✗ FAIL'}")
    print(f"{'='*70}\n")

    return results


def main():
    """Run Phase 2 validation suite."""
    print("\nv0.3.0 PHASE 2: CONSTRAINT-AWARE LANGEVIN VALIDATION\n")

    all_results = {}

    # Test 1: Baseline control (dt=0.5fs)
    print("\n[TEST 1/3] Baseline Control (dt=0.5fs)")
    results_05 = run_validation_test(dt_fs=0.5, duration_ps=50.0, n_waters=4)
    all_results["dt_05fs"] = results_05

    if "error" not in results_05:
        # Test 2: Primary goal (dt=1.0fs)
        print("\n[TEST 2/3] Primary Goal (dt=1.0fs) — CRITICAL GATE")
        results_10 = run_validation_test(dt_fs=1.0, duration_ps=100.0, n_waters=4)
        all_results["dt_10fs"] = results_10

        # Test 3: Stretch goal (dt=2.0fs) — only if 1.0fs passed
        if all([
            results_10.get("T_mean", float("inf")) - 300 < 5,
            results_10.get("E_drift_percent", float("inf")) < 1.0,
            results_10.get("KS_pvalue", 0) > 0.05,
        ]):
            print("\n[TEST 3/3] Stretch Goal (dt=2.0fs)")
            results_20 = run_validation_test(dt_fs=2.0, duration_ps=100.0, n_waters=4)
            all_results["dt_20fs"] = results_20

    # Summary report
    print("\n" + "="*70)
    print("SUMMARY REPORT")
    print("="*70)

    for key, results in all_results.items():
        if "error" in results:
            print(f"\n{key}: ERROR - {results['error']}")
        else:
            dt = results["dt_fs"]
            t_mean = results["T_mean"]
            e_drift = results["E_drift_percent"]
            ks_p = results["KS_pvalue"]

            print(f"\ndt={dt:.1f}fs:")
            print(f"  T={t_mean:6.1f}K ({'PASS' if abs(t_mean-300)<5 else 'FAIL'})")
            print(f"  Drift={e_drift:5.2f}% ({'PASS' if e_drift<1.0 else 'FAIL'})")
            print(f"  KS p={ks_p:.4f} ({'PASS' if ks_p>0.05 else 'FAIL'})")

    print("\n" + "="*70)
    print("DECISION")
    print("="*70)

    if "dt_10fs" in all_results and "error" not in all_results["dt_10fs"]:
        r = all_results["dt_10fs"]
        if (abs(r["T_mean"] - 300) < 5 and
            r["E_drift_percent"] < 1.0 and
            r["KS_pvalue"] > 0.05):
            print("\n✓ dt=1.0fs PASSED - Constraint-aware thermostat validated!")
            print("→ Proceed to Phase 3: Edge cases, performance tuning, release")
            return 0
        else:
            print("\n✗ dt=1.0fs FAILED - Thermostat-SETTLE coupling persists")
            print("→ Escalate per insurance plan (see PHASE2_VALIDATION_SPEC.md)")
            return 1
    else:
        print("\n✗ Tests encountered errors - See output above")
        return 2


if __name__ == "__main__":
    exit(main())
