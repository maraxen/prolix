"""NPT 20ps background run: 4 TIP3P waters, 300K, 1 bar, dt=0.5fs, 40,000 steps."""

import sys
sys.path.insert(0, '/home/marielle/projects/prolix/src')

import jax
import jax.numpy as jnp
import numpy as np
import json
from pathlib import Path

jax.config.update("jax_enable_x64", True)

from prolix.physics import pbc, settle, system, pressure, stress
from prolix.physics.rigid_water_ke import rigid_tip3p_box_ke_kcal
from prolix.physics.units import BAR_PER_AKMA_PRESSURE, AKMA_PRESSURE_PER_BAR
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL

# Import test utilities
import sys
sys.path.insert(0, '/home/marielle/projects/prolix/tests')
from physics.test_explicit_langevin_tip3p_parity import _grid_water_positions, _prolix_params_pure_water

def run_npt_20ps():
    """Run NPT 20ps simulation for 16 TIP3P waters."""
    n_waters = 16
    dt_fs = 0.5
    temperature_k = 300.0
    pressure_bar = 1.0
    steps = 40000
    record_interval = 100
    health_check_step = 10000

    print(f"NPT 20ps run: n_waters={n_waters}, dt={dt_fs}fs, T={temperature_k}K, P={pressure_bar}bar")
    print(f"Total steps: {steps} (20ps at {dt_fs}fs)")

    # Initialize system
    positions_a, box_edge = _grid_water_positions(n_waters, spacing_angstrom=10.0)
    box_vec = jnp.array([box_edge, box_edge, box_edge], dtype=jnp.float64)
    dt_akma = float(dt_fs) / float(AKMA_TIME_UNIT_FS)
    kT = float(temperature_k) * BOLTZMANN_KCAL
    tau_baro_akma = 2000.0  # 0.1 ps barostat time constant
    tau_thermo_akma = 2000.0  # 0.1 ps thermostat time constant

    sys_dict = _prolix_params_pure_water(n_waters)
    displacement_fn, shift_fn = pbc.create_periodic_space(box_vec)
    energy_fn = system.make_energy_fn(
        displacement_fn, sys_dict, box=box_vec, use_pbc=True, implicit_solvent=False,
        pme_grid_points=32, pme_alpha=0.34, cutoff_distance=9.0, strict_parameterization=False
    )

    n_atoms = n_waters * 3
    mass = jnp.array([[15.999], [1.008], [1.008]] * n_waters).reshape(n_atoms)
    water_indices = settle.get_water_indices(0, n_waters)

    # Initialize settle_csvr_npt
    init_s, apply_s = settle.settle_csvr_npt(
        energy_fn, shift_fn,
        dt=dt_akma, kT=kT,
        target_pressure_bar=pressure_bar,
        tau_barostat_akma=tau_baro_akma,
        tau_thermostat_akma=tau_thermo_akma,
        mass=mass,
        water_indices=water_indices,
        box_init=box_vec,
    )

    apply_j = jax.jit(apply_s)
    state = init_s(jax.random.PRNGKey(2026), jnp.array(positions_a), mass=mass, box=box_vec)

    # Storage for results
    T_hist = []
    P_hist = []
    V_hist = []
    nan_detected = False

    print(f"\nRunning {steps} steps...")
    for step in range(steps):
        state = apply_j(state, box=state.box)

        # NaN check
        if not jnp.all(jnp.isfinite(state.position)):
            print(f"ERROR: NaN in positions at step {step}")
            nan_detected = True
            break
        if not jnp.all(jnp.isfinite(state.momentum)):
            print(f"ERROR: NaN in momentum at step {step}")
            nan_detected = True
            break
        if not jnp.all(jnp.isfinite(state.box)):
            print(f"ERROR: NaN in box at step {step}")
            nan_detected = True
            break

        # Record every 100 steps
        if (step + 1) % record_interval == 0:
            # Compute temperature from kinetic energy
            ke = rigid_tip3p_box_ke_kcal(state.position, state.momentum, mass, n_waters)
            dof = 6.0 * n_waters - 3.0
            T_inst = float(2.0 * ke / (dof * BOLTZMANN_KCAL))
            T_hist.append(T_inst)

            # Compute pressure from virial + kinetic energy
            volume = float(jnp.prod(state.box))
            virial_w = float(stress.virial_trace(state.position, state.force))
            pressure_akma = pressure.instantaneous_pressure_akma(ke, virial_w, volume, ndim=3)
            P_inst = float(pressure_akma * BAR_PER_AKMA_PRESSURE)
            P_hist.append(P_inst)

            # Store box volume
            V_inst = float(volume)
            V_hist.append(V_inst)

            if (step + 1) % (record_interval * 10) == 0:
                print(f"Step {step + 1}: T={T_inst:.2f}K, V={V_inst:.2f}Ų, P={P_inst:.2f} bar")

        # Health check at 10K steps
        if step == health_check_step:
            ke = rigid_tip3p_box_ke_kcal(state.position, state.momentum, mass, n_waters)
            dof = 6.0 * n_waters - 3.0
            T_check = float(2.0 * ke / (dof * BOLTZMANN_KCAL))
            volume_check = float(jnp.prod(state.box))
            print(f"\nHealth check at 5ps (step {step}): T={T_check:.2f}K, V={volume_check:.2f}Ų")
            if not (200 <= T_check <= 400):
                print(f"WARNING: Temperature out of bounds at health check: {T_check}K")
            if not jnp.isfinite(volume_check) or volume_check <= 0:
                print(f"WARNING: Invalid volume at health check: {volume_check}")

    # Save results
    results = {
        "n_waters": n_waters,
        "dt_fs": dt_fs,
        "target_temperature_k": temperature_k,
        "target_pressure_bar": pressure_bar,
        "total_steps": steps,
        "steps_completed": len(T_hist) * record_interval,
        "nan_detected": nan_detected,
        "T_mean_k": float(np.mean(T_hist[-200:])) if len(T_hist) > 200 else float(np.mean(T_hist)),
        "T_std_k": float(np.std(T_hist[-200:])) if len(T_hist) > 200 else float(np.std(T_hist)),
        "P_mean_bar": float(np.mean(P_hist[-200:])) if len(P_hist) > 200 else float(np.mean(P_hist)),
        "P_std_bar": float(np.std(P_hist[-200:])) if len(P_hist) > 200 else float(np.std(P_hist)),
        "V_mean_angstrom3": float(np.mean(V_hist[-200:])) if len(V_hist) > 200 else float(np.mean(V_hist)),
        "T_hist": [float(x) for x in T_hist],
        "P_hist": [float(x) for x in P_hist],
        "V_hist": [float(x) for x in V_hist],
    }

    # Compute density (g/cm³) from volume
    if results["V_mean_angstrom3"] > 0:
        N_A = 6.022e23
        V_cm3 = results["V_mean_angstrom3"] * 1e-24  # Ų to cm³
        mass_g = n_waters * 18.015 / N_A
        density_g_cm3 = mass_g / V_cm3 if V_cm3 > 0 else 0.0
        results["density_g_cm3"] = float(density_g_cm3)
        results["density_error_percent"] = float(abs(density_g_cm3 - 0.985) / 0.985 * 100.0)

    Path("/tmp/npt_20ps_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to /tmp/npt_20ps_results.json")
    print(f"T: {results['T_mean_k']:.2f} ± {results['T_std_k']:.2f} K")
    print(f"P: {results['P_mean_bar']:.2f} ± {results['P_std_bar']:.2f} bar")
    if "density_g_cm3" in results:
        print(f"Density: {results['density_g_cm3']:.3f} g/cm³ (error: {results['density_error_percent']:.2f}%)")
    print(f"NaN detected: {nan_detected}")

if __name__ == "__main__":
    try:
        run_npt_20ps()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
