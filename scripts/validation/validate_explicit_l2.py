"""Distribution-level dynamics parity test (Phase 3).

Compares ensemble statistics (mean Temperature and energy drift)
between Prolix's BAOAB Langevin integrator (with SETTLE) and OpenMM's
Langevin integrator on a solvated 1UAO system.
"""

import os
import time
import tempfile
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from pathlib import Path
from jax_md import space, partition, quantity

from prolix.physics import pbc, settle, solvation, system, simulate
from prolix.physics import neighbor_list as nl
from proxide import CoordFormat, OutputSpec, parse_structure

# Enable x64 for physics precision
jax.config.update("jax_enable_x64", True)

# Paths
DATA_DIR = Path("data/pdb")
FF_PATH = Path("../proxide/src/proxide/assets/protein.ff19SB.xml")

# Regression parameters from conftest.py
REGRESSION_PME = {
  "pme_alpha": 0.34,
  "pme_grid_points": 32,
  "cutoff_distance": 9.0,
}

def openmm_available():
  try:
    import openmm
    return True
  except ImportError:
    return False

def _get_prolix_params_from_omm(omm_system):
    """Extract physics parameters from OpenMM System into Prolix-compatible dict."""
    import openmm
    from openmm import unit
    
    n_particles = omm_system.getNumParticles()
    charges = np.zeros(n_particles)
    sigmas = np.zeros(n_particles)
    epsilons = np.zeros(n_particles)
    
    # Nonbonded
    for i in range(omm_system.getNumForces()):
        force = omm_system.getForce(i)
        if isinstance(force, openmm.NonbondedForce):
            for j in range(n_particles):
                q, sig, eps = force.getParticleParameters(j)
                charges[j] = q.value_in_unit(unit.elementary_charge)
                sigmas[j] = sig.value_in_unit(unit.angstrom)
                epsilons[j] = eps.value_in_unit(unit.kilocalories_per_mole)

    return {
        "charges": jnp.array(charges),
        "sigmas": jnp.array(sigmas),
        "epsilons": jnp.array(epsilons),
    }

def run_openmm_stats(n_steps=10000, target_temp=300.0, dt_fs=2.0):
    """Run OpenMM and collect T statistics."""
    from openmm import app, unit, openmm
    
    pdb_path = DATA_DIR / "1UAO.pdb"
    pdb = app.PDBFile(str(pdb_path))
    ff = app.ForceField(str(FF_PATH), "amber14/tip3p.xml")
    
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(ff)
    modeller.addSolvent(ff, padding=0.8 * unit.nanometer, model="tip3p")
    
    cutoff = REGRESSION_PME["cutoff_distance"]
    
    omm_system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=(cutoff / 10.0) * unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
        removeCMMotion=True
    )
    
    # Precise PME
    for i in range(omm_system.getNumForces()):
        force = omm_system.getForce(i)
        if isinstance(force, openmm.NonbondedForce):
            force.setPMEParameters(REGRESSION_PME["pme_alpha"] * 10.0, 
                                 REGRESSION_PME["pme_grid_points"], 
                                 REGRESSION_PME["pme_grid_points"], 
                                 REGRESSION_PME["pme_grid_points"])
            force.setUseDispersionCorrection(False)

    integrator = openmm.LangevinMiddleIntegrator(
        target_temp * unit.kelvin,
        1.0 / unit.picosecond,
        dt_fs * unit.femtoseconds
    )
    
    platform = openmm.Platform.getPlatformByName("Reference")
    simulation = app.Simulation(modeller.topology, omm_system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    
    # Thermalize briefly
    simulation.step(500)
    
    temps = []
    energies = []
    
    for i in range(n_steps // 100):
        simulation.step(100)
        state = simulation.context.getState(getEnergy=True)
        ke = state.getKineticEnergy().value_in_unit(unit.kilocalories_per_mole)
        n_degrees = 3 * omm_system.getNumParticles() - omm_system.getNumConstraints()
        temp = (2.0 * ke) / (n_degrees * 0.0019872041)
        temps.append(temp)
        energies.append(state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole) + ke)
        
    return np.array(temps), np.array(energies)

def run_prolix_stats(n_steps=10000, target_temp=300.0, dt_fs=2.0):
    """Run Prolix and collect T statistics."""
    from openmm import app, unit
    
    # 1. Setup solvated system via OpenMM Modeller
    pdb_path = DATA_DIR / "1UAO.pdb"
    pdb = app.PDBFile(str(pdb_path))
    ff = app.ForceField(str(FF_PATH), "amber14/tip3p.xml")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(ff)
    modeller.addSolvent(ff, padding=0.8 * unit.nanometer, model="tip3p")
    
    positions_nm = modeller.positions.value_in_unit(unit.nanometer)
    positions_A = np.array([[p[0] * 10, p[1] * 10, p[2] * 10] for p in positions_nm])

    omm_system = ff.createSystem(
      modeller.topology,
      nonbondedMethod=app.PME,
      nonbondedCutoff=0.9 * unit.nanometer,
      constraints=None,
    )

    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w") as tmp:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, tmp)
        tmp.flush()
        spec = OutputSpec(parameterize_md=True, coord_format=CoordFormat.Full, force_field=str(FF_PATH))
        protein = parse_structure(tmp.name, spec)

    # 2. Build Prolix system
    box_vecs = modeller.topology.getPeriodicBoxVectors()
    box = jnp.array([
        box_vecs[0][0].value_in_unit(unit.angstrom),
        box_vecs[1][1].value_in_unit(unit.angstrom),
        box_vecs[2][2].value_in_unit(unit.angstrom),
    ])
    displacement_fn, shift_fn = pbc.create_periodic_space(box)
    
    # MANUAL WATER EXCLUSIONS (O-H1, O-H2, H1-H2)
    water_excl = []
    water_indices_list = []
    for atom in modeller.topology.atoms():
        if atom.residue.name in ("HOH", "WAT", "TIP3"):
            if atom.name == "O":
                o_idx = atom.index
                h1_idx = o_idx + 1
                h2_idx = o_idx + 2
                water_excl.extend([[o_idx, h1_idx], [o_idx, h2_idx], [h1_idx, h2_idx]])
                water_indices_list.append([o_idx, h1_idx, h2_idx])
    
    # Sync charges/sigmas/epsilons with OpenMM to ensure 1-1 match
    omm_params = _get_prolix_params_from_omm(omm_system)
    protein = protein.replace(
        charges=omm_params["charges"],
        sigmas=omm_params["sigmas"],
        epsilons=omm_params["epsilons"]
    )

    n_atoms = protein.charges.shape[0]
    self_excl = jnp.stack([jnp.arange(n_atoms), jnp.arange(n_atoms)], axis=1)
    
    exclusion_spec = nl.ExclusionSpec.from_protein(protein)
    all_1213 = jnp.concatenate([exclusion_spec.idx_12_13, self_excl, jnp.array(water_excl, dtype=jnp.int32)], axis=0)
    exclusion_spec = nl.ExclusionSpec(
        n_atoms=n_atoms,
        idx_12_13=all_1213,
        idx_14=exclusion_spec.idx_14,
        scale_14_elec=exclusion_spec.scale_14_elec,
        scale_14_vdw=exclusion_spec.scale_14_vdw
    )
    
    energy_fn = system.make_energy_fn(
        displacement_fn,
        protein,
        exclusion_spec=exclusion_spec,
        box=box,
        use_pbc=True,
        implicit_solvent=False,
        pme_alpha=REGRESSION_PME["pme_alpha"],
        pme_grid_points=REGRESSION_PME["pme_grid_points"],
        cutoff_distance=REGRESSION_PME["cutoff_distance"],
        strict_parameterization=False
    )
    
    # 3. Setup Langevin + SETTLE
    dt = dt_fs / 1000.0
    kT = 0.0019872041 * target_temp
    
    water_indices = jnp.array(water_indices_list, dtype=jnp.int32)
    
    init_fn, apply_fn = settle.settle_langevin(
        energy_fn,
        shift_fn=shift_fn,
        dt=dt,
        kT=kT,
        gamma=1.0,
        water_indices=water_indices
    )
    
    key = jax.random.PRNGKey(42)
    state = init_fn(key, jnp.array(positions_A))
    
    # Thermalize
    print("  Prolix thermalization (500 steps)...")
    for _ in range(500):
        state = apply_fn(state)
        
    temps = []
    energies = []
    
    n_constraints = len(water_indices_list) * 3 # SETTLE removes 3 DOF per water
    n_degrees = 3 * n_atoms - n_constraints
    
    print(f"  Prolix production ({n_steps} steps)...")
    for i in range(n_steps // 100):
        for _ in range(100):
            state = apply_fn(state)
        
        ke = jnp.sum(0.5 * jnp.sum(state.momentum**2, axis=-1) / state.mass.squeeze())
        temp = (2.0 * ke) / (n_degrees * 0.0019872041)
        pe = energy_fn(state.position)
        
        temps.append(float(temp))
        energies.append(float(pe + ke))
        
    return np.array(temps), np.array(energies)

def main():
    if not openmm_available():
        print("OpenMM not found. Skipping statistical parity.")
        return

    n_steps = 1000 # Short run for smoke test
    target_temp = 300.0
    
    print(f"Running OpenMM ({n_steps} steps)...")
    omm_t, omm_e = run_openmm_stats(n_steps, target_temp)
    
    print(f"Running Prolix ({n_steps} steps)...")
    jax_t, jax_e = run_prolix_stats(n_steps, target_temp)
    
    print(f"\nResults (Target T = {target_temp} K):")
    print(f"  Integrator | Mean T (K) | Std T (K) | Drift (kcal/mol/step)")
    print(f"  -----------|------------|-----------|----------------------")
    
    omm_drift = (omm_e[-1] - omm_e[0]) / n_steps
    jax_drift = (jax_e[-1] - jax_e[0]) / n_steps
    
    print(f"  OpenMM     | {np.mean(omm_t):10.2f} | {np.std(omm_t):9.2f} | {omm_drift:10.6f}")
    print(f"  Prolix     | {np.mean(jax_t):10.2f} | {np.std(jax_t):9.2f} | {jax_drift:10.6f}")
    
    t_diff = abs(np.mean(omm_t) - np.mean(jax_t))
    print(f"\nTemperature difference: {t_diff:.2f} K")
    
    if np.isnan(t_diff):
        print("\033[91mFAIL: NaNs detected in dynamics.\033[0m")
    elif t_diff < 15.0: # Allow larger margin for short smoke run
        print("\033[92mPASS: Distribution-level dynamics are consistent.\033[0m")
    else:
        print("\033[91mFAIL: Significant temperature deviation detected.\033[0m")

if __name__ == "__main__":
    main()
