"""Distribution-level dynamics parity test for Implicit Solvent (GBSA).

Compares ensemble statistics (Mean Temperature and Energy Drift)
between Prolix's Langevin integrator (with GBSA) and OpenMM's
Langevin integrator on a solvated protein (1UAO).
"""

import os
import time
import tempfile
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from jax_md import space, partition, simulate as jax_md_simulate

from prolix.physics import generalized_born, system, simulate, neighbor_list as nl
from proxide import CoordFormat, OutputSpec, parse_structure

# Enable x64 for physics precision
jax.config.update("jax_enable_x64", True)

# Paths
DATA_DIR = Path("data/pdb")
# FF_PATH updated to be relative to script location
FF_PATH = (
  Path(__file__).parent.parent.parent / "proxide" / "src" / "proxide" / "assets" / "protein.ff19SB.xml"
)

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
    
    # Find forces
    nb_force = None
    gbsa_force = None
    for i in range(omm_system.getNumForces()):
        f = omm_system.getForce(i)
        if isinstance(f, openmm.NonbondedForce):
            nb_force = f
        if isinstance(f, (openmm.GBSAOBCForce, openmm.CustomGBForce)):
            gbsa_force = f

    n_particles = omm_system.getNumParticles()
    charges = np.zeros(n_particles)
    sigmas = np.zeros(n_particles)
    epsilons = np.zeros(n_particles)
    radii = np.zeros(n_particles)
    scales = np.zeros(n_particles)
    masses = np.zeros(n_particles)
    
    for i in range(n_particles):
        q, sig, eps = nb_force.getParticleParameters(i)
        charges[i] = q.value_in_unit(unit.elementary_charge)
        sigmas[i] = sig.value_in_unit(unit.angstrom)
        epsilons[i] = eps.value_in_unit(unit.kilocalories_per_mole)
        masses[i] = omm_system.getParticleMass(i).value_in_unit(unit.amu)
        
        if isinstance(gbsa_force, openmm.GBSAOBCForce):
            _, r, s = gbsa_force.getParticleParameters(i)
            radii[i] = r.value_in_unit(unit.angstrom)
            scales[i] = s
        else:
            p = gbsa_force.getParticleParameters(i)
            radii[i] = p[1] * 10.0
            scales[i] = p[2]

    return {
        "charges": jnp.array(charges),
        "sigmas": jnp.array(sigmas),
        "epsilons": jnp.array(epsilons),
        "radii": jnp.array(radii),
        "scaled_radii": jnp.array(radii * scales),
        "masses": jnp.array(masses)
    }

def run_openmm_stats(n_steps=5000, target_temp=300.0, dt_fs=2.0):
    """Run OpenMM GBSA and collect T statistics."""
    from openmm import app, unit, openmm
    
    pdb_path = DATA_DIR / "1UAO.pdb"
    pdb = app.PDBFile(str(pdb_path))
    ff = app.ForceField("amber14-all.xml", "implicit/obc2.xml")
    
    omm_system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False
    )
    
    integrator = openmm.LangevinMiddleIntegrator(
        target_temp * unit.kelvin,
        1.0 / unit.picosecond,
        dt_fs * unit.femtoseconds
    )
    
    platform = openmm.Platform.getPlatformByName("Reference")
    simulation = app.Simulation(pdb.topology, omm_system, integrator, platform)
    simulation.context.setPositions(pdb.positions)
    
    # Thermalize
    simulation.step(500)
    
    temps = []
    energies = []
    
    for i in range(n_steps // 100):
        simulation.step(100)
        state = simulation.context.getState(getEnergy=True)
        ke = state.getKineticEnergy().value_in_unit(unit.kilocalories_per_mole)
        n_degrees = 3 * omm_system.getNumParticles() - 3
        temp = (2.0 * ke) / (n_degrees * 0.0019872041)
        temps.append(temp)
        energies.append(state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole) + ke)
        
    return np.array(temps), np.array(energies)

def run_prolix_stats(n_steps=5000, target_temp=300.0, dt_fs=2.0):
    """Run Prolix GBSA and collect T statistics."""
    from openmm import app, unit
    
    # 1. Setup system
    pdb_path = DATA_DIR / "1UAO.pdb"
    pdb = app.PDBFile(str(pdb_path))
    ff = app.ForceField("amber14-all.xml", "implicit/obc2.xml")
    
    omm_system = ff.createSystem(
      pdb.topology,
      nonbondedMethod=app.NoCutoff,
      constraints=None,
    )
    
    spec = OutputSpec(parameterize_md=True, coord_format=CoordFormat.Full, force_field=str(FF_PATH))
    protein = parse_structure(str(pdb_path), spec)
    
    # Sync parameters
    omm_params = _get_prolix_params_from_omm(omm_system)
    protein = protein.replace(
        charges=omm_params["charges"],
        sigmas=omm_params["sigmas"],
        epsilons=omm_params["epsilons"],
        radii=omm_params["radii"],
        scaled_radii=omm_params["scaled_radii"],
        masses=omm_params["masses"]
    )

    exclusion_spec = nl.ExclusionSpec.from_protein(protein)
    displacement_fn, shift_fn = space.free()
    
    energy_fn = system.make_energy_fn(
        displacement_fn,
        protein,
        exclusion_spec=exclusion_spec,
        implicit_solvent=True,
        use_pbc=False,
        strict_parameterization=False
    )
    
    # 2. Setup Langevin
    dt = dt_fs / 1000.0
    kT = 0.0019872041 * target_temp
    
    init_fn, apply_fn = jax_md_simulate.nvt_langevin(
        energy_fn,
        shift_fn=shift_fn,
        dt=dt,
        kT=kT,
        gamma=1.0
    )
    
    key = jax.random.PRNGKey(42)
    state = init_fn(key, jnp.array(np.array(pdb.positions.value_in_unit(unit.nanometer)) * 10.0), mass=protein.masses)
    
    # Thermalize
    print("  Prolix thermalization...")
    for _ in range(500):
        state = apply_fn(state)
        
    temps = []
    energies = []
    
    n_atoms = protein.charges.shape[0]
    n_degrees = 3 * n_atoms - 3
    
    print(f"  Prolix production ({n_steps} steps)...")
    for i in range(n_steps // 100):
        for _ in range(100):
            state = apply_fn(state)
        
        ke = jnp.sum(0.5 * protein.masses * jnp.sum(state.velocity**2, axis=-1))
        temp = (2.0 * ke) / (n_degrees * 0.0019872041)
        pe = energy_fn(state.position)
        
        temps.append(float(temp))
        energies.append(float(pe + ke))
        
    return np.array(temps), np.array(energies)

def main():
    if not openmm_available():
        print("OpenMM not found. Skipping.")
        return

    n_steps = 2000 
    target_temp = 300.0
    
    print(f"Running OpenMM GBSA ({n_steps} steps)...")
    omm_t, omm_e = run_openmm_stats(n_steps, target_temp)
    
    print(f"Running Prolix GBSA ({n_steps} steps)...")
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
    
    if t_diff < 15.0: 
        print("\033[92mPASS: GBSA dynamics are consistent.\033[0m")
    else:
        print("\033[91mFAIL: Significant temperature deviation detected.\033[0m")

if __name__ == "__main__":
    main()
