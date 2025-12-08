"""Run Chignolin simulation and export trajectory for visualization.

This script properly sets up and runs an MD simulation with:
1. Energy minimization before dynamics
2. Correct unit handling for jax_md
3. Proper NVT Langevin dynamics
"""

import time
import jax
import jax.numpy as jnp
from jax import random
import numpy as np

# Enable x64 for physics
jax.config.update("jax_enable_x64", True)

from jax_md import space, simulate as jax_simulate

# Prolix imports
from prolix.physics import system, simulate as physics_simulate, constants
from prolix.simulate import TrajectoryWriter, SimulationState

# Priox imports
from priox.md.bridge.core import parameterize_system
from priox.physics.force_fields.loader import load_force_field
from priox.io.parsing import biotite as parsing_biotite
import biotite.structure as struc


def main():
    # 1. Load PDB
    pdb_path = "data/pdb/1UAO.pdb"
    print(f"Loading {pdb_path}...")
    
    # Load structure using Priox/Biotite tool
    atom_array = parsing_biotite.load_structure_with_hydride(pdb_path, model=1)
    
    # Extract lists for parameterize_system
    residues = []
    atom_names = []
    atom_counts = []
    
    # Iterate residues
    res_starts = struc.get_residue_starts(atom_array)
    for i, start_idx in enumerate(res_starts):
        if i < len(res_starts) - 1:
            end_idx = res_starts[i+1]
            res_atoms = atom_array[start_idx:end_idx]
        else:
            res_atoms = atom_array[start_idx:]
            
        res_name = res_atoms.res_name[0]
        residues.append(res_name)
        
        names = res_atoms.atom_name.tolist()
        # Fix H name for N-term
        if len(residues) == 1:
            for k in range(len(names)):
                if names[k] == "H":
                     names[k] = "H1"
        
        atom_names.extend(names)
        atom_counts.append(len(names))

    # Rename terminals for Amber FF
    if residues:
        residues[0] = "N" + residues[0]
        residues[-1] = "C" + residues[-1]
        
    print(f"Parsed {len(residues)} residues, {len(atom_names)} atoms")

    # 2. Parameterize
    ff_path = "data/force_fields/protein19SB.eqx"
    print(f"Loading force field from {ff_path}...")
    ff = load_force_field(ff_path)
    
    print("Parameterizing system...")
    system_params = parameterize_system(
        ff, residues, atom_names, atom_counts
    )
    
    # 3. Setup Physics
    print("Setting up simulation...")
    displacement_fn, shift_fn = space.free()
    
    # Create energy function with implicit solvent (GBSA)
    energy_fn = system.make_energy_fn(
        displacement_fn, 
        system_params,
        implicit_solvent=True,
        dielectric_constant=1.0,
        solvent_dielectric=78.5,
    )
    
    # Initial coordinates (in Angstroms)
    coords = jnp.array(atom_array.coord)
    print(f"Initial coords shape: {coords.shape}")
    
    # 4. CRITICAL: Energy Minimization BEFORE dynamics
    print("Running energy minimization...")
    e_initial = energy_fn(coords)
    print(f"  Initial energy: {e_initial:.2f} kcal/mol")
    
    r_min = physics_simulate.run_minimization(energy_fn, coords, steps=1000)
    e_minimized = energy_fn(r_min)
    print(f"  Minimized energy: {e_minimized:.2f} kcal/mol")
    
    # 5. Setup NVT Langevin dynamics
    temperature = 300.0
    dt = 2e-3  # jax_md reduced units (roughly ps)
    gamma = 0.1  # friction coefficient
    kT = constants.BOLTZMANN_KCAL * temperature
    
    print(f"Setting up NVT Langevin: T={temperature}K, dt={dt}, gamma={gamma}, kT={kT:.4f}")
    
    init_fn, apply_fn = jax_simulate.nvt_langevin(
        energy_fn,
        shift_fn=shift_fn,
        dt=dt,
        kT=kT,
        gamma=gamma
    )
    
    key = jax.random.PRNGKey(42)
    state = init_fn(key, r_min)
    
    # 6. Run simulation with trajectory saving
    save_path = "chignolin_traj.array_record"
    save_interval = 100  # Save every N steps
    total_steps = 5000  # Total steps to run
    n_saves = total_steps // save_interval
    
    print(f"Running {total_steps} steps, saving every {save_interval} steps ({n_saves} frames)...")
    
    writer = TrajectoryWriter(save_path)
    
    # JIT the step block
    @jax.jit
    def step_block(state):
        def body_fn(i, s):
            return apply_fn(s)
        return jax.lax.fori_loop(0, save_interval, body_fn, state)
    
    t0 = time.time()
    for block in range(n_saves):
        state = step_block(state)
        
        # Compute energy for logging
        e = energy_fn(state.position)
        step_num = (block + 1) * save_interval
        
        if block % 10 == 0 or block == n_saves - 1:
            print(f"  Step {step_num}: Energy = {e:.2f} kcal/mol")
        
        # Save state
        sim_state = SimulationState(
            positions=state.position,
            velocities=state.momentum / state.mass if hasattr(state, 'mass') and state.mass is not None else state.momentum,
            step=step_num,
            time_ns=step_num * dt * 1e-3,  # Approximate
            potential_energy=e,
        )
        writer.write(sim_state)
        
        # Check for explosion
        if not jnp.isfinite(e):
            print(f"WARNING: Energy became NaN/Inf at step {step_num}!")
            break
            
    writer.close()
    elapsed = time.time() - t0
    print(f"Simulation complete! Took {elapsed:.1f}s")
    print(f"Trajectory saved to: {save_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"CRITICAL ERROR: {e}", flush=True)
