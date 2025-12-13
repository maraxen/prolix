
import jax
import jax.numpy as jnp
import numpy as np
import biotite.structure as struc

from jax_md import simulate, energy, space
from prolix.physics import solvation, system, simulate as p_simulate

from proxide.io.parsing import biotite as parsing_biotite
from proxide.md.bridge import core as bridge_core
from proxide.physics.force_fields import loader as ff_loader

def test_solvation_stability():
    print("===========================================================")
    print("   Testing Solvation Stability (TIP3P - Minimal)")
    print("===========================================================")

    # 1. Load Protein
    pdb_path = "data/pdb/1UAO.pdb"
    print(f"Loading {pdb_path}...")
    # Load with model=1 to get AtomArray
    atom_array = parsing_biotite.load_structure_with_hydride(pdb_path, model=1)
    
    protein_pos = jnp.array(atom_array.coord)
    print(f"Protein atoms: {len(protein_pos)}")
    
    # 2. Solvate
    print("Solvating...")
    # Radii: approx 2.0 A for exclusion
    radii = jnp.ones(len(protein_pos)) * 2.0
    
    # Use explicit box shape to match water box multiple (30.0) to avoid boundary clashes
    # 1UAO fits in 30.0 box.
    target_box = jnp.array([30.0, 30.0, 30.0])
    combined_pos, box_size = solvation.solvate(protein_pos, radii, padding=None, target_box_shape=target_box)
    
    n_total = len(combined_pos)
    n_protein = len(protein_pos)
    n_water_atoms = n_total - n_protein
    n_waters = n_water_atoms // 3
    
    print(f"Solvated System: {n_total} atoms")
    print(f"  Protein: {n_protein}")
    print(f"  Water:   {n_waters} molecules ({n_water_atoms} atoms)")
    print(f"  Box:     {box_size}")
    
    # 3. Build Topology Lists
    # We need full list of residues and atom names for parameterize_system
    
    # Extract protein residues (per atom)
    # Biotite res_name is per-atom
    protein_res_names = list(atom_array.res_name)
    protein_atom_names = list(atom_array.atom_name)
    
    # Water residues
    water_res_names = ["HOH"] * n_water_atoms # per atom!
    # Water atoms
    # Order in solvation.py was O, H1, H2 repeated
    water_atom_names = ["O", "H1", "H2"] * n_waters
    
    full_res_names = protein_res_names + water_res_names
    full_atom_names = protein_atom_names + water_atom_names
    
    # But parameterize_system expects `residues` list to be condensed (per residue)?
    # Wait, my logic in core.py: `for r_i, res_name in enumerate(residues):`
    #   `if atom_counts is not None: count = atom_counts[r_i]`
    # So `residues` MUST be per-residue list.
    
    # Get residue starts for protein
    res_starts = struc.get_residue_starts(atom_array)
    # Protein residue list
    prot_res_list = [atom_array.res_name[i] for i in res_starts]
    
    # Protein atom counts
    prot_counts = []
    for i in range(len(res_starts)-1):
        prot_counts.append(res_starts[i+1] - res_starts[i])
    prot_counts.append(len(atom_array) - res_starts[-1])
    
    # Water residue list
    water_res_list = ["HOH"] * n_waters
    water_counts = [3] * n_waters
    
    full_residues = prot_res_list + water_res_list
    full_counts = prot_counts + water_counts
    
    print(f"Total Residues: {len(full_residues)}")
    
    # 4. Parameterize
    print("Parameterizing System...")
    # Using ff14SB from assets
    ff_name = "amber/ff14SB"
    print(f"Loading Force Field: {ff_name}")
    ff = ff_loader.load_force_field(ff_name)
    
    params = bridge_core.parameterize_system(
        ff, 
        full_residues, 
        full_atom_names, # Flat list of all atom names
        atom_counts=full_counts
    )
    
    print("System Parameters generated.")
    print(f"  Total Charge: {jnp.sum(params['charges']):.4f}")
    print(f"  Bonds: {len(params['bonds'])}")
    print(f"  Constraints: {len(params['constrained_bonds'])}")
    
    # 5. Energy Function
    print("Creating Energy Function (PME)...")
    # displacement_fn for periodic box
    displacement_fn, shift_fn = space.periodic(box_size)
    
    energy_fn = system.make_energy_fn(
        displacement_fn=displacement_fn,
        system_params=params,
        box=box_size,
        use_pbc=True,
        implicit_solvent=False, # EXPLICIT!
        pme_grid_points=64, # Good enough for ~40A box
        cutoff_distance=9.0,
    )
    
    # 6. Minimize
    print("Minimizing Energy...")
    init_energy = energy_fn(combined_pos)
    print(f"  Initial Energy: {init_energy:.4f} kcal/mol")
    
    # Use Prolix minimize wrapper (uses FIRE/Adam)
    # Be careful with shift_fn! periodic shift needed.
    # The wrapper likely takes shift_fn.
    
    # check wrapper signature: minimize(energy_fn, shift_fn, positions, ...)
    final_pos = p_simulate.run_minimization(
        energy_fn, 
        combined_pos, # r_init is 2nd arg
        steps=500,
        dt_start=0.001
    )
    final_energy = energy_fn(final_pos)
    print(f"  Final Energy:   {final_energy:.4f} kcal/mol")
    
    if not jnp.isfinite(final_energy):
        print("ERROR: Energy is NaN or Inf after minimization!")
        return # Fail
        
    # 7. Short NVT Simulation
    print("Running Short NVT (Langevin)...")
    
    # 100 steps
    n_steps = 100
    dt_fs = 2.0
    dt = dt_fs * 1e-3 # ps
    kT = 298.15 * 0.001987
    gamma = 1.0 # friction
    
    init_fn, apply_fn = simulate.nvt_langevin(
        energy_fn,
        shift_fn,
        dt=dt,
        kT=kT,
        gamma=gamma
    )
    
    # Initialize state
    key = jax.random.PRNGKey(0)
    state = init_fn(key, final_pos, mass=params['masses'])
    
    # Loop
    # We can use lax.scan but python loop is fine for 100 steps debugging
    traj_energies = []
    
    import time
    t0 = time.time()
    for i in range(n_steps):
        state = apply_fn(state, time=i*dt)
        if i % 10 == 0:
            e = energy_fn(state.position)
            traj_energies.append(e)
            # print(f"Step {i}: E={e:.4f}")
            
    elapsed = time.time() - t0
    final_step_e = energy_fn(state.position)
    print(f"NVT Complete. {n_steps} steps took {elapsed:.2f}s")
    print(f"Final NVT Energy: {final_step_e:.4f} kcal/mol")
    
    # Check Stability
    if not jnp.isfinite(final_step_e):
         print("FAILED: Simulation exploded (NaN/Inf energy)")
    elif final_step_e > 0 and final_step_e > final_energy + 10000:
         print("FAILED: Energy skyrocketed (Explosion)")
    else:
         print("SUCCESS: Simulation stable.")
         
if __name__ == "__main__":
    test_solvation_stability()
