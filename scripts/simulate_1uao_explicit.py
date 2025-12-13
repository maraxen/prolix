"""Run 1UAO simulation in explicit solvent (TIP3P)."""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import os

# Prolix imports
from prolix import simulate
from prolix.physics import system
from prolix.visualization import animate_trajectory, TrajectoryReader

# Priox imports
from proxide.md.bridge.core import parameterize_system
from proxide.physics.force_fields.loader import load_force_field
from proxide.io.parsing import biotite as parsing_biotite
import biotite.structure as struc

def main():
  # 1. Load Solvated PDB
  pdb_path = "data/pdb/1UAO_solvated_tip3p.pdb"
  if not os.path.exists(pdb_path):
      raise FileNotFoundError(f"Missing {pdb_path}. Run tests/physics/test_explicit_parity.py to generate it.")
      
  print(f"Loading {pdb_path}...")
  
  # Load structure (keep solvent!)
  atom_array = parsing_biotite.load_structure_with_hydride(pdb_path, model=1, remove_solvent=False)
  
  # Extract box from PDB (CRYST1)
  if atom_array.box is not None:
      box = atom_array.box
      if box.ndim == 3:
           box = box[0]
      box_size = jnp.array([box[0,0], box[1,1], box[2,2]])
      print(f"Box Size from PDB: {box_size} Angstroms")
  else:
      # Fallback if PDB has no CRYST1 (should verify this doesn't happen for verified PDB)
      print("WARNING: No CRYST1 found in PDB, using estimated box.")
      min_coords = np.min(atom_array.coord, axis=0)
      max_coords = np.max(atom_array.coord, axis=0)
      box_size = jnp.array((max_coords - min_coords))
      print(f"Estimated Box Size: {box_size}")

  # 2. Prepare Topology for Parameterization
  residues = []
  atom_names = []
  atom_counts = []
  
  res_starts = struc.get_residue_starts(atom_array)
  
  # We need to correctly handle N-term H renaming for the protein part
  # AND handle WAT residue naming.
  
  for i, start_idx in enumerate(res_starts):
      if i < len(res_starts) - 1:
          end_idx = res_starts[i+1]
          res_atoms = atom_array[start_idx:end_idx]
      else:
          res_atoms = atom_array[start_idx:]
          
      res_name = res_atoms.res_name[0]
      # Fix water residue name if needed (OpenMM uses HOH, Amber uses WAT)
      if res_name == "HOH":
           res_name = "WAT"
      if res_name == "TIP3": # possible variant
           res_name = "WAT"
           
      residues.append(res_name)
      
      names = res_atoms.atom_name.tolist()
      
      # Handle N-term H -> H1 for protein only (residue 0 usually)
      # 1UAO has 20 residues? waters follow.
      # Solvated PDB has protein first.
      if i == 0 and res_name not in ["WAT", "HOH", "NA", "CL"]:
          for k in range(len(names)):
              if names[k] == "H":
                   names[k] = "H1"
                   
      atom_names.extend(names)
      atom_counts.append(len(names))

  # Rename protein terminals for Amber FF (N+ResName, C+ResName)
  # Find protein range
  protein_indices = [i for i, r in enumerate(residues) if r not in ["WAT", "HOH", "NA", "CL"]]
  if protein_indices:
      start = protein_indices[0]
      end = protein_indices[-1]
      residues[start] = "N" + residues[start]
      residues[end] = "C" + residues[end]
      
  print(f"Parsed {len(residues)} residues ({len(protein_indices)} protein), {len(atom_names)} atoms")

  # 3. Parameterize
  ff_path = "data/force_fields/protein19SB.eqx" # Using best FF
  # OR ff14SB.eqx as per plan? 
  # Plan said ff14SB, but existing scripts used ff19SB. 
  # Let's use ff14SB.eqx if available, else protein19SB.eqx 
  ff_path = "data/force_fields/ff14SB.eqx" 
  if not os.path.exists(ff_path):
      ff_path = "data/force_fields/protein19SB.eqx"
      
  print(f"Loading force field from {ff_path}...")
  ff = load_force_field(ff_path)
  
  print("Parameterizing system (this may take a moment)...")
  system_params = parameterize_system(
      ff, residues, atom_names, atom_counts,
      water_model="TIP3P",
      rigid_water=True
  )
  
  # 4. Setup Simulation
  positions = jnp.array(atom_array.coord) # Angstroms
  
  key = random.PRNGKey(42)
  
  # 50ps simulation -> Reduced to 2ps for rapid verification
  spec = simulate.SimulationSpec(
    total_time_ns=0.002, 
    step_size_fs=2.0,
    save_interval_ns=0.001, # 1 ps save
    save_path="1uao_explicit_traj.array_record",
    temperature_k=300.0,
    gamma=1.0,
    box=box_size,         # Explicit box
    use_pbc=True,         # Enable PBC
    pme_grid_size=48      # ~1A spacing (29A box -> 32 or 48 is fine)
  )
  
  print(f"Running SHORT verification simulation for {spec.total_time_ns} ns...")
  print(f"  Box: {box_size}")
  print(f"  PME Grid: {spec.pme_grid_size}")
  
  final_state = simulate.run_simulation(
     system_params=system_params,
     r_init=positions,
     spec=spec,
     key=key
  )
  print("Simulation complete!")
  print(f"Final Energy: {final_state.potential_energy} kcal/mol")

if __name__ == "__main__":
  main()
