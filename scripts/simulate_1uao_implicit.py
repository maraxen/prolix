"""Run 1UAO simulation in implicit solvent (GBSA) - fast demo."""

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import os

# Prolix imports
from prolix import simulate
from prolix.visualization import animate_trajectory, TrajectoryReader, save_trajectory_html

# Priox imports
from priox.md.bridge.core import parameterize_system
from priox.physics.force_fields.loader import load_force_field
from priox.io.parsing import biotite as parsing_biotite
import biotite.structure as struc

def main():
  # 1. Load PDB (just protein, no water)
  pdb_path = "data/pdb/1UAO.pdb"
  if not os.path.exists(pdb_path):
      raise FileNotFoundError(f"Missing {pdb_path}")
      
  print(f"Loading {pdb_path}...")
  
  atom_array = parsing_biotite.load_structure_with_hydride(pdb_path, model=1)
  
  # 2. Prepare Topology
  residues = []
  atom_names = []
  atom_counts = []
  
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
      
      # Handle N-term H -> H1
      if i == 0:
          for k in range(len(names)):
              if names[k] == "H":
                   names[k] = "H1"
                   
      atom_names.extend(names)
      atom_counts.append(len(names))

  # N/C terminal naming
  residues[0] = "N" + residues[0]
  residues[-1] = "C" + residues[-1]
      
  print(f"Parsed {len(residues)} residues, {len(atom_names)} atoms")

  # 3. Parameterize
  ff_path = "data/force_fields/ff14SB.eqx"
  if not os.path.exists(ff_path):
      ff_path = "data/force_fields/protein19SB.eqx"
      
  print(f"Loading force field from {ff_path}...")
  ff = load_force_field(ff_path)
  
  print("Parameterizing system...")
  system_params = parameterize_system(ff, residues, atom_names, atom_counts)
  
  # 4. Setup Simulation (implicit solvent - fast!)
  positions = jnp.array(atom_array.coord)
  key = random.PRNGKey(42)
  
  # 10ps simulation - fast for demo
  spec = simulate.SimulationSpec(
    total_time_ns=0.01,  # 10 ps
    step_size_fs=2.0,
    save_interval_ns=0.001,
    save_path="1uao_implicit_traj.array_record",
    temperature_k=300.0,
    gamma=1.0,
    use_pbc=False,  # No PBC for implicit
  )
  
  print(f"Running implicit solvent simulation for {spec.total_time_ns} ns...")
  
  final_state = simulate.run_simulation(
     system_params=system_params,
     r_init=positions,
     spec=spec,
     key=key
  )
  print("Simulation complete!")
  print(f"Final Energy: {final_state.potential_energy} kcal/mol")
  
  # 5. Generate visualization immediately
  print("\nGenerating GIF...")
  animate_trajectory(
      "1uao_implicit_traj.array_record",
      "1uao_movie.gif",
      pdb_path=pdb_path,
      frame_stride=1,
      fps=15,
      title="1UAO Implicit Solvent MD"
  )
  print("GIF saved to 1uao_movie.gif")

if __name__ == "__main__":
  main()
