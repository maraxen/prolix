"""Run Chignolin simulation and export GIF."""

# Using biotite directly to inspect structure if needed
import biotite.structure as struc
import jax.numpy as jnp
from jax import random
from proxide.io.parsing import biotite as parsing_biotite

# Priox imports
from proxide.md.bridge.core import parameterize_system
from proxide.physics.force_fields.loader import load_force_field

# Prolix imports
from prolix import simulate
from prolix.visualization import animate_trajectory


def main():
  # 1. Load PDB
  pdb_path = "data/pdb/1UAO.pdb"
  print(f"Loading {pdb_path}...")

  # Load structure using Priox/Biotite tool (handles H naming etc?)
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

      # Use the first atom's res_name
      res_name = res_atoms.res_name[0]
      residues.append(res_name)

      names = res_atoms.atom_name.tolist()
      # Fix H name for N-term like in verify script?
      # Assuming 1UAO is standard, maybe verify script logic is safer
      # In Verify: if i==0 and j==0 and name=="H" -> "H1"
      # Let's perform a simple check
      if len(residues) == 1:
          # N-term check
          for k in range(len(names)):
              if names[k] == "H":
                   names[k] = "H1"

      atom_names.extend(names)
      atom_counts.append(len(names))

  # Rename terminals for Amber FF (N+ResName, C+ResName)
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

  # 3. Setup Simulation
  print("Setting up simulation...")
  # Create energy function (Implicit Solvent)
  # verify_end_to_end used dielectric_constant=1.0, solvent=78.5

  # Need to prep positions
  positions = jnp.array(atom_array.coord / 10.0) # Convert Angstrom to nm?
  # Wait, JAX MD usually uses Angstrom or nm consistently?
  # OpenMM uses nm. verify script used Angstrom for JAX positions?
  # verify script:
  #   jax_positions = jnp.array(positions.value_in_unit(unit.angstrom))
  #   ... make_energy_fn(..., displacement_fn)
  #   So JAX MD implementation here uses Angstroms!
  positions = jnp.array(atom_array.coord) # Angstroms

  from jax_md import space
  displacement_fn, shift_fn = space.free()

  # Initial state
  key = random.PRNGKey(0)

  # 50ps simulation
  spec = simulate.SimulationSpec(
    total_time_ns=0.05,
    step_size_fs=2.0,
    save_interval_ns=0.001,
    save_path="chignolin_traj.array_record",
    temperature_k=300.0,
    gamma=1.0,
    box=None,
    use_pbc=False
  )

  print(f"Running simulation for {spec.total_time_ns} ns...")
  final_state = simulate.run_simulation(
     system_params=system_params,
     r_init=positions,
     spec=spec,
     key=key
  )
  print("Simulation complete!")

  # 4. Export GIF
  print("Generating GIF animation...")
  output_gif = "outputs/chignolin_movie.gif"

  animate_trajectory(
      trajectory_path=spec.save_path,
      output_path=output_gif,
      pdb_path=pdb_path,
      frame_stride=2,
      fps=15,
      title="Chignolin (Implicit Solvent)"
  )
  print(f"GIF saved to {output_gif}")

if __name__ == "__main__":
  try:
    main()
  except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"CRITICAL ERROR: {e}", flush=True)
