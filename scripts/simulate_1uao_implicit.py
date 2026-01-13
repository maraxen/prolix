"""Run 1UAO simulation in implicit solvent (GBSA) - fast demo."""

import os

from jax import random
import jax.numpy as jnp
import argparse
import proxide
from proxide import CoordFormat, OutputSpec


# Prolix imports
from prolix import simulate
from prolix.visualization import animate_trajectory
from proxide.io.parsing.backend import parse_structure
import logging

logging.basicConfig(level=logging.INFO)



def main():
    parser = argparse.ArgumentParser(description="Run implicit solvent simulation of 1UAO") # Added by instruction
    args = parser.parse_args() # Added by instruction

    # 1. Load PDB using Rust parser (with hydrogen addition)
    pdb_path = "data/pdb/1UAO.pdb"
    # Use protein.ff19SB for parameters
    ff_path = "proxide/src/proxide/assets/protein.ff19SB.xml"
    
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"Missing {pdb_path}")
    if not os.path.exists(ff_path):
        print(f"WARNING: {ff_path} not found, check path") # Modified by instruction

    print(f"Loading {pdb_path}...")
    
    # Parse structure with Rust parser (handles H-addition and parameterization)
    # Using proxide.io.parsing.rust.parse_structure which returns a Protein (AtomicSystem)
    spec = OutputSpec(
        coord_format=CoordFormat.Full,
        force_field=ff_path,
        parameterize_md=True,
        add_hydrogens=False,
    )
    
    protein = parse_structure(pdb_path, spec)
    
    # Fix for missing implicit solvent radii (ff19SB xml doesn't have them) # Added by instruction
    if protein.radii is None: # Added by instruction
        print("Assigning MBONDI2 radii manually...")
        # Need to convert JAX/numpy arrays to python lists for Rust
        # atom_names is already a list of strings
        radii_list = proxide.assign_mbondi2_radii(protein.atom_names, protein.bonds.tolist())
        protein = protein.replace(radii=jnp.array(radii_list))

    n_atoms = protein.coordinates.shape[0] if protein.coordinates.ndim == 2 else len(protein.coordinates) // 3
    print(f"Loaded structure: {n_atoms} atoms")

    # 2. Setup Simulation (implicit solvent - fast!)
    key = random.PRNGKey(42)

    # 10ps simulation - fast for demo
    sim_spec = simulate.SimulationSpec(
        total_time_ns=0.01,  # 10 ps
        step_size_fs=0.1,
        save_interval_ns=0.001,
        save_path="1uao_implicit_traj.array_record",
        temperature_k=300.0,
        gamma=1.0,
        use_pbc=False,  # No PBC for implicit
    )

    print(f"Running implicit solvent simulation for {sim_spec.total_time_ns} ns...")

    # Pass Protein (AtomicSystem) directly - prolix extracts what it needs
    final_state = simulate.run_simulation(
        system=protein,
        spec=sim_spec,
        key=key
    )
    print("Simulation complete!")
    print(f"Final Energy: {final_state.potential_energy} kcal/mol")

    # 3. Generate visualization
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
