import os
import sys
import time

import jax
import jax.numpy as jnp
from jax_md import space
from termcolor import colored

# Enable x64
jax.config.update("jax_enable_x64", True)

# Add source path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from biotite import structure
from proxide.io.parsing import biotite as parsing_biotite
from proxide.md import jax_md_bridge
from proxide.physics import force_fields

from prolix.physics import system


def profile_energy_fn(pdb_code="1UAO", n_steps=100):
    print(colored(f"Profiling JIT Compilation for {pdb_code}", "cyan"))

    # 1. Load Data
    pdb_path = f"data/pdb/{pdb_code}.pdb"
    if not os.path.exists(pdb_path):
        os.makedirs("data/pdb", exist_ok=True)
        rcsb.fetch(pdb_code, "pdb", "data/pdb")

    atom_array = parsing_biotite.load_structure_with_hydride(pdb_path, model=1)

    # 2. Parameterize
    ff_path = "data/force_fields/protein19SB.eqx"
    if not os.path.exists(ff_path):
        print(colored(f"Force field not found at {ff_path}", "red"))
        return None

    ff = force_fields.load_force_field(ff_path)

    # Extract info (simplified from verify script)
    full_residues = []
    atom_names = []
    atom_counts = []

    # Iterate residues using biotite.structure.residue_iter
    for i, res_atoms in enumerate(structure.residue_iter(atom_array)):
        res_name = res_atoms[0].res_name
        full_residues.append(res_name)
        atom_counts.append(len(res_atoms))

        for atom in res_atoms:
            name = atom.atom_name
            # Amber N-term H fix (verify script logic)
            if i == 0 and name == "H": name = "H1"
            atom_names.append(name)

    if full_residues:
        full_residues[0] = "N" + full_residues[0]
        full_residues[-1] = "C" + full_residues[-1]


    system_params = jax_md_bridge.parameterize_system(
        ff, full_residues, atom_names, atom_counts
    )

    displacement_fn, shift_fn = space.free()
    positions = jnp.array(atom_array.coord)

    # 3. Create Energy Fn
    print("\nCreating energy function...", end="", flush=True)
    start_create = time.time()
    energy_fn = system.make_energy_fn(
        displacement_fn,
        system_params,
        implicit_solvent=True
    )
    print(f" Done ({time.time() - start_create:.4f}s)")

    # 4. JIT Compile
    print("JIT Compiling...", end="", flush=True)
    jit_start = time.time()
    compiled_fn = jax.jit(energy_fn)
    # Trigger compilation
    _ = compiled_fn(positions).block_until_ready()
    jit_end = time.time()
    print(f" Done ({jit_end - jit_start:.4f}s)")

    # 5. Execution Benchmarking
    print(f"Running {n_steps} steps...", end="", flush=True)
    run_start = time.time()
    for _ in range(n_steps):
        _ = compiled_fn(positions).block_until_ready()
    run_end = time.time()

    avg_time = (run_end - run_start) / n_steps
    print(" Done")
    print(colored(f"\nAverage Execution Time: {avg_time*1000:.4f} ms/step", "green"))

    return jit_end - jit_start, avg_time

if __name__ == "__main__":
    profile_energy_fn()
