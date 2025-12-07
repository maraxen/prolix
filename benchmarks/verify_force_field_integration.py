import jax
import jax.numpy as jnp
import equinox as eqx
from priox.physics.force_fields import load_force_field
from priox.io.parsing import biotite as parsing_biotite
from prolix.physics import system
from jax_md import space
import time
import os
import traceback
import sys

# Target FFs to verify
TARGET_FFS = [
    "ff14SB", 
    "ff99SB", 
    "charmm36_protein" 
]

# Path to test protein
PDB_PATH = "data/pdb/1UAO.pdb"
FF_DIR = "data/force_fields"

# Enable x64 (often needed for physics)
jax.config.update("jax_enable_x64", True)

def verify_ff(ff_name):
    eqx_path = os.path.join(FF_DIR, f"{ff_name}.eqx")
    
    if not os.path.exists(eqx_path):
        return f"SKIP (File not found: {eqx_path})"

    try:
        print(f"Loading {ff_name}...", flush=True)
        ff = load_force_field(eqx_path)
        
        print(f"Loading PDB for {ff_name}...", flush=True)
        # Ensure directory exists or fetch happens if needed (handled by parsing_biotite potentially if path is valid)
        if not os.path.exists(PDB_PATH):
             # Try to fetch if missing? verify_end_to_end_physics does it.
             # For now assume it exists or fail.
             pass

        atom_array = parsing_biotite.load_structure_with_hydride(PDB_PATH)
        
        print(f"Parameterizing {ff_name}...", flush=True)
        # Use helper to get system params and coords
        params, coords = parsing_biotite.biotite_to_jax_md_system(atom_array, ff)
        
        print(f"Initializing Energy Fn for {ff_name}...", flush=True)
        # Use free space displacement for simple energy check
        displacement_fn, _ = space.free()
        
        # Create energy function
        # Using implicit solvent defaults
        energy_fn = system.make_energy_fn(
            displacement_fn, 
            params,
            implicit_solvent=True
        )
        
        print(f"Computing Energy for {ff_name}...", flush=True)
        E = energy_fn(coords)
        
        if jnp.isnan(E) or jnp.isinf(E):
            return f"FAIL (Energy is NaN/Inf: {E})"
        
        return f"PASS (E={E:.4f})"

    except Exception as e:
        # print(traceback.format_exc(), flush=True) # Optional: print full trace for details
        return f"FAIL (Error: {e})"

def main():
    print(f"Verifying {len(TARGET_FFS)} Force Fields on {PDB_PATH}...\n", flush=True)
    print(f"{'Force Field':<20} | {'Status'}", flush=True)
    print("-" * 50, flush=True)
    
    results = []
    for ff in TARGET_FFS:
        status = verify_ff(ff)
        results.append((ff, status))
        print(f"{ff:<20} | {status}", flush=True)
    
    print("\n--- Summary ---", flush=True)
    passes = sum(1 for _, s in results if "PASS" in s)
    fails = sum(1 for _, s in results if "FAIL" in s)
    skips = sum(1 for _, s in results if "SKIP" in s)
    print(f"PASS: {passes}", flush=True)
    print(f"FAIL: {fails}", flush=True)
    print(f"SKIP: {skips}", flush=True)
    
    if fails > 0:
        exit(1)

if __name__ == "__main__":
    main()
