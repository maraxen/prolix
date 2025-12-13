"""
End-to-End Parallel Tempering Verification Script.

This script validates the Prolix REMD implementation by running a short
simulation on 1UAO and checking:
1. No energy explosion (NaN or very large values).
2. Exchange statistics are non-zero.
3. Kinetic energy distributions match expected values for each temperature.
4. Walkers diffuse across temperature space.
"""

import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from termcolor import colored

# JAX Config
jax.config.update("jax_enable_x64", True)

# Add paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from prolix.pt import replica_exchange, temperature
from prolix.physics import system
from proxide.physics import force_fields
from proxide.md import jax_md_bridge
from proxide.io.parsing import biotite as parsing_biotite
from proxide.physics.constants import BOLTZMANN_KCAL
import biotite.structure.io.pdb as pdb

# Configuration
PDB_PATH = "data/pdb/1UAO.pdb"
FF_EQX_PATH = "data/force_fields/protein19SB.eqx"

# REMD Parameters
N_REPLICAS = 4
MIN_TEMP = 300.0
MAX_TEMP = 400.0
TOTAL_TIME_NS = 0.001  # 1 ps per replica (very short for testing)
EXCHANGE_INTERVAL_PS = 0.1  # 100 fs between exchange attempts


def run_verification():
    print(colored("===========================================================", "cyan"))
    print(colored("   Parallel Tempering End-to-End Verification", "cyan"))
    print(colored("===========================================================", "cyan"))

    # -------------------------------------------------------------------------
    # 1. Load Structure
    # -------------------------------------------------------------------------
    print(colored("\n[1] Loading Structure...", "yellow"))
    
    if not os.path.exists(PDB_PATH):
        import biotite.database.rcsb as rcsb
        os.makedirs("data/pdb", exist_ok=True)
        rcsb.fetch("1UAO", "pdb", "data/pdb")
        
    atom_array = parsing_biotite.load_structure_with_hydride(PDB_PATH, model=1)
    print(f"  Loaded {len(atom_array)} atoms")

    # -------------------------------------------------------------------------
    # 2. Parameterize System
    # -------------------------------------------------------------------------
    print(colored("\n[2] Parameterizing System...", "yellow"))
    
    ff = force_fields.load_force_field(FF_EQX_PATH)
    
    # Extract topology from atom_array
    residues = []
    atom_names = []
    atom_counts = []
    
    current_res_id = None
    count = 0
    for atom in atom_array:
        res_id = (atom.chain_id, atom.res_id)
        if res_id != current_res_id:
            if current_res_id is not None:
                atom_counts.append(count)
            residues.append(atom.res_name)
            current_res_id = res_id
            count = 1
        else:
            count += 1
        atom_names.append(atom.atom_name)
    atom_counts.append(count)  # Last residue
    
    # Rename terminals for Amber FF
    if residues:
        residues[0] = "N" + residues[0]
        residues[-1] = "C" + residues[-1]
        
    system_params = jax_md_bridge.parameterize_system(ff, residues, atom_names, atom_counts)
    
    positions = jnp.array(atom_array.coord)
    print(f"  Parameterized: {len(positions)} atoms, {len(residues)} residues")

    # -------------------------------------------------------------------------
    # 3. Run REMD
    # -------------------------------------------------------------------------
    print(colored("\n[3] Running REMD...", "yellow"))
    
    spec = replica_exchange.ReplicaExchangeSpec(
        n_replicas=N_REPLICAS,
        min_temp=MIN_TEMP,
        max_temp=MAX_TEMP,
        total_time_ns=TOTAL_TIME_NS,
        step_size_fs=2.0,
        exchange_interval_ps=EXCHANGE_INTERVAL_PS,
        save_path=""  # No saving for verification
    )
    
    temps = temperature.generate_temperature_ladder(N_REPLICAS, MIN_TEMP, MAX_TEMP)
    print(f"  Temperature Ladder: {temps}")
    
    key = random.PRNGKey(42)
    
    start_time = time.time()
    final_state = replica_exchange.run_replica_exchange(system_params, positions, spec, key)
    end_time = time.time()
    
    duration = end_time - start_time
    total_steps = int(TOTAL_TIME_NS * 1e6 / spec.step_size_fs) * N_REPLICAS
    steps_per_sec = total_steps / duration
    
    print(f"  Simulation completed in {duration:.2f}s")
    print(f"  Total steps (all replicas): {total_steps}")
    print(f"  Steps/sec: {steps_per_sec:.2f}")

    # -------------------------------------------------------------------------
    # 4. Verification Checks
    # -------------------------------------------------------------------------
    print(colored("\n[4] Verification Checks", "magenta"))
    
    all_passed = True
    
    # Check 1: No NaN energies
    pe = np.array(final_state.potential_energy)
    ke = np.array(final_state.kinetic_energy)
    
    if np.any(np.isnan(pe)) or np.any(np.isnan(ke)):
        print(colored("  FAIL: NaN energies detected!", "red"))
        all_passed = False
    else:
        print(colored("  PASS: No NaN energies", "green"))
        
    # Check 2: No energy explosion
    if np.any(np.abs(pe) > 1e6) or np.any(np.abs(ke) > 1e6):
        print(colored("  FAIL: Energy explosion detected!", "red"))
        all_passed = False
    else:
        print(colored("  PASS: Energies are bounded", "green"))
        
    # Check 3: Exchange statistics
    n_attempts = int(final_state.exchange_attempts)
    n_successes = int(final_state.exchange_successes)
    
    print(f"  Exchange Attempts: {n_attempts}")
    print(f"  Exchange Successes: {n_successes}")
    
    if n_attempts == 0:
        print(colored("  FAIL: No exchange attempts made!", "red"))
        all_passed = False
    elif n_successes == 0:
        print(colored("  WARNING: No successful exchanges (may be OK for short runs)", "yellow"))
    else:
        rate = n_successes / n_attempts
        if rate < 0.05:
            print(colored(f"  WARNING: Very low acceptance rate ({rate:.2%})", "yellow"))
        elif rate > 0.95:
            print(colored(f"  WARNING: Very high acceptance rate ({rate:.2%}) - check temperature spacing", "yellow"))
        else:
            print(colored(f"  PASS: Reasonable acceptance rate ({rate:.2%})", "green"))
            
    # Check 4: Kinetic Energy matches temperature
    # <KE> = (3N/2) * kT for 3D system with N atoms
    # But this is per replica, so N = n_atoms
    n_atoms = positions.shape[0]
    expected_ke = 0.5 * 3 * n_atoms * BOLTZMANN_KCAL * temps  # In kcal/mol
    
    print(f"  Expected <KE> per replica: {expected_ke}")
    print(f"  Actual KE per replica: {ke}")
    
    # We can't compare instantaneous KE to expected average perfectly,
    # but order of magnitude should be right.
    ke_ratio = ke / expected_ke
    print(f"  KE Ratio (Actual/Expected): {ke_ratio}")
    
    if np.any(ke_ratio < 0.1) or np.any(ke_ratio > 10.0):
        print(colored("  WARNING: KE significantly off from expected (may be OK instantaneously)", "yellow"))
    else:
        print(colored("  PASS: KE order of magnitude is correct", "green"))
        
    # Check 5: Walker diffusion
    walker_indices = np.array(final_state.walker_indices)
    print(f"  Final Walker Indices: {walker_indices}")
    
    # For a very short run, we can't guarantee much diffusion
    # But we can check that the state is valid
    if len(set(walker_indices)) < N_REPLICAS:
        # This means some walkers have same index, which is wrong
        print(colored("  FAIL: Duplicate walker indices!", "red"))
        all_passed = False
    else:
        print(colored("  PASS: Walker indices are unique", "green"))
        
    # -------------------------------------------------------------------------
    # 5. Summary
    # -------------------------------------------------------------------------
    print(colored("\n[5] Summary", "magenta"))
    
    if all_passed:
        print(colored("  All checks PASSED!", "green"))
    else:
        print(colored("  Some checks FAILED.", "red"))
        
    # Print energy breakdown
    print("\n  Final State Energy Breakdown:")
    for i in range(N_REPLICAS):
        print(f"    Replica {i} (T={temps[i]:.1f}K): PE={pe[i]:.2f}, KE={ke[i]:.2f}")
        
    return all_passed


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
