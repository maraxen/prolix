#!/usr/bin/env python3
"""Profile memory usage of run_simulation with different settings."""

import argparse
import logging
import os
import time
import jax
import jax.numpy as jnp
import numpy as np
from proxide.io.parsing.backend import parse_structure
from proxide import OutputSpec, CoordFormat

from prolix import simulate

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_memory_stats():
    try:
        device = jax.devices()[0]
        if device.platform == "gpu":
            stats = device.memory_stats()
            return {
                "bytes_in_use": stats.get("bytes_in_use", 0),
                "peak_bytes_in_use": stats.get("peak_bytes_in_use", 0),
            }
    except Exception:
        pass
    return {}

def profile_run(pdb_path: str, accumulate_steps: int, use_neighbor_list: bool, steps: int = 1000):
    logger.info(f"Profiling {pdb_path} | Accumulate: {accumulate_steps} | NL: {use_neighbor_list}")
    
    # 1. Load System
    ff_path = "proxide/src/proxide/assets/protein.ff19SB.xml"
    if not os.path.exists(ff_path):
         ff_path = os.path.join(os.getcwd(), "proxide/src/proxide/assets/protein.ff19SB.xml")
    
    spec = OutputSpec(
        coord_format=CoordFormat.Full,
        force_field=ff_path,
        parameterize_md=True,
    )
    protein = parse_structure(pdb_path, spec)
    
    # Simple box estimation if not present (for explicit solvent)
    box = None
    if use_neighbor_list: # NL requires box usually
        # Derive box from coords just for testing if not explicit
        coords = protein.coordinates.reshape(-1, 3)
        extent = np.max(coords, axis=0) - np.min(coords, axis=0)
        box = jnp.array(extent + 20.0) # Padding
        # Ideally this PDB has box info, but proxide parser might not expose it easily yet 
        # unless we check 'pbc' field if it exists. 
        # For solvated pdb, let's assume valid box if we can find it.
        # Actually, simulate.py uses spec.box.
    
    if "solvated" in pdb_path:
        # Hardcode a reasonable box for 1UAO Solvated if parameters missing
        # Or try to get it. 
        # proxide parse_structure usually handles pbc if explicitly requested?
        # Let's just use a large box to avoid errors if not found.
        box = jnp.array([60.0, 60.0, 60.0])

    # 2. Setup Spec
    sim_spec = simulate.SimulationSpec(
        total_time_ns=(steps * 2.0) / 1e6, # just enough for 'steps'
        step_size_fs=2.0,
        save_interval_ns=0.002, # save every step? No, save every 1000 steps normally
        # We want to match accumulate_steps to force saving regularily
        accumulate_steps=accumulate_steps,
        save_path=f"profile_{accumulate_steps}_{use_neighbor_list}.array_record",
        use_neighbor_list=use_neighbor_list,
        use_pbc=(box is not None),
        box=box,
        neighbor_cutoff=9.0,
    )

    # Align save interval to accumulate steps roughly for this test
    # save every 'accumulate_steps' steps
    sim_spec.save_interval_ns = (accumulate_steps * 2.0) / 1e6

    # 3. Reset stats
    jax.clear_caches() # Clear cache if possible
    
    mem_before = get_memory_stats()
    logger.info(f"Memory Before: {mem_before}")

    # 4. Run
    start_t = time.time()
    try:
        simulate.run_simulation(
            system=protein, # Passing AtomicSystem provided by proxide
            spec=sim_spec
        )
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        # import traceback
        # traceback.print_exc()
    
    jax.block_until_ready(jnp.array(0)) # Sync
    end_t = time.time()
    
    mem_after = get_memory_stats()
    logger.info(f"Memory After: {mem_after}")
    logger.info(f"Time: {end_t - start_t:.2f}s")
    
    peak_diff = mem_after.get("peak_bytes_in_use", 0) - mem_before.get("bytes_in_use", 0)
    logger.info(f"Approx Peak Increase: {peak_diff / 1024**2:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb", default="prolix/data/pdb/1UAO_solvated_tip3p.pdb")
    parser.add_argument("--accumulate", type=int, default=100)
    parser.add_argument("--neighbor", action="store_true")
    args = parser.parse_args()
    
    if not os.path.exists(args.pdb):
        # Fallback to 1UBQ if solvated not found
        args.pdb = "prolix/data/pdb/1UBQ.pdb"
        print(f"File not found, falling back to {args.pdb}")

    profile_run(args.pdb, args.accumulate, args.neighbor)
