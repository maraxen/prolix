"""Validation script for Chignolin/Trp-cage simulation and analysis."""
import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import space

from prolix.simulate import SimulationSpec, run_simulation, SimulationState, TrajectoryWriter
from prolix import analysis
from priox.md.jax_md_bridge import create_system_from_pdb

def analyze_trajectory(trajectory_path: str, pdb_path: str, output_dir: str = "analysis_results"):
    """Analyze a trajectory file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Analyzing {trajectory_path} using reference {pdb_path}...")
    
    # 1. Load Reference (Native)
    # create_system_from_pdb returns (params, positions)
    params, ref_positions = create_system_from_pdb(pdb_path)
    
    # 2. Iterate Trajectory
    # In a real scenario, we might iterate efficiently using ArrayRecordReader
    # For this script, we'll assume we can load it via SimulationState.from_array_record 
    # if we iterate manually, OR we use the Grain loader from priox.
    
    # Let's use array_record_module directly since we want raw access
    from array_record.python.array_record_module import ArrayRecordReader
    
    reader = ArrayRecordReader(trajectory_path)
    n_frames = reader.num_records()
    print(f"Found {n_frames} frames.")
    
    rmsds = []
    rgs = []
    fraction_native = []
    
    # Pre-compute native contact map
    native_cmap = analysis.compute_contact_map(ref_positions, threshold_angstrom=8.0)
    n_native_contacts = jnp.sum(native_cmap)
    
    # We'll batch this loop if possible, but reading is sequential
    for i in range(n_frames):
        data = reader.read(i, i+1)[0]
        state = SimulationState.from_array_record(data)
        
        pos = state.positions
        
        # 3. Compute Metrics
        
        # RMSD
        rmsd = analysis.compute_rmsd(pos, ref_positions)
        rmsds.append(float(rmsd))
        
        # Rg
        rg = analysis.compute_radius_of_gyration(pos, params.get("masses"))
        rgs.append(float(rg))
        
        # Contacts (Q)
        curr_cmap = analysis.compute_contact_map(pos, threshold_angstrom=8.0) # Or wider for Q?
        # Usually Q uses slightly wider threshold or sigmoid.
        # We'll use simple shared contacts
        shared = jnp.sum(native_cmap * curr_cmap)
        q = shared / (n_native_contacts + 1e-8)
        fraction_native.append(float(q))
        
        if i % 100 == 0:
            print(f"Processed {i}/{n_frames} frames. Current RMSD: {rmsd:.2f}, Q: {q:.2f}")
            
    reader.close()
    
    # 4. Save Results
    csv_path = output_path / "metrics.csv"
    import csv
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "rmsd", "rg", "Q"])
        for i in range(n_frames):
            writer.writerow([i, rmsds[i], rgs[i], fraction_native[i]])
            
    print(f"Saved metrics to {csv_path}")
    
    print("Optimization Summary:")
    print(f"Mean RMSD: {np.mean(rmsds):.3f}")
    print(f"Mean Q: {np.mean(fraction_native):.3f}")
    
    # Calculate Free Energy (1D along RMSD)
    rmsd_array = jnp.array(rmsds)
    centers, fes = analysis.compute_free_energy_surface(rmsd_array, temperature=300.0)
    
    # Save FES
    np.savez(output_path / "fes_rmsd.npz", centers=centers, fes=fes)
    print(f"Saved FES to {output_path / 'fes_rmsd.npz'}")

def main():
    parser = argparse.ArgumentParser(description="Run simulation and analysis for protein folding validation.")
    parser.add_argument("--pdb", type=str, required=True, help="Path to PDB file (e.g. data/pdb/1UAO.pdb)")
    parser.add_argument("--steps", type=int, default=1000, help="Number of accumulated steps per save")
    parser.add_argument("--time_ns", type=float, default=0.01, help="Total simulation time in ns")
    parser.add_argument("--run", action="store_true", help="Run new simulation")
    parser.add_argument("--analyze", action="store_true", help="Analyze existing trajectory")
    parser.add_argument("--trajectory", type=str, default="trajectory.array_record", help="Trajectory file path")
    
    args = parser.parse_args()
    
    if args.run:
        print(f"Running simulation for {args.pdb}...")
        params, positions = create_system_from_pdb(args.pdb)
        
        # Determine accumulation
        # Default save interval 1ps (0.001 ns)
        # We want to save reasonable number of frames.
        spec = SimulationSpec(
            total_time_ns=args.time_ns,
            save_interval_ns=0.001,
            accumulate_steps=args.steps,
            save_path=args.trajectory
        )
        
        start = time.time()
        final_state = run_simulation(params, positions, spec)
        end = time.time()
        print(f"Simulation completed in {end - start:.2f}s")
        
    if args.analyze:
        analyze_trajectory(args.trajectory, args.pdb)

if __name__ == "__main__":
    main()
