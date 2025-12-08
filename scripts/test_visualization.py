"""Test visualization suite on generated trajectory."""
import os
import matplotlib.pyplot as plt
import numpy as np
import biotite.structure as struc
import biotite.structure.io as struc_io
from prolix.visualization import TrajectoryReader, plot_rmsd, plot_energy, plot_contact_map, plot_ramachandran, plot_free_energy_surface

def get_backbone_indices(pdb_path):
    """Get Phi/Psi indices manually."""
    # Load structure
    array = struc_io.load_structure(pdb_path, model=1)
    
    # Filter only protein
    protein = array[struc.filter_amino_acids(array)]
    
    # Get residues
    res_ids, starts = struc.get_residue_starts(protein)
    
    phi_ind = []
    psi_ind = []
    
    # Needs C-N-CA-C and N-CA-C-N
    # Iterate residues
    for i in range(len(starts)):
        # Check if we have prev and current for Phi
        # Phi: C(i-1) - N(i) - CA(i) - C(i)
        if i > 0:
            # Get atoms
            curr_res = protein[starts[i]:starts[i+1] if i+1 < len(starts) else None]
            prev_res = protein[starts[i-1]:starts[i]]
            
            try:
                c_prev = prev_res[prev_res.atom_name == "C"][0].i
                n_curr = curr_res[curr_res.atom_name == "N"][0].i
                ca_curr = curr_res[curr_res.atom_name == "CA"][0].i
                c_curr = curr_res[curr_res.atom_name == "C"][0].i
                phi_ind.append([c_prev, n_curr, ca_curr, c_curr])
            except IndexError:
                pass
                
        # Psi: N(i) - CA(i) - C(i) - N(i+1)
        if i < len(starts) - 1:
            curr_res = protein[starts[i]:starts[i+1] if i+1 < len(starts) else None]
            next_res = protein[starts[i+1]:starts[i+2] if i+2 < len(starts) else None]
            
            try:
                n_curr = curr_res[curr_res.atom_name == "N"][0].i
                ca_curr = curr_res[curr_res.atom_name == "CA"][0].i
                c_curr = curr_res[curr_res.atom_name == "C"][0].i
                n_next = next_res[next_res.atom_name == "N"][0].i
                psi_ind.append([n_curr, ca_curr, c_curr, n_next])
            except IndexError:
                pass
                
    return np.array(phi_ind), np.array(psi_ind)

def main():
    traj_path = "chignolin_traj.array_record"
    pdb_path = "data/pdb/1UAO.pdb"
    
    if not os.path.exists(traj_path):
        print("Trajectory not found. Waiting or run scripts/simulate_chignolin_gif.py first.")
        return

    print("Loading trajectory...")
    reader = TrajectoryReader(traj_path)
    print(f"Frames: {len(reader)}")
    
    # 1. RMSD
    print("1. Plotting RMSD...")
    visualization.plot_rmsd(reader, output_path="chignolin_rmsd.png")
    
    # 2. Energy
    print("2. Plotting Energy...")
    visualization.plot_energy(reader, output_path="chignolin_energy.png")
    
    # 3. Contact Map
    print("3. Plotting Contact Map...")
    visualization.plot_contact_map(reader, output_path="chignolin_cmap.png")
    
    # 4. Ramachandran
    print("4. Plotting Ramachandran...")
    if os.path.exists(pdb_path):
        phi, psi = get_backbone_indices(pdb_path)
        if len(phi) > 0 and len(psi) > 0:
            visualization.plot_ramachandran(reader, phi, psi, output_path="chignolin_rama.png")
        else:
            print("Could not extract backbone indices.")
    else:
        print(f"PDB {pdb_path} not found, skipping Ramachandran.")

    # 5. FES (on RMSD)
    print("5. Plotting FES (RMSD)...")
    # Need metric fn
    from prolix import analysis
    import jax.numpy as jnp
    
    # Need reference for RMSD
    # Let's just use first frame
    ref = jnp.array(reader.get_positions()[0])
    
    def rmsd_metric(pos):
        return analysis.compute_rmsd(pos, ref)
        
    visualization.plot_free_energy_surface(
        reader, rmsd_metric, output_path="chignolin_fes.png"
    )

    print("Visualization tests completed. Check .png files.")

if __name__ == "__main__":
    main()
