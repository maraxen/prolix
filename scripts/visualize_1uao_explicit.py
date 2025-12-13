"""Visualize 1UAO explicit solvent trajectory."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

import jax.numpy as jnp

from prolix.visualization import TrajectoryReader, save_trajectory_html
from proxide.io.parsing import biotite as parsing_biotite
import biotite.structure as struc

def main():
    # Paths
    pdb_path = "data/pdb/1UAO_solvated_tip3p.pdb"
    traj_path = "1uao_explicit_traj.array_record"
    
    if not os.path.exists(traj_path):
        print(f"Error: Trajectory {traj_path} not found. Run simulation first.")
        return

    print("Loading topology...")
    atom_array = parsing_biotite.load_structure_with_hydride(pdb_path, model=1, remove_solvent=False)
    
    # Identify atoms
    # Water oxygens: Resname WAT/HOH/TIP3 and Name O
    # Water hydrogens: Resname WAT/HOH/TIP3 and Name H1/H2
    # Protein: Everything else
    
    res_names = atom_array.res_name
    atom_names = atom_array.atom_name
    
    is_water_res = np.isin(res_names, ["WAT", "HOH", "TIP3", "SOL"])
    is_water_o = is_water_res & (atom_names == "O")
    is_water_h = is_water_res & np.isin(atom_names, ["H1", "H2", "H"])
    
    is_protein = ~is_water_res & ~np.isin(res_names, ["NA", "CL"]) # Exclude ions from "protein" group (handle separately or with protein)
    # Actually let's include ions with protein or separately.
    is_ion = np.isin(res_names, ["NA", "CL"])
    
    print(f"Atoms: {len(atom_array)}")
    print(f"Protein atoms: {np.sum(is_protein)}")
    print(f"Water Oxygens: {np.sum(is_water_o)}")
    print(f"Water Hydrogens: {np.sum(is_water_h)}")
    
    # 1. HTML Visualization with py2Dmol
    # -----------------------------------------------
    print("\nGenerating HTML visualization...")
    
    # Define Styles
    # High alpha protein, low alpha water spheres
    custom_styles = [
        # Protein (cartoon + stick)
        ({'not': {'resn': ['WAT', 'HOH', 'NA', 'CL']}}, {'cartoon': {'color': 'spectrum'}, 'stick': {'radius': 0.15, 'colorscheme': 'pujm'}}),
        
        # Water Oxygens (Blue spheres, low opacity)
        ({'resn': ['WAT', 'HOH'], 'atom': 'O'}, {'sphere': {'radius': 0.5, 'color': 'blue', 'opacity': 0.3}}),
        
        # Hide Water Hydrogens? py2Dmol default is to show everything unless hidden.
        # We can style everything else hidden? Or just not style them?
        # If we specify styles, do others get hidden? 
        # save_trajectory_html logic applies a default style if custom_styles is None.
        # But if custom_styles is provided, it does `addStyle`. 
        # addStyle ADDS to existing? No, py2Dmol models start with no style usually?
        # Actually save_trajectory_html applies default style first if no custom styles?
        # My implementation:
        # if custom_styles:
        #    viewer.setStyle({ style: ... }) // Applies default to ALL
        #    viewer.addStyle(...) // Overrides/Adds
        # So water hydrogens will get the default 'cartoon' style? 
        # Cartoon on atoms doesn't do much.
        # But maybe we want to HIDE them explicitly.
        ({'resn': ['WAT', 'HOH'], 'atom': ['H1', 'H2']}, {'hidden': True}) 
        # Note: 'hidden': True might not be valid py2Dmol syntax directly in style dict?
        # py2Dmol style object keys are render types (sphere, stick).
        # To hide, we just DON'T style them, provided we didn't apply a global style.
        # BUT my implementation applies global style first!
        # `viewer.setStyle({ {style}: { color: 'spectrum' } });`
        # Cartoon style on single H atoms usually does nothing.
        # So it might be fine.
    ]
    
    # Actually, applying cartoon to water might be weird if py2Dmol tries to trace it.
    # But usually it ignores non-protein.
    
    output_html = "1uao_explicit_viz.html"
    save_trajectory_html(
        trajectory=traj_path,
        pdb_path=pdb_path,
        output_path=output_html,
        stride=2, # Stride for HTML
        style="cartoon", # Base style
        title="1UAO Explicit Solvent (50ps)",
        custom_styles=custom_styles
    )
    
    # 2. GIF Visualization (Matplotlib)
    # -----------------------------------------------
    print("\nGenerating GIF visualization...")
    output_gif = "1uao_explicit_movie.gif"
    
    reader = TrajectoryReader(traj_path)
    full_positions = reader.get_positions()
    
    stride = 2
    positions = full_positions[::stride]
    
    # Pre-calculate indices
    idx_prot = np.where(is_protein | is_ion)[0]
    idx_wat_o = np.where(is_water_o)[0]
    
    # Setup Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Bounds
    min_bound = np.min(positions, axis=(0, 1))
    max_bound = np.max(positions, axis=(0, 1))
    mid = (min_bound + max_bound) / 2
    max_range = np.max(max_bound - min_bound) / 2
    
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    ax.set_title("1UAO Explicit Solvent")
    
    # Initial Scatter
    # Protein: colorful or gray? Let's do orange for contrast
    scat_prot = ax.scatter(
        positions[0, idx_prot, 0], 
        positions[0, idx_prot, 1], 
        positions[0, idx_prot, 2],
        c='orange', s=20, alpha=1.0, label='Protein'
    )
    
    # Water O: Blue, low alpha
    scat_wat = ax.scatter(
        positions[0, idx_wat_o, 0],
        positions[0, idx_wat_o, 1],
        positions[0, idx_wat_o, 2],
        c='blue', s=10, alpha=0.15, label='Water' # 0.15 opacity
    )
    
    ax.legend()
    
    txt = fig.text(0.1, 0.9, f"0 / {len(positions)} frames")
    
    def update(frame_idx):
        pos = positions[frame_idx]
        
        # Update Protein
        scat_prot._offsets3d = (pos[idx_prot, 0], pos[idx_prot, 1], pos[idx_prot, 2])
        
        # Update Water
        scat_wat._offsets3d = (pos[idx_wat_o, 0], pos[idx_wat_o, 1], pos[idx_wat_o, 2])
        
        txt.set_text(f"Frame {frame_idx}")
        return scat_prot, scat_wat, txt
        
    ani = animation.FuncAnimation(fig, update, frames=len(positions), blit=False)
    
    writer = animation.PillowWriter(fps=15)
    ani.save(output_gif, writer=writer)
    plt.close(fig)
    print(f"GIF saved to {output_gif}")

if __name__ == "__main__":
    main()
