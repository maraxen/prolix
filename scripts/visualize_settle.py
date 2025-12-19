import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import os

def main():
    path = "1uao_explicit_settle_traj.npz"
    output_gif = "1uao_settle_movie.gif"
    
    if not os.path.exists(path):
        print(f"Trajectory {path} not found.")
        return

    print(f"Loading {path}...")
    data = np.load(path)
    positions = data["positions"] # (n_frames, n_atoms, 3)
    n_protein = int(data["n_protein_atoms"])
    n_waters = int(data["n_waters"])
    
    n_frames, n_atoms, _ = positions.shape
    print(f"Loaded {n_frames} frames, {n_atoms} atoms ({n_protein} protein, {n_waters} waters)")
    
    # Setup Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate bounds (centered on protein)
    # Use protein center of geometry
    prot_pos_0 = positions[0, :n_protein]
    center = np.mean(prot_pos_0, axis=0)
    
    # View window size (e.g. +/- 15 Angstroms)
    r = 15.0
    
    ax.set_xlim(center[0]-r, center[0]+r)
    ax.set_ylim(center[1]-r, center[1]+r)
    ax.set_zlim(center[2]-r, center[2]+r)
    
    ax.set_xlabel("X (A)")
    ax.set_ylabel("Y (A)")
    ax.set_zlabel("Z (A)")
    ax.set_title("1UAO Explicit Solvent (SETTLE)")
    
    # Plot Elements
    # 1. Protein CA trace? Or just all protein atoms as scatter?
    # All protein atoms (blue)
    prot_scat = ax.scatter([], [], [], c='blue', s=20, label='Protein', alpha=1.0)
    
    # 2. Water Oxygens
    # Indices: n_protein + 3*i
    water_O_indices = [n_protein + 3*i for i in range(n_waters)]
    water_scat = ax.scatter([], [], [], c='cyan', s=5, label='Water (O)', alpha=0.3)
    
    txt = fig.text(0.05, 0.95, "", transform=fig.transFigure, fontsize=12)
    
    ax.legend()
    
    def update(frame):
        pos = positions[frame]
        
        # Update Protein
        prot_p = pos[:n_protein]
        prot_scat._offsets3d = (prot_p[:,0], prot_p[:,1], prot_p[:,2])
        
        # Update Water Oxygens
        wat_p = pos[water_O_indices]
        water_scat._offsets3d = (wat_p[:,0], wat_p[:,1], wat_p[:,2])
        
        txt.set_text(f"Frame {frame}/{n_frames}")
        return prot_scat, water_scat, txt
        
    print(f"Rendering {n_frames} frames to {output_gif}...")
    ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    
    writer = animation.PillowWriter(fps=10)
    ani.save(output_gif, writer=writer)
    print("Done!")

if __name__ == "__main__":
    main()
