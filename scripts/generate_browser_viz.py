"""Generate browser visualization for Chignolin trajectory."""
import os
from prolix.visualization import save_trajectory_html

def main():
    traj_path = "chignolin_traj.array_record"
    pdb_path = "data/pdb/1UAO.pdb"
    output_path = "chignolin_viz.html"
    
    if not os.path.exists(traj_path):
        print("Trajectory not found. Run scripts/simulate_chignolin_gif.py first.")
        return
        
    print(f"Generating browser visualization in {output_path}...")
    
    save_trajectory_html(
        trajectory=traj_path,
        pdb_path=pdb_path,
        output_path=output_path,
        stride=2, # Skip frames to keep file size reasonable
        style="cartoon", # or stick
        title="Chignolin (1UAO) Trajectory"
    )
    
    print(f"Done! Open {output_path} in your browser.")
    
if __name__ == "__main__":
    main()
