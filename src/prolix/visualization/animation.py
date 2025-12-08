"""Animation tools."""
from __future__ import annotations

import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional

from .trajectory import TrajectoryReader

logger = logging.getLogger(__name__)

def animate_trajectory(
    trajectory_path: str,
    output_path: str,
    pdb_path: Optional[str] = None,
    frame_stride: int = 1,
    fps: int = 15,
    title: str = "Trajectory"
):
    """Create a GIF/Video animation of the trajectory.
    
    Args:
        trajectory_path: Path to .array_record file
        output_path: Output filename (e.g. .gif or .mp4)
        pdb_path: Optional PDB file for topology/coloring (currently unused for topology)
        frame_stride: Skip frames
        fps: Frames per second
        title: Plot title
    """
    logger.info(f"Loading trajectory from {trajectory_path}")
    reader = TrajectoryReader(trajectory_path)
    
    # Load all positions (careful with memory for huge files, but safe for typical demos)
    # If huge, should read on demand. checking size first?
    # For now, load all, but stride.
    
    full_positions = reader.get_positions() # (T, N, 3)
    positions = full_positions[::frame_stride]
    
    if len(positions) == 0:
        raise ValueError("No frames found or stride too large.")
        
    n_frames, n_atoms, _ = positions.shape
    logger.info(f"Animating {n_frames} frames (stride={frame_stride})...")
    
    # Setup plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate bounds for consistent view
    # Center on mean position of first frame to separate translation from internal motion?
    # Or just global bounds.
    
    min_bound = np.min(positions, axis=(0, 1))
    max_bound = np.max(positions, axis=(0, 1))
    mid = (min_bound + max_bound) / 2
    max_range = np.max(max_bound - min_bound) / 2
    
    # Draw first frame
    # We use a simple scatter plot
    # TODO: Color by element if PDB known?
    # For now, simple blue dots.
    
    scat = ax.scatter(positions[0, :, 0], positions[0, :, 1], positions[0, :, 2], 
                      c='blue', alpha=0.6, s=10)
    
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(title)
    
    txt = fig.text(0.1, 0.9, f"Time: 0.00 ns", transform=fig.transFigure)
    
    # Get times if available
    times = reader.get_time()[::frame_stride]
    
    def update(frame_idx):
        pos = positions[frame_idx]
        scat._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
        if frame_idx < len(times):
            txt.set_text(f"Time: {times[frame_idx]:.3f} ns")
        return scat, txt
        
    ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    
    # Save
    logger.info(f"Saving to {output_path}...")
    if output_path.endswith('.gif'):
        writer = animation.PillowWriter(fps=fps)
    else:
        # Require ffmpeg
        writer = animation.FFMpegWriter(fps=fps, extra_args=['-vcodec', 'libx264'])
        
    ani.save(output_path, writer=writer)
    plt.close(fig)
    logger.info("Done.")
