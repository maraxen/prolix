"""Flask-based server for the prolix trajectory viewer.

This server:
1. Serves the built webapp (from viewer/dist)
2. Provides API endpoints to read trajectory files
3. Converts array_record trajectories to JSON for browser consumption

Usage:
    python -m prolix.viewer_server trajectory.array_record --pdb protein.pdb
    
Or programmatically:
    from prolix.viewer_server import launch_viewer
    launch_viewer(trajectory_path="traj.array_record", pdb_path="protein.pdb")
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import traceback
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

# Try importing Flask
try:
  from flask import Flask, jsonify, request, send_from_directory
  HAS_FLASK = True
except ImportError:
  HAS_FLASK = False

# Try importing visualization module for TrajectoryReader
try:
  from prolix.visualization import TrajectoryReader
  HAS_READER = True
except ImportError:
  HAS_READER = False

logger = logging.getLogger(__name__)

# Find the viewer dist directory
# Path is: src/prolix/viewer_server.py -> ../../.. -> project root -> viewer/dist
VIEWER_DIST = Path(__file__).parent.parent.parent / "viewer" / "dist"
if not VIEWER_DIST.exists():
  # Fallback: Try from installed package location (editable install)
  VIEWER_DIST = Path(__file__).parent.parent.parent.parent / "viewer" / "dist"


def create_app(
  trajectory_path: str | None = None,
  pdb_path: str | None = None,
) -> "Flask":
  """Create the Flask app for serving the viewer.
  
  Args:
      trajectory_path: Path to trajectory file (array_record, json, etc.)
      pdb_path: Path to PDB structure file
      
  Returns:
      Flask app instance
  """
  if not HAS_FLASK:
    msg = "Flask is not installed. Install with: pip install flask"
    raise ImportError(msg)

  app = Flask(__name__, static_folder=str(VIEWER_DIST))
  
  # Store paths as app config
  app.config["TRAJECTORY_PATH"] = trajectory_path
  app.config["PDB_PATH"] = pdb_path

  @app.route("/")
  def index():
    """Serve the main app."""
    return send_from_directory(VIEWER_DIST, "index.html")

  @app.route("/<path:path>")
  def static_files(path: str):
    """Serve static files."""
    return send_from_directory(VIEWER_DIST, path)

  @app.route("/api/config")
  def get_config():
    """Return the configured file paths."""
    return jsonify({
      "trajectoryPath": app.config["TRAJECTORY_PATH"],
      "pdbPath": app.config["PDB_PATH"],
    })

  @app.route("/api/pdb")
  def get_pdb():
    """Return the PDB file contents."""
    pdb = app.config["PDB_PATH"]
    if not pdb or not os.path.exists(pdb):
      return jsonify({"error": "PDB file not found"}), 404
    
    with open(pdb) as f:
      return f.read(), 200, {"Content-Type": "text/plain"}

  @app.route("/api/trajectory")
  def get_trajectory():
    """Return trajectory data as JSON with CA atoms only."""
    traj_path = app.config["TRAJECTORY_PATH"]
    pdb_path = app.config["PDB_PATH"]
    
    if not traj_path or not os.path.exists(traj_path):
      return jsonify({"error": "Trajectory file not found"}), 404
    
    if not HAS_READER:
      return jsonify({"error": "TrajectoryReader not available"}), 500
    
    # Parse CA indices from PDB
    ca_indices = []
    if pdb_path and os.path.exists(pdb_path):
      with open(pdb_path) as f:
        for line in f:
          if line.startswith("ATOM") or line.startswith("HETATM"):
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            # Skip water
            if res_name in ("HOH", "WAT"):
              continue
            # For proteins, only CA atoms
            if atom_name == "CA":
              # The index is the current count (0-based)
              atom_idx = int(line[6:11].strip()) - 1  # PDB is 1-indexed
              ca_indices.append(len(ca_indices))  # Use sequential index in positions array
      
      # Actually we need to track which positions correspond to CA
      # Re-parse to get the actual position indices
      ca_indices = []
      atom_counter = 0
      with open(pdb_path) as f:
        for line in f:
          # Only use MODEL 1
          if line.startswith("ENDMDL"):
            break
          if line.startswith("ATOM") or line.startswith("HETATM"):
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            # Skip water
            if res_name in ("HOH", "WAT"):
              continue
            if atom_name == "CA":
              ca_indices.append(atom_counter)
            atom_counter += 1
    
    logger.info(f"Found {len(ca_indices)} CA atoms in PDB")
    
    try:
      frames = []
      reader = TrajectoryReader(traj_path)
      for state in reader:
        positions = np.asarray(state.positions)
        # Filter to CA only if we have indices
        if ca_indices:
          ca_positions = positions[ca_indices].tolist()
        else:
          ca_positions = positions.tolist()
        
        frames.append({
          "positions": ca_positions,
          "time_ns": float(np.asarray(state.time_ns)) if state.time_ns is not None else None,
          "potential_energy": float(np.asarray(state.potential_energy)) if state.potential_energy is not None else None,
        })
      
      return jsonify({"frames": frames, "num_ca_atoms": len(ca_indices) if ca_indices else len(frames[0]["positions"]) if frames else 0})
    except Exception as e:
      logger.exception("Error reading trajectory")
      print(traceback.format_exc())
      return jsonify({"error": str(e)}), 500

  @app.route("/api/trajectory/<int:frame_idx>")
  def get_frame(frame_idx: int):
    """Return a single frame from the trajectory."""
    traj_path = app.config["TRAJECTORY_PATH"]
    if not traj_path or not os.path.exists(traj_path):
      return jsonify({"error": "Trajectory file not found"}), 404
    
    if not HAS_READER:
      return jsonify({"error": "TrajectoryReader not available"}), 500
    
    try:
      with TrajectoryReader(traj_path) as reader:
        if frame_idx >= len(reader) or frame_idx < 0:
          return jsonify({"error": "Frame index out of range"}), 400
        
        state = reader[frame_idx]
        return jsonify({
          "positions": np.asarray(state.positions).tolist(),
          "time_ns": float(np.asarray(state.time_ns)) if state.time_ns is not None else None,
          "potential_energy": float(np.asarray(state.potential_energy)) if state.potential_energy is not None else None,
        })
    except Exception as e:
      logger.exception("Error reading frame")
      return jsonify({"error": str(e)}), 500

  @app.route("/api/trajectory/info")
  def get_trajectory_info():
    """Return trajectory metadata."""
    traj_path = app.config["TRAJECTORY_PATH"]
    if not traj_path or not os.path.exists(traj_path):
      return jsonify({"error": "Trajectory file not found"}), 404
    
    if not HAS_READER:
      return jsonify({"error": "TrajectoryReader not available"}), 500
    
    try:
      with TrajectoryReader(traj_path) as reader:
        return jsonify({
          "numFrames": len(reader),
          "path": traj_path,
        })
    except Exception as e:
      return jsonify({"error": str(e)}), 500

  return app


def launch_viewer(
  trajectory_path: str | None = None,
  pdb_path: str | None = None,
  port: int = 5000,
  open_browser: bool = True,
  debug: bool = False,
) -> None:
  """Launch the viewer server.
  
  Args:
      trajectory_path: Path to trajectory file
      pdb_path: Path to PDB file
      port: Server port (default 5000)
      open_browser: Whether to open browser automatically
      debug: Enable Flask debug mode
  """
  app = create_app(trajectory_path, pdb_path)
  
  url = f"http://localhost:{port}"
  print(f"Prolix Trajectory Viewer running at {url}")
  
  if open_browser:
    webbrowser.open(url)
  
  app.run(host="0.0.0.0", port=port, debug=debug)


def main():
  """CLI entry point."""
  parser = argparse.ArgumentParser(
    description="Launch the Prolix Trajectory Viewer"
  )
  parser.add_argument(
    "trajectory",
    nargs="?",
    help="Path to trajectory file (array_record, json)",
  )
  parser.add_argument(
    "--pdb",
    help="Path to PDB structure file",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=5000,
    help="Server port (default: 5000)",
  )
  parser.add_argument(
    "--no-browser",
    action="store_true",
    help="Don't open browser automatically",
  )
  parser.add_argument(
    "--debug",
    action="store_true",
    help="Enable debug mode",
  )
  
  args = parser.parse_args()
  
  launch_viewer(
    trajectory_path=args.trajectory,
    pdb_path=args.pdb,
    port=args.port,
    open_browser=not args.no_browser,
    debug=args.debug,
  )


if __name__ == "__main__":
  main()
