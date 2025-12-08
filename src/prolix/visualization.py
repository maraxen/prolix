"""Visualization tools for prolix trajectories.

This module provides tools for visualizing molecular dynamics trajectories
saved in array_record format using py2Dmol for 2D protein visualization
in Jupyter notebooks and Google Colab.

Example:
    >>> from prolix.visualization import visualize_trajectory
    >>> viewer = visualize_trajectory("trajectory.array_record", pdb_path="protein.pdb")

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Iterator, Sequence

import msgpack_numpy as m
import numpy as np

from prolix.simulate import SimulationState

if TYPE_CHECKING:
  from pathlib import Path

# Try importing array_record
try:
  from array_record.python.array_record_module import ArrayRecordReader
except ImportError:
  ArrayRecordReader = None  # type: ignore[assignment, misc]

# Try importing py2Dmol
try:
  import py2Dmol

  HAS_PY2DMOL = True
except ImportError:
  py2Dmol = None  # type: ignore[assignment]
  HAS_PY2DMOL = False

# Try importing gemmi for PDB parsing
try:
  import gemmi

  HAS_GEMMI = True
except ImportError:
  gemmi = None  # type: ignore[assignment]
  HAS_GEMMI = False

# Try importing matplotlib for animation export
try:
  import matplotlib.pyplot as plt
  import matplotlib.animation as animation
  from matplotlib.collections import LineCollection
  HAS_MATPLOTLIB = True
except ImportError:
  HAS_MATPLOTLIB = False

m.patch()

logger = logging.getLogger(__name__)


class TrajectoryReader:
  """Iterator for reading array_record trajectory files.

  Example:
      >>> reader = TrajectoryReader("trajectory.array_record")
      >>> for state in reader:
      ...     print(state.positions.shape)
      >>> reader.close()

      Or as a context manager:
      >>> with TrajectoryReader("trajectory.array_record") as reader:
      ...     for state in reader:
      ...         process(state)

  """

  def __init__(self, path: str | Path) -> None:
    """Initialize the reader.

    Args:
        path: Path to the array_record file.

    Raises:
        ImportError: If array_record is not installed.

    """
    if ArrayRecordReader is None:
      msg = (
        "array_record is not installed. "
        "Install it with: pip install array_record"
      )
      raise ImportError(msg)

    self.path = str(path)
    self.reader = ArrayRecordReader(self.path)
    self._num_records: int | None = None
    self._closed = False

  def __len__(self) -> int:
    """Return the number of frames in the trajectory."""
    if self._num_records is None:
      self._num_records = self.reader.num_records()
    return self._num_records

  def __iter__(self) -> Iterator[SimulationState]:
    """Iterate over all frames in the trajectory."""
    for i in range(len(self)):
      yield self[i]

  def __getitem__(self, idx: int) -> SimulationState:
    """Get a specific frame by index.

    Args:
        idx: Frame index (0-based).

    Returns:
        SimulationState for the frame.

    Raises:
        IndexError: If idx is out of range.

    """
    if idx < 0:
      idx = len(self) + idx
    if idx < 0 or idx >= len(self):
      msg = f"Index {idx} out of range for trajectory with {len(self)} frames"
      raise IndexError(msg)

    record = self.reader.read([idx])[0]
    return SimulationState.from_array_record(record)

  def close(self) -> None:
    """Close the reader."""
    if not self._closed:
      self.reader.close()
      self._closed = True

  def __enter__(self) -> "TrajectoryReader":
    """Context manager entry."""
    return self

  def __exit__(self, *args: object) -> None:
    """Context manager exit."""
    self.close()

  def __del__(self) -> None:
    """Destructor to ensure cleanup."""
    self.close()


def _extract_atom_info_from_pdb(
  pdb_path: str,
  chains: list[str] | None = None,
) -> tuple[np.ndarray, list[str], list[str]]:
  """Extract C-alpha positions and metadata from a PDB file.

  Args:
      pdb_path: Path to the PDB file.
      chains: Optional list of chains to include.

  Returns:
      Tuple of (ca_indices, chain_ids, atom_types).

  Raises:
      ImportError: If gemmi is not installed.

  """
  if not HAS_GEMMI:
    msg = "gemmi is not installed. Install it with: pip install gemmi"
    raise ImportError(msg)

  structure = gemmi.read_structure(pdb_path)
  model = structure[0]

  ca_indices: list[int] = []
  chain_ids: list[str] = []
  atom_types: list[str] = []

  atom_idx = 0
  for chain in model:
    if chains is not None and chain.name not in chains:
      for residue in chain:
        atom_idx += len(residue)
      continue

    for residue in chain:
      residue_info = gemmi.find_tabulated_residue(residue.name)
      if residue_info.is_amino_acid():
        # Find CA atom
        for atom in residue:
          if atom.name == "CA":
            ca_indices.append(atom_idx)
            chain_ids.append(chain.name)
            atom_types.append("P")  # Protein
            break
          atom_idx += 1
        else:
          atom_idx += len(residue) - len([a for a in residue if a.name == "CA"])
      elif residue_info.is_nucleic_acid():
        # Find C4' atom for nucleic acids
        for atom in residue:
          if atom.name == "C4'":
            ca_indices.append(atom_idx)
            chain_ids.append(chain.name)
            # 'D' for DNA, 'R' for RNA
            atom_types.append("D" if "D" in residue.name else "R")
            break
          atom_idx += 1
        else:
          atom_idx += len(residue) - len([a for a in residue if a.name == "C4'"])
      else:
        # Ligand - use all heavy atoms
        for atom in residue:
          if atom.element.name != "H":
            ca_indices.append(atom_idx)
            chain_ids.append(chain.name)
            atom_types.append("L")  # Ligand
          atom_idx += 1

  return np.array(ca_indices), chain_ids, atom_types


def visualize_trajectory(
  trajectory_path: str | Path,
  pdb_path: str | Path | None = None,
  ca_indices: np.ndarray | Sequence[int] | None = None,
  chains: list[str] | None = None,
  atom_types: list[str] | None = None,
  frame_stride: int = 1,
  max_frames: int | None = None,
  show_progress: bool = True,
  **viewer_kwargs: object,
) -> "py2Dmol.view":
  """Visualize a trajectory from an array_record file using py2Dmol.

  This function creates an interactive 2D visualization of a molecular dynamics
  trajectory that can be displayed in Jupyter notebooks or Google Colab.

  Args:
      trajectory_path: Path to the .array_record trajectory file.
      pdb_path: Optional path to reference PDB for extracting atom metadata.
          If provided, C-alpha positions, chain IDs, and atom types will be
          extracted automatically.
      ca_indices: Indices of C-alpha or representative atoms to visualize.
          Required if pdb_path is not provided.
      chains: Optional list of chain identifiers for each atom.
          Extracted from PDB if pdb_path is provided.
      atom_types: Optional list of atom types ('P', 'D', 'R', 'L').
          Extracted from PDB if pdb_path is provided.
      frame_stride: Only visualize every Nth frame. Default 1 (all frames).
      max_frames: Maximum number of frames to visualize. Default None (all).
      show_progress: Whether to print progress info. Default True.
      **viewer_kwargs: Additional kwargs passed to py2dmol.View() constructor.
          Common options: size=(width, height), color='chain'|'rainbow',
          shadow=True, outline=True, width=3.0, rotate=False.

  Returns:
      py2dmol.View object. Display in notebook by returning it or calling show().

  Raises:
      ImportError: If py2Dmol is not installed.
      ValueError: If neither pdb_path nor ca_indices is provided.

  Example:
      >>> viewer = visualize_trajectory(
      ...     "trajectory.array_record",
      ...     pdb_path="protein.pdb",
      ...     frame_stride=10,
      ...     size=(800, 600)
      ... )

  """
  if not HAS_PY2DMOL:
    msg = (
      "py2Dmol is not installed. "
      "Install it with: pip install py2Dmol[viz] or uv add py2Dmol"
    )
    raise ImportError(msg)

  # Extract atom info from PDB if provided
  if pdb_path is not None:
    pdb_ca_indices, pdb_chains, pdb_atom_types = _extract_atom_info_from_pdb(
      str(pdb_path), chains
    )
    if ca_indices is None:
      ca_indices = pdb_ca_indices
    if chains is None:
      chains = pdb_chains
    if atom_types is None:
      atom_types = pdb_atom_types
  elif ca_indices is None:
    msg = "Either pdb_path or ca_indices must be provided"
    raise ValueError(msg)

  ca_indices = np.asarray(ca_indices)

  # Create viewer
  viewer = py2Dmol.view(**viewer_kwargs)

  # Read trajectory
  with TrajectoryReader(trajectory_path) as reader:
    n_frames = len(reader)
    frames_to_viz = list(range(0, n_frames, frame_stride))
    if max_frames is not None:
      frames_to_viz = frames_to_viz[:max_frames]

    if show_progress:
      logger.info(
        "Visualizing %d frames from trajectory with %d total frames",
        len(frames_to_viz),
        n_frames,
      )

    for i, frame_idx in enumerate(frames_to_viz):
      state = reader[frame_idx]

      # Extract C-alpha coordinates
      coords = np.asarray(state.positions)
      if ca_indices.max() >= len(coords):
        msg = (
          f"C-alpha index {ca_indices.max()} out of range "
          f"for frame with {len(coords)} atoms"
        )
        raise ValueError(msg)

      ca_coords = coords[ca_indices]

      # Use pLDDT-like scores if available, otherwise use constant
      if state.potential_energy is not None:
        # Normalize energy to 0-100 range for color
        # Lower energy -> higher "confidence"
        pe = float(np.asarray(state.potential_energy))
        plddt = np.full(len(ca_coords), max(0, min(100, 100 - abs(pe) / 10)))
      else:
        plddt = np.full(len(ca_coords), 70.0)

      # Add frame to viewer
      viewer.add(
        ca_coords,
        plddt,
        chains,
        atom_types,
        new_traj=(i == 0),
        trajectory_name="MD Trajectory",
      )

  return viewer


def view_frame(
  positions: np.ndarray,
  ca_indices: np.ndarray | Sequence[int] | None = None,
  chains: list[str] | None = None,
  atom_types: list[str] | None = None,
  plddt: np.ndarray | None = None,
  pdb_path: str | Path | None = None,
  **viewer_kwargs: object,
) -> "py2Dmol.view":
  """Visualize a single frame/structure using py2Dmol.

  Args:
      positions: (N, 3) array of atom positions.
      ca_indices: Indices of C-alpha atoms to visualize.
          If None and pdb_path is provided, extracted from PDB.
          If None and pdb_path is None, uses all atoms.
      chains: List of chain IDs for each visualized atom.
      atom_types: List of atom types ('P', 'D', 'R', 'L').
      plddt: Array of pLDDT-like confidence scores (0-100).
      pdb_path: Optional PDB path to extract atom metadata from.
      **viewer_kwargs: Passed to py2dmol.View() constructor.

  Returns:
      py2dmol.View object.

  """
  if not HAS_PY2DMOL:
    msg = (
      "py2Dmol is not installed. "
      "Install it with: pip install py2Dmol[viz] or uv add py2Dmol"
    )
    raise ImportError(msg)

  # Extract info from PDB if provided
  if pdb_path is not None:
    pdb_ca_indices, pdb_chains, pdb_atom_types = _extract_atom_info_from_pdb(
      str(pdb_path)
    )
    if ca_indices is None:
      ca_indices = pdb_ca_indices
    if chains is None:
      chains = pdb_chains
    if atom_types is None:
      atom_types = pdb_atom_types

  positions = np.asarray(positions)

  if ca_indices is not None:
    ca_indices = np.asarray(ca_indices)
    coords = positions[ca_indices]
  else:
    coords = positions
    ca_indices = np.arange(len(positions))

  n_atoms = len(coords)

  if chains is None:
    chains = ["A"] * n_atoms
  if atom_types is None:
    atom_types = ["P"] * n_atoms
  if plddt is None:
    plddt = np.full(n_atoms, 70.0)

  viewer = py2Dmol.view(**viewer_kwargs)
  viewer.add(coords, plddt, chains, atom_types)

  return viewer


def view_pdb(
  pdb_path: str | Path,
  chains: list[str] | None = None,
  **viewer_kwargs: object,
) -> "py2Dmol.view":
  """Visualize a PDB file using py2Dmol.

  This is a simple wrapper around py2Dmol's add_pdb method.

  Args:
      pdb_path: Path to the PDB or CIF file.
      chains: Optional list of chains to include.
      **viewer_kwargs: Passed to py2dmol.View() constructor.

  Returns:
      py2dmol.View object.

  """
  if not HAS_PY2DMOL:
    msg = (
      "py2Dmol is not installed. "
      "Install it with: pip install py2Dmol[viz] or uv add py2Dmol"
    )
    raise ImportError(msg)

  viewer = py2Dmol.view(**viewer_kwargs)
  viewer.add_pdb(str(pdb_path), chains=chains)
  return viewer


def animate_trajectory(
  trajectory_path: str | Path,
  output_path: str | Path,
  pdb_path: str | Path | None = None,
  ca_indices: np.ndarray | Sequence[int] | None = None,
  chains: list[str] | None = None,
  frame_stride: int = 1,
  max_frames: int | None = None,
  fps: int = 20,
  dpi: int = 100,
  title: str = "MD Trajectory",
) -> None:
  """Create a GIF animation of the trajectory using matplotlib.

  This provides a way to export visualizations programmatically without
  needing a browser or py2Dmol.

  Args:
      trajectory_path: Path to array_record file.
      output_path: Path to save the GIF file.
      pdb_path: Path to PDB structure (for connectivity).
      ca_indices: Indices of atoms to visualize (usually C-alpha).
      chains: Chain IDs for atoms (for coloring).
      frame_stride: Stride for frames.
      max_frames: Max frames to render.
      fps: Frames per second for the GIF.
      dpi: output DPI.
      title: Animation title.

  """
  if not HAS_MATPLOTLIB:
    msg = "matplotlib is not installed. Install with: pip install matplotlib"
    raise ImportError(msg)

  # Extract atom info
  if pdb_path is not None:
    pdb_ca_indices, pdb_chains, _ = _extract_atom_info_from_pdb(
      str(pdb_path), chains
    )
    if ca_indices is None:
      ca_indices = pdb_ca_indices
    if chains is None:
      chains = pdb_chains
  elif ca_indices is None:
    msg = "Either pdb_path or ca_indices must be provided"
    raise ValueError(msg)

  ca_indices = np.asarray(ca_indices)
  if chains is None:
    chains = ["A"] * len(ca_indices)

  # Setup plot
  fig = plt.figure(figsize=(8, 6), dpi=dpi)
  ax = fig.add_subplot(111)
  ax.set_aspect("equal")
  ax.set_title(title)
  ax.grid(False)
  ax.set_axis_off()

  # prepare colors
  unique_chains = sorted(list(set(chains)))
  chain_map = {c: i for i, c in enumerate(unique_chains)}
  colors = plt.cm.tab10(np.linspace(0, 1, len(unique_chains)))
  atom_colors = [colors[chain_map[c]] for c in chains]

  # Read trajectory first to get limits
  with TrajectoryReader(trajectory_path) as reader:
    states = []
    n_frames = len(reader)
    indices = list(range(0, n_frames, frame_stride))
    if max_frames:
      indices = indices[:max_frames]
    
    logger.info("loading %d frames for animation logic...", len(indices))
    all_coords = []
    for i in indices:
      s = reader[i]
      pos = np.asarray(s.positions)[ca_indices]
      # Center
      pos = pos - np.mean(pos, axis=0)
      states.append(pos)
      all_coords.append(pos)
    
    # Calculate bounds
    all_coords_concat = np.concatenate(all_coords, axis=0)
    min_xyz = np.min(all_coords_concat, axis=0)
    max_xyz = np.max(all_coords_concat, axis=0)
    
    # Padding
    pad = 5.0
    ax.set_xlim(min_xyz[0] - pad, max_xyz[0] + pad)
    ax.set_ylim(min_xyz[1] - pad, max_xyz[1] + pad)

  # Plot elements
  lines = []
  
  # Group indices by chain to draw connected lines
  current_chain = None
  current_segment = []
  segments = [] # List of (start, end) tuples of *indices in ca_indices*
  segment_colors = []

  for i, chain in enumerate(chains):
    if chain != current_chain:
      if len(current_segment) > 1:
        # add segments
        for j in range(len(current_segment) - 1):
           segments.append((current_segment[j], current_segment[j+1]))
           segment_colors.append(atom_colors[current_segment[j]])
      current_chain = chain
      current_segment = [i]
    else:
      # Check distance to avoid long bonds across PBC or breaks
      # Here assuming simpler "connected if sequential" logic for now
      current_segment.append(i)
  
  # Add last segment
  if len(current_segment) > 1:
    for j in range(len(current_segment) - 1):
        segments.append((current_segment[j], current_segment[j+1]))
        segment_colors.append(atom_colors[current_segment[j]])

  lc = LineCollection([], linewidths=2)
  ax.add_collection(lc)
  
  # Scatter for C-alpha - Init with first frame to establish colors
  # We use the projected coordinates of the first frame
  coords_0 = states[0]
  # Apply projection logic (angle=0) -> x=x, y=y
  projected_0 = coords_0[:, [0, 1]]
  
  scat = ax.scatter(projected_0[:, 0], projected_0[:, 1], s=20, c=atom_colors, alpha=0.7)

  timestamp = ax.text(0.02, 0.95, "", transform=ax.transAxes)

  def init():
    # If we clear segments and offsets, blit is fine
    lc.set_segments([])
    # scat.set_offsets(np.empty((0, 2))) # This might cause issues if color array persists? 
    # Actually, let's just letting update handle it and return artists
    return lc, scat, timestamp

  def update(frame_idx):
    coords = states[frame_idx]
    
    # Compute 2D projection (just XY for now, maybe add rotation later)
    # Simple rotation around Y
    angle = frame_idx * 0.05
    c, s = np.cos(angle), np.sin(angle)
    # Rotate around Y axis: x' = x*c + z*s, z' = -x*s + z*c
    # We project to XY plane, so visualize x' and y
    
    xyz = coords
    x = xyz[:, 0] * c + xyz[:, 2] * s
    y = xyz[:, 1]
    
    projected = np.stack([x, y], axis=1)
    
    # Update lines
    segs = []
    cols = []
    valid_color_indices = []
    
    for k, (i, j) in enumerate(segments):
       p1 = projected[i]
       p2 = projected[j]
       # Draw line
       segs.append([p1, p2])
       valid_color_indices.append(k)

    if segs:
        lc.set_segments(segs)
        # set colors
        lc.set_color([segment_colors[k] for k in valid_color_indices])
    
    scat.set_offsets(projected)
    timestamp.set_text(f"Frame {frame_idx * frame_stride}")
    return lc, scat, timestamp

  anim = animation.FuncAnimation(
    fig, update, frames=len(states), init_func=init, blit=True
  )
  
  logger.info("Saving animation to %s", output_path)
  anim.save(str(output_path), writer="pillow", fps=fps)
  plt.close(fig)
