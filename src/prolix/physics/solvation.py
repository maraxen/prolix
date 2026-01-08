"""Explicit solvation tools."""

from __future__ import annotations

import os
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax_md import util

Array = util.Array

# Constants from OpenMM (converted to Angstroms where appropriate)
# vdwRadiusPerSigma = 0.56123...
# tip3p waterRadius (sigma*0.56...) = 0.315... nm * 10 = 3.15 A (approx? No, sigma is ~3.15A)
# OpenMM: waterRadius = 0.315075 * 0.56123 = 0.1768 nm = 1.768 Angstroms
TIP3P_WATER_RADIUS = 1.768  # Angstroms


@dataclass
class WaterBox:
  positions: Array  # (N_waters * 3, 3)
  box_size: Array  # (3,)


def _load_water_npz(path: str) -> WaterBox:
  """Loads a pre-converted water box from .npz format."""
  data = np.load(path)
  return WaterBox(positions=jnp.array(data["positions"]), box_size=jnp.array(data["box_size"]))


# Placeholder for _parse_water_pdb, as it's removed from load_tip3p_box
# If it's used elsewhere, it should be kept or defined.
# For now, assuming it's only used by load_tip3p_box and will be removed if not needed.


def load_tip3p_box() -> WaterBox:
  """Loads the pre-equilibrated TIP3P water box from .npz."""
  # Look for .npz relative to this file
  # File is in src/prolix/physics/solvation.py
  # Data is in data/water_boxes/tip3p.npz (relative to project root)

  current_dir = os.path.dirname(os.path.abspath(__file__))
  # Go up: physics -> prolix -> src -> project_root
  project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))

  path = os.path.join(project_root, "data", "water_boxes", "tip3p.npz")

  if not os.path.exists(path):
    # Fallback for when installed as package?
    # Or checking if running from script in root
    if os.path.exists("data/water_boxes/tip3p.npz"):
      path = "data/water_boxes/tip3p.npz"
    else:
      msg = f"Could not find TIP3P water box at {path} or relative to CWD"
      raise FileNotFoundError(msg)

  return _load_water_npz(path)


def solvate(
  solute_positions: Array,
  solute_radii: Array,  # VDW radii for exclusion
  padding: float = 10.0,
  water_box: WaterBox | None = None,
  target_box_shape: Array | None = None,
) -> tuple[Array, Array, Array]:
  """Adds solvent around solute.

  Args:
      solute_positions: (N_solute, 3)
      solute_radii: (N_solute,) VDW radii for exclusion
      padding: Padding in Angstroms
      water_box: Optional pre-loaded water box. If None, loads TIP3P.
      target_box_shape: Optional explicit box size (3,). If None, computed from padding.

  Returns:
      (combined_positions, water_indices, box_size)
      water_indices is simple range starting after solute

  """
  if water_box is None:
    water_box = load_tip3p_box()

  # 1. Determine box size
  min_coords = jnp.min(solute_positions, axis=0)
  max_coords = jnp.max(solute_positions, axis=0)

  if target_box_shape is not None:
    target_box_size = jnp.array(target_box_shape)
  else:
    # Initial box size based on solute + padding
    # OpenMM: max(2*radius+padding, 2*padding) logic or just bounds + padding
    # Let's use simple bounds + padding for now
    target_box_size = (max_coords - min_coords) + 2 * padding

  # 2. Re-center solute
  center = (max_coords + min_coords) / 2
  box_center = target_box_size / 2
  shift = box_center - center
  centered_solute = solute_positions + shift

  # 3. Tile water box
  # We need to fill target_box_size with water_box
  # Replicate water_box symmetrically around the center

  # Generate replicas
  # We need to loop or vectorize. Since this is setup code (not JIT), numpy is fine/better.
  wb_pos = np.array(water_box.positions)
  wb_size = np.array(water_box.box_size)
  target_box_np = np.array(target_box_size)

  # Calculate number of replicas needed in each direction from center
  # We need enough replicas to cover from 0 to target_box_size
  # Add +1 to ensure complete coverage at boundaries
  n_reps_per_dim = np.ceil(target_box_np / wb_size).astype(int) + 1

  all_waters = []

  # Water box contains complete molecules (O, H1, H2) in order
  # N_waters_in_box = len(wb_pos) // 3

  # Tile symmetrically: go from -n to +n in each dimension to ensure coverage
  for i in range(-1, n_reps_per_dim[0] + 1):
    for j in range(-1, n_reps_per_dim[1] + 1):
      for k in range(-1, n_reps_per_dim[2] + 1):
        offset = np.array([i, j, k]) * wb_size
        pos = wb_pos + offset
        # Filter waters that are outside the target box
        # Check if oxygen is inside [0, target_box_size) in all dimensions
        oxygens = pos[0::3]

        # Check BOTH lower and upper bounds
        valid_mask = np.all((oxygens >= 0) & (oxygens < target_box_np), axis=1)

        # If valid, keep the whole molecule
        # Expand mask (N_waters) -> (N_atoms)
        valid_atoms_mask = np.repeat(valid_mask, 3)
        all_waters.append(pos[valid_atoms_mask])

  if not all_waters:
    msg = "No waters generated! Check box sizes."
    raise ValueError(msg)

  tiled_waters = np.concatenate(all_waters, axis=0)
  # Shape (N_total_water_atoms, 3)

  # 4a. Deduplicate waters at periodic boundaries
  # Waters at x=0 and x=box_size are duplicates under PBC
  # Use minimum image convention to detect duplicates
  tile_oxygens_pre = tiled_waters[0::3]
  n_waters_pre = len(tile_oxygens_pre)

  # Find duplicate water pairs (O-O distance < 0.5 Å under PBC)
  from scipy.spatial.distance import cdist

  # Apply PBC to oxygen positions for comparison
  oxy_pbc = tile_oxygens_pre % target_box_np
  oxy_dists = cdist(oxy_pbc, oxy_pbc)
  np.fill_diagonal(oxy_dists, 100.0)  # Exclude self

  # Also check minimum image distances for edge cases
  # For waters near boundaries, check if they're duplicates
  keep_water_mask = np.ones(n_waters_pre, dtype=bool)
  for wi in range(n_waters_pre):
    if not keep_water_mask[wi]:
      continue
    for wj in range(wi + 1, n_waters_pre):
      if not keep_water_mask[wj]:
        continue
      # Minimum image distance
      delta = tile_oxygens_pre[wi] - tile_oxygens_pre[wj]
      delta = delta - target_box_np * np.round(delta / target_box_np)
      dist_pbc = np.linalg.norm(delta)
      if dist_pbc < 2.0:  # Remove overlapping waters (O-O clash)
        keep_water_mask[wj] = False

  n_removed = n_waters_pre - np.sum(keep_water_mask)
  if n_removed > 0:
    # Filter out duplicates
    keep_indices_dedup = np.where(keep_water_mask)[0]
    dedup_atom_indices = []
    for idx in keep_indices_dedup:
      dedup_atom_indices.extend([3 * idx, 3 * idx + 1, 3 * idx + 2])
    tiled_waters = tiled_waters[dedup_atom_indices]

  # 4b. Prune waters overlapping with solute
  # Check distances from each Oxygen to all solute atoms

  tile_oxygens = tiled_waters[0::3]
  # tiled_waters is [O1, H1, H2, O2, H1, H2 ...]

  # We need to check distance(O_water, Atom_solute) < (R_water + R_solute)
  # R_water = TIP3P_WATER_RADIUS

  # Using JAX for distance check might be faster if large
  # But for setup stability, standard numpy/scipy cdist is fine.
  # brute force (N_water * N_solute) can be large.
  # Use simple blocking or kd-tree if needed. For 1UAO (138 atoms) it's fast.

  from scipy.spatial.distance import cdist

  dists = cdist(tile_oxygens, np.array(centered_solute))  # (N_w, N_s)

  # Solute radii broadcasting
  # condition: dist < (r_solute + r_water)
  # dist - r_solute < r_water

  radii_matrix = np.array(solute_radii)[None, :]  # (1, N_s)
  # Check if any solute atom clashes
  clashes = dists < (radii_matrix + TIP3P_WATER_RADIUS)
  clash_mask = np.any(clashes, axis=1)  # (N_w,) True if clash

  # Keep non-clashing
  keep_mask = ~clash_mask
  keep_indices = np.where(keep_mask)[0]

  # Reconstruct atoms
  # keep_indices refers to oxygens (0, 1, 2...)
  # We need atom indices: 3*i, 3*i+1, 3*i+2

  final_water_indices = []
  for idx in keep_indices:
    final_water_indices.extend([3 * idx, 3 * idx + 1, 3 * idx + 2])

  final_waters = tiled_waters[final_water_indices]

  n_solute = centered_solute.shape[0]
  n_waters = final_waters.shape[0]
  water_indices = jnp.arange(n_solute, n_solute + n_waters)

  combined_pos = jnp.concatenate([centered_solute, jnp.array(final_waters)])

  return combined_pos, water_indices, target_box_size


def add_ions(
  positions: Array,  # (N_atoms, 3)
  water_indices: Array,  # Indices of water ATOMS (N_waters * 3)
  solute_charge: float,
  positive_ion_name: str = "NA",
  negative_ion_name: str = "CL",
  ionic_strength: float = 0.0,  # Molar
  neutralize: bool = True,
  box_size: Array | None = None,
) -> tuple[Array, list[str], list[str]]:
  """Replaces waters with ions to neutralize and/or reach ionic strength.

  Args:
      positions: JAX Array of positions
      water_indices: Array of indices of water atoms. Assumes waters are 3-atom molecules (O,H,H).
      solute_charge: Total charge of solute to neutralize.
      positive_ion_name: Residue/Atom name for positive ion.
      negative_ion_name: Residue/Atom name for negative ion.
      ionic_strength: Target ionic strength in Molar (mol/L).
      neutralize: Whether to add ions to neutralize solute charge.
      box_size: Box size in Angstroms (required for concentration calc).

  Returns:
      (new_positions, new_atom_names, new_res_names)
      Note: The returned arrays contain ONLY the modified water/ion part?
      No, we should probably return the Full arrays?
      But we don't have input names.
      We return (new_positions, ion_atom_names, ion_res_names) corresponding to the water/ion region.
      Caller must handle merging with solute topology.

  """
  if box_size is None and ionic_strength > 0:
    msg = "box_size required for ionic_strength"
    raise ValueError(msg)

  n_waters = len(water_indices) // 3

  n_pos = 0
  n_neg = 0

  # 1. Neutralization
  if neutralize:
    if solute_charge < -0.5:
      # Need positive ions
      n_pos += int(jnp.round(-solute_charge))
    elif solute_charge > 0.5:
      # Need negative ions
      n_neg += int(jnp.round(solute_charge))

  # 2. Ionic Strength
  if ionic_strength > 0 and box_size is not None:
    # Volume in Liters
    # Box size in A. 1 A = 1e-8 cm. 1 A^3 = 1e-24 cm^3 = 1e-27 L.
    # Wait. 1 cm = 1e8 A. 1 L = 1000 cm^3 = 1000 * (1e8)^3 A^3 = 1e27 A^3.
    # Volume (L) = Volume(A^3) * 1e-27.

    vol_A3 = box_size[0] * box_size[1] * box_size[2]
    vol_L = vol_A3 * 1.0e-27

    n_salt = int(jnp.round(ionic_strength * vol_L * 6.022e23))
    n_pos += n_salt
    n_neg += n_salt

  total_ions = n_pos + n_neg
  if total_ions > n_waters:
    msg = f"Not enough waters ({n_waters}) to place {total_ions} ions!"
    raise ValueError(msg)

  if total_ions == 0:
    return positions, [], []

  # Select waters to replace
  # We select random waters.
  # water_indices has shape (3*N_waters,).
  # water molecules are at indices 0, 3, 6... relative to start of water block?
  # No, water_indices are absolute.
  # But we assume they are contiguous blocks of 3?
  # Yes, typical from solvate().

  # Helper to get water molecule indices
  # We take every 3rd index from water_indices array?
  # Or just range(n_waters)?
  # We need to pick n_waters indices out of 0..n_waters-1.

  rng = np.random.default_rng()
  replace_indices = rng.choice(n_waters, size=total_ions, replace=False)

  pos_indices = replace_indices[:n_pos]
  neg_indices = replace_indices[n_pos:]

  # Build new position array
  # We will delete the atoms of replaced waters (3 atoms each)
  # And add Ion atoms (1 atom each).
  # Actually, easiest is to keep the Oxygen position for the Ion, and remove Hydrogens.

  # Convert jax array to numpy for manipulation
  pos_np = np.array(positions)

  # Identify atoms to keep and their new identities
  # Default: Keep all, unless replaced.

  # Mask of atoms to REMOVE
  remove_mask = np.zeros(len(pos_np), dtype=bool)

  # Map for new names {atom_idx: (atom_name, res_name)}
  new_identities = {}  # For ions

  # waters start at water_indices[0] usually
  start_idx = int(water_indices[0])  # Assuming contiguous

  # Handle Positive Ions
  for w_idx in pos_indices:
    # Abs atom indices
    # w_idx is the i-th water (0-indexed relative to waters)
    # assuming waters are contiguous 3-atom blocks
    base = start_idx + w_idx * 3

    # Keep O (base), Remove H1 (base+1), H2 (base+2)
    # remove_mask[base] = False # Keep O
    remove_mask[base + 1] = True
    remove_mask[base + 2] = True

    # Update Identity
    new_identities[base] = (positive_ion_name, positive_ion_name)

  # Handle Negative Ions
  for w_idx in neg_indices:
    base = start_idx + w_idx * 3
    remove_mask[base + 1] = True
    remove_mask[base + 2] = True
    new_identities[base] = (negative_ion_name, negative_ion_name)

  # Construct new positions
  keep_mask = ~remove_mask
  new_pos = pos_np[keep_mask]

  # Reconstruct names?
  # We return lists of names for the *entire* system?
  # Or just for the waters/ions part?
  # Ideally checking signature -> returns (new_pos, atom_names, res_names)
  # But we don't have input names.
  # So we can only return the NAMES for the WATER/ION block?
  # The caller needs to stitch it with the Solute names.

  # Generating names for the water/ion block:
  # We iterate 0..n_waters.
  # If replaced by ion -> Add Ion names.
  # If water -> Add WAT/O/H/H names.

  final_atom_names = []
  final_res_names = []

  # We iterate in order of waters to maintain position alignment
  for w_idx in range(n_waters):
    base = start_idx + w_idx * 3

    if base in new_identities:
      # It's an ion
      aname, rname = new_identities[base]
      final_atom_names.append(aname)
      final_res_names.append(rname)
    else:
      # It's a water
      final_atom_names.extend(["O", "H1", "H2"])
      final_res_names.extend(["WAT", "WAT", "WAT"])  # Or HOH

  # And we assume the caller handles the solute part?
  # Yes. The caller passed water_indices. The return values are implicitly for the *modified water Block*?

  # Wait. new_pos includes Solute (if input positions did).
  # If we return partial names, length mismatch.
  # We should return partial positions for the water block?
  # No, returning full positions is good.
  # But names...
  # We can't generate names for solute.
  # So we returns names ONLY for the waters/ions.

  return jnp.array(new_pos), final_atom_names, final_res_names


def fix_water_geometry(
  positions: Array,
  box_size: Array,
  n_solute: int,
  n_waters: int,
) -> Array:
  """Unwraps water hydrogens relative to oxygens to fix broken molecules.

  And then re-wraps the whole molecule into the box.

  Args:
      positions: (N_atoms, 3)
      box_size: (3,)
      n_solute: Number of solute atoms
      n_waters: Number of waters (following solute)

  Returns:
      Fixed positions.
  """
  box_arr = jnp.array(box_size)

  solute_pos = positions[:n_solute]
  waters = positions[n_solute:].reshape(n_waters, 3, 3)

  O = waters[:, 0, :]
  H1 = waters[:, 1, :]
  H2 = waters[:, 2, :]

  # Unwrap H relative to O
  # d = H - O
  # d -= box * round(d/box)
  # H_new = O + d

  d1 = H1 - O
  d1 = d1 - box_arr * jnp.round(d1 / box_arr)
  H1_unwrapped = O + d1

  d2 = H2 - O
  d2 = d2 - box_arr * jnp.round(d2 / box_arr)
  H2_unwrapped = O + d2

  # Now molecules are whole.
  # Wrap them back to box (keeping them whole)

  O_wrapped = jnp.mod(O, box_arr)
  wrap_disp = O_wrapped - O

  H1_final = H1_unwrapped + wrap_disp
  H2_final = H2_unwrapped + wrap_disp

  waters_final = jnp.stack([O_wrapped, H1_final, H2_final], axis=1).reshape(-1, 3)

  return jnp.concatenate([solute_pos, waters_final], axis=0)


def wrap_solvated_molecules(
  positions: Array,
  box_size: Array,
  n_solute: int,
  n_waters: int,
) -> Array:
  """Wraps positions with molecule-aware periodic boundary conditions.

  Assumes molecules are already whole (hydrogens close to oxygens).
  """
  return fix_water_geometry(positions, box_size, n_solute, n_waters)
