"""Explicit solvation tools."""

from __future__ import annotations

import os
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from scipy.spatial.distance import cdist
from jax_md import util

Array = util.Array

# Constants from OpenMM (converted to Angstroms where appropriate)
# vdwRadiusPerSigma = 0.56123...
# tip3p waterRadius (sigma*0.56...) = 0.315... nm * 10 = 3.15 A (approx? No, sigma is ~3.15A)
# OpenMM: waterRadius = 0.315075 * 0.56123 = 0.1768 nm = 1.768 Angstroms
TIP3P_WATER_RADIUS = 1.768  # Angstroms


@dataclass
class WaterBox:
  """Container for a pre-equilibrated water box.

  Fields:
      positions: Cartesian coordinates (N_atoms, 3).
      box_size: Box vectors (3,).
  """

  positions: Array
  box_size: Array

  def get_oxygens(self) -> Array:
    """Return oxygen coordinates (N_waters, 3)."""
    return self.positions[0::3]

  def get_hydrogens_1(self) -> Array:
    """Return first hydrogen coordinates (N_waters, 3)."""
    return self.positions[1::3]

  def get_hydrogens_2(self) -> Array:
    """Return second hydrogen coordinates (N_waters, 3)."""
    return self.positions[2::3]

  def n_waters(self) -> int:
    """Return number of water molecules."""
    return len(self.positions) // 3


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
  # Go up: physics -> prolix -> src -> package_root
  package_root = os.path.abspath(os.path.join(current_dir, "../../../"))

  path = os.path.join(package_root, "data", "water_boxes", "tip3p.npz")

  if not os.path.exists(path):
    # Fallback for when installed as package?
    # Or checking if running from script in root
    if os.path.exists("data/water_boxes/tip3p.npz"):
      path = "data/water_boxes/tip3p.npz"
    else:
      msg = f"Could not find TIP3P water box at {path} or relative to CWD"
      raise FileNotFoundError(msg)

  box = _load_water_npz(path)

  # Unwrap waters to ensure they are whole before tiling
  pos = np.array(box.positions)
  size = np.array(box.box_size)
  n_waters = len(pos) // 3

  for i in range(n_waters):
    o = pos[3 * i]
    h1 = pos[3 * i + 1]
    h2 = pos[3 * i + 2]

    # MIC unwrap H1 and H2 relative to O
    d1 = h1 - o
    h1 = o + (d1 - size * np.round(d1 / size))
    d2 = h2 - o
    h2 = o + (d2 - size * np.round(d2 / size))

    pos[3 * i + 1] = h1
    pos[3 * i + 2] = h2

  box.positions = jnp.array(pos)
  return box


def _tile_water_box(water_box: WaterBox, target_box_size: np.ndarray) -> np.ndarray:
  """Replicates a water box to fill the target volume.

  Args:
      water_box: Prototype water box.
      target_box_size: Dimensions of the target volume (3,).

  Returns:
      Numpy array of tiled water positions (N_atoms, 3).
  """
  wb_pos = np.array(water_box.positions)
  wb_size = np.array(water_box.box_size)

  # Determine repetitions needed to cover the box plus buffer for partials
  n_reps_per_dim = np.ceil(target_box_size / wb_size).astype(int) + 1

  all_waters = []
  for i in range(-1, n_reps_per_dim[0] + 1):
    for j in range(-1, n_reps_per_dim[1] + 1):
      for k in range(-1, n_reps_per_dim[2] + 1):
        offset = np.array([i, j, k]) * wb_size
        pos = wb_pos + offset
        oxygens = pos[0::3]
        # Keep waters whose Oxygen is within [0, target_box_size)
        valid_mask = np.all((oxygens >= 0) & (oxygens < target_box_size), axis=1)
        valid_atoms_mask = np.repeat(valid_mask, 3)
        all_waters.append(pos[valid_atoms_mask])

  if not all_waters:
    msg = "No waters generated! Check box sizes."
    raise ValueError(msg)

  return np.concatenate(all_waters, axis=0)


def _deduplicate_waters(tiled_waters: np.ndarray, box_size: np.ndarray) -> np.ndarray:
  """Removes overlapping waters at periodic boundaries using minimum image convention.

  Args:
      tiled_waters: Candidate water positions.
      box_size: Periodic box dimensions.

  Returns:
      Deduplicated water positions.
  """
  tile_oxygens_pre = tiled_waters[0::3]
  n_waters_pre = len(tile_oxygens_pre)

  keep_water_mask = np.ones(n_waters_pre, dtype=bool)
  for wi in range(n_waters_pre):
    if not keep_water_mask[wi]:
      continue
    # O(N^2) but typically small set of candidates for solvation setup
    for wj in range(wi + 1, n_waters_pre):
      if not keep_water_mask[wj]:
        continue
      # Minimum image distance check (O-O clash)
      delta = tile_oxygens_pre[wi] - tile_oxygens_pre[wj]
      delta = delta - box_size * np.round(delta / box_size)
      dist_pbc = np.linalg.norm(delta)
      if dist_pbc < 2.0:
        keep_water_mask[wj] = False

  # Filter and return
  keep_indices_dedup = np.where(keep_water_mask)[0]
  dedup_atom_indices = []
  for idx in keep_indices_dedup:
    dedup_atom_indices.extend([3 * idx, 3 * idx + 1, 3 * idx + 2])
  return tiled_waters[dedup_atom_indices]


def _prune_solute_clashes(
  tiled_waters: np.ndarray, solute_positions: np.ndarray, solute_radii: np.ndarray
) -> np.ndarray:
  """Removes waters that clash with solute atoms based on VDW radii.

  Args:
      tiled_waters: (3*N_w, 3) water atom positions.
      solute_positions: (N_s, 3) solute positions.
      solute_radii: (N_s,) atom radii.

  Returns:
      Pruned water positions.
  """
  tile_oxygens = tiled_waters[0::3]
  dists = cdist(tile_oxygens, solute_positions)  # (N_w, N_s)

  # Check condition: dist < (r_solute + r_water)
  radii_matrix = solute_radii[None, :]
  clashes = dists < (radii_matrix + TIP3P_WATER_RADIUS)
  clash_mask = np.any(clashes, axis=1)  # (N_w,)

  # Reconstruct atomic indices for non-clashing waters
  keep_indices = np.where(~clash_mask)[0]
  final_atom_indices = []
  for idx in keep_indices:
    final_atom_indices.extend([3 * idx, 3 * idx + 1, 3 * idx + 2])

  return tiled_waters[final_atom_indices]


def solvate(
  solute_positions: Array,
  solute_radii: Array,
  padding: float = 10.0,
  water_box: WaterBox | None = None,
  target_box_shape: Array | None = None,
) -> tuple[Array, Array]:
  r"""Add solvent molecules around a solute.

  Process:
  1.  **Box Size**: Determine target dimensions from solute bounds and padding.
  2.  **Center**: Re-center solute at $(\frac{1}{2} L, \frac{1}{2} L, \frac{1}{2} L)$.
  3.  **Tile**: Replicate `water_box` to fill the target volume.
  4.  **Deduplicate**: Remove overlapping waters at periodic boundaries (O-O distance < 2.0 Å).
  5.  **Prune**: Remove waters within exclusion radius of any solute atom.
  6.  **Combine**: Merge solute and remaining solvent coordinates.

  Args:
      solute_positions: (N_solute, 3) coordinates.
      solute_radii: (N_solute,) VDW radii for overlap check.
      padding: Buffer distance (Å).
      water_box: Prototype water box. Loads TIP3P if None.
      target_box_shape: Explicit box size override (3,).

  Returns:
      (positions, water_indices, box_size).
  """
  if water_box is None:
    water_box = load_tip3p_box()

  # 1. Box Size Determination
  min_coords = jnp.min(solute_positions, axis=0)
  max_coords = jnp.max(solute_positions, axis=0)

  if target_box_shape is not None:
    target_box_size = jnp.array(target_box_shape)
  else:
    target_box_size = (max_coords - min_coords) + 2 * padding

  # 2. Re-center Solute
  center = (max_coords + min_coords) / 2
  box_center = target_box_size / 2
  shift = box_center - center
  centered_solute = solute_positions + shift

  # 3. Solvent Placement Pipeline (using Numpy for setup logic)
  solute_np = np.array(centered_solute)
  radii_np = np.array(solute_radii)
  box_np = np.array(target_box_size)

  tiled_waters = _tile_water_box(water_box, box_np)
  deduped_waters = _deduplicate_waters(tiled_waters, box_np)
  final_waters = _prune_solute_clashes(deduped_waters, solute_np, radii_np)

  # 4. Integrate results
  n_solute = centered_solute.shape[0]
  n_waters_mol = final_waters.shape[0] // 3

  combined_pos = jnp.concatenate([centered_solute, jnp.array(final_waters)])

  # 5. Final wrap to ensure everything is within [0, L) and whole
  wrapped_pos = fix_water_geometry(combined_pos, target_box_size, n_solute, n_waters_mol)

  return wrapped_pos, target_box_size


def add_ions(
  positions: Array,
  water_indices: Array,
  solute_charge: float,
  positive_ion_name: str = "NA",
  negative_ion_name: str = "CL",
  ionic_strength: float = 0.0,
  neutralize: bool = True,
  box_size: Array | None = None,
) -> tuple[Array, list[str], list[str]]:
  r"""Replace water molecules with ions to reach target concentration.

  Process:
  1.  **Count**: Determine $N_{pos}$ and $N_{neg}$ needed for neutralization and concentration.
  2.  **Select**: Randomly pick water molecules to replace.
  3.  **Place**: Replace water Oxygen with Ion and remove Hydrogens.
  4.  **Rename**: Generate residue and atom names for the solvent block.

  Args:
      positions: (N, 3) coordinates.
      water_indices: Absolute indices of water atoms.
      solute_charge: Total net charge of protein/solute.
      positive_ion_name: Label for cation (e.g., "NA").
      negative_ion_name: Label for anion (e.g., "CL").
      ionic_strength: Target concentration (Molar).
      neutralize: Whether to zero out net charge.
      box_size: (3,) dimensions (A).

  Returns:
      (new_positions, atom_names, res_names).
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
  r"""Fix broken water molecules by unwrapping hydrogens relative to oxygen.

  Process:
  1.  **Extract**: Identify O, H1, H2 for each water molecule.
  2.  **Unwrap**: $\vec{r}_{H, unwrapped} = \vec{r}_O + \text{MIC}(\vec{r}_H - \vec{r}_O)$.
  3.  **Wrap**: Map the whole molecule back into the box using $\text{mod}(\vec{r}_O, L)$.

  Notes:
  Ensures all H atoms are within bonding distance of their parent Oxygen across PBC.

  Args:
      positions: (N, 3) coordinates.
      box_size: (3,) dimensions.
      n_solute: Number of solute atoms.
      n_waters: Number of water molecules.

  Returns:
      Corrected positions array.
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
