"""Explicit solvation tools."""

from __future__ import annotations

import os
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from flax import struct
from jax_md import util
from scipy.spatial.distance import cdist

from prolix.physics.ion_params import get_ion_params
from prolix.physics.topology_merger import MergedTopology, merge_solvated_topology
from prolix.physics.water_models import WaterModelType, get_water_params

Array = util.Array

# Default water radius if model not specified (TIP3P)
TIP3P_WATER_RADIUS = 1.768  # Angstroms


@struct.dataclass
class SolventTopology:
    """Structured container for pre-parameterized solvent (waters + ions)."""
    positions: Array       # (N_solvent, 3)
    charges: Array         # (N_solvent,)
    sigmas: Array          # (N_solvent,)
    epsilons: Array        # (N_solvent,)
    masses: Array          # (N_solvent,)
    element_ids: Array     # (N_solvent,)
    is_hydrogen: Array     # (N_solvent,) bool
    water_indices: Array   # (N_waters, 3) relative to solvent start
    n_waters: int
    n_ions: int


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


def load_water_box(model_type: WaterModelType | str = WaterModelType.TIP3P) -> WaterBox:
  """Loads the pre-equilibrated water box for a specific model."""
  if isinstance(model_type, str):
      model_type = WaterModelType(model_type.lower())
  
  # Look for .npz relative to this file
  current_dir = os.path.dirname(os.path.abspath(__file__))
  package_root = os.path.abspath(os.path.join(current_dir, "../../../"))

  filename = f"{model_type.value}.npz"
  path = os.path.join(package_root, "data", "water_boxes", filename)

  if not os.path.exists(path):
    # Fallback for when installed as package?
    if os.path.exists(f"data/water_boxes/{filename}"):
      path = f"data/water_boxes/{filename}"
    else:
      msg = f"Could not find {model_type.value} water box at {path} or relative to CWD"
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
  """Removes overlapping waters at periodic boundaries using scipy.spatial.KDTree.

  Args:
      tiled_waters: Candidate water positions.
      box_size: Periodic box dimensions.

  Returns:
      Deduplicated water positions.
  """
  from scipy.spatial import KDTree
  
  tile_oxygens_pre = tiled_waters[0::3]
  n_waters_pre = len(tile_oxygens_pre)

  # Use KDTree to find all pairs within 2.0A
  # We use the box_size for periodic wrap (scipy KDTree supports this)
  tree = KDTree(tile_oxygens_pre, boxsize=box_size)
  
  # Find all self-pairs within 2.0A
  pairs = tree.query_pairs(r=2.0)
  
  keep_water_mask = np.ones(n_waters_pre, dtype=bool)
  for i, j in pairs:
    # If both are still marked, discard the second one
    if keep_water_mask[i]:
      keep_water_mask[j] = False

  # Filter and return
  keep_indices_dedup = np.where(keep_water_mask)[0]
  dedup_atom_indices = []
  for idx in keep_indices_dedup:
    dedup_atom_indices.extend([3 * idx, 3 * idx + 1, 3 * idx + 2])
  return tiled_waters[dedup_atom_indices]


def _prune_solute_clashes(
  tiled_waters: np.ndarray, solute_positions: np.ndarray, solute_radii: np.ndarray,
  water_radius: float = TIP3P_WATER_RADIUS
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

  # Check condition: dist < (r_solute + r_water + 1.0)
  # Adding 1.0 A margin to account for water hydrogen extent (~0.96 A)
  radii_matrix = solute_radii[None, :]
  clashes = dists < (radii_matrix + water_radius + 1.0)
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
  model_type: WaterModelType = WaterModelType.TIP3P,
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
      water_box: Prototype water box. Loads according to model_type if None.
      target_box_shape: Explicit box size override (3,).
      model_type: Supported water model (TIP3P, OPC3).

  Returns:
      (positions, water_indices, box_size).
  """
  if water_box is None:
    water_box = load_water_box(model_type)

  model_params = get_water_params(model_type)
  water_radius = model_params.water_radius

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
  final_waters = _prune_solute_clashes(deduped_waters, solute_np, radii_np, water_radius)

  # 4. Integrate results
  n_solute = centered_solute.shape[0]
  n_waters_mol = final_waters.shape[0] // 3

  combined_pos = jnp.concatenate([centered_solute, jnp.array(final_waters)])

  # 5. Final wrap to ensure everything is within [0, L) and whole
  water_indices = jnp.arange(n_solute, n_solute + 3 * n_waters_mol).reshape(-1, 3)
  wrapped_pos = fix_water_geometry_padded(combined_pos, water_indices, target_box_size)

  return wrapped_pos, target_box_size


def add_ions(
    positions: Array,
    water_indices: Array,
    total_charge: float,
    positive_ion_name: str = "NA",
    negative_ion_name: str = "CL",
    ionic_strength: float = 0.15,
    neutralize: bool = True,
    box_size: Array | None = None,
    model_type: WaterModelType = WaterModelType.TIP3P,
) -> tuple[Array, SolventTopology]:
    """Replaces random waters with ions to neutralize and set ionic strength.

    Returns:
        New positions and a SolventTopology containing ion/water parameters.
    """

    if box_size is None and ionic_strength > 0:
        raise ValueError("box_size required for ionic_strength")

    n_solute_atoms = len(positions) - len(water_indices)
    n_waters_initial = len(water_indices) // 3
    
    n_pos = 0
    n_neg = 0

    # 1. Neutralization
    if neutralize:
        if total_charge < -0.5:
            n_pos += int(jnp.round(-total_charge))
        elif total_charge > 0.5:
            n_neg += int(jnp.round(total_charge))

    # 2. Ionic Strength
    if ionic_strength > 0 and box_size is not None:
        vol_A3 = box_size[0] * box_size[1] * box_size[2]
        vol_L = vol_A3 * 1.0e-27
        n_salt = int(jnp.round(ionic_strength * vol_L * 6.022e23))
        n_pos += n_salt
        n_neg += n_salt

    total_ions = n_pos + n_neg
    if total_ions > n_waters_initial:
        raise ValueError(f"Not enough waters ({n_waters_initial}) to place {total_ions} ions!")

    # Select waters to replace
    rng = np.random.default_rng()
    replace_indices = rng.choice(n_waters_initial, size=total_ions, replace=False)
    
    pos_indices_set = set(replace_indices[:n_pos].tolist())
    neg_indices_set = set(replace_indices[n_pos:].tolist())

    # Parameters
    w_params = get_water_params(model_type)
    p_ion = get_ion_params(model_type, positive_ion_name)
    n_ion = get_ion_params(model_type, negative_ion_name)
    
    # Atomic numbers for monovalent ions
    ION_ELEMENTS = {"NA": 11, "CL": 17, "LI": 3, "K": 19, "RB": 37, "CS": 55, "F": 9, "BR": 35, "I": 53}
    p_elem = ION_ELEMENTS.get(positive_ion_name.upper(), 11)
    n_elem = ION_ELEMENTS.get(negative_ion_name.upper(), 17)

    solvent_pos = []
    solvent_charges = []
    solvent_sigmas = []
    solvent_epsilons = []
    solvent_masses = []
    solvent_elements = []
    solvent_is_h = []
    water_molecule_indices = []

    current_atom_idx = 0
    for w_idx in range(n_waters_initial):
        base = w_idx * 3
        o_pos = positions[n_solute_atoms + base]
        
        if w_idx in pos_indices_set:
            # Positive Ion
            solvent_pos.append(o_pos)
            solvent_charges.append(p_ion.charge)
            solvent_sigmas.append(p_ion.sigma)
            solvent_epsilons.append(p_ion.epsilon)
            solvent_masses.append(p_ion.mass)
            solvent_elements.append(p_elem)
            solvent_is_h.append(False)
            current_atom_idx += 1
        elif w_idx in neg_indices_set:
            # Negative Ion
            solvent_pos.append(o_pos)
            solvent_charges.append(n_ion.charge)
            solvent_sigmas.append(n_ion.sigma)
            solvent_epsilons.append(n_ion.epsilon)
            solvent_masses.append(n_ion.mass)
            solvent_elements.append(n_elem)
            solvent_is_h.append(False)
            current_atom_idx += 1
        else:
            # Keep as water
            h1_pos = positions[n_solute_atoms + base + 1]
            h2_pos = positions[n_solute_atoms + base + 2]
            solvent_pos.extend([o_pos, h1_pos, h2_pos])
            solvent_charges.extend([w_params.charge_O, w_params.charge_H, w_params.charge_H])
            solvent_sigmas.extend([w_params.sigma_O, 1e-6, 1e-6])
            solvent_epsilons.extend([w_params.epsilon_O, 0.0, 0.0])
            solvent_masses.extend([15.999, 1.008, 1.008])
            solvent_elements.extend([8, 1, 1])
            solvent_is_h.extend([False, True, True])
            
            # Record 3-atom water unit
            water_molecule_indices.append([
                current_atom_idx, current_atom_idx + 1, current_atom_idx + 2
            ])
            current_atom_idx += 3

    new_positions = jnp.concatenate([
        positions[:n_solute_atoms],
        jnp.array(solvent_pos)
    ], axis=0)

    solvent_topo = SolventTopology(
        positions=jnp.array(solvent_pos),
        charges=jnp.array(solvent_charges, dtype=jnp.float32),
        sigmas=jnp.array(solvent_sigmas, dtype=jnp.float32),
        epsilons=jnp.array(solvent_epsilons, dtype=jnp.float32),
        masses=jnp.array(solvent_masses, dtype=jnp.float32),
        element_ids=jnp.array(solvent_elements, dtype=jnp.int32),
        is_hydrogen=jnp.array(solvent_is_h, dtype=jnp.bool_),
        water_indices=jnp.array(water_molecule_indices, dtype=jnp.int32),
        n_waters=len(water_molecule_indices),
        n_ions=total_ions
    )

    return new_positions, solvent_topo


def fix_water_geometry(
  positions: Array,
  box_size: Array,
  n_solute: int,
  solvent_topo: SolventTopology,
) -> Array:
  r"""Fix broken water molecules by unwrapping hydrogens relative to oxygen."""
  solute_pos = positions[:n_solute]
  solvent_pos = positions[n_solute:]
  
  new_solvent = fix_water_geometry_padded(
      solvent_pos, solvent_topo.water_indices, box_size
  )
  return jnp.concatenate([solute_pos, new_solvent], axis=0)


def fix_water_geometry_padded(
    positions: Array,
    water_indices: Array,
    box_size: Array,
) -> Array:
    """Fix broken water molecules in a padded positions array."""
    box_arr = jnp.array(box_size)
    O = positions[water_indices[:, 0]]
    H1 = positions[water_indices[:, 1]]
    H2 = positions[water_indices[:, 2]]

    # Unwrap H relative to O
    d1 = H1 - O
    d1 = d1 - box_arr * jnp.round(d1 / box_arr)
    H1_unwrapped = O + d1

    d2 = H2 - O
    d2 = d2 - box_arr * jnp.round(d2 / box_arr)
    H2_unwrapped = O + d2

    # Wrap O
    O_wrapped = jnp.mod(O, box_arr)
    wrap_disp = O_wrapped - O

    H1_final = H1_unwrapped + wrap_disp
    H2_final = H2_unwrapped + wrap_disp

    # Re-insert into solvent positions
    new_positions = positions.at[water_indices[:, 0]].set(O_wrapped)
    new_positions = new_positions.at[water_indices[:, 1]].set(H1_final)
    new_positions = new_positions.at[water_indices[:, 2]].set(H2_final)

    return new_positions


def solvate_protein(
    protein: Protein,
    padding: float = 10.0,
    model_type: WaterModelType = WaterModelType.TIP3P,
    ionic_strength: float = 0.15,
    neutralize: bool = True,
    target_box_size: Array | None = None,
) -> MergedTopology:
    """High-level solvation pipeline for Protein objects."""
    # 1. Solvate positions
    mask = (protein.atom_mask > 0.5).flatten()
    pos_full = protein.coordinates.reshape(-1, 3)
    pos_real = pos_full[mask]
    
    radii_full = getattr(protein, "radii", None)
    if radii_full is None:
        radii_full = jnp.ones(len(pos_full)) * 1.5
    radii_real = radii_full[mask]
    
    min_coords = jnp.min(pos_real, axis=0)
    max_coords = jnp.max(pos_real, axis=0)
    
    if target_box_size is not None:
        box_size = jnp.array(target_box_size)
    else:
        box_size = (max_coords - min_coords) + 2 * padding
    
    center = (max_coords + min_coords) / 2
    box_center = box_size / 2
    shift = box_center - center
    
    # Shift real atoms, but keep ghost atoms at their original positions (0,0,0)
    # to avoid polluting the water box center.
    shift_mask = mask[:, None]
    pos_full_centered = jnp.where(shift_mask, pos_full + shift, pos_full)
    pos_real_centered = pos_real + shift
    
    water_box_obj = load_water_box(model_type)
    model_params = get_water_params(model_type)
    
    tiled_waters = _tile_water_box(water_box_obj, np.array(box_size))
    deduped_waters = _deduplicate_waters(tiled_waters, np.array(box_size))
    final_waters = _prune_solute_clashes(
        deduped_waters,
        np.array(pos_real_centered),
        np.array(radii_real),
        model_params.water_radius
    )
    
    pos_solv = jnp.concatenate([pos_full_centered, jnp.array(final_waters)])
    n_solute = len(pos_full)
    water_indices = jnp.arange(n_solute, len(pos_solv))
    
    # 2. Add ions (replaces some waters)
    charges = getattr(protein, "charges", None)
    total_charge = float(jnp.sum(charges)) if charges is not None else 0.0
    
    pos_ionized, solvent_topo = add_ions(
        pos_solv,
        water_indices,
        total_charge,
        ionic_strength=ionic_strength,
        neutralize=neutralize,
        box_size=box_size,
        model_type=model_type
    )
    
    # 3. Final molecule-aware wrapping
    pos_final = fix_water_geometry(pos_ionized, box_size, n_solute, solvent_topo)
    
    # 4. Merge topologies
    # solvent_topo now contains ALL parameters for ions and waters
    return merge_solvated_topology(
        protein,
        solvent_topo,
        model_type=model_type,
        box_size=box_size,
        merged_positions=pos_final
    )
