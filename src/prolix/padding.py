"""Padding and batching utilities for cross-topology simulation."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax_md import util

if TYPE_CHECKING:
  from proxide.core.containers import Protein
  from prolix.physics.topology_merger import MergedTopology

Array = util.Array

ATOM_BUCKETS = (1024, 2048, 2816, 3072, 4096, 5120, 6144, 7168, 8192, 16384, 32768, 65536)

class PaddedSystem(eqx.Module):
  """A protein system padded to a fixed atom count for vmap compatibility."""
  
  # Per-atom arrays (all shape: (N_padded, ...))
  positions: Array        # (N_padded, 3)
  charges: Array          # (N_padded,)
  sigmas: Array           # (N_padded,)
  epsilons: Array         # (N_padded,)
  radii: Array            # (N_padded,)   — GB radii
  scaled_radii: Array     # (N_padded,)   — OBC scaling factors
  masses: Array           # (N_padded,)
  element_ids: Array      # (N_padded,) int — atomic number (1=H, 6=C, 7=N, 8=O, 16=S)
  atom_mask: Array        # (N_padded,) bool — True for real atoms
  is_hydrogen: Array      # (N_padded,) bool — True for hydrogen atoms
  is_backbone: Array      # (N_padded,) bool — True for backbone atoms (N, CA, C, O)
  is_heavy: Array         # (N_padded,) bool — True for real non-hydrogen atoms
  protein_atom_mask: Array # (N_padded,) bool — True for protein atoms
  water_atom_mask: Array   # (N_padded,) bool — True for water atoms
  
  # Bonded term arrays (padded to max per bucket)
  bonds: Array                                     # (N_bonds_padded, 2) int
  bond_params: Array      # (N_bonds_padded, 2) float
  bond_mask: Array        # (N_bonds_padded,) bool
  
  angles: Array                                    # (N_angles_padded, 3) int
  angle_params: Array     # (N_angles_padded, 2) float
  angle_mask: Array       # (N_angles_padded,) bool
  
  dihedrals: Array                                 # (N_dih_padded, 4) int
  dihedral_params: Array  # (N_dih_padded, 3) float
  dihedral_mask: Array    # (N_dih_padded,) bool
  
  impropers: Array                                 # (N_imp_padded, 4) int
  improper_params: Array  # (N_imp_padded, 3) float
  improper_mask: Array    # (N_imp_padded,) bool

  # Urey-Bradley
  urey_bradley_bonds: Array | None  # (N_ub_padded, 2) int
  urey_bradley_params: Array | None # (N_ub_padded, 2) float
  urey_bradley_mask: Array | None   # (N_ub_padded,) bool

  # CMAP (optional)
  cmap_torsions: Array | None = None               # (N_cmap_padded, 5) int
  cmap_indices: Array | None = None                # (N_cmap_padded,) int
  cmap_mask: Array | None = None       # (N_cmap_padded,) bool
  cmap_coeffs: Array | None = None     # (N_maps, G, G, 16) — shared across batch

  # Non-bonded exclusions — sparse per-atom arrays
  # Used for 1-2/1-3 (fully excluded) and 1-4 (scaled) interactions.
  excl_indices: Array = None                       # (N_padded, max_excl) int32 — excluded atom indices, -1 = unused
  excl_scales_vdw: Array | None = None    # (N_padded, max_excl) float32 — LJ scale (0.0 or 0.5 or 1.0)
  excl_scales_elec: Array | None = None   # (N_padded, max_excl) float32 — elec scale (0.0 or 1/1.2 or 1.0)

  # RATTLE/SHAKE constraints — X-H bond pairs with target lengths
  constraint_pairs: Array = None                   # (N_constr_padded, 2) int — atom indices for constrained bonds
  constraint_lengths: Array | None = None   # (N_constr_padded,) float — equilibrium bond lengths (Å)
  constraint_mask: Array | None = None      # (N_constr_padded,) bool — True for real constraints

  # Metadata
  n_real_atoms: Array | None = None
  n_padded_atoms: int | Array = eqx.field(static=True, default=0)
  bucket_size: int | Array = eqx.field(static=True, default=0)

  # Water molecule indices for SETTLE
  water_indices: Array | None = None               # (N_waters_padded, 3) int
  water_mask: Array | None = None      # (N_waters_padded,) bool
  box_size: Array | None = eqx.field(static=True, default=None) # (3,) static for PME grid shapes
  pme_alpha: float = eqx.field(static=True, default=0.0)


  # Precomputed dense exclusion matrices (N_padded, N_padded).
  # These are topology constants — they never change during simulation.
  # Precomputing once avoids the costly fori_loop+scatter rebuild every step.
  dense_excl_scale_vdw: Array | None = None   # (N_padded, N_padded) float32
  dense_excl_scale_elec: Array | None = None  # (N_padded, N_padded) float32


def select_bucket(n_atoms: int, buckets: tuple[int, ...] = ATOM_BUCKETS) -> int:
  """Select the smallest bucket that fits n_atoms."""
  for b in sorted(buckets):
    if n_atoms <= b:
      return b
  raise ValueError(f"Atom count {n_atoms} exceeds maximum bucket size {max(buckets)}")

def pad_array(arr: Array | np.ndarray, target_length: int, pad_value: Any) -> Array:
  """Pads the first dimension of an array to target_length with pad_value."""
  arr = jnp.asarray(arr)
  if len(arr) == target_length:
    return arr
  if len(arr) > target_length:
    raise ValueError(f"Array length {len(arr)} exceeds target padding {target_length}")
  
  pad_width = [(0, target_length - len(arr))]
  for _ in range(1, arr.ndim):
    pad_width.append((0, 0))
    
  return jnp.pad(arr, pad_width, constant_values=pad_value)

def pad_bonded_indices(indices: Array | None, target_length: int, width: int) -> Array:
  """Pad bonded indices with zeros. Real indices will use safe `0` values for ghost terms,
  and the energy function's `k=0` padding ensures they won't contribute."""
  if indices is None or len(indices) == 0:
    return jnp.zeros((target_length, width), dtype=jnp.int32)
  return pad_array(indices, target_length, 0)

def pad_bonded_params(params: Array | None, target_length: int, width: int) -> Array:
  """Pad bonded parameters. For force constants, `k=0` must be used so ghost terms
  have 0 energy contribution."""
  if params is None or len(params) == 0:
    return jnp.zeros((target_length, width), dtype=jnp.float32)
  return pad_array(params, target_length, 0.0)

def create_mask(n_real: int, n_padded: int) -> Array:
  """Create boolean mask matching length."""
  return jnp.arange(n_padded) < n_real

def pad_protein(
  protein: Protein,
  target_atoms: int | None = None,
  target_bonds: int | None = None,
  target_angles: int | None = None,
  target_dihedrals: int | None = None,
  target_impropers: int | None = None,
  target_cmaps: int | None = None,
  target_ub: int | None = None,
  target_constraints: int | None = None,
  pme_alpha: float = 0.0,
) -> PaddedSystem:
  """Pad a protein to specific array sizes."""
  
  pos = jnp.asarray(protein.coordinates).reshape(-1, 3)
  n_real = len(pos)

  if target_atoms is None:
    target_atoms = select_bucket(n_real)
  
  if n_real > target_atoms:
    raise ValueError(f"Protein has {n_real} atoms, exceeds target padding {target_atoms}")
    
  # Ghost atom conventions
  # Non-bonded ghost atoms are positioned far away to not interact
  # Note: if use_pbc=True, 9999 may wrap around depending on box size.
  padded_pos = pad_array(pos, target_atoms, 9999.0)
  
  charges = jnp.asarray(protein.charges) if protein.charges is not None else jnp.zeros(n_real)
  padded_charges = pad_array(charges, target_atoms, 0.0)
  
  sigmas = jnp.asarray(protein.sigmas) if protein.sigmas is not None else jnp.ones(n_real)*3.0
  padded_sigmas = pad_array(sigmas, target_atoms, 1e-6)  # non-zero to avoid singularities 
  
  epsilons = jnp.asarray(protein.epsilons) if protein.epsilons is not None else jnp.zeros(n_real)
  padded_epsilons = pad_array(epsilons, target_atoms, 0.0)
  
  radii = jnp.asarray(protein.radii) if protein.radii is not None else jnp.ones(n_real)
  padded_radii = pad_array(radii, target_atoms, 1.5)  # Safe GB radii
  
  scaled_radii = getattr(protein, 'scaled_radii', None)
  if scaled_radii is None:
      scaled_radii = jnp.ones(n_real)*0.8
  padded_scaled_radii = pad_array(jnp.asarray(scaled_radii), target_atoms, 0.8) # Standard OBC2 default
  
  # Derive masses if none
  from prolix.constants import masses_from_elements, atomic_numbers_from_elements, DEFAULT_MASS

  masses = protein.masses
  elements_list = getattr(protein, "elements", None)
  if masses is None or jnp.all(jnp.asarray(masses) == 0):
      if elements_list is not None and len(elements_list) == n_real:
          masses = jnp.array(masses_from_elements(list(elements_list)), dtype=jnp.float32)
      else:
          masses = jnp.ones(n_real) * DEFAULT_MASS
  # Pad with DEFAULT_MASS so ghost atoms have safe non-zero mass for Langevin p/m
  padded_masses = pad_array(masses, target_atoms, DEFAULT_MASS)

  # Real atom mask from protein (handles missing atoms in PDB)
  p_mask = getattr(protein, 'atom_mask', None)
  if p_mask is None:
    p_mask = jnp.ones(n_real, dtype=bool)
  else:
    p_mask = jnp.asarray(p_mask, dtype=bool)

  atom_mask = pad_array(p_mask, target_atoms, False)
  
  # Derive additional masks for selective restraints
  atom_names = getattr(protein, "atom_names", None)
  if atom_names is not None and len(atom_names) == n_real:
      is_h = jnp.array([str(n).strip().startswith("H") for n in atom_names], dtype=jnp.bool_)
      is_bb = jnp.array([str(n).strip() in {"N", "CA", "C", "O"} for n in atom_names], dtype=jnp.bool_)
  else:
      # Fallback: if no names, hydrogen defined by element_id==1
      is_h = (element_ids == 1)
      is_bb = jnp.zeros(n_real, dtype=jnp.bool_) # Cannot infer backbone without names
  
  padded_is_hydrogen = pad_array(is_h, target_atoms, False)
  padded_is_backbone = pad_array(is_bb, target_atoms, False)
  padded_is_heavy = atom_mask & ~padded_is_hydrogen

  # Exclusion data — sparse (N, max_excl) arrays
  from prolix.physics.neighbor_list import ExclusionSpec, map_exclusions_to_dense_padded
  MAX_EXCL = 32
  try:
    excl_spec = ExclusionSpec.from_protein(protein)
    ei, sv, se = map_exclusions_to_dense_padded(excl_spec, max_exclusions=MAX_EXCL)
    # ei: (n_real, MAX_EXCL) int32, sv/se: (n_real, MAX_EXCL) float32
    padded_excl_indices = pad_array(ei, target_atoms, -1)
    padded_excl_scales_vdw = pad_array(sv, target_atoms, 1.0)
    padded_excl_scales_elec = pad_array(se, target_atoms, 1.0)
  except (ValueError, AttributeError):
    # Fallback: if protein has no bonds/charges, create empty exclusions
    padded_excl_indices = jnp.full((target_atoms, MAX_EXCL), -1, dtype=jnp.int32)
    padded_excl_scales_vdw = jnp.ones((target_atoms, MAX_EXCL), dtype=jnp.float32)
    padded_excl_scales_elec = jnp.ones((target_atoms, MAX_EXCL), dtype=jnp.float32)

  # Target logic
  bonds = jnp.asarray(protein.bonds) if protein.bonds is not None else jnp.zeros((0, 2), dtype=jnp.int32)
  if target_bonds is None: target_bonds = len(bonds)
  
  angles = jnp.asarray(protein.angles) if protein.angles is not None else jnp.zeros((0, 3), dtype=jnp.int32)
  if target_angles is None: target_angles = len(angles)
  
  proper_dihedrals = getattr(protein, 'proper_dihedrals', None)
  dihedrals = jnp.asarray(proper_dihedrals) if proper_dihedrals is not None else jnp.zeros((0, 4), dtype=jnp.int32)
  if target_dihedrals is None: target_dihedrals = len(dihedrals)
  impropers = jnp.asarray(protein.impropers) if protein.impropers is not None else jnp.zeros((0, 4), dtype=jnp.int32)
  if target_impropers is None: target_impropers = len(impropers)
  
  improper_mask = create_mask(len(impropers), target_impropers)
  
  # Urey-Bradley
  ub_bonds = getattr(protein, 'urey_bradley_bonds', None)
  if ub_bonds is None: ub_bonds = jnp.zeros((0, 2), dtype=jnp.int32)
  ub_params = getattr(protein, 'urey_bradley_params', None)
  if ub_params is None: ub_params = jnp.zeros((0, 2), dtype=jnp.float32)
  if target_ub is None: target_ub = len(ub_bonds)
  
  padded_ub_bonds = pad_bonded_indices(ub_bonds, target_ub, 2)
  padded_ub_params = pad_bonded_params(ub_params, target_ub, 2)
  padded_ub_mask = create_mask(len(ub_bonds), target_ub)

  cmaps = getattr(protein, 'cmap_torsions', None)
  if cmaps is not None:
    cmaps = jnp.asarray(cmaps)
  else:
    cmaps = jnp.zeros((0, 5), dtype=jnp.int32)

  if target_cmaps is None: target_cmaps = len(cmaps)

  # Padded Arrays
  padded_bonds = pad_bonded_indices(bonds, target_bonds, 2)
  padded_bond_params = pad_bonded_params(protein.bond_params, target_bonds, 2)
  # Bond is real only if both atoms are in the mask
  real_bonds = p_mask[bonds].all(axis=-1) if len(bonds) > 0 else jnp.zeros(0, dtype=bool)
  bond_mask = pad_array(real_bonds, target_bonds, False)

  padded_angles = pad_bonded_indices(angles, target_angles, 3)
  padded_angle_params = pad_bonded_params(protein.angle_params, target_angles, 2)
  # Angle is real only if all three atoms are in the mask
  real_angles = p_mask[angles].all(axis=-1) if len(angles) > 0 else jnp.zeros(0, dtype=bool)
  angle_mask = pad_array(real_angles, target_angles, False)

  padded_dihedrals = pad_bonded_indices(dihedrals, target_dihedrals, 4)
  padded_dih_params = pad_bonded_params(protein.dihedral_params, target_dihedrals, 3)
  # Dihedral is real only if all four atoms are in the mask
  real_dih = p_mask[dihedrals].all(axis=-1) if len(dihedrals) > 0 else jnp.zeros(0, dtype=bool)
  dih_mask = pad_array(real_dih, target_dihedrals, False)

  padded_impropers = pad_bonded_indices(impropers, target_impropers, 4)
  padded_imp_params = pad_bonded_params(protein.improper_params, target_impropers, 3)
  # Improper is real only if all four atoms are in the mask
  real_imp = p_mask[impropers].all(axis=-1) if len(impropers) > 0 else jnp.zeros(0, dtype=bool)
  imp_mask = pad_array(real_imp, target_impropers, False)

  if target_cmaps > 0:
    padded_cmaps = pad_bonded_indices(cmaps, target_cmaps, 5)
    # CMAP is real only if all five atoms are in the mask
    real_cmap = p_mask[cmaps].all(axis=-1) if len(cmaps) > 0 else jnp.zeros(0, dtype=bool)
    cmap_mask = pad_array(real_cmap, target_cmaps, False)
    cmap_coeffs = jnp.asarray(protein.cmap_coeffs) if getattr(protein, 'cmap_coeffs', None) is not None else None
  else:
    padded_cmaps = None
    cmap_mask = None
    cmap_coeffs = None

  # -------------------------------------------------------------------
  # RATTLE constraints: identify X-H bonds by element identity (not mass).
  # Element-based check is HMR-safe — if hydrogen masses are repartitioned,
  # constraints are still correctly identified by atomic number == 1.
  # -------------------------------------------------------------------
  elem_np = np.asarray(element_ids)  # (n_real,) int — atomic numbers
  bonds_np = np.asarray(bonds)
  bond_params_np = np.asarray(protein.bond_params) if protein.bond_params is not None else np.zeros((0, 2))

  constr_pairs_list = []
  constr_lengths_list = []
  if len(bonds_np) > 0 and len(bond_params_np) > 0:
    for i in range(len(bonds_np)):
      a1, a2 = int(bonds_np[i, 0]), int(bonds_np[i, 1])
      # Check bounds (padding indices point to atom 0)
      if a1 >= n_real or a2 >= n_real:
        continue
      # Hydrogen has atomic number 1
      if int(elem_np[a1]) == 1 or int(elem_np[a2]) == 1:
        constr_pairs_list.append([a1, a2])
        # Equilibrium bond length from bond_params[:, 0] (r0)
        constr_lengths_list.append(float(bond_params_np[i, 0]))

  if constr_pairs_list:
    constr_pairs = jnp.array(constr_pairs_list, dtype=jnp.int32)
    constr_lengths = jnp.array(constr_lengths_list, dtype=jnp.float32)
  else:
    constr_pairs = jnp.zeros((0, 2), dtype=jnp.int32)
    constr_lengths = jnp.zeros((0,), dtype=jnp.float32)

  n_constraints = len(constr_pairs)
  if target_constraints is None:
    target_constraints = n_constraints

  padded_constr_pairs = pad_bonded_indices(constr_pairs, target_constraints, 2)
  padded_constr_lengths = pad_array(constr_lengths, target_constraints, 0.0) if n_constraints > 0 else jnp.zeros(target_constraints, dtype=jnp.float32)
  # Constraint is real only if both atoms are in the mask
  real_constr = p_mask[constr_pairs].all(axis=-1) if n_constraints > 0 else jnp.zeros(0, dtype=bool)
  constr_mask = pad_array(real_constr, target_constraints, False)

  sys = PaddedSystem(
      positions=padded_pos,
      charges=padded_charges,
      sigmas=padded_sigmas,
      epsilons=padded_epsilons,
      radii=padded_radii,
      scaled_radii=padded_scaled_radii,
      masses=padded_masses,
      element_ids=padded_element_ids,
      atom_mask=atom_mask,
      is_hydrogen=padded_is_hydrogen,
      is_backbone=padded_is_backbone,
      is_heavy=padded_is_heavy,
      protein_atom_mask=atom_mask,
      water_atom_mask=jnp.zeros(target_atoms, dtype=jnp.bool_),
      bonds=padded_bonds,
      bond_params=padded_bond_params,
      bond_mask=bond_mask,
      angles=padded_angles,
      angle_params=padded_angle_params,
      angle_mask=angle_mask,
      dihedrals=padded_dihedrals,
      dihedral_params=padded_dih_params,
      dihedral_mask=dih_mask,
      impropers=padded_impropers,
      improper_params=padded_imp_params,
      improper_mask=imp_mask,
      urey_bradley_bonds=padded_ub_bonds,
      urey_bradley_params=padded_ub_params,
      urey_bradley_mask=padded_ub_mask,
      cmap_torsions=padded_cmaps,
      cmap_indices=None,
      cmap_mask=cmap_mask,
      cmap_coeffs=cmap_coeffs,
      excl_indices=padded_excl_indices,
      excl_scales_vdw=padded_excl_scales_vdw,
      excl_scales_elec=padded_excl_scales_elec,
      constraint_pairs=padded_constr_pairs,
      constraint_lengths=padded_constr_lengths,
      constraint_mask=constr_mask,
      n_real_atoms=jnp.array(n_real, dtype=jnp.int32),
      n_padded_atoms=target_atoms,
      bucket_size=target_atoms,
      pme_alpha=pme_alpha,
  )
  return sys


def pad_solvated_system(
    topology: MergedTopology,
    target_atoms: int | None = None,
    target_bonds: int | None = None,
    target_angles: int | None = None,
    target_dihedrals: int | None = None,
    target_impropers: int | None = None,
    target_cmaps: int | None = None,
    target_ub: int | None = None,
    target_constraints: int | None = None,
    target_waters: int | None = None,
    pme_alpha: float | None = None,
) -> PaddedSystem:
    """Pad a merged solvated topology to specific array sizes."""
    n_real = len(topology.positions)
    if target_atoms is None:
        target_atoms = select_bucket(n_real)

    if n_real > target_atoms:
        raise ValueError(f"System has {n_real} atoms, exceeds target padding {target_atoms}")

    if pme_alpha is None:
        pme_alpha = 0.34 if topology.box_size is not None else 0.0

    # 1. Base parameters
    padded_pos = pad_array(topology.positions, target_atoms, 9999.0)
    padded_charges = pad_array(topology.charges, target_atoms, 0.0)
    padded_sigmas = pad_array(topology.sigmas, target_atoms, 1e-6)
    padded_epsilons = pad_array(topology.epsilons, target_atoms, 0.0)
    padded_radii = pad_array(topology.radii, target_atoms, 0.0)
    padded_scaled_radii = pad_array(topology.scaled_radii, target_atoms, 0.0)
    padded_masses = pad_array(topology.masses, target_atoms, 12.0) # default mass
    
    # Real atom mask from topology
    t_mask = getattr(topology, 'atom_mask', None)
    if t_mask is None:
        t_mask = jnp.ones(n_real, dtype=bool)
    
    atom_mask = pad_array(t_mask, target_atoms, False)
    padded_protein_mask = pad_array(topology.protein_atom_mask, target_atoms, False)
    padded_water_mask = pad_array(topology.water_atom_mask, target_atoms, False)
    
    padded_element_ids = pad_array(topology.element_ids, target_atoms, 0)
    
    is_h = topology.is_hydrogen
    padded_is_hydrogen = pad_array(is_h, target_atoms, False)
    is_bb = topology.is_backbone
    padded_is_backbone = pad_array(is_bb, target_atoms, False)
    is_hv = topology.is_heavy
    padded_is_heavy = pad_array(is_hv, target_atoms, False)

    # 2. Exclusions
    from prolix.physics.neighbor_list import map_exclusions_to_dense_padded
    MAX_EXCL = 32
    ei, sv, se = map_exclusions_to_dense_padded(topology.exclusion_spec, max_exclusions=MAX_EXCL)
    padded_excl_indices = pad_array(ei, target_atoms, -1)
    padded_excl_scales_vdw = pad_array(sv, target_atoms, 1.0)
    padded_excl_scales_elec = pad_array(se, target_atoms, 1.0)

    # 3. Bonded terms
    if target_bonds is None: target_bonds = len(topology.bonds)
    if target_angles is None: target_angles = len(topology.angles)
    
    dihedrals = topology.proper_dihedrals if topology.proper_dihedrals is not None else jnp.zeros((0, 4), dtype=jnp.int32)
    dihedral_params = topology.dihedral_params if topology.dihedral_params is not None else jnp.zeros((0, 3), dtype=jnp.float32)
    if target_dihedrals is None: target_dihedrals = len(dihedrals)
    
    impropers = topology.impropers if topology.impropers is not None else jnp.zeros((0, 4), dtype=jnp.int32)
    improper_params = topology.improper_params if topology.improper_params is not None else jnp.zeros((0, 3), dtype=jnp.float32)
    if target_impropers is None: target_impropers = len(impropers)

    cmaps = topology.cmap_torsions if topology.cmap_torsions is not None else jnp.zeros((0, 5), dtype=jnp.int32)
    if target_cmaps is None: target_cmaps = len(cmaps)
    
    padded_bonds = pad_bonded_indices(topology.bonds, target_bonds, 2)
    padded_bond_params = pad_bonded_params(topology.bond_params, target_bonds, 2)
    real_bonds = t_mask[topology.bonds].all(axis=-1) if len(topology.bonds) > 0 else jnp.zeros(0, dtype=bool)
    bond_mask = pad_array(real_bonds, target_bonds, False)

    padded_angles = pad_bonded_indices(topology.angles, target_angles, 3)
    padded_angle_params = pad_bonded_params(topology.angle_params, target_angles, 2)
    real_angles = t_mask[topology.angles].all(axis=-1) if len(topology.angles) > 0 else jnp.zeros(0, dtype=bool)
    angle_mask = pad_array(real_angles, target_angles, False)
    
    padded_dihedrals = pad_bonded_indices(dihedrals, target_dihedrals, 4)
    padded_dih_params = pad_bonded_params(dihedral_params, target_dihedrals, 3)
    real_dih = t_mask[dihedrals].all(axis=-1) if len(dihedrals) > 0 else jnp.zeros(0, dtype=bool)
    dih_mask = pad_array(real_dih, target_dihedrals, False)

    padded_impropers = pad_bonded_indices(impropers, target_impropers, 4)
    padded_imp_params = pad_bonded_params(improper_params, target_impropers, 3)
    real_imp = t_mask[impropers].all(axis=-1) if len(impropers) > 0 else jnp.zeros(0, dtype=bool)
    imp_mask = pad_array(real_imp, target_impropers, False)

    # Urey-Bradley
    ub_bonds = topology.urey_bradley_bonds if topology.urey_bradley_bonds is not None else jnp.zeros((0, 2), dtype=jnp.int32)
    ub_params = topology.urey_bradley_params if topology.urey_bradley_params is not None else jnp.zeros((0, 2), dtype=jnp.float32)
    if target_ub is None: target_ub = len(ub_bonds)
    
    padded_ub_bonds = pad_bonded_indices(ub_bonds, target_ub, 2)
    padded_ub_params = pad_bonded_params(ub_params, target_ub, 2)
    real_ub = t_mask[ub_bonds].all(axis=-1) if len(ub_bonds) > 0 else jnp.zeros(0, dtype=bool)
    padded_ub_mask = pad_array(real_ub, target_ub, False)

    if target_cmaps > 0:
        padded_cmaps = pad_bonded_indices(cmaps, target_cmaps, 5)
        cmap_idx = topology.cmap_indices if getattr(topology, 'cmap_indices', None) is not None else jnp.zeros(len(cmaps), dtype=jnp.int32)
        padded_cmap_indices = pad_bonded_indices(jnp.expand_dims(cmap_idx, -1), target_cmaps, 1).squeeze(-1)
        real_cmap = t_mask[cmaps].all(axis=-1) if len(cmaps) > 0 else jnp.zeros(0, dtype=bool)
        cmap_mask = pad_array(real_cmap, target_cmaps, False)
        cmap_coeffs = topology.cmap_energy_grids
    else:
        padded_cmaps = None
        padded_cmap_indices = None
        cmap_mask = None
        cmap_coeffs = None

    # 4. Constraints
    # Protein X-H constraints
    elem_np = np.asarray(topology.element_ids)
    bonds_np = np.asarray(topology.bonds)
    bond_params_np = np.asarray(topology.bond_params)
    
    constr_pairs_list = []
    constr_lengths_list = []
    if len(bonds_np) > 0 and len(bond_params_np) > 0:
        for i in range(len(bonds_np)):
            # We only build constraints for the protein part (X-H), water relies on SETTLE
            # which does not use constraint_pairs
            a1, a2 = int(bonds_np[i, 0]), int(bonds_np[i, 1])
            if a1 >= topology.n_protein_atoms and a2 >= topology.n_protein_atoms:
                continue # Water-water bonds are skipped (handled by SETTLE)
            if a1 >= n_real or a2 >= n_real:
                continue
            if int(elem_np[a1]) == 1 or int(elem_np[a2]) == 1:
                constr_pairs_list.append([a1, a2])
                constr_lengths_list.append(float(bond_params_np[i, 0]))

    if constr_pairs_list:
        constr_pairs = jnp.array(constr_pairs_list, dtype=jnp.int32)
        constr_lengths = jnp.array(constr_lengths_list, dtype=jnp.float32)
    else:
        constr_pairs = jnp.zeros((0, 2), dtype=jnp.int32)
        constr_lengths = jnp.zeros((0,), dtype=jnp.float32)

    n_constraints = len(constr_pairs)
    if target_constraints is None:
        target_constraints = n_constraints

    padded_constr_pairs = pad_bonded_indices(constr_pairs, target_constraints, 2)
    padded_constr_lengths = pad_array(constr_lengths, target_constraints, 0.0) if n_constraints > 0 else jnp.zeros(target_constraints, dtype=jnp.float32)
    real_constr = t_mask[constr_pairs].all(axis=-1) if n_constraints > 0 else jnp.zeros(0, dtype=bool)
    constr_mask = pad_array(real_constr, target_constraints, False)

    # 5. Water-specific indices for SETTLE
    n_waters = len(topology.water_indices)
    if target_waters is None: target_waters = n_waters
    
    padded_water_indices = pad_bonded_indices(topology.water_indices, target_waters, 3)
    water_molecule_mask = create_mask(n_waters, target_waters)

    return PaddedSystem(
        box_size=np.array(topology.box_size) if topology.box_size is not None else None,
        positions=padded_pos,
        charges=padded_charges,
        sigmas=padded_sigmas,
        epsilons=padded_epsilons,
        radii=padded_radii,
        scaled_radii=padded_scaled_radii,
        masses=padded_masses,
        element_ids=padded_element_ids,
        atom_mask=atom_mask,
        is_hydrogen=padded_is_hydrogen,
        is_backbone=padded_is_backbone,
        is_heavy=padded_is_heavy,
        protein_atom_mask=padded_protein_mask,
        water_atom_mask=padded_water_mask,
        bonds=padded_bonds,
        bond_params=padded_bond_params,
        bond_mask=bond_mask,
        angles=padded_angles,
        angle_params=padded_angle_params,
        angle_mask=angle_mask,
        dihedrals=padded_dihedrals,
        dihedral_params=padded_dih_params,
        dihedral_mask=dih_mask,
        impropers=padded_impropers,
        improper_params=padded_imp_params,
        improper_mask=imp_mask,
        urey_bradley_bonds=padded_ub_bonds,
        urey_bradley_params=padded_ub_params,
        urey_bradley_mask=padded_ub_mask,
        cmap_torsions=padded_cmaps,
        cmap_indices=padded_cmap_indices,
        cmap_mask=cmap_mask,
        cmap_coeffs=cmap_coeffs,
        excl_indices=padded_excl_indices,
        excl_scales_vdw=padded_excl_scales_vdw,
        excl_scales_elec=padded_excl_scales_elec,
        constraint_pairs=padded_constr_pairs,
        constraint_lengths=padded_constr_lengths,
        constraint_mask=constr_mask,
        water_indices=padded_water_indices,
        water_mask=water_molecule_mask,
        n_real_atoms=jnp.array(n_real, dtype=jnp.int32),
        n_padded_atoms=target_atoms,
        bucket_size=target_atoms,
        pme_alpha=pme_alpha,
    )


def precompute_dense_exclusions(sys: PaddedSystem) -> PaddedSystem:
    """Precompute dense (N, N) exclusion scale matrices from sparse arrays.

    These matrices are pure topology constants — they depend only on bond
    connectivity, never on positions. Computing them once and caching on the
    PaddedSystem eliminates a costly fori_loop+scatter from every MD step.

    This gave a measured ~4x speedup in per-step force evaluation
    (0.437ms → 0ms for exclusion construction).
    """
    from prolix.batched_energy import _build_dense_exclusion_scales

    N = sys.n_padded_atoms
    dense_vdw = _build_dense_exclusion_scales(
        sys.excl_indices, sys.excl_scales_vdw, N,
    )
    dense_elec = _build_dense_exclusion_scales(
        sys.excl_indices, sys.excl_scales_elec, N,
    )
    return eqx.tree_at(
        lambda s: (s.dense_excl_scale_vdw, s.dense_excl_scale_elec),
        sys,
        (dense_vdw, dense_elec),
        is_leaf=lambda x: x is None,
    )

def bucket_proteins(
    proteins: list[Protein],
    buckets: tuple[int, ...] = ATOM_BUCKETS,
) -> dict[int, list[PaddedSystem]]:
    """Group proteins into buckets and pad each to its bucket size and max bonded terms."""
    
    # First, separate by target bucket
    groups = {}
    for p in proteins:
      pos = jnp.asarray(p.coordinates).reshape(-1, 3)
      n_atoms = len(pos)
      bucket = select_bucket(n_atoms, buckets)
      if bucket not in groups:
        groups[bucket] = []
      groups[bucket].append(p)
      
    # Next, pad each group to the max elements within that bucket
    ready_buckets = {}
    for bucket_size, prot_list in groups.items():
      # find maximum terms across this bucket
      # We use fixed multipliers to avoid JAX JIT recompiles for different molecules:
      max_bonds = int(1.2 * bucket_size)
      max_angles = int(2.2 * bucket_size)
      max_dihedrals = int(3.5 * bucket_size)
      max_impropers = int(0.5 * bucket_size)
      max_cmaps = int(0.3 * bucket_size)
      # Constraints: ~1 H-bond per heavy atom → ~0.6 × bucket_size
      max_constraints = int(0.7 * bucket_size)
      
      padded_list = []
      for p in prot_list:
        padded = pad_protein(
            p, bucket_size,
            target_bonds=max_bonds,
            target_angles=max_angles,
            target_dihedrals=max_dihedrals,
            target_impropers=max_impropers,
            target_cmaps=max_cmaps,
            target_constraints=max_constraints,
        )
        padded_list.append(padded)
      ready_buckets[bucket_size] = padded_list
      
    return ready_buckets

def collate_batch(systems: list[PaddedSystem]) -> PaddedSystem:
    """Stack multiple PaddedSystems into a batched PaddedSystem."""
    if not systems:
        raise ValueError("Cannot collate empty list.")
    
    b_size = systems[0].bucket_size
    def count(s, attr):
      val = getattr(s, attr)
      return val.shape[0] if val is not None else 0
      
    n_bonds = count(systems[0], "bonds")
    n_angles = count(systems[0], "angles")
    n_dih = count(systems[0], "dihedrals")
    n_imp = count(systems[0], "impropers")
    n_cmap = count(systems[0], "cmap_torsions")
    n_constr = count(systems[0], "constraint_pairs")
    
    for s in systems:
      assert s.bucket_size == b_size, "Systems must belong to identical ATOM bucket."
      assert count(s, "bonds") == n_bonds, "Systems must belong to identical BONDS length."
      assert count(s, "angles") == n_angles, "Systems must belong to identical ANGLES length."
      assert count(s, "dihedrals") == n_dih, "Systems must belong to identical DIHEDRALS length."
      assert count(s, "impropers") == n_imp, "Systems must belong to identical IMPROPERS length."
      assert count(s, "cmap_torsions") == n_cmap, "Systems must belong to identical CMAPS length."
      assert count(s, "constraint_pairs") == n_constr, "Systems must belong to identical CONSTRAINTS length."
      
    # Equinox modules are PyTrees, so tree_map stacks their arrays naturally
    return jax.tree_util.tree_map(lambda *args: jnp.stack(args), *systems)
