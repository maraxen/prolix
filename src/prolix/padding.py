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

Array = util.Array

ATOM_BUCKETS = (4096, 8192, 16384, 32768, 65536)

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
  atom_mask: Array        # (N_padded,) bool — True for real atoms
  
  # Bonded term arrays (padded to max per bucket)
  bonds: Array            # (N_bonds_padded, 2) int
  bond_params: Array      # (N_bonds_padded, 2) float
  bond_mask: Array        # (N_bonds_padded,) bool
  
  angles: Array           # (N_angles_padded, 3) int
  angle_params: Array     # (N_angles_padded, 2) float
  angle_mask: Array       # (N_angles_padded,) bool
  
  dihedrals: Array        # (N_dih_padded, 4) int
  dihedral_params: Array  # (N_dih_padded, 3) float
  dihedral_mask: Array    # (N_dih_padded,) bool
  
  impropers: Array        # (N_imp_padded, 4) int
  improper_params: Array  # (N_imp_padded, 3) float
  improper_mask: Array    # (N_imp_padded,) bool

  # CMAP (optional)
  cmap_torsions: Array | None   # (N_cmap_padded, 5) int
  cmap_mask: Array | None       # (N_cmap_padded,) bool
  cmap_coeffs: Array | None     # (N_maps, G, G, 16) — shared across batch

  # Metadata
  n_real_atoms: Array
  n_padded_atoms: int | Array = eqx.field(static=True)
  bucket_size: int | Array = eqx.field(static=True)


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
  target_atoms: int,
  target_bonds: int | None = None,
  target_angles: int | None = None,
  target_dihedrals: int | None = None,
  target_impropers: int | None = None,
  target_cmaps: int | None = None,
) -> PaddedSystem:
  """Pad a protein to specific array sizes."""
  
  pos = jnp.asarray(protein.coordinates).reshape(-1, 3)
  n_real = len(pos)
  
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
  _ELEMENT_MASS = {
      "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "S": 32.06,
      "P": 30.974, "F": 18.998, "Cl": 35.45, "Br": 79.904, "I": 126.904,
      "Fe": 55.845, "Zn": 65.38, "Ca": 40.078, "Mg": 24.305, "Na": 22.990,
      "K": 39.098, "Se": 78.971, "Mn": 54.938, "Cu": 63.546, "Co": 58.933,
  }
  masses = protein.masses
  if masses is None or jnp.all(jnp.asarray(masses) == 0):
      elements = getattr(protein, "elements", None)
      if elements is not None and len(elements) == n_real:
          mass_list = [_ELEMENT_MASS.get(e, _ELEMENT_MASS.get(e.capitalize(), 12.011)) for e in elements]
          masses = jnp.array(mass_list, dtype=jnp.float32)
      else:
          masses = jnp.ones(n_real) * 12.0
  padded_masses = pad_array(masses, target_atoms, 1.0)
  
  atom_mask = create_mask(n_real, target_atoms)

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

  cmaps = getattr(protein, 'cmap_torsions', None)
  if cmaps is not None:
    cmaps = jnp.asarray(cmaps)
  else:
    cmaps = jnp.zeros((0, 5), dtype=jnp.int32)

  if target_cmaps is None: target_cmaps = len(cmaps)

  # Padded Arrays
  padded_bonds = pad_bonded_indices(bonds, target_bonds, 2)
  padded_bond_params = pad_bonded_params(protein.bond_params, target_bonds, 2)
  bond_mask = create_mask(len(bonds), target_bonds)

  padded_angles = pad_bonded_indices(angles, target_angles, 3)
  padded_angle_params = pad_bonded_params(protein.angle_params, target_angles, 2)
  angle_mask = create_mask(len(angles), target_angles)

  padded_dihedrals = pad_bonded_indices(dihedrals, target_dihedrals, 4)
  padded_dih_params = pad_bonded_params(protein.dihedral_params, target_dihedrals, 3)
  dih_mask = create_mask(len(dihedrals), target_dihedrals)

  padded_impropers = pad_bonded_indices(impropers, target_impropers, 4)
  padded_imp_params = pad_bonded_params(protein.improper_params, target_impropers, 3)
  imp_mask = create_mask(len(impropers), target_impropers)

  if target_cmaps > 0:
    padded_cmaps = pad_bonded_indices(cmaps, target_cmaps, 5)
    cmap_mask = create_mask(len(cmaps), target_cmaps)
    cmap_coeffs = jnp.asarray(protein.cmap_coeffs) if getattr(protein, 'cmap_coeffs', None) is not None else None
  else:
    padded_cmaps = None
    cmap_mask = None
    cmap_coeffs = None
    
  return PaddedSystem(
      positions=padded_pos,
      charges=padded_charges,
      sigmas=padded_sigmas,
      epsilons=padded_epsilons,
      radii=padded_radii,
      scaled_radii=padded_scaled_radii,
      masses=padded_masses,
      atom_mask=atom_mask,
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
      cmap_torsions=padded_cmaps,
      cmap_mask=cmap_mask,
      cmap_coeffs=cmap_coeffs,
      n_real_atoms=jnp.array(n_real, dtype=jnp.int32),
      n_padded_atoms=target_atoms,
      bucket_size=target_atoms,
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
      
      padded_list = []
      for p in prot_list:
        padded = pad_protein(
            p, bucket_size,
            target_bonds=max_bonds,
            target_angles=max_angles,
            target_dihedrals=max_dihedrals,
            target_impropers=max_impropers,
            target_cmaps=max_cmaps
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
    
    for s in systems:
      assert s.bucket_size == b_size, "Systems must belong to identical ATOM bucket."
      assert count(s, "bonds") == n_bonds, "Systems must belong to identical BONDS length."
      assert count(s, "angles") == n_angles, "Systems must belong to identical ANGLES length."
      assert count(s, "dihedrals") == n_dih, "Systems must belong to identical DIHEDRALS length."
      assert count(s, "impropers") == n_imp, "Systems must belong to identical IMPROPERS length."
      assert count(s, "cmap_torsions") == n_cmap, "Systems must belong to identical CMAPS length."
      
    # Equinox modules are PyTrees, so tree_map stacks their arrays naturally
    return jax.tree_util.tree_map(lambda *args: jnp.stack(args), *systems)
