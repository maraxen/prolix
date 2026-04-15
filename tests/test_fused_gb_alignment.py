"""Fused implicit GB path uses scaled_radii like ``single_padded_energy_nl``."""

import jax
import jax.numpy as jnp
import numpy as np

from prolix.fused_energy import _gb_energy_from_positions
from prolix.padding import PaddedSystem


def _build_dense_neighbor_idx(n_atoms: int) -> jnp.ndarray:
  """Every atom lists all others (same as test_neighbor_list_energy)."""
  idx = np.zeros((n_atoms, n_atoms - 1), dtype=np.int32)
  for i in range(n_atoms):
    neighbors = list(range(i)) + list(range(i + 1, n_atoms))
    idx[i] = neighbors
  return jnp.array(idx)


def _minimal_padded(n: int, scaled_radii: jnp.ndarray) -> PaddedSystem:
  rng = np.random.RandomState(0)
  positions = jnp.array(rng.uniform(2.0, 18.0, (n, 3)), dtype=jnp.float32)
  charges = jnp.array(rng.uniform(-0.5, 0.5, n), dtype=jnp.float32)
  charges = charges - jnp.mean(charges)
  sigmas = jnp.ones(n, dtype=jnp.float32) * 1.5
  epsilons = jnp.ones(n, dtype=jnp.float32) * 0.1
  radii = jnp.ones(n, dtype=jnp.float32) * 1.5
  masses = jnp.ones(n, dtype=jnp.float32) * 12.0
  elem = jnp.ones(n, dtype=jnp.int32) * 6
  mask = jnp.ones(n, dtype=jnp.bool_)
  excl_indices = jnp.full((n, 1), -1, dtype=jnp.int32)
  excl_scales_vdw = jnp.ones((n, 1), dtype=jnp.float32)
  excl_scales_elec = jnp.ones((n, 1), dtype=jnp.float32)

  return PaddedSystem(
    positions=positions,
    charges=charges,
    sigmas=sigmas,
    epsilons=epsilons,
    radii=radii,
    scaled_radii=scaled_radii,
    masses=masses,
    element_ids=elem,
    atom_mask=mask,
    is_hydrogen=jnp.zeros(n, dtype=jnp.bool_),
    is_backbone=jnp.zeros(n, dtype=jnp.bool_),
    is_heavy=mask,
    protein_atom_mask=mask,
    water_atom_mask=jnp.zeros(n, dtype=jnp.bool_),
    bonds=jnp.zeros((0, 2), dtype=jnp.int32),
    bond_params=jnp.zeros((0, 2), dtype=jnp.float32),
    bond_mask=jnp.zeros(0, dtype=jnp.bool_),
    angles=jnp.zeros((0, 3), dtype=jnp.int32),
    angle_params=jnp.zeros((0, 2), dtype=jnp.float32),
    angle_mask=jnp.zeros(0, dtype=jnp.bool_),
    dihedrals=jnp.zeros((0, 4), dtype=jnp.int32),
    dihedral_params=jnp.zeros((0, 3), dtype=jnp.float32),
    dihedral_mask=jnp.zeros(0, dtype=jnp.bool_),
    impropers=jnp.zeros((0, 4), dtype=jnp.int32),
    improper_params=jnp.zeros((0, 3), dtype=jnp.float32),
    improper_mask=jnp.zeros(0, dtype=jnp.bool_),
    urey_bradley_bonds=jnp.zeros((0, 2), dtype=jnp.int32),
    urey_bradley_params=jnp.zeros((0, 2), dtype=jnp.float32),
    urey_bradley_mask=jnp.zeros(0, dtype=jnp.bool_),
    cmap_torsions=None,
    cmap_indices=None,
    cmap_mask=None,
    cmap_coeffs=None,
    excl_indices=excl_indices,
    excl_scales_vdw=excl_scales_vdw,
    excl_scales_elec=excl_scales_elec,
    n_real_atoms=jnp.array(n, dtype=jnp.int32),
    n_padded_atoms=n,
    bucket_size=n,
  )


def test_gb_energy_changes_with_scaled_radii():
  """``scaled_radii`` affects NL Born radii and thus GB+ACE; fused path must forward it."""
  jax.config.update("jax_enable_x64", True)
  n = 8
  neighbor_idx = _build_dense_neighbor_idx(n)

  sr_a = jnp.ones(n, dtype=jnp.float64) * 0.85
  sr_b = jnp.ones(n, dtype=jnp.float64) * 0.95
  sys_a = _minimal_padded(n, sr_a)
  sys_b = _minimal_padded(n, sr_b)
  r = sys_a.positions

  e_a = _gb_energy_from_positions(r, sys_a, neighbor_idx)
  e_b = _gb_energy_from_positions(r, sys_b, neighbor_idx)

  assert jnp.isfinite(e_a) and jnp.isfinite(e_b)
  assert not bool(jnp.allclose(e_a, e_b, rtol=1e-9))
