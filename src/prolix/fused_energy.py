"""Fused energy + force computation for neighbor-list MD.

Eliminates jax.grad for LJ by computing analytical forces in the forward pass.
GB still uses checkpointed jax.grad (the chain rule is complex), but the overall
function returns (energy, forces) directly — no need for outer jax.grad.

The key speedup: LJ gradient was the most expensive autodiff component
(~12ms for gradient alone). Analytical forces eliminate this entirely.

Usage:
    energy, forces = fused_energy_and_forces_nl(sys, neighbor_idx, displacement_fn)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
  from prolix.padding import PaddedSystem


def compute_pairwise_nl(
  positions: jnp.ndarray,
  neighbor_idx: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Precompute displacements and distances for all neighbor slots.

  Args:
      positions: (N, 3) atom positions.
      neighbor_idx: (N, K) neighbor indices; padding uses sentinel >= N.

  Returns:
      dr: (N, K, 3) central − neighbor displacement.
      dist: (N, K) center–neighbor distances.
      safe_idx: (N, K) neighbor indices clamped for safe gathering.
  """
  N = positions.shape[0]
  safe_idx = jnp.minimum(neighbor_idx, N - 1)
  pos_neighbors = positions[safe_idx]
  pos_central = positions[:, None, :]
  dr = pos_central - pos_neighbors
  dist = jnp.sqrt(jnp.sum(dr**2, axis=-1) + jnp.float32(1e-12))
  return dr, dist, safe_idx


def lj_energy_and_force_nl(
  dr: jnp.ndarray,
  dist: jnp.ndarray,
  sigmas: jnp.ndarray,
  epsilons: jnp.ndarray,
  safe_idx: jnp.ndarray,
  neighbor_idx: jnp.ndarray,
  soft_core_lambda: float = 1.0,
  excl_scales_vdw: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """LJ energy and analytical forces using precomputed pairwise dr/dist (N, K).

  Args:
      dr: (N, K, 3) displacements.
      dist: (N, K) distances.
      sigmas, epsilons: (N,) LJ parameters.
      safe_idx: (N, K) clamped neighbor indices for gathering.
      neighbor_idx: (N, K) raw neighbor indices (for padding mask).
      excl_scales_vdw: (N, K) LJ scale factors from exclusion lookup.

  Returns:
      energy: scalar total LJ energy.
      forces: (N, 3) per-atom force array.
  """
  N = dr.shape[0]
  K = dr.shape[1]

  lam = jnp.float32(soft_core_lambda)
  alpha = jnp.float32(0.5)
  soft_term = alpha * jnp.maximum(jnp.float32(1.0) - lam, jnp.float32(1e-8))

  def _scan_fn(carry, k):
    e_acc, f_acc = carry

    dr_k = dr[:, k, :]
    dist_k = dist[:, k]
    idx_k = neighbor_idx[:, k]
    mask_k = idx_k < N
    safe_idx_k = safe_idx[:, k]

    sigma_ij = 0.5 * (sigmas + sigmas[safe_idx_k])
    epsilon_ij = jnp.sqrt(epsilons * epsilons[safe_idx_k])

    r_over_sig = dist_k / jnp.maximum(sigma_ij, jnp.float32(1e-8))
    r6 = r_over_sig**6
    denom = soft_term + r6 + jnp.float32(1e-12)

    e_pair = jnp.float32(4.0) * epsilon_ij * lam * (
      jnp.float32(1.0) / (denom * denom) - jnp.float32(1.0) / denom
    )
    e_pair = jnp.where(mask_k, e_pair, 0.0)

    if excl_scales_vdw is not None:
      excl_scale_k = excl_scales_vdw[:, k]
      e_pair = e_pair * excl_scale_k

    sigma_ij6 = jnp.maximum(sigma_ij**6, jnp.float32(1e-48))
    ddenom_ddist = jnp.float32(6.0) * dist_k**5 / sigma_ij6

    de_ddist = (
      jnp.float32(4.0)
      * epsilon_ij
      * lam
      * (jnp.float32(1.0) / (denom**2) - jnp.float32(2.0) / (denom**3))
      * ddenom_ddist
    )

    de_ddist = jnp.where(mask_k, de_ddist, 0.0)
    if excl_scales_vdw is not None:
      excl_scale_k = excl_scales_vdw[:, k]
      de_ddist = de_ddist * excl_scale_k

    unit_dr = dr_k / (dist_k[..., None] + 1e-12)
    f_pair = de_ddist[..., None] * unit_dr
    f_pair_j = -f_pair

    return (e_acc + e_pair, f_acc + f_pair_j), None

  init_carry = (jnp.zeros(N, dtype=jnp.float32), jnp.zeros((N, 3), dtype=jnp.float32))

  (e_total_arr, f_total), _ = jax.lax.scan(_scan_fn, init_carry, jnp.arange(K))

  energy = 0.5 * jnp.sum(e_total_arr)

  return energy, f_total


# ==============================================================================
# BONDED ENERGY (for jax.grad — O(N), cheap)
# ==============================================================================


def _bonded_energy_from_positions(
  positions: jnp.ndarray,
  sys: PaddedSystem,
  displacement_fn,
) -> jnp.ndarray:
  """Compute total bonded energy from positions (for jax.grad)."""
  from prolix.batched_energy import (
    _angle_energy_masked,
    _bond_energy_masked,
    _cmap_energy_masked,
    _dihedral_energy_masked,
  )

  e_bond = _bond_energy_masked(
    positions, sys.bonds, sys.bond_params, sys.bond_mask, displacement_fn
  )
  e_angle = _angle_energy_masked(
    positions, sys.angles, sys.angle_params, sys.angle_mask, displacement_fn
  )
  e_dih = _dihedral_energy_masked(
    positions, sys.dihedrals, sys.dihedral_params, sys.dihedral_mask, displacement_fn
  )
  e_imp = _dihedral_energy_masked(
    positions, sys.impropers, sys.improper_params, sys.improper_mask, displacement_fn
  )
  e_cmap = _cmap_energy_masked(
    positions, sys.cmap_torsions, sys.cmap_indices, sys.cmap_mask, sys.cmap_coeffs, displacement_fn
  )
  return e_bond + e_angle + e_dih + e_imp + e_cmap


# ==============================================================================
# GB ENERGY (for jax.grad — matches batched NL path + optional OpenMM-style masks)
# ==============================================================================


def _gb_energy_from_positions(
  positions: jnp.ndarray,
  sys: PaddedSystem,
  neighbor_idx: jnp.ndarray,
  dielectric_offset: float = 0.09,
) -> jnp.ndarray:
  """GB polar + ACE nonpolar using the same NL kernel as ``single_padded_energy_nl``."""
  from prolix.physics.generalized_born import (
    compute_ace_nonpolar_energy,
    compute_gb_energy_neighbor_list,
  )

  N = positions.shape[0]
  scaled_radii = sys.scaled_radii
  pair_mask_born = None
  pair_mask_energy = None
  if sys.dense_excl_scale_vdw is not None:
    gb_mask = jnp.ones((N, N), dtype=jnp.float32)
    gb_energy_mask = jnp.ones((N, N), dtype=jnp.float32)
    idx_nl = neighbor_idx
    safe_j = jnp.minimum(idx_nl, N - 1)
    ii = jnp.arange(N)[:, None]
    valid = idx_nl < N
    pair_mask_born = gb_mask[ii, safe_j]
    pair_mask_born = jnp.where(valid, pair_mask_born, 0.0)
    pair_mask_energy = gb_energy_mask[ii, safe_j]
    pair_mask_energy = jnp.where(valid, pair_mask_energy, 0.0)

  e_gb, born_radii = compute_gb_energy_neighbor_list(
    positions=positions,
    charges=sys.charges,
    radii=sys.radii,
    neighbor_idx=neighbor_idx,
    dielectric_offset=dielectric_offset,
    scaled_radii=scaled_radii,
    pair_mask_born=pair_mask_born,
    pair_mask_energy=pair_mask_energy,
  )
  e_np = compute_ace_nonpolar_energy(sys.radii, born_radii)
  e_np = jnp.sum(e_np * sys.atom_mask)
  return e_gb + e_np


# ==============================================================================
# PUBLIC API: FUSED ENERGY + FORCES
# ==============================================================================


def fused_energy_and_forces_nl(
  sys: PaddedSystem,
  neighbor_idx: jnp.ndarray,
  displacement_fn,
  implicit_solvent: bool = True,
  soft_core_lambda: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Compute total energy and forces with analytical LJ gradient.

  Architecture:
  - LJ: fully analytical energy + force (no jax.grad, single pass)
  - GB: checkpointed jax.grad (complex chain rule, recomputed on backward)
  - Bonded: jax.grad (O(N), ~0.1ms, negligible cost)

  The energy sum matches single_padded_energy_nl_cvjp:
      e_bonded + e_lj + 2 * e_solv
  where e_solv = e_gb_polar + e_nonpolar (factor 2 matches e_elec + e_solv convention).

  Args:
      sys: PaddedSystem with positions, charges, radii, etc.
      neighbor_idx: (N, K) neighbor indices, sentinel = N.
      displacement_fn: JAX-MD displacement function.
      implicit_solvent: Whether to include GB solvation.
      soft_core_lambda: Soft-core LJ coupling (1.0 = standard).

  Returns:
      (total_energy, forces): scalar energy and (N, 3) force array.
  """
  r = sys.positions

  from prolix.physics.neighbor_list import get_neighbor_exclusion_scales

  excl_scales_vdw_nl, _ = get_neighbor_exclusion_scales(
    sys.excl_indices, sys.excl_scales_vdw, sys.excl_scales_elec, neighbor_idx,
  )

  dr, dist, safe_idx = compute_pairwise_nl(r, neighbor_idx)

  e_lj, f_lj = lj_energy_and_force_nl(
    dr,
    dist,
    sys.sigmas,
    sys.epsilons,
    safe_idx,
    neighbor_idx,
    soft_core_lambda=soft_core_lambda,
    excl_scales_vdw=excl_scales_vdw_nl,
  )

  if implicit_solvent:
    gb_fn = jax.checkpoint(lambda pos: _gb_energy_from_positions(pos, sys, neighbor_idx))
    e_solv = gb_fn(r)
    f_solv = -jax.grad(gb_fn)(r)
    e_gb_total = 2.0 * e_solv
    f_gb_total = 2.0 * f_solv
  else:
    e_gb_total = jnp.float32(0.0)
    f_gb_total = jnp.zeros_like(r)

  def bonded_fn(pos):
    return _bonded_energy_from_positions(pos, sys, displacement_fn)

  e_bonded = bonded_fn(r)
  f_bonded = -jax.grad(bonded_fn)(r)

  total_energy = e_bonded + e_lj + e_gb_total
  total_forces = f_lj + f_gb_total + f_bonded

  return total_energy, total_forces
