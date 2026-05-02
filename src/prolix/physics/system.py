from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
from jax_md import energy, partition, space, util
from proxide.physics import constants

from prolix.physics import bonded, cmap, explicit_corrections, generalized_born, pme, virtual_sites
from prolix.physics import neighbor_list as nl
from prolix.physics.electrostatic_methods import (
  ElectrostaticMethod,
  openmm_reaction_field_coefficients,
)
from prolix.types import CmapTorsionIndices
from prolix.utils import topology

from .optimization import chunked_lj_energy, chunked_coulomb_energy
from .types import PhysicsSystem, EnergyParams

if TYPE_CHECKING:
  from collections.abc import Callable
  from proxide.core.containers import Protein

Array = util.Array

class _DictSystemWrapper:
  """Wraps a plain dict to provide attribute access with defaults for optional fields."""
  _DEFAULTS = {
    "cmap_energy_grids": None, "cmap_torsions": None, "cmap_indices": None,
    "proper_dihedrals": None, "impropers": None, "improper_params": None,
    "urey_bradley_bonds": None, "urey_bradley_params": None, "virtual_site_def": None,
    "virtual_site_params": None, "radii": None, "scaled_radii": None,
    "scale_matrix_vdw": None, "scale_matrix_elec": None, "coulomb14scale": None,
    "lj14scale": None, "exception_14_params": None, "pairs_14": None,
    "exclusion_mask": None, "excl_indices": None, "excl_scales_vdw": None,
    "excl_scales_elec": None, "dense_excl_scale_vdw": None, "dense_excl_scale_elec": None,
  }
  def __init__(self, d: dict): self._d = d
  def __getattr__(self, name: str):
    if name.startswith("_"): raise AttributeError(name)
    try: return self._d[name]
    except KeyError:
      if name in self._DEFAULTS: return self._DEFAULTS[name]
      raise AttributeError(f"System dict has no key '{name}' and no default is defined")

def compute_dihedral_angles(r: Array, indices: Array, displacement_fn: space.DisplacementFn) -> Array:
  r_i, r_j, r_k, r_l = r[indices[:, 0]], r[indices[:, 1]], r[indices[:, 2]], r[indices[:, 3]]
  b0, b1, b2 = jax.vmap(displacement_fn)(r_i, r_j), jax.vmap(displacement_fn)(r_k, r_j), jax.vmap(displacement_fn)(r_l, r_k)
  b1_unit = b1 / (jnp.linalg.norm(b1, axis=-1, keepdims=True) + 1e-8)
  v = b0 - jnp.sum(b0 * b1_unit, axis=-1, keepdims=True) * b1_unit
  w = b2 - jnp.sum(b2 * b1_unit, axis=-1, keepdims=True) * b1_unit
  x, y = jnp.sum(v * w, axis=-1), jnp.sum(jnp.cross(b1_unit, v) * w, axis=-1)
  m = (x == 0.0) & (y == 0.0); return jnp.arctan2(jnp.where(m, 0.0, y), jnp.where(m, 1.0, x))

def make_energy_fn(displacement_fn, system, neighbor_list=None, exclusion_spec=None, **kwargs):
  """Legacy closure-based energy factory."""
  import warnings
  warnings.warn("make_energy_fn is deprecated. Use make_energy_fn_pure.", DeprecationWarning, stacklevel=2)
  if isinstance(system, dict): system = _DictSystemWrapper(system)
  
  def _sba(idx, prm, ic, pc):
    i = jnp.asarray(idx if idx is not None else jnp.zeros((0, ic), dtype=jnp.int32))
    p = jnp.asarray(prm if prm is not None else jnp.zeros((0, pc), dtype=jnp.float32))
    return i, p

  bond_idx, bond_prm = _sba(system.bonds, system.bond_params, 2, 2)
  bond_energy_fn = bonded.make_bond_energy_fn(displacement_fn, bond_idx, bond_prm)
  angle_idx, angle_prm = _sba(system.angles, system.angle_params, 3, 2)
  angle_energy_fn = bonded.make_angle_energy_fn(displacement_fn, angle_idx, angle_prm)
  dih_idx, dih_prm = _sba(system.proper_dihedrals, system.dihedral_params, 4, 3)
  dihedral_energy_fn = bonded.make_dihedral_energy_fn(displacement_fn, dih_idx, dih_prm)
  imp_idx, imp_prm = _sba(system.impropers, system.improper_params, 4, 3)
  improper_energy_fn = bonded.make_dihedral_energy_fn(displacement_fn, imp_idx, imp_prm)
  ub_idx, ub_prm = _sba(system.urey_bradley_bonds, system.urey_bradley_params, 2, 2)
  ub_energy_fn = bonded.make_bond_energy_fn(displacement_fn, ub_idx, ub_prm)

  charges, sigmas, epsilons = system.charges, jnp.maximum(system.sigmas, 1e-6), system.epsilons
  box, use_pbc = kwargs.get("box"), kwargs.get("use_pbc", False)
  pme_alpha, cutoff = kwargs.get("pme_alpha", 0.34), kwargs.get("cutoff_distance", 9.0)
  pme_grid = kwargs.get("pme_grid_points", 64)

  # SPME Setup
  _spme = None
  if use_pbc and box is not None:
    box_arr = jnp.asarray(box)
    gs = kwargs.get("pme_grid_spacing") or float(jnp.mean(box_arr.astype(jnp.float64))) / float(max(int(pme_grid), 1))
    _spme = pme.make_spme_energy_fn(box_arr, alpha=float(pme_alpha), grid_spacing=gs)

  def total_energy(r, neighbor=None, **kwargs_run):
    e_total = bond_energy_fn(r) + angle_energy_fn(r) + ub_energy_fn(r) + dihedral_energy_fn(r) + improper_energy_fn(r)
    # This legacy path is kept for parity; it uses full N^2 direct space if no neighbor list
    dr = space.map_product(displacement_fn)(r, r); dist = space.distance(dr); ds = dist + 1e-6
    mask = 1.0 - jnp.eye(r.shape[0])
    
    # Check for exclusions in both legacy and modern formats
    excl_mask = getattr(system, "exclusion_mask", None)
    if excl_mask is None:
        excl_mask = getattr(system, "dense_excl_scale_vdw", None)
    
    if excl_mask is not None:
        mask *= excl_mask
    
    e_lj = 0.5 * jnp.sum(energy.lennard_jones(dist, 0.5*(sigmas[:,None]+sigmas[None,:]), jnp.sqrt(epsilons[:,None]*epsilons[None,:])) * mask)
    e_elec = 0.5 * jnp.sum(332.0637 * (charges[:,None]*charges[None,:]) / ds * jax.scipy.special.erfc(pme_alpha * dist) * mask)
    if _spme:
      e_elec += _spme(r, charges, jnp.ones(r.shape[0], bool)) + pme.spme_background_energy(charges, jnp.ones(r.shape[0], bool), pme_alpha, jnp.asarray(box))
      # Note: Legacy skip correction if no explicit bonds found
    return e_total + e_lj + e_elec

  return total_energy

def make_energy_fn_pure(
  displacement_fn: space.DisplacementFn,
  physics_system: PhysicsSystem,
  cutoff_distance: float = 9.0,
  pme_grid_points: int | Array = 64,
  pme_alpha: float = 0.34,
  pme_grid_spacing: float | None = None,
  strict_parameterization: bool = True,
  tile_size: int = 128,
) -> tuple[EnergyParams, Any]:
  """Exportable energy factory with memory-efficient kernels."""
  def _sba(idx, prm, ic, pc):
    i = jnp.asarray(idx if idx is not None else jnp.zeros((0, ic), dtype=jnp.int32))
    p = jnp.asarray(prm if prm is not None else jnp.zeros((0, pc), dtype=jnp.float32))
    return i, p

  bond_idx, bond_prm = _sba(physics_system.bonds, physics_system.bond_params, 2, 2)
  bond_energy_fn = bonded.make_bond_energy_fn(displacement_fn, bond_idx, bond_prm)
  angle_idx, angle_prm = _sba(physics_system.angles, physics_system.angle_params, 3, 2)
  angle_energy_fn = bonded.make_angle_energy_fn(displacement_fn, angle_idx, angle_prm)
  dih_idx, dih_prm = _sba(physics_system.dihedrals, physics_system.dihedral_params, 4, 3)
  dihedral_energy_fn = bonded.make_dihedral_energy_fn(displacement_fn, dih_idx, dih_prm)
  imp_idx, imp_prm = _sba(physics_system.impropers, physics_system.improper_params, 4, 3)
  improper_energy_fn = bonded.make_dihedral_energy_fn(displacement_fn, imp_idx, imp_prm)
  ub_idx, ub_prm = _sba(physics_system.urey_bradley_bonds, physics_system.urey_bradley_params, 2, 2)
  ub_energy_fn = bonded.make_bond_energy_fn(displacement_fn, ub_idx, ub_prm)

  box_arr = physics_system.box_size if physics_system.box_size is not None else jnp.zeros(3)
  gs = float(pme_grid_spacing) if pme_grid_spacing is not None else float(jnp.mean(box_arr.astype(jnp.float64))) / float(max(int(pme_grid_points), 1))
  _spme_fn = pme.make_spme_energy_fn(box_arr, alpha=float(pme_alpha), grid_spacing=gs)
  COULOMB_CONSTANT = 332.0636

  def total_energy_pure_impl(params: EnergyParams, system: PhysicsSystem) -> Array:
    r, charges_p = system.positions, jnp.asarray(params.params['charges'])
    sigmas_p, epsilons_p = jnp.maximum(jnp.asarray(params.params['sigmas']), 1e-6), jnp.asarray(params.params['epsilons'])
    
    e_total = bond_energy_fn(r) + angle_energy_fn(r) + ub_energy_fn(r) + dihedral_energy_fn(r) + improper_energy_fn(r)
    e_lj = chunked_lj_energy(r, sigmas_p, epsilons_p, displacement_fn, tile_size)
    e_direct = chunked_coulomb_energy(r, charges_p, displacement_fn, float(pme_alpha), COULOMB_CONSTANT, tile_size)
    e_recip = _spme_fn(r, charges_p, jnp.ones(charges_p.shape[0], bool)) + pme.spme_background_energy(charges_p, jnp.ones(charges_p.shape[0], bool), float(pme_alpha), jnp.asarray(system.box_size))
    
    # Topological Corrections
    e_corr_vdw = e_corr_elec_direct = e_corr_elec_recip = 0.0
    excl_idx, excl_sv, excl_se = system.excl_indices, system.excl_scales_vdw, system.excl_scales_elec
    if excl_idx is not None and excl_idx.shape[0] > 0:
      def pair_corr(ri, rj, si, sj, ei, ej, qi, qj, sv, se):
        dr = displacement_fn(ri, rj); d = jnp.sqrt(jnp.sum(dr**2) + 1e-12); ds = d + 1e-6
        s, e = 0.5*(si+sj), jnp.sqrt(ei*ej)
        ev = 4.0*e*((s/ds)**12-(s/ds)**6)
        ed = COULOMB_CONSTANT*(qi*qj/ds)*jax.scipy.special.erfc(pme_alpha*ds)
        er = COULOMB_CONSTANT*(qi*qj/ds)*jax.scipy.special.erf(pme_alpha*ds)
        return (1.0-sv)*ev, (1.0-se)*ed, (1.0-se)*er
      v_corr = jax.vmap(jax.vmap(pair_corr, (None,0,None,0,None,0,None,0,0,0)), (0,0,0,0,0,0,0,0,0,0))
      ev, ed, er = v_corr(r, r[jnp.where(excl_idx>=0, excl_idx, 0)], sigmas_p, sigmas_p[jnp.where(excl_idx>=0, excl_idx, 0)], epsilons_p, epsilons_p[jnp.where(excl_idx>=0, excl_idx, 0)], charges_p, charges_p[jnp.where(excl_idx>=0, excl_idx, 0)], excl_sv, excl_se)
      m = excl_idx >= 0; e_corr_vdw += 0.5*jnp.sum(jnp.where(m, ev, 0.0)); e_corr_elec_direct += 0.5*jnp.sum(jnp.where(m, ed, 0.0)); e_corr_elec_recip += 0.5*jnp.sum(jnp.where(m, er, 0.0))

    if system.dense_excl_scale_vdw is not None or system.dense_excl_scale_elec is not None:
      dmv, dme = system.dense_excl_scale_vdw, system.dense_excl_scale_elec
      dr_m = jax.vmap(jax.vmap(displacement_fn, (None,0)), (0,None))(r, r); dist = jnp.sqrt(jnp.sum(dr_m**2, axis=-1)+1e-12); ds = dist+1e-6
      q_ij, sig_ij, eps_ij = charges_p[:,None]*charges_p[None,:], 0.5*(sigmas_p[:,None]+sigmas_p[None,:]), jnp.sqrt(epsilons_p[:,None]*epsilons_p[None,:]); md = 1.0-jnp.eye(r.shape[0])
      if dmv is not None: e_corr_vdw += 0.5*jnp.sum(4.0*eps_ij*((sig_ij/ds)**12-(sig_ij/ds)**6)*(1.0-dmv)*md)
      if dme is not None:
        e_corr_elec_direct += 0.5*jnp.sum(COULOMB_CONSTANT*(q_ij/ds)*jax.scipy.special.erfc(pme_alpha*ds)*(1.0-dme)*md)
        e_corr_elec_recip += 0.5*jnp.sum(COULOMB_CONSTANT*(q_ij/ds)*jax.scipy.special.erf(pme_alpha*ds)*(1.0-dme)*md)

    return e_total + (e_lj - e_corr_vdw) + (e_direct - e_corr_elec_direct) + (e_recip - e_corr_elec_recip)

  def fn(p: EnergyParams, r: Array) -> Array:
    return total_energy_pure_impl(p, physics_system.replace(positions=r))

  params = EnergyParams(params={
      'charges': physics_system.charges,
      'sigmas': jnp.maximum(physics_system.sigmas, 1e-6),
      'epsilons': physics_system.epsilons
  })
  return params, fn
