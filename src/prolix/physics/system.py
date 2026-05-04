from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Any, Callable, Tuple, Optional
from jaxtyping import Array, Float
from jax_md import space

from prolix.physics.types import PhysicsSystem, DifferentiableParams

def make_energy_fn(
  displacement_fn: space.DisplacementFn,
  system: PhysicsSystem,
  cutoff_distance: float = 9.0,
  **kwargs
) -> Callable:
  """Standard factory for non-pure energy functions."""
  from prolix.physics import bonded, pme, optimization
  
  def _sba(idx, prm, ic, pc):
    idx = jnp.asarray(idx if idx is not None else jnp.zeros((0, ic), dtype=jnp.int32))
    prm = jnp.asarray(prm if prm is not None else jnp.zeros((0, pc), dtype=jnp.float32))
    return idx, prm

  bond_idx, bond_prm = _sba(system.bonds, system.bond_params, 2, 2)
  bond_energy_fn = bonded.make_bond_energy_fn(displacement_fn, bond_idx, bond_prm)
  angle_idx, angle_prm = _sba(system.angles, system.angle_params, 3, 2)
  angle_energy_fn = bonded.make_angle_energy_fn(displacement_fn, angle_idx, angle_prm)
  dih_idx, dih_prm = _sba(system.dihedrals, system.dihedral_params, 4, 3)
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
    dr = space.map_product(displacement_fn)(r, r); dist = space.distance(dr); ds = dist + 1e-6
    mask = 1.0 - jnp.eye(r.shape[0])
    
    excl_mask = getattr(system, "exclusion_mask", None)
    if excl_mask is None:
        excl_mask = getattr(system, "dense_excl_scale_vdw", None)
    
    if excl_mask is not None:
        mask *= excl_mask
    
    e_lj = 0.5 * jnp.sum(optimization.energy.lennard_jones(dist, 0.5*(sigmas[:,None]+sigmas[None,:]), jnp.sqrt(epsilons[:,None]*epsilons[None,:])) * mask)
    e_elec = 0.5 * jnp.sum(332.0637 * (charges[:,None]*charges[None,:]) / ds * jax.scipy.special.erfc(pme_alpha * dist) * mask)
    if _spme:
      e_elec += _spme(r, charges, jnp.ones(r.shape[0], bool)) + pme.spme_background_energy(charges, jnp.ones(r.shape[0], bool), pme_alpha, jnp.asarray(box))
    return e_total + e_lj + e_elec

  return total_energy

def make_energy_fn_pure(
  displacement_fn: space.DisplacementFn,
  physics_system: PhysicsSystem,
  cutoff_distance: float = 9.0,
  tile_size: int = 128,
  pme_grid_points: int = 64
) -> tuple[DifferentiableParams, Callable]:
  """Exportable energy factory with memory-efficient kernels."""
  from prolix.physics import bonded, pme, optimization, explicit_corrections
  from prolix.physics.optimization import chunked_lj_energy, chunked_lj_energy_nl, chunked_coulomb_energy, chunked_coulomb_energy_nl

  def _sba(idx, ic):
    return jnp.asarray(idx if idx is not None else jnp.zeros((0, ic), dtype=jnp.int32))

  bond_idx = _sba(physics_system.bonds, 2)
  angle_idx = _sba(physics_system.angles, 3)
  dih_idx = _sba(physics_system.dihedrals, 4)
  imp_idx = _sba(physics_system.impropers, 4)
  ub_idx = _sba(physics_system.urey_bradley_bonds, 2)

  bond_energy_fn = bonded.make_bond_energy_fn(displacement_fn, bond_idx)
  angle_energy_fn = bonded.make_angle_energy_fn(displacement_fn, angle_idx)
  dihedral_energy_fn = bonded.make_dihedral_energy_fn(displacement_fn, dih_idx)
  improper_energy_fn = bonded.make_dihedral_energy_fn(displacement_fn, imp_idx)
  ub_energy_fn = bonded.make_bond_energy_fn(displacement_fn, ub_idx)

  COULOMB_CONSTANT = 332.0637

  # SPME Setup - Precompute grid dims from initial box
  initial_box = physics_system.box_size
  grid_dims = pme.compute_pme_grid_dims(initial_box, jnp.mean(initial_box) / float(max(pme_grid_points, 1)))

  def total_energy_pure_impl(params: DifferentiableParams, r: Array, neighbor: Any = None) -> Array:
    charges_p = params.charges
    sigmas_p, epsilons_p = jnp.maximum(params.sigmas, 1e-6), params.epsilons
    pme_alpha = params.pme_alpha
    box_arr = params.box_size
    
    # Bonded terms
    e_total = (
        bond_energy_fn(r, params.bond_params) +
        angle_energy_fn(r, params.angle_params) +
        ub_energy_fn(r, params.urey_bradley_params) +
        dihedral_energy_fn(r, params.dihedral_params) +
        improper_energy_fn(r, params.improper_params)
    )
    
    # Non-bonded terms
    if neighbor is not None:
      nb_idx = getattr(neighbor, "idx", neighbor)
      e_lj = chunked_lj_energy_nl(r, sigmas_p, epsilons_p, nb_idx, displacement_fn, cutoff_distance, tile_size)
      e_direct = chunked_coulomb_energy_nl(r, charges_p, nb_idx, displacement_fn, pme_alpha, COULOMB_CONSTANT, cutoff_distance, tile_size)
    else:
      e_lj = chunked_lj_energy(r, sigmas_p, epsilons_p, displacement_fn, cutoff_distance, tile_size)
      e_direct = chunked_coulomb_energy(r, charges_p, displacement_fn, pme_alpha, COULOMB_CONSTANT, cutoff_distance, tile_size)
      
    # PME Reciprocal
    spme_fn = lambda pos, q, m: pme.spme_energy_with_forces(pos, q, m, box_arr, grid_dims, pme_alpha, 4)
    e_recip = spme_fn(r, charges_p, physics_system.atom_mask) + pme.spme_background_energy(charges_p, physics_system.atom_mask, pme_alpha, box_arr)
    
    # Topological Corrections
    e_corr_vdw = e_corr_elec_direct = e_corr_elec_recip = 0.0
    excl_idx, excl_sv, excl_se = physics_system.excl_indices, physics_system.excl_scales_vdw, physics_system.excl_scales_elec
    if excl_idx is not None and excl_idx.shape[0] > 0:
      def pair_corr(ri, rj, si, sj, ei, ej, qi, qj, sv, se):
        dr = displacement_fn(ri, rj); d = jnp.sqrt(jnp.sum(dr**2) + 1e-12); ds = d + 1e-6
        s, e = 0.5*(si+sj), jnp.sqrt(ei*ej)
        ev = 4.0*e*((s/ds)**12-(s/ds)**6)
        ed = COULOMB_CONSTANT*(qi*qj/ds)*jax.scipy.special.erfc(pme_alpha*ds)
        er = COULOMB_CONSTANT*(qi*qj/ds)*jax.scipy.special.erf(pme_alpha*ds)
        if cutoff_distance > 0:
            ev = jnp.where(ds < cutoff_distance, ev, 0.0)
            ed = jnp.where(ds < cutoff_distance, ed, 0.0)
        return (1.0-sv)*ev, (1.0-se)*ed, (1.0-se)*er
      v_corr = jax.vmap(jax.vmap(pair_corr, (None,0,None,0,None,0,None,0,0,0)), (0,0,0,0,0,0,0,0,0,0))
      ev, ed, er = v_corr(r, r[jnp.where(excl_idx>=0, excl_idx, 0)], sigmas_p, sigmas_p[jnp.where(excl_idx>=0, excl_idx, 0)], epsilons_p, epsilons_p[jnp.where(excl_idx>=0, excl_idx, 0)], charges_p, charges_p[jnp.where(excl_idx>=0, excl_idx, 0)], excl_sv, excl_se)
      m = excl_idx >= 0; e_corr_vdw += 0.5*jnp.sum(jnp.where(m, ev, 0.0)); e_corr_elec_direct += 0.5*jnp.sum(jnp.where(m, ed, 0.0)); e_corr_elec_recip += 0.5*jnp.sum(jnp.where(m, er, 0.0))

    if physics_system.dense_excl_scale_vdw is not None or physics_system.dense_excl_scale_elec is not None:
      dmv, dme = physics_system.dense_excl_scale_vdw, physics_system.dense_excl_scale_elec
      dr_m = jax.vmap(jax.vmap(displacement_fn, (None,0)), (0,None))(r, r); dist = jnp.sqrt(jnp.sum(dr_m**2, axis=-1)+1e-12); ds = dist+1e-6
      q_ij, sig_ij, eps_ij = charges_p[:,None]*charges_p[None,:], 0.5*(sigmas_p[:,None]+sigmas_p[None,:]), jnp.sqrt(epsilons_p[:,None]*epsilons_p[None,:]); md = 1.0-jnp.eye(r.shape[0])
      c_mask = (ds < cutoff_distance) if cutoff_distance > 0 else jnp.ones_like(ds, dtype=bool)
      if dmv is not None: e_corr_vdw += 0.5*jnp.sum(4.0*eps_ij*((sig_ij/ds)**12-(sig_ij/ds)**6)*(1.0-dmv)*md*c_mask)
      if dme is not None:
        e_corr_elec_direct += 0.5*jnp.sum(COULOMB_CONSTANT*(q_ij/ds)*jax.scipy.special.erfc(pme_alpha*ds)*(1.0-dme)*md*c_mask)
        e_corr_elec_recip += 0.5*jnp.sum(COULOMB_CONSTANT*(q_ij/ds)*jax.scipy.special.erf(pme_alpha*ds)*(1.0-dme)*md)

    e_tail = explicit_corrections.lj_dispersion_tail_energy(box_arr, sigmas_p, epsilons_p, cutoff_distance, physics_system.atom_mask)
    return e_total + (e_lj - e_corr_vdw) + (e_direct - e_corr_elec_direct) + (e_recip - e_corr_elec_recip) + e_tail

  params = DifferentiableParams.from_system(physics_system)
  return params, total_energy_pure_impl
