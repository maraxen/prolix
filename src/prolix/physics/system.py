from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax_md import space
from jaxtyping import Array

from prolix.physics import bonded, cmap, explicit_corrections, generalized_born, pme
from prolix.physics.bonded import compute_dihedral_angles
from prolix.typing import DifferentiableParams, PhysicsSystem


def make_energy_fn(
  displacement_fn: space.DisplacementFn,
  system: PhysicsSystem | dict,
  cutoff_distance: float = 9.0,
  return_decomposed: bool = False,
  implicit_solvent: bool = False,
  exclusion_spec: Any = None,
  **kwargs
) -> Callable | dict:
  """Standard factory for non-pure energy functions."""
  
  if isinstance(system, dict):
      positions = kwargs.get("positions", jnp.zeros((0, 3)))
      box = kwargs.get("box")
      system = PhysicsSystem.from_dict(system, positions, box_size=box, cutoff_distance=cutoff_distance)
  
  def _sba(idx, prm, ic, pc):
    idx = jnp.asarray(idx if idx is not None else jnp.zeros((0, ic), dtype=jnp.int32))
    prm = jnp.asarray(prm if prm is not None else jnp.zeros((0, pc), dtype=jnp.float32))
    return idx, prm

  bond_idx, bond_prm = _sba(getattr(system, "bonds", None), getattr(system, "bond_params", None), 2, 2)
  _bond_energy_fn = bonded.make_bond_energy_fn(displacement_fn, bond_idx)
  bond_energy_fn_bound = lambda r, n=None: _bond_energy_fn(r, bond_prm)
  
  angle_idx, angle_prm = _sba(getattr(system, "angles", None), getattr(system, "angle_params", None), 3, 2)
  _angle_energy_fn = bonded.make_angle_energy_fn(displacement_fn, angle_idx)
  angle_energy_fn_bound = lambda r, n=None: _angle_energy_fn(r, angle_prm)
  
  dih_idx = getattr(system, "dihedrals", None)
  if dih_idx is None:
      dih_idx = getattr(system, "proper_dihedrals", None)
      
  dih_prm = getattr(system, "dihedral_params", None)
  if dih_prm is None:
      dih_prm = getattr(system, "proper_dihedral_params", None)
  
  if dih_idx is not None and dih_prm is not None:
      dih_idx, dih_prm = _sba(dih_idx, dih_prm, 4, 3)
      _dihedral_energy_fn = bonded.make_dihedral_energy_fn(displacement_fn, dih_idx)
      dihedral_energy_fn_bound = lambda r, n=None: _dihedral_energy_fn(r, dih_prm)
  else:
      dihedral_energy_fn_bound = lambda r, n=None: 0.0

  imp_idx = getattr(system, "impropers", None)
  imp_prm = getattr(system, "improper_params", None)
  if imp_idx is not None:
      imp_idx, imp_prm = _sba(imp_idx, imp_prm, 4, 3)
      _improper_energy_fn = bonded.make_dihedral_energy_fn(displacement_fn, imp_idx)
      improper_energy_fn_bound = lambda r, n=None: _improper_energy_fn(r, imp_prm)
  else:
      improper_energy_fn_bound = lambda r, n=None: 0.0
  
  ub_idx, ub_prm = _sba(getattr(system, "urey_bradley_bonds", None), getattr(system, "urey_bradley_params", None), 2, 2)
  _ub_energy_fn = bonded.make_bond_energy_fn(displacement_fn, ub_idx)
  ub_energy_fn_bound = lambda r, n=None: _ub_energy_fn(r, ub_prm)

  # CMAP
  if hasattr(system, "cmap_torsions") and system.cmap_torsions is not None:
      def _cmap_fn(r, n=None):
          phi = compute_dihedral_angles(r, system.cmap_torsions[:, 0:4], displacement_fn)
          psi = compute_dihedral_angles(r, system.cmap_torsions[:, 1:5], displacement_fn)
          return cmap.compute_cmap_energy(phi, psi, system.cmap_indices, system.cmap_energy_grids)
      cmap_energy_fn_bound = _cmap_fn
  else:
      cmap_energy_fn_bound = lambda r, n=None: 0.0

  charges, sigmas, epsilons = system.charges, jnp.maximum(system.sigmas, 1e-6), system.epsilons
  box, use_pbc = kwargs.get("box"), kwargs.get("use_pbc", False)
  default_pme_alpha = 0.34 if use_pbc else 0.0
  pme_alpha = kwargs.get("pme_alpha", default_pme_alpha)
  cutoff = kwargs.get("cutoff_distance", 9.0)
  COULOMB_CONSTANT = 332.0637

  # Sparse Non-bonded Setup
  from prolix.physics.optimization import (
    chunked_coulomb_energy,
    chunked_coulomb_energy_nl,
    chunked_lj_energy_nl,
  )
  
  # SPME Setup
  _spme = None
  if use_pbc and box is not None:
    box_arr = jnp.asarray(box)
    pme_grid = kwargs.get("pme_grid_points", 64)
    gs = kwargs.get("pme_grid_spacing") or float(jnp.mean(box_arr.astype(jnp.float64))) / float(max(int(pme_grid), 1))
    _spme = pme.make_spme_energy_fn(box_arr, alpha=float(pme_alpha), grid_spacing=gs)

  # Sparse Correction Function
  def _compute_sparse_corrections(r, qi, si, ei, excl_indices=None, excl_scales_vdw=None, excl_scales_elec=None, exclusion_spec=None):
    total_ev = total_ed = total_er = 0.0

    def pair_corr(i, j, sv, se):
      dr = displacement_fn(r[i], r[j])
      d = jnp.sqrt(jnp.sum(dr**2) + 1e-12)
      ds = jnp.where(d < 1e-10, 1.0, d)
      si_j, ei_j = 0.5*(si[i]+si[j]), jnp.sqrt(ei[i]*ei[j])
      
      inv_ds = 1.0 / ds
      
      # DEBUG 1-4
      # jax.debug.print("DEBUG pair_corr: i={i} j={j} qi={qi} qj={qj} ds={ds}", i=i, j=j, qi=qi[i], qj=qi[j], ds=ds)

      inv_r6 = (si_j * inv_ds)**6
      ev = 4.0 * ei_j * (inv_r6**2 - inv_r6)
      
      erfc_val = jax.scipy.special.erfc(pme_alpha * ds)
      erf_val = jax.scipy.special.erf(pme_alpha * ds)
      
      ed = COULOMB_CONSTANT * (qi[i] * qi[j] * inv_ds) * erfc_val
      er = COULOMB_CONSTANT * (qi[i] * qi[j] * inv_ds) * erf_val
      
      # Mask out i=j case and apply cutoff
      mask = (i != j) & (d > 1e-10)
      if cutoff > 0:
          mask &= (ds < cutoff)
          
      ev = jnp.where(mask, ev, 0.0)
      # DEBUG: print first few ev values if they are suspiciously large
      # jax.debug.print("DEBUG pair_corr: i={i} j={j} ev={ev} ed={ed}", i=i, j=j, ev=ev, ed=ed)
      jax.debug.print("DEBUG pair_corr: COULOMB_CONSTANT={c}", c=COULOMB_CONSTANT)
      
      ed = jnp.where(mask, ed, 0.0)
      er = jnp.where(mask, er, 0.0)
      
      return (1.0-sv)*ev, (1.0-se)*ed, (1.0-se)*er

    # 1. Base corrections from system.excl_indices (N, max_excl)
    if excl_indices is not None and excl_indices.shape[0] > 0:
        def _compute_excl(i, j_idx, sv, se):
            return jnp.where(j_idx >= 0, pair_corr(i, j_idx, sv, se)[0], 0.0), \
                   jnp.where(j_idx >= 0, pair_corr(i, j_idx, sv, se)[1], 0.0), \
                   jnp.where(j_idx >= 0, pair_corr(i, j_idx, sv, se)[2], 0.0)

        v_corr = jax.vmap(jax.vmap(_compute_excl, (None, 0, None, None)), (0, 0, 0, 0))
        ev, ed, er = v_corr(jnp.arange(r.shape[0]), excl_indices, excl_scales_vdw, excl_scales_elec)
        total_ev += 0.5 * jnp.sum(ev)
        total_ed += 0.5 * jnp.sum(ed)
        total_er += 0.5 * jnp.sum(er)

    # 2. Pair-list exclusions (M, 2)
    if exclusion_spec is not None:
        def _compute_pairs(pairs, sv, se):
            if pairs is None or pairs.shape[0] == 0:
                return 0.0, 0.0, 0.0
            ev, ed, er = jax.vmap(lambda p: pair_corr(p[0], p[1], sv, se))(pairs)
            return jnp.sum(ev), jnp.sum(ed), jnp.sum(er)
        
        ev1, ed1, er1 = _compute_pairs(getattr(exclusion_spec, "idx_12_13", None), 0.0, 0.0)
        jax.debug.print("DEBUG: 12_13 VdW={v} Elec={e}", v=ev1, e=ed1)
        total_ev += ev1; total_ed += ed1; total_er += er1
        
        scale_14_vdw = getattr(exclusion_spec, "scale_14_vdw", 0.0)
        scale_14_elec = getattr(exclusion_spec, "scale_14_elec", 0.0)
        jax.debug.print("DEBUG: 1-4 Scales: VdW={v} Elec={e}", v=scale_14_vdw, e=scale_14_elec)
        ev2, ed2, er2 = _compute_pairs(getattr(exclusion_spec, "idx_14", None),
                                       scale_14_vdw,
                                       scale_14_elec)
        jax.debug.print("DEBUG: 14 VdW={v} Elec={e}", v=ev2, e=ed2)
        total_ev += ev2; total_ed += ed2; total_er += er2

    # 3. Dense fallback
    if getattr(system, "dense_excl_scale_vdw", None) is not None:
        dmv, dme = system.dense_excl_scale_vdw, system.dense_excl_scale_elec
        dr_m = jax.vmap(jax.vmap(displacement_fn, (None, 0)), (0, None))(r, r)
        ds = jnp.sqrt(jnp.sum(dr_m**2, axis=-1) + 1e-12)
        q_ij = qi[:, None] * qi[None, :]
        sig_ij = 0.5 * (si[:, None] + si[None, :])
        eps_ij = jnp.sqrt(ei[:, None] * ei[None, :])
        md = 1.0 - jnp.eye(r.shape[0])
        c_mask = (ds < cutoff) if cutoff > 0 else jnp.ones_like(ds, dtype=bool)
        if dmv is not None:
            total_ev += 0.5 * jnp.sum(4.0 * eps_ij * ((sig_ij/ds)**12 - (sig_ij/ds)**6) * (1.0 - dmv) * md * c_mask)
        if dme is not None:
            total_ed += 0.5 * jnp.sum(COULOMB_CONSTANT * (q_ij/ds) * jax.scipy.special.erfc(pme_alpha * ds) * (1.0 - dme) * md * c_mask)
            total_er += 0.5 * jnp.sum(COULOMB_CONSTANT * (q_ij/ds) * jax.scipy.special.erf(pme_alpha * ds) * (1.0 - dme) * md)
            
    return total_ev, total_ed, total_er

  def lj_energy_fn_bound(r, neighbor=None):
    if neighbor is not None:
        nb_idx = getattr(neighbor, "idx", neighbor)
        e_lj = chunked_lj_energy_nl(r, sigmas, epsilons, nb_idx, displacement_fn, cutoff)
    else:
        dr = space.map_product(displacement_fn)(r, r)
        dist = jnp.sqrt(jnp.sum(dr**2, axis=-1) + 1e-12)
        
        # Mask out diagonal
        mask = (1.0 - jnp.eye(r.shape[0]))
        # Use a safe distance for powers
        dist_safe = jnp.where(dist < 1e-10, 1.0, dist)
        
        if cutoff > 0: mask = mask * (dist < cutoff)
        
        sig_ij = 0.5 * (sigmas[:, None] + sigmas[None, :])
        eps_ij = jnp.sqrt(epsilons[:, None] * epsilons[None, :])
        
        inv_r6 = jnp.where(mask > 0, (sig_ij / dist_safe)**6, 0.0)
        # Mask the energy where distance was zero (diagonal)
        e_lj_vals = 4.0 * eps_ij * (inv_r6**2 - inv_r6)
        e_lj = 0.5 * jnp.sum(e_lj_vals * mask)
    
    # Apply Sparse Corrections
    c_vdw, c_elec_d, _ = _compute_sparse_corrections(r, charges, sigmas, epsilons,
                                             excl_indices=getattr(system, "excl_indices", None),
                                             excl_scales_vdw=getattr(system, "excl_scales_vdw", None),
                                             excl_scales_elec=getattr(system, "excl_scales_elec", None),
                                             exclusion_spec=exclusion_spec)
    jax.debug.print("DEBUG: LJ={lj} Corr={c}", lj=e_lj, c=c_vdw)
    e_lj -= c_vdw
    
    if use_pbc and box is not None:
        mask = getattr(system, "atom_mask", jnp.ones(r.shape[0], bool))
        e_lj += explicit_corrections.lj_dispersion_tail_energy(jnp.asarray(box), sigmas, epsilons, cutoff, mask)
    return e_lj

  def electrostatics_energy_fn_bound(r, neighbor=None):
    if implicit_solvent:
        radii = getattr(system, "radii", None)
        if radii is None: radii = jnp.ones_like(charges)
        e_gb, born_radii = generalized_born.compute_gb_energy(
            r, charges, radii,
            mask=None,
            energy_mask=None,
            scaled_radii=getattr(system, "scaled_radii", None)
        )
        if neighbor is not None:
            nb_idx = getattr(neighbor, "idx", neighbor)
            e_direct = chunked_coulomb_energy_nl(r, charges, nb_idx, displacement_fn, pme_alpha, COULOMB_CONSTANT, cutoff)
        else:
            e_direct = chunked_coulomb_energy(r, charges, displacement_fn, pme_alpha, COULOMB_CONSTANT, cutoff)
            
        _, c_elec_d, _ = _compute_sparse_corrections(r, charges, sigmas, epsilons,
                                                    excl_indices=getattr(system, "excl_indices", None),
                                                    excl_scales_vdw=getattr(system, "excl_scales_vdw", None),
                                                    excl_scales_elec=getattr(system, "excl_scales_elec", None),
                                                    exclusion_spec=exclusion_spec)
        e_direct -= c_elec_d
        return e_gb, e_direct, born_radii
    if neighbor is not None:
        nb_idx = getattr(neighbor, "idx", neighbor)
        e_direct = chunked_coulomb_energy_nl(r, charges, nb_idx, displacement_fn, pme_alpha, COULOMB_CONSTANT, cutoff)
    else:
        e_direct = chunked_coulomb_energy(r, charges, displacement_fn, pme_alpha, COULOMB_CONSTANT, cutoff)

    _, c_elec_d, _ = _compute_sparse_corrections(r, charges, sigmas, epsilons,
                                                excl_indices=getattr(system, "excl_indices", None),
                                                excl_scales_vdw=getattr(system, "excl_scales_vdw", None),
                                                excl_scales_elec=getattr(system, "excl_scales_elec", None),
                                                exclusion_spec=exclusion_spec)
    e_direct -= c_elec_d

    if _spme:
      mask = getattr(system, "atom_mask", jnp.ones(r.shape[0], bool))
      e_direct += _spme(r, charges, mask) + pme.spme_background_energy(charges, mask, pme_alpha, jnp.asarray(box))
    return e_direct



  def total_energy(r, neighbor=None, **kwargs_run):
    elec = electrostatics_energy_fn_bound(r, neighbor)
    if implicit_solvent:
        e_elec = elec[0] + elec[1]
    else:
        e_elec = elec
    return (
        bond_energy_fn_bound(r, neighbor) +
        angle_energy_fn_bound(r, neighbor) +
        ub_energy_fn_bound(r, neighbor) +
        dihedral_energy_fn_bound(r, neighbor) +
        improper_energy_fn_bound(r, neighbor) +
        cmap_energy_fn_bound(r, neighbor) +
        lj_energy_fn_bound(r, neighbor) +
        e_elec
    )

  if return_decomposed:
      # Nonpolar energy fn
      def _nonpolar_fn(r, born_radii=None):
          if implicit_solvent and hasattr(system, "radii") and system.radii is not None:
              return jnp.sum(generalized_born.compute_ace_nonpolar_energy(system.radii, born_radii))
          return 0.0

      res = {
          "bond": bond_energy_fn_bound,
          "angle": angle_energy_fn_bound,
          "dihedral": dihedral_energy_fn_bound,
          "improper": improper_energy_fn_bound,
          "urey_bradley": ub_energy_fn_bound,
          "cmap": cmap_energy_fn_bound,
          "lj": lj_energy_fn_bound,
          "electrostatics": electrostatics_energy_fn_bound,
          "nonpolar": _nonpolar_fn,
          "total": total_energy,
      }
      return res
  return total_energy


def make_energy_fn_pure(
  displacement_fn: space.DisplacementFn,
  physics_system: PhysicsSystem,
  cutoff_distance: float = 9.0,
  tile_size: int = 128,
  pme_grid_points: int = 64,
  pme_alpha: float | None = None,
  **kwargs
) -> tuple[DifferentiableParams, Callable]:
  """Exportable energy factory with memory-efficient kernels."""
  from prolix.physics.optimization import (
    chunked_coulomb_energy,
    chunked_coulomb_energy_nl,
    chunked_lj_energy,
    chunked_lj_energy_nl,
  )

  # Use system's pme_alpha if not provided
  if pme_alpha is None:
    pme_alpha = getattr(physics_system, "pme_alpha", 0.34)

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
  if pme_alpha is not None:
      params = eqx.tree_at(lambda p: p.pme_alpha, params, jnp.asarray(pme_alpha))
  return params, total_energy_pure_impl
