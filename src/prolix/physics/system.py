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
      n_atoms = int(jnp.asarray(system["charges"]).shape[0])
      if n_atoms == 0:
          raise ValueError("make_energy_fn: system['charges'] is empty — cannot infer n_atoms")
      positions = kwargs.get("positions", jnp.zeros((n_atoms, 3)))
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
  if dih_idx is None or dih_idx.size == 0:
      dih_idx = getattr(system, "proper_dihedrals", None)

  dih_prm = getattr(system, "dihedral_params", None)
  if dih_prm is None or dih_prm.size == 0:
      dih_prm = getattr(system, "proper_dihedral_params", None)

  imp_idx = getattr(system, "impropers", None)
  if imp_idx is None or imp_idx.size == 0:
      imp_idx = getattr(system, "improper_dihedrals", None)

  imp_prm = getattr(system, "improper_params", None)
  if imp_prm is None or imp_prm.size == 0:
      imp_prm = getattr(system, "improper_dihedral_params", None)

  _dihedral_energy_fn = None
  if dih_idx is not None and dih_idx.size > 0:
      _dihedral_energy_fn = bonded.make_dihedral_energy_fn(displacement_fn, dih_idx)

  _improper_energy_fn = None
  if imp_idx is not None and imp_idx.size > 0:
      # Selection logic for harmonic vs periodic impropers
      # With 3D params, we check the last dimension
      if imp_prm is not None and imp_prm.shape[-1] == 2:
          _improper_energy_fn = bonded.make_harmonic_improper_energy_fn(displacement_fn, imp_idx)
      else:
          _improper_energy_fn = bonded.make_dihedral_energy_fn(displacement_fn, imp_idx)

  dihedral_energy_fn_bound = lambda r, n=None: _dihedral_energy_fn(r, dih_prm) if _dihedral_energy_fn is not None else 0.0
  improper_energy_fn_bound = lambda r, n=None: _improper_energy_fn(r, imp_prm) if _improper_energy_fn is not None else 0.0

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

  # Radii handling
  radii = getattr(system, "radii", None)
  scaled_radii = getattr(system, "scaled_radii", None)

  if radii is None or radii.size == 0:
      from proxide import assign_mbondi2_radii, assign_obc2_scaling_factors
      # Ensure we have atom names available
      atom_names = getattr(system, "atom_names", None)
      if atom_names is not None:
           radii = jnp.array(assign_mbondi2_radii(list(atom_names), system.bonds))
           scaled_radii = jnp.array(assign_obc2_scaling_factors(list(atom_names)))

  if radii is None: radii = jnp.ones_like(charges)
  box, use_pbc = kwargs.get("box"), kwargs.get("use_pbc", False)
  default_pme_alpha = 0.34 if use_pbc else 0.0
  pme_alpha = kwargs.get("pme_alpha", default_pme_alpha)
  cutoff = kwargs.get("cutoff_distance", kwargs.get("cutoff", 9.0))
  COULOMB_CONSTANT = 332.0637

  # Sparse Non-bonded Setup
  from prolix.physics import neighbor_list
  from prolix.physics.optimization import (
    chunked_coulomb_energy,
    chunked_coulomb_energy_nl,
    chunked_lj_energy,
    chunked_lj_energy_nl,
  )

  excl_idx = getattr(system, "excl_indices", None)
  n_atoms = system.charges.shape[0]

  # Build exception pair energy fn from ExclusionSpec (explicit AMBER 1-4 overrides)
  _exception_energy_fn = None
  if exclusion_spec is not None and exclusion_spec.exception_pairs.shape[0] > 0:
      _exception_energy_fn = bonded.make_exception_pair_energy_fn(
          displacement_fn,
          exclusion_spec.exception_pairs,
          exclusion_spec.exception_sigmas,
          exclusion_spec.exception_epsilons,
          exclusion_spec.exception_chargeprods,
      )
  exception_energy_fn_bound = lambda r, n=None: _exception_energy_fn(r) if _exception_energy_fn is not None else 0.0

  if exclusion_spec is not None and (excl_idx is None or excl_idx.size == 0):
      excl_idx, excl_sv, excl_se = neighbor_list.map_exclusions_to_dense_padded(exclusion_spec, max_exclusions=32)
  else:
      excl_sv = getattr(system, "excl_scales_vdw", None)
      excl_se = getattr(system, "excl_scales_elec", None)
      if excl_idx is None:
          excl_idx = jnp.zeros((n_atoms, 0), dtype=jnp.int32)
      if excl_sv is None:
          excl_sv = jnp.zeros_like(excl_idx, dtype=jnp.float32)
      if excl_se is None:
          excl_se = jnp.zeros_like(excl_idx, dtype=jnp.float32)

  # jax.debug.print("DEBUG exclusions: excl_idx shape={s}", s=excl_idx.shape)
  # if excl_idx.size > 0:
  #     jax.debug.print("DEBUG min/max scales_vdw: {min}, {max}", min=jnp.min(excl_sv), max=jnp.max(excl_sv))
  # SPME Setup
  _spme = None
  if use_pbc and box is not None:
    box_arr = jnp.asarray(box)
    pme_grid = kwargs.get("pme_grid_points", 64)
    gs = kwargs.get("pme_grid_spacing") or float(jnp.mean(box_arr.astype(jnp.float64))) / float(max(int(pme_grid), 1))
    _spme = pme.make_spme_energy_fn(box_arr, alpha=float(pme_alpha), grid_spacing=gs)

  def lj_energy_fn_bound(r, neighbor=None):
    if neighbor is not None:
        nb_idx = getattr(neighbor, "idx", neighbor)
        e_lj = chunked_lj_energy_nl(r, sigmas, epsilons, excl_idx, excl_sv, nb_idx, displacement_fn, cutoff, 128)
    else:
        e_lj = chunked_lj_energy(r, sigmas, epsilons, excl_idx, excl_sv, displacement_fn, cutoff, 128)

    if use_pbc and box is not None:
        mask = getattr(system, "atom_mask", jnp.ones(r.shape[0], bool))
        e_lj += explicit_corrections.lj_dispersion_tail_energy(jnp.asarray(box), sigmas, epsilons, cutoff, mask)
    return e_lj

  def electrostatics_energy_fn_bound(r, neighbor=None):
    if implicit_solvent:
        # OBC2 descreening MUST include 1-2 and 1-3 neighbors.
        # Currently we use a dense self-interaction mask (1.0 - eye).
        # TODO: Move to sparse neighbor-list based descreening for large systems.
        n_atoms = r.shape[0]
        gb_mask = 1.0 - jnp.eye(n_atoms)

        e_gb, born_radii = generalized_born.compute_gb_energy(
            r, charges, radii,
            mask=gb_mask,
            energy_mask=None,
            scaled_radii=scaled_radii
        )
        if neighbor is not None:
            nb_idx = getattr(neighbor, "idx", neighbor)
            e_direct = chunked_coulomb_energy_nl(r, charges, excl_idx, excl_se, nb_idx, displacement_fn, pme_alpha, COULOMB_CONSTANT, cutoff, 128)
        else:
            e_direct = chunked_coulomb_energy(r, charges, excl_idx, excl_se, displacement_fn, pme_alpha, COULOMB_CONSTANT, cutoff, 128)

        return e_gb, e_direct, born_radii

    if neighbor is not None:
        nb_idx = getattr(neighbor, "idx", neighbor)
        e_direct = chunked_coulomb_energy_nl(r, charges, excl_idx, excl_se, nb_idx, displacement_fn, pme_alpha, COULOMB_CONSTANT, cutoff, 128)
    else:
        e_direct = chunked_coulomb_energy(r, charges, excl_idx, excl_se, displacement_fn, pme_alpha, COULOMB_CONSTANT, cutoff, 128)

    if _spme:
      mask = getattr(system, "atom_mask", jnp.ones(r.shape[0], bool))
      e_direct += _spme(r, charges, mask) + pme.spme_background_energy(charges, mask, pme_alpha, jnp.asarray(box))
    return e_direct

  def total_energy(r, neighbor=None, **kwargs_run):
    elec = electrostatics_energy_fn_bound(r, neighbor)
    if implicit_solvent:
        e_elec = elec[0] + elec[1]
        born_radii = elec[2]
        e_ace = jnp.sum(jnp.where(getattr(system, "atom_mask", jnp.ones(r.shape[0], bool)), generalized_born.compute_ace_nonpolar_energy(radii, born_radii), 0.0))
        e_elec += e_ace
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
        exception_energy_fn_bound(r, neighbor) +
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
          "exception": exception_energy_fn_bound,
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

  _dihedral_energy_fn = None
  if physics_system.dihedrals is not None:
      _dihedral_energy_fn = bonded.make_dihedral_energy_fn(displacement_fn, dih_idx)
  dihedral_energy_fn = lambda r, p: _dihedral_energy_fn(r, p) if _dihedral_energy_fn is not None else 0.0

  _improper_energy_fn = None
  if physics_system.impropers is not None:
      _improper_energy_fn = bonded.make_dihedral_energy_fn(displacement_fn, imp_idx)
  improper_energy_fn = lambda r, p: _improper_energy_fn(r, p) if _improper_energy_fn is not None else 0.0

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
    excl_idx = physics_system.excl_indices
    excl_sv, excl_se = physics_system.excl_scales_vdw, physics_system.excl_scales_elec

    if neighbor is not None:
      nb_idx = getattr(neighbor, "idx", neighbor)
      e_lj = chunked_lj_energy_nl(r, sigmas_p, epsilons_p, excl_idx, excl_sv, nb_idx, displacement_fn, cutoff_distance, tile_size)
      e_direct = chunked_coulomb_energy_nl(r, charges_p, excl_idx, excl_se, nb_idx, displacement_fn, pme_alpha, COULOMB_CONSTANT, cutoff_distance, tile_size)
    else:
      e_lj = chunked_lj_energy(r, sigmas_p, epsilons_p, excl_idx, excl_sv, displacement_fn, cutoff_distance, tile_size)
      e_direct = chunked_coulomb_energy(r, charges_p, excl_idx, excl_se, displacement_fn, pme_alpha, COULOMB_CONSTANT, cutoff_distance, tile_size)

    # PME Reciprocal
    spme_fn = lambda pos, q, m: pme.spme_energy_with_forces(pos, q, m, box_arr, grid_dims, pme_alpha, 4)
    e_recip = spme_fn(r, charges_p, physics_system.atom_mask) + pme.spme_background_energy(charges_p, physics_system.atom_mask, pme_alpha, box_arr)

    # ACE nonpolar term if implicit solvent
    e_ace = 0.0
    if hasattr(physics_system, "implicit_solvent") and physics_system.implicit_solvent:
        # We need born radii which require GB, but here we are in a simplified pure impl.
        # Assuming for pure implementation GB calculation is handled or ACE uses a placeholder.
        # Based on current structure, might need to call GB energy here or pass it if precomputed.
        pass

    e_tail = explicit_corrections.lj_dispersion_tail_energy(box_arr, sigmas_p, epsilons_p, cutoff_distance, physics_system.atom_mask)
    return e_total + e_lj + e_direct + e_recip + e_tail + e_ace

  params = DifferentiableParams.from_system(physics_system)
  if pme_alpha is not None:
      params = eqx.tree_at(lambda p: p.pme_alpha, params, jnp.asarray(pme_alpha))
  return params, total_energy_pure_impl
