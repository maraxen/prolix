from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jax_md import space
from jaxtyping import Array

from prolix.physics import bonded, cmap, explicit_corrections, generalized_born, pme
from prolix.physics.bonded import compute_dihedral_angles
from prolix.types.bundles import (
  ANGLE_BUCKETS,
  ATOM_BUCKETS,
  BOND_BUCKETS,
  CMAP_BUCKETS,
  DIHEDRAL_BUCKETS,
  EXCEPTION_BUCKETS,
  EXCL_BUCKETS,
  WATER_BUCKETS,
  MolecularBundle,
  MolecularShapeSpec,
  _bucket_idx,
)
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
  # cutoff_distance is a named parameter (see signature above), so it never
  # lands in kwargs -- kwargs.get("cutoff_distance", ...) could never succeed,
  # silently discarding whatever cutoff_distance the caller passed and
  # falling through to the 9.0 default regardless. A separate `cutoff=`
  # kwarg alias is used by some callers (e.g. test_openmm_parity.py's
  # cutoff=0) and did work via the second fallback -- preserved here by
  # keeping kwargs["cutoff"] as the higher-priority override, with the
  # actual cutoff_distance parameter (not a dead kwargs lookup) as the
  # correct fallback instead of a hardcoded 9.0.
  cutoff = kwargs.get("cutoff", cutoff_distance)
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


# ---------------------------------------------------------------------------
# Phase 1 migration: PhysicsSystem -> MolecularBundle factory
# ---------------------------------------------------------------------------


def _next_bucket(n: int, buckets: tuple) -> int:
    """Return the smallest bucket >= n, or the largest if n exceeds all."""
    for b in buckets:
        if n <= b:
            return b
    return buckets[-1]


def _pad_1d(arr, bucket: int, dtype=None):
    """Pad a 1D array to bucket length; return zeros if arr is None or empty."""
    import jax.numpy as jnp

    if arr is None or (hasattr(arr, "size") and arr.size == 0):
        return jnp.zeros(bucket, dtype=dtype or jnp.float32)
    pad = bucket - arr.shape[0]
    return jnp.pad(arr, (0, pad))


def _pad_2d(arr, bucket: int, cols: int, dtype=None):
    """Pad a 2D array to (bucket, cols); return zeros if arr is None or empty."""
    import jax.numpy as jnp

    if arr is None or (hasattr(arr, "size") and arr.size == 0):
        return jnp.zeros((bucket, cols), dtype=dtype or jnp.int32)
    pad = bucket - arr.shape[0]
    return jnp.pad(arr, ((0, pad), (0, 0)))


def _mask(n_real: int, bucket: int):
    """Bool mask: True for first n_real entries, False for padding."""
    import jax.numpy as jnp

    return jnp.concatenate(
        [jnp.ones(n_real, dtype=bool), jnp.zeros(bucket - n_real, dtype=bool)]
    )


def _pad_positions_with_ghost_lattice(pos, bucket: int, box_size, dtype=None):
    """Pad positions to ``bucket`` rows, placing padding (ghost) rows on a
    uniform sub-lattice filling the periodic box instead of the coordinate
    origin (debt 772).

    Zero-filling ghost positions (the previous behavior, matching
    ``_pad_2d``'s general convention) puts every ghost atom exactly at the
    box origin under periodic boundary conditions. For a bundle with a large
    padding fraction (e.g. a 1963-real-atom protein padded to 5000), this
    means hundreds of real atoms near the origin see hundreds of ghost
    "neighbors" once a fixed-capacity neighbor list is built on these
    positions — evicting real neighbor-list entries and silently corrupting
    energy/forces. This is invisible in the current dense O(N^2) energy path
    (every ghost-involving pair is exactly zeroed via ``atom_mask``,
    regardless of position — see ``_lj_energy_masked``/``_coulomb_energy_masked``
    in batched_energy.py — so where ghosts sit has never mattered before now),
    but is a hard blocker for any neighbor-list-based dispatch (debt 760).

    Ghost atoms carry zero force everywhere in the energy path already, so
    their position is otherwise physically irrelevant — a deterministic,
    non-overlapping uniform lattice is the correct, cheap choice: no bias,
    bounded and predictable neighbor-list occupancy (measured: ~1.75x
    direct-space overhead for the 5000/32A-box/9A-cutoff class, vs ~10x and
    silently wrong when ghosts pile up at the origin).

    Note: this only fixes the *initial* placement. Ghost atoms have nonzero
    mass (see ``masses_padded`` below, ``constant_values=1.0``) and the
    Langevin O-step (``settle.py``) is not atom_mask-aware, so ghosts still
    receive stochastic momentum kicks and will drift from this lattice over
    a trajectory. Not fixed here — deferred to debt 760's dispatch-carry
    implementation (the first place a live, running trajectory with a real
    neighbor list actually exists to test against); see debt 772.

    Args:
        pos: (n, 3) real positions, or None/empty.
        bucket: total padded row count.
        box_size: (3,) box edge lengths, or None/non-positive for
            non-periodic systems — falls back to zero-fill (no periodic
            wraparound to place ghosts sensibly against, and non-periodic
            dispatch has no neighbor-list capacity concern to protect).
        dtype: output dtype.

    Returns:
        (bucket, 3) array.
    """
    import jax.numpy as jnp
    import numpy as np

    # Preserve pos's own dtype (matching _pad_2d's actual behavior, which
    # pads via jnp.pad and never touches dtype for the non-empty case --
    # under jax_enable_x64=True with a float64 pos, that means output stays
    # float64 despite the dtype=jnp.float32 call-site argument). Forcing
    # out_dtype unconditionally here broke that under x64
    # (tests/physics/test_b1_water_pme_parity.py enables x64 explicitly) --
    # a real regression caught by the existing test suite, not just a
    # theoretical risk. `dtype` is only a fallback for the all-empty case.
    if pos is None or (hasattr(pos, "size") and pos.size == 0):
        out_dtype = dtype or jnp.float32
        real = np.zeros((0, 3), dtype=np.float64)
    else:
        out_dtype = pos.dtype if hasattr(pos, "dtype") else (dtype or jnp.float32)
        real = np.asarray(pos, dtype=np.float64)

    n = real.shape[0]
    n_ghost = bucket - n
    if n_ghost <= 0:
        return jnp.asarray(real, dtype=out_dtype)

    box_np = None if box_size is None else np.asarray(box_size, dtype=np.float64)
    if box_np is None or box_np.shape != (3,) or np.any(box_np <= 0):
        ghost = np.zeros((n_ghost, 3), dtype=np.float64)
        return jnp.asarray(np.concatenate([real, ghost], axis=0), dtype=out_dtype)

    m = max(int(np.ceil(n_ghost ** (1.0 / 3.0))), 1)
    axes = [np.linspace(0.0, box_np[d], num=m, endpoint=False) for d in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    ghost = grid[:n_ghost]

    return jnp.asarray(np.concatenate([real, ghost], axis=0), dtype=out_dtype)


def _pad_3d(arr, bucket: int, d1: int, d2: int, dtype=None):
    """Pad a 3D array to (bucket, d1, d2); return zeros if arr is None or empty."""
    import jax.numpy as jnp

    if arr is None or (hasattr(arr, "size") and arr.size == 0):
        return jnp.zeros((bucket, d1, d2), dtype=dtype or jnp.float32)
    arr = jnp.asarray(arr)
    if arr.ndim != 3:
        msg = f"expected 3D cmap grid, got ndim={arr.ndim}"
        raise ValueError(msg)
    pad = bucket - arr.shape[0]
    return jnp.pad(arr, ((0, pad), (0, 0), (0, 0)))


def _dense_excl_to_pair_list(
    excl_indices,
    excl_scales_vdw,
    excl_scales_elec,
) -> tuple[object | None, object | None, object | None, int]:
    """Convert PhysicsSystem dense ``(N, M)`` excl layout to bundle ``(E, 2)`` pairs."""
    import numpy as np

    if excl_indices is None or getattr(excl_indices, "size", 0) == 0:
        return None, None, None, 0

    idx = np.asarray(excl_indices)
    if idx.ndim == 2 and idx.shape[1] == 2 and not np.any(idx < 0):
        n = int(idx.shape[0])
        sv = np.asarray(excl_scales_vdw) if excl_scales_vdw is not None else np.ones(n)
        se = np.asarray(excl_scales_elec) if excl_scales_elec is not None else np.ones(n)
        return idx.astype(np.int32), sv.astype(np.float32), se.astype(np.float32), n

    if idx.ndim != 2:
        msg = f"excl_indices must be 2D, got ndim={idx.ndim}"
        raise ValueError(msg)

    sv = np.asarray(excl_scales_vdw, dtype=np.float32)
    se = np.asarray(excl_scales_elec, dtype=np.float32)
    pairs: list[list[int]] = []
    vdw: list[float] = []
    elec: list[float] = []
    seen: set[tuple[int, int]] = set()
    n_atoms, n_slots = idx.shape
    for i in range(n_atoms):
        for k in range(n_slots):
            j = int(idx[i, k])
            if j < 0:
                continue
            a, b = (i, j) if i < j else (j, i)
            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            pairs.append([a, b])
            vdw.append(float(sv[i, k] if sv.size else 1.0))
            elec.append(float(se[i, k] if se.size else 1.0))
    n = len(pairs)
    if n == 0:
        return None, None, None, 0
    return (
        np.array(pairs, dtype=np.int32),
        np.array(vdw, dtype=np.float32),
        np.array(elec, dtype=np.float32),
        n,
    )


def _flatten_multi_term_torsions(
    indices,
    params,
) -> tuple[object, object, int]:
    """Flatten PhysicsSystem (N, T, P) torsion params to bundle (N*T, P) layout.

    PhysicsSystem stores ``dihedral_params`` / ``improper_params`` as
    ``(N_torsions, N_terms, P)``. MolecularBundle stores one row per term.
    """
    import jax.numpy as jnp

    if indices is None or (hasattr(indices, "size") and indices.size == 0):
        return indices, params, 0
    indices = jnp.asarray(indices)
    params = jnp.asarray(params)
    if params.ndim == 2:
        return indices, params, int(indices.shape[0])
    if params.ndim == 3:
        n_terms = int(params.shape[1])
        flat_idx = jnp.repeat(indices, n_terms, axis=0)
        flat_params = params.reshape(-1, params.shape[-1])
        return flat_idx, flat_params, int(flat_idx.shape[0])
    msg = f"expected params ndim 2 or 3, got {params.ndim}"
    raise ValueError(msg)


def make_bundle_from_system(
    system,
    boundary_condition: str = "periodic",
    exclusion_spec: Any = None,
) -> MolecularBundle:
    """Convert a PhysicsSystem to a MolecularBundle with bucketed padding.

    All Optional fields are resolved; None becomes zeros. This factory is
    the entry point for the Phase 2a OpenMM parity harness.

    Args:
        system: PhysicsSystem (or compatible object with getattr access).
        boundary_condition: "periodic" or "free".
        exclusion_spec: Optional ``ExclusionSpec``; when provided, populates
            ``excl_*`` pair lists and ``exception_*`` fields on the bundle.

    Returns:
        MolecularBundle with all arrays padded to bucket sizes.

    Notes:
        Multi-term ``dihedral_params`` / ``improper_params`` ``(N, T, P)`` are
        flattened to one bundle row per term via ``_flatten_multi_term_torsions``.
        Dense ``excl_indices`` ``(N, M)`` are converted to ``(E, 2)`` pair lists.
    """
    import jax.numpy as jnp

    def _get(attr, default=None):
        return getattr(system, attr, default)

    pos = system.positions
    n = pos.shape[0]
    a = _next_bucket(n, ATOM_BUCKETS)

    # --- bonded topology -------------------------------------------------
    bonds = _get("bonds")
    bp = _get("bond_params")
    nb = 0 if bonds is None or bonds.size == 0 else bonds.shape[0]
    bb = _next_bucket(max(nb, 1), BOND_BUCKETS)

    angles = _get("angles")
    ap = _get("angle_params")
    na = 0 if angles is None or angles.size == 0 else angles.shape[0]
    ab = _next_bucket(max(na, 1), ANGLE_BUCKETS)

    dihs = _get("dihedrals")
    if dihs is None:
        # MergedTopology (topology_merger.py) names this field proper_dihedrals,
        # not dihedrals -- same fallback convention already used a few lines up
        # in this file's make_energy_fn (system.py:61-63) for the same mismatch.
        dihs = _get("proper_dihedrals")
    dp = _get("dihedral_params")
    dihs, dp, nd = _flatten_multi_term_torsions(dihs, dp)
    db = _next_bucket(max(nd, 1), DIHEDRAL_BUCKETS)

    imps = _get("impropers")
    imp_p = _get("improper_params")
    imps, imp_p, ni = _flatten_multi_term_torsions(imps, imp_p)

    ub = _get("urey_bradley_bonds")
    ub_p = _get("urey_bradley_params")
    nub = 0 if ub is None or ub.size == 0 else ub.shape[0]

    # --- water -----------------------------------------------------------
    wi = _get("water_indices")
    nw = 0 if wi is None or wi.size == 0 else wi.shape[0]
    wb = _next_bucket(max(nw, 1), WATER_BUCKETS)

    # --- exclusions -------------------------------------------------------
    # PhysicsSystem: dense (N, max_excl) or pair list (E, 2). Bundle: (E, 2).
    excl = _get("excl_indices")
    excl_sv = _get("excl_scales_vdw")
    excl_se = _get("excl_scales_elec")
    if exclusion_spec is not None:
        from prolix.physics.neighbor_list import map_exclusions_to_dense_padded

        dense_i, dense_sv, dense_se = map_exclusions_to_dense_padded(exclusion_spec)
        excl, excl_sv, excl_se, ne = _dense_excl_to_pair_list(
            dense_i, dense_sv, dense_se
        )
    else:
        excl, excl_sv, excl_se, ne = _dense_excl_to_pair_list(excl, excl_sv, excl_se)
    eb = _next_bucket(max(ne, 1), EXCL_BUCKETS)

    # --- exclusions (per-atom-row form, for NL/flash kernels; debt 765) ---
    # Every neighbor-list/flash consumer (this file's own make_energy_fn NL
    # branch, optimization.py's chunked_*_nl, flash_explicit.py,
    # flash_nonbonded.py) needs excl_indices/excl_scales_vdw/excl_scales_elec
    # in the documented (N, max_excl) per-atom-row shape (typing.py:300-302)
    # -- NOT the (E, 2) pair-list MolecularBundle stores above (excl/excl_sv/
    # excl_se). Computed once here, host-side, construction-time-only -- NEVER
    # inside bundle_md.physics_system_from_bundle, which runs per-replica
    # under jax.vmap/jit during real dispatch (this session's heterogeneous-
    # batching confirmation requires per-replica-varying bundle fields, so
    # bundle.excl_indices is a genuine tracer there, not a host array; a
    # numpy-loop reconstruction is only safe at true bundle-construction time,
    # same rationale as map_exclusions_to_dense_padded / _dense_excl_to_pair_list
    # above). exclusion_spec path uses map_exclusions_to_dense_padded directly
    # (same source data _dense_excl_to_pair_list above converts to pairs);
    # the no-exclusion_spec fallback instead converts the just-computed
    # pair-list excl/excl_sv/excl_se (ne real pairs, all valid) via
    # pair_list_to_dense_padded.
    _EXCL_DENSE_MAX = 32
    excl_dense_indices = excl_dense_scales_vdw = excl_dense_scales_elec = None
    if exclusion_spec is not None:
        from prolix.physics.neighbor_list import map_exclusions_to_dense_padded

        dense_idx, dense_sv, dense_se = map_exclusions_to_dense_padded(
            exclusion_spec, max_exclusions=_EXCL_DENSE_MAX
        )
        pad_rows = a - dense_idx.shape[0]
        excl_dense_indices = jnp.pad(
            dense_idx, ((0, pad_rows), (0, 0)), constant_values=-1
        )
        excl_dense_scales_vdw = jnp.pad(
            dense_sv, ((0, pad_rows), (0, 0)), constant_values=1.0
        )
        excl_dense_scales_elec = jnp.pad(
            dense_se, ((0, pad_rows), (0, 0)), constant_values=1.0
        )
    elif ne > 0:
        from prolix.physics.neighbor_list import pair_list_to_dense_padded

        dense_idx, dense_sv, dense_se = pair_list_to_dense_padded(
            excl, excl_sv, excl_se, jnp.ones(ne, dtype=bool), a,
            max_exclusions=_EXCL_DENSE_MAX,
        )
        excl_dense_indices = dense_idx
        excl_dense_scales_vdw = dense_sv
        excl_dense_scales_elec = dense_se

    # --- exceptions (1-4 pairs) -------------------------------------------
    exc_pairs = _get("exception_pairs")
    exc_sig = _get("exception_sigmas")
    exc_eps = _get("exception_epsilons")
    exc_q = _get("exception_chargeprods")
    if exclusion_spec is not None and exclusion_spec.exception_pairs.shape[0] > 0:
        exc_pairs = exclusion_spec.exception_pairs
        exc_sig = exclusion_spec.exception_sigmas
        exc_eps = exclusion_spec.exception_epsilons
        exc_q = exclusion_spec.exception_chargeprods
    nx = 0 if exc_pairs is None or exc_pairs.size == 0 else int(exc_pairs.shape[0])
    xb = _next_bucket(max(nx, 1), EXCEPTION_BUCKETS)

    # --- CMAP -------------------------------------------------------------
    cmap_t = _get("cmap_torsions")
    cmap_g = _get("cmap_energy_grids")
    if cmap_g is None:
        cmap_g = _get("cmap_energy")
    nc = 0 if cmap_t is None or cmap_t.size == 0 else int(cmap_t.shape[0])
    cb = _next_bucket(max(nc, 1), CMAP_BUCKETS)

    # --- box / PBC -------------------------------------------------------
    box_size = _get("box_size")
    has_pbc = box_size is not None and jnp.any(jnp.asarray(box_size) != 0).item()
    if box_size is not None:
        # box_size is (3,); MolecularBundle.box is (3, 3) — use diagonal
        box_mat = jnp.diag(jnp.asarray(box_size, dtype=jnp.float32))
    else:
        box_mat = jnp.zeros((3, 3), dtype=jnp.float32)

    # --- shape spec -------------------------------------------------------
    # Compute bucket indices (coarse; enables identical static hashing for same-bucket systems)
    atom_bucket_idx = _bucket_idx(n, ATOM_BUCKETS)
    bond_bucket_idx = _bucket_idx(max(nb, 1), BOND_BUCKETS)
    angle_bucket_idx = _bucket_idx(max(na, 1), ANGLE_BUCKETS)
    dihedral_bucket_idx = _bucket_idx(max(nd, 1), DIHEDRAL_BUCKETS)
    water_bucket_idx = _bucket_idx(max(nw, 1), WATER_BUCKETS)
    excl_bucket_idx = _bucket_idx(max(ne, 1), EXCL_BUCKETS)
    cmap_bucket_idx = _bucket_idx(max(nc, 1), CMAP_BUCKETS)
    exception_bucket_idx = _bucket_idx(max(nx, 1), EXCEPTION_BUCKETS)

    spec = MolecularShapeSpec(
        atom_bucket_idx=atom_bucket_idx,
        bond_bucket_idx=bond_bucket_idx,
        angle_bucket_idx=angle_bucket_idx,
        dihedral_bucket_idx=dihedral_bucket_idx,
        water_bucket_idx=water_bucket_idx,
        excl_bucket_idx=excl_bucket_idx,
        cmap_bucket_idx=cmap_bucket_idx,
        exception_bucket_idx=exception_bucket_idx,
        has_pbc=has_pbc,
        has_implicit_solvent=False,
        boundary_condition=boundary_condition,
    )

    # --- per-atom helpers -------------------------------------------------
    def _pad_atom(attr):
        arr = _get(attr)
        return _pad_1d(arr, a)

    # masses: unlike other per-atom fields, missing data must NOT zero-fill
    # (mass=0 is physically invalid — div-by-zero in p/m). Default to unit mass,
    # matching the historical masses_for_bundle() fallback for systems that don't
    # supply real masses (e.g. energy-only fixtures); real masses (from_pdb,
    # proxide-backed loaders) flow through unchanged.
    masses_in = _get("masses")
    if masses_in is None or (hasattr(masses_in, "size") and masses_in.size == 0):
        masses_padded = jnp.ones(a, dtype=jnp.float32)
    else:
        masses_in = jnp.asarray(masses_in, dtype=jnp.float32)
        masses_padded = jnp.pad(masses_in, (0, a - masses_in.shape[0]), constant_values=1.0)

    return MolecularBundle(
        positions=_pad_positions_with_ghost_lattice(pos, a, box_size, dtype=jnp.float32),
        masses=masses_padded,
        charges=_pad_atom("charges"),
        sigmas=_pad_atom("sigmas"),
        epsilons=_pad_atom("epsilons"),
        radii=_pad_atom("radii"),
        scaled_radii=_pad_atom("scaled_radii"),
        atom_mask=_mask(n, a),
        n_atoms=jnp.array(n, dtype=jnp.int32),
        box=box_mat,
        # bonds
        bond_idx=_pad_2d(bonds, bb, 2, dtype=jnp.int32),
        bond_params=_pad_2d(bp, bb, 2, dtype=jnp.float32),
        bond_mask=_mask(nb, bb),
        n_bonds=jnp.array(nb, dtype=jnp.int32),
        # angles
        angle_idx=_pad_2d(angles, ab, 3, dtype=jnp.int32),
        angle_params=_pad_2d(ap, ab, 2, dtype=jnp.float32),
        angle_mask=_mask(na, ab),
        n_angles=jnp.array(na, dtype=jnp.int32),
        # proper dihedrals (flattened multi-term)
        dihedral_idx=_pad_2d(dihs, db, 4, dtype=jnp.int32),
        dihedral_params=_pad_2d(dp, db, 3, dtype=jnp.float32),
        dihedral_mask=_mask(nd, db),
        n_dihedrals=jnp.array(nd, dtype=jnp.int32),
        # impropers (flattened multi-term)
        improper_idx=_pad_2d(imps, db, 4, dtype=jnp.int32),
        improper_params=_pad_2d(imp_p, db, 3, dtype=jnp.float32),
        improper_mask=_mask(ni, db),
        improper_is_periodic=jnp.array(
            imp_p is not None
            and getattr(imp_p, "size", 0) > 0
            and int(jnp.asarray(imp_p).shape[-1]) == 3,
            dtype=bool,
        ),
        n_impropers=jnp.array(ni, dtype=jnp.int32),
        # Urey-Bradley
        urey_bradley_idx=_pad_2d(ub, ab, 3, dtype=jnp.int32),
        urey_bradley_params=_pad_2d(ub_p, ab, 2, dtype=jnp.float32),
        urey_bradley_mask=_mask(nub, ab),
        n_urey_bradley=jnp.array(nub, dtype=jnp.int32),
        # CMAP
        cmap_torsion_idx=_pad_2d(cmap_t, cb, 8, dtype=jnp.int32),
        cmap_energy_grids=_pad_3d(
            cmap_g,
            cb,
            int(cmap_g.shape[1]) if cmap_g is not None and getattr(cmap_g, "size", 0) else 24,
            int(cmap_g.shape[2]) if cmap_g is not None and getattr(cmap_g, "size", 0) else 24,
            dtype=jnp.float32,
        ),
        cmap_mask=_mask(nc, cb),
        n_cmap=jnp.array(nc, dtype=jnp.int32),
        # SETTLE water
        water_indices=_pad_2d(wi, wb, 3, dtype=jnp.int32),
        water_mask=_mask(nw, wb),
        n_waters=jnp.array(nw, dtype=jnp.int32),
        # nonbonded exclusions
        excl_indices=_pad_2d(excl, eb, 2, dtype=jnp.int32),
        excl_scales_vdw=_pad_1d(excl_sv, eb, dtype=jnp.float32),
        excl_scales_elec=_pad_1d(excl_se, eb, dtype=jnp.float32),
        excl_mask=_mask(ne, eb),
        n_excl=jnp.array(ne, dtype=jnp.int32),
        # nonbonded exclusions, per-atom-row form (debt 765)
        excl_dense_indices=excl_dense_indices,
        excl_dense_scales_vdw=excl_dense_scales_vdw,
        excl_dense_scales_elec=excl_dense_scales_elec,
        # 1-4 exception pairs
        exception_pairs=_pad_2d(exc_pairs, xb, 2, dtype=jnp.int32),
        exception_sigmas=_pad_1d(exc_sig, xb, dtype=jnp.float32),
        exception_epsilons=_pad_1d(exc_eps, xb, dtype=jnp.float32),
        exception_chargeprods=_pad_1d(exc_q, xb, dtype=jnp.float32),
        exception_mask=_mask(nx, xb),
        n_exception_pairs=jnp.array(nx, dtype=jnp.int32),
        # nonbonded parameters
        pme_alpha=jnp.array(float(_get("pme_alpha") or 0.0)),
        cutoff_distance=jnp.array(float(_get("nonbonded_cutoff") or 9.0)),
        shape_spec=spec,
    )
