"""System setup and energy function for implicit solvent MD."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from jax_md import energy, partition, space, util
from proxide.physics import constants

from prolix.physics import bonded, cmap, generalized_born, pme, virtual_sites
from prolix.physics import neighbor_list as nl
from prolix.types import CmapTorsionIndices
from prolix.utils import topology

if TYPE_CHECKING:
  from collections.abc import Callable
  from proxide.core.containers import Protein

Array = util.Array


class _DictSystemWrapper:
  """Wraps a plain dict to provide attribute access with defaults for optional fields.

  This allows make_energy_fn to accept both Protein objects and plain dicts
  (used in unit tests for simple particle systems).
  """

  # Optional attributes with their defaults
  _DEFAULTS = {
    "cmap_energy_grids": None,
    "cmap_torsions": None,
    "cmap_indices": None,
    "proper_dihedrals": None,
    "impropers": None,
    "improper_params": None,
    "urey_bradley_bonds": None,
    "urey_bradley_params": None,
    "virtual_site_def": None,
    "virtual_site_params": None,
    "radii": None,
    "scaled_radii": None,
    "scale_matrix_vdw": None,
    "scale_matrix_elec": None,
    "coulomb14scale": None,
  }

  def __init__(self, d: dict):
    self._d = d

  def __getattr__(self, name: str):
    if name.startswith("_"):
      raise AttributeError(name)
    try:
      return self._d[name]
    except KeyError:
      if name in self._DEFAULTS:
        return self._DEFAULTS[name]
      raise AttributeError(f"System dict has no key '{name}' and no default is defined")


def compute_dihedral_angles(
  r: Array, indices: Array, displacement_fn: space.DisplacementFn
) -> Array:
  """Computes dihedral angles for a batch of indices (N, 4)."""
  r_i = r[indices[:, 0]]
  r_j = r[indices[:, 1]]
  r_k = r[indices[:, 2]]
  r_l = r[indices[:, 3]]

  b0 = jax.vmap(displacement_fn)(r_i, r_j)
  b1 = jax.vmap(displacement_fn)(r_k, r_j)
  b2 = jax.vmap(displacement_fn)(r_l, r_k)

  b1_norm = jnp.linalg.norm(b1, axis=-1, keepdims=True) + 1e-8
  b1_unit = b1 / b1_norm

  v = b0 - jnp.sum(b0 * b1_unit, axis=-1, keepdims=True) * b1_unit
  w = b2 - jnp.sum(b2 * b1_unit, axis=-1, keepdims=True) * b1_unit

  x = jnp.sum(v * w, axis=-1)
  y = jnp.sum(jnp.cross(b1_unit, v) * w, axis=-1)

  return jnp.arctan2(y, x)


def make_energy_fn(
  displacement_fn: space.DisplacementFn,
  system: Protein | dict[str, Any],
  neighbor_list: partition.NeighborList | None = None,
  exclusion_spec: nl.ExclusionSpec | None = None,
  dielectric_constant: float = 1.0,
  implicit_solvent: bool = True,
  solvent_dielectric: float = 78.5,
  solute_dielectric: float = constants.DIELECTRIC_PROTEIN,
  surface_tension: float = constants.SURFACE_TENSION,
  dielectric_offset: float = constants.DIELECTRIC_OFFSET,
  # PBC / Explicit Solvent parameters
  box: Array | None = None,
  use_pbc: bool = False,
  cutoff_distance: float = 9.0,
  pme_grid_points: int | Array = 64,
  pme_alpha: float = 0.34,
  return_decomposed: bool = False,
) -> Callable[[Array], Array] | dict[str, Callable]:
  """Creates the total potential energy function.

  U(R) = U_bond + U_angle + U_vdw + U_elec + U_cmap + U_sasa

  Args:
      displacement_fn: JAX MD displacement function.
      system: Protein structure container.
      neighbor_list: Optional neighbor list. If provided, non-bonded terms
                     will use it. If None, they will be N^2 (slow).
                     NOTE: For proteins, N^2 is often acceptable for small systems,
                     but neighbor lists are better for >500 atoms.
                     We strongly recommend using neighbor lists.
      dielectric_constant: Dielectric constant for explicit solvent/vacuum (default 1.0).
      implicit_solvent: Whether to use implicit solvent (GBSA).
      solvent_dielectric: Solvent dielectric constant (default 78.5).
      solute_dielectric: Solute dielectric constant (default 1.0).
      surface_tension: Surface tension for SASA term (kcal/mol/A^2).
      dielectric_offset: Offset for Born radius calculation (default 0.09 A).
      box: Simulation box dimensions (optional).
      use_pbc: Whether to use periodic boundary conditions.
      cutoff_distance: Cutoff for non-bonded interactions (Angstroms).
      pme_grid_points: Grid points for PME (e.g. 64).
      pme_alpha: Ewald splitting parameter.

  Returns:
      A function energy(R, neighbor=None) -> float.

  NOTES:
      The energy function includes several terms:
      * Bonded: bonds, angles, dihedrals, impropers, urey-bradley.
      * Non-Bonded: Lennard-Jones and Electrostatics (Coulomb or PME).
      * Solvation: Generalized Born (if implicit_solvent=True).
      * Other: CMAP corrections, SASA non-polar terms.

      CMAP coefficients are pre-computed during function creation to avoid
      spline fitting overhead during runtime. Sparse exclusion lookups are
      also pre-computed if an ExclusionSpec is provided.

  """
  # Wrap plain dicts for attribute access with defaults
  if isinstance(system, dict):
    system = _DictSystemWrapper(system)

  import logging as _logging_setup
  _log_setup = _logging_setup.getLogger("prolix.physics.system")

  def _safe_bonded_arrays(
    indices, params, term_name: str, idx_cols: int, param_cols: int,
  ) -> tuple[Array, Array]:
    """Validate index-param consistency for bonded terms.

    Returns safe (indices, params) arrays. If indices exist but params are
    None or have incompatible row count, logs a warning and returns empty
    arrays for both to prevent shape-mismatch crashes.
    """
    idx = jnp.asarray(indices if indices is not None else jnp.zeros((0, idx_cols), dtype=jnp.int32))
    prm = jnp.asarray(params if params is not None else jnp.zeros((0, param_cols), dtype=jnp.float32))
    if idx.shape[0] > 0 and prm.shape[0] == 0:
      _log_setup.warning(
        "%s: %d indices but no parameters — skipping term. "
        "Fix root cause in proxide parameterization.",
        term_name, idx.shape[0],
      )
      idx = jnp.zeros((0, idx_cols), dtype=jnp.int32)
    elif idx.shape[0] != prm.shape[0] and idx.shape[0] > 0:
      _log_setup.warning(
        "%s: index/param row mismatch (%d vs %d) — skipping term.",
        term_name, idx.shape[0], prm.shape[0],
      )
      idx = jnp.zeros((0, idx_cols), dtype=jnp.int32)
      prm = jnp.zeros((0, param_cols), dtype=jnp.float32)
    return idx, prm

  cmap_grids = system.cmap_energy_grids
  cmap_coeffs_precomputed = None

  if cmap_grids is not None and cmap_grids.shape[0] > 0:
    if cmap_grids.ndim == 3:
      cmap_coeffs_precomputed = cmap.precompute_cmap_coefficients(cmap_grids)
    elif cmap_grids.ndim == 4:
      cmap_coeffs_precomputed = cmap_grids

  bond_idx, bond_prm = _safe_bonded_arrays(system.bonds, system.bond_params, "bonds", 2, 2)
  bond_energy_fn = bonded.make_bond_energy_fn(displacement_fn, bond_idx, bond_prm)

  angle_idx, angle_prm = _safe_bonded_arrays(system.angles, system.angle_params, "angles", 3, 2)
  angle_energy_fn = bonded.make_angle_energy_fn(displacement_fn, angle_idx, angle_prm)

  dih_idx, dih_prm = _safe_bonded_arrays(system.proper_dihedrals, system.dihedral_params, "proper_dihedrals", 4, 3)
  dihedral_energy_fn = bonded.make_dihedral_energy_fn(displacement_fn, dih_idx, dih_prm)

  imp_idx, imp_prm = _safe_bonded_arrays(system.impropers, system.improper_params, "impropers", 4, 3)
  improper_energy_fn = bonded.make_dihedral_energy_fn(displacement_fn, imp_idx, imp_prm)

  ub_idx, ub_prm = _safe_bonded_arrays(system.urey_bradley_bonds, system.urey_bradley_params, "urey_bradley", 2, 2)
  ub_energy_fn = bonded.make_bond_energy_fn(displacement_fn, ub_idx, ub_prm)

  vs_def = system.virtual_site_def if system.virtual_site_def is not None else jnp.zeros((0, 4), dtype=jnp.int32)
  vs_params = system.virtual_site_params if system.virtual_site_params is not None else jnp.zeros((0, 12), dtype=jnp.float32)
  has_virtual_sites = vs_params.shape[0] > 0

  charges = system.charges
  sigmas = system.sigmas
  epsilons = system.epsilons

  # Defensive: clamp sigmas to prevent singular LJ from unparameterized atoms (sigma=0.0).
  # This can happen when oxidize/ff parameterization skips atoms it can't match.
  n_zero_sigma = int(jnp.sum(sigmas <= 0.0))
  if n_zero_sigma > 0:
    import logging as _logging
    _log = _logging.getLogger("prolix.physics.system")
    _log.warning(
      "Found %d atoms with sigma <= 0.0 (unparameterized). "
      "Clamping to 1e-6 to prevent singular LJ. "
      "Fix root cause in force field parameterization.",
      n_zero_sigma,
    )
    sigmas = jnp.maximum(sigmas, 1e-6)

  assert sigmas.shape[0] == charges.shape[0], f"Sigmas shape mismatch: {sigmas.shape} vs {charges.shape}"

  scale_matrix_vdw = system.scale_matrix_vdw if hasattr(system, "scale_matrix_vdw") else None
  scale_matrix_elec = system.scale_matrix_elec if hasattr(system, "scale_matrix_elec") else None
  exclusion_mask = system.exclusion_mask

  # Auto-build ExclusionSpec from bonds when neither explicit spec nor
  # pre-built exclusion_mask is available (e.g. CoordFormat.Full proteins).
  if exclusion_spec is None and exclusion_mask is None:
    _bonds = system.bonds
    if _bonds is not None and _bonds.shape[0] > 0:
      import logging as _logging_excl
      _log_excl = _logging_excl.getLogger("prolix.physics.system")
      exclusion_spec = nl.ExclusionSpec.from_protein(system)
      _log_excl.info(
        "Auto-built ExclusionSpec from bonds: %d 1-2/1-3 pairs, %d 1-4 pairs",
        len(exclusion_spec.idx_12_13),
        len(exclusion_spec.idx_14),
      )

  if exclusion_spec is not None:
    excl_indices, excl_scales_vdw, excl_scales_elec = nl.map_exclusions_to_dense_padded(
      exclusion_spec
    )
    use_sparse_exclusions = True

    # If dense scaling matrices are missing, build them from spec (required for N^2 path)
    if scale_matrix_vdw is None and scale_matrix_elec is None:
      # Build dense scaling matrices
      N = charges.shape[0]
      mat_vdw = jnp.ones((N, N), dtype=jnp.float32)
      mat_elec = jnp.ones((N, N), dtype=jnp.float32)

      # Self-exclusion
      idx = jnp.arange(N)
      mat_vdw = mat_vdw.at[idx, idx].set(0.0)
      mat_elec = mat_elec.at[idx, idx].set(0.0)

      # 1-2/1-3 Exclusions (Fully Excluded)
      idx1213 = exclusion_spec.idx_12_13
      if idx1213.shape[0] > 0:
        mat_vdw = mat_vdw.at[idx1213[:, 0], idx1213[:, 1]].set(0.0)
        mat_vdw = mat_vdw.at[idx1213[:, 1], idx1213[:, 0]].set(0.0)
        mat_elec = mat_elec.at[idx1213[:, 0], idx1213[:, 1]].set(0.0)
        mat_elec = mat_elec.at[idx1213[:, 1], idx1213[:, 0]].set(0.0)

      # 1-4 Exclusions (Scaled)
      idx14 = exclusion_spec.idx_14
      if idx14.shape[0] > 0:
        mat_vdw = mat_vdw.at[idx14[:, 0], idx14[:, 1]].set(exclusion_spec.scale_14_vdw)
        mat_vdw = mat_vdw.at[idx14[:, 1], idx14[:, 0]].set(exclusion_spec.scale_14_vdw)
        mat_elec = mat_elec.at[idx14[:, 0], idx14[:, 1]].set(exclusion_spec.scale_14_elec)
        mat_elec = mat_elec.at[idx14[:, 1], idx14[:, 0]].set(exclusion_spec.scale_14_elec)

      scale_matrix_vdw = mat_vdw
      scale_matrix_elec = mat_elec

      # Also set exclusion_mask for binary checks (though scale matrix path takes precedence)
      # We treat anything with non-zero scale as "allowed" for binary mask,
      # but really scale_matrix handles it all.
      exclusion_mask = (mat_vdw > 0.0).astype(jnp.float32)

  else:
    excl_indices = excl_scales_vdw = excl_scales_elec = None
    use_sparse_exclusions = False

  # Constants
  if implicit_solvent:
    eff_dielectric = solute_dielectric
    kappa = 0.0
  else:
    eff_dielectric = dielectric_constant
    kappa = 0.0
    if use_pbc:
      kappa = 0.0
      eff_dielectric = 1.0

  COULOMB_CONSTANT = 332.0637 / eff_dielectric

  # Pre-create PME function at setup time (NOT inside compute_electrostatics)
  # This prevents recompilation on every energy evaluation
  if use_pbc and box is not None:
    pme_recip_fn = pme.make_pme_energy_fn(
      jnp.asarray(charges), box, grid_points=pme_grid_points, alpha=pme_alpha
    )
  else:
    pme_recip_fn = None

  def lj_pair(dr, sigma_i, sigma_j, eps_i, eps_j, **kwargs):
    sigma = 0.5 * (sigma_i + sigma_j)
    epsilon = jnp.sqrt(eps_i * eps_j)
    return energy.lennard_jones(dr, sigma, epsilon)

  # Electrostatics
  # -------------------------------------------------------------------------
  def compute_electrostatics(r, neighbor_idx=None):
    if system.radii is not None:
      radii = system.radii
    else:
      import logging as _log_gb
      _log_gb.getLogger("prolix.physics.system").warning(
        "No GB radii found — using sigma/2 fallback. "
        "This is physically incorrect for implicit solvent. "
        "Call assign_mbondi2_radii() or use CoordFormat.Full with parse_structure."
      )
      radii = sigmas * 0.5

    e_gb = 0.0
    born_radii = None

    if implicit_solvent:
      # Generalized Born (OBC) - Solvation Term

      if scale_matrix_vdw is not None:
        # OpenMM CustomGBForce (OBC2) behavior:
        # GBSA Hypothesis: Include All in Radii.
        # Energy: Include 1-2/1-3 (1.0), Exclude 1-4 (0.0), Include others (1.0)
        gb_mask = jnp.ones_like(scale_matrix_vdw)

        if scale_matrix_elec is not None:
          # OpenMM GBSAOBCForce includes ALL pairs (1-2, 1-3, 1-4) in the energy calculation
          # with scaling factor 1.0.
          # "The GBSA interaction is calculated between all pairs of particles,
          # including those that are excluded from the nonbonded force."
          gb_energy_mask = jnp.ones_like(scale_matrix_vdw)
        else:
          gb_energy_mask = jnp.ones_like(scale_matrix_vdw)

      else:
        gb_mask = exclusion_mask
        gb_energy_mask = None

      scaled_radii = system.scaled_radii

      if neighbor_idx is None:
        e_gb, born_radii = generalized_born.compute_gb_energy(
          r,
          jnp.asarray(charges),
          jnp.asarray(radii),
          solvent_dielectric=solvent_dielectric,
          solute_dielectric=solute_dielectric,
          dielectric_offset=dielectric_offset,
          mask=gb_mask,  # Radii: Scale 1-4 (0.5)
          energy_mask=gb_energy_mask,  # Energy: Full (1.0)
          scaled_radii=jnp.asarray(scaled_radii) if scaled_radii is not None else None,
        )
      else:
        # TODO: Update neighbor list version of GBSA to accept mask
        # For now, we assume neighbor list version handles exclusions via neighbor list construction?
        # No, neighbor list usually includes everything within cutoff.
        # But we don't have mask support in compute_gb_energy_neighbor_list yet.
        # Since validation script uses N^2 (neighbor_idx=None), this is fine for now.
        e_gb, born_radii = generalized_born.compute_gb_energy_neighbor_list(
          r,
          jnp.asarray(charges),
          jnp.asarray(radii),
          neighbor_idx,
          solvent_dielectric=solvent_dielectric,
          solute_dielectric=solute_dielectric,
          dielectric_offset=dielectric_offset,
        )

      # Non-polar Solvation (SASA) - Now computed separately using ACE

    # Direct Coulomb / Screened Coulomb / PME
    if use_pbc and box is not None:
      # PME Electrostatics (Explicit Solvent / Periodic)
      # Use pre-computed PME function (captured from outer scope)
      if pme_recip_fn is None:
        raise RuntimeError("PME reciprocal function not initialized.")
      e_recip = pme_recip_fn(r)
      # NOTE: Direct space (erfc term) is computed in the neighbor block below

    else:
      e_recip = 0.0

    # COULOMB_CONSTANT is defined in outer scope

    # Apply scaling to PME Reciprocal
    # jax_md PME returns energy in internal units (assuming q in e, r in A -> V ~ e^2/A)
    # We need to scale by COULOMB_CONSTANT to get kcal/mol
    if use_pbc and box is not None:
      e_recip = e_recip * COULOMB_CONSTANT

      q_sq_sum = jnp.sum(charges**2)
      e_self = COULOMB_CONSTANT * (pme_alpha / jnp.sqrt(jnp.pi)) * q_sq_sum

      e_recip = e_recip - e_self

    if neighbor_idx is None:
      # Dense
      dr = space.map_product(displacement_fn)(r, r)
      dist = space.distance(jnp.asarray(dr))
      q_ij = charges[:, None] * charges[None, :]

      dist_safe = dist + 1e-6

      if use_pbc and box is not None:
        # Dense PME Direct Space
        # E = C * q_i * q_j * erfc(alpha * r) / r
        erfc_term = jax.scipy.special.erfc(pme_alpha * dist)
        e_coul = COULOMB_CONSTANT * (q_ij / dist_safe) * erfc_term
      elif kappa > 0:
        e_coul = COULOMB_CONSTANT * (q_ij / dist_safe) * jnp.exp(-kappa * dist)
      else:
        e_coul = COULOMB_CONSTANT * (q_ij / dist_safe)

      # Apply scaling/masking
      if scale_matrix_elec is not None:
        e_coul = e_coul * scale_matrix_elec
      else:
        # Fallback to binary mask
        mask = 1.0 - jnp.eye(charges.shape[0])
        e_coul = jnp.where(mask, e_coul, 0.0)
        e_coul = jnp.where(exclusion_mask, e_coul, 0.0)

      e_direct = 0.5 * jnp.sum(e_coul)

    else:
      # Neighbor List
      idx = neighbor_idx
      r_neighbors = r[idx]  # (N, K, 3)
      # Use jax_md's map_neighbor for correct displacement
      map_neighbor_disp = space.map_neighbor(displacement_fn)
      dr = map_neighbor_disp(r, r_neighbors)  # (N, K, 3)
      dist = space.distance(jnp.asarray(dr))  # (N, K)

      q_neighbors = charges[idx]
      q_central = charges[:, None]
      q_ij = q_central * q_neighbors

      dist_safe = dist + 1e-6

      if use_pbc and box is not None:
        # Neighbor List PME Direct Space
        erfc_term = jax.scipy.special.erfc(pme_alpha * dist)
        e_coul = COULOMB_CONSTANT * (q_ij / dist_safe) * erfc_term
      elif kappa > 0:
        e_coul = COULOMB_CONSTANT * (q_ij / dist_safe) * jnp.exp(-kappa * dist)
      else:
        e_coul = COULOMB_CONSTANT * (q_ij / dist_safe)

      # Mask padding
      mask_neighbors = idx < r.shape[0]

      # Apply scaling using sparse exclusion lookups (O(N*K)) or dense matrix (O(N^2))
      if use_sparse_exclusions and excl_indices is not None:
        # Sparse exclusion lookup - efficient for large systems
        if excl_indices is None or excl_scales_vdw is None or excl_scales_elec is None:
          raise RuntimeError("Sparse exclusions requested but not initialized.")
        _, scale_elec = nl.get_neighbor_exclusion_scales(
          excl_indices, excl_scales_vdw, excl_scales_elec, idx
        )
        e_coul = e_coul * scale_elec
      elif scale_matrix_elec is not None:
        # Dense scaling matrix (legacy path)
        i_idx = jnp.arange(r.shape[0])[:, None]
        safe_idx = jnp.minimum(idx, r.shape[0] - 1)
        scale = scale_matrix_elec[i_idx, safe_idx]
        e_coul = e_coul * scale
      elif exclusion_mask is not None:
        # Dense binary mask (legacy path)
        i_idx = jnp.arange(r.shape[0])[:, None]
        safe_idx = jnp.minimum(idx, r.shape[0] - 1)
        interaction_allowed = exclusion_mask[i_idx, safe_idx]
        e_coul = jnp.where(interaction_allowed, e_coul, 0.0)

      final_mask = mask_neighbors
      e_coul = jnp.where(final_mask, e_coul, 0.0)

      e_direct = 0.5 * jnp.sum(e_coul)

    if implicit_solvent:
      return e_gb, e_direct, born_radii
    return 0.0, e_direct + e_recip, None

  # Combine Non-Bonded
  # -------------------------------------------------------------------------
  def compute_lj(r, neighbor_idx=None):
    if neighbor_idx is None:
      dr = space.map_product(displacement_fn)(r, r)
      dist = space.distance(jnp.asarray(dr))

      sig_ij = 0.5 * (sigmas[:, None] + sigmas[None, :])
      eps_ij = jnp.sqrt(epsilons[:, None] * epsilons[None, :])

      e_lj = energy.lennard_jones(jnp.asarray(dist), sig_ij, eps_ij)

      if scale_matrix_vdw is not None:
        e_lj = e_lj * scale_matrix_vdw
      else:
        mask = exclusion_mask
        e_lj = jnp.where(mask, e_lj, 0.0)

      return 0.5 * jnp.sum(e_lj)

    idx = neighbor_idx
    r_neighbors = r[idx]
    map_neighbor_disp = space.map_neighbor(displacement_fn)
    dr = map_neighbor_disp(r, r_neighbors)
    dist = space.distance(jnp.asarray(dr))

    sig_neighbors = sigmas[idx]
    eps_neighbors = epsilons[idx]
    sig_central = sigmas[:, None]
    eps_central = epsilons[:, None]

    sig_ij = 0.5 * (sig_central + sig_neighbors)
    eps_ij = jnp.sqrt(eps_central * eps_neighbors)

    e_lj = energy.lennard_jones(dist, sig_ij, eps_ij)

    mask_neighbors = idx < r.shape[0]

    if use_sparse_exclusions and excl_indices is not None:
      if excl_indices is None or excl_scales_vdw is None or excl_scales_elec is None:
        raise RuntimeError("Sparse exclusions requested but not initialized.")
      scale_vdw, _ = nl.get_neighbor_exclusion_scales(
        excl_indices, excl_scales_vdw, excl_scales_elec, idx
      )
      e_lj = e_lj * scale_vdw
    elif scale_matrix_vdw is not None:
      i_idx = jnp.arange(r.shape[0])[:, None]
      safe_idx = jnp.minimum(idx, r.shape[0] - 1)
      scale = scale_matrix_vdw[i_idx, safe_idx]
      e_lj = e_lj * scale
    elif exclusion_mask is not None:
      i_idx = jnp.arange(r.shape[0])[:, None]
      safe_idx = jnp.minimum(idx, r.shape[0] - 1)
      interaction_allowed = exclusion_mask[i_idx, safe_idx]
      e_lj = jnp.where(interaction_allowed, e_lj, 0.0)

    final_mask = mask_neighbors
    e_lj = jnp.where(final_mask, e_lj, 0.0)

    return 0.5 * jnp.sum(e_lj)

  def compute_nonpolar(r, born_radii, neighbor_idx=None):
    if not implicit_solvent or born_radii is None:
      return 0.0

    if system.radii is not None:
      radii = system.radii
    else:
      radii = sigmas * 0.5  # Fallback (warning already emitted in compute_electrostatics)

    return generalized_born.compute_ace_nonpolar_energy(
      jnp.asarray(radii),
      born_radii,
      surface_tension=surface_tension,
      probe_radius=constants.PROBE_RADIUS,
    )

  def compute_cmap_term(r):
    if system.cmap_torsions is None or system.cmap_energy_grids is None:
      return 0.0

    cmap_torsions = system.cmap_torsions
    if cmap_torsions.shape[0] == 0:
      return 0.0

    cmap_indices = system.cmap_indices
    cmap_grids = system.cmap_energy_grids

    torsion_indices = jax.vmap(CmapTorsionIndices.from_row)(cmap_torsions)

    phi = compute_dihedral_angles(r, jnp.asarray(torsion_indices.phi_indices), displacement_fn)
    psi = compute_dihedral_angles(r, jnp.asarray(torsion_indices.psi_indices), displacement_fn)

    coeffs_to_use = cmap_coeffs_precomputed if cmap_coeffs_precomputed is not None else cmap_grids

    return cmap.compute_cmap_energy(psi, phi, cmap_indices, coeffs_to_use)

  # PME Exclusion Corrections (Pre-computation)
  # -------------------------------------------------------------------------

  # Robust Topological Search for Exclusions
  pme_bonds = system.bonds
  n_atoms = charges.shape[0]

  if pme_bonds is not None and pme_bonds.shape[0] > 0:
    excl = topology.find_bonded_exclusions(pme_bonds, n_atoms)
    pme_idx_12 = excl.idx_12
    pme_idx_13 = excl.idx_13
    pme_idx_14 = excl.idx_14
  else:
    empty = jnp.zeros((0, 2), dtype=jnp.int32)
    pme_idx_12 = pme_idx_13 = pme_idx_14 = empty

  # Scaling for 1-4
  coul_14_scale = system.coulomb14scale if system.coulomb14scale is not None else 0.83333333

  def compute_lj_tail_correction(
    box: Array, sigma: Array, epsilon: Array, cutoff: float, N_atoms: int
  ) -> Array:
    r"""Compute the long-range dispersion correction for Lennard-Jones.

    Process:
    1.  **Volume**: Compute box volume from lattice vectors.
    2.  **Average**: Determine mean $\sigma$ and $\epsilon$.
    3.  **Integral**: Evaluate tail integral from cutoff to $\infty$.

    Notes:
    $$ E_{LRC} = \frac{8 \pi N^2}{3 V} \langle \epsilon \sigma^6 \rangle \left[ \frac{1}{3} \frac{\sigma^6}{R_c^9} - \frac{1}{R_c^3} \right] \text{ (approx)} $$

    Args:
        box: Box dimensions.
        sigma: Atomic sigmas.
        epsilon: Atomic epsilons.
        cutoff: Potential cutoff.
        N_atoms: Total atoms.

    Returns:
        Tail correction scalar.
    """
    volume = box[0] * box[1] * box[2] if box.ndim == 1 else jnp.linalg.det(box)

    avg_sig = jnp.mean(sigma)
    avg_eps = jnp.mean(epsilon)

    rc3 = cutoff**3
    rc9 = rc3**3
    sig3 = avg_sig**3
    sig6 = sig3**2
    sig9 = sig3**3

    term = (1.0 / 9.0) * (sig9 / rc9) - (1.0 / 3.0) * (sig3 / rc3)
    return (8.0 * jnp.pi * (N_atoms**2) / volume) * avg_eps * sig6 * term

  def compute_pme_exceptions(r: Array) -> Array:
    """Computes the reciprocal space correction for excluded/scaled pairs."""
    if not use_pbc or box is None:
      return jnp.array(0.0)

    # Correction Formula: E_corr = - (1.0 - scale) * q_i * q_j * erf(alpha * r) / r
    # For 1-2 and 1-3: scale = 0.0 => E_corr = -1.0 * Recip
    # For 1-4: scale = sc => E_corr = -(1.0 - sc) * Recip

    def calc_correction_term(indices, scale_factor):
      if indices.shape[0] == 0:
        return 0.0

      r_i = r[indices[:, 0]]
      r_j = r[indices[:, 1]]

      # Displacement
      dr = jax.vmap(displacement_fn)(r_i, r_j)
      dist = space.distance(dr)

      # Charges
      q_i = charges[indices[:, 0]]
      q_j = charges[indices[:, 1]]

      # Reciprocal Part of Energy (what we want to remove)
      # E_recip_pair = q_i * q_j * erf(alpha * r) / r
      # We use COULOMB_CONSTANT for units

      # Avoid singularity
      dist_safe = dist + 1e-6
      erf_term = jax.scipy.special.erf(pme_alpha * dist)

      e_pair_recip = COULOMB_CONSTANT * (q_i * q_j / dist_safe) * erf_term

      # We subtract (1 - scale) * E_recip_pair
      factor = 1.0 - scale_factor
      return jnp.sum(e_pair_recip * factor)

    e_corr = 0.0
    # 1-2 (Scale 0.0)
    e_corr += calc_correction_term(pme_idx_12, 0.0)
    # 1-3 (Scale 0.0)
    e_corr += calc_correction_term(pme_idx_13, 0.0)
    # 1-4 (Scale coul_14_scale)
    e_corr += calc_correction_term(pme_idx_14, coul_14_scale)

    # We subtract this correction
    return -e_corr

  # Total Energy Function
  # -------------------------------------------------------------------------
  def total_energy(r: Array, neighbor: partition.NeighborList | None = None, **kwargs) -> Array:
    if has_virtual_sites:
      r = virtual_sites.reconstruct_virtual_sites(r, vs_def, vs_params)

    e_bond = bond_energy_fn(r)
    e_angle = angle_energy_fn(r)
    e_ub = ub_energy_fn(r)
    e_dihedral = dihedral_energy_fn(r)
    e_improper = improper_energy_fn(r)
    e_cmap = compute_cmap_term(r)

    neighbor_idx = neighbor.idx if neighbor is not None else None

    e_lj = compute_lj(r, neighbor_idx)
    e_gb, e_direct, born_radii = compute_electrostatics(r, neighbor_idx)
    e_elec = e_gb + e_direct
    e_np = compute_nonpolar(r, born_radii, neighbor_idx)

    if use_pbc and box is not None:
      e_pme_corr = compute_pme_exceptions(r)
      e_elec += e_pme_corr

      # Long-Range LJ Correction
      e_lj_lrc = compute_lj_tail_correction(
        box=box,
        sigma=jnp.asarray(sigmas),
        epsilon=jnp.asarray(epsilons),
        cutoff=cutoff_distance,
        N_atoms=r.shape[0],
      )
      e_lj += e_lj_lrc

    return e_bond + e_angle + e_ub + e_dihedral + e_improper + e_cmap + e_lj + e_elec + e_np

  if return_decomposed:
    return {
      "bond": bond_energy_fn,
      "angle": angle_energy_fn,
      "urey_bradley": ub_energy_fn,
      "dihedral": dihedral_energy_fn,
      "improper": improper_energy_fn,
      "cmap": compute_cmap_term,
      "lj": lambda r: compute_lj(r),
      "electrostatics": lambda r: compute_electrostatics(r),
      "nonpolar": lambda r, born_radii: compute_nonpolar(r, born_radii),
      "total": total_energy,
    }

  return total_energy
