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

  from proxide.md import SystemParams

Array = util.Array


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
  system_params: SystemParams,
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
) -> Callable[[Array], Array]:
  """Creates the total potential energy function.

  U(R) = U_bond + U_angle + U_vdw + U_elec + U_cmap + U_sasa

  Args:
      displacement_fn: JAX MD displacement function.
      system_params: System parameters from `proxide.md`.
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
  cmap_grids = system_params.get("cmap_energy_grids")
  cmap_coeffs_precomputed = None

  if cmap_grids is not None and cmap_grids.shape[0] > 0:
    if cmap_grids.ndim == 3:
      cmap_coeffs_precomputed = cmap.precompute_cmap_coefficients(cmap_grids)
    elif cmap_grids.ndim == 4:
      cmap_coeffs_precomputed = cmap_grids

  bond_energy_fn = bonded.make_bond_energy_fn(
    displacement_fn,
    jnp.asarray(system_params["bonds"]),
    jnp.asarray(system_params["bond_params"]),
  )

  angle_energy_fn = bonded.make_angle_energy_fn(
    displacement_fn,
    jnp.asarray(system_params["angles"]),
    jnp.asarray(system_params["angle_params"]),
  )

  dihedral_energy_fn = bonded.make_dihedral_energy_fn(
    displacement_fn,
    jnp.asarray(system_params["dihedrals"]),
    jnp.asarray(system_params["dihedral_params"]),
  )

  improper_energy_fn = bonded.make_dihedral_energy_fn(
    displacement_fn,
    jnp.asarray(system_params["impropers"]),
    jnp.asarray(system_params["improper_params"]),
  )

  ub_energy_fn = bonded.make_bond_energy_fn(
    displacement_fn,
    jnp.asarray(system_params.get("urey_bradley_bonds", jnp.zeros((0, 2), dtype=jnp.int32))),
    jnp.asarray(system_params.get("urey_bradley_params", jnp.zeros((0, 2), dtype=jnp.float32))),
  )

  vs_def = system_params.get("virtual_site_def", jnp.zeros((0, 4), dtype=jnp.int32))
  vs_params = system_params.get("virtual_site_params", jnp.zeros((0, 12), dtype=jnp.float32))
  has_virtual_sites = vs_params.shape[0] > 0

  charges = system_params["charges"]
  sigmas = system_params["sigmas"]
  epsilons = system_params["epsilons"]

  scale_matrix_vdw = system_params.get("scale_matrix_vdw")
  scale_matrix_elec = system_params.get("scale_matrix_elec")
  exclusion_mask = system_params.get("exclusion_mask")

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
    if "gb_radii" in system_params and system_params["gb_radii"] is not None:
      radii = system_params["gb_radii"]
    else:
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

      scaled_radii = system_params.get("scaled_radii")

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

    if "gb_radii" in system_params and system_params["gb_radii"] is not None:
      radii = system_params["gb_radii"]
    else:
      radii = sigmas * 0.5

    return generalized_born.compute_ace_nonpolar_energy(
      jnp.asarray(radii),
      born_radii,
      surface_tension=surface_tension,
      probe_radius=constants.PROBE_RADIUS,
    )

  def compute_cmap_term(r):
    if "cmap_torsions" not in system_params or "cmap_energy_grids" not in system_params:
      return 0.0

    cmap_torsions = system_params["cmap_torsions"]
    if cmap_torsions.shape[0] == 0:
      return 0.0

    cmap_indices = system_params["cmap_indices"]
    cmap_grids = system_params["cmap_energy_grids"]

    torsion_indices = jax.vmap(CmapTorsionIndices.from_row)(cmap_torsions)

    phi = compute_dihedral_angles(r, jnp.asarray(torsion_indices.phi_indices), displacement_fn)
    psi = compute_dihedral_angles(r, jnp.asarray(torsion_indices.psi_indices), displacement_fn)

    coeffs_to_use = cmap_coeffs_precomputed if cmap_coeffs_precomputed is not None else cmap_grids

    return cmap.compute_cmap_energy(psi, phi, cmap_indices, coeffs_to_use)

  # PME Exclusion Corrections (Pre-computation)
  # -------------------------------------------------------------------------

  # Robust Topological Search for Exclusions
  pme_bonds = system_params.get("bonds")
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
  coul_14_scale = system_params.get("coulomb14scale", 0.83333333)

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

  return total_energy
