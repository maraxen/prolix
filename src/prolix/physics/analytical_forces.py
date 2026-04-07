"""Analytical force computation for dense N² nonbonded terms.

Replaces jax.grad for LJ, Coulomb, GB, and ACE terms to eliminate
autodiff pathologies (scatter NaN, 0/0 gradients) on padded systems
and reduce memory usage from reverse-mode AD tape.

All functions return (N, 3) force arrays in kcal/mol/Å.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from proxide.physics import constants
from proxide.physics.constants import COULOMB_CONSTANT

if TYPE_CHECKING:
    from jax_md.util import Array


def lj_forces_dense(
    r: Array,
    sigmas: Array,
    epsilons: Array,
    atom_mask: Array,
    soft_core_lambda: Array = jnp.float32(1.0),
    excl_scale_vdw: Array | None = None,
) -> Array:
    """Analytical Lennard-Jones forces on the dense N² path.

    Uses the Beutler soft-core formulation:
      U(r,λ) = 4ελ [1/(α(1-λ) + (r/σ)⁶)² - 1/(α(1-λ) + (r/σ)⁶)]

    The force on atom i from atom j is:
      F_ij = -dU/dr · r̂_ij
      dU/dr = 4ελ [1/D² - 2/D³] · dD/dr
      dD/dr = 6r⁵/σ⁶

    Args:
        r: Positions (N, 3).
        sigmas: LJ sigma (N,).
        epsilons: LJ epsilon (N,).
        atom_mask: Bool mask for real atoms (N,).
        soft_core_lambda: Soft-core coupling. 1.0 = standard LJ.
        excl_scale_vdw: (N, N) exclusion scale matrix (must have stop_gradient).

    Returns:
        Forces (N, 3) in kcal/mol/Å. Zero for padded atoms.
    """
    N = r.shape[0]

    # Pairwise displacement vectors: dr[i,j] = r[i] - r[j]
    dr = r[:, None, :] - r[None, :, :]  # (N, N, 3)
    dist_sq = jnp.sum(dr ** 2, axis=-1) + jnp.float32(1e-10)  # (N, N)
    dist = jnp.sqrt(dist_sq)  # (N, N)

    # Combined mask: real pairs, no self-interaction
    mask_ij = (atom_mask[:, None] & atom_mask[None, :]) * (1.0 - jnp.eye(N))
    if excl_scale_vdw is not None:
        mask_ij = mask_ij * excl_scale_vdw
    mask_bool = mask_ij > 0.0

    # Mixing rules — safe values for masked pairs
    sigma_ij = 0.5 * (sigmas[:, None] + sigmas[None, :])
    sigma_ij_safe = jnp.where(mask_bool, jnp.maximum(sigma_ij, jnp.float32(1e-4)),
                              jnp.float32(1.0))
    epsilon_ij = jnp.sqrt(jnp.maximum(epsilons[:, None] * epsilons[None, :],
                                       jnp.float32(0.0)))

    # Soft-core terms
    lam = jnp.float32(soft_core_lambda)
    alpha = jnp.float32(0.5)
    soft_term = alpha * jnp.maximum(jnp.float32(1.0) - lam, jnp.float32(1e-8))

    r_over_sig = dist / sigma_ij_safe
    r6 = r_over_sig ** 6
    D = soft_term + r6 + jnp.float32(1e-12)  # denominator

    # dU/dr = 4ε·λ·[1/D² - 2/D³] · dD/dr
    # dD/dr = 6·r⁵/σ⁶ = 6·dist⁵/σ⁶
    sigma6 = jnp.maximum(sigma_ij_safe ** 6, jnp.float32(1e-48))
    dD_dr = 6.0 * dist ** 5 / sigma6

    dU_dr = jnp.float32(4.0) * epsilon_ij * lam * (
        1.0 / (D * D) - 2.0 / (D * D * D)
    ) * dD_dr

    # Apply mask with jnp.where (not multiply) to prevent NaN * 0
    dU_dr = jnp.where(mask_bool, dU_dr * mask_ij, jnp.float32(0.0))

    # Unit displacement vectors: r̂_ij = dr_ij / |dr_ij|
    unit_dr = dr / (dist[..., None] + jnp.float32(1e-12))  # (N, N, 3)

    # Force on atom i from pair (i,j): F_ij = -dU/dr · r̂_ij
    # Sum over j: F_i = Σ_j F_ij
    # Factor 0.5 because we double-count, but forces ARE Newton's third law:
    # F_ij on i = -dU/dr * r̂_ij, and F_ji on j = +dU/dr * r̂_ij
    # So no 0.5 needed — just sum the (i,j) direction.
    # But with symmetric mask, each pair appears twice; the forces naturally
    # cancel the double-counting because r̂_ij = -r̂_ji.
    pair_force = -dU_dr[..., None] * unit_dr  # (N, N, 3)
    forces = jnp.sum(pair_force, axis=1)  # (N, 3)

    return forces


def coulomb_forces_dense(
    r: Array,
    charges: Array,
    atom_mask: Array,
    excl_scale_elec: Array | None = None,
) -> Array:
    """Analytical Coulomb forces on the dense N² path.

    F_ij = COULOMB_CONSTANT · q_i · q_j / r² · r̂_ij  (pointing from j to i)
    Force on atom i: F_i = -Σ_j COULOMB_CONSTANT · q_i · q_j / r²_ij · r̂_ij

    The sign: E = k·q_i·q_j/r, so F = -dE/dr · r̂ = k·q_i·q_j/r² · r̂
    For like charges (positive product), the force is repulsive (along r̂_ij).

    Args:
        r: Positions (N, 3).
        charges: Atomic charges (N,).
        atom_mask: Bool mask for real atoms (N,).
        excl_scale_elec: (N, N) exclusion scale matrix (must have stop_gradient).

    Returns:
        Forces (N, 3) in kcal/mol/Å. Zero for padded atoms.
    """
    N = r.shape[0]

    # Pairwise displacements
    dr = r[:, None, :] - r[None, :, :]  # (N, N, 3)
    dist_sq = jnp.sum(dr ** 2, axis=-1) + jnp.float32(1e-10)  # (N, N)
    dist = jnp.sqrt(dist_sq)  # (N, N)

    # Mask: real pairs, no self
    mask_ij = (atom_mask[:, None] & atom_mask[None, :]).astype(jnp.float32)
    mask_ij = mask_ij * (1.0 - jnp.eye(N))
    if excl_scale_elec is not None:
        mask_ij = mask_ij * excl_scale_elec
    mask_bool = mask_ij > 0.0

    # Charge products
    qq = charges[:, None] * charges[None, :]  # (N, N)

    # Force magnitude: F = k·q_i·q_j / r²
    # (Force on i from j, along r̂_ij = (r_i - r_j)/|r_i - r_j|)
    # E = k·qq/r → F_i = -dE/dr_i = k·qq/r² · r̂_ij
    f_mag = COULOMB_CONSTANT * qq / dist_sq  # (N, N)

    # Mask with jnp.where
    f_mag = jnp.where(mask_bool, f_mag * mask_ij, jnp.float32(0.0))

    # Unit vectors
    unit_dr = dr / (dist[..., None] + jnp.float32(1e-12))  # (N, N, 3)

    # F_i = Σ_j f_mag_ij · r̂_ij  (repulsive for like charges)
    pair_force = f_mag[..., None] * unit_dr  # (N, N, 3)
    forces = jnp.sum(pair_force, axis=1)  # (N, 3)

    return forces


def gb_ace_forces_dense(
    positions: Array,
    charges: Array,
    radii: Array,
    scaled_radii: Array | None,
    atom_mask: Array,
    dielectric_offset: float = 0.09,
    solvent_dielectric: float = constants.DIELECTRIC_WATER,
    solute_dielectric: float = constants.DIELECTRIC_PROTEIN,
) -> Array:
    """Compute GB + ACE solvation forces via decomposed VJP.

    Instead of jax.grad on the entire GB+ACE energy (which checkpoints
    the full N² Born radii + Coulomb graph and causes OOM), we decompose
    the backward pass into 3 targeted vjp calls:

      1. VJP of GB Coulomb w.r.t. (positions, born_radii)
         → gives dE_gb/dr_direct AND dE_gb/dB
      2. Analytical dE_ace/dB (simple per-atom formula)
      3. VJP of Born radii w.r.t. positions
         → gives dB/dr

    Then chain rule: F = -(dE/dr_direct + (dE_gb/dB + dE_ace/dB) · dB/dr)

    Same physics as jax.grad but ~14 GiB less peak memory because XLA
    doesn't need to checkpoint the entire graph simultaneously.

    Args:
        positions: (N, 3) atom positions.
        charges: (N,) partial charges.
        radii: (N,) intrinsic atomic radii in Angstroms.
        scaled_radii: (N,) optional OBC scaled radii.
        atom_mask: (N,) bool mask for real vs padded atoms.
        dielectric_offset: Born radius offset (default 0.09 Å).
        solvent_dielectric: Solvent dielectric constant.
        solute_dielectric: Solute dielectric constant.

    Returns:
        Forces (N, 3) in kcal/mol/Å. Zero for padded atoms.
    """
    from prolix.physics.generalized_born import (
        compute_born_radii,
        compute_ace_nonpolar_energy,
        f_gb,
        safe_norm,
    )

    N = positions.shape[0]
    mask_ij = atom_mask[:, None] & atom_mask[None, :]
    mask_float = atom_mask.astype(jnp.float32)

    # Sanitize radii for padded atoms: compute_born_radii does 1/(r - offset)
    # which is NaN when r=0. Use a safe dummy radius for padded atoms.
    safe_radii = jnp.where(atom_mask, radii, jnp.float32(1.5))
    safe_scaled = (
        jnp.where(atom_mask, scaled_radii, jnp.float32(1.2))
        if scaled_radii is not None else None
    )
    safe_charges = jnp.where(atom_mask, charges, jnp.float32(0.0))

    # --- Stage 0: Forward pass — Born radii ---
    def _born_radii_fn(pos):
        br = compute_born_radii(
            pos, safe_radii,
            dielectric_offset=dielectric_offset,
            mask=mask_ij.astype(jnp.float32),
            scaled_radii=safe_scaled,
        )
        # Sanitize: ensure padded atoms have safe born_radii (not NaN)
        return jnp.where(atom_mask, br, jnp.float32(1.5))

    born_radii = _born_radii_fn(positions)

    # --- Stage 1: Forward + VJP of dense GB Coulomb ---
    # GB energy from precomputed Born radii (differentiable w.r.t. pos AND B)
    def _gb_coulomb_fn(pos, br):
        delta = pos[:, None, :] - pos[None, :, :]
        distances = safe_norm(delta, axis=-1)

        br_i = br[:, None]
        br_j = br[None, :]
        eff_dist = f_gb(distances, br_i, br_j)

        tau = (1.0 / solute_dielectric) - (1.0 / solvent_dielectric)
        prefactor = jnp.float32(-0.5) * jnp.float32(COULOMB_CONSTANT) * jnp.float32(tau)

        charge_prod = safe_charges[:, None] * safe_charges[None, :]
        energy_terms = charge_prod / eff_dist

        # Mask: include self-solvation (diagonal), exclude padded atoms
        energy_mask = mask_ij.astype(jnp.float32)
        energy_terms = energy_terms * energy_mask

        return prefactor * jnp.sum(energy_terms)

    # VJP gives us dE_gb/dr_direct and dE_gb/dB in one call
    _, gb_vjp_fn = jax.vjp(_gb_coulomb_fn, positions, born_radii)
    dE_gb_dr_direct, dE_gb_dB = gb_vjp_fn(jnp.ones((), dtype=positions.dtype))

    # --- Stage 2: ACE gradient w.r.t. Born radii (analytical) ---
    # ACE: E_i = C * (R_i + 0.14)^2 * (R_i / B_i)^6
    # dE_i/dB_i = C * (R_i + 0.14)^2 * 6 * R_i^6 * (-1) / B_i^7
    #           = -6 * E_i / B_i
    def _ace_fn(br):
        return jnp.sum(
            compute_ace_nonpolar_energy(safe_radii, br) * mask_float
        )

    # Use vjp for robustness (ACE formula has unit conversions)
    _, ace_vjp_fn = jax.vjp(_ace_fn, born_radii)
    dE_ace_dB = ace_vjp_fn(jnp.ones((), dtype=positions.dtype))[0]

    # --- Stage 3: VJP of Born radii w.r.t. positions ---
    # Total dE/dB = GB contribution + ACE contribution
    total_dE_dB = dE_gb_dB + dE_ace_dB

    _, born_vjp_fn = jax.vjp(_born_radii_fn, positions)
    dE_dr_via_born = born_vjp_fn(total_dE_dB)[0]

    # --- Stage 4: Total force = -(direct + chain rule) ---
    total_grad = dE_gb_dr_direct + dE_dr_via_born

    # Mask padded atoms
    forces = -total_grad * atom_mask[:, None]

    return forces
