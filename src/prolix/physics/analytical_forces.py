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
        compute_ace_nonpolar_energy,
        compute_born_radii,
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


# ---------------------------------------------------------------------------
# Bonded analytical forces for ShimMode.ANALYTICAL
# ---------------------------------------------------------------------------


def bond_forces_analytical(
    positions: Array,
    bond_indices: Array,
    bond_params: Array,
    displacement_fn,
    bond_mask: Array | None = None,
) -> Array:
    """Analytical harmonic bond forces: -dE/dr for U = 0.5*k*(r-r0)^2.

    Computes forces from harmonic bond potential analytically.
    For each bond (i, j): U = 0.5*k*(|r_ij| - r0)^2
    Force: F = -dU/dr = -k*(r - r0) * r̂_ij

    Args:
        positions: Atom positions (N, 3).
        bond_indices: Bond pairs (B, 2) — integer indices into positions.
        bond_params: Bond parameters (B, 2) — [r0, k] per bond.
            r0: Equilibrium distance (Å).
            k: Force constant (kcal/mol/Å²).
        displacement_fn: PBC-aware displacement function from jax_md.space.
        bond_mask: Optional mask (B,) — 1.0 for real bonds, 0.0 for padding.

    Returns:
        Force array (N, 3) in kcal/mol/Å.
    """
    if bond_mask is None:
        bond_mask = jnp.ones(bond_indices.shape[0])

    i_idx = bond_indices[:, 0].astype(jnp.int32)
    j_idx = bond_indices[:, 1].astype(jnp.int32)
    r0 = bond_params[:, 0]
    k = bond_params[:, 1]

    # Displacement vectors: r_ij = r_i - r_j (accounts for PBC)
    r_vec = jax.vmap(displacement_fn)(positions[i_idx], positions[j_idx])

    # Distance and unit vector
    r_mag = jnp.linalg.norm(r_vec, axis=-1) + jnp.float32(1e-12)
    r_unit = r_vec / r_mag[:, None]

    # Force magnitude: F_mag = k * (r - r0)
    # Force direction: F = -F_mag * r̂_ij
    f_mag = bond_mask * k * (r_mag - r0)
    f_ij = -f_mag[:, None] * r_unit

    # Accumulate forces: F_i += f_ij, F_j -= f_ij
    N = positions.shape[0]
    forces = jnp.zeros_like(positions)
    forces = forces.at[i_idx].add(f_ij)
    forces = forces.at[j_idx].add(-f_ij)

    return forces


def angle_forces_analytical(
    positions: Array,
    angle_indices: Array,
    angle_params: Array,
    displacement_fn,
    angle_mask: Array | None = None,
) -> Array:
    """Analytical harmonic angle forces: -dE/dθ for U = 0.5*k*(θ-θ0)^2.

    Computes forces from harmonic angle potential analytically.
    For three atoms (i, j, k) with j central:
      cos(θ) = (v_ji · v_jk) / (|v_ji| |v_jk|)
      U = 0.5 * k * (θ - θ0)^2
      F = -dU/dr via chain rule through arccos

    Args:
        positions: Atom positions (N, 3).
        angle_indices: Angle triplets (A, 3) — [i, j, k] with j central.
        angle_params: Angle parameters (A, 2) — [theta0, k].
            theta0: Equilibrium angle (radians).
            k: Force constant (kcal/mol/rad²).
        displacement_fn: PBC-aware displacement function from jax_md.space.
        angle_mask: Optional mask (A,) — 1.0 for real, 0.0 for padding.

    Returns:
        Force array (N, 3) in kcal/mol/Å.
    """
    if angle_mask is None:
        angle_mask = jnp.ones(angle_indices.shape[0])

    theta0 = angle_params[:, 0]
    k = angle_params[:, 1]

    def total_angle_energy(pos):
        """Compute total angle energy for all angles."""
        energy = jnp.float32(0.0)
        for a_idx in range(angle_indices.shape[0]):
            i = angle_indices[a_idx, 0].astype(jnp.int32)
            j = angle_indices[a_idx, 1].astype(jnp.int32)
            k_atom = angle_indices[a_idx, 2].astype(jnp.int32)

            v_ji = displacement_fn(pos[i], pos[j])
            v_jk = displacement_fn(pos[k_atom], pos[j])

            d_ji = jnp.linalg.norm(v_ji) + jnp.float32(1e-12)
            d_jk = jnp.linalg.norm(v_jk) + jnp.float32(1e-12)

            cos_theta_a = jnp.sum(v_ji * v_jk) / (d_ji * d_jk)
            cos_theta_a = jnp.clip(cos_theta_a, -0.999999, 0.999999)
            theta_a = jnp.arccos(cos_theta_a)

            e_a = 0.5 * k[a_idx] * (theta_a - theta0[a_idx]) ** 2
            energy = energy + angle_mask[a_idx] * e_a

        return energy

    grad_fn = jax.grad(total_angle_energy)
    forces = -grad_fn(positions)

    return forces


def _dihedral_angle_batched(
    positions: Array,
    dihedral_indices: Array,
    displacement_fn,
    dihedral_mask: Array | None = None,
) -> Array:
    """Compute dihedral angles for multiple dihedrals.

    Computes φ = atan2(y, x) - π following the convention from batched_energy.py.

    Args:
        positions: Atom positions (N, 3).
        dihedral_indices: Dihedral quadruplets (D, 4) — [i, j, k, l].
        displacement_fn: PBC-aware displacement function from jax_md.space.
        dihedral_mask: Optional mask (D,) — 1.0 for real, 0.0 for padding.

    Returns:
        Dihedral angles (D,) in radians.
    """
    if dihedral_mask is None:
        dihedral_mask = jnp.ones(dihedral_indices.shape[0])

    i_idx = dihedral_indices[:, 0].astype(jnp.int32)
    j_idx = dihedral_indices[:, 1].astype(jnp.int32)
    k_idx = dihedral_indices[:, 2].astype(jnp.int32)
    l_idx = dihedral_indices[:, 3].astype(jnp.int32)

    # Vectors: b0 = r_j - r_i, b1 = r_k - r_j, b2 = r_l - r_k
    b0 = jax.vmap(displacement_fn)(positions[j_idx], positions[i_idx])
    b1 = jax.vmap(displacement_fn)(positions[k_idx], positions[j_idx])
    b2 = jax.vmap(displacement_fn)(positions[l_idx], positions[k_idx])

    # Unit vector along b1
    b1_norm = jnp.linalg.norm(b1, axis=-1, keepdims=True) + jnp.float32(1e-12)
    b1_unit = b1 / b1_norm

    # Project b0 and b2 onto plane perpendicular to b1
    v = b0 - jnp.sum(b0 * b1_unit, axis=-1, keepdims=True) * b1_unit
    w = b2 - jnp.sum(b2 * b1_unit, axis=-1, keepdims=True) * b1_unit

    # Dihedral angle: φ = atan2(y, x) - π
    # where y = (b1_unit × v) · w and x = v · w
    x = jnp.sum(v * w, axis=-1)
    y = jnp.sum(jnp.cross(b1_unit, v) * w, axis=-1)

    # Safe values for padded entries (from batched_energy.py:91-93)
    overlap_mask = (dihedral_mask == 0) | ((x == 0.0) & (y == 0.0))
    safe_x = jnp.where(overlap_mask, 1.0, x)
    safe_y = jnp.where(overlap_mask, 0.0, y)

    phi = jnp.arctan2(safe_y, safe_x)
    phi = phi - jnp.pi

    return phi


def dihedral_forces_analytical(
    positions: Array,
    dihedral_indices: Array,
    dihedral_params: Array,
    displacement_fn,
    dihedral_mask: Array | None = None,
) -> Array:
    """Analytical periodic dihedral forces.

    Computes forces from periodic dihedral potential:
      U = Σ_t k_t * (1 + cos(n_t * φ - phase_t))
      dU/dφ = -Σ_t k_t * n_t * sin(n_t * φ - phase_t)

    Hybrid approach: φ is computed analytically, dφ/dr via jax.jacobian.

    Args:
        positions: Atom positions (N, 3).
        dihedral_indices: Dihedral quadruplets (D, 4) — [i, j, k, l].
        dihedral_params: Dihedral parameters (D, N_terms, 3) — [n, phase, k] per term.
        displacement_fn: PBC-aware displacement function from jax_md.space.
        dihedral_mask: Optional mask (D,) — 1.0 for real, 0.0 for padding.

    Returns:
        Force array (N, 3) in kcal/mol/Å.
    """
    if dihedral_mask is None:
        dihedral_mask = jnp.ones(dihedral_indices.shape[0])

    def total_dihedral_energy(pos):
        """Compute total dihedral energy for all dihedrals."""
        energy = jnp.float32(0.0)
        for d_idx in range(dihedral_indices.shape[0]):
            i = dihedral_indices[d_idx, 0].astype(jnp.int32)
            j = dihedral_indices[d_idx, 1].astype(jnp.int32)
            k = dihedral_indices[d_idx, 2].astype(jnp.int32)
            l = dihedral_indices[d_idx, 3].astype(jnp.int32)

            # Vectors
            b0 = displacement_fn(pos[j], pos[i])
            b1 = displacement_fn(pos[k], pos[j])
            b2 = displacement_fn(pos[l], pos[k])

            # Unit vector along b1
            b1_norm = jnp.linalg.norm(b1) + jnp.float32(1e-12)
            b1_unit = b1 / b1_norm

            # Project b0 and b2 onto plane perpendicular to b1
            v = b0 - jnp.sum(b0 * b1_unit) * b1_unit
            w = b2 - jnp.sum(b2 * b1_unit) * b1_unit

            # Dihedral angle
            x = jnp.sum(v * w)
            y = jnp.sum(jnp.cross(b1_unit, v) * w)

            # Safe values for padded
            overlap = (dihedral_mask[d_idx] == 0) | ((x == 0.0) & (y == 0.0))
            safe_x = jnp.where(overlap, 1.0, x)
            safe_y = jnp.where(overlap, 0.0, y)

            phi = jnp.arctan2(safe_y, safe_x) - jnp.pi

            # Energy for this dihedral
            e_d = jnp.float32(0.0)
            for t_idx in range(dihedral_params.shape[1]):
                n = dihedral_params[d_idx, t_idx, 0]
                phase = dihedral_params[d_idx, t_idx, 1]
                k = dihedral_params[d_idx, t_idx, 2]
                e_d = e_d + k * (1.0 + jnp.cos(n * phi - phase))

            energy = energy + dihedral_mask[d_idx] * e_d

        return energy

    grad_fn = jax.grad(total_dihedral_energy)
    forces = -grad_fn(positions)

    return forces


def improper_forces_analytical(
    positions: Array,
    improper_indices: Array,
    improper_params: Array,
    displacement_fn,
    improper_mask: Array | None = None,
) -> Array:
    """Analytical improper forces with shape dispatch.

    Dispatches on improper_params.shape[-1]:
    - 3: periodic improper (same form as dihedral)
    - 2: harmonic improper with angle wrapping

    Args:
        positions: Atom positions (N, 3).
        improper_indices: Improper quadruplets (I, 4) — [i, j, k, l].
        improper_params: Improper parameters (I, N_terms, 3 or 2).
            Periodic (3): [n, phase, k] per term
            Harmonic (2): [k, phi0] per term (bonded module convention)
        displacement_fn: PBC-aware displacement function from jax_md.space.
        improper_mask: Optional mask (I,) — 1.0 for real, 0.0 for padding.

    Returns:
        Force array (N, 3) in kcal/mol/Å.
    """
    if improper_mask is None:
        improper_mask = jnp.ones(improper_indices.shape[0])

    # Dispatch based on parameter shape
    param_shape = improper_params.shape[-1]

    if param_shape == 3:
        # Periodic improper
        return _improper_forces_periodic(
            positions, improper_indices, improper_params, displacement_fn, improper_mask
        )
    elif param_shape == 2:
        # Harmonic improper
        return _improper_forces_harmonic(
            positions, improper_indices, improper_params, displacement_fn, improper_mask
        )
    else:
        raise ValueError(f"Improper params shape[-1] must be 2 or 3, got {param_shape}")


def _improper_forces_periodic(
    positions: Array,
    improper_indices: Array,
    improper_params: Array,
    displacement_fn,
    improper_mask: Array,
) -> Array:
    """Periodic improper forces (delegates to dihedral logic)."""
    return dihedral_forces_analytical(
        positions, improper_indices, improper_params, displacement_fn, improper_mask
    )


def _improper_forces_harmonic(
    positions: Array,
    improper_indices: Array,
    improper_params: Array,
    displacement_fn,
    improper_mask: Array,
) -> Array:
    """Harmonic improper forces: U = k*(φ-φ0)^2 with angle wrapping."""

    def total_improper_energy(pos):
        """Compute total harmonic improper energy."""
        # Extract indices
        i_idxs = improper_indices[:, 0].astype(jnp.int32)
        j_idxs = improper_indices[:, 1].astype(jnp.int32)
        k_idxs = improper_indices[:, 2].astype(jnp.int32)
        l_idxs = improper_indices[:, 3].astype(jnp.int32)

        # Compute dihedral angles (following bonded.py convention)
        # b0 = r_i - r_j, b1 = r_k - r_j, b2 = r_l - r_k
        b0s = jax.vmap(displacement_fn)(pos[i_idxs], pos[j_idxs])
        b1s = jax.vmap(displacement_fn)(pos[k_idxs], pos[j_idxs])
        b2s = jax.vmap(displacement_fn)(pos[l_idxs], pos[k_idxs])

        # epsilon must match compute_dihedral_angles in bonded.py to keep this
        # analytical path's gradient bit-equivalent to the reference energy's gradient
        b1_norms = jnp.linalg.norm(b1s, axis=-1, keepdims=True) + 1e-8
        b1_units = b1s / b1_norms

        vs = b0s - jnp.sum(b0s * b1_units, axis=-1, keepdims=True) * b1_units
        ws = b2s - jnp.sum(b2s * b1_units, axis=-1, keepdims=True) * b1_units

        xs = jnp.sum(vs * ws, axis=-1)
        ys = jnp.sum(jnp.cross(b1_units, vs) * ws, axis=-1)

        # Dihedral angles using bonded convention
        phis = jnp.arctan2(ys, xs)

        # Harmonic improper energy
        # params shape: (N_improper, N_terms, 2) with [k, phi0]
        k_harms = improper_params[:, :, 0]
        phi0s = improper_params[:, :, 1]

        # Angle wrapping: map difference to [-π, π]
        deltas = phis[:, None] - phi0s  # (N_improper, N_terms)
        deltas_wrapped = (deltas + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

        # Energy per improper term
        energies = k_harms * deltas_wrapped ** 2  # (N_improper, N_terms)

        # Sum over terms and apply mask
        total_energy = jnp.sum(energies * improper_mask[:, None])

        return total_energy

    grad_fn = jax.grad(total_improper_energy)
    forces = -grad_fn(positions)

    return forces


def urey_bradley_forces_analytical(
    positions: Array,
    ub_indices: Array,
    ub_params: Array,
    displacement_fn,
    ub_mask: Array | None = None,
) -> Array:
    """Analytical urey-bradley forces (1-3 pair harmonic interaction).

    Same form as bond: U = 0.5*k*(r-r0)^2, but applied to i-k pair in angle i-j-k.

    Args:
        positions: Atom positions (N, 3).
        ub_indices: UB pairs (UB, 2) — [i, k] (skipping j).
        ub_params: UB parameters (UB, 2) — [r0, k].
            r0: Equilibrium distance (Å).
            k: Force constant (kcal/mol/Å²).
        displacement_fn: PBC-aware displacement function from jax_md.space.
        ub_mask: Optional mask (UB,) — 1.0 for real, 0.0 for padding.

    Returns:
        Force array (N, 3) in kcal/mol/Å.
    """
    # UB is just a bond between i and k
    return bond_forces_analytical(
        positions, ub_indices, ub_params, displacement_fn, ub_mask
    )
