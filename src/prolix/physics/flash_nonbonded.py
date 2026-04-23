"""FlashMD: Chunked gradient rematerialization for dense O(N²) forces.

Eliminates the HBM bandwidth bottleneck by tiling the N² pair computation
into L2-cache-resident T×T blocks. Uses jax.checkpoint (remat) to discard
forward-pass intermediates and recompute them during the backward pass,
keeping all (T,T) working arrays in L2/SRAM instead of HBM.

Architecture (two-pass):
  Pass 1: Chunked Born radii — tiled descreening integrals
  Pass 2: Fused LJ + Coulomb + GB energy — all from one tiled distance pass

Forces via jax.grad on the total energy. JAX computes LJ, Coulomb, and
the full GB chain-rule force simultaneously in a single backward sweep.

References:
  - FlashAttention (Dao et al., 2022) — same recompute-instead-of-save principle
  - OBC Model II: Onufriev, Bashford, Case, Proteins 55, 383 (2004)

All intermediate arrays are (N, T) or (T, T) — never (N, N).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from proxide.physics import constants
from proxide.physics.constants import COULOMB_CONSTANT

if TYPE_CHECKING:
    from jax_md.util import Array

    from prolix.padding import PaddedSystem


# OBC-II constants (matching generalized_born.py)
ALPHA_OBC = 1.0
BETA_OBC = 0.8
GAMMA_OBC = 4.85

# ACE constants
ACE_COEFF_KJ_NM = 28.3919551
KJ_TO_KCAL = 0.239006


# ═══════════════════════════════════════════════════════════════════════
# Tile helpers
# ═══════════════════════════════════════════════════════════════════════

def _safe_norm(x: Array, axis: int = -1, eps: float = 1e-12) -> Array:
    """Safe norm avoiding NaN gradients at zero."""
    return jnp.sqrt(jnp.sum(x ** 2, axis=axis) + eps)


def _pad_to_tile(arr: Array, T: int, axis: int = 0, pad_value=0.0):
    """Pad array along axis so its length is divisible by T."""
    n = arr.shape[axis]
    remainder = n % T
    if remainder == 0:
        return arr
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, T - remainder)
    return jnp.pad(arr, pad_width, constant_values=pad_value)


# ═══════════════════════════════════════════════════════════════════════
# Pass 1: Chunked Born Radii
# ═══════════════════════════════════════════════════════════════════════

def _pair_integral_tile(
    dist: Array,
    offset_radii_i: Array,
    scaled_radii_j: Array,
) -> Array:
    """Compute OBC-II pair integral for a (N_i, T_j) tile of distances.

    OpenMM formula:
      I = 0.5*(1/L-1/U + 0.25*log(L/U)/r + 0.125*(r-sr2^2/r)*(1/U^2-1/L^2))
    where L = max(or_i, |r-sr_j|), U = r+sr_j.

    Args:
        dist: (N_i, T_j) pairwise distances.
        offset_radii_i: (N_i, 1) offset radii of i atoms.
        scaled_radii_j: (1, T_j) scaled radii of j atoms.

    Returns:
        (N_i, T_j) pair integral values.
    """
    D = jnp.abs(dist - scaled_radii_j)
    L = jnp.maximum(offset_radii_i, D)
    L_safe = jnp.maximum(L, 1e-4)
    U = dist + scaled_radii_j
    U_safe = jnp.maximum(U, 1e-4)
    r_safe = jnp.maximum(dist, 1e-4)

    inv_L = 1.0 / L_safe
    inv_U = 1.0 / U_safe

    term1 = 0.5 * (inv_L - inv_U)
    term2 = 0.25 * jnp.log(L_safe / U_safe) / r_safe
    sr2_sq = scaled_radii_j ** 2
    term3 = 0.125 * (dist - sr2_sq / r_safe) * (inv_U ** 2 - inv_L ** 2)

    total = term1 + term2 + term3
    condition = (dist + scaled_radii_j) > offset_radii_i
    return jnp.where(condition, total, 0.0)


def chunked_born_radii(
    positions: Array,
    radii: Array,
    scaled_radii: Array,
    atom_mask: Array,
    T: int = 256,
    dielectric_offset: float = 0.09,
) -> Array:
    """Compute Born radii via tiled O(N²) descreening.

    Tiles the j-dimension into chunks of T. Each (N, T) tile fits in L2
    cache. jax.checkpoint discards intermediates during forward pass;
    they are recomputed from positions during the backward pass.

    Args:
        positions: (N, 3) atom positions.
        radii: (N,) intrinsic atomic radii.
        scaled_radii: (N,) OBC scaled radii.
        atom_mask: (N,) boolean mask.
        T: Tile size (must evenly divide N).
        dielectric_offset: Born radius offset.

    Returns:
        (N,) Born radii.
    """
    N = positions.shape[0]
    n_tiles = N // T

    offset_radii = radii - dielectric_offset
    offset_radii_col = offset_radii[:, None]  # (N, 1) for broadcasting

    def _tile_descreening(I_sum, j_idx):
        """Accumulate descreening from one j-tile."""
        start = j_idx * T

        @jax.checkpoint
        def _compute_tile(pos, start_idx):
            r_j = jax.lax.dynamic_slice(pos, (start_idx, 0), (T, 3))  # (T, 3)
            sr_j = jax.lax.dynamic_slice(scaled_radii, (start_idx,), (T,))
            mask_j = jax.lax.dynamic_slice(atom_mask, (start_idx,), (T,))

            # Distances: (N, T)
            dr = pos[:, None, :] - r_j[None, :, :]  # (N, T, 3)
            dist = _safe_norm(dr, axis=-1)  # (N, T)

            # Self-exclusion: build self-mask for this tile
            i_indices = jnp.arange(N)[:, None]  # (N, 1)
            j_indices = jnp.arange(T)[None, :] + start_idx  # (1, T)
            not_self = (i_indices != j_indices).astype(jnp.float32)

            # Pair mask
            mask_tile = (atom_mask[:, None] & mask_j[None, :]).astype(
                jnp.float32
            ) * not_self

            # Pair integrals for this tile
            sr_j_row = sr_j[None, :]  # (1, T)
            pair_int = _pair_integral_tile(dist, offset_radii_col, sr_j_row)
            pair_int = pair_int * mask_tile

            return jnp.sum(pair_int, axis=1)  # (N,)

        tile_I = _compute_tile(positions, start)
        return I_sum + tile_I, None

    I_total, _ = jax.lax.scan(
        _tile_descreening, jnp.zeros(N, dtype=positions.dtype),
        jnp.arange(n_tiles),
    )

    # OBC-II Born radii from descreening integral
    psi = offset_radii * I_total
    tanh_arg = ALPHA_OBC * psi - BETA_OBC * psi ** 2 + GAMMA_OBC * psi ** 3
    inv_born = 1.0 / offset_radii - jnp.tanh(tanh_arg) / radii
    born_radii = 1.0 / inv_born

    # Sanitize padding atoms
    born_radii = jnp.where(atom_mask, born_radii, jnp.float32(1.5))

    return born_radii


# ═══════════════════════════════════════════════════════════════════════
# Pass 2: Fused LJ + Coulomb + GB Energy (tiled)
# ═══════════════════════════════════════════════════════════════════════

def chunked_fused_energy(
    positions: Array,
    born_radii: Array,
    charges: Array,
    sigmas: Array,
    epsilons: Array,
    atom_mask: Array,
    T: int = 256,
    soft_core_lambda: float = 1.0,
    solvent_dielectric: float = constants.DIELECTRIC_WATER,
    solute_dielectric: float = constants.DIELECTRIC_PROTEIN,
) -> Array:
    """Fused LJ + Coulomb + GB energy in one tiled O(N²) pass.

    All three potentials share the same tiled distance computation.
    jax.checkpoint ensures intermediates stay in L2 cache.

    Args:
        positions: (N, 3) atom positions.
        born_radii: (N,) precomputed Born radii.
        charges: (N,) partial charges.
        sigmas: (N,) LJ sigma.
        epsilons: (N,) LJ epsilon.
        atom_mask: (N,) boolean mask.
        T: Tile size.
        soft_core_lambda: Soft-core coupling (1.0=standard LJ).
        solvent_dielectric: Solvent dielectric constant.
        solute_dielectric: Solute dielectric constant.

    Returns:
        Scalar total nonbonded energy (kcal/mol).
    """
    N = positions.shape[0]
    n_tiles = N // T

    lam = jnp.float32(soft_core_lambda)
    alpha = jnp.float32(0.5)
    soft_term = alpha * jnp.maximum(1.0 - lam, jnp.float32(1e-8))

    tau = (1.0 / solute_dielectric) - (1.0 / solvent_dielectric)
    gb_prefactor = jnp.float32(-0.5) * jnp.float32(COULOMB_CONSTANT) * jnp.float32(tau)

    def tile_energy_outer(carry, j_idx):
        start = j_idx * T

        @jax.checkpoint
        def _compute_tile_inner(pos, start_idx):
            r_j = jax.lax.dynamic_slice(pos, (start_idx, 0), (T, 3))
            br_j = jax.lax.dynamic_slice(born_radii, (start_idx,), (T,))
            q_j = jax.lax.dynamic_slice(charges, (start_idx,), (T,))
            s_j = jax.lax.dynamic_slice(sigmas, (start_idx,), (T,))
            e_j = jax.lax.dynamic_slice(epsilons, (start_idx,), (T,))
            m_j = jax.lax.dynamic_slice(atom_mask, (start_idx,), (T,))

            # Shared distances (N, T)
            dr = pos[:, None, :] - r_j[None, :, :]
            dist_sq = jnp.sum(dr ** 2, axis=-1) + 1e-10
            dist = jnp.sqrt(dist_sq)

            # Self-exclusion
            i_indices = jnp.arange(N)[:, None]
            j_indices = jnp.arange(T)[None, :] + start_idx
            not_self = (i_indices != j_indices).astype(jnp.float32)

            # Pair mask (no exclusions: scale=1.0 assumed, corrections applied later)
            pair_mask = (atom_mask[:, None] & m_j[None, :]).astype(
                jnp.float32
            ) * not_self

            # --- LJ ---
            sig_ij = 0.5 * (sigmas[:, None] + s_j[None, :])
            sig_ij_safe = jnp.where(
                pair_mask > 0, jnp.maximum(sig_ij, 1e-4), 1.0
            )
            eps_ij = jnp.sqrt(jnp.maximum(
                epsilons[:, None] * e_j[None, :], 0.0
            ))
            r_over_sig = dist / sig_ij_safe
            r6 = r_over_sig ** 6
            D = soft_term + r6 + 1e-12
            e_lj = 4.0 * eps_ij * lam * (1.0 / (D * D) - 1.0 / D)
            e_lj = jnp.where(pair_mask > 0, e_lj, 0.0)

            # --- Coulomb ---
            qq = charges[:, None] * q_j[None, :]
            e_coul = COULOMB_CONSTANT * qq / dist
            e_coul = jnp.where(pair_mask > 0, e_coul, 0.0)

            # --- GB (with self-solvation on diagonal) ---
            br_i = born_radii[:, None]
            br_j_tile = br_j[None, :]
            br_prod = br_i * br_j_tile
            f_gb_d = jnp.sqrt(
                dist_sq + br_prod * jnp.exp(-dist_sq / (4.0 * br_prod + 1e-12))
            )
            gb_mask = (atom_mask[:, None] & m_j[None, :]).astype(jnp.float32)
            e_gb = qq / f_gb_d * gb_mask

            # 0.5× for LJ/Coulomb double-count, GB uses full sum with 0.5 in prefactor
            return 0.5 * jnp.sum(e_lj) + 0.5 * jnp.sum(e_coul) + gb_prefactor * jnp.sum(e_gb)

        tile_e = _compute_tile_inner(positions, start)
        return carry + tile_e, None

    total_energy, _ = jax.lax.scan(
        tile_energy_outer,
        jnp.float32(0.0),
        jnp.arange(n_tiles),
    )

    return total_energy


# ═══════════════════════════════════════════════════════════════════════
# ACE Nonpolar Energy (per-atom, no tiling needed)
# ═══════════════════════════════════════════════════════════════════════

def ace_energy(
    radii: Array,
    born_radii: Array,
    atom_mask: Array,
    dielectric_offset: float = 0.09,
) -> Array:
    """ACE nonpolar solvation energy (scalar).

    E_i = 28.3919551 * (radius_nm + 0.14)^2 * (radius_nm / B_nm)^6  [kJ/mol]

    This is O(N), so no tiling needed.
    """
    offset_radii_nm = (radii - dielectric_offset) / 10.0
    radius_nm = offset_radii_nm + 0.009
    born_radii_nm = born_radii / 10.0

    term1 = (radius_nm + 0.14) ** 2
    term2 = (radius_nm / born_radii_nm) ** 6

    energy_kj = ACE_COEFF_KJ_NM * term1 * term2
    energy_kcal = energy_kj * KJ_TO_KCAL

    return jnp.sum(energy_kcal * atom_mask.astype(jnp.float32))


# ═══════════════════════════════════════════════════════════════════════
# Sparse Exclusion Correction
# ═══════════════════════════════════════════════════════════════════════

def _sparse_exclusion_energy(
    positions: Array,
    charges: Array,
    sigmas: Array,
    epsilons: Array,
    atom_mask: Array,
    excl_indices: Array,
    excl_scales_vdw: Array,
    excl_scales_elec: Array,
    soft_core_lambda: float = 1.0,
) -> Array:
    """Compute the exclusion correction energy.

    For each atom i and its excluded partners j (from excl_indices):
      E_correction_ij = (1 - scale_vdw) * E_lj(i,j) + (1 - scale_elec) * E_coul(i,j)

    This is O(N × max_excl) ≈ O(N), not O(N²).

    The main energy was computed assuming scale=1.0 for all pairs.
    We subtract: E_main - E_correction = E_correct.
    """
    N = positions.shape[0]
    max_excl = excl_indices.shape[1]

    lam = jnp.float32(soft_core_lambda)
    alpha = jnp.float32(0.5)
    soft_term = alpha * jnp.maximum(1.0 - lam, 1e-8)

    # Gather excluded partner positions and params
    # excl_indices: (N, max_excl), values in [0, N), -1 for padding
    safe_idx = jnp.maximum(excl_indices, 0)  # clamp -1 → 0

    r_i = positions  # (N, 3)
    r_j = positions[safe_idx]  # (N, max_excl, 3)

    dr = r_i[:, None, :] - r_j  # (N, max_excl, 3)
    dist = _safe_norm(dr, axis=-1)  # (N, max_excl)

    # Valid exclusion mask: index >= 0 AND real atoms
    valid = (excl_indices >= 0) & atom_mask[:, None]
    valid = valid & atom_mask[safe_idx]
    valid_float = valid.astype(jnp.float32)

    # LJ for excluded pairs
    sig_i = sigmas[:, None]  # (N, 1)
    sig_j = sigmas[safe_idx]  # (N, max_excl)
    sig_ij = 0.5 * (sig_i + sig_j)
    sig_ij_safe = jnp.maximum(sig_ij, 1e-4)

    eps_i = epsilons[:, None]
    eps_j = epsilons[safe_idx]
    eps_ij = jnp.sqrt(jnp.maximum(eps_i * eps_j, 0.0))

    r_over_sig = dist / sig_ij_safe
    r6 = r_over_sig ** 6
    D = soft_term + r6 + 1e-12
    e_lj = 4.0 * eps_ij * lam * (1.0 / (D * D) - 1.0 / D)

    # Coulomb for excluded pairs
    q_i = charges[:, None]
    q_j = charges[safe_idx]
    e_coul = COULOMB_CONSTANT * q_i * q_j / dist

    # Correction = (1 - scale) * energy_at_full_scale
    # excl_scales_vdw is the SCALE factor (0.0=fully excluded, 0.5=1-4, 1.0=no excl)
    # The main energy computed all with scale=1.0.
    # We need to subtract the EXCESS: (1 - scale) * E_pair
    vdw_correction = (1.0 - excl_scales_vdw) * e_lj
    elec_correction = (1.0 - excl_scales_elec) * e_coul

    # 0.5 for double-counting: each (i,j) pair appears as both i→j and j→i
    # in the sparse exclusion list. But check: does the exclusion list
    # list each pair once or twice? If symmetric (both i→j and j→i entries),
    # use 0.5. If one-directional, use 1.0.
    # Our format: excl_indices has both directions, so 0.5×
    total = 0.5 * jnp.sum(
        (vdw_correction + elec_correction) * valid_float
    )

    return total


# ═══════════════════════════════════════════════════════════════════════
# Main Entry Point: FlashMD Forces
# ═══════════════════════════════════════════════════════════════════════

def flash_nonbonded_forces(
    sys: PaddedSystem,
    T: int = 256,
    soft_core_lambda: float = 1.0,
    solvent_dielectric: float = constants.DIELECTRIC_WATER,
    solute_dielectric: float = constants.DIELECTRIC_PROTEIN,
) -> Array:
    """Compute nonbonded forces via FlashMD architecture.

    Two-pass chunked energy with jax.grad for forces:
      Pass 1: Tiled Born radii (descreening integrals)
      Pass 2: Fused LJ + Coulomb + GB energy in one tiled pass
      + ACE nonpolar energy (O(N), no tiling)
      - Sparse exclusion correction

    Forces = -grad(total_energy) w.r.t. positions.

    Args:
        sys: PaddedSystem with all force field parameters.
        T: Tile size for chunking. Must evenly divide N.
        soft_core_lambda: Soft-core coupling (1.0=standard).
        solvent_dielectric: Solvent dielectric.
        solute_dielectric: Solute dielectric.

    Returns:
        Forces (N, 3) in kcal/mol/Å. Zero for padded atoms.
    """
    # Sanitize inputs for padding safety
    safe_radii = jnp.where(sys.atom_mask, sys.radii, jnp.float32(1.5))
    safe_scaled = jnp.where(
        sys.atom_mask, sys.scaled_radii, jnp.float32(1.2)
    )
    safe_charges = jnp.where(sys.atom_mask, sys.charges, jnp.float32(0.0))
    safe_sigmas = jnp.where(sys.atom_mask, sys.sigmas, jnp.float32(1.0))
    safe_epsilons = jnp.where(sys.atom_mask, sys.epsilons, jnp.float32(0.0))

    def _total_energy(positions):
        # Pass 1: Born radii
        born_r = chunked_born_radii(
            positions, safe_radii, safe_scaled, sys.atom_mask, T=T,
        )

        # Pass 2: Fused LJ + Coulomb + GB
        e_fused = chunked_fused_energy(
            positions, born_r, safe_charges, safe_sigmas, safe_epsilons,
            sys.atom_mask, T=T,
            soft_core_lambda=soft_core_lambda,
            solvent_dielectric=solvent_dielectric,
            solute_dielectric=solute_dielectric,
        )

        # ACE nonpolar (O(N), not tiled)
        e_ace = ace_energy(safe_radii, born_r, sys.atom_mask)

        return e_fused + e_ace

    # Forces via jax.grad — JAX computes full chain rule
    # (LJ + Coulomb + GB d/dr + GB dE/dB * dB/dr) in one backward sweep
    forces = -jax.grad(_total_energy)(sys.positions)

    # Sparse exclusion correction
    excl_correction = -jax.grad(
        lambda pos: _sparse_exclusion_energy(
            pos, safe_charges, safe_sigmas, safe_epsilons,
            sys.atom_mask, sys.excl_indices,
            sys.excl_scales_vdw, sys.excl_scales_elec,
            soft_core_lambda=soft_core_lambda,
        )
    )(sys.positions)

    total_forces = (forces - excl_correction) * sys.atom_mask[:, None]
    return total_forces
