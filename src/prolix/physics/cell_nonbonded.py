"""Cell-list tiled nonbonded kernels for explicit solvent MD.

Two stencil strategies implemented for benchmarking:
  Option A: lax.scan over 27 shift vectors (lower memory)
  Option B: Grid-shift half-shell with jnp.roll (higher throughput)

Both compute LJ + direct-space Coulomb within cutoff.
Erfc-damped Ewald direct-space is supported via the `alpha` parameter.

All operations are on the dense (Nx, Ny, Nz, M) cell grid.
Ghost atoms (mask=False) have sigma=1.0, epsilon=0.0, charge=0.0
and contribute zero energy by construction.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.lax as lax

from prolix.physics.cell_list import CellList, HALF_SHELL_SHIFTS

if TYPE_CHECKING:
    pass


# ===========================================================================
# Core pair energy computation (shared by both strategies)
# ===========================================================================

def _cell_pair_energy(
    pos_i: jnp.ndarray,     # (Nx, Ny, Nz, M, 3)
    pos_j: jnp.ndarray,     # (Nx, Ny, Nz, M, 3)
    sig_i: jnp.ndarray,     # (Nx, Ny, Nz, M)
    sig_j: jnp.ndarray,     # (Nx, Ny, Nz, M)
    eps_i: jnp.ndarray,     # (Nx, Ny, Nz, M)
    eps_j: jnp.ndarray,     # (Nx, Ny, Nz, M)
    chg_i: jnp.ndarray,     # (Nx, Ny, Nz, M)
    chg_j: jnp.ndarray,     # (Nx, Ny, Nz, M)
    mask_i: jnp.ndarray,    # (Nx, Ny, Nz, M) bool
    mask_j: jnp.ndarray,    # (Nx, Ny, Nz, M) bool
    box_size: jnp.ndarray,  # (3,)
    cutoff: float,
    alpha: float | None = None,
    is_self_interaction: bool = False,
) -> jnp.ndarray:
    """Compute pairwise LJ + Coulomb energy between two cell grids.

    Computes (Nx, Ny, Nz, M_i, M_j) pairwise interactions.

    Args:
        pos_i, pos_j: Cell-grid positions.
        sig_i, sig_j: LJ sigma parameters.
        eps_i, eps_j: LJ epsilon parameters.
        chg_i, chg_j: Partial charges.
        mask_i, mask_j: Real-atom masks.
        box_size: Periodic box dimensions.
        cutoff: Interaction cutoff in Å.
        alpha: Ewald splitting parameter. If None, plain Coulomb.
            If set, computes erfc(α·r)/r instead of 1/r.
        is_self_interaction: If True, exclude i==j diagonal.

    Returns:
        Scalar total energy (kcal/mol).
    """
    # Expand for pairwise: (Nx, Ny, Nz, M, 1, 3) - (Nx, Ny, Nz, 1, M, 3)
    ri = pos_i[..., None, :, :]   # (..., M, 1, 3)
    rj = pos_j[..., None, :, :]   # (..., 1, M, 3)
    # Actually we need: pos_i is (Nx,Ny,Nz,M,3) → expand to (Nx,Ny,Nz,M,1,3)
    #                   pos_j is (Nx,Ny,Nz,M,3) → expand to (Nx,Ny,Nz,1,M,3)
    ri = jnp.expand_dims(pos_i, axis=-2)   # (Nx, Ny, Nz, M, 1, 3)
    rj = jnp.expand_dims(pos_j, axis=-3)   # (Nx, Ny, Nz, 1, M, 3)

    # Minimum image displacement
    dr = ri - rj  # (Nx, Ny, Nz, M, M, 3)
    dr = dr - jnp.round(dr / box_size) * box_size

    # Distance with safe epsilon for gradient
    dist_sq = jnp.sum(dr ** 2, axis=-1)  # (Nx, Ny, Nz, M, M)

    # Pair mask: both real atoms AND within cutoff
    mi = jnp.expand_dims(mask_i, axis=-1)   # (Nx, Ny, Nz, M, 1)
    mj = jnp.expand_dims(mask_j, axis=-2)   # (Nx, Ny, Nz, 1, M)
    pair_mask = (mi & mj).astype(jnp.float32)

    # Self-interaction exclusion (for same-cell computation)
    if is_self_interaction:
        M = pos_i.shape[-2]
        eye = jnp.eye(M, dtype=jnp.float32)
        not_self = 1.0 - eye  # (M, M)
        pair_mask = pair_mask * not_self

    # Cutoff mask
    within_cutoff = (dist_sq < cutoff ** 2).astype(jnp.float32)
    pair_mask = pair_mask * within_cutoff

    # Safe distance: replace masked pairs with 1.0 to prevent NaN
    # (the result is zeroed out by pair_mask anyway)
    safe_dist_sq = jnp.where(pair_mask > 0.0, dist_sq, jnp.float32(1.0))
    dist = jnp.sqrt(safe_dist_sq + jnp.float32(1e-10))

    # --- LJ energy ---
    si = jnp.expand_dims(sig_i, axis=-1)    # (Nx, Ny, Nz, M, 1)
    sj = jnp.expand_dims(sig_j, axis=-2)    # (Nx, Ny, Nz, 1, M)
    sigma_ij = 0.5 * (si + sj)
    sigma_ij_safe = jnp.maximum(sigma_ij, jnp.float32(1e-4))

    ei = jnp.expand_dims(eps_i, axis=-1)
    ej = jnp.expand_dims(eps_j, axis=-2)
    epsilon_ij = jnp.sqrt(jnp.maximum(ei * ej, jnp.float32(0.0)))

    r_over_sig = dist / sigma_ij_safe
    r6 = r_over_sig ** 6
    r12 = r6 ** 2
    # Standard LJ (no soft-core in cell-list — that's for minimization)
    e_lj = jnp.float32(4.0) * epsilon_ij * (1.0 / r12 - 1.0 / r6)

    # --- Coulomb energy ---
    qi = jnp.expand_dims(chg_i, axis=-1)
    qj = jnp.expand_dims(chg_j, axis=-2)
    qq = qi * qj

    from proxide.physics.constants import COULOMB_CONSTANT

    if alpha is not None:
        # Ewald direct-space: erfc(α·r) / r
        erfc_val = jax.scipy.special.erfc(alpha * dist)
        e_coul = COULOMB_CONSTANT * qq * erfc_val / dist
    else:
        # Plain cutoff Coulomb
        e_coul = COULOMB_CONSTANT * qq / dist

    # Apply mask (zero for ghost, out-of-cutoff, and self pairs)
    e_pair = (e_lj + e_coul) * pair_mask

    return jnp.sum(e_pair)


# ===========================================================================
# Option A: lax.scan over 27 shift vectors
# ===========================================================================

def cell_energy_scan(
    cells: CellList,
    box_size: jnp.ndarray,
    cutoff: float,
    alpha: float | None = None,
) -> jnp.ndarray:
    """Compute nonbonded energy via lax.scan over 27-cell stencil.

    Lower peak memory than grid-shift: processes one cell-pair at a time.
    But serial execution limits GPU utilization.

    Args:
        cells: CellList from build_cell_list().
        box_size: (3,) periodic box dimensions.
        cutoff: Interaction cutoff in Å.
        alpha: Ewald splitting parameter (None = plain Coulomb).

    Returns:
        Scalar total nonbonded energy (kcal/mol).
    """
    pos = cells.positions   # (Nx, Ny, Nz, M, 3)
    sig = cells.sigmas      # (Nx, Ny, Nz, M)
    eps = cells.epsilons     # (Nx, Ny, Nz, M)
    chg = cells.charges      # (Nx, Ny, Nz, M)
    msk = cells.mask         # (Nx, Ny, Nz, M)

    # Self-cell energy (exclude i==j diagonal)
    e_self = _cell_pair_energy(
        pos, pos, sig, sig, eps, eps, chg, chg, msk, msk,
        box_size, cutoff, alpha, is_self_interaction=True,
    )
    # 0.5 for double-counting within same cell
    e_self = 0.5 * e_self

    # All 26 neighbor shifts (not just half-shell — simpler for scan)
    all_shifts = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                all_shifts.append((dx, dy, dz))
    shifts_array = jnp.array(all_shifts, dtype=jnp.int32)  # (26, 3)

    def _neighbor_energy(carry, shift):
        """Energy from one neighbor shift direction."""
        dx, dy, dz = shift[0], shift[1], shift[2]
        # jnp.roll shifts the grid — equivalent to looking at neighbor
        nbr_pos = jnp.roll(pos, (-dx, -dy, -dz), axis=(0, 1, 2))
        nbr_sig = jnp.roll(sig, (-dx, -dy, -dz), axis=(0, 1, 2))
        nbr_eps = jnp.roll(eps, (-dx, -dy, -dz), axis=(0, 1, 2))
        nbr_chg = jnp.roll(chg, (-dx, -dy, -dz), axis=(0, 1, 2))
        nbr_msk = jnp.roll(msk, (-dx, -dy, -dz), axis=(0, 1, 2))

        e = _cell_pair_energy(
            pos, nbr_pos, sig, nbr_sig, eps, nbr_eps,
            chg, nbr_chg, msk, nbr_msk,
            box_size, cutoff, alpha, is_self_interaction=False,
        )
        return carry + e, None

    e_neighbors, _ = lax.scan(_neighbor_energy, jnp.float32(0.0), shifts_array)
    # 0.5 for double-counting: each pair appears as (i→j) and (j→i)
    e_neighbors = 0.5 * e_neighbors

    return e_self + e_neighbors


# ===========================================================================
# Option B: Grid-shift half-shell (13 parallel jnp.roll ops)
# ===========================================================================

def cell_energy_grid_shift(
    cells: CellList,
    box_size: jnp.ndarray,
    cutoff: float,
    alpha: float | None = None,
) -> jnp.ndarray:
    """Compute nonbonded energy via grid-shift half-shell.

    Uses 13 positive shift vectors and Newton's 3rd law.
    Each shift is a parallel tensor operation (no serial loop).
    jnp.roll is free in XLA (fused into memory load index).

    Higher peak memory than scan: 13 × (Nx, Ny, Nz, M, M) intermediates.
    But fully parallel, high GPU utilization.

    Args:
        cells: CellList from build_cell_list().
        box_size: (3,) periodic box dimensions.
        cutoff: Interaction cutoff in Å.
        alpha: Ewald splitting parameter (None = plain Coulomb).

    Returns:
        Scalar total nonbonded energy (kcal/mol).
    """
    pos = cells.positions
    sig = cells.sigmas
    eps = cells.epsilons
    chg = cells.charges
    msk = cells.mask

    # Self-cell energy (exclude diagonal, include 0.5× for double-count)
    e_self = 0.5 * _cell_pair_energy(
        pos, pos, sig, sig, eps, eps, chg, chg, msk, msk,
        box_size, cutoff, alpha, is_self_interaction=True,
    )

    # 13 half-shell shifts — each pair counted exactly once
    e_neighbors = jnp.float32(0.0)
    for dx, dy, dz in HALF_SHELL_SHIFTS:
        nbr_pos = jnp.roll(pos, (-dx, -dy, -dz), axis=(0, 1, 2))
        nbr_sig = jnp.roll(sig, (-dx, -dy, -dz), axis=(0, 1, 2))
        nbr_eps = jnp.roll(eps, (-dx, -dy, -dz), axis=(0, 1, 2))
        nbr_chg = jnp.roll(chg, (-dx, -dy, -dz), axis=(0, 1, 2))
        nbr_msk = jnp.roll(msk, (-dx, -dy, -dz), axis=(0, 1, 2))

        # Newton's 3rd law: no 0.5× needed, each pair counted once
        e = _cell_pair_energy(
            pos, nbr_pos, sig, nbr_sig, eps, nbr_eps,
            chg, nbr_chg, msk, nbr_msk,
            box_size, cutoff, alpha, is_self_interaction=False,
        )
        e_neighbors = e_neighbors + e

    return e_self + e_neighbors


# ===========================================================================
# Force computation via jax.grad
# ===========================================================================

def cell_forces_scan(
    cells: CellList,
    all_positions: jnp.ndarray,
    box_size: jnp.ndarray,
    cutoff: float,
    alpha: float | None = None,
) -> jnp.ndarray:
    """Compute forces via cell-list energy + jax.grad.

    Forces are computed w.r.t. the original flat position array,
    then the cell-list is rebuilt internally. This ensures correct
    force routing back to atom indices.

    Args:
        cells: Prebuilt CellList for structure reference.
        all_positions: (N, 3) original flat positions.
        box_size: (3,) box dimensions.
        cutoff: Interaction cutoff.
        alpha: Ewald splitting parameter.

    Returns:
        (N, 3) force array in kcal/mol/Å.
    """
    # For now, use scan strategy. Grid-shift is benchmark variant.
    def _energy_fn(positions):
        # Rebuild cell list from positions (CPU operation — acceptable
        # for force validation, not for production lax.scan)
        from prolix.physics.cell_list import build_cell_list
        new_cells = build_cell_list(
            positions, box_size,
            atom_mask=jnp.ones(positions.shape[0], dtype=bool),
            sigmas=jnp.ones(positions.shape[0]),
            epsilons=jnp.zeros(positions.shape[0]),
            charges=jnp.zeros(positions.shape[0]),
            cutoff=cutoff,
            grid_shape=cells.grid_shape,
        )
        return cell_energy_scan(new_cells, box_size, cutoff, alpha)

    return -jax.grad(_energy_fn)(all_positions)


# ===========================================================================
# Ewald exclusion correction (Layer 2 of two-layer architecture)
# ===========================================================================

def ewald_exclusion_correction(
    positions: jnp.ndarray,      # (N, 3)
    charges: jnp.ndarray,        # (N,)
    atom_mask: jnp.ndarray,      # (N,)
    excl_indices: jnp.ndarray,   # (N, max_excl) int32
    excl_scales_elec: jnp.ndarray,  # (N, max_excl) float32
    alpha: float,
    box_size: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Ewald exclusion correction for bonded pairs.

    Computes: E_correction = sum_{excluded pairs} q_i * q_j * [1/r - erfc(α·r)/r]
                            = sum_{excluded pairs} q_i * q_j * erf(α·r) / r

    For 1-2/1-3 pairs (scale=0.0): subtract full direct-space contribution.
    For 1-4 pairs (scale=0.5): subtract (1-0.5) × direct-space.

    This is O(N × max_excl) and runs once per step.

    Args:
        positions: Atom positions.
        charges: Partial charges.
        atom_mask: Real-atom mask.
        excl_indices: Excluded partner indices (-1 = padding).
        excl_scales_elec: Electrostatic scale factors.
        alpha: Ewald splitting parameter.
        box_size: For minimum image (periodic). None = non-periodic.

    Returns:
        Scalar correction energy (kcal/mol). Subtract from total.
    """
    from proxide.physics.constants import COULOMB_CONSTANT

    N = positions.shape[0]
    safe_idx = jnp.maximum(excl_indices, 0)  # clamp -1 → 0

    r_i = positions[:, None, :]           # (N, 1, 3)
    r_j = positions[safe_idx]             # (N, max_excl, 3)

    dr = r_i - r_j                         # (N, max_excl, 3)

    # Minimum image if periodic
    if box_size is not None:
        dr = dr - jnp.round(dr / box_size) * box_size

    dist = jnp.sqrt(jnp.sum(dr ** 2, axis=-1) + jnp.float32(1e-10))

    # Valid exclusion mask
    valid = (excl_indices >= 0) & atom_mask[:, None]
    valid = valid & atom_mask[safe_idx]
    valid_float = valid.astype(jnp.float32)

    # Charges
    q_i = charges[:, None]
    q_j = charges[safe_idx]

    # erf(α·r) / r = [1 - erfc(α·r)] / r
    # The direct-space Ewald computed erfc(α·r)/r for all pairs.
    # PME reciprocal computed the full 1/r for all pairs.
    # For excluded pairs, we need to subtract the direct-space contribution
    # that shouldn't be there: subtract erfc(α·r)/r, and also subtract
    # the reciprocal contribution: subtract [1/r - erfc(α·r)/r] = erf(α·r)/r
    # Net: subtract erf(α·r)/r for excluded pairs.
    erf_val = jax.scipy.special.erf(alpha * dist)
    e_corr = COULOMB_CONSTANT * q_i * q_j * erf_val / dist

    # Scale: (1 - scale) is how much to subtract
    correction_scale = (1.0 - excl_scales_elec)
    e_corr = e_corr * correction_scale * valid_float

    # 0.5 for symmetric exclusion list (both i→j and j→i present)
    return 0.5 * jnp.sum(e_corr)
