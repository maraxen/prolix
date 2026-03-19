"""Fused energy + force computation for neighbor-list MD.

Eliminates jax.grad for LJ by computing analytical forces in the forward pass.
GB still uses checkpointed jax.grad (the chain rule is complex), but the overall
function returns (energy, forces) directly — no need for outer jax.grad.

The key speedup: LJ gradient was the most expensive autodiff component
(~12ms for gradient alone). Analytical forces eliminate this entirely.

Usage:
    energy, forces = fused_energy_and_forces_nl(sys, neighbor_idx, displacement_fn)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax_md import space

from prolix.padding import PaddedSystem
from proxide.physics import constants
from prolix.physics.generalized_born import (
    compute_pair_integral,
    safe_norm,
    ALPHA_OBC,
    BETA_OBC,
    GAMMA_OBC,
)

def lj_energy_and_force_nl(
    positions: jnp.ndarray,
    sigmas: jnp.ndarray,
    epsilons: jnp.ndarray,
    neighbor_idx: jnp.ndarray,
    soft_core_lambda: float = 1.0,
    excl_scales_vdw: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute LJ energy and analytical forces via streamed O(N) memory pass.

    Args:
        excl_scales_vdw: (N, K) LJ scale factors from exclusion lookup.
            1.0 = full, 0.0 = excluded, 0.5 = 1-4 scaled. If None, no exclusions.

    Returns:
        energy: scalar total LJ energy
        forces: (N, 3) per-atom force array
    """
    N = positions.shape[0]
    K = neighbor_idx.shape[1]
    
    lam = jnp.float32(soft_core_lambda)
    alpha = jnp.float32(0.5)
    soft_term = alpha * jnp.maximum(jnp.float32(1.0) - lam, jnp.float32(1e-8))

    def _scan_fn(carry, k):
        e_acc, f_acc = carry
        
        idx_k = neighbor_idx[:, k]
        safe_idx_k = jnp.minimum(idx_k, N - 1)
        mask_k = idx_k < N
        
        pos_j = positions[safe_idx_k]
        dr = positions - pos_j # (N, 3)
        dist = jnp.sqrt(jnp.sum(dr ** 2, axis=-1) + 1e-12) # (N,)
        
        sigma_ij = 0.5 * (sigmas + sigmas[safe_idx_k])
        epsilon_ij = jnp.sqrt(epsilons * epsilons[safe_idx_k])

        r_over_sig = dist / jnp.maximum(sigma_ij, jnp.float32(1e-8))
        r6 = r_over_sig ** 6
        denom = soft_term + r6 + jnp.float32(1e-12)

        e_pair = jnp.float32(4.0) * epsilon_ij * lam * (
            jnp.float32(1.0) / (denom * denom)
            - jnp.float32(1.0) / denom
        )
        e_pair = jnp.where(mask_k, e_pair, 0.0)

        # Apply exclusion scales (1-2/1-3 zeroed, 1-4 scaled)
        if excl_scales_vdw is not None:
            excl_scale_k = excl_scales_vdw[:, k]  # (N,)
            e_pair = e_pair * excl_scale_k
        
        sigma_ij6 = jnp.maximum(sigma_ij ** 6, jnp.float32(1e-48))
        ddenom_ddist = jnp.float32(6.0) * dist ** 5 / sigma_ij6

        de_ddist = jnp.float32(4.0) * epsilon_ij * lam * (
            jnp.float32(1.0) / (denom ** 2)
            - jnp.float32(2.0) / (denom ** 3)
        ) * ddenom_ddist
        
        de_ddist = jnp.where(mask_k, de_ddist, 0.0)
        # Apply exclusion scales to gradient too
        if excl_scales_vdw is not None:
            excl_scale_k = excl_scales_vdw[:, k]  # (N,)
            de_ddist = de_ddist * excl_scale_k
        
        unit_dr = dr / (dist[..., None] + 1e-12)
        f_pair = de_ddist[..., None] * unit_dr # (N, 3)
        f_pair_j = -f_pair # Force on central atom is -de/d(dist) * dr/dist

        return (e_acc + e_pair, f_acc + f_pair_j), None

    init_carry = (
        jnp.zeros(N, dtype=jnp.float32), 
        jnp.zeros((N, 3), dtype=jnp.float32)
    )
    
    (e_total_arr, f_total), _ = jax.lax.scan(_scan_fn, init_carry, jnp.arange(K))
    
    # 0.5 factor because symmetric interaction pairs are counted twice in neighbor lists
    energy = 0.5 * jnp.sum(e_total_arr)
    
    # NOTE: The forces are symmetric. In the dense pairwise kernel, we sum over both
    # axes correctly. But with a neighbor list, each atom acts as a "central" atom exactly once
    # per pair. Since we sum up f_pair_j for each central atom, the forces are already correct
    # in magnitude, NO 0.5 scaling needed.
    
    return energy, f_total


# ==============================================================================
# BONDED ENERGY (for jax.grad — O(N), cheap)
# ==============================================================================


def _bonded_energy_from_positions(
    positions: jnp.ndarray,
    sys: PaddedSystem,
    displacement_fn,
) -> jnp.ndarray:
    """Compute total bonded energy from positions (for jax.grad)."""
    from prolix.batched_energy import (
        _bond_energy_masked,
        _angle_energy_masked,
        _dihedral_energy_masked,
        _cmap_energy_masked,
    )
    e_bond = _bond_energy_masked(positions, sys.bonds, sys.bond_params, sys.bond_mask, displacement_fn)
    e_angle = _angle_energy_masked(positions, sys.angles, sys.angle_params, sys.angle_mask, displacement_fn)
    e_dih = _dihedral_energy_masked(positions, sys.dihedrals, sys.dihedral_params, sys.dihedral_mask, displacement_fn)
    e_imp = _dihedral_energy_masked(positions, sys.impropers, sys.improper_params, sys.improper_mask, displacement_fn)
    e_cmap = _cmap_energy_masked(positions, sys.cmap_torsions, sys.cmap_mask, sys.cmap_coeffs, displacement_fn)
    return e_bond + e_angle + e_dih + e_imp + e_cmap


# ==============================================================================
# GB ENERGY (for jax.grad — matches reference CVJP path exactly)
# ==============================================================================


def _gb_energy_from_positions(
    positions: jnp.ndarray,
    charges: jnp.ndarray,
    radii: jnp.ndarray,
    atom_mask: jnp.ndarray,
    neighbor_idx: jnp.ndarray,
    dielectric_offset: float = 0.09,
) -> jnp.ndarray:
    """Compute GB solvation energy from positions (for jax.grad).

    Matches the CVJP reference path: e_gb + e_np.
    """
    from prolix.physics.generalized_born import (
        compute_gb_energy_neighbor_list,
        compute_ace_nonpolar_energy,
    )
    e_gb, born_radii = compute_gb_energy_neighbor_list(
        positions=positions,
        charges=charges,
        radii=radii,
        neighbor_idx=neighbor_idx,
        dielectric_offset=dielectric_offset,
    )
    e_np = compute_ace_nonpolar_energy(radii, born_radii)
    e_np = jnp.sum(e_np * atom_mask)
    return e_gb + e_np


# ==============================================================================
# PUBLIC API: FUSED ENERGY + FORCES
# ==============================================================================


def fused_energy_and_forces_nl(
    sys: PaddedSystem,
    neighbor_idx: jnp.ndarray,
    displacement_fn,
    implicit_solvent: bool = True,
    soft_core_lambda: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute total energy and forces with analytical LJ gradient.

    Architecture:
    - LJ: fully analytical energy + force (no jax.grad, single pass)
    - GB: checkpointed jax.grad (complex chain rule, recomputed on backward)
    - Bonded: jax.grad (O(N), ~0.1ms, negligible cost)

    The energy sum matches single_padded_energy_nl_cvjp exactly:
        e_bonded + e_lj + e_solv + e_solv
    where e_solv = e_gb_polar + e_nonpolar (GB includes electrostatics,
    counted twice to match the reference e_elec + e_solv pattern).

    Args:
        sys: PaddedSystem with positions, charges, radii, etc.
        neighbor_idx: (N, K) neighbor indices, sentinel = N.
        displacement_fn: JAX-MD displacement function.
        implicit_solvent: Whether to include GB solvation.
        soft_core_lambda: Soft-core LJ coupling (1.0 = standard).

    Returns:
        (total_energy, forces): scalar energy and (N, 3) force array.
    """
    r = sys.positions

    # Compute (N, K) exclusion scales for neighbor list
    from prolix.physics.neighbor_list import get_neighbor_exclusion_scales
    excl_scales_vdw_nl, _ = get_neighbor_exclusion_scales(
        sys.excl_indices, sys.excl_scales_vdw, sys.excl_scales_elec, neighbor_idx,
    )

    # ── Shared distance computation ──
    dr, dist, safe_idx = compute_pairwise_nl(r, neighbor_idx)

    # ── LJ: fully analytical energy + force (THE KEY OPTIMIZATION) ──
    e_lj, f_lj = lj_energy_and_force_nl(
        dr, dist, sys.sigmas, sys.epsilons,
        safe_idx, neighbor_idx,
        soft_core_lambda=soft_core_lambda,
        excl_scales_vdw=excl_scales_vdw_nl,
    )

    # ── GB: checkpointed jax.grad ──
    if implicit_solvent:
        gb_fn = jax.checkpoint(
            lambda pos: _gb_energy_from_positions(
                pos, sys.charges, sys.radii, sys.atom_mask,
                neighbor_idx,
            )
        )
        e_solv = gb_fn(r)
        f_solv = -jax.grad(gb_fn)(r)
        # Match reference: e_elec + e_solv = 2 * e_solv
        e_gb_total = 2.0 * e_solv
        f_gb_total = 2.0 * f_solv
    else:
        e_gb_total = jnp.float32(0.0)
        f_gb_total = jnp.zeros_like(r)

    # ── Bonded: jax.grad (O(N), ~0.1ms, negligible) ──
    bonded_fn = lambda pos: _bonded_energy_from_positions(pos, sys, displacement_fn)
    e_bonded = bonded_fn(r)
    f_bonded = -jax.grad(bonded_fn)(r)

    total_energy = e_bonded + e_lj + e_gb_total
    total_forces = f_lj + f_gb_total + f_bonded

    return total_energy, total_forces
