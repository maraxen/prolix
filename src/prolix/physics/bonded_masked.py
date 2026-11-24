"""Masked (padding-safe) bonded energy terms: bonds, angles, dihedrals, CMAP.

Split out of ``prolix.batched_energy`` so ``prolix.physics.flash_explicit`` can use them
without importing ``batched_energy`` (which dispatches back into ``flash_explicit``) --
removing the mutual lazy import between the two modules (Phase 1.2).
``prolix.batched_energy`` re-imports every name for backward compatibility.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax_md import space

from prolix.physics.bonded import compute_dihedral_angles
from prolix.physics.cmap import compute_cmap_energies
from prolix.typing import CmapTorsionIndices

if TYPE_CHECKING:
    from jax_md.util import Array

__all__ = [
    "_angle_energy_masked",
    "_bond_energy_masked",
    "_cmap_energy_masked",
    "_dihedral_energy_masked",
]

# ==============================================================================
# MASKED BONDED TERMS
# ==============================================================================

def _bond_energy_masked(r: Array, indices: Array, params: Array, mask: Array, displacement_fn: space.DisplacementFn) -> Array:
    """Computes harmonic bond energy with per-bond masking."""
    # Cast to int32 to prevent float64 indexing errors under JAX_ENABLE_X64
    r_i = r[indices[:, 0].astype(jnp.int32)]
    r_j = r[indices[:, 1].astype(jnp.int32)]
    dr = jax.vmap(displacement_fn)(r_i, r_j)
    dist = jnp.sqrt(jnp.sum(dr ** 2, axis=-1) + 1e-10)
    
    # params shape is (N, 2): length, force_constant
    r0 = params[:, 0]
    k = params[:, 1]
    
    e_per_bond = 0.5 * k * (dist - r0) ** 2
    return jnp.sum(e_per_bond * mask)

def _angle_energy_masked(r: Array, indices: Array, params: Array, mask: Array, displacement_fn: space.DisplacementFn) -> Array:
    """Computes harmonic angle energy with per-angle masking."""
    # Cast to int32
    r_i = r[indices[:, 0].astype(jnp.int32)]
    r_j = r[indices[:, 1].astype(jnp.int32)]
    r_k = r[indices[:, 2].astype(jnp.int32)]

    v_ji = jax.vmap(displacement_fn)(r_i, r_j)
    v_jk = jax.vmap(displacement_fn)(r_k, r_j)

    d_ji = jnp.sqrt(jnp.sum(v_ji ** 2, axis=-1) + 1e-10)
    d_jk = jnp.sqrt(jnp.sum(v_jk ** 2, axis=-1) + 1e-10)

    denom = d_ji * d_jk + 1e-8
    cos_theta = jnp.sum(v_ji * v_jk, axis=-1) / denom
    cos_theta = jnp.clip(cos_theta, -0.999999, 0.999999)
    theta = jnp.arccos(cos_theta)
    
    theta0 = params[:, 0]
    k = params[:, 1]

    e_per_angle = 0.5 * k * (theta - theta0) ** 2
    return jnp.sum(e_per_angle * mask)

def _dihedral_energy_masked(r: Array, indices: Array, params: Array, mask: Array, displacement_fn: space.DisplacementFn) -> Array:
    """Computes periodic dihedral energy with per-dihedral masking.

    Supports single-term ``(N, 3|4)`` and multi-term ``(N, T, 3)`` params.
    Padded rows are sanitized before ``arctan2`` so NaN gradients cannot leak.
    """
    r_i = r[indices[:, 0].astype(jnp.int32)]
    r_j = r[indices[:, 1].astype(jnp.int32)]
    r_k = r[indices[:, 2].astype(jnp.int32)]
    r_l = r[indices[:, 3].astype(jnp.int32)]

    b0 = jax.vmap(displacement_fn)(r_j, r_i)
    b1 = jax.vmap(displacement_fn)(r_k, r_j)
    b2 = jax.vmap(displacement_fn)(r_l, r_k)

    b1_norm = jnp.sqrt(jnp.sum(b1**2, axis=-1, keepdims=True) + 1e-12)
    b1_unit = b1 / b1_norm

    v = b0 - jnp.sum(b0 * b1_unit, axis=-1, keepdims=True) * b1_unit
    w = b2 - jnp.sum(b2 * b1_unit, axis=-1, keepdims=True) * b1_unit

    x = jnp.sum(v * w, axis=-1)
    y = jnp.sum(jnp.cross(b1_unit, v) * w, axis=-1)

    # Pad safe: Prevent arctan2(0,0) which has a NaN gradient
    overlap_mask = (mask == 0) | ((x == 0.0) & (y == 0.0))
    safe_x = jnp.where(overlap_mask, 1.0, x)
    safe_y = jnp.where(overlap_mask, 0.0, y)

    phi = jnp.arctan2(safe_y, safe_x)
    phi = phi - jnp.pi

    # Multi-term (N, T, 3) vs single-term (N, 3|4).
    if params.ndim == 3:
        n = params[:, :, 0]
        phase = params[:, :, 1]
        k = params[:, :, 2]
        e_per_dih = jnp.sum(k * (1.0 + jnp.cos(n * phi[:, None] - phase)), axis=-1)
    else:
        periodicity = params[:, 0]
        phase = params[:, 1]
        k = params[:, 2]
        e_per_dih = k * (1.0 + jnp.cos(periodicity * phi - phase))
    return jnp.sum(jnp.where(mask != 0, e_per_dih, 0.0))


def _cmap_energy_masked(
    r: Array,
    indices: Array,
    map_indices: Array,
    mask: Array,
    coeffs: Array | None,
    displacement_fn: space.DisplacementFn,
    *,
    first_is_psi: bool = True,
) -> Array:
    """Computes CMAP energy with per-torsion masking."""
    if indices is None or coeffs is None or len(indices) == 0:
        return jnp.array(0.0, dtype=r.dtype)

    indices = indices.astype(jnp.int32)
    torsion_indices = jax.vmap(CmapTorsionIndices.from_row)(indices)

    phi = compute_dihedral_angles(r, jnp.asarray(torsion_indices.phi_indices), displacement_fn)
    psi = compute_dihedral_angles(r, jnp.asarray(torsion_indices.psi_indices), displacement_fn)

    if first_is_psi:
        e_per_torsion = compute_cmap_energies(psi, phi, map_indices, coeffs)
    else:
        e_per_torsion = compute_cmap_energies(phi, psi, map_indices, coeffs)
    return jnp.sum(jnp.where(mask != 0, e_per_torsion * mask, 0.0))


