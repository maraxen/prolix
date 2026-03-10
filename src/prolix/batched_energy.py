"""Batched energy function for evaluating multiple padded systems via vmap."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
from jax_md import space, energy

from prolix.padding import PaddedSystem
from prolix.types import BondParams, AngleParams, DihedralParams, CmapTorsionIndices
from proxide.physics.constants import COULOMB_CONSTANT
from prolix.physics.generalized_born import compute_ace_nonpolar_energy, compute_born_radii, compute_gb_energy
from prolix.physics.neighbor_list import compute_exclusion_mask_neighbor_list, get_neighbor_exclusion_scales

if TYPE_CHECKING:
    from jax_md.util import Array

# ==============================================================================
# MASKED BONDED TERMS
# ==============================================================================

def _bond_energy_masked(r: Array, indices: Array, params: Array, mask: Array, displacement_fn: space.DisplacementFn) -> Array:
    """Computes harmonic bond energy with per-bond masking."""
    r_i = r[indices[:, 0]]
    r_j = r[indices[:, 1]]
    dr = jax.vmap(displacement_fn)(r_i, r_j)
    dist = space.distance(dr)
    
    # params shape is (N, 2): length, force_constant
    r0 = params[:, 0]
    k = params[:, 1]
    
    e_per_bond = 0.5 * k * (dist - r0) ** 2
    return jnp.sum(e_per_bond * mask)

def _angle_energy_masked(r: Array, indices: Array, params: Array, mask: Array, displacement_fn: space.DisplacementFn) -> Array:
    """Computes harmonic angle energy with per-angle masking."""
    r_i = r[indices[:, 0]]
    r_j = r[indices[:, 1]]
    r_k = r[indices[:, 2]]

    v_ji = jax.vmap(displacement_fn)(r_i, r_j)
    v_jk = jax.vmap(displacement_fn)(r_k, r_j)

    d_ji = space.distance(v_ji)
    d_jk = space.distance(v_jk)

    denom = d_ji * d_jk + 1e-8
    cos_theta = jnp.sum(v_ji * v_jk, axis=-1) / denom
    cos_theta = jnp.clip(cos_theta, -0.999999, 0.999999)
    theta = jnp.arccos(cos_theta)
    
    theta0 = params[:, 0]
    k = params[:, 1]

    e_per_angle = 0.5 * k * (theta - theta0) ** 2
    return jnp.sum(e_per_angle * mask)

def _dihedral_energy_masked(r: Array, indices: Array, params: Array, mask: Array, displacement_fn: space.DisplacementFn) -> Array:
    """Computes periodic dihedral energy with per-dihedral masking."""
    r_i = r[indices[:, 0]]
    r_j = r[indices[:, 1]]
    r_k = r[indices[:, 2]]
    r_l = r[indices[:, 3]]

    b0 = jax.vmap(displacement_fn)(r_j, r_i)
    b1 = jax.vmap(displacement_fn)(r_k, r_j)
    b2 = jax.vmap(displacement_fn)(r_l, r_k)

    b1_norm = jnp.linalg.norm(b1, axis=-1, keepdims=True) + 1e-8
    b1_unit = b1 / b1_norm

    v = b0 - jnp.sum(b0 * b1_unit, axis=-1, keepdims=True) * b1_unit
    w = b2 - jnp.sum(b2 * b1_unit, axis=-1, keepdims=True) * b1_unit

    x = jnp.sum(v * w, axis=-1)
    y = jnp.sum(jnp.cross(b1_unit, v) * w, axis=-1)
    phi = jnp.arctan2(y, x)
    phi = phi - jnp.pi
    
    periodicity = params[:, 0]
    phase = params[:, 1]
    k = params[:, 2]

    e_per_dih = k * (1.0 + jnp.cos(periodicity * phi - phase))
    return jnp.sum(e_per_dih * mask)

def _cmap_energy_masked(r: Array, indices: Array, mask: Array, coeffs: Array | None, displacement_fn: space.DisplacementFn) -> Array:
    """Computes CMAP energy with per-torsion masking."""
    if indices is None or coeffs is None or len(indices) == 0:
        return jnp.array(0.0)
        
    from prolix.physics.cmap import compute_cmap_energy
    from prolix.physics.bonded import compute_dihedral_angles
    
    torsion_indices = jax.vmap(CmapTorsionIndices.from_row)(indices)
    
    phi = compute_dihedral_angles(r, jnp.asarray(torsion_indices.phi_indices), displacement_fn)
    psi = compute_dihedral_angles(r, jnp.asarray(torsion_indices.psi_indices), displacement_fn)
    
    # We maps all torsions to map index 0 if not provided
    map_indices = jnp.zeros(len(indices), dtype=jnp.int32)
    
    e_per_torsion = compute_cmap_energy(psi, phi, map_indices, coeffs)
    return jnp.sum(e_per_torsion * mask)


# ==============================================================================
# MASKED NON-BONDED TERMS (N^2 path only for now)
# ==============================================================================

def _lj_energy_masked(r: Array, sigmas: Array, epsilons: Array, atom_mask: Array, displacement_fn: space.DisplacementFn, soft_core_lambda: Array | None = None) -> Array:
    """Computes Lennard-Jones energy with atom masking.
    
    Uses soft-core Beutler (1994) formulation. When soft_core_lambda=1.0,
    the formula reduces to standard LJ. This avoids recompilation when
    varying lambda across minimization stages.
    """
    # Create N x N masks
    mask_ij = atom_mask[:, None] & atom_mask[None, :]
    
    dr = space.map_product(displacement_fn)(r, r)
    dist = space.distance(dr)
    
    # Mixing rules
    sigma_ij = 0.5 * (sigmas[:, None] + sigmas[None, :])
    epsilon_ij = jnp.sqrt(epsilons[:, None] * epsilons[None, :])
    
    # Unified soft-core LJ: U(r,λ) = 4ελ[1/(α(1-λ)+(r/σ)⁶)² - 1/(α(1-λ)+(r/σ)⁶)]
    # When λ=1.0: α(1-λ)=0, so this becomes standard 4ε[(r/σ)⁻¹² - (r/σ)⁻⁶]
    # Explicit float32 to prevent promotion under jax_enable_x64=True
    lam = jnp.float32(soft_core_lambda) if soft_core_lambda is not None else jnp.float32(1.0)
    alpha = jnp.float32(0.5)
    soft_term = alpha * jnp.maximum(jnp.float32(1.0) - lam, jnp.float32(1e-8))
    r_over_sig = dist / jnp.maximum(sigma_ij, jnp.float32(1e-8))
    r6 = r_over_sig ** 6
    # Add small epsilon to avoid division by zero for self-interactions
    denom = soft_term + r6 + jnp.float32(1e-12)
    e_pair = jnp.float32(4.0) * epsilon_ij * lam * (jnp.float32(1.0) / (denom * denom) - jnp.float32(1.0) / denom)
    
    # Remove self interaction
    n = len(sigmas)
    no_self = 1.0 - jnp.eye(n)
    
    e_pair = e_pair * mask_ij * no_self
    
    # Divide by 2 because we double count (i,j) and (j,i)
    return 0.5 * jnp.sum(e_pair)

def _coulomb_energy_masked(r: Array, charges: Array, atom_mask: Array, displacement_fn: space.DisplacementFn) -> Array:
    """Computes pure Coulomb energy with atom masking."""
    mask_ij = atom_mask[:, None] & atom_mask[None, :]
    
    dr = space.map_product(displacement_fn)(r, r)
    dist = space.distance(dr)
    
    dist = jnp.where(dist < 1e-4, 1.0, dist)
    
    e_pair = COULOMB_CONSTANT * (charges[:, None] * charges[None, :]) / dist
    
    n = len(charges)
    no_self = 1.0 - jnp.eye(n)
    
    e_pair = e_pair * mask_ij * no_self
    return 0.5 * jnp.sum(e_pair)


def _lj_energy_neighbor_list(
    r: 'Array',
    sigmas: 'Array',
    epsilons: 'Array',
    neighbor_idx: 'Array',
    soft_core_lambda: 'Array | None' = None,
) -> 'Array':
    """Computes Lennard-Jones energy using neighbor list indices.

    O(N*K) scaling instead of O(N^2). Each atom i has K neighbor slots.
    Padding neighbors (idx >= N) contribute zero energy.

    Uses the same soft-core Beutler (1994) formulation as _lj_energy_masked.

    Args:
        r: Positions (N, 3).
        sigmas: LJ sigma params (N,).
        epsilons: LJ epsilon params (N,).
        neighbor_idx: Neighbor indices (N, K). Padding sentinel = N.
        soft_core_lambda: Soft-core coupling parameter. 1.0 = standard LJ.
    """
    N = r.shape[0]
    # _K = neighbor_idx.shape[1]

    # Gather neighbor data: (N, K, 3) and (N, K)
    # Clamp indices to [0, N-1] for safe gather; mask handles OOB
    safe_idx = jnp.minimum(neighbor_idx, N - 1)
    r_j = r[safe_idx]                    # (N, K, 3)
    sigma_j = sigmas[safe_idx]           # (N, K)
    epsilon_j = epsilons[safe_idx]       # (N, K)

    # Central atom data broadcast: (N, 1, ...)
    r_i = r[:, None, :]                  # (N, 1, 3)
    sigma_i = sigmas[:, None]            # (N, 1)
    epsilon_i = epsilons[:, None]        # (N, 1)

    # Pairwise distances
    dr = r_i - r_j                       # (N, K, 3)
    dist = jnp.sqrt(jnp.sum(dr ** 2, axis=-1) + 1e-12)  # (N, K)

    # Mixing rules
    sigma_ij = 0.5 * (sigma_i + sigma_j)
    epsilon_ij = jnp.sqrt(epsilon_i * epsilon_j)

    # Soft-core LJ (same formula as _lj_energy_masked)
    lam = jnp.float32(soft_core_lambda) if soft_core_lambda is not None else jnp.float32(1.0)
    alpha = jnp.float32(0.5)
    soft_term = alpha * jnp.maximum(jnp.float32(1.0) - lam, jnp.float32(1e-8))
    r_over_sig = dist / jnp.maximum(sigma_ij, jnp.float32(1e-8))
    r6 = r_over_sig ** 6
    denom = soft_term + r6 + jnp.float32(1e-12)
    e_pair = jnp.float32(4.0) * epsilon_ij * lam * (
        jnp.float32(1.0) / (denom * denom) - jnp.float32(1.0) / denom
    )

    # Mask out padding neighbors (sentinel = N)
    mask = neighbor_idx < N  # (N, K) bool
    e_pair = jnp.where(mask, e_pair, 0.0)

    # Sum. No 0.5 factor: neighbor list is directional (i->j only once
    # per pair IF the NL is symmetric). JAX-MD NLs are symmetric,
    # meaning both (i,j) and (j,i) appear, so we need 0.5.
    return 0.5 * jnp.sum(e_pair)


# ==============================================================================
# BATCHED EVALUATION
# ==============================================================================

def single_padded_energy(sys: PaddedSystem, displacement_fn: space.DisplacementFn, implicit_solvent: bool = True, soft_core_lambda: Array = jnp.array(1.0)) -> Array:
    """Computes total potential energy for a single padded system.

    This is the public API for computing energy of one PaddedSystem.
    It can be used standalone with jax.grad, or batched via make_batched_energy_fn.
    
    Args:
        soft_core_lambda: JAX array for soft-core LJ coupling.
            λ=1.0 → standard LJ, λ<1.0 → soft-core (for staged minimization).
            None defaults to λ=1.0 (standard LJ).
    """
    r = sys.positions
    
    # Bonded terms
    e_bond = _bond_energy_masked(r, sys.bonds, sys.bond_params, sys.bond_mask, displacement_fn)
    e_angle = _angle_energy_masked(r, sys.angles, sys.angle_params, sys.angle_mask, displacement_fn)
    e_dih = _dihedral_energy_masked(r, sys.dihedrals, sys.dihedral_params, sys.dihedral_mask, displacement_fn)
    e_imp = _dihedral_energy_masked(r, sys.impropers, sys.improper_params, sys.improper_mask, displacement_fn)
    
    e_cmap = _cmap_energy_masked(r, sys.cmap_torsions, sys.cmap_mask, sys.cmap_coeffs, displacement_fn)
    
    # Non-bonded
    # NOTE: For phase 2 and simplicity in batched vmap (testing gradients), we use the N^2 path.
    # N*K neighbor list paths require extra state/handling.
    e_lj = _lj_energy_masked(r, sys.sigmas, sys.epsilons, sys.atom_mask, displacement_fn, soft_core_lambda=soft_core_lambda)
    
    if implicit_solvent:
        mask_ij = sys.atom_mask[:, None] & sys.atom_mask[None, :]
        n = len(sys.atom_mask)
        energy_mask = mask_ij * (1.0 - jnp.eye(n))
        e_gb, born_radii = compute_gb_energy(
            positions=r,
            charges=sys.charges,
            radii=sys.radii,
            scaled_radii=sys.scaled_radii,
            mask=sys.atom_mask,
            energy_mask=energy_mask,
            dielectric_offset=0.09,
        )
        e_np = compute_ace_nonpolar_energy(sys.radii, born_radii)
        # Mask out nonpolar energy for padding atoms (ACE is per-atom)
        e_np = jnp.sum(e_np * sys.atom_mask)
        e_elec = e_gb
        e_solv = e_gb + e_np
    else:
        e_elec = _coulomb_energy_masked(r, sys.charges, sys.atom_mask, displacement_fn)
        e_solv = 0.0

    # We assume 1-4 scaling etc. are handled correctly via masking, but for simplicity
    # we omit the full ExclusionSpec processing here. This is focused on batching overhead.
    # In a real system you would add 1-4 exclusions here using map_product + masks.

    return e_bond + e_angle + e_dih + e_imp + e_cmap + e_lj + e_elec + e_solv

def make_batched_energy_fn(displacement_fn: space.DisplacementFn, implicit_solvent: bool = True) -> Callable[[PaddedSystem], Array]:
    """Create a vmap-compatible energy function for padded systems.
    
    Currently implements the full N^2 pairwise computation. Designed for batched
    execution of heterogeneous small systems.
    """
    return jax.vmap(lambda sys: single_padded_energy(sys, displacement_fn, implicit_solvent))
