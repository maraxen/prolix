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

    b1_norm = jnp.sqrt(jnp.sum(b1**2, axis=-1, keepdims=True) + 1e-12)
    b1_unit = b1 / b1_norm

    v = b0 - jnp.sum(b0 * b1_unit, axis=-1, keepdims=True) * b1_unit
    w = b2 - jnp.sum(b2 * b1_unit, axis=-1, keepdims=True) * b1_unit

    x = jnp.sum(v * w, axis=-1)
    y = jnp.sum(jnp.cross(b1_unit, v) * w, axis=-1)
    
    # Pad safe: Prevent arctan2(0,0) which has a NaN gradient
    safe_x = jnp.where(mask == 0, 1.0, x)
    safe_y = jnp.where(mask == 0, 0.0, y)
    
    phi = jnp.arctan2(safe_y, safe_x)
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
    from prolix.physics.system import compute_dihedral_angles
    
    torsion_indices = jax.vmap(CmapTorsionIndices.from_row)(indices)
    
    phi = compute_dihedral_angles(r, jnp.asarray(torsion_indices.phi_indices), displacement_fn)
    psi = compute_dihedral_angles(r, jnp.asarray(torsion_indices.psi_indices), displacement_fn)
    
    # We maps all torsions to map index 0 if not provided
    map_indices = jnp.zeros(len(indices), dtype=jnp.int32)
    
    e_per_torsion = compute_cmap_energy(psi, phi, map_indices, coeffs)
    return jnp.sum(e_per_torsion * mask)


# ==============================================================================
# POSITION RESTRAINTS (for FIRE minimization)
# ==============================================================================

def _position_restraint_energy(
    r: Array, r_ref: Array,
    k_restraint: Array, atom_mask: Array,
) -> Array:
    """Harmonic position restraint on real atoms.

    E_restraint = 0.5 * k * sum_i(|r_i - r_ref_i|^2 * mask_i)

    Used during FIRE minimization to prevent bonded geometry distortion.
    k_restraint is a JAX scalar so the same traced function works across
    all minimization stages (staged decrease from 100 to 0).

    Args:
        r: Current positions (N, 3).
        r_ref: Reference positions (N, 3) — typically the starting geometry.
        k_restraint: Spring constant (kcal/mol/Å²). JAX scalar, NOT Python float.
        atom_mask: (N,) bool mask for real atoms.
    """
    dr = r - r_ref
    dist_sq = jnp.sum(dr ** 2, axis=-1)  # (N,)
    return 0.5 * k_restraint * jnp.sum(dist_sq * atom_mask)


# ==============================================================================
# MASKED NON-BONDED TERMS (N^2 path only for now)
# ==============================================================================

def _build_dense_exclusion_scales(
    excl_indices: 'Array',
    excl_scales: 'Array',
    N: int,
) -> 'Array':
    """Convert sparse (N, M) exclusion arrays to dense (N, N) scale matrix.

    Uses fori_loop over M (typically 32) exclusion slots. Each iteration
    scatters one column of exclusion data into the N×N output.

    Returns:
        (N, N) float32 scale matrix. 1.0 = full interaction, 0.0 = excluded,
        0.5 = 1-4 scaled (for vdw), 1/1.2 = 1-4 scaled (for elec).
    """
    M = excl_indices.shape[1]
    scale = jnp.ones((N, N), dtype=jnp.float32)

    row_idx = jnp.arange(N)

    def body(i, s):
        idx_col = excl_indices[:, i]   # (N,)
        sc = excl_scales[:, i]         # (N,)
        valid = idx_col >= 0           # (N,) bool
        safe_col = jnp.where(valid, idx_col, 0)
        # Scatter: for each row n, set s[n, safe_col[n]] = sc[n] if valid
        new_vals = jnp.where(valid, sc, s[row_idx, safe_col])
        s = s.at[row_idx, safe_col].set(new_vals)
        return s

    return jax.lax.fori_loop(0, M, body, scale)

def _lj_energy_masked(
    r: Array, sigmas: Array, epsilons: Array, atom_mask: Array,
    displacement_fn: space.DisplacementFn,
    soft_core_lambda: Array | None = None,
    excl_scale_vdw: Array | None = None,
) -> Array:
    """Computes Lennard-Jones energy with atom masking and exclusion scaling.
    
    Uses soft-core Beutler (1994) formulation. When soft_core_lambda=1.0,
    the formula reduces to standard LJ. This avoids recompilation when
    varying lambda across minimization stages.

    Gradient safety: Padded atoms have sigma=0, causing (dist/sigma)^6 to
    overflow float32. Even masked by mask_ij, NaN*0=NaN in IEEE arithmetic.
    We use jnp.where to substitute safe values BEFORE the LJ computation,
    ensuring the backward pass never encounters overflow on any code path.

    Args:
        excl_scale_vdw: (N, N) scale matrix for excluded/scaled pairs.
            1.0 = full interaction, 0.0 = excluded, 0.5 = 1-4 scaled.
            If None, no exclusions are applied (legacy behavior).
    """
    n = len(sigmas)
    # Combined mask: real-atom pair AND not self-interaction
    mask_ij = (atom_mask[:, None] & atom_mask[None, :]) * (1.0 - jnp.eye(n))
    
    # Apply exclusion scales into the mask (0 = excluded, 0.5 = 1-4 scaled)
    if excl_scale_vdw is not None:
        mask_ij = mask_ij * excl_scale_vdw
    
    dr = space.map_product(displacement_fn)(r, r)
    # Gradient-safe distance: sqrt(sum(dr²) + eps) avoids d/dr sqrt(0) = NaN
    dist = jnp.sqrt(jnp.sum(dr ** 2, axis=-1) + jnp.float32(1e-10))
    
    # Mixing rules with safe values for padded atoms
    # Use jnp.where to substitute 1.0 for sigma/epsilon where mask=0,
    # preventing overflow in (dist/sigma)^6 for padded-atom pairs.
    mask_bool = mask_ij > 0.0
    sigma_ij = 0.5 * (sigmas[:, None] + sigmas[None, :])
    sigma_ij_safe = jnp.where(mask_bool, jnp.maximum(sigma_ij, jnp.float32(1e-4)), jnp.float32(1.0))
    epsilon_ij = jnp.sqrt(jnp.maximum(epsilons[:, None] * epsilons[None, :], jnp.float32(0.0)))
    
    # Soft-core LJ: U(r,λ) = 4ελ[1/(α(1-λ)+(r/σ)⁶)² - 1/(α(1-λ)+(r/σ)⁶)]
    lam = jnp.float32(soft_core_lambda) if soft_core_lambda is not None else jnp.float32(1.0)
    alpha = jnp.float32(0.5)
    soft_term = alpha * jnp.maximum(jnp.float32(1.0) - lam, jnp.float32(1e-8))
    r_over_sig = dist / sigma_ij_safe
    r6 = r_over_sig ** 6
    denom = soft_term + r6 + jnp.float32(1e-12)
    e_pair = jnp.float32(4.0) * epsilon_ij * lam * (
        jnp.float32(1.0) / (denom * denom) - jnp.float32(1.0) / denom
    )
    
    # Apply combined mask (real pairs, no-self, exclusion scales)
    # Use jnp.where instead of multiplication to prevent NaN * 0 = NaN
    e_pair = jnp.where(mask_bool, e_pair * mask_ij, jnp.float32(0.0))
    
    # Divide by 2 because we double count (i,j) and (j,i)
    return 0.5 * jnp.sum(e_pair)

def _coulomb_energy_masked(
    r: Array, charges: Array, atom_mask: Array,
    displacement_fn: space.DisplacementFn,
    excl_scale_elec: Array | None = None,
) -> Array:
    """Computes pure Coulomb energy with atom masking and exclusion scaling.

    Args:
        excl_scale_elec: (N, N) scale matrix for excluded/scaled pairs.
            If None, no exclusions are applied (legacy behavior).
    """
    mask_ij = atom_mask[:, None] & atom_mask[None, :]
    
    dr = space.map_product(displacement_fn)(r, r)
    # Gradient-safe distance (same rationale as LJ — sqrt(0) = NaN gradient)
    dist = jnp.sqrt(jnp.sum(dr ** 2, axis=-1) + jnp.float32(1e-10))
    
    dist = jnp.where(dist < 1e-4, 1.0, dist)
    
    e_pair = COULOMB_CONSTANT * (charges[:, None] * charges[None, :]) / dist
    
    n = len(charges)
    no_self = 1.0 - jnp.eye(n)
    
    e_pair = e_pair * mask_ij * no_self

    # Apply exclusion scales (1-2/1-3 zeroed, 1-4 scaled)
    if excl_scale_elec is not None:
        e_pair = e_pair * excl_scale_elec

    return 0.5 * jnp.sum(e_pair)


def _lj_energy_neighbor_list(
    r: 'Array',
    sigmas: 'Array',
    epsilons: 'Array',
    neighbor_idx: 'Array',
    soft_core_lambda: 'Array | None' = None,
    excl_scales_vdw: 'Array | None' = None,
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
        excl_scales_vdw: (N, K) LJ scale factors from exclusion lookup.
            1.0 = full, 0.0 = excluded, 0.5 = 1-4 scaled. If None, no exclusions.
    """
    N = r.shape[0]

    # Gather neighbor data: (N, K, 3) and (N, K)
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

    # Apply exclusion scales (1-2/1-3 zeroed, 1-4 scaled)
    if excl_scales_vdw is not None:
        e_pair = e_pair * excl_scales_vdw

    # Sum. JAX-MD NLs are symmetric (both (i,j) and (j,i) appear), so 0.5.
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
    
    # Non-bonded — use precomputed dense exclusion scale matrices if available.
    # These are topology constants (independent of r). Precomputing once
    # avoids the costly fori_loop+scatter rebuild every step.
    # stop_gradient is critical: the fori_loop+scatter backward pass in
    # _build_dense_exclusion_scales produces NaN due to overlapping scatter
    # indices. These matrices are topology constants (independent of r),
    # so cutting the gradient is both correct and necessary.
    N = len(sys.atom_mask)
    if sys.dense_excl_scale_vdw is not None:
        excl_scale_vdw = sys.dense_excl_scale_vdw
    else:
        excl_scale_vdw = jax.lax.stop_gradient(_build_dense_exclusion_scales(
            sys.excl_indices, sys.excl_scales_vdw, N,
        ))
    if sys.dense_excl_scale_elec is not None:
        excl_scale_elec = sys.dense_excl_scale_elec
    else:
        excl_scale_elec = jax.lax.stop_gradient(_build_dense_exclusion_scales(
            sys.excl_indices, sys.excl_scales_elec, N,
        ))

    e_lj = _lj_energy_masked(
        r, sys.sigmas, sys.epsilons, sys.atom_mask, displacement_fn,
        soft_core_lambda=soft_core_lambda, excl_scale_vdw=excl_scale_vdw,
    )
    
    # Unconditionally compute dense Coulomb interaction (vacuum electrostatic)
    # excl_scale_elec already computed above with stop_gradient — reuse it
    e_elec = _coulomb_energy_masked(
        r, sys.charges, sys.atom_mask, displacement_fn,
        excl_scale_elec=excl_scale_elec,
    )

    if implicit_solvent:
        mask_ij = sys.atom_mask[:, None] & sys.atom_mask[None, :]
        energy_mask = mask_ij * (1.0 - jnp.eye(N))
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
        e_solv = e_gb + e_np
    else:
        e_solv = 0.0

    return e_bond + e_angle + e_dih + e_imp + e_cmap + e_lj + e_elec + e_solv


def single_padded_force(
    sys: PaddedSystem,
    displacement_fn: space.DisplacementFn,
    implicit_solvent: bool = True,
    soft_core_lambda: Array = jnp.array(1.0),
    use_flash: bool = True,
) -> Array:
    """Compute forces for a padded system.

    Two paths:
      - FlashMD (use_flash=True, default): Chunked+checkpointed nonbonded
        energy via tiled O(N²) with sparse exclusion correction. No dense
        (N,N) exclusion matrices needed — memory is O(N) per replica.
      - Legacy (use_flash=False): Separate analytical LJ, Coulomb, GB
        force functions using dense (N,N) exclusion matrices.

    Bonded terms (bonds, angles, dihedrals, impropers, CMAP) are always
    computed via jax.grad — safe because they use indexed gathers (no N²).

    Args:
        sys: A PaddedSystem with all force field parameters.
        displacement_fn: JAX-MD displacement function.
        implicit_solvent: Whether to include GB solvation.
        soft_core_lambda: Soft-core coupling for LJ. 1.0 = standard.
        use_flash: Use FlashMD architecture (default True). Set False to
            use legacy dense-matrix path (requires dense_excl_scale_* fields).

    Returns:
        Forces (N, 3) in kcal/mol/Å. F = -dE/dr.
    """
    r = sys.positions
    N = len(sys.atom_mask)

    # --- Bonded energy only (safe for jax.grad — no N² operations) ---
    def bonded_energy(positions):
        e_bond = _bond_energy_masked(
            positions, sys.bonds, sys.bond_params, sys.bond_mask, displacement_fn,
        )
        e_angle = _angle_energy_masked(
            positions, sys.angles, sys.angle_params, sys.angle_mask, displacement_fn,
        )
        e_dih = _dihedral_energy_masked(
            positions, sys.dihedrals, sys.dihedral_params,
            sys.dihedral_mask, displacement_fn,
        )
        e_imp = _dihedral_energy_masked(
            positions, sys.impropers, sys.improper_params,
            sys.improper_mask, displacement_fn,
        )
        e_cmap = _cmap_energy_masked(
            positions, sys.cmap_torsions, sys.cmap_mask,
            sys.cmap_coeffs, displacement_fn,
        )

        return e_bond + e_angle + e_dih + e_imp + e_cmap

    # Bonded forces via autodiff (safe path — no N² operations)
    bonded_force = -jax.grad(bonded_energy)(r)

    if use_flash:
        # --- FlashMD path: O(N) memory, sparse exclusions ---
        from prolix.physics.flash_nonbonded import flash_nonbonded_forces
        f_nonbonded = flash_nonbonded_forces(
            sys,
            soft_core_lambda=soft_core_lambda,
        )
        total_force = bonded_force + f_nonbonded
    else:
        # --- Legacy path: dense (N,N) exclusion matrices required ---
        from prolix.physics.analytical_forces import (
            lj_forces_dense,
            coulomb_forces_dense,
            gb_ace_forces_dense,
        )

        if sys.dense_excl_scale_vdw is not None:
            excl_scale_vdw = sys.dense_excl_scale_vdw
        else:
            excl_scale_vdw = jax.lax.stop_gradient(_build_dense_exclusion_scales(
                sys.excl_indices, sys.excl_scales_vdw, N,
            ))
        if sys.dense_excl_scale_elec is not None:
            excl_scale_elec = sys.dense_excl_scale_elec
        else:
            excl_scale_elec = jax.lax.stop_gradient(_build_dense_exclusion_scales(
                sys.excl_indices, sys.excl_scales_elec, N,
            ))

        f_lj = lj_forces_dense(
            r, sys.sigmas, sys.epsilons, sys.atom_mask,
            soft_core_lambda=soft_core_lambda,
            excl_scale_vdw=excl_scale_vdw,
        )
        f_coulomb = coulomb_forces_dense(
            r, sys.charges, sys.atom_mask,
            excl_scale_elec=excl_scale_elec,
        )

        f_solv = jnp.zeros_like(r)
        if implicit_solvent:
            f_solv = gb_ace_forces_dense(
                r, sys.charges, sys.radii, sys.scaled_radii, sys.atom_mask,
            )

        total_force = bonded_force + f_lj + f_coulomb + f_solv

    # Belt-and-suspenders: zero forces on padding atoms.
    total_force = total_force * sys.atom_mask[:, None]
    return total_force.astype(r.dtype)

def make_batched_energy_fn(displacement_fn: space.DisplacementFn, implicit_solvent: bool = True) -> Callable[[PaddedSystem], Array]:
    """Create a vmap-compatible energy function for padded systems.
    
    Currently implements the full N^2 pairwise computation. Designed for batched
    execution of heterogeneous small systems.
    """
    return jax.vmap(lambda sys: single_padded_energy(sys, displacement_fn, implicit_solvent))


# ==============================================================================
# CUSTOM VJP LJ ENERGY — avoids scatter-add in backward pass
# ==============================================================================

def _make_lj_energy_nl_cvjp(neighbor_idx, soft_core_lambda=jnp.float32(1.0), excl_scales_vdw=None):
    """Factory creating a custom-VJP LJ energy function.

    Captures neighbor_idx and excl_scales_vdw via closure (non-differentiable).
    Returns a function f(r, sigmas, epsilons) → scalar energy.

    Key insight: with a symmetric NL and the 0.5 factor in the forward pass,
    the gradient w.r.t. r_i is simply the sum of pair-force vectors from
    row i of the NL. No scatter-add to other atoms is needed because the
    (j,i) pair in the NL already contributes the reverse force.

    This eliminates the expensive gather→scatter-add pattern that standard
    JAX autodiff generates for r[safe_idx] operations.
    """
    N_sentinel = neighbor_idx.shape[0]  # = N (padded atom count)

    def _compute_intermediates(r, sigmas, epsilons):
        """Shared forward computation for both energy and gradient."""
        safe_idx = jnp.minimum(neighbor_idx, N_sentinel - 1)
        r_j = r[safe_idx]                    # (N, K, 3)
        r_i = r[:, None, :]                  # (N, 1, 3)
        dr = r_i - r_j                       # (N, K, 3)
        dist = jnp.sqrt(jnp.sum(dr ** 2, axis=-1) + 1e-12)  # (N, K)

        sigma_ij = 0.5 * (sigmas[:, None] + sigmas[safe_idx])
        epsilon_ij = jnp.sqrt(epsilons[:, None] * epsilons[safe_idx])

        lam = jnp.float32(soft_core_lambda)
        alpha = jnp.float32(0.5)
        soft_term = alpha * jnp.maximum(jnp.float32(1.0) - lam, jnp.float32(1e-8))
        r_over_sig = dist / jnp.maximum(sigma_ij, jnp.float32(1e-8))
        r6 = r_over_sig ** 6
        denom = soft_term + r6 + jnp.float32(1e-12)

        mask = neighbor_idx < N_sentinel  # (N, K)
        return dr, dist, sigma_ij, epsilon_ij, lam, denom, mask

    @jax.custom_vjp
    def lj_energy_cvjp(r, sigmas, epsilons):
        dr, dist, sigma_ij, epsilon_ij, lam, denom, mask = \
            _compute_intermediates(r, sigmas, epsilons)
        e_pair = jnp.float32(4.0) * epsilon_ij * lam * (
            jnp.float32(1.0) / (denom * denom)
            - jnp.float32(1.0) / denom
        )
        e_pair = jnp.where(mask, e_pair, 0.0)
        # Apply exclusion scales (1-2/1-3 zeroed, 1-4 scaled)
        if excl_scales_vdw is not None:
            e_pair = e_pair * excl_scales_vdw
        return 0.5 * jnp.sum(e_pair)

    def _fwd(r, sigmas, epsilons):
        energy = lj_energy_cvjp(r, sigmas, epsilons)
        # Save minimal residuals for backward (recompute intermediates)
        return energy, (r, sigmas, epsilons)

    def _bwd(res, g):
        r, sigmas, epsilons = res
        dr, dist, sigma_ij, epsilon_ij, lam, denom, mask = \
            _compute_intermediates(r, sigmas, epsilons)

        # Analytical derivative: de_pair/d(dist)
        sigma_ij6 = jnp.maximum(sigma_ij ** 6, jnp.float32(1e-48))
        ddenom_ddist = jnp.float32(6.0) * dist ** 5 / sigma_ij6
        de_ddist = jnp.float32(4.0) * epsilon_ij * lam * (
            jnp.float32(1.0) / (denom ** 2)
            - jnp.float32(2.0) / (denom ** 3)
        ) * ddenom_ddist

        de_ddist = jnp.where(mask, de_ddist, 0.0)
        # Apply exclusion scales to gradient too
        if excl_scales_vdw is not None:
            de_ddist = de_ddist * excl_scales_vdw

        # Gradient: de/d(r_i) = de/d(dist) · (r_i - r_j) / dist
        unit_dr = dr / (dist[..., None] + 1e-12)  # (N, K, 3)
        pair_grad = de_ddist[..., None] * unit_dr   # (N, K, 3)
        grad_r = jnp.sum(pair_grad, axis=1)         # (N, 3)

        return (g * grad_r, jnp.zeros_like(sigmas), jnp.zeros_like(epsilons))

    lj_energy_cvjp.defvjp(_fwd, _bwd)
    return lj_energy_cvjp


def single_padded_energy_nl_cvjp(
    sys: PaddedSystem,
    neighbor_idx: 'Array',
    displacement_fn: space.DisplacementFn,
    implicit_solvent: bool = True,
    soft_core_lambda: 'Array' = jnp.array(1.0),
) -> 'Array':
    """Like single_padded_energy_nl but with custom VJP for LJ + checkpoint for GB.

    This variant:
    1. Uses analytical LJ forces via custom_vjp (no scatter-add)
    2. Wraps GB in jax.checkpoint (recompute forward during backward)

    Together this minimizes both compute and memory in the gradient pass.
    """
    r = sys.positions

    # Bonded terms (unchanged — already O(N))
    e_bond = _bond_energy_masked(r, sys.bonds, sys.bond_params, sys.bond_mask, displacement_fn)
    e_angle = _angle_energy_masked(r, sys.angles, sys.angle_params, sys.angle_mask, displacement_fn)
    e_dih = _dihedral_energy_masked(r, sys.dihedrals, sys.dihedral_params, sys.dihedral_mask, displacement_fn)
    e_imp = _dihedral_energy_masked(r, sys.impropers, sys.improper_params, sys.improper_mask, displacement_fn)
    e_cmap = _cmap_energy_masked(r, sys.cmap_torsions, sys.cmap_mask, sys.cmap_coeffs, displacement_fn)

    # Compute (N, K) exclusion scales for neighbor list
    from prolix.physics.neighbor_list import get_neighbor_exclusion_scales
    excl_scales_vdw_nl, _excl_scales_elec_nl = get_neighbor_exclusion_scales(
        sys.excl_indices, sys.excl_scales_vdw, sys.excl_scales_elec, neighbor_idx,
    )

    # LJ with custom VJP (analytical forces, no scatter-add)
    lj_fn = _make_lj_energy_nl_cvjp(
        neighbor_idx, soft_core_lambda, excl_scales_vdw=excl_scales_vdw_nl,
    )
    e_lj = lj_fn(r, sys.sigmas, sys.epsilons)

    # Unconditionally compute dense Coulomb interaction (vacuum electrostatic)
    N = len(sys.atom_mask)
    excl_scale_elec = _build_dense_exclusion_scales(
        sys.excl_indices, sys.excl_scales_elec, N,
    )
    e_elec = _coulomb_energy_masked(
        r, sys.charges, sys.atom_mask, displacement_fn,
        excl_scale_elec=excl_scale_elec,
    )

    if implicit_solvent:
        from prolix.physics.generalized_born import (
            compute_gb_energy_neighbor_list,
            compute_ace_nonpolar_energy,
        )
        # Checkpoint GB to save memory (recompute forward during backward)
        @jax.checkpoint
        def _gb_energy(positions, charges, radii):
            e_gb, born_radii = compute_gb_energy_neighbor_list(
                positions=positions,
                charges=charges,
                radii=radii,
                neighbor_idx=neighbor_idx,
                dielectric_offset=0.09,
            )
            e_np = compute_ace_nonpolar_energy(radii, born_radii)
            e_np = jnp.sum(e_np * sys.atom_mask)
            return e_gb + e_np

        e_solv = _gb_energy(r, sys.charges, sys.radii)
    else:
        e_solv = 0.0

    return e_bond + e_angle + e_dih + e_imp + e_cmap + e_lj + e_elec + e_solv



def single_padded_energy_nl(
    sys: PaddedSystem,
    neighbor_idx: 'Array',
    displacement_fn: space.DisplacementFn,
    implicit_solvent: bool = True,
    soft_core_lambda: 'Array' = jnp.array(1.0),
) -> 'Array':
    """Computes total potential energy using neighbor lists for non-bonded terms.

    O(N*K) scaling for LJ and GB instead of O(N^2).
    Bonded terms remain identical to single_padded_energy.

    Args:
        sys: Padded protein system.
        neighbor_idx: Neighbor indices (N, K). Padding sentinel = N.
        displacement_fn: JAX-MD displacement function.
        implicit_solvent: Whether to use GB/SA implicit solvent.
        soft_core_lambda: Soft-core LJ coupling (1.0 = standard).

    Returns:
        Total potential energy (scalar).
    """
    r = sys.positions

    # Bonded terms (unchanged — already O(N))
    e_bond = _bond_energy_masked(r, sys.bonds, sys.bond_params, sys.bond_mask, displacement_fn)
    e_angle = _angle_energy_masked(r, sys.angles, sys.angle_params, sys.angle_mask, displacement_fn)
    e_dih = _dihedral_energy_masked(r, sys.dihedrals, sys.dihedral_params, sys.dihedral_mask, displacement_fn)
    e_imp = _dihedral_energy_masked(r, sys.impropers, sys.improper_params, sys.improper_mask, displacement_fn)
    e_cmap = _cmap_energy_masked(r, sys.cmap_torsions, sys.cmap_mask, sys.cmap_coeffs, displacement_fn)

    # Non-bonded: O(N*K) via neighbor list
    from prolix.physics.neighbor_list import get_neighbor_exclusion_scales
    excl_scales_vdw_nl, _excl_scales_elec_nl = get_neighbor_exclusion_scales(
        sys.excl_indices, sys.excl_scales_vdw, sys.excl_scales_elec, neighbor_idx,
    )

    e_lj = _lj_energy_neighbor_list(
        r, sys.sigmas, sys.epsilons, neighbor_idx,
        soft_core_lambda=soft_core_lambda,
        excl_scales_vdw=excl_scales_vdw_nl,
    )

    # Unconditionally compute dense Coulomb interaction (vacuum electrostatic)
    N = len(sys.atom_mask)
    excl_scale_elec = _build_dense_exclusion_scales(
        sys.excl_indices, sys.excl_scales_elec, N,
    )
    e_elec = _coulomb_energy_masked(
        r, sys.charges, sys.atom_mask, displacement_fn,
        excl_scale_elec=excl_scale_elec,
    )

    if implicit_solvent:
        from prolix.physics.generalized_born import (
            compute_gb_energy_neighbor_list,
            compute_ace_nonpolar_energy,
        )
        e_gb, born_radii = compute_gb_energy_neighbor_list(
            positions=r,
            charges=sys.charges,
            radii=sys.radii,
            neighbor_idx=neighbor_idx,
            dielectric_offset=0.09,
        )
        e_np = compute_ace_nonpolar_energy(sys.radii, born_radii)
        e_np = jnp.sum(e_np * sys.atom_mask)
        e_solv = e_gb + e_np
    else:
        e_solv = 0.0

    return e_bond + e_angle + e_dih + e_imp + e_cmap + e_lj + e_elec + e_solv


def make_batched_energy_fn_nl(
    displacement_fn: space.DisplacementFn,
    implicit_solvent: bool = True,
) -> Callable:
    """Create a vmap-compatible energy function using neighbor lists.

    Unlike make_batched_energy_fn, this takes neighbor_idx as a separate
    argument (shape [B, N, K]) since it's updated dynamically during simulation.

    Returns:
        Function(sys_batch: PaddedSystem, neighbor_idx: Array) -> Array[B]
    """
    def _energy_single(sys, nbr_idx):
        return single_padded_energy_nl(sys, nbr_idx, displacement_fn, implicit_solvent)

    return jax.vmap(_energy_single)
