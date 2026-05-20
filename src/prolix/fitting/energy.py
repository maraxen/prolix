"""Differentiable bonded energy computation for force-field fitting.

Computes CHARMM/AMBER-style bonded energy (bonds, angles, torsions).
JAX-pure and vmap-friendly. Gradients w.r.t. positions give forces;
gradients w.r.t. params give parameter gradient for optimization.
"""

from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from prolix.fitting.params import BondedParams
from prolix.fitting.topology import BondedTopology


def bonded_energy(
    positions: Float[Array, "N_atoms 3"],
    params: BondedParams,
    topology: BondedTopology,
    *,
    bond_mask: Optional[Float[Array, "N_bonds"]] = None,
    angle_mask: Optional[Float[Array, "N_angles"]] = None,
    torsion_mask: Optional[Float[Array, "N_torsions"]] = None,
) -> Float[Array, ""]:
    """Compute total bonded energy with optional per-term masking.

    Energy decomposition:
        E_bond = sum_b mask_b * k_b * (r_b - r0_b)^2
        E_angle = sum_a mask_a * k_theta_a * (theta_a - theta0_a)^2
        E_torsion = sum_t mask_t * sum_term k_phi_term * (1 + cos(periodicity_term * phi_t - phase_term))
        E_total = E_bond + E_angle + E_torsion

    Args:
        positions: (N_atoms, 3) Cartesian coordinates in Å.
        params: BondedParams holding k_bond, r0, k_theta, theta0_rad, k_phi.
        topology: BondedTopology holding static atom indices and periodicity.
        bond_mask: Optional (N_bonds,) boolean mask; when provided, multiply per-bond energy by mask.
        angle_mask: Optional (N_angles,) boolean mask; when provided, multiply per-angle energy by mask.
        torsion_mask: Optional (N_torsions,) boolean mask; when provided, multiply per-torsion energy by mask.

    Returns:
        Scalar energy in kcal/mol.
    """
    energy = 0.0

    # ===== BOND ENERGY =====
    if topology.n_bonds > 0:
        energy = energy + _bond_energy(positions, params, topology, bond_mask=bond_mask)

    # ===== ANGLE ENERGY =====
    if topology.n_angles > 0:
        energy = energy + _angle_energy(positions, params, topology, angle_mask=angle_mask)

    # ===== TORSION ENERGY =====
    if topology.n_torsions > 0:
        energy = energy + _torsion_energy(positions, params, topology, torsion_mask=torsion_mask)

    return energy


def _bond_energy(
    positions: Float[Array, "N_atoms 3"],
    params: BondedParams,
    topology: BondedTopology,
    bond_mask: Optional[Float[Array, "N_bonds"]] = None,
) -> Float[Array, ""]:
    """Harmonic bond energy: sum_b mask_b * k_b * (r_b - r0_b)^2."""
    i_idx = topology.bond_idx[:, 0].astype(jnp.int32)
    j_idx = topology.bond_idx[:, 1].astype(jnp.int32)

    pos_i = positions[i_idx]  # (N_bonds, 3)
    pos_j = positions[j_idx]  # (N_bonds, 3)

    dr = pos_i - pos_j  # (N_bonds, 3)
    dist = jnp.linalg.norm(dr, axis=-1)  # (N_bonds,)

    energy_per_bond = params.k_bond * (dist - params.r0) ** 2

    # Apply mask if provided
    if bond_mask is not None:
        energy_per_bond = energy_per_bond * bond_mask

    return jnp.sum(energy_per_bond)


def _angle_energy(
    positions: Float[Array, "N_atoms 3"],
    params: BondedParams,
    topology: BondedTopology,
    angle_mask: Optional[Float[Array, "N_angles"]] = None,
) -> Float[Array, ""]:
    """Harmonic angle energy: sum_a mask_a * k_theta_a * (theta_a - theta0_a)^2."""
    i_idx = topology.angle_idx[:, 0].astype(jnp.int32)
    j_idx = topology.angle_idx[:, 1].astype(jnp.int32)
    k_idx = topology.angle_idx[:, 2].astype(jnp.int32)

    pos_i = positions[i_idx]  # (N_angles, 3)
    pos_j = positions[j_idx]  # (N_angles, 3)
    pos_k = positions[k_idx]  # (N_angles, 3)

    v_ji = pos_i - pos_j  # (N_angles, 3)
    v_jk = pos_k - pos_j  # (N_angles, 3)

    norm_ji = jnp.linalg.norm(v_ji, axis=-1, keepdims=True)  # (N_angles, 1)
    norm_jk = jnp.linalg.norm(v_jk, axis=-1, keepdims=True)  # (N_angles, 1)

    # Avoid division by zero
    denom = norm_ji * norm_jk + 1e-8

    cos_theta = jnp.sum(v_ji * v_jk, axis=-1) / denom[:, 0]

    # Clip to valid range for arccos
    cos_theta = jnp.clip(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)

    theta = jnp.arccos(cos_theta)  # (N_angles,)

    energy_per_angle = params.k_theta * (theta - params.theta0_rad) ** 2

    # Apply mask if provided
    if angle_mask is not None:
        energy_per_angle = energy_per_angle * angle_mask

    return jnp.sum(energy_per_angle)


def _torsion_energy(
    positions: Float[Array, "N_atoms 3"],
    params: BondedParams,
    topology: BondedTopology,
    torsion_mask: Optional[Float[Array, "N_torsions"]] = None,
) -> Float[Array, ""]:
    """Periodic torsion energy: sum_t mask_t * sum_term k_phi_term * (1 + cos(periodicity_term * phi_t - phase_term))."""
    i_idx = topology.torsion_idx[:, 0].astype(jnp.int32)
    j_idx = topology.torsion_idx[:, 1].astype(jnp.int32)
    k_idx = topology.torsion_idx[:, 2].astype(jnp.int32)
    l_idx = topology.torsion_idx[:, 3].astype(jnp.int32)

    pos_i = positions[i_idx]  # (N_torsions, 3)
    pos_j = positions[j_idx]
    pos_k = positions[k_idx]
    pos_l = positions[l_idx]

    # Compute dihedral angle using standard 4-atom formula
    # b0 = r_j - r_i
    # b1 = r_k - r_j
    # b2 = r_l - r_k
    b0 = pos_j - pos_i  # (N_torsions, 3)
    b1 = pos_k - pos_j
    b2 = pos_l - pos_k

    # Normalize b1 (the central bond)
    b1_norm = jnp.linalg.norm(b1, axis=-1, keepdims=True) + 1e-12
    b1_unit = b1 / b1_norm  # (N_torsions, 3)

    # Project b0 and b2 onto the plane perpendicular to b1
    # v = b0 - (b0 · b1_unit) * b1_unit
    # w = b2 - (b2 · b1_unit) * b1_unit
    dot_b0_b1 = jnp.sum(b0 * b1_unit, axis=-1, keepdims=True)  # (N_torsions, 1)
    dot_b2_b1 = jnp.sum(b2 * b1_unit, axis=-1, keepdims=True)

    v = b0 - dot_b0_b1 * b1_unit  # (N_torsions, 3)
    w = b2 - dot_b2_b1 * b1_unit

    # Compute dihedral angle: phi = atan2(w · (b1_unit × v), w · v)
    b1_cross_v = jnp.cross(b1_unit, v)  # (N_torsions, 3)

    x = jnp.sum(v * w, axis=-1)  # (N_torsions,)
    y = jnp.sum(b1_cross_v * w, axis=-1)

    # Use arctan2 for robust angle computation
    # Safe: handle (0, 0) case by substituting x=1, y=0
    safe_x = jnp.where((x == 0.0) & (y == 0.0), 1.0, x)
    safe_y = jnp.where((x == 0.0) & (y == 0.0), 0.0, y)

    phi = jnp.arctan2(safe_y, safe_x)  # (N_torsions,)

    # Standard convention: phi -= pi to shift range to [-pi, pi]
    phi = phi - jnp.pi

    # Compute energy per torsion term
    # In v0, each torsion has n_terms=1
    # E_dih = k_phi * (1 + cos(periodicity * phi - phase))
    periodicity = topology.torsion_periodicity  # (N_torsions, n_terms)
    phase = topology.torsion_phase_rad  # (N_torsions, n_terms)
    k_phi = params.k_phi  # (N_torsions, n_terms)

    # Broadcast phi to match shape (N_torsions, n_terms)
    phi_expanded = jnp.expand_dims(phi, axis=-1)  # (N_torsions, 1)

    cos_term = jnp.cos(periodicity * phi_expanded - phase)
    energy_per_term = k_phi * (1.0 + cos_term)  # (N_torsions, n_terms)

    # Apply mask if provided
    if torsion_mask is not None:
        # Expand mask to match (N_torsions, n_terms) shape
        torsion_mask_expanded = jnp.expand_dims(torsion_mask, axis=-1)
        energy_per_term = energy_per_term * torsion_mask_expanded

    return jnp.sum(energy_per_term)
