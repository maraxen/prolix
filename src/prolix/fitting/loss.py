"""Differentiable bonded-parameter fitting loss.

Implements the loss formulation from spec §6:
  L_m = (1/N_atoms_m) * mean[(F_pred - F_ref)^2]_atoms,conf
        + α * (1/N_conf_m) * mean[(E_pred - E_ref - shift_m)^2]_conf
        + w_reg * sum[(θ - θ_init)^2 / σ_θ^2]_params

Gradient flows through both positions (for forces) and parameters (for fitting).
"""

from typing import Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from prolix.fitting.energy import bonded_energy
from prolix.fitting.params import BondedParams
from prolix.fitting.topology import BondedTopology


# Unit conversion
HA_TO_KCAL_PER_MOL = 627.5094740631


def default_sigma(params_init: BondedParams) -> BondedParams:
    """Default per-parameter regularization widths (spec §6, non-contractual).

    Args:
        params_init: Reference BondedParams (shapes only; values ignored).

    Returns:
        BondedParams with constant sigma values broadcast to each array shape.
    """
    return BondedParams(
        k_bond=jnp.full_like(params_init.k_bond, 100.0),  # kcal/mol/Å²
        r0=jnp.full_like(params_init.r0, 0.05),  # Å
        k_theta=jnp.full_like(params_init.k_theta, 30.0),  # kcal/mol/rad²
        theta0_rad=jnp.full_like(params_init.theta0_rad, 5.0 * jnp.pi / 180.0),  # 5° in rad
        k_phi=jnp.full_like(params_init.k_phi, 1.0),  # kcal/mol
    )


def bonded_loss(
    positions_per_conf: Float[Array, "N_conf N_atoms 3"],
    forces_ref: Float[Array, "N_conf N_atoms 3"],
    energies_ref: Float[Array, "N_conf"],
    params: BondedParams,
    params_init: BondedParams,
    topology: BondedTopology,
    *,
    alpha: float = 0.25,
    w_reg: float = 0.01,
    sigma: Optional[BondedParams] = None,
    bond_mask: Optional[Array] = None,
    angle_mask: Optional[Array] = None,
    torsion_mask: Optional[Array] = None,
) -> Float[Array, ""]:
    """Per-molecule bonded loss with energy + force + regularization.

    Args:
        positions_per_conf: (N_conf, N_atoms, 3) atom positions in Å.
        forces_ref: (N_conf, N_atoms, 3) reference forces in kcal/mol/Å (already unit-converted).
        energies_ref: (N_conf,) reference energies in Hartree (NOT yet converted).
        params: BondedParams to optimize.
        params_init: BondedParams θ_init (frozen reference for prior).
        topology: BondedTopology (static atom indices, periodicity).
        alpha: Weight for energy loss (default 0.25 from spec).
        w_reg: Weight for harmonic prior (default 0.01 from spec).
        sigma: BondedParams with per-parameter regularization widths.
            If None, uses default_sigma(params_init).

    Returns:
        Scalar loss value in kcal/mol (roughly, mixed units).
    """
    if sigma is None:
        sigma = default_sigma(params_init)

    n_conf = positions_per_conf.shape[0]
    n_atoms = positions_per_conf.shape[1]

    # ===== COMPUTE PREDICTED FORCES AND ENERGIES =====
    # Define loss_fn that computes energy (so we can take grad w.r.t. positions)
    def energy_fn(positions_single):
        """Energy for a single conformer (will be vmapped)."""
        return bonded_energy(
            positions_single, params, topology,
            bond_mask=bond_mask,
            angle_mask=angle_mask,
            torsion_mask=torsion_mask,
        )

    # Compute forces via gradient: F = -dE/dr
    forces_fn = jax.grad(energy_fn)
    forces_pred_per_conf = jax.vmap(forces_fn)(positions_per_conf)  # (N_conf, N_atoms, 3)

    # Compute energies (direct evaluation, vmapped)
    energies_pred_per_conf = jax.vmap(energy_fn)(positions_per_conf)  # (N_conf,)

    # ===== ZERO-MEAN ENERGY SHIFT (per-molecule espaloma precedent) =====
    # Both predicted and reference energies are shifted to zero mean
    energies_pred_mean = jnp.mean(energies_pred_per_conf)
    energies_ref_mean = jnp.mean(energies_ref)

    energies_pred_shifted = energies_pred_per_conf - energies_pred_mean
    energies_ref_converted = energies_ref * HA_TO_KCAL_PER_MOL
    energies_ref_shifted = energies_ref_converted - energies_ref_mean * HA_TO_KCAL_PER_MOL

    # ===== FORCE LOSS =====
    # Normalize per atom (not per molecule)
    force_diff = forces_pred_per_conf - forces_ref  # (N_conf, N_atoms, 3)
    force_mse_per_atom = jnp.mean(force_diff**2)  # scalar, averaged over all atoms and confs
    force_loss = force_mse_per_atom / n_atoms

    # ===== ENERGY LOSS =====
    # Normalize per conformer
    energy_diff = energies_pred_shifted - energies_ref_shifted  # (N_conf,)
    energy_mse = jnp.mean(energy_diff**2)
    energy_loss = energy_mse / n_conf

    # ===== REGULARIZATION (harmonic prior) =====
    # sum[(p - p_init)^2 / sigma^2] across all parameters
    # When masks are provided, zero out padded-slot contributions so the prior
    # doesn't pull padded-zero params toward zero (which they already are).
    def _maybe_mask(arr, mask):
        return arr if mask is None else arr * mask

    reg_bond = jnp.sum(_maybe_mask(
        ((params.k_bond - params_init.k_bond) / sigma.k_bond) ** 2, bond_mask))
    reg_r0 = jnp.sum(_maybe_mask(
        ((params.r0 - params_init.r0) / sigma.r0) ** 2, bond_mask))
    reg_theta = jnp.sum(_maybe_mask(
        ((params.k_theta - params_init.k_theta) / sigma.k_theta) ** 2, angle_mask))
    reg_theta0 = jnp.sum(_maybe_mask(
        ((params.theta0_rad - params_init.theta0_rad) / sigma.theta0_rad) ** 2, angle_mask))
    # k_phi has an extra trailing dim (n_terms); broadcast mask if provided
    if torsion_mask is not None:
        reg_phi_terms = ((params.k_phi - params_init.k_phi) / sigma.k_phi) ** 2
        reg_phi = jnp.sum(reg_phi_terms * jnp.expand_dims(torsion_mask, axis=-1))
    else:
        reg_phi = jnp.sum(((params.k_phi - params_init.k_phi) / sigma.k_phi) ** 2)

    reg_loss = w_reg * (reg_bond + reg_r0 + reg_theta + reg_theta0 + reg_phi)

    # ===== TOTAL LOSS =====
    total_loss = force_loss + alpha * energy_loss + reg_loss

    return total_loss
