"""Hydrogen Mass Repartitioning (HMR).

Transfers mass from heavy atoms to their bonded hydrogens, enabling
stable integration at 4 fs timesteps (vs 2 fs standard). The total
system mass is conserved.

Standard HMR doubles or triples the hydrogen mass (from ~1 amu to
~3-4 amu), which slows the fastest vibrational modes (C-H, N-H, O-H
bonds) below the integration frequency.

This is an OPTIONAL optimization enabled via configuration.
Uses AMBER-style reduced units (mass in amu).

References:
- Hopkins, Le Grand, Walker, Simmerling. JCTC 11(4), 1864-1874, 2015.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


# ===========================================================================
# Configuration
# ===========================================================================

class HMRConfig(NamedTuple):
    """HMR configuration.

    Attributes:
        enabled: Whether HMR is active.
        target_h_mass: Target hydrogen mass in amu. Default 3.024
            (3× standard hydrogen) which supports 4 fs timestep.
            Some codes use 4.032 for even larger timesteps.
        min_heavy_mass: Minimum heavy atom mass after repartitioning.
            Prevents instability from over-depleting heavy atoms.
    """
    enabled: bool = False
    target_h_mass: float = 3.024
    min_heavy_mass: float = 1.5


# Standard atomic masses (amu)
STANDARD_MASSES = {
    'H': 1.008,
    'C': 12.011,
    'N': 14.007,
    'O': 15.999,
    'S': 32.065,
    'P': 30.974,
    'F': 18.998,
    'Cl': 35.453,
    'Fe': 55.845,
    'Zn': 65.38,
    'Ca': 40.078,
    'Mg': 24.305,
    'Na': 22.990,
    'K': 39.098,
    'Se': 78.971,
}


def is_hydrogen(
    elements: list[str] | None = None,
    masses: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Identify hydrogen atoms from elements or masses.

    Args:
        elements: List of element symbols. Takes priority if provided.
        masses: (N,) array of atomic masses. Hydrogen if mass < 1.1 amu.

    Returns:
        (N,) boolean array: True for hydrogens.
    """
    if elements is not None:
        return jnp.array([e.strip().upper() in ('H', 'D') for e in elements])
    elif masses is not None:
        return masses < 1.1
    else:
        raise ValueError("Must provide either elements or masses")


def repartition_masses(
    masses: jnp.ndarray,         # (N,) original masses
    bond_pairs: jnp.ndarray,     # (B, 2) bonded atom index pairs
    h_mask: jnp.ndarray,         # (N,) True for hydrogen atoms
    atom_mask: jnp.ndarray,      # (N,) True for real atoms
    config: HMRConfig | None = None,
) -> jnp.ndarray:
    """Apply hydrogen mass repartitioning.

    For each H-heavy bond:
    1. Compute mass_transfer = target_h_mass - m_H
    2. Add mass_transfer to H, subtract from heavy atom
    3. Skip if heavy atom would go below min_heavy_mass

    Total system mass is conserved.

    Args:
        masses: (N,) original atomic masses in amu
        bond_pairs: (B, 2) pairs of bonded atom indices
        h_mask: (N,) boolean mask identifying hydrogen atoms
        atom_mask: (N,) boolean mask for real atoms
        config: HMR configuration. None = default.

    Returns:
        (N,) repartitioned masses.

    Raises:
        ValueError: If total mass changes (conservation check).
    """
    if config is None:
        config = HMRConfig()

    if not config.enabled:
        return masses

    target_h = config.target_h_mass
    min_heavy = config.min_heavy_mass

    new_masses = jnp.array(masses, dtype=jnp.float32).copy()

    # Process each bond
    for b in range(bond_pairs.shape[0]):
        i, j = int(bond_pairs[b, 0]), int(bond_pairs[b, 1])

        # Skip bonds involving ghost atoms
        if not bool(atom_mask[i]) or not bool(atom_mask[j]):
            continue

        # Identify H-heavy pairs
        i_is_h = bool(h_mask[i])
        j_is_h = bool(h_mask[j])

        if i_is_h and not j_is_h:
            h_idx, heavy_idx = i, j
        elif j_is_h and not i_is_h:
            h_idx, heavy_idx = j, i
        elif i_is_h and j_is_h:
            # H-H bond (e.g., in H2) — skip
            continue
        else:
            # Heavy-heavy bond — skip
            continue

        # Compute mass transfer
        current_h_mass = float(new_masses[h_idx])
        current_heavy_mass = float(new_masses[heavy_idx])
        transfer = target_h - current_h_mass

        if transfer <= 0:
            continue  # Already at or above target

        # Check heavy atom can afford the transfer
        if current_heavy_mass - transfer < min_heavy:
            # Transfer only what we can without going below min
            transfer = max(0.0, current_heavy_mass - min_heavy)
            if transfer <= 0:
                continue

        # Apply transfer
        new_masses = new_masses.at[h_idx].set(current_h_mass + transfer)
        new_masses = new_masses.at[heavy_idx].set(current_heavy_mass - transfer)

    # Conservation check
    original_total = jnp.sum(masses * atom_mask.astype(jnp.float32))
    new_total = jnp.sum(new_masses * atom_mask.astype(jnp.float32))
    if abs(float(original_total - new_total)) > 1e-3:
        raise ValueError(
            f"Mass conservation violated: {float(original_total)} → {float(new_total)}"
        )

    return new_masses


def compute_hmr_timestep(config: HMRConfig | None = None) -> float:
    """Compute recommended timestep based on HMR configuration.

    Args:
        config: HMR configuration. None = default (disabled).

    Returns:
        Recommended timestep in ps.
    """
    if config is None or not config.enabled:
        return 0.002  # 2 fs standard

    if config.target_h_mass >= 3.0:
        return 0.004  # 4 fs with 3× hydrogen mass
    elif config.target_h_mass >= 2.0:
        return 0.003  # 3 fs with 2× hydrogen mass
    else:
        return 0.002  # Conservative


def report_hmr(
    original_masses: jnp.ndarray,
    new_masses: jnp.ndarray,
    h_mask: jnp.ndarray,
    atom_mask: jnp.ndarray,
) -> dict:
    """Generate HMR diagnostic report.

    Args:
        original_masses: (N,) original masses
        new_masses: (N,) repartitioned masses
        h_mask: (N,) hydrogen mask
        atom_mask: (N,) real atom mask

    Returns:
        Dictionary with diagnostic information.
    """
    real = atom_mask.astype(jnp.float32)
    h_real = (h_mask & atom_mask).astype(jnp.float32)
    heavy_real = (~h_mask & atom_mask).astype(jnp.float32)

    return {
        'n_atoms': int(jnp.sum(real)),
        'n_hydrogen': int(jnp.sum(h_real)),
        'n_heavy': int(jnp.sum(heavy_real)),
        'total_mass_original': float(jnp.sum(original_masses * real)),
        'total_mass_new': float(jnp.sum(new_masses * real)),
        'avg_h_mass_original': float(
            jnp.sum(original_masses * h_real) / jnp.maximum(jnp.sum(h_real), 1)
        ),
        'avg_h_mass_new': float(
            jnp.sum(new_masses * h_real) / jnp.maximum(jnp.sum(h_real), 1)
        ),
        'min_heavy_mass': float(
            jnp.min(jnp.where(heavy_real > 0, new_masses, jnp.float32(1e6)))
        ),
        'mass_conserved': bool(
            abs(float(jnp.sum(original_masses * real) - jnp.sum(new_masses * real))) < 1e-3
        ),
    }
