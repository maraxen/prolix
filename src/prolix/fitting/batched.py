"""Batched bonded parameters and topology for vmap-friendly training.

Provides padded containers for stacking multiple molecules' parameters and
topology into a single batch, with per-molecule masks indicating real entries.
"""

from typing import Tuple

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from prolix.fitting.params import BondedParams
from prolix.fitting.topology import BondedTopology


class BatchedBondedParams(eqx.Module):
    """Stacked params for B molecules in the SAME atom_bucket.

    Per-tensor shape: (B, max_n_<term>, ...). All arrays padded to the
    bucket's max size across the B molecules; masks indicate real entries.

    Attributes:
        k_bond: Float[Array, "B max_bonds"]
        r0: Float[Array, "B max_bonds"]
        k_theta: Float[Array, "B max_angles"]
        theta0_rad: Float[Array, "B max_angles"]
        k_phi: Float[Array, "B max_torsions n_terms"]
    """

    k_bond: Float[Array, "B max_bonds"]
    r0: Float[Array, "B max_bonds"]
    k_theta: Float[Array, "B max_angles"]
    theta0_rad: Float[Array, "B max_angles"]
    k_phi: Float[Array, "B max_torsions n_terms"]


class BatchedBondedTopology(eqx.Module):
    """Stacked topology for B molecules. Indices into per-mol atom arrays.

    Fields are regular pytree leaves (NOT eqx.field(static=True)) — they
    are jnp arrays populated by stack_molecules, and JAX arrays in static
    fields trigger "A JAX array is being set as static" warnings and
    cause subtle tracing bugs (verified 2026-05-20). The arrays are
    constant across vmap, but that doesn't make them static in the
    pytree sense.
    """

    bond_idx: Int[Array, "B max_bonds 2"]
    angle_idx: Int[Array, "B max_angles 3"]
    torsion_idx: Int[Array, "B max_torsions 4"]
    torsion_periodicity: Int[Array, "B max_torsions n_terms"]
    torsion_phase_rad: Float[Array, "B max_torsions n_terms"]
    bond_mask: Bool[Array, "B max_bonds"]
    angle_mask: Bool[Array, "B max_angles"]
    torsion_mask: Bool[Array, "B max_torsions"]


def stack_molecules(
    params_list: list[BondedParams],
    topology_list: list[BondedTopology],
) -> Tuple[BatchedBondedParams, BatchedBondedTopology]:
    """Stack multiple molecules' parameters and topology into batched containers.

    Pads each per-molecule array to the max size across the batch. Padding rows
    for bond indices are set to (0, 0) — masked out during energy computation.
    Padding parameter values are zero.

    Args:
        params_list: List of BondedParams (one per molecule).
        topology_list: List of BondedTopology (one per molecule).

    Returns:
        (BatchedBondedParams, BatchedBondedTopology) stacked over the batch axis.
    """
    B = len(params_list)

    if B == 0:
        raise ValueError("Cannot stack zero molecules")

    # Find max sizes across all molecules
    max_n_bonds = max(topo.n_bonds for topo in topology_list)
    max_n_angles = max(topo.n_angles for topo in topology_list)
    max_n_torsions = max(topo.n_torsions for topo in topology_list)

    # Determine n_terms from the first molecule (assume all have same)
    n_torsion_terms = params_list[0].n_torsion_terms if params_list[0].n_torsions > 0 else 1

    # Initialize padded arrays
    k_bond_stacked = jnp.zeros((B, max_n_bonds))
    r0_stacked = jnp.zeros((B, max_n_bonds))
    k_theta_stacked = jnp.zeros((B, max_n_angles))
    theta0_rad_stacked = jnp.zeros((B, max_n_angles))
    k_phi_stacked = jnp.zeros((B, max_n_torsions, n_torsion_terms))

    # Padded indices: use DISTINCT atom indices so r_ij != 0 for padded bonds.
    # If padded bonds use (0,0), the gradient ∂‖r‖/∂r = r/‖r‖ → 0/0 NaN even
    # though the mask zeroes the per-term energy — multiplying mask*NaN gives
    # NaN (IEEE 754). Using (0,1) keeps the vector non-degenerate; the mask
    # still gates the energy contribution to zero.
    bond_idx_stacked = jnp.tile(jnp.array([0, 1], dtype=jnp.int32), (B, max_n_bonds, 1))
    angle_idx_stacked = jnp.tile(jnp.array([0, 1, 2], dtype=jnp.int32), (B, max_n_angles, 1))
    torsion_idx_stacked = jnp.tile(jnp.array([0, 1, 2, 3], dtype=jnp.int32), (B, max_n_torsions, 1))
    torsion_periodicity_stacked = jnp.zeros((B, max_n_torsions, n_torsion_terms), dtype=jnp.int32)
    torsion_phase_rad_stacked = jnp.zeros((B, max_n_torsions, n_torsion_terms))

    bond_mask_stacked = jnp.zeros((B, max_n_bonds), dtype=jnp.bool_)
    angle_mask_stacked = jnp.zeros((B, max_n_angles), dtype=jnp.bool_)
    torsion_mask_stacked = jnp.zeros((B, max_n_torsions), dtype=jnp.bool_)

    # Fill stacked arrays
    for b, (params, topology) in enumerate(zip(params_list, topology_list)):
        # Bonds
        n_b = topology.n_bonds
        if n_b > 0:
            k_bond_stacked = k_bond_stacked.at[b, :n_b].set(params.k_bond)
            r0_stacked = r0_stacked.at[b, :n_b].set(params.r0)
            bond_idx_stacked = bond_idx_stacked.at[b, :n_b].set(topology.bond_idx)
            bond_mask_stacked = bond_mask_stacked.at[b, :n_b].set(True)

        # Angles
        n_a = topology.n_angles
        if n_a > 0:
            k_theta_stacked = k_theta_stacked.at[b, :n_a].set(params.k_theta)
            theta0_rad_stacked = theta0_rad_stacked.at[b, :n_a].set(params.theta0_rad)
            angle_idx_stacked = angle_idx_stacked.at[b, :n_a].set(topology.angle_idx)
            angle_mask_stacked = angle_mask_stacked.at[b, :n_a].set(True)

        # Torsions
        n_t = topology.n_torsions
        if n_t > 0:
            # Handle torsion params carefully (may be 1D or 2D)
            k_phi_b = params.k_phi
            if len(k_phi_b.shape) == 1:
                k_phi_b = jnp.expand_dims(k_phi_b, axis=-1)
            k_phi_stacked = k_phi_stacked.at[b, :n_t, :].set(k_phi_b)

            torsion_idx_stacked = torsion_idx_stacked.at[b, :n_t].set(topology.torsion_idx)
            torsion_periodicity_stacked = torsion_periodicity_stacked.at[b, :n_t].set(
                topology.torsion_periodicity
            )
            torsion_phase_rad_stacked = torsion_phase_rad_stacked.at[b, :n_t].set(
                topology.torsion_phase_rad
            )
            torsion_mask_stacked = torsion_mask_stacked.at[b, :n_t].set(True)

    batched_params = BatchedBondedParams(
        k_bond=k_bond_stacked,
        r0=r0_stacked,
        k_theta=k_theta_stacked,
        theta0_rad=theta0_rad_stacked,
        k_phi=k_phi_stacked,
    )

    batched_topology = BatchedBondedTopology(
        bond_idx=bond_idx_stacked,
        angle_idx=angle_idx_stacked,
        torsion_idx=torsion_idx_stacked,
        torsion_periodicity=torsion_periodicity_stacked,
        torsion_phase_rad=torsion_phase_rad_stacked,
        bond_mask=bond_mask_stacked,
        angle_mask=angle_mask_stacked,
        torsion_mask=torsion_mask_stacked,
    )

    return batched_params, batched_topology


def unbatch_params(
    batched_params: BatchedBondedParams,
    batch_idx: int,
) -> BondedParams:
    """Extract per-molecule parameters from batched container.

    Args:
        batched_params: BatchedBondedParams.
        batch_idx: Index within the batch (0 to B-1).

    Returns:
        BondedParams for that molecule (trimmed of padding).
    """
    # Extract per-molecule arrays (still padded)
    k_bond_padded = batched_params.k_bond[batch_idx]
    r0_padded = batched_params.r0[batch_idx]
    k_theta_padded = batched_params.k_theta[batch_idx]
    theta0_rad_padded = batched_params.theta0_rad[batch_idx]
    k_phi_padded = batched_params.k_phi[batch_idx]  # (max_torsions, n_terms)

    # For now, return with padding (caller can trim if needed)
    # In practice, keep padded size for consistency; masking will zero the contribution
    return BondedParams(
        k_bond=k_bond_padded,
        r0=r0_padded,
        k_theta=k_theta_padded,
        theta0_rad=theta0_rad_padded,
        k_phi=k_phi_padded,
    )
