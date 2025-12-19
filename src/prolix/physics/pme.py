"""Particle Mesh Ewald (PME) electrostatics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax_md import energy, partition, space, util

if TYPE_CHECKING:
    from collections.abc import Callable

Array = util.Array


def make_pme_energy_fn(
    charges: Array,
    box: Array,
    grid_points: int | Array = 64,
    alpha: float = 0.34,
) -> Callable[[Array], Array]:
    """Creates a PME energy function (reciprocal space).

    Args:
        charges: (N,) atomic charges
        box: (3,) or (3,3) simulation box
        grid_points: Number of grid points for FFT (int or (3,))
        alpha: Ewald splitting parameter

    Returns:
        Energy function E(R) -> float

    """
    # jax_md.energy.coulomb_recip_pme expects grid_points as tuple if not int
    grid_dim = grid_points if isinstance(grid_points, int) else tuple(grid_points)

    # jax_md.energy.coulomb_recip_pme expects box as (3,3) matrix for det() if not scalar
    # If box is (3,), convert to diagonal matrix
    box_matrix = jnp.diag(box) if box.ndim == 1 and box.shape[0] == 3 else box

    pme_fn = energy.coulomb_recip_pme(
        charges, box_matrix, grid_dim, alpha=alpha
    )

    def energy_fn(r: Array, **kwargs) -> Array:
        # pme_fn returns total energy
        return pme_fn(r, **kwargs)

    return energy_fn


def make_direct_coulomb_energy_fn(
    displacement_fn: space.DisplacementFn,
    charges: Array,
    cutoff: float,
    neighbor_list: partition.NeighborList,
    exclusion_mask: Array | None = None,
    alpha: float = 0.34,
) -> Callable[[Array], Array]:
    """Creates a direct-space Coulomb energy function with cutoff.

    E_direct = sum_ij [ q_i*q_j * erfc(alpha*r_ij) / r_ij ]

    This needs to be implemented manually or using a custom neighbor list function
    since jax_md doesn't have a canned "damped coulomb with neighbor list" function
    that exactly matches PME's direct part easily exposed, although `energy.erfc` exists.

    For now, we can use `energy.coulomb_neighbor_list` tailored with a custom interaction fn.
    """

    # Custom pair interaction for Ewald direct space
    def ewald_direct_pair(r, q_i, q_j):
        # E = q_i * q_j * erfc(alpha * r) / r
        # Avoid singularity at r=0
        r_safe = r + 1e-6
        return q_i * q_j * jax.scipy.special.erfc(alpha * r) / r_safe

    # Helper to compute energy over neighbor list
    def energy_fn(r: Array, neighbor: partition.NeighborList | None = None, **kwargs) -> Array:
        if neighbor is None:
            # Fallback to dense if no neighbor list (not recommended for PME)
            # Or assume neighbor is passed via closure if baked in (but neighbor_list arg is dynamic usually)
             neighbor = neighbor_list

        idx = neighbor.idx

        # Calculate pairwise displacements/distances via neighbor list logic
        # Easier to reuse jax_md's generic neighbor list energy capacity?
        # Not straightforward without a predefined energy function.

        # Let's implement manually using vectorization over neighbor list
        # (N, MaxNeighbors)

        d = space.map_neighbor(displacement_fn)(r, r[idx])
        dist = space.distance(d)

        # Gather charges
        q_i = charges[:, None]
        q_j = charges[idx]

        # Calculate energy terms
        e_pair = ewald_direct_pair(dist, q_i, q_j)

        # Mask out-of-range neighbors (idx == N)
        mask = idx < r.shape[0]

        # Apply exclusion mask if provided
        # exclusion_mask usually (N, N) dense. For neighbor list, we need to gather it?
        # Or if exclusion_mask is None, we assume all pairs in neighbor list are valid?
        # Wait, neighbor list includes exclusions usually unless masked out.
        # But standard neighbor_list includes everything within cutoff.
        # We need to mask 1-2, 1-3, etc.

        # If exclusion_mask is provided as (N, N), we can gather:
        if exclusion_mask is not None:
             # exclusion_mask[i, j]
             # We need mask[i, idx[i, k]]
             # This is efficient enough?
             # i indices: (N, 1) broadcasted
             i_idx = jnp.arange(r.shape[0])[:, None]
             # Safe idx to avoid OOB
             safe_idx = jnp.minimum(idx, r.shape[0]-1)

             excl = exclusion_mask[i_idx, safe_idx]
             # exclusion_mask is 1 if allowed, 0 if excluded?
             # Based on system.py: "interaction_allowed = exclusion_mask..."
             # So 1 means Allow, 0 means Exclude.
             e_pair = e_pair * excl

        # Cutoff check (redundant if neighbor list is strict, but good for erfc smoothing)
        # Usually direct space is just simple cutoff.

        # Apply neighbor mask
        e_pair = jnp.where(mask, e_pair, 0.0)

        return 0.5 * jnp.sum(e_pair)

    return energy_fn
