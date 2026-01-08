"""Neighbor list management and sparse exclusion handling."""
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from jax_md import partition, space, util

if TYPE_CHECKING:
    from proxide.md import SystemParams

Array = util.Array


@dataclasses.dataclass(frozen=True)
class ExclusionSpec:
    """Sparse specification of excluded interactions for neighbor lists.
    
    Instead of N×N dense matrices, we store pairs of atoms that should be
    excluded or scaled. This allows O(N×K) filtering in neighbor lists.
    """
    # 1-2 and 1-3 pairs (fully excluded)
    idx_12_13: Array  # (M1, 2) int32

    # 1-4 pairs (scaled)
    idx_14: Array     # (M2, 2) int32
    scale_14_elec: float
    scale_14_vdw: float

    # Total number of atoms (for validation/creation)
    n_atoms: int

    @classmethod
    def from_system_params(
        cls,
        system_params: SystemParams,
        coulomb14scale: float = 0.83333333,
        lj14scale: float = 0.5
    ) -> ExclusionSpec:
        """Build ExclusionSpec from standard system parameters."""
        # Use existing graph search logic from system.py (refactored here or copied)
        # For now, we'll re-implement the graph search efficiently using numpy

        charges = system_params["charges"]
        n_atoms = len(charges)
        bonds = system_params.get("bonds")

        if bonds is None or len(bonds) == 0:
            return cls(
                idx_12_13=jnp.zeros((0, 2), dtype=jnp.int32),
                idx_14=jnp.zeros((0, 2), dtype=jnp.int32),
                scale_14_elec=coulomb14scale,
                scale_14_vdw=lj14scale,
                n_atoms=n_atoms
            )

        import collections
        adj = collections.defaultdict(list)
        bonds_np = np.array(bonds)

        for b in bonds_np:
            adj[b[0]].append(b[1])
            adj[b[1]].append(b[0])

        excl_12 = set()
        excl_13 = set()
        excl_14 = set()

        # Iterate all atoms
        for i in range(n_atoms):
            # 1-2
            for j in adj[i]:
                if j > i:
                    excl_12.add((i, j))

                # 1-3
                for k in adj[j]:
                    if k == i: continue
                    if k > i:
                        excl_13.add((i, k))

                    # 1-4
                    for l in adj[k]:
                        if l == j: continue
                        if l == i: continue
                        if l > i:
                            excl_14.add((i, l))

        # Clean up overlaps
        # 1-2 takes precedence over 1-3 and 1-4
        excl_13 = excl_13 - excl_12
        excl_14 = excl_14 - excl_12 - excl_13

        # Combine 1-2 and 1-3 (both fully excluded usually)
        # Note: In some FFs 1-3 might be scaled differently, but standard AMBER is 0.0 for both.
        full_exclusions = sorted(list(excl_12 | excl_13))
        scaled_exclusions = sorted(list(excl_14))

        return cls(
            idx_12_13=jnp.array(full_exclusions, dtype=jnp.int32) if full_exclusions else jnp.zeros((0, 2), dtype=jnp.int32),
            idx_14=jnp.array(scaled_exclusions, dtype=jnp.int32) if scaled_exclusions else jnp.zeros((0, 2), dtype=jnp.int32),
            scale_14_elec=coulomb14scale,
            scale_14_vdw=lj14scale,
            n_atoms=n_atoms
        )


def make_neighbor_list_fn(
    displacement_fn: space.DisplacementFn,
    box_size: Array,
    cutoff: float,
    capacity_multiplier: float = 1.25,
    disable_cell_list: bool = False,
) -> partition.NeighborListFn:
    """Creates a neighbor list function optimized for solvent systems."""
    return partition.neighbor_list(
        displacement_fn,
        box_size,
        r_cutoff=cutoff,
        dr_threshold=0.5, # buffer distance for updates
        capacity_multiplier=capacity_multiplier,
        disable_cell_list=disable_cell_list,
        mask_self=True,
        fractional_coordinates=False
    )


def compute_exclusion_mask_neighbor_list(
    exclusion_spec: ExclusionSpec,
    neighbor_idx: Array,  # (N, K)
    n_atoms: int
) -> tuple[Array, Array, Array]:
    """Computes exclusion masks for a neighbor list.
    
    Returns:
        mask_vdw: (N, K) - 1.0 if interacting, 0.0 if excluded, scale if 1-4
        mask_elec: (N, K) - 1.0 if interacting, 0.0 if excluded, scale if 1-4
        mask_hard: (N, K) - 0.0 if excluded (1-2, 1-3), 1.0 otherwise (used for non-scaled terms)
    """
    N, K = neighbor_idx.shape

    # We need to check if pair (i, neighbor_idx[i, k]) is in exclusion lists.
    # Since K is small (~100-300) and exclusion lists are sparse, this is tricky in JAX.
    # N*K lookups into a list of M pairs is expensive if M is large.
    # But for molecular systems, M is O(N).

    # Approach:
    # We can't use Python sets in JIT.
    # We can use a dense lookup table if we can strictly limit it to bond-topology range (e.g. max separation 3 bonds).
    # But that's complicated.

    # Better Approach for JAX:
    # Encode exclusions into a dense (N, max_exclusions_per_atom) array?
    # Max exclusions is usually small (< 20 for organic molecules).
    # Then for each neighbor, check if it exists in the exclusion row.

    # Let's perform this encoding in a helper function one-time on setup (or JIT-able).

    # Wait, sparse exclusion logic in JAX is usually done via masking logic:
    # 1. Expand exclusions to dense mask? NO, that's what we want to avoid.
    # 2. Check overlap of neighbor list with exclusion list.

    # Efficient way:
    # Pre-process exclusion_spec into padded dense arrays:
    # exclusions: (N, MaxExclusions) - contains indices of excluded atoms for each atom
    # scales_vdw: (N, MaxExclusions)
    # scales_elec: (N, MaxExclusions)

    # We need to construct this from the pair format.
    # This is best done ONCE when building the Energy Function (closure capture).
    # But compute_exclusion_mask_neighbor_list implies dynamic calculation.

    # Let's assume this function is called inside the energy function, but we can helper it.

def map_exclusions_to_dense_padded(exclusion_spec: ExclusionSpec, max_exclusions: int = 32) -> tuple[Array, Array, Array]:
    """Maps pair exclusions to per-atom padded arrays for efficient lookup."""
    N = exclusion_spec.n_atoms

    # Initialize with -1 (invalid index)
    excl_indices = jnp.full((N, max_exclusions), -1, dtype=jnp.int32)
    scales_vdw = jnp.ones((N, max_exclusions), dtype=jnp.float32)
    scales_elec = jnp.ones((N, max_exclusions), dtype=jnp.float32)

    # Helper to scatter pairs into the arrays
    # Since we can't use dynamic loops easily in JAX to fill this without scan,
    # and this is setup code, we can use numpy or simple loops if not traced?
    # Actually, this should be done during `make_energy_fn`, so numpy is fine!

    # Convert JAX arrays to numpy for population if they are arrays
    idx_12_13 = np.array(exclusion_spec.idx_12_13)
    idx_14 = np.array(exclusion_spec.idx_14)

    # Use lists to collect per atom
    atom_excls = [[] for _ in range(N)]

    for i, j in idx_12_13:
        if i < N: atom_excls[i].append((int(j), 0.0, 0.0))
        if j < N: atom_excls[j].append((int(i), 0.0, 0.0))

    for i, j in idx_14:
        s_v = exclusion_spec.scale_14_vdw
        s_e = exclusion_spec.scale_14_elec
        if i < N: atom_excls[i].append((int(j), s_v, s_e))
        if j < N: atom_excls[j].append((int(i), s_v, s_e))

    # Fill arrays
    excl_indices_np = np.full((N, max_exclusions), -1, dtype=np.int32)
    scales_vdw_np = np.ones((N, max_exclusions), dtype=np.float32)
    scales_elec_np = np.ones((N, max_exclusions), dtype=np.float32)

    for i in range(N):
        entries = atom_excls[i]
        # Sort by index for determinism
        entries.sort(key=lambda x: x[0])
        n_entries = min(len(entries), max_exclusions)
        for k in range(n_entries):
            excl_indices_np[i, k] = entries[k][0]
            scales_vdw_np[i, k] = entries[k][1]
            scales_elec_np[i, k] = entries[k][2]

    return jnp.array(excl_indices_np), jnp.array(scales_vdw_np), jnp.array(scales_elec_np)

def get_neighbor_exclusion_scales(
    excl_indices: Array, # (N, MaxExcl)
    excl_scales_vdw: Array, # (N, MaxExcl)
    excl_scales_elec: Array, # (N, MaxExcl)
    neighbor_idx: Array, # (N, K)
) -> tuple[Array, Array]:
    """Computes scale factors for neighbors based on exclusion lists.
    
    Returns:
        scale_vdw: (N, K)
        scale_elec: (N, K)
    """
    N, K = neighbor_idx.shape
    M = excl_indices.shape[1]

    # Broadcast to (N, K, M)
    # neighbor_idx: (N, K, 1)
    # excl_indices: (N, 1, M)

    n_idx_broad = neighbor_idx[:, :, None]
    excl_idx_broad = excl_indices[:, None, :] # (N, 1, M)

    # Check for matches
    # match: (N, K, M) boolean
    # neighbor index == excluded index
    match = (n_idx_broad == excl_idx_broad) & (excl_idx_broad != -1)

    # If any match is found for a neighbor (k), we take the scale.
    # Since an atom appears at most once in exclusion list, max() works.

    # However, if NO match, scale should be 1.0.
    # Current arrays have scales, but we need to select them.

    # Extract scales where match is True
    # (N, K, M)
    vdw_matches = match * excl_scales_vdw[:, None, :]
    elec_matches = match * excl_scales_elec[:, None, :]

    # Sum over M (should be at most one match)
    # If no match, sum is 0.0. But we want 1.0 for non-excluded.
    # So we need to detect IF there was a match.

    has_match = jnp.any(match, axis=2) # (N, K)

    sum_vdw = jnp.sum(vdw_matches, axis=2) # (N, K)
    sum_elec = jnp.sum(elec_matches, axis=2) # (N, K)

    final_vdw = jnp.where(has_match, sum_vdw, 1.0)
    final_elec = jnp.where(has_match, sum_elec, 1.0)

    return final_vdw, final_elec
