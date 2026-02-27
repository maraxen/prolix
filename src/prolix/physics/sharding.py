"""Multi-GPU sharding utilities for pmap-based parallel simulation.

Provides padding, sharding, and unpadding for distributing N^2 non-bonded
computations across multiple GPUs using JAX pmap.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import jax.numpy as jnp
from jax_md import util

Array = util.Array


@dataclasses.dataclass(frozen=True)
class PaddedSystem:
    """System padded to pmap-compatible size.

    Attributes:
        positions: Padded positions (N_padded, 3).
        charges: Padded charges (N_padded,). Ghost atoms have charge 0.
        atom_mask: Boolean mask (N_padded,). True for real atoms, False for ghosts.
        sigmas: Padded LJ sigma (N_padded,). Ghost atoms have sigma 1e-6.
        epsilons: Padded LJ epsilon (N_padded,). Ghost atoms have epsilon 0.
        radii: Padded Born radii (N_padded,) or None.
        scaled_radii: Padded scaled Born radii (N_padded,) or None.
        exclusion_mask: Padded exclusion mask (N_padded, N_padded) or None.
        n_real: Number of real atoms.
        n_padded: Number of atoms after padding.
    """

    positions: Array
    charges: Array
    atom_mask: Array
    sigmas: Array | None = None
    epsilons: Array | None = None
    radii: Array | None = None
    scaled_radii: Array | None = None
    exclusion_mask: Array | None = None
    n_real: int = 0
    n_padded: int = 0


@dataclasses.dataclass(frozen=True)
class UnpaddedResults:
    """Results with ghost atoms stripped."""

    energy: Array | None = None
    forces: Array | None = None
    born_radii: Array | None = None


def _next_divisible(n: int, d: int) -> int:
    """Return smallest integer >= n that is divisible by d."""
    return math.ceil(n / d) * d


def _next_power_of_2(n: int) -> int:
    """Return smallest power of 2 >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def pad_system_for_pmap(
    positions: Array,
    charges: Array,
    atom_mask: Array,
    num_devices: int = 2,
    power_of_2: bool = False,
    sigmas: Array | None = None,
    epsilons: Array | None = None,
    radii: Array | None = None,
    scaled_radii: Array | None = None,
    exclusion_mask: Array | None = None,
) -> PaddedSystem:
    """Pad a system so N_atoms is divisible by num_devices.

    Ghost atoms are placed far away (9999 A) with zero charge and zero
    epsilon to ensure they do not contribute to energies or forces.

    Args:
        positions: Atom positions (N, 3).
        charges: Atom charges (N,).
        atom_mask: Boolean mask (N,). True for real atoms.
        num_devices: Number of GPU devices (typically 2 for Blackwell).
        power_of_2: If True, pad to next power of 2 instead.
        sigmas: Optional LJ sigma (N,).
        epsilons: Optional LJ epsilon (N,).
        radii: Optional Born radii (N,).
        scaled_radii: Optional scaled Born radii (N,).
        exclusion_mask: Optional exclusion mask (N, N).

    Returns:
        PaddedSystem with all arrays padded to N_padded.
    """
    n_real = positions.shape[0]

    if power_of_2:
        n_padded = _next_power_of_2(n_real)
        # Ensure also divisible by num_devices
        n_padded = _next_divisible(n_padded, num_devices)
    else:
        n_padded = _next_divisible(n_real, num_devices)

    pad_count = n_padded - n_real

    if pad_count == 0:
        return PaddedSystem(
            positions=positions,
            charges=charges,
            atom_mask=atom_mask,
            sigmas=sigmas,
            epsilons=epsilons,
            radii=radii,
            scaled_radii=scaled_radii,
            exclusion_mask=exclusion_mask,
            n_real=n_real,
            n_padded=n_padded,
        )

    # Pad positions: place ghost atoms far away
    ghost_positions = jnp.full((pad_count, 3), 9999.0)
    padded_positions = jnp.concatenate([positions, ghost_positions], axis=0)

    # Pad charges: ghost atoms have zero charge
    padded_charges = jnp.concatenate([charges, jnp.zeros(pad_count)])

    # Pad atom mask: ghost atoms are False
    padded_mask = jnp.concatenate(
        [atom_mask, jnp.zeros(pad_count, dtype=jnp.bool_)]
    )

    # Pad sigmas: ghost atoms get minimal sigma
    padded_sigmas = None
    if sigmas is not None:
        padded_sigmas = jnp.concatenate([sigmas, jnp.full(pad_count, 1e-6)])

    # Pad epsilons: ghost atoms get zero epsilon (no LJ)
    padded_epsilons = None
    if epsilons is not None:
        padded_epsilons = jnp.concatenate([epsilons, jnp.zeros(pad_count)])

    # Pad radii: ghost atoms get large radius (minimal Born contribution)
    padded_radii = None
    if radii is not None:
        padded_radii = jnp.concatenate([radii, jnp.full(pad_count, 1.0)])

    padded_scaled_radii = None
    if scaled_radii is not None:
        padded_scaled_radii = jnp.concatenate(
            [scaled_radii, jnp.full(pad_count, 1.0)]
        )

    # Pad exclusion mask: ghost interactions are excluded
    padded_exclusion_mask = None
    if exclusion_mask is not None:
        padded_exclusion_mask = jnp.pad(
            exclusion_mask,
            ((0, pad_count), (0, pad_count)),
            constant_values=False,
        )

    return PaddedSystem(
        positions=padded_positions,
        charges=padded_charges,
        atom_mask=padded_mask,
        sigmas=padded_sigmas,
        epsilons=padded_epsilons,
        radii=padded_radii,
        scaled_radii=padded_scaled_radii,
        exclusion_mask=padded_exclusion_mask,
        n_real=n_real,
        n_padded=n_padded,
    )


def unpad_results(
    n_real: int,
    n_padded: int,
    energy: Array | None = None,
    forces: Array | None = None,
    born_radii: Array | None = None,
) -> UnpaddedResults:
    """Strip ghost atom data from results.

    Args:
        n_real: Number of real atoms.
        n_padded: Number of atoms after padding.
        energy: Scalar energy (passed through unchanged).
        forces: Forces (N_padded, 3) -> trimmed to (N_real, 3).
        born_radii: Born radii (N_padded,) -> trimmed to (N_real,).

    Returns:
        UnpaddedResults with ghost atoms stripped.
    """
    trimmed_forces = forces[:n_real] if forces is not None else None
    trimmed_born_radii = born_radii[:n_real] if born_radii is not None else None

    return UnpaddedResults(
        energy=energy,
        forces=trimmed_forces,
        born_radii=trimmed_born_radii,
    )


def shard_for_pmap(
    arr: Array,
    num_devices: int = 2,
) -> Array:
    """Split an array's first axis across devices for pmap.

    Args:
        arr: Array with shape (N, ...) where N must be divisible by num_devices.
        num_devices: Number of devices to shard across.

    Returns:
        Array with shape (num_devices, N // num_devices, ...).

    Raises:
        ValueError: If N is not divisible by num_devices.
    """
    n = arr.shape[0]
    if n % num_devices != 0:
        msg = (
            f"Array size {n} is not divisible by num_devices={num_devices}. "
            f"Use pad_system_for_pmap first."
        )
        raise ValueError(msg)

    shard_size = n // num_devices
    return arr.reshape(num_devices, shard_size, *arr.shape[1:])


def unshard_from_pmap(
    arr: Array,
    reduce: str | None = None,
) -> Array:
    """Reassemble a sharded array from pmap output.

    Args:
        arr: Sharded array with shape (num_devices, shard_size, ...) or
             (num_devices,) for per-device scalars.
        reduce: If "sum", sum across devices. If None, concatenate.

    Returns:
        Reassembled array.
    """
    if reduce == "sum":
        return jnp.sum(arr, axis=0)

    if arr.ndim == 1:
        # Per-device scalars -- just return as-is or concatenate
        return arr

    num_devices = arr.shape[0]
    shard_size = arr.shape[1]
    return arr.reshape(num_devices * shard_size, *arr.shape[2:])


import jax
from functools import partial

COULOMB_CONSTANT = 332.0637  # kcal*A/(mol*e^2), vacuum dielectric


def sharded_coulomb_energy(
    positions: Array,
    charges: Array,
    atom_mask: Array,
    num_devices: int = 2,
    dielectric: float = 1.0,
) -> Array:
    """Compute Coulomb energy with pmap sharding across devices.

    Each device computes interactions for N/D "central" atoms against ALL N
    atoms, then results are summed. This distributes the O(N^2) memory.

    Args:
        positions: Atom positions (N, 3). N must be divisible by num_devices.
        charges: Atom charges (N,).
        atom_mask: Boolean mask (N,). True = real atom.
        num_devices: Number of devices.
        dielectric: Dielectric constant (default 1.0).

    Returns:
        Total Coulomb energy (scalar).
    """
    C = COULOMB_CONSTANT / dielectric

    # Shard the "central atom" dimension
    pos_sharded = shard_for_pmap(positions, num_devices)       # (D, N/D, 3)
    q_sharded = shard_for_pmap(charges, num_devices)           # (D, N/D)
    mask_sharded = shard_for_pmap(atom_mask, num_devices)      # (D, N/D)

    @partial(jax.pmap, axis_name="devices")
    def _compute_shard(pos_shard, q_shard, mask_shard, all_positions, all_charges, all_mask):
        """Compute Coulomb for a shard of central atoms against all atoms."""
        # pos_shard: (N/D, 3), all_positions: (N, 3)
        # Compute distances: (N/D, N)
        dr = pos_shard[:, None, :] - all_positions[None, :, :]
        dist = jnp.sqrt(jnp.sum(dr**2, axis=-1) + 1e-12)  # (N/D, N)

        # Charge products: (N/D, N)
        q_ij = q_shard[:, None] * all_charges[None, :]

        # Interaction mask: both atoms must be real
        pair_mask = mask_shard[:, None] * all_mask[None, :]  # (N/D, N)

        # Self-interaction mask: need global indices
        shard_size = pos_shard.shape[0]
        device_idx = jax.lax.axis_index("devices")
        global_idx = jnp.arange(shard_size) + device_idx * shard_size
        self_mask = global_idx[:, None] != jnp.arange(all_positions.shape[0])[None, :]

        # Combined mask
        full_mask = pair_mask * self_mask

        # Energy
        dist_safe = dist + 1e-6
        e_coul = C * q_ij / dist_safe * full_mask

        # Half-sum (avoid double counting)
        return 0.5 * jnp.sum(e_coul)

    # Replicate full arrays across devices
    all_pos = jnp.broadcast_to(positions, (num_devices, *positions.shape))
    all_q = jnp.broadcast_to(charges, (num_devices, *charges.shape))
    all_mask_rep = jnp.broadcast_to(atom_mask, (num_devices, *atom_mask.shape))

    # Run pmap
    per_device_energy = _compute_shard(
        pos_sharded, q_sharded, mask_sharded,
        all_pos, all_q, all_mask_rep,
    )

    # Sum across devices
    return jnp.sum(per_device_energy)
