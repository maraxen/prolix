"""FlashMD tiling and padding primitives for O(N^2) and O(N*K) kernels.

Adapted from jaxbeans.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, TypeVar

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool

if TYPE_CHECKING:
    pass

N = "N"
T = "T"
_T = TypeVar("_T")


def pad_to_tile(
    arr: Array, tile_size: int, axis: int = 0, pad_value: Any = 0.0
) -> tuple[Array, Bool[Array, N]]:  # noqa: F821
    """Pad an array so its length is divisible by `tile_size`.

    Args:
        arr: The input array.
        tile_size: The target tile dimension.
        axis: The axis to pad along.
        pad_value: The value to use for padding (default 0.0).

    Returns:
        A tuple of (padded_array, mask), where the mask is `True` for real
        elements and `False` for padded ones.
    """
    n = arr.shape[axis]
    remainder = n % tile_size
    if remainder == 0:
        return arr, jnp.ones(n, dtype=jnp.bool_)

    padding_needed = tile_size - remainder
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, padding_needed)

    padded_arr = jnp.pad(arr, pad_width, constant_values=pad_value)

    # Create the mask
    mask = jnp.arange(n + padding_needed) < n

    return padded_arr, mask


def tile_reduction(
    positions: Array,
    atom_mask: Bool[Array, N],  # noqa: F821
    f_tile: Callable[[Array, Array, Array, Bool[Array, T], int], Array],  # noqa: F821
    init_state: Any,
    tile_size: int,
) -> Any:
    """Execute a tiled O(N^2) reduction using the FlashMD pattern.

    $$ Peak Memory = O(N \\cdot tile_size) $$

    Args:
        positions: (N, 3) coordinate array.
        atom_mask: (N,) boolean mask for real atoms.
        f_tile: Kernel function (pos_i, pos_j, mask_i, mask_j, start_idx) -> tile_result.
        init_state: Initial accumulation state (usually zeros).
        tile_size: Hyperparameter for chunking (T).

    Returns:
        The accumulated reduction result.
    """
    n = positions.shape[0]
    n_tiles = n // tile_size

    def _tile_step(carry, j_idx):
        start_idx = j_idx * tile_size

        pos_j = jax.lax.dynamic_slice(positions, (start_idx, 0), (tile_size, positions.shape[-1]))
        mask_j = jax.lax.dynamic_slice(atom_mask, (start_idx,), (tile_size,))

        # Execute kernel
        tile_res = f_tile(positions, pos_j, atom_mask, mask_j, start_idx)
        return carry + tile_res, None

    final_state, _ = jax.lax.scan(_tile_step, init_state, jnp.arange(n_tiles))
    return final_state


def tile_reduction_nl(
    positions: Array,
    neighbor_idx: Array,
    atom_mask: Bool[Array, N],  # noqa: F821
    f_tile: Callable[[Array, Array, Array, Array, Array, int], Array],  # noqa: F821
    init_state: Any,
    tile_size: int,
) -> Any:
    """Execute a tiled O(N*K) reduction over neighbor lists.

    Args:
        positions: (N, 3) coordinate array.
        neighbor_idx: (N, K) neighbor indices.
        atom_mask: (N,) boolean mask for real atoms.
        f_tile: Kernel function (pos_i, pos_j, mask_i, mask_j, nb_idx_tile, start_idx) -> tile_result.
        init_state: Initial accumulation state.
        tile_size: Hyperparameter for chunking the K dimension.

    Returns:
        The accumulated reduction result.
    """
    n, k = neighbor_idx.shape
    
    # Ensure k is divisible by tile_size
    remainder = k % tile_size
    if remainder != 0:
        padding_needed = tile_size - remainder
        neighbor_idx = jnp.pad(neighbor_idx, ((0, 0), (0, padding_needed)), constant_values=-1)
        k = k + padding_needed
        
    n_tiles = k // tile_size

    def _tile_step(carry, j_idx):
        start_idx = j_idx * tile_size

        # Slice neighbors for this tile
        nb_idx_tile = jax.lax.dynamic_slice(neighbor_idx, (0, start_idx), (n, tile_size)) # (N, T)
        
        # In JAX_MD, neighbor_idx == N represents a padded/empty neighbor.
        # We also support -1 as a sentinel from our own padding.
        # mask_j: (N, T)
        mask_j = (nb_idx_tile < n) & (nb_idx_tile >= 0)

        # Gather neighbor positions
        # pos_j: (N, T, 3)
        pos_j = jnp.where(mask_j[..., None], positions[nb_idx_tile], 0.0)

        # Execute kernel
        tile_res = f_tile(positions, pos_j, atom_mask, mask_j, nb_idx_tile, start_idx)
        return carry + tile_res, None

    final_state, _ = jax.lax.scan(_tile_step, init_state, jnp.arange(n_tiles))
    return final_state
