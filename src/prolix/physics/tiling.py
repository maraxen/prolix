"""FlashMD tiling and padding primitives for O(N^2) kernels.

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
    *,
    rematerialize: bool = True,
) -> Any:
    """Execute a tiled O(N^2) reduction using the FlashMD pattern.

    $$ Peak Memory = O(N \\cdot tile_size) $$

    Args:
        positions: (N, 3) coordinate array.
        atom_mask: (N,) boolean mask for real atoms.
        f_tile: Kernel function (pos_i, pos_j, mask_i, mask_j, start_idx) -> tile_result.
        init_state: Initial accumulation state (usually zeros).
        tile_size: Hyperparameter for chunking (T).
        rematerialize: Whether to wrap the tile in `jax.checkpoint`.

    Returns:
        The accumulated reduction result.
    """
    n = positions.shape[0]
    n_tiles = n // tile_size

    def _tile_step(carry, j_idx):
        start_idx = j_idx * tile_size

        def _compute_tile(pos, m, start):
            pos_j = jax.lax.dynamic_slice(pos, (start, 0), (tile_size, pos.shape[-1]))
            mask_j = jax.lax.dynamic_slice(m, (start,), (tile_size,))

            # Execute kernel
            return f_tile(pos, pos_j, atom_mask, mask_j, start)

        if rematerialize:
            _compute_tile = jax.checkpoint(_compute_tile)

        tile_res = _compute_tile(positions, atom_mask, start_idx)
        return carry + tile_res, None

    final_state, _ = jax.lax.scan(_tile_step, init_state, jnp.arange(n_tiles))
    return final_state
