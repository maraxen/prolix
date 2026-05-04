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
    remainder = int(n % tile_size)
    if remainder == 0:
        return arr, jnp.ones(n, dtype=jnp.bool_)

    padding_needed = int(tile_size - remainder)
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, padding_needed)

    padded_arr = jnp.pad(arr, pad_width, constant_values=pad_value)

    # Create the mask
    mask = jnp.arange(n + padding_needed) < n

    return padded_arr, mask


def tile_reduction(
    positions: Array,
    atom_mask: Bool[Array, N],  # noqa: F821
    f_tile: Callable[[Array, Array, Bool[Array, Any], Bool[Array, T], int, int], Any],  # noqa: F821
    init_state: Any,
    tile_size: int,
    inner_tile_size: int = 1024,
) -> Any:
    """Execute a tiled O(N^2) reduction using nested lax.scan.

    $$ Peak Memory = O(inner_tile_size \\cdot tile_size) $$

    Args:
        positions: (N, 3) coordinate array.
        atom_mask: (N,) boolean mask for real atoms.
        f_tile: Kernel function (pos_i, pos_j, mask_i, mask_j, start_i, start_j) -> tile_result.
        init_state: Initial accumulation state (usually zeros).
        tile_size: Hyperparameter for chunking the j dimension (T).
        inner_tile_size: Hyperparameter for chunking the i dimension.

    Returns:
        The accumulated reduction result.
    """
    n = positions.shape[0]
    n_tiles_j = int(n // tile_size)
    n_tiles_i = int(n // inner_tile_size)

    def _j_tile_step(carry, j_idx):
        start_j = j_idx * tile_size
        pos_j = jax.lax.dynamic_slice(positions, (start_j, 0), (tile_size, 3))
        mask_j = jax.lax.dynamic_slice(atom_mask, (start_j,), (tile_size,))

        def _i_tile_step(inner_carry, i_idx):
            start_i = i_idx * inner_tile_size
            pos_i = jax.lax.dynamic_slice(positions, (start_i, 0), (inner_tile_size, 3))
            mask_i = jax.lax.dynamic_slice(atom_mask, (start_i,), (inner_tile_size,))

            # Execute kernel for this (i, j) tile pair
            tile_res = f_tile(pos_i, pos_j, mask_i, mask_j, start_i, start_j)
            
            # Accumulate results.
            # If tile_res is a per-atom array (first dim is inner_tile_size),
            # we update the inner_carry at start_i using advanced indexing.
            if hasattr(tile_res, "shape") and len(tile_res.shape) > 0 and tile_res.shape[0] == inner_tile_size:
                idx = jnp.arange(inner_tile_size) + start_i
                new_inner_carry = inner_carry.at[idx].add(tile_res)
            else:
                # For scalars or global reductions, use standard addition
                new_inner_carry = inner_carry + tile_res
            
            return new_inner_carry, None

        final_inner_state, _ = jax.lax.scan(_i_tile_step, carry, jnp.arange(n_tiles_i))
        return final_inner_state, None

    final_state, _ = jax.lax.scan(_j_tile_step, init_state, jnp.arange(n_tiles_j))
    return final_state


def tile_reduction_nl(
    positions: Array,
    neighbor_idx: Array,
    atom_mask: Bool[Array, N],  # noqa: F821
    f_tile: Callable[[Array, Array, Bool[Array, Any], Bool[Array, Any], Array, int, int], Any],  # noqa: F821
    init_state: Any,
    tile_size: int,
    inner_tile_size: int = 1024,
) -> Any:
    """Execute a tiled O(N*K) reduction over neighbor lists using nested lax.scan.

    Args:
        positions: (N, 3) coordinate array. Already padded to inner_tile_size.
        neighbor_idx: (N_real, K) neighbor indices.
        atom_mask: (N,) boolean mask for real atoms.
        f_tile: Kernel function (pos_i, pos_j, mask_i, mask_j, nb_idx_tile, start_i, start_j) -> tile_result.
        init_state: Initial accumulation state.
        tile_size: Hyperparameter for chunking the K dimension.
        inner_tile_size: Hyperparameter for chunking the N dimension.

    Returns:
        The accumulated reduction result.
    """
    n_padded = positions.shape[0]
    n_atoms_real, k = neighbor_idx.shape
    
    # 1. Ensure neighbor_idx has n_padded rows
    if n_atoms_real < n_padded:
        padding_needed_n = int(n_padded - n_atoms_real)
        neighbor_idx = jnp.pad(neighbor_idx, ((0, padding_needed_n), (0, 0)), constant_values=-1)
    elif n_atoms_real > n_padded:
        # This shouldn't happen if optimization.py pads correctly
        pass

    # 2. Ensure k is divisible by tile_size
    remainder_k = int(k % tile_size)
    if remainder_k != 0:
        padding_needed_k = int(tile_size - remainder_k)
        neighbor_idx = jnp.pad(neighbor_idx, ((0, 0), (0, padding_needed_k)), constant_values=-1)
        k = k + padding_needed_k
        
    n_tiles_j = int(k // tile_size)
    n_tiles_i = int(n_padded // inner_tile_size)

    def _j_tile_step(carry, j_idx):
        start_j = j_idx * tile_size
        
        def _i_tile_step(inner_carry, i_idx):
            start_i = i_idx * inner_tile_size
            
            # Slice atoms and their neighbor indices for this tile
            pos_i = jax.lax.dynamic_slice(positions, (start_i, 0), (inner_tile_size, 3))
            mask_i = jax.lax.dynamic_slice(atom_mask, (start_i,), (inner_tile_size,))
            nb_idx_tile = jax.lax.dynamic_slice(neighbor_idx, (start_i, start_j), (inner_tile_size, tile_size))
            
            # mask_j: (inner_tile_size, tile_size)
            # Support both -1 (our padding) and n_atoms_real (JAX-MD padding)
            mask_j = (nb_idx_tile < n_atoms_real) & (nb_idx_tile >= 0)

            # Gather neighbor positions
            # pos_j: (inner_tile_size, tile_size, 3)
            pos_j = jnp.where(mask_j[..., None], positions[nb_idx_tile], 0.0)

            # Execute kernel
            tile_res = f_tile(pos_i, pos_j, mask_i, mask_j, nb_idx_tile, start_i, start_j)
            
            # Accumulate results
            if hasattr(tile_res, "shape") and len(tile_res.shape) > 0 and tile_res.shape[0] == inner_tile_size:
                idx = jnp.arange(inner_tile_size) + start_i
                new_inner_carry = inner_carry.at[idx].add(tile_res)
            else:
                new_inner_carry = inner_carry + tile_res
                
            return new_inner_carry, None

        final_inner_state, _ = jax.lax.scan(_i_tile_step, carry, jnp.arange(n_tiles_i))
        return final_inner_state, None

    final_state, _ = jax.lax.scan(_j_tile_step, init_state, jnp.arange(n_tiles_j))
    return final_state
