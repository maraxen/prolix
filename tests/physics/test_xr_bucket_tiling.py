"""XR-BUCKET (#746): tiling bucketing helpers and loud invariant checks."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from prolix.physics.tiling import (
    compute_dense_tiling_dims,
    round_up_to_multiple,
    tile_reduction,
)


def test_round_up_to_multiple():
    assert round_up_to_multiple(1, 128) == 128
    assert round_up_to_multiple(128, 128) == 128
    assert round_up_to_multiple(129, 128) == 256
    assert round_up_to_multiple(0, 64) == 64
    with pytest.raises(ValueError):
        round_up_to_multiple(10, 0)


def test_compute_dense_tiling_dims_invariants():
    # 895 waters → 2685 atoms; many excl rows — classic #746 case shape.
    n_atoms, n_excl, tile = 2685, 5000, 128
    pad_dim, inner = compute_dense_tiling_dims(n_atoms, n_excl, tile)
    assert inner % tile == 0
    assert pad_dim % tile == 0
    assert pad_dim >= tile
    assert pad_dim >= n_atoms
    assert pad_dim >= n_excl
    assert pad_dim == max(tile, inner)


def test_compute_dense_tiling_dims_matches_legacy_formula_when_n_atoms_small():
    n_excl, tile = 200, 128
    _need = max(1024, n_excl + 128)
    legacy_inner = ((_need + tile - 1) // tile) * tile
    legacy_pad = max(tile, legacy_inner)
    pad_dim, inner = compute_dense_tiling_dims(64, n_excl, tile)
    assert (pad_dim, inner) == (legacy_pad, legacy_inner)


def test_tile_reduction_raises_on_non_multiple():
    positions = jnp.zeros((100, 3))  # 100 % 128 != 0
    mask = jnp.ones(100, dtype=bool)

    def f_tile(*_a):
        return 0.0

    with pytest.raises(ValueError, match="#746"):
        tile_reduction(positions, mask, f_tile, 0.0, tile_size=128, inner_tile_size=1024)


def test_optimization_uses_compute_dense_tiling_dims():
    import inspect

    from prolix.physics import optimization as opt

    src = inspect.getsource(opt)
    assert "compute_dense_tiling_dims" in src
    assert src.count("((_need + tile_size - 1) // tile_size) * tile_size") == 0
