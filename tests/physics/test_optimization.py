import jax
import jax.numpy as jnp
from jax_md import space
import pytest
from prolix.physics.optimization import chunked_lj_energy, chunked_coulomb_energy, chunked_lj_energy_nl


def test_chunked_lj_tiling_alignment_regression():
    """Regression for tiling atom-drop bug: inner_tile_size must be tile_size multiple.

    At excl_count=897 (excl_count+128=1025, not divisible by 128), the pre-fix code
    used inner_tile_size=1025, silently dropped atoms in tile_reduction, and produced
    10^62 K blowup at 895-water scale.

    The fix (src/prolix/physics/optimization.py, line 28) uses ceiling division:
        inner_tile_size = ((_need + tile_size - 1) // tile_size) * tile_size
    """
    from jax_md import space
    displacement_fn, _ = space.free()

    n_atoms = 8
    key = jax.random.PRNGKey(0)
    positions = jax.random.normal(key, (n_atoms, 3)) * 2.0
    sigmas = jnp.ones(n_atoms) * 3.0
    epsilons = jnp.ones(n_atoms) * 0.1

    # Case 1: excl_count=897 → _need=max(1024,1025)=1025 → NOT tile-aligned without the fix
    excl_count = 897
    excl_indices = jnp.zeros((excl_count, 2), dtype=jnp.int32)
    excl_scales = jnp.ones((excl_count, 2))

    result = chunked_lj_energy(positions, sigmas, epsilons, excl_indices, excl_scales, displacement_fn)
    assert jnp.isfinite(result), f"NaN/Inf at excl_count={excl_count} (tiling regression)"

    # Case 2: Aligned baseline (excl_count=896 → _need=1024, aligned)
    excl_count_2 = 896
    excl_indices_2 = jnp.zeros((excl_count_2, 2), dtype=jnp.int32)
    excl_scales_2 = jnp.ones((excl_count_2, 2))
    result_2 = chunked_lj_energy(positions, sigmas, epsilons, excl_indices_2, excl_scales_2, displacement_fn)
    assert jnp.isfinite(result_2), f"NaN/Inf at excl_count={excl_count_2} (aligned baseline)"

    # Case 3: Test chunked_lj_energy_nl with the same misaligned excl_count
    neighbor_idx = jnp.array([[0, 1], [1, 2], [2, 3]], dtype=jnp.int32)
    result_nl = chunked_lj_energy_nl(positions, sigmas, epsilons, excl_indices, excl_scales, neighbor_idx, displacement_fn)
    assert jnp.isfinite(result_nl), f"NaN/Inf at excl_count={excl_count} in chunked_lj_energy_nl"

def test_chunked_coulomb_parity():
    n_atoms = 2
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0]
    ])
    charges = jnp.array([1.0, -1.0])

    displacement_fn, _ = space.free()
    pme_alpha = 0.34
    C = 332.0636 # Match optimization.py constant

    # Empty exclusions (no special pairs)
    excl_indices = jnp.zeros((0, 2), dtype=jnp.int32)
    excl_scales = jnp.zeros((0, 2))

    # Baseline
    def dense_coul(r):
        dr_mat = jax.vmap(jax.vmap(displacement_fn, (None, 0)), (0, None))(r, r)
        dist = jnp.sqrt(jnp.sum(dr_mat**2, axis=-1) + 1e-12)
        mask_diag = (1.0 - jnp.eye(n_atoms)).astype(bool)
        q_ij = charges[:, None] * charges[None, :]
        dist_safe = jnp.where(mask_diag, dist, 1.0)
        erfc_term = jax.scipy.special.erfc(pme_alpha * dist_safe)
        e_pair = C * (q_ij / dist_safe) * erfc_term
        return 0.5 * jnp.sum(jnp.where(mask_diag, e_pair, 0.0))

    res_dense = dense_coul(positions)
    res_chunked = chunked_coulomb_energy(positions, charges, excl_indices, excl_scales, displacement_fn, pme_alpha, C, tile_size=32)

    assert jnp.allclose(res_dense, res_chunked, rtol=1e-5)

    grad_dense = jax.grad(dense_coul)(positions)
    grad_chunked = jax.grad(lambda r: chunked_coulomb_energy(r, charges, excl_indices, excl_scales, displacement_fn, pme_alpha, C, tile_size=32))(positions)

    assert jnp.allclose(grad_dense, grad_chunked, rtol=1e-4)

def test_chunked_lj_parity():
    n_atoms = 2
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0]
    ])
    sigmas = jnp.array([3.0, 3.0])
    epsilons = jnp.array([0.1, 0.1])

    displacement_fn, _ = space.free()

    # Empty exclusions (no special pairs)
    excl_indices = jnp.zeros((0, 2), dtype=jnp.int32)
    excl_scales = jnp.zeros((0, 2))

    # Baseline (dense materialization)
    def dense_lj(r):
        dr_mat = jax.vmap(jax.vmap(displacement_fn, (None, 0)), (0, None))(r, r)
        dist = jnp.sqrt(jnp.sum(dr_mat**2, axis=-1) + 1e-12)
        mask_diag = (1.0 - jnp.eye(n_atoms)).astype(bool)
        sig_ij = 0.5 * (sigmas[:, None] + sigmas[None, :])
        eps_ij = jnp.sqrt(epsilons[:, None] * epsilons[None, :])

        # Masked LJ 12-6
        dist_safe = jnp.where(mask_diag, dist, 1.0)
        inv_r6 = (sig_ij / dist_safe)**6
        e_pair = 4.0 * eps_ij * (inv_r6**2 - inv_r6)
        return 0.5 * jnp.sum(jnp.where(mask_diag, e_pair, 0.0))

    # Chunked optimized version
    res_dense = dense_lj(positions)
    res_chunked = chunked_lj_energy(positions, sigmas, epsilons, excl_indices, excl_scales, displacement_fn, tile_size=32)

    assert jnp.allclose(res_dense, res_chunked, rtol=1e-5)

    # Gradient parity
    grad_dense = jax.grad(dense_lj)(positions)
    grad_chunked = jax.grad(lambda r: chunked_lj_energy(r, sigmas, epsilons, excl_indices, excl_scales, displacement_fn, tile_size=32))(positions)

    assert jnp.allclose(grad_dense, grad_chunked, rtol=1e-4)
