import jax
import jax.numpy as jnp
from jax_md import space
import pytest
from prolix.physics.optimization import chunked_lj_energy, chunked_coulomb_energy

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
    res_chunked = chunked_coulomb_energy(positions, charges, displacement_fn, pme_alpha, C, tile_size=32)
    
    assert jnp.allclose(res_dense, res_chunked, rtol=1e-5)
    
    grad_dense = jax.grad(dense_coul)(positions)
    grad_chunked = jax.grad(lambda r: chunked_coulomb_energy(r, charges, displacement_fn, pme_alpha, C, tile_size=32))(positions)
    
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
    res_chunked = chunked_lj_energy(positions, sigmas, epsilons, displacement_fn, tile_size=32)
    
    assert jnp.allclose(res_dense, res_chunked, rtol=1e-5)
    
    # Gradient parity
    grad_dense = jax.grad(dense_lj)(positions)
    grad_chunked = jax.grad(lambda r: chunked_lj_energy(r, sigmas, epsilons, displacement_fn, tile_size=32))(positions)
    
    assert jnp.allclose(grad_dense, grad_chunked, rtol=1e-4)
