"""Tests for batched energy function via vmap."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_md import space
from proxide.core.containers import Protein

from prolix.batched_energy import make_batched_energy_fn
from prolix.padding import bucket_proteins, collate_batch


@pytest.fixture
def fake_proteins() -> list[Protein]:
    """Create 3 small fake proteins of varying sizes."""
    proteins = []
    sizes = [5, 12, 8]
    
    for n in sizes:
        p = Protein(
            coordinates=np.random.randn(n, 3),
            aatype=np.zeros(n, dtype=np.int32),
            residue_index=np.zeros(n, dtype=np.int32),
            chain_index=np.zeros(n, dtype=np.int32),
            charges=np.linspace(-0.2, 0.2, n),
            sigmas=np.ones(n) * 3.0,
            epsilons=np.ones(n) * 0.1,
            radii=np.ones(n) * 1.5,
            scaled_radii=np.ones(n) * 0.8,
            masses=np.ones(n) * 12.0,
            bonds=np.array([[i, i+1] for i in range(n-1)]),
            bond_params=np.array([[1.5, 300.0] for _ in range(n-1)]),
            angles=np.array([[i, i+1, i+2] for i in range(n-2)]),
            angle_params=np.array([[2.0, 100.0] for _ in range(n-2)]),
        )
        proteins.append(p)
    return proteins


def test_make_batched_energy_fn(fake_proteins):
    """Test that we can construct a batched energy fn and apply it."""
    # 1. Bucket and pad
    buckets = bucket_proteins(fake_proteins, buckets=(32, 64))
    padded_list = buckets[32]  # All 3 fit in 32
    
    assert len(padded_list) == 3
    
    # 2. Collate
    batch = collate_batch(padded_list)
    assert batch.positions.shape == (3, 32, 3)
    
    # 3. Define fn
    displacement_fn, _ = space.free()
    batched_energy = make_batched_energy_fn(displacement_fn, implicit_solvent=True)
    
    # 4. Evaluate
    energies = batched_energy(batch)
    
    assert energies.shape == (3,)
    assert jnp.all(jnp.isfinite(energies))
    
    # Values should all be distinct due to different protein sizes
    assert not jnp.allclose(energies[0], energies[1])
    assert not jnp.allclose(energies[1], energies[2])


def test_grad_batched_energy_fn(fake_proteins):
    """Verify that taking gradient of the batched energy is numerically safe for padded elements."""
    buckets = bucket_proteins(fake_proteins, buckets=(32, 64))
    batch = collate_batch(buckets[32])
    
    displacement_fn, _ = space.free()
    batched_energy = make_batched_energy_fn(displacement_fn, implicit_solvent=True)
    
    import equinox as eqx

    def sum_energy(pos):
        new_batch = eqx.tree_at(lambda b: b.positions, batch, pos)
        return jnp.sum(batched_energy(new_batch))
        
    grad_pos = jax.grad(sum_energy)(batch.positions)
    
    assert grad_pos.shape == (3, 32, 3)
    assert jnp.all(jnp.isfinite(grad_pos))
    
    # Verify that ghost atoms receive EXACTLY 0.0 gradient
    N1 = fake_proteins[0].coordinates.shape[0]  # size 5
    ghost_grads = grad_pos[0, N1:]
    np.testing.assert_allclose(ghost_grads, 0.0, atol=1e-6)
