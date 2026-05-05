import jax
import jax.numpy as jnp
import pytest
import equinox as eqx
from prolix.physics.pure_utils import wrap_energy_fn_pure
from prolix.typing import EnergyParams

def test_energy_parity():
    # Simple harmonic potential: 0.5 * k * (x - x0)^2
    def legacy_harmonic_energy(positions, box, k, x0):
        # Closure uses k, x0
        return 0.5 * k * jnp.sum((positions - x0)**2)

    # Pure energy function creation
    pure_energy_fn = wrap_energy_fn_pure(legacy_harmonic_energy)

    # Test parameters
    k = 10.0
    x0 = 0.5
    params = EnergyParams(params={'k': k, 'x0': x0})

    # Test over 10 geometries
    key = jax.random.PRNGKey(42)
    for _ in range(10):
        key, subkey = jax.random.split(key)
        positions = jax.random.normal(subkey, (10, 3))
        box = None # Assuming non-PBC for simplicity
        
        legacy_res = legacy_harmonic_energy(positions, box, k=k, x0=x0)
        pure_res = pure_energy_fn(params, positions, box)
        
        assert jnp.abs(legacy_res - pure_res) < 1e-6
        
    # Test JIT/Grad
    jit_energy = jax.jit(pure_energy_fn)
    grad_energy = jax.grad(pure_energy_fn, argnums=1)
    
    positions = jax.random.normal(key, (10, 3))
    jit_res = jit_energy(params, positions, None)
    grad_res = grad_energy(params, positions, None)
    
    assert jit_res.shape == ()
    assert grad_res.shape == (10, 3)
