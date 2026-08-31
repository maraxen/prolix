import jax
import jax.numpy as jnp
from jax_md import space
from prolix.physics import system, pressure, explicit_corrections
from prolix.typing import PhysicsSystem, EnergyParams
import pytest


@pytest.fixture
def enable_x64():
    """Enable JAX x64 mode for this test and restore on teardown."""
    old_value = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old_value)

def test_tail_pressure_derivative():
    # Setup a simple periodic system
    box_size = jnp.array([20.0, 20.0, 20.0])
    N = 100
    positions = jnp.linspace(0, 19.0, N).reshape(-1, 1) * jnp.ones((N, 3))
    
    # Fake system
    d = {
        "charges": jnp.zeros(N),
        "sigmas": jnp.ones(N) * 0.3,
        "epsilons": jnp.ones(N) * 0.1,
    }
    physics_system = PhysicsSystem.from_dict(
        d, positions, box_size=box_size
    )
    
    params = EnergyParams(params={
        'charges': physics_system.charges,
        'sigmas': physics_system.sigmas,
        'epsilons': physics_system.epsilons
    })
    
    cutoff = 9.0
    
    # 1. Compute E_tail
    e_tail_val = explicit_corrections.lj_dispersion_tail_energy(
        box_size, physics_system.sigmas, physics_system.epsilons, cutoff, physics_system.atom_mask
    )
    
    # 2. Compute P_tail numerically dE/dV
    def get_e_tail(box):
        return explicit_corrections.lj_dispersion_tail_energy(
            box, physics_system.sigmas, physics_system.epsilons, cutoff, physics_system.atom_mask
        )
    
    # Perturb volume
    def get_e_tail_by_vol(v):
        # assume cubic box
        side = v**(1.0/3.0)
        return get_e_tail(jnp.array([side, side, side]))
        
    vol = jnp.prod(box_size)
    eps = 1e-3
    e1 = get_e_tail_by_vol(vol - eps)
    e2 = get_e_tail_by_vol(vol + eps)
    p_num = -(e2 - e1) / (2 * eps)
    
    # 3. Compute P_tail from analytical formula
    p_tail_val = explicit_corrections.lj_dispersion_tail_pressure(
        box_size, physics_system.sigmas, physics_system.epsilons, cutoff, physics_system.atom_mask
    )
    
    assert jnp.allclose(p_num, p_tail_val, atol=1e-3)


def test_lj_tail_energy_f64_autodiff(enable_x64):
    """Test that lj_dispersion_tail_energy supports f64 and autodiff without dtype crash.

    Regression test for debt 832: hardcoded float32 casts caused:
      TypeError: lax.add requires arguments to have the same dtypes, got float32, float64
    when differentiating under jax_enable_x64=True.
    """
    box_size = jnp.array([20.0, 20.0, 20.0], dtype=jnp.float64)
    N = 10
    sigmas = jnp.ones(N, dtype=jnp.float64) * 0.3
    epsilons = jnp.ones(N, dtype=jnp.float64) * 0.1
    atom_mask = jnp.ones(N, dtype=jnp.bool_)
    cutoff = 9.0

    # Forward pass should work
    e_tail = explicit_corrections.lj_dispersion_tail_energy(
        box_size, sigmas, epsilons, cutoff, atom_mask
    )
    assert e_tail.dtype == jnp.float64

    # Backward pass (autodiff) should NOT crash with dtype mismatch
    def energy_fn(eps):
        return explicit_corrections.lj_dispersion_tail_energy(
            box_size, sigmas, eps, cutoff, atom_mask
        )

    grad_fn = jax.grad(energy_fn)
    grad_e = grad_fn(epsilons)
    assert grad_e.dtype == jnp.float64
    assert jnp.all(jnp.isfinite(grad_e))


def test_lj_tail_impulsive_pressure_f64_autodiff(enable_x64):
    """Test that lj_dispersion_tail_impulsive_pressure supports f64 and autodiff.

    Regression test for debt 832 in the impulsive pressure variant.
    """
    box_size = jnp.array([20.0, 20.0, 20.0], dtype=jnp.float64)
    N = 10
    sigmas = jnp.ones(N, dtype=jnp.float64) * 0.3
    epsilons = jnp.ones(N, dtype=jnp.float64) * 0.1
    atom_mask = jnp.ones(N, dtype=jnp.bool_)
    cutoff = 9.0

    # Forward pass should work
    p_tail = explicit_corrections.lj_dispersion_tail_impulsive_pressure(
        box_size, sigmas, epsilons, cutoff, atom_mask
    )
    assert p_tail.dtype == jnp.float64

    # Backward pass (autodiff) should NOT crash with dtype mismatch
    def pressure_fn(eps):
        return explicit_corrections.lj_dispersion_tail_impulsive_pressure(
            box_size, sigmas, eps, cutoff, atom_mask
        )

    grad_fn = jax.grad(pressure_fn)
    grad_p = grad_fn(epsilons)
    assert grad_p.dtype == jnp.float64
    assert jnp.all(jnp.isfinite(grad_p))
