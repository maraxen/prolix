import jax
import jax.numpy as jnp
from jax_md import space
from prolix.physics import system, pressure, explicit_corrections
from prolix.typing import PhysicsSystem, EnergyParams
import pytest

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
