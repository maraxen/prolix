import pytest
import jax
import jax.numpy as jnp
from prolix.physics.integrator_builder import make_integrator
from prolix.physics.step_system import step_sequences

def test_modular_scr_barostat_smoke():
    """Verify that baoab_csvr_npt instantiates and runs without errors."""
    n_atoms = 10
    positions = jax.random.uniform(jax.random.PRNGKey(0), (n_atoms, 3)) * 10.0
    box = jnp.array([20.0, 20.0, 20.0])
    mass = jnp.ones((n_atoms, 1))
    
    def energy_fn(r, box=None, params=None):
        return 0.5 * jnp.sum(r**2)
    
    def shift_fn(r, box=None):
        return r
        
    energy_params = {
        "charges": jnp.zeros(n_atoms),
        "sigmas": jnp.ones(n_atoms),
        "epsilons": jnp.ones(n_atoms),
    }
    
    # Instantiate modular NPT integrator
    # Use high target pressure to trigger significant scaling
    init_fn, apply_fn = make_integrator(
        energy_fn=energy_fn,
        shift_fn=shift_fn,
        mass=mass,
        energy_params=energy_params,
        sequence_name="baoab_csvr_npt",
        dt=1.0,
        kT=1.0,
        target_pressure_bar=100000.0, 
        tau_barostat_akma=50.0, # Very fast relaxation
    )
    
    state = init_fn(jax.random.PRNGKey(42), positions, box=box)
    
    # Run a few steps
    curr_state = state
    for _ in range(5):
        curr_state = apply_fn(curr_state)
        
    # Check that box has changed
    assert not jnp.allclose(curr_state.box, box, atol=1e-5)
    assert jnp.all(curr_state.box > 0)
    assert not jnp.any(jnp.isnan(curr_state.positions))
    assert curr_state.step_count == 5

def test_modular_npt_with_settle():
    """Verify that SETTLE steps in NPT sequence work."""
    # 1 water molecule
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [0.9572, 0.0, 0.0],
        [-0.24, 0.92, 0.0]
    ])
    mass = jnp.array([[15.999], [1.008], [1.008]])
    water_indices = jnp.array([[0, 1, 2]])
    box = jnp.array([20.0, 20.0, 20.0])
    
    def energy_fn(r, box=None, params=None):
        return 0.0 # Zero energy, only constraints and barostat
        
    energy_params = {
        "charges": jnp.zeros(3),
        "sigmas": jnp.ones(3),
        "epsilons": jnp.ones(3),
    }
    
    init_fn, apply_fn = make_integrator(
        energy_fn=energy_fn,
        shift_fn=lambda r, b: r,
        mass=mass,
        energy_params=energy_params,
        sequence_name="baoab_csvr_npt",
        water_indices=water_indices,
        target_pressure_bar=1.0,
    )
    
    state = init_fn(jax.random.PRNGKey(42), positions, box=box)
    
    # Before scaling, check constraints
    def get_dist(p, i, j):
        return jnp.sqrt(jnp.sum((p[i]-p[j])**2))
    
    d_OH1 = get_dist(state.positions, 0, 1)
    assert jnp.allclose(d_OH1, 0.9572, atol=1e-3)
    
    # Run 1 step (includes SCR and SETTLE)
    state_new = apply_fn(state)
    
    # After scaling, check constraints are still preserved
    d_OH1_new = get_dist(state_new.positions, 0, 1)
    assert jnp.allclose(d_OH1_new, 0.9572, atol=1e-3)

if __name__ == "__main__":
    test_modular_scr_barostat_smoke()
    test_modular_npt_with_settle()
