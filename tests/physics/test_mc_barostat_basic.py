import jax
import jax.numpy as jnp
import pytest
from prolix.physics.barostat import MC_Barostat_Step
from prolix.physics.step_system import IntegratorState
from prolix.physics.types import IntegratorParams, EnergyParams

def test_mc_barostat_basic():
    # Setup state
    key = jax.random.PRNGKey(0)
    pos = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    box = jnp.array([10.0, 10.0, 10.0])
    
    state = IntegratorState(
        position=pos,
        momentum=jnp.zeros_like(pos),
        force=jnp.zeros_like(pos),
        mass=jnp.ones(2),
        rng=key,
        box=box,
        step_count=0
    )
    
    # Mock energy function: just returns sum of positions squared (arbitrary)
    def energy_fn(pos, box):
        return jnp.sum(pos**2)

    params = IntegratorParams(
        dt=0.5,
        kT=2.479,
        gamma=1.0,
        energy_params=EnergyParams(params={}),
        molecule_indices=jnp.array([0, 1])
    )
    
    # Barostat step: attempt every 2 steps
    barostat = MC_Barostat_Step(barostat_interval=2, pressure=1.0, energy_fn=energy_fn, n_molecules=2)
    
    # Step 1: Count should increment, but no barostat move
    state1 = barostat.apply(state, params)
    assert state1.step_count == 1
    assert jnp.allclose(state1.position, pos)
    assert jnp.allclose(state1.box, box)
    
    # Step 2: Barostat should trigger
    state2 = barostat.apply(state1, params)
    assert state2.step_count == 2
    # Check that position/box changed (or not)
    # With this simple energy function and barostat, it should perform *some* change
    assert not jnp.allclose(state2.position, pos) or not jnp.allclose(state2.box, box)

if __name__ == "__main__":
    test_mc_barostat_basic()
