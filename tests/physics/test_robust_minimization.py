
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_md import space

from prolix import simulate
from prolix.physics import system

# Enable x64 for physics
jax.config.update("jax_enable_x64", True)

@pytest.fixture
def clash_system():
    """Setup a simple harmonic systems with two particles that clash."""
    
    # Simple harmonic 2-particle system
    # V = k * (r - r0)^2
    # But we want to test LJ-like behavior where V -> inf as r -> 0
    
    # Let's use actual LJ potential for testing robustness
    sigma = 1.0
    epsilon = 1.0
    
    def energy_fn(R, neighbor=None):
        dr = R[0] - R[1]
        dist = jnp.linalg.norm(dr)
        
        # LJ potential
        sr6 = (sigma / dist) ** 6
        energy = 4 * epsilon * (sr6**2 - sr6)
        return energy

    # Initial position: Very close (clash)
    # sigma=1.0. Minimum at 2^(1/6) ~= 1.12
    # Place at 0.5 -> Huge Repulsion
    R_clash = jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    
    return energy_fn, R_clash

def test_clash_minimization_survives(clash_system):
    """Test that minimization doesn't produce NaNs even with severe clashes."""
    energy_fn, R_init = clash_system
    
    # Standard minimization (might fail with simple GD, expecting success with robust)
    spec = simulate.SimulationSpec(total_time_ns=0.001) # Dummy spec
    
    # We invoke the robust minimization logic directly if possible, or via run_simulation
    # Since we are testing internal logic, we might need to expose it or run a minimal simulation loop
    
    # But prolix's minimize is inside run_simulation currently.
    # We will test using run_simulation with a mock system if possible, 
    # but run_simulation requires a SystemParams or AtomicSystem.
    
    # Let's verify the energy explosion with a simple manually constructed GD loop first
    # to confirm the baseline behavior, then verify the fix.
    
    # Baseline check: Does simple GD explode?
    grad_fn = jax.grad(energy_fn)
    
    R = R_init
    step_size = 0.001
    
    # Simple GD for 10 steps
    for i in range(10):
        g = grad_fn(R)
        R = R - step_size * g
        if not jnp.all(jnp.isfinite(R)):
            print(f"Exploded at step {i}")
            break
            
    # We expect this to likely be unstable or result in huge jumps.
    # The real test is running the NEW code in simulate.py.
    pass


@pytest.fixture
def minimal_lj_system():
    """Create a minimal 2-particle system params dict."""
    # 2 particles
    n_atoms = 2
    
    # LJ parameters
    sigmas = jnp.array([1.0, 1.0])
    epsilons = jnp.array([1.0, 1.0])
    charges = jnp.zeros(n_atoms)
    
    # Empty topology
    bonds = jnp.zeros((0, 2), dtype=jnp.int32)
    bond_params = jnp.zeros((0, 2))
    angles = jnp.zeros((0, 3), dtype=jnp.int32)
    angle_params = jnp.zeros((0, 2))
    dihedrals = jnp.zeros((0, 4), dtype=jnp.int32)
    dihedral_params = jnp.zeros((0, 3))
    impropers = jnp.zeros((0, 4), dtype=jnp.int32)
    improper_params = jnp.zeros((0, 3))
    
    system_params = {
        "charges": charges,
        "sigmas": sigmas,
        "epsilons": epsilons,
        "bonds": bonds,
        "bond_params": bond_params,
        "angles": angles,
        "angle_params": angle_params,
        "dihedrals": dihedrals,
        "dihedral_params": dihedral_params,
        "impropers": impropers,
        "improper_params": improper_params,
        "gb_radii": jnp.ones(n_atoms) * 1.5, # Dummy radii
    }
    return system_params

def test_clash_minimization_survives(minimal_lj_system):
    """Test that minimization handles severe overlap without NaN."""
    system_params = minimal_lj_system
    
    # Initial position: Very close overlap (r=0.5 < sigma=1.0)
    # Potential is ~ (1/0.5)^12 = 4096 * 4 = ~16000 epsilon
    # Gradients will be huge.
    initial_positions = jnp.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0]
    ])
    
    spec = simulate.SimulationSpec(
        total_time_ns=0.0001, # Minimal run
        step_size_fs=1.0,
        save_interval_ns=0.0001,
        accumulate_steps=1,
        use_pbc=False,
        use_neighbor_list=False
    )
    
    # This should NOT rasie NaN/Inf error or RuntimeWarning
    # `run_simulation` runs minimization internally
    final_state = simulate.run_simulation(
        system=system_params,
        initial_positions=initial_positions,
        spec=spec
    )
    
    # Check that positions are finite and separated
    pos = final_state.positions
    dist = jnp.linalg.norm(pos[0] - pos[1])
    
    print(f"Final distance: {dist}")
    
    assert jnp.all(jnp.isfinite(pos))
    assert dist > 0.8 # Should have pushed apart significantly
    # Ideally < 0.0, but if it pushes to > cutoff (9.0) it might be 0. 
    # Or if slightly repulsive but safe, that's okay too compared to start.
    assert final_state.potential_energy < 100.0 

def test_standard_minimization_trajectory(minimal_lj_system):
    """Test that robust minimization still converges for normal cases."""
    system_params = minimal_lj_system
    
    # Start slightly outside minimum
    initial_positions = jnp.array([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0]
    ])
    
    spec = simulate.SimulationSpec(
        total_time_ns=0.0001,
        step_size_fs=1.0,
    )
    
    final_state = simulate.run_simulation(
        system=system_params,
        initial_positions=initial_positions,
        spec=spec
    )
    
    pos = final_state.positions
    dist = jnp.linalg.norm(pos[0] - pos[1])
    
    # Expected min is 2^(1/6) * sigma = 1.122
    assert jnp.abs(dist - 1.122) < 0.05

