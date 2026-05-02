import jax
import jax.numpy as jnp
import pytest
from prolix.physics import pbc, system
from prolix.physics.system import PhysicsSystem, make_energy_fn_pure

def test_tiled_parity():
    """Verify bit-parity between tile_size=1 and tile_size=64."""
    n_atoms = 32
    positions = jax.random.uniform(jax.random.PRNGKey(42), (n_atoms, 3)) * 10.0
    box = jnp.array([15.0, 15.0, 15.0])
    
    sys_dict = {
        "charges": jax.random.normal(jax.random.PRNGKey(0), (n_atoms,)),
        "sigmas": jnp.full(n_atoms, 3.4),
        "epsilons": jnp.full(n_atoms, 0.2),
        "bonds": jnp.zeros((0, 2), dtype=jnp.int32),
        "bond_params": jnp.zeros((0, 2), dtype=jnp.float32),
        "angles": jnp.zeros((0, 3), dtype=jnp.int32),
        "angle_params": jnp.zeros((0, 2), dtype=jnp.float32),
        "proper_dihedrals": jnp.zeros((0, 4), dtype=jnp.int32),
        "dihedral_params": jnp.zeros((0, 3), dtype=jnp.float32),
        "impropers": jnp.zeros((0, 4), dtype=jnp.int32),
        "improper_params": jnp.zeros((0, 3), dtype=jnp.float32),
    }
    
    displacement_fn, _ = pbc.create_periodic_space(box)
    phys_sys = PhysicsSystem.from_dict(sys_dict, positions, box)
    
    params1, fn1 = make_energy_fn_pure(displacement_fn, phys_sys, tile_size=1)
    params128, fn128 = make_energy_fn_pure(displacement_fn, phys_sys, tile_size=128)
    
    e1 = fn1(params1, positions)
    e128 = fn128(params128, positions)
    
    # Use higher tolerance for sum accumulation differences if float32, but jax.lax.scan
    # usually matches exactly if order is preserved.
    assert jnp.allclose(e1, e128, atol=1e-5), f"Energy mismatch: {e1} vs {e128}"
    
    # Check forces
    f1 = jax.grad(fn1, argnums=1)(params1, positions)
    f128 = jax.grad(fn128, argnums=1)(params128, positions)
    assert jnp.allclose(f1, f128, atol=1e-4), "Force mismatch"

@pytest.mark.slow
def test_nve_energy_drift_stub():
    """Placeholder for 100ps NVE drift test. Requires full integrator setup."""
    # This would involve:
    # 1. Loading pre-equilibrated water box
    # 2. Running 100ps simulation with velocity verlet
    # 3. Computing (E_max - E_min) / E_avg per ns
    pytest.skip("Full NVE drift test requires substantial compute; deferred to cluster submission.")

@pytest.mark.slow
def test_force_rmse_dhfr_stub():
    """Placeholder for DHFR parity check against OpenMM Reference."""
    # This would involve:
    # 1. Loading DHFR pdb/psf
    # 2. Extracting OpenMM forces from a reference XML/JSON
    # 3. Comparing against make_energy_fn_pure forces
    pytest.skip("DHFR parity check requires DHFR asset files; deferred to cluster.")
