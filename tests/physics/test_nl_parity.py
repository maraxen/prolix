import jax
import jax.numpy as jnp
import numpy as np
from jax_md import space, partition
from prolix.physics.system import make_energy_fn_pure
from prolix.typing import PhysicsSystem, EnergyParams
import pytest

# XA-CI: API/physics drift or heavy compile — deselect from GitHub-faithful suite; tracked under XA-DRIFT.
pytestmark = pytest.mark.slow

def test_nl_dense_parity():
    # Setup a small system
    box_size = jnp.array([30.0, 30.0, 30.0])
    N = 256
    key = jax.random.key(42)
    pos = jax.random.uniform(key, (N, 3)) * 25.0
    
    # Random parameters
    charges = jax.random.normal(key, (N,)) * 0.1
    sigmas = jnp.ones(N) * 0.3
    epsilons = jnp.ones(N) * 0.1
    
    phys_sys = PhysicsSystem.from_dict({
        "charges": charges,
        "sigmas": sigmas,
        "epsilons": epsilons,
    }, positions=pos, box_size=box_size)
    
    disp_fn, _ = space.periodic(box_size)
    
    # 1. Dense Energy
    e_params_d, energy_fn_dense = make_energy_fn_pure(
        disp_fn, phys_sys, pme_alpha=0.34, pme_grid_spacing=1.0, tile_size=64
    )
    
    e_dense = energy_fn_dense(e_params_d, pos, neighbor=None)
    f_dense = -jax.grad(energy_fn_dense, argnums=1)(e_params_d, pos, neighbor=None)
    
    # 2. NL Energy
    neighbor_fn = partition.neighbor_list(disp_fn, box_size, r_cutoff=9.0, dr_threshold=1.0)
    nb = neighbor_fn.allocate(pos)
    
    e_params_nl, energy_fn_nl = make_energy_fn_pure(
        disp_fn, phys_sys, pme_alpha=0.34, pme_grid_spacing=1.0, tile_size=32
    )
    
    e_nl = energy_fn_nl(e_params_nl, pos, neighbor=nb)
    f_nl = -jax.grad(energy_fn_nl, argnums=1)(e_params_nl, pos, neighbor=nb)
    
    # Check Energy Parity (1e-5 relative)
    # Energy includes PME which has some grid error, but Direct-space should match.
    # Actually, we should check LJ and Direct Coulomb separately if we want 1e-5.
    # But let's check total energy first.
    rel_diff_e = abs(e_nl - e_dense) / (abs(e_dense) + 1e-6)
    print(f"Energy Dense: {e_dense:.6f}, NL: {e_nl:.6f}, Rel Diff: {rel_diff_e:.2e}")
    assert rel_diff_e < 1e-5
    
    # Check Force Parity
    f_rmse = jnp.sqrt(jnp.mean((f_nl - f_dense)**2))
    f_max = jnp.max(jnp.abs(f_dense))
    print(f"Force RMSE: {f_rmse:.2e}, Max Force: {f_max:.2e}")
    assert f_rmse < 1e-5 * (f_max + 1.0)

def test_nl_overflow_detection():
    from prolix.batched_simulate import make_langevin_step_nl_dynamic, LangevinState, PaddedSystem
    from jax_md import partition, space
    
    # Setup a system where overflow is likely
    N = 100
    box_size = jnp.array([10.0, 10.0, 10.0])
    pos = jnp.zeros((N, 3))
    
    disp_fn, shift_fn = space.free()
    
    # Use from_dict to get all mandatory fields
    phys_sys = PhysicsSystem.from_dict({
        "charges": jnp.zeros(N),
        "sigmas": jnp.ones(N) * 0.3,
        "epsilons": jnp.ones(N) * 0.1,
        "bonds": jnp.zeros((0, 2), dtype=jnp.int32),
        "bond_params": jnp.zeros((0, 2)),
        "angles": jnp.zeros((0, 3), dtype=jnp.int32),
        "angle_params": jnp.zeros((0, 2)),
        "dihedrals": jnp.zeros((0, 4), dtype=jnp.int32),
        "dihedral_params": jnp.zeros((0, 3)),
        "impropers": jnp.zeros((0, 4), dtype=jnp.int32),
        "improper_params": jnp.zeros((0, 3)),
        "urey_bradley_bonds": jnp.zeros((0, 2), dtype=jnp.int32),
        "urey_bradley_params": jnp.zeros((0, 2)),
        "cmap_torsions": jnp.zeros((0, 5), dtype=jnp.int32),
        "excl_indices": jnp.full((N, 1), -1, dtype=jnp.int32),
        "excl_scales_vdw": jnp.ones((N, 1)),
        "excl_scales_elec": jnp.ones((N, 1)),
        "constraint_pairs": jnp.zeros((0, 2), dtype=jnp.int32),
        "constraint_lengths": jnp.zeros((0,)),
    }, positions=pos, box_size=box_size)
    
    state = LangevinState(
        positions=pos,
        momentum=jnp.zeros_like(pos),
        force=jnp.zeros_like(pos),
        mass=jnp.ones(N),
        key=jax.random.key(0),
        cap_count=jnp.array(0)
    )
    
    step_fn = make_langevin_step_nl_dynamic(0.001, 1.0, 1.0)
    
    # Move atoms to trigger overflow (K will increase)
    # Actually, JAX-MD NeighborList.update() checks if max_occupancy is exceeded.
    # In partition.Dense, occupancy is always N-1? No, it's the number of neighbors within cutoff.
    # If we move all atoms to the origin, occupancy = N-1.
    # If we allocated with small capacity, it will overflow.
    
    # Let's just manually trigger it by setting a very small capacity if possible.
    # Or just use a very large cutoff.
    neighbor_fn_tight = partition.neighbor_list(disp_fn, box_size, r_cutoff=1.0, dr_threshold=0.1, capacity_multiplier=1.0)
    # Allocate with few atoms far apart
    pos_far = jnp.arange(N).reshape(-1, 1) * jnp.array([2.0, 0, 0])
    nbrs_tight = neighbor_fn_tight.allocate(pos_far)
    print(f"Initial occupancy: {nbrs_tight.max_occupancy}")
    
    # Now move them all to 0
    pos_near = jnp.zeros((N, 3))
    new_state = state.replace(positions=pos_near, last_update_positions=pos_far)
    
    # Run step
    final_state, final_nbrs = step_fn(phys_sys, new_state, nbrs_tight)
    
    print(f"Did overflow: {final_state.did_overflow}")
    assert final_state.did_overflow == True
