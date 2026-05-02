import jax
import jax.numpy as jnp
import pytest
import equinox as eqx
from prolix.physics.types import PhysicsSystem, EnergyParams
from prolix.physics.optimization import StaticBakingWrapper, chunked_lj_energy, chunked_coulomb_energy

def test_chunked_coulomb_parity():
    n_atoms = 2
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0]
    ])
    charges = jnp.array([1.0, -1.0])
    
    displacement_fn, _ = space.free()
    pme_alpha = 0.34
    C = 332.0637
    
    # Baseline
    def dense_coul(r):
        dr_mat = space.map_product(displacement_fn)(r, r)
        dist = space.distance(jnp.asarray(dr_mat))
        mask_diag = (1.0 - jnp.eye(n_atoms)).astype(bool)
        q_ij = charges[:, None] * charges[None, :]
        dist_safe = jnp.where(mask_diag, dist, 1.0)
        erfc_term = jax.scipy.special.erfc(pme_alpha * dist_safe)
        e_pair = C * (q_ij / dist_safe) * erfc_term
        return 0.5 * jnp.sum(jnp.where(mask_diag, e_pair, 0.0))

    res_dense = dense_coul(positions)
    res_chunked = chunked_coulomb_energy(positions, charges, displacement_fn, pme_alpha, C, chunk_size=32)
    
    assert jnp.allclose(res_dense, res_chunked, atol=1e-5)
    
    grad_dense = jax.grad(dense_coul)(positions)
    grad_chunked = jax.grad(lambda r: chunked_coulomb_energy(r, charges, displacement_fn, pme_alpha, C, chunk_size=32))(positions)
    
    assert jnp.allclose(grad_dense, grad_chunked, atol=1e-4)
from jax_md import space

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
        dr_mat = space.map_product(displacement_fn)(r, r)
        dist = space.distance(jnp.asarray(dr_mat))
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
    res_chunked = chunked_lj_energy(positions, sigmas, epsilons, displacement_fn, chunk_size=32)
    
    assert jnp.allclose(res_dense, res_chunked, atol=1e-5)
    
    # Gradient parity (Analytical vs Autodiff)
    grad_dense = jax.grad(dense_lj)(positions)
    grad_chunked = jax.grad(lambda r: chunked_lj_energy(r, sigmas, epsilons, displacement_fn, chunk_size=32))(positions)
    
    assert jnp.allclose(grad_dense, grad_chunked, atol=1e-4)

def test_static_field_baking():
    # Setup a dummy system
    n_atoms = 10
    positions = jnp.zeros((n_atoms, 3))
    charges = jnp.ones(n_atoms)
    sigmas = jnp.ones(n_atoms)
    epsilons = jnp.ones(n_atoms)
    
    # Static fields
    radii = jnp.array([1.5] * n_atoms)
    atom_mask = jnp.ones(n_atoms, dtype=bool)
    
    system = PhysicsSystem(
        positions=positions,
        charges=charges,
        sigmas=sigmas,
        epsilons=epsilons,
        radii=radii,
        scaled_radii=radii,
        masses=jnp.ones(n_atoms),
        element_ids=jnp.zeros(n_atoms, dtype=jnp.int32),
        atom_mask=atom_mask,
        is_hydrogen=jnp.zeros(n_atoms, dtype=bool),
        is_backbone=jnp.zeros(n_atoms, dtype=bool),
        is_heavy=jnp.ones(n_atoms, dtype=bool),
        protein_atom_mask=jnp.ones(n_atoms, dtype=bool),
        water_atom_mask=jnp.zeros(n_atoms, dtype=bool),
        bonds=jnp.zeros((0, 2), dtype=jnp.int32),
        bond_params=jnp.zeros((0, 2)),
        bond_mask=jnp.zeros(0, dtype=bool),
        angles=jnp.zeros((0, 3), dtype=jnp.int32),
        angle_params=jnp.zeros((0, 2)),
        angle_mask=jnp.zeros(0, dtype=bool),
        dihedrals=jnp.zeros((0, 4), dtype=jnp.int32),
        dihedral_params=jnp.zeros((0, 3)),
        dihedral_mask=jnp.zeros(0, dtype=bool),
        impropers=jnp.zeros((0, 4), dtype=jnp.int32),
        improper_params=jnp.zeros((0, 3)),
        improper_mask=jnp.zeros(0, dtype=bool),
        nonbonded_cutoff=9.0
    )
    
    params = EnergyParams(params={'charges': charges, 'sigmas': sigmas, 'epsilons': epsilons})
    
    # Dummy energy function
    def dummy_energy(p: EnergyParams, s: PhysicsSystem):
        # Uses both dynamic and static fields
        return jnp.sum(s.positions) + jnp.sum(s.radii)
    
    # Bake static fields
    wrapper = StaticBakingWrapper(dummy_energy, system, ("positions", "charges", "sigmas", "epsilons"))
    
    # Check that it works
    res = wrapper(params, system)
    assert res == jnp.sum(system.positions) + jnp.sum(system.radii)
    
    # Check that radii is NOT in the leaves of the wrapper's dynamic part 
    # (actually StaticBakingWrapper itself marks static_struct as static)
    
    leaves = jax.tree_util.tree_leaves(wrapper)
    # Radii should NOT be in leaves because it's in static_struct which is marked as static
    # Wait, eqx.field(static=True) only works for Module attributes.
    # In StaticBakingWrapper, static_struct is marked static=True.
    
    # Verify by checking if JIT compiles it away
    @jax.jit
    def jitted_res(p, s):
        return wrapper(p, s)
        
    # If we change system.radii, jitted_res should NOT change because it's baked
    system_new = eqx.tree_at(lambda s: s.radii, system, radii * 2)
    res_new = jitted_res(params, system_new)
    assert res_new == res # Still original value!
    
    # Check jax.export compatibility
    # jax.export requires a jitted function
    export_fn = jax.jit(lambda p, s: wrapper(p, s))
    lowered = export_fn.lower(params, system)
    assert lowered is not None

if __name__ == "__main__":
    test_static_field_baking()
