import jax
import jax.numpy as jnp
import os
import pathlib
import pytest
from prolix.physics import pbc, system, types
from prolix.physics.system import make_energy_fn_pure
from prolix.export import export_energy_fn, export_langevin_step, save_artifact, load_artifact

def test_export_roundtrip():
    """Verify that energy functions can be exported, saved, and re-loaded as MLIR."""
    # 1. Setup a simple Argon system
    n_atoms = 8
    positions = jax.random.uniform(jax.random.key(42), (n_atoms, 3)) * 10.0
    box_vec = jnp.array([15.0, 15.0, 15.0])
    
    sys_dict = {
        "charges": jnp.zeros(n_atoms),
        "sigmas": jnp.full(n_atoms, 3.405),
        "epsilons": jnp.full(n_atoms, 0.238),
        "bonds": jnp.zeros((0, 2), dtype=jnp.int32),
        "bond_params": jnp.zeros((0, 2), dtype=jnp.float32),
        "angles": jnp.zeros((0, 3), dtype=jnp.int32),
        "angle_params": jnp.zeros((0, 2), dtype=jnp.float32),
        "proper_dihedrals": jnp.zeros((0, 4), dtype=jnp.int32),
        "dihedral_params": jnp.zeros((0, 3), dtype=jnp.float32),
        "impropers": jnp.zeros((0, 4), dtype=jnp.int32),
        "improper_params": jnp.zeros((0, 3), dtype=jnp.float32),
    }
    
    displacement_fn, _ = pbc.create_periodic_space(box_vec)
    physics_system = system.PhysicsSystem.from_dict(sys_dict, positions, box_vec)
    
    params, fn_pure = make_energy_fn_pure(displacement_fn, physics_system)
    
    # 2. Get baseline energy
    e_baseline = float(fn_pure(params, positions))
    
    # 3. Export to StableHLO
    lowered = export_energy_fn(fn_pure, params, positions)
    
    # 4. Save and Load Artifact
    tmp_path = "outputs/tests/argon_energy.mlir"
    save_artifact(lowered, tmp_path)
    mlir_text = load_artifact(tmp_path)
    
    assert "module" in mlir_text
    assert "func.func" in mlir_text
    
    # 5. Compile and Verify
    compiled = lowered.compile()
    e_compiled = float(compiled(params, positions))
    
    assert abs(e_baseline - e_compiled) < 1e-6
    
    # Clean up
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

def test_export_langevin_step_raises():
    """Verify that export_langevin_step raises NotImplementedError in v1.1."""
    with pytest.raises(NotImplementedError, match="deferred to v1.2"):
        export_langevin_step(None, None)
