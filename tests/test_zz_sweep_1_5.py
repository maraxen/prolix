import os
import jax
import jax.numpy as jnp
import pytest
from jax.test_util import check_grads
from proxide.io.parsing.backend import parse_structure, OutputSpec
from proxide import CoordFormat
from prolix.physics import system
from jax_md import space

# NOTE: do NOT set jax_enable_x64, jax_debug_nans, or disable_jit at module
# level — these are irreversible global state and contaminate all subsequent
# tests in the pytest session.  Scoped inside the test function below.


@pytest.mark.order("last")
def test_sweep_1_5_masked_eager_gradients():
    # Enable x64, NaN debugging and disable JIT for this test only.
    # WARNING: these are irreversible within a pytest session —
    # that is why this test is marked order("last").
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_debug_nans", True)
    jax.disable_jit()

    print("Testing Sweep 1.5 Protocol...")
    
    # 1. Parse a real protein
    pdb_path = "tests/data/test_ala.pdb"
    ff_path = "proxide/src/proxide/assets/amber/protein.ff19SB.xml"
    spec = OutputSpec(parameterize_md=True, force_field=ff_path, coord_format=CoordFormat.Full)
    protein = parse_structure(pdb_path, spec)
    
    # 2. Power-of-two padding (Sweep 1.5 Protocol requirement)
    # Pad to N=32 atoms
    N_real = protein.coordinates.shape[0]
    N_pad = 32
    
    padded_coords = jnp.zeros((N_pad, 3))
    padded_coords = padded_coords.at[:N_real].set(protein.coordinates)
    
    # Create padding mask for ExclusionSpec
    # Pad the original mask with 0.0 for interactions involving padded atoms
    exclusion_mask = jnp.zeros((N_pad, N_pad))
    if protein.exclusion_mask is not None:
        exclusion_mask = exclusion_mask.at[:N_real, :N_real].set(protein.exclusion_mask)
    else:
        # Fallback to (1 - eye) if no original mask
        exclusion_mask = exclusion_mask.at[:N_real, :N_real].set(1.0 - jnp.eye(N_real))
    
    import dataclasses
    protein = dataclasses.replace(
        protein,
        exclusion_mask=exclusion_mask,
        charges=jnp.pad(protein.charges, (0, N_pad - N_real)),
        sigmas=jnp.pad(protein.sigmas, (0, N_pad - N_real), constant_values=1.0),
        epsilons=jnp.pad(protein.epsilons, (0, N_pad - N_real))
    )
    
    # Empty bonds, angles etc for simplicity in gradient testing of non-bonded
    # (since bonded terms are usually safe, but we test the whole energy_fn anyway)
    
    displacement_fn, _ = space.free()
    
    energy_fn = system.make_energy_fn(
        displacement_fn=displacement_fn,
        system=protein,
        implicit_solvent=False,  # Test LJ and Coulomb directly
        use_pbc=False
    )
    
    def loss_wrapper(coords):
        return energy_fn(coords)
    
    # 3. Check gradients using jax.test_util.check_grads
    print("Running check_grads...")
    check_grads(loss_wrapper, (padded_coords,), order=1, modes=['rev'])
    print("Sweep 1.5 Protocol successful. No NaNs detected in backward pass.")

if __name__ == "__main__":
    test_sweep_1_5_masked_eager_gradients()
