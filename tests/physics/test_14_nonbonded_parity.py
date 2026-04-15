"""Parity test for 1-4 nonbonded scaling in explicit solvent.

Validates that 1-4 Coulomb and LJ scaling factors are correctly applied
within Prolix's energy functions (both dense and neighbor-list paths)
by comparing against a minimal 4-atom OpenMM system.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_md import space, partition

from prolix.physics import system, neighbor_list as nl

# Enable x64 for physics precision
jax.config.update("jax_enable_x64", True)

def openmm_available():
  """Check if OpenMM is available."""
  try:
    import openmm  # noqa: F401
    return True
  except ImportError:
    return False

@pytest.mark.integration
@pytest.mark.skipif(not openmm_available(), reason="OpenMM not installed")
class Test14NonbondedParity:
  """Tests 1-4 scaling parity against OpenMM."""

  @pytest.fixture
  def simple_14_system(self):
    """Create a minimal 4-atom system with 1-4 interactions."""
    import openmm
    from openmm import app, unit

    # 4 atoms in a line: 1-2-3-4
    # 1-4 interaction between atom 0 and 3
    n_atoms = 4
    omm_system = openmm.System()
    for _ in range(n_atoms):
        omm_system.addParticle(12.0 * unit.amu) # Carbon-like mass

    # Nonbonded
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    
    # Standard parameters (Carbon-like)
    q = 0.5
    sig = 3.4
    eps = 0.15
    for _ in range(n_atoms):
        nb.addParticle(q, sig / 10.0, eps * 4.184) # nm, kJ/mol
    
    # Exclusions & 1-4 scaling
    # 1-2: (0,1), (1,2), (2,3) -> Scale 0
    # 1-3: (0,2), (1,3) -> Scale 0
    # 1-4: (0,3) -> Scale 0.5 (Elec) and 0.5 (VDW)
    nb.addException(0, 1, 0.0, 1.0, 0.0)
    nb.addException(1, 2, 0.0, 1.0, 0.0)
    nb.addException(2, 3, 0.0, 1.0, 0.0)
    nb.addException(0, 2, 0.0, 1.0, 0.0)
    nb.addException(1, 3, 0.0, 1.0, 0.0)
    
    scale_elec = 0.8333333333333334 # Amber 1/1.2
    scale_vdw = 0.5 # Amber 1/2.0
    nb.addException(0, 3, (q*q)*scale_elec, (sig/10.0), (eps*4.184)*scale_vdw)
    
    omm_system.addForce(nb)
    
    # Positions: 1-2-3-4 zig-zag
    pos = np.array([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [1.5, 1.5, 0.0],
        [0.0, 1.5, 0.0]
    ]) # Å
    
    return {
        "system": omm_system,
        "positions": pos,
        "charges": np.array([q]*n_atoms),
        "sigmas": np.array([sig]*n_atoms),
        "epsilons": np.array([eps]*n_atoms),
        "scale_elec": scale_elec,
        "scale_vdw": scale_vdw,
        "idx_12_13": np.array([[0,1], [1,2], [2,3], [0,2], [1,3]]),
        "idx_14": np.array([[0,3]])
    }

  def test_14_dense_parity(self, simple_14_system):
    """Verify 1-4 scaling in dense N^2 path."""
    import openmm
    from openmm import unit

    data = simple_14_system
    
    # 1. OpenMM Energy
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    context = openmm.Context(data["system"], integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions((data["positions"] * 0.1) * unit.nanometer)
    omm_energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

    # 2. Prolix Energy
    displacement_fn, _ = space.free()
    
    exclusion_spec = nl.ExclusionSpec(
        n_atoms=4,
        idx_12_13=jnp.array(data["idx_12_13"]),
        idx_14=jnp.array(data["idx_14"]),
        scale_14_elec=data["scale_elec"],
        scale_14_vdw=data["scale_vdw"]
    )
    
    params = {
        "charges": jnp.array(data["charges"]),
        "sigmas": jnp.array(data["sigmas"]),
        "epsilons": jnp.array(data["epsilons"]),
        "exclusion_mask": None,
        "bonds": jnp.zeros((0, 2), dtype=jnp.int32),
        "bond_params": jnp.zeros((0, 2)),
        "angles": jnp.zeros((0, 3), dtype=jnp.int32),
        "angle_params": jnp.zeros((0, 2)),
        "proper_dihedrals": jnp.zeros((0, 4), dtype=jnp.int32),
        "dihedral_params": jnp.zeros((0, 3)),
        "impropers": jnp.zeros((0, 4), dtype=jnp.int32),
        "improper_params": jnp.zeros((0, 3)),
    }
    
    energy_fn = system.make_energy_fn(
        displacement_fn,
        params,
        exclusion_spec=exclusion_spec,
        implicit_solvent=False,
        use_pbc=False,
        strict_parameterization=False
    )
    
    jax_energy = float(energy_fn(jnp.array(data["positions"])))
    
    print(f"\nOpenMM Energy: {omm_energy:.6f}")
    print(f"Prolix Energy: {jax_energy:.6f}")
    
    assert np.isclose(omm_energy, jax_energy, atol=1e-4)

  def test_14_neighbor_list_parity(self, simple_14_system):
    """Verify 1-4 scaling in sparse neighbor-list path."""
    import openmm
    from openmm import unit

    data = simple_14_system
    
    # 1. OpenMM Energy
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    context = openmm.Context(data["system"], integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions((data["positions"] * 0.1) * unit.nanometer)
    omm_energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

    # 2. Prolix Energy
    displacement_fn, _ = space.free()
    
    exclusion_spec = nl.ExclusionSpec(
        n_atoms=4,
        idx_12_13=jnp.array(data["idx_12_13"]),
        idx_14=jnp.array(data["idx_14"]),
        scale_14_elec=data["scale_elec"],
        scale_14_vdw=data["scale_vdw"]
    )
    
    params = {
        "charges": jnp.array(data["charges"]),
        "sigmas": jnp.array(data["sigmas"]),
        "epsilons": jnp.array(data["epsilons"]),
        "exclusion_mask": None,
        "bonds": jnp.zeros((0, 2), dtype=jnp.int32),
        "bond_params": jnp.zeros((0, 2)),
        "angles": jnp.zeros((0, 3), dtype=jnp.int32),
        "angle_params": jnp.zeros((0, 2)),
        "proper_dihedrals": jnp.zeros((0, 4), dtype=jnp.int32),
        "dihedral_params": jnp.zeros((0, 3)),
        "impropers": jnp.zeros((0, 4), dtype=jnp.int32),
        "improper_params": jnp.zeros((0, 3)),
    }
    
    # Force Prolix to use sparse exclusions by providing a neighbor list
    neighbor_list_fn = partition.neighbor_list(displacement_fn, box=100.0, r_cutoff=5.0)
    nbr = neighbor_list_fn.allocate(jnp.array(data["positions"]))
    
    energy_fn = system.make_energy_fn(
        displacement_fn,
        params,
        neighbor_list=nbr,
        exclusion_spec=exclusion_spec,
        implicit_solvent=False,
        use_pbc=False,
        strict_parameterization=False
    )
    
    jax_energy = float(energy_fn(jnp.array(data["positions"]), neighbor=nbr))
    
    print(f"\nOpenMM Energy: {omm_energy:.6f}")
    print(f"Prolix Energy (NL): {jax_energy:.6f}")
    
    assert np.isclose(omm_energy, jax_energy, atol=1e-4)

if __name__ == "__main__":
  pytest.main([__file__, "-v"])
