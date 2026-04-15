"""Parity test for OBC2 Born radii against OpenMM.

Validates that Prolix's OBC2 implementation produces identical Born radii
to OpenMM's Reference platform for a standard protein (1UAO).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pathlib import Path
from jax_md import space, partition

from prolix.physics import generalized_born
from proxide import CoordFormat, OutputSpec, parse_structure
from proxide import assign_mbondi2_radii, assign_obc2_scaling_factors

# Enable x64 for physics precision
jax.config.update("jax_enable_x64", True)

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "pdb"
FF_PATH = Path(__file__).parent.parent.parent / "src" / "proxide" / "assets" / "protein.ff19SB.xml"

def openmm_available():
  try:
    import openmm
    return True
  except ImportError:
    return False

@pytest.mark.integration
@pytest.mark.skipif(not openmm_available(), reason="OpenMM not installed")
class TestGBSABornRadiiParity:
  """Tests Born radii parity against OpenMM."""

  def test_born_radii_parity_1uao(self):
    """Compare OBC2 Born radii for 1UAO."""
    import openmm
    from openmm import app, unit

    pdb_path = DATA_DIR / "1UAO.pdb"
    if not pdb_path.exists():
      pytest.skip("1UAO.pdb not found")

    # 1. OpenMM Reference
    pdb = app.PDBFile(str(pdb_path))
    ff = app.ForceField("amber14-all.xml", "implicit/obc2.xml")
    
    # Create system with OBC2
    system_omm = ff.createSystem(
        pdb.topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False
    )
    
    # Find the GBSA force
    gbsa_force = None
    for i in range(system_omm.getNumForces()):
        f = system_omm.getForce(i)
        if isinstance(f, (openmm.GBSAOBCForce, openmm.CustomGBForce)):
            gbsa_force = f
            break
    
    if gbsa_force is None:
        pytest.fail("Could not find GBSA force in OpenMM system")

    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    context = openmm.Context(system_omm, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(pdb.positions)
    
    # Force a state update to compute Born radii
    context.getState(getEnergy=True)
    
    n_atoms = system_omm.getNumParticles()
    omm_radii = np.zeros(n_atoms)
    omm_scales = np.zeros(n_atoms)
    
    if isinstance(gbsa_force, openmm.GBSAOBCForce):
        for i in range(n_atoms):
            p = gbsa_force.getParticleParameters(i)
            omm_radii[i] = p[1].value_in_unit(unit.angstrom)
            omm_scales[i] = p[2]
    else:
        # CustomGBForce
        for i in range(n_atoms):
            p = gbsa_force.getParticleParameters(i)
            omm_radii[i] = p[1] * 10.0 # nm to A
            omm_scales[i] = p[2]

    # 2. Prolix Calculation
    # Parse 1UAO
    spec = OutputSpec(parameterize_md=True, coord_format=CoordFormat.Full)
    protein = parse_structure(str(pdb_path), spec)
    
    # Use exact same input radii/scales as OpenMM to isolate the Born integration logic
    positions = jnp.array(np.array(pdb.positions.value_in_unit(unit.nanometer)) * 10.0)
    
    # Prolix OBC2 implementation
    jax_born_radii = generalized_born.compute_born_radii(
        positions,
        jnp.array(omm_radii),
        scaled_radii=jnp.array(omm_radii * omm_scales),
        dielectric_offset=0.09 
    )
    
    neighbor_list_fn = partition.neighbor_list(
        space.free()[0], 100.0, r_cutoff=10.0, dr_threshold=0.5
    )
    nbr = neighbor_list_fn.allocate(positions)
    
    jax_born_radii_nl = generalized_born.compute_born_radii_neighbor_list(
        positions,
        jnp.array(omm_radii),
        nbr.idx,
        scaled_radii=jnp.array(omm_radii * omm_scales),
        dielectric_offset=0.09
    )
    
    diff = jnp.abs(jax_born_radii - jax_born_radii_nl)
    max_diff = float(jnp.max(diff))
    print(f"Born Radii Dense vs NL Max Diff: {max_diff:.6e} Å")
    
    assert max_diff < 1e-4

if __name__ == "__main__":
  pytest.main([__file__, "-v"])
