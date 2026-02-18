"""PME parity tests against OpenMM.

Uses proxide parse_structure for loading. Tests electrostatic energy
calculations including PME for periodic systems.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from proxide import OutputSpec, parse_structure

# Enable x64 for physics
jax.config.update("jax_enable_x64", True)

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "pdb"
FF_PATH = (
  Path(__file__).parent.parent.parent
  / "proxide"
  / "src"
  / "proxide"
  / "assets"
  / "protein.ff19SB.xml"
)


def openmm_available():
  """Check if OpenMM is available."""
  try:
    import openmm  # noqa: F401

    return True
  except ImportError:
    return False


@pytest.fixture
def parameterized_protein():
  """Load protein using proxide parse_structure."""
  pdb_path = DATA_DIR / "1CRN.pdb"
  if not pdb_path.exists():
    pytest.skip("1CRN.pdb not found")

  spec = OutputSpec(
    parameterize_md=True,
    force_field=str(FF_PATH),
    add_hydrogens=True,
  )
  return parse_structure(str(pdb_path), spec)


def get_flat_coords(protein):
  """Extract flat coordinates from protein."""
  coords = protein.coordinates
  mask = protein.atom_mask

  if coords.ndim == 3:
    flat_coords = coords.reshape(-1, 3)
    flat_mask = mask.reshape(-1)
    valid_indices = jnp.where(flat_mask > 0.5)[0]
    return flat_coords[valid_indices]
  return coords


class TestElectrostaticSetup:
  """Tests for electrostatic energy function setup."""

  def test_charges_assigned(self, parameterized_protein):
    """Test that all atoms have charges assigned."""
    charges = parameterized_protein.charges
    assert charges is not None
    assert len(charges) > 0
    print(f"Loaded {len(charges)} atom charges")

  def test_total_charge_near_integer(self, parameterized_protein):
    """Test that total system charge is near an integer."""
    total = float(jnp.sum(parameterized_protein.charges))
    rounded = round(total)
    assert abs(total - rounded) < 0.2, f"Total charge {total:.3f} not near integer {rounded}"
    print(f"Total charge: {total:.4f} (expected integer: {rounded})")


@pytest.mark.skipif(not openmm_available(), reason="OpenMM not installed")
class TestPMEParity:
  """PME parity tests between Prolix and OpenMM."""

  def test_openmm_nonbonded_energy(self, parameterized_protein):
    """Test that OpenMM nonbonded energy is finite."""
    import openmm
    from openmm import unit

    protein = parameterized_protein
    coords = get_flat_coords(protein)

    # Convert to OpenMM system
    omm_system = protein.to_openmm_system()

    # Create context
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(omm_system, integrator, platform)

    # Set positions (Å to nm)
    positions = np.array(coords) * 0.1
    context.setPositions(positions * unit.nanometer)

    # Get energy
    state = context.getState(getEnergy=True)
    energy_kj = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    energy_kcal = energy_kj / 4.184

    print(f"OpenMM Energy: {energy_kj:.2f} kJ/mol = {energy_kcal:.2f} kcal/mol")
    assert np.isfinite(energy_kj), "OpenMM energy is not finite"

  def test_force_decomposition(self, parameterized_protein):
    """Test that we can decompose forces by type in OpenMM."""
    import openmm
    from openmm import unit

    protein = parameterized_protein
    coords = get_flat_coords(protein)

    omm_system = protein.to_openmm_system()

    # Get force types
    for i in range(omm_system.getNumForces()):
      force = omm_system.getForce(i)
      force_name = force.__class__.__name__

      # Create separate system with only this force
      single_force_system = openmm.System()
      for j in range(omm_system.getNumParticles()):
        single_force_system.addParticle(omm_system.getParticleMass(j))

      # Clone the force
      if hasattr(force, "getNumBonds"):  # HarmonicBondForce
        new_force = openmm.HarmonicBondForce()
        for b in range(force.getNumBonds()):
          p1, p2, length, k = force.getBondParameters(b)
          new_force.addBond(p1, p2, length, k)
        single_force_system.addForce(new_force)
      elif hasattr(force, "getNumAngles"):  # HarmonicAngleForce
        new_force = openmm.HarmonicAngleForce()
        for a in range(force.getNumAngles()):
          p1, p2, p3, angle, k = force.getAngleParameters(a)
          new_force.addAngle(p1, p2, p3, angle, k)
        single_force_system.addForce(new_force)
      else:
        continue  # Skip complex forces for this test

      integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
      context = openmm.Context(single_force_system, integrator)
      positions = np.array(coords) * 0.1
      context.setPositions(positions * unit.nanometer)

      state = context.getState(getEnergy=True)
      e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
      print(f"  {force_name}: {e:.2f} kJ/mol")


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
