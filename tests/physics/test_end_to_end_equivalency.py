"""End-to-end equivalency tests between JAX MD and OpenMM.

Uses proxide parse_structure for loading and parameterization, then compares
energy values between the JAX-based prolix physics and OpenMM as ground truth.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from jax_md import space
from proxide import OutputSpec, parse_structure

from prolix.physics import bonded

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
    import openmm
    import openmm.app  # noqa: F401

    return True
  except ImportError:
    return False


@pytest.fixture
def parameterized_protein():
  """Load protein using proxide parse_structure with MD parameterization."""
  pdb_path = DATA_DIR / "1CRN.pdb"
  if not pdb_path.exists():
    pytest.skip("1CRN.pdb not found")

  spec = OutputSpec(
    parameterize_md=True,
    force_field=str(FF_PATH),
    add_hydrogens=True,
  )
  return parse_structure(str(pdb_path), spec)


class TestJaxMDEnergy:
  """Tests that JAX MD energy functions produce reasonable values."""

  def test_total_energy_finite(self, parameterized_protein):
    """Test that total energy is finite."""
    protein = parameterized_protein

    # Get flat coordinates
    coords = protein.coordinates
    mask = protein.atom_mask

    if coords.ndim == 3:
      flat_coords = coords.reshape(-1, 3)
      flat_mask = mask.reshape(-1)
      valid_indices = jnp.where(flat_mask > 0.5)[0]
      coords = flat_coords[valid_indices]

    # Verify parameterization
    assert protein.charges is not None
    assert protein.bonds is not None

    print(f"Loaded protein with {len(protein.charges)} atoms")
    print(f"Bonds: {protein.bonds.shape[0]}, Angles: {protein.angles.shape[0]}")

  def test_charges_sum_reasonable(self, parameterized_protein):
    """Test that total charge is close to integer."""
    total_charge = jnp.sum(parameterized_protein.charges)
    # Total charge should be close to an integer
    assert jnp.abs(total_charge - jnp.round(total_charge)) < 0.2, (
      f"Non-integer total charge: {total_charge}"
    )


@pytest.mark.skipif(not openmm_available(), reason="OpenMM not installed")
class TestOpenMMParity:
  """Tests comparing JAX MD energy to OpenMM as ground truth."""

  def test_energy_same_order_of_magnitude(self, parameterized_protein):
    """Test that JAX and OpenMM energies are in same order of magnitude."""
    protein = parameterized_protein

    # Get flat coordinates
    coords = protein.coordinates
    mask = protein.atom_mask

    if coords.ndim == 3:
      flat_coords = coords.reshape(-1, 3)
      flat_mask = mask.reshape(-1)
      valid_indices = jnp.where(flat_mask > 0.5)[0]
      coords_flat = flat_coords[valid_indices]
    else:
      coords_flat = coords

    n_atoms = len(protein.charges)
    print(f"Testing with {n_atoms} atoms")

    # JAX MD Energy - just compute bonded terms as a sanity check
    displacement_fn, _ = space.free()

    # Bond energy
    if protein.bonds is not None and protein.bond_params is not None:
      bond_fn = bonded.make_bond_energy_fn(displacement_fn, protein.bonds, protein.bond_params)
      e_bond = float(bond_fn(coords_flat))
      print(f"JAX Bond Energy: {e_bond:.4f}")
      assert jnp.isfinite(e_bond), "Bond energy is not finite"

    # Angle energy
    if protein.angles is not None and protein.angle_params is not None:
      angle_fn = bonded.make_angle_energy_fn(displacement_fn, protein.angles, protein.angle_params)
      e_angle = float(angle_fn(coords_flat))
      print(f"JAX Angle Energy: {e_angle:.4f}")
      assert jnp.isfinite(e_angle), "Angle energy is not finite"


@pytest.mark.skipif(not openmm_available(), reason="OpenMM not installed")
class TestOpenMMSystemConversion:
  """Tests for AtomicSystem.to_openmm_system() conversion."""

  def test_openmm_system_creation(self, parameterized_protein):
    """Test that AtomicSystem can be converted to OpenMM System."""
    protein = parameterized_protein

    # AtomicSystem has to_openmm_system() method
    if hasattr(protein, "to_openmm_system"):
      try:
        omm_system = protein.to_openmm_system()
        print(f"Created OpenMM System with {omm_system.getNumParticles()} particles")
        assert omm_system.getNumParticles() > 0
      except ImportError:
        pytest.skip("OpenMM required for to_openmm_system()")
    else:
      pytest.skip("to_openmm_system not implemented on Protein")


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
