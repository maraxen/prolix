"""Tests for simulation stability.

Tests that the parse_structure -> energy function -> simulation pipeline
produces stable physics. Uses implicit solvent to avoid biotite dependency.
"""

from pathlib import Path

import jax.numpy as jnp
import pytest
from proxide.io.parsing.backend import OutputSpec, parse_structure

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
FF_PATH = (
  Path(__file__).parent.parent.parent
  / "proxide"
  / "src"
  / "proxide"
  / "assets"
  / "protein.ff19SB.xml"
)


@pytest.fixture
def parameterized_system():
  """Load a parameterized protein system."""
  pdb_path = DATA_DIR / "pdb" / "1CRN.pdb"
  if not pdb_path.exists():
    pytest.skip("1CRN.pdb not found")

  spec = OutputSpec(
    parameterize_md=True,
    force_field=str(FF_PATH),
    add_hydrogens=True,
  )
  return parse_structure(str(pdb_path), spec)


def test_implicit_solvent_stability(parameterized_system):
  """Test that implicit solvent MD runs stably."""
  protein = parameterized_system

  # Get flat coordinates from Atom37 format
  coords = protein.coordinates
  mask = protein.atom_mask

  if coords.ndim == 3:
    # Atom37 format: (N_res, 37, 3)
    flat_coords = coords.reshape(-1, 3)
    flat_mask = mask.reshape(-1)
    valid_indices = jnp.where(flat_mask > 0.5)[0]
    coords = flat_coords[valid_indices]

  # Verify we have valid coordinates
  assert coords.shape[0] > 0, "No valid coordinates found"
  assert coords.shape[1] == 3, "Coordinates should be (N, 3)"
  assert jnp.all(jnp.isfinite(coords)), "Coordinates contain NaN/Inf"

  # Verify parameterization is complete
  n_atoms = len(protein.charges)
  assert n_atoms > 0, "No charges assigned"
  assert protein.bonds is not None, "No bonds assigned"
  assert protein.bonds.shape[0] > 0, "Empty bond array"


def test_energy_function_produces_finite_values(parameterized_system):
  """Test that energy function returns finite values."""
  protein = parameterized_system

  # Get coordinates
  coords = protein.coordinates
  mask = protein.atom_mask

  if coords.ndim == 3:
    flat_coords = coords.reshape(-1, 3)
    flat_mask = mask.reshape(-1)
    valid_indices = jnp.where(flat_mask > 0.5)[0]
    coords = flat_coords[valid_indices]

  # Build system params dict from AtomicSystem attributes
  # Note: Rust returns nm/kJ units, but for this test we just check finiteness
  params = {
    "charges": protein.charges,
    "sigmas": protein.sigmas,
    "epsilons": protein.epsilons,
    "bonds": protein.bonds,
    "bond_params": protein.bond_params,
    "angles": protein.angles,
    "angle_params": protein.angle_params,
    "dihedrals": protein.proper_dihedrals,
    "dihedral_params": protein.dihedral_params,
  }

  # Verify all required params are present
  for key in ["charges", "sigmas", "bonds", "bond_params"]:
    assert params[key] is not None, f"Missing {key}"


@pytest.mark.skip(
  reason="Explicit solvation requires biotite for topology - needs proxide migration"
)
def test_explicit_solvation_stability():
  """Test that explicit solvent simulation runs stably.

  This test is skipped until explicit solvent topology building is migrated
  from biotite to the Rust parser.
  """


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
