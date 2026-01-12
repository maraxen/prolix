"""Tests for ligand and LJ parameter validation.

Tests that the Rust-based parse_structure correctly assigns LJ parameters
to molecular systems. GAFF-specific tests are skipped until proxide supports
GAFF charge assignment.
"""

from pathlib import Path

import jax.numpy as jnp
import pytest
from proxide.io.parsing.rust import OutputSpec, parse_structure

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


class TestGAFFParameterization:
  """Tests for GAFF atom typing and LJ parameter assignment."""

  @pytest.mark.skip(reason="GAFF charge assignment not yet supported in proxide")
  def test_gaff_atom_types_from_mol2(self):
    """Test that MOL2 files with GAFF types are parsed correctly."""

  @pytest.mark.skip(reason="GAFF charge assignment not yet supported in proxide")
  def test_parse_structure_with_gaff(self):
    """Test parse_structure with GAFF force field for ligands."""


class TestAtomicSystemMDAttributes:
  """Tests that AtomicSystem has all expected static attributes for MD."""

  @pytest.fixture
  def parameterized_system(self):
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

  def test_has_static_md_attributes(self, parameterized_system):
    """Test that AtomicSystem has all static MD attributes needed."""
    system = parameterized_system

    # All these should be present (static params for partial binding)
    static_attrs = [
      "coordinates",  # Dynamic during sim
      "atom_mask",
      "charges",
      "sigmas",
      "epsilons",
      "bonds",
      "bond_params",
      "angles",
      "angle_params",
      "proper_dihedrals",
      "dihedral_params",
    ]

    for attr in static_attrs:
      value = getattr(system, attr, None)
      assert value is not None, f"AtomicSystem missing {attr}"

  def test_shapes_consistent(self, parameterized_system):
    """Test that array shapes are consistent."""
    system = parameterized_system
    n_atoms = len(system.charges)

    assert system.sigmas.shape == (n_atoms,)
    assert system.epsilons.shape == (n_atoms,)
    assert system.bonds.shape[1] == 2, "Bonds should be (N_bonds, 2)"
    assert system.angles.shape[1] == 3, "Angles should be (N_angles, 3)"
    assert system.proper_dihedrals.shape[1] == 4, "Dihedrals should be (N_dihedrals, 4)"


class TestLJParameterValues:
  """Tests for reasonable LJ parameter values."""

  @pytest.fixture
  def parameterized_system(self):
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

  def test_sigma_values_physical(self, parameterized_system):
    """Test that sigma values are in physical range."""
    sigmas = parameterized_system.sigmas
    nonzero_sigmas = sigmas[sigmas > 0]

    if len(nonzero_sigmas) > 0:
      # Rust returns units in nm: 0.1-0.4 nm = 1.0-4.0 Angstroms
      assert jnp.all(nonzero_sigmas > 0.05), "Sigma too small"
      assert jnp.all(nonzero_sigmas < 0.5), "Sigma too large"

  def test_epsilon_values_physical(self, parameterized_system):
    """Test that epsilon values are in physical range."""
    epsilons = parameterized_system.epsilons

    assert jnp.all(epsilons >= 0), "Found negative epsilon"
    nonzero_eps = epsilons[epsilons > 0]

    if len(nonzero_eps) > 0:
      # Rust returns units in kJ/mol
      assert jnp.all(nonzero_eps < 5.0), "Epsilon too large"


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
