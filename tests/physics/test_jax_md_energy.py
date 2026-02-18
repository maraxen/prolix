"""Tests for JAX MD energy function with parse_structure API.

This module tests that the Protein object from parse_structure contains
the correct parameters for building energy functions.
"""

from pathlib import Path

import jax.numpy as jnp
import pytest
from proxide import OutputSpec, parse_structure

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


@pytest.fixture
def parameterized_protein():
  """Load a parameterized protein."""
  pdb_path = DATA_DIR / "1CRN.pdb"

  spec = OutputSpec()
  spec.parameterize_md = True
  spec.force_field = str(FF_PATH)
  spec.add_hydrogens = True

  return parse_structure(str(pdb_path), spec)


def test_charges_shape(parameterized_protein):
  """Test that charges have the correct shape."""
  protein = parameterized_protein
  n_atoms = len(protein.charges)
  assert protein.charges.shape == (n_atoms,)
  assert n_atoms > 0


def test_lj_params_shape(parameterized_protein):
  """Test that LJ params have matching shapes."""
  protein = parameterized_protein
  assert protein.sigmas.shape == protein.charges.shape
  assert protein.epsilons.shape == protein.charges.shape


def test_bond_params_count(parameterized_protein):
  """Test that bond params match bond indices."""
  protein = parameterized_protein
  assert protein.bonds.shape[0] == protein.bond_params.shape[0]


def test_angle_params_count(parameterized_protein):
  """Test that angle params match angle indices."""
  protein = parameterized_protein
  assert protein.angles.shape[0] == protein.angle_params.shape[0]


def test_dihedral_params_count(parameterized_protein):
  """Test that dihedral params match dihedral indices."""
  protein = parameterized_protein
  assert protein.proper_dihedrals.shape[0] == protein.dihedral_params.shape[0]


def test_bond_indices_valid(parameterized_protein):
  """Test that bond indices are within valid range."""
  protein = parameterized_protein
  n_atoms = len(protein.charges)
  bonds = protein.bonds

  assert jnp.all(bonds >= 0), "Negative bond index found"
  assert jnp.all(bonds < n_atoms), f"Bond index >= n_atoms ({n_atoms})"


def test_angle_indices_valid(parameterized_protein):
  """Test that angle indices are within valid range."""
  protein = parameterized_protein
  n_atoms = len(protein.charges)
  angles = protein.angles

  assert jnp.all(angles >= 0), "Negative angle index found"
  assert jnp.all(angles < n_atoms), f"Angle index >= n_atoms ({n_atoms})"


def test_dihedral_indices_valid(parameterized_protein):
  """Test that dihedral indices are within valid range."""
  protein = parameterized_protein
  n_atoms = len(protein.charges)
  dihedrals = protein.proper_dihedrals

  assert jnp.all(dihedrals >= 0), "Negative dihedral index found"
  assert jnp.all(dihedrals < n_atoms), f"Dihedral index >= n_atoms ({n_atoms})"


def test_epsilons_non_negative(parameterized_protein):
  """Test that LJ epsilon values are non-negative."""
  protein = parameterized_protein
  assert jnp.all(protein.epsilons >= 0), "Negative epsilon found"


def test_sigmas_non_negative(parameterized_protein):
  """Test that LJ sigma values are non-negative."""
  protein = parameterized_protein
  assert jnp.all(protein.sigmas >= 0), "Negative sigma found"
