"""Tests for force field loading.

These tests use the new Rust-based force field loading from XML files.
The old equinox-based save/load has been deprecated.
"""

import pytest
from proxide.physics.force_fields import (
  FullForceField,
  list_available_force_fields,
  load_force_field,
)


def test_list_available_force_fields():
  """Test listing available force fields."""
  ff_list = list_available_force_fields()
  assert isinstance(ff_list, list)
  # Should have at least some force fields if assets are populated
  # This may be empty if assets are not installed


def test_load_force_field_from_assets():
  """Test loading force field from assets directory."""
  ff_list = list_available_force_fields()
  if not ff_list:
    pytest.skip("No force field assets available")

  # Load the first available force field
  ff = load_force_field(ff_list[0])
  assert isinstance(ff, FullForceField)
  assert len(ff.source_files) > 0


def test_load_force_field_gaff():
  """Test loading GAFF force field."""
  try:
    ff = load_force_field("gaff")
    assert isinstance(ff, FullForceField)
  except ValueError:
    pytest.skip("gaff.xml not available in assets")


def test_force_field_get_charge():
  """Test getting charge for specific atom."""
  try:
    ff = load_force_field("protein.ff14SB")
  except ValueError:
    pytest.skip("protein.ff14SB.xml not available")

  # For a typical amino acid
  charge = ff.get_charge("ALA", "CA")
  # CA typically has a small or zero charge
  assert isinstance(charge, float)


def test_force_field_get_lj_params():
  """Test getting LJ parameters for specific atom."""
  try:
    ff = load_force_field("protein.ff14SB")
  except ValueError:
    pytest.skip("protein.ff14SB.xml not available")

  sigma, epsilon = ff.get_lj_params("ALA", "CA")
  assert isinstance(sigma, float)
  assert isinstance(epsilon, float)
  assert sigma > 0  # Sigma should be positive
  assert epsilon > 0  # Epsilon should be positive
