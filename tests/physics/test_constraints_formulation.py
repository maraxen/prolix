"""Unit tests for ConstraintDOFMask formulation (Phase 1.1).

Tests validate:
1. DOF mask construction for various system sizes
2. Orthogonality and completeness of DOF decomposition
3. Projection operator properties (idempotence, nullspace)
4. Constraint equation count
"""

import jax.numpy as jnp
import pytest

from prolix.physics.constraints import ConstraintDOFMask


class TestConstraintDOFMaskConstruction:
  """Test ConstraintDOFMask instantiation and basic properties."""

  def test_construction_single_water(self):
    """Verify instantiation with N=1 water."""
    water_indices = jnp.array([[0, 1, 2]], dtype=jnp.int32)  # One water: O at 0, H1 at 1, H2 at 2
    n_atoms = 3
    mask = ConstraintDOFMask(water_indices=water_indices, n_atoms=n_atoms)

    assert mask.n_waters == 1
    assert mask.n_atoms == 3
    assert mask.n_constraint_equations == 3
    assert mask.n_rigid_atoms == 3  # All atoms in water
    assert mask.n_free_atoms == 0

  def test_construction_64_waters(self):
    """Verify instantiation with N=64 waters (typical MD system)."""
    n_waters = 64
    n_atoms = n_waters * 3
    # Simple layout: water i has atoms [3*i, 3*i+1, 3*i+2]
    water_indices = jnp.arange(n_atoms, dtype=jnp.int32).reshape(n_waters, 3)
    mask = ConstraintDOFMask(water_indices=water_indices, n_atoms=n_atoms)

    assert mask.n_waters == 64
    assert mask.n_atoms == 192
    assert mask.n_constraint_equations == 192
    assert mask.n_rigid_atoms == 192
    assert mask.n_free_atoms == 0

  def test_construction_512_waters(self):
    """Verify instantiation with N=512 waters (large system)."""
    n_waters = 512
    n_atoms = n_waters * 3
    water_indices = jnp.arange(n_atoms, dtype=jnp.int32).reshape(n_waters, 3)
    mask = ConstraintDOFMask(water_indices=water_indices, n_atoms=n_atoms)

    assert mask.n_waters == 512
    assert mask.n_constraint_equations == 1536
    assert mask.n_rigid_atoms == 1536
    assert mask.n_free_atoms == 0

  def test_construction_with_solute(self):
    """Verify instantiation with water + solute atoms."""
    # 4 water molecules + 8 solute atoms
    water_indices = jnp.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=jnp.int32)
    n_atoms = 20  # 12 water + 8 solute
    mask = ConstraintDOFMask(water_indices=water_indices, n_atoms=n_atoms)

    assert mask.n_waters == 4
    assert mask.n_constraint_equations == 12
    assert mask.n_rigid_atoms == 12
    assert mask.n_free_atoms == 8

  def test_construction_no_waters(self):
    """Verify instantiation with N=0 waters (implicit solvent)."""
    water_indices = jnp.zeros((0, 3), dtype=jnp.int32)
    n_atoms = 10
    mask = ConstraintDOFMask(water_indices=water_indices, n_atoms=n_atoms)

    assert mask.n_waters == 0
    assert mask.n_constraint_equations == 0
    assert mask.n_rigid_atoms == 0
    assert mask.n_free_atoms == 10


class TestConstraintDOFMaskOrthogonality:
  """Test orthogonality and completeness of DOF masks."""

  def test_rigid_free_are_orthogonal(self):
    """Verify rigid_dof_mask and free_dof_mask have no overlap."""
    water_indices = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
    n_atoms = 10  # 6 in water, 4 free
    mask = ConstraintDOFMask(water_indices=water_indices, n_atoms=n_atoms)

    rigid = mask.rigid_dof_mask
    free = mask.free_dof_mask

    # No atom should be both rigid and free
    overlap = rigid & free
    assert jnp.all(~overlap), "rigid_dof_mask and free_dof_mask overlap"

  def test_rigid_free_are_complete(self):
    """Verify rigid_dof_mask and free_dof_mask cover all atoms."""
    water_indices = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
    n_atoms = 10  # 6 in water, 4 free
    mask = ConstraintDOFMask(water_indices=water_indices, n_atoms=n_atoms)

    rigid = mask.rigid_dof_mask
    free = mask.free_dof_mask

    # Every atom should be either rigid or free
    union = rigid | free
    assert jnp.all(union), "rigid_dof_mask and free_dof_mask do not cover all atoms"

  def test_is_orthogonal_complement_true(self):
    """Verify is_orthogonal_complement returns True for valid system."""
    water_indices = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
    n_atoms = 10
    mask = ConstraintDOFMask(water_indices=water_indices, n_atoms=n_atoms)

    assert mask.is_orthogonal_complement(), "Expected orthogonal complement check to pass"

  def test_is_orthogonal_complement_for_pure_water(self):
    """Verify is_orthogonal_complement for pure-water system (N waters only)."""
    n_waters = 64
    n_atoms = n_waters * 3
    water_indices = jnp.arange(n_atoms, dtype=jnp.int32).reshape(n_waters, 3)
    mask = ConstraintDOFMask(water_indices=water_indices, n_atoms=n_atoms)

    assert mask.is_orthogonal_complement()

  def test_is_orthogonal_complement_for_no_water(self):
    """Verify is_orthogonal_complement for implicit-solvent system (no water)."""
    water_indices = jnp.zeros((0, 3), dtype=jnp.int32)
    n_atoms = 10
    mask = ConstraintDOFMask(water_indices=water_indices, n_atoms=n_atoms)

    assert mask.is_orthogonal_complement()


class TestProjectionOperator:
  """Test projection operator construction and properties."""

  def test_projection_operator_shape(self):
    """Verify projection operator has correct shape."""
    water_indices = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
    n_atoms = 8  # 6 in water, 2 free
    mask = ConstraintDOFMask(water_indices=water_indices, n_atoms=n_atoms)

    P = mask.projection_operator()

    expected_size = 3 * n_atoms
    assert P.shape == (expected_size, expected_size), f"Expected shape {(expected_size, expected_size)}, got {P.shape}"

  def test_projection_operator_is_idempotent(self):
    """Verify P @ P = P (idempotence)."""
    water_indices = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
    n_atoms = 10
    mask = ConstraintDOFMask(water_indices=water_indices, n_atoms=n_atoms)

    P = mask.projection_operator()
    P2 = P @ P

    # Verify idempotence: P @ P ≈ P
    error = jnp.max(jnp.abs(P2 - P))
    assert error < 1e-14, f"Projection operator not idempotent; max error: {error}"

  def test_projection_operator_removes_rigid_dof(self):
    """Verify P @ v_rigid ≈ 0 for rigid-body velocities."""
    water_indices = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
    n_atoms = 10
    mask = ConstraintDOFMask(water_indices=water_indices, n_atoms=n_atoms)

    P = mask.projection_operator()

    # Create a synthetic rigid-body velocity: all atoms in water have same velocity
    v_rigid = jnp.ones((n_atoms, 3), dtype=jnp.float32)
    v_flat = v_rigid.reshape(-1)

    # Apply projection
    v_proj_flat = P @ v_flat
    v_proj = v_proj_flat.reshape(n_atoms, 3)

    # Verify rigid atoms get zero velocity after projection
    rigid_mask = mask.rigid_dof_mask
    v_rigid_after = jnp.where(rigid_mask, v_proj, 0.0)
    error = jnp.max(jnp.abs(v_rigid_after))
    assert error < 1e-14, f"Projection failed to remove rigid DOF; max error: {error}"

  def test_projection_operator_preserves_free_dof(self):
    """Verify P @ v_free ≈ v_free for free-DOF velocities."""
    water_indices = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
    n_atoms = 10
    mask = ConstraintDOFMask(water_indices=water_indices, n_atoms=n_atoms)

    P = mask.projection_operator()

    # Create a velocity on free atoms only
    v = jnp.zeros((n_atoms, 3), dtype=jnp.float32)
    free_mask = mask.free_dof_mask
    v = jnp.where(free_mask, 1.0, 0.0)
    v_flat = v.reshape(-1)

    # Apply projection
    v_proj_flat = P @ v_flat
    v_proj = v_proj_flat.reshape(n_atoms, 3)

    # Verify free atoms remain unchanged
    error = jnp.max(jnp.abs(v_proj - v))
    assert error < 1e-14, f"Projection failed to preserve free DOF; max error: {error}"

  def test_projection_operator_zero_for_pure_water(self):
    """Verify P is zero matrix when all atoms are in water."""
    n_waters = 4
    n_atoms = n_waters * 3
    water_indices = jnp.arange(n_atoms, dtype=jnp.int32).reshape(n_waters, 3)
    mask = ConstraintDOFMask(water_indices=water_indices, n_atoms=n_atoms)

    P = mask.projection_operator()

    # P should be zero because all DOF are constrained
    assert jnp.allclose(P, 0.0, atol=1e-15), "Projection operator should be zero for pure-water system"

  def test_projection_operator_identity_for_no_water(self):
    """Verify P is identity matrix when no water (implicit solvent)."""
    water_indices = jnp.zeros((0, 3), dtype=jnp.int32)
    n_atoms = 5
    mask = ConstraintDOFMask(water_indices=water_indices, n_atoms=n_atoms)

    P = mask.projection_operator()

    # P should be identity because no DOF are constrained
    expected_identity = jnp.eye(3 * n_atoms)
    assert jnp.allclose(P, expected_identity, atol=1e-15), "Projection operator should be identity for implicit-solvent system"


class TestConstraintCounts:
  """Test constraint equation counting."""

  def test_constraint_count_single_water(self):
    """Verify n_constraint_equations = 3 for 1 water."""
    water_indices = jnp.array([[0, 1, 2]], dtype=jnp.int32)
    mask = ConstraintDOFMask(water_indices=water_indices, n_atoms=3)

    assert mask.n_constraint_equations == 3

  def test_constraint_count_multiple_waters(self):
    """Verify n_constraint_equations = 3 * n_waters."""
    for n_waters in [1, 4, 64, 512]:
      water_indices = jnp.arange(n_waters * 3, dtype=jnp.int32).reshape(n_waters, 3)
      mask = ConstraintDOFMask(water_indices=water_indices, n_atoms=n_waters * 3)

      assert mask.n_constraint_equations == 3 * n_waters

  def test_constraint_count_zero_waters(self):
    """Verify n_constraint_equations = 0 for no water."""
    water_indices = jnp.zeros((0, 3), dtype=jnp.int32)
    mask = ConstraintDOFMask(water_indices=water_indices, n_atoms=10)

    assert mask.n_constraint_equations == 0


class TestDOFMaskConsistency:
  """Test internal consistency of DOF masks."""

  def test_rigid_plus_free_equals_total(self):
    """Verify n_rigid_atoms + n_free_atoms = n_atoms."""
    water_indices = jnp.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=jnp.int32)
    n_atoms = 12  # 9 in water, 3 free
    mask = ConstraintDOFMask(water_indices=water_indices, n_atoms=n_atoms)

    total_atoms = mask.n_rigid_atoms + mask.n_free_atoms
    assert total_atoms == n_atoms, f"Expected {n_atoms} total atoms, got {total_atoms}"

  def test_rigid_dof_count_correct(self):
    """Verify n_rigid_atoms = 3 * n_waters."""
    for n_waters in [1, 10, 64]:
      water_indices = jnp.arange(n_waters * 3, dtype=jnp.int32).reshape(n_waters, 3)
      n_atoms = n_waters * 3 + 10  # Add some solute
      mask = ConstraintDOFMask(water_indices=water_indices, n_atoms=n_atoms)

      assert mask.n_rigid_atoms == 3 * n_waters

  def test_free_dof_count_correct(self):
    """Verify n_free_atoms = n_atoms - 3*n_waters."""
    water_indices = jnp.array([[0, 1, 2], [3, 4, 5]], dtype=jnp.int32)
    n_atoms = 20
    mask = ConstraintDOFMask(water_indices=water_indices, n_atoms=n_atoms)

    expected_free = n_atoms - 6  # 2 waters = 6 atoms
    assert mask.n_free_atoms == expected_free


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
