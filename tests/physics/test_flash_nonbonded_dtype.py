"""Tests for flash_nonbonded dtype consistency (float32/float64 compatibility).

Validates that float32 literal dtype mismatches are fixed and that flash_nonbonded
handles both float32 and float64 arrays correctly, auto-promoting plain Python
floats to match the surrounding array dtype.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Flash nonbonded force compiles — deselect from GitHub-faithful CI (XA-CI).
pytestmark = pytest.mark.slow

from prolix.physics.flash_nonbonded import flash_nonbonded_forces
from prolix.padding import PaddedSystem


@pytest.fixture
def enable_x64():
    """Enable JAX x64 mode for this test and restore on teardown."""
    old_value = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", old_value)


@pytest.fixture
def minimal_padded_system_float32() -> PaddedSystem:
    """Create a minimal 8-atom system in float32 for dtype testing."""
    n = 8

    # Positions and parameters all cast to float32
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [5.0, 5.0, 0.0],
            [2.5, 2.5, 0.0],
            [7.5, 2.5, 0.0],
            [2.5, 7.5, 0.0],
            [7.5, 7.5, 0.0],
        ],
        dtype=jnp.float32,
    )

    atom_mask = jnp.ones(n, dtype=jnp.bool_)

    return PaddedSystem(
        positions=positions,
        charges=jnp.ones(n, dtype=jnp.float32) * 0.5,
        sigmas=jnp.ones(n, dtype=jnp.float32) * 3.0,
        epsilons=jnp.ones(n, dtype=jnp.float32) * 0.1,
        radii=jnp.ones(n, dtype=jnp.float32) * 1.5,
        scaled_radii=jnp.ones(n, dtype=jnp.float32) * 0.8,
        masses=jnp.ones(n, dtype=jnp.float32) * 12.0,
        element_ids=jnp.ones(n, dtype=jnp.int32) * 6,
        atom_mask=atom_mask,
        is_hydrogen=jnp.zeros(n, dtype=jnp.bool_),
        is_backbone=jnp.ones(n, dtype=jnp.bool_),
        is_heavy=jnp.ones(n, dtype=jnp.bool_),
        protein_atom_mask=jnp.ones(n, dtype=jnp.bool_),
        water_atom_mask=jnp.zeros(n, dtype=jnp.bool_),
        bonds=jnp.zeros((0, 2), dtype=jnp.int32),
        bond_params=jnp.zeros((0, 2), dtype=jnp.float32),
        bond_mask=jnp.zeros((0,), dtype=jnp.bool_),
        angles=jnp.zeros((0, 3), dtype=jnp.int32),
        angle_params=jnp.zeros((0, 2), dtype=jnp.float32),
        angle_mask=jnp.zeros((0,), dtype=jnp.bool_),
        dihedrals=jnp.zeros((0, 4), dtype=jnp.int32),
        dihedral_params=jnp.zeros((0, 3), dtype=jnp.float32),
        dihedral_mask=jnp.zeros((0,), dtype=jnp.bool_),
        impropers=jnp.zeros((0, 4), dtype=jnp.int32),
        improper_params=jnp.zeros((0, 3), dtype=jnp.float32),
        improper_mask=jnp.zeros((0,), dtype=jnp.bool_),
        urey_bradley_bonds=None,
        urey_bradley_params=None,
        urey_bradley_mask=None,
        excl_indices=jnp.full((n, 4), -1, dtype=jnp.int32),
        excl_scales_vdw=jnp.ones((n, 4), dtype=jnp.float32),
        excl_scales_elec=jnp.ones((n, 4), dtype=jnp.float32),
        n_padded_atoms=n,
    )


@pytest.fixture
def minimal_padded_system_float64() -> PaddedSystem:
    """Create a minimal 8-atom system in float64 for dtype testing."""
    n = 8

    # Positions and parameters all cast to float64
    positions = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
            [5.0, 5.0, 0.0],
            [2.5, 2.5, 0.0],
            [7.5, 2.5, 0.0],
            [2.5, 7.5, 0.0],
            [7.5, 7.5, 0.0],
        ],
        dtype=jnp.float64,
    )

    atom_mask = jnp.ones(n, dtype=jnp.bool_)

    return PaddedSystem(
        positions=positions,
        charges=jnp.ones(n, dtype=jnp.float64) * 0.5,
        sigmas=jnp.ones(n, dtype=jnp.float64) * 3.0,
        epsilons=jnp.ones(n, dtype=jnp.float64) * 0.1,
        radii=jnp.ones(n, dtype=jnp.float64) * 1.5,
        scaled_radii=jnp.ones(n, dtype=jnp.float64) * 0.8,
        masses=jnp.ones(n, dtype=jnp.float64) * 12.0,
        element_ids=jnp.ones(n, dtype=jnp.int32) * 6,
        atom_mask=atom_mask,
        is_hydrogen=jnp.zeros(n, dtype=jnp.bool_),
        is_backbone=jnp.ones(n, dtype=jnp.bool_),
        is_heavy=jnp.ones(n, dtype=jnp.bool_),
        protein_atom_mask=jnp.ones(n, dtype=jnp.bool_),
        water_atom_mask=jnp.zeros(n, dtype=jnp.bool_),
        bonds=jnp.zeros((0, 2), dtype=jnp.int32),
        bond_params=jnp.zeros((0, 2), dtype=jnp.float64),
        bond_mask=jnp.zeros((0,), dtype=jnp.bool_),
        angles=jnp.zeros((0, 3), dtype=jnp.int32),
        angle_params=jnp.zeros((0, 2), dtype=jnp.float64),
        angle_mask=jnp.zeros((0,), dtype=jnp.bool_),
        dihedrals=jnp.zeros((0, 4), dtype=jnp.int32),
        dihedral_params=jnp.zeros((0, 3), dtype=jnp.float64),
        dihedral_mask=jnp.zeros((0,), dtype=jnp.bool_),
        impropers=jnp.zeros((0, 4), dtype=jnp.int32),
        improper_params=jnp.zeros((0, 3), dtype=jnp.float64),
        improper_mask=jnp.zeros((0,), dtype=jnp.bool_),
        urey_bradley_bonds=None,
        urey_bradley_params=None,
        urey_bradley_mask=None,
        excl_indices=jnp.full((n, 4), -1, dtype=jnp.int32),
        excl_scales_vdw=jnp.ones((n, 4), dtype=jnp.float64),
        excl_scales_elec=jnp.ones((n, 4), dtype=jnp.float64),
        n_padded_atoms=n,
    )


class TestFlashNonbondedDtype:
    """Test dtype consistency in flash_nonbonded forces."""

    def test_float32_forces_dtype(self, minimal_padded_system_float32):
        """Verify forces output float32 when input is float32."""
        sys = minimal_padded_system_float32
        forces = flash_nonbonded_forces(sys, T=4)

        # Output dtype should match input
        assert forces.dtype == jnp.float32, f"Expected float32, got {forces.dtype}"
        assert forces.shape == sys.positions.shape
        assert jnp.all(jnp.isfinite(forces)), "Forces contain NaN or Inf"

    def test_float64_forces_dtype(self, minimal_padded_system_float64, enable_x64):
        """Verify forces output float64 when input is float64."""
        sys = minimal_padded_system_float64
        forces = flash_nonbonded_forces(sys, T=4)

        # Output dtype should match input
        assert forces.dtype == jnp.float64, f"Expected float64, got {forces.dtype}"
        assert forces.shape == sys.positions.shape
        assert jnp.all(jnp.isfinite(forces)), "Forces contain NaN or Inf"

    def test_float32_force_magnitudes(self, minimal_padded_system_float32):
        """Verify float32 forces are reasonable (non-zero, finite)."""
        sys = minimal_padded_system_float32
        forces = flash_nonbonded_forces(sys, T=4)

        # Check that forces are not all zero (there are Coulomb interactions)
        assert jnp.max(jnp.abs(forces)) > 0.0, "All forces are zero"

        # Check magnitudes are in reasonable range (not NaN or huge)
        assert jnp.all(jnp.abs(forces) < 1e6), "Forces suspiciously large"

    @pytest.mark.slow
    def test_float64_force_magnitudes(self, minimal_padded_system_float64, enable_x64):
        """Verify float64 forces are reasonable (non-zero, finite)."""
        sys = minimal_padded_system_float64
        forces = flash_nonbonded_forces(sys, T=4)

        # Check that forces are not all zero
        assert jnp.max(jnp.abs(forces)) > 0.0, "All forces are zero"

        # Check magnitudes are in reasonable range
        assert jnp.all(jnp.abs(forces) < 1e6), "Forces suspiciously large"

    @pytest.mark.slow
    def test_float32_float64_consistency(
        self, minimal_padded_system_float32, minimal_padded_system_float64, enable_x64
    ):
        """Verify float32 and float64 give consistent energies (within tolerance).

        This is a loose consistency check: we allow 1e-4 relative tolerance
        because float32 and float64 have different precisions and can accumulate
        errors differently.
        """
        sys32 = minimal_padded_system_float32
        sys64 = minimal_padded_system_float64

        forces32 = flash_nonbonded_forces(sys32, T=4)
        forces64 = flash_nonbonded_forces(sys64, T=4)

        # Cast float64 forces to float32 for comparison
        forces64_cast = forces64.astype(jnp.float32)

        # Check relative consistency: should be close but not identical
        # Use 1e-4 relative tolerance for float32 precision differences
        rel_diff = jnp.abs(forces32 - forces64_cast) / (
            jnp.abs(forces64_cast) + 1e-6
        )

        # Most forces should be consistent within 1e-4 relative error
        consistent_count = jnp.sum(rel_diff < 1e-4)
        total_forces = forces32.size
        consistency_fraction = consistent_count / total_forces

        assert (
            consistency_fraction > 0.8
        ), f"Only {consistency_fraction*100:.1f}% of forces are consistent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
