"""Tests for Hydrogen Mass Repartitioning (HMR).

Tests Phase 5 functionality:
- Mass conservation
- Hydrogen mass increase
- Heavy atom mass decrease (bounded)
- Ghost atom exclusion
- Timestep recommendation
- Diagnostic report
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from prolix.physics.hmr import (
    HMRConfig,
    compute_hmr_timestep,
    is_hydrogen,
    repartition_masses,
    report_hmr,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def methane():
    """CH4 molecule: 1 carbon + 4 hydrogens."""
    masses = jnp.array([12.011, 1.008, 1.008, 1.008, 1.008])
    elements = ['C', 'H', 'H', 'H', 'H']
    bond_pairs = jnp.array([
        [0, 1], [0, 2], [0, 3], [0, 4],  # C-H bonds
    ])
    h_mask = is_hydrogen(elements=elements)
    atom_mask = jnp.ones(5, dtype=bool)
    return {
        'masses': masses,
        'elements': elements,
        'bond_pairs': bond_pairs,
        'h_mask': h_mask,
        'atom_mask': atom_mask,
    }


@pytest.fixture
def water():
    """H2O molecule."""
    masses = jnp.array([15.999, 1.008, 1.008])
    elements = ['O', 'H', 'H']
    bond_pairs = jnp.array([[0, 1], [0, 2]])
    h_mask = is_hydrogen(elements=elements)
    atom_mask = jnp.ones(3, dtype=bool)
    return {
        'masses': masses,
        'elements': elements,
        'bond_pairs': bond_pairs,
        'h_mask': h_mask,
        'atom_mask': atom_mask,
    }


@pytest.fixture
def with_ghosts(methane):
    """Methane with 2 ghost atoms appended."""
    masses = jnp.concatenate([methane['masses'], jnp.array([1.008, 12.011])])
    bond_pairs = jnp.concatenate([
        methane['bond_pairs'],
        jnp.array([[5, 6]]),  # ghost-ghost bond
    ])
    h_mask = jnp.concatenate([methane['h_mask'], jnp.array([True, False])])
    atom_mask = jnp.concatenate([methane['atom_mask'], jnp.array([False, False])])
    return {
        'masses': masses,
        'bond_pairs': bond_pairs,
        'h_mask': h_mask,
        'atom_mask': atom_mask,
    }


# ===========================================================================
# Tests
# ===========================================================================

class TestIsHydrogen:
    """Test hydrogen identification."""

    def test_from_elements(self):
        """Identify H from element strings."""
        h = is_hydrogen(elements=['C', 'H', 'O', 'H', 'N'])
        expected = jnp.array([False, True, False, True, False])
        np.testing.assert_array_equal(h, expected)

    def test_from_masses(self):
        """Identify H from masses (< 1.1 amu)."""
        masses = jnp.array([12.0, 1.008, 14.0, 1.008, 1.008])
        h = is_hydrogen(masses=masses)
        expected = jnp.array([False, True, False, True, True])
        np.testing.assert_array_equal(h, expected)

    def test_deuterium(self):
        """Deuterium should also be identified."""
        h = is_hydrogen(elements=['C', 'D', 'H'])
        expected = jnp.array([False, True, True])
        np.testing.assert_array_equal(h, expected)


class TestRepartition:
    """Test mass repartitioning."""

    def test_disabled_noop(self, methane):
        """Disabled HMR should return original masses."""
        config = HMRConfig(enabled=False)
        result = repartition_masses(
            methane['masses'], methane['bond_pairs'],
            methane['h_mask'], methane['atom_mask'],
            config,
        )
        np.testing.assert_array_equal(result, methane['masses'])

    def test_mass_conservation(self, methane):
        """Total mass must be conserved."""
        config = HMRConfig(enabled=True, target_h_mass=3.024)
        result = repartition_masses(
            methane['masses'], methane['bond_pairs'],
            methane['h_mask'], methane['atom_mask'],
            config,
        )
        original_total = float(jnp.sum(methane['masses']))
        new_total = float(jnp.sum(result))
        np.testing.assert_allclose(new_total, original_total, atol=1e-3)

    def test_hydrogen_mass_increases(self, methane):
        """Hydrogen masses should increase to target."""
        config = HMRConfig(enabled=True, target_h_mass=3.024)
        result = repartition_masses(
            methane['masses'], methane['bond_pairs'],
            methane['h_mask'], methane['atom_mask'],
            config,
        )
        # All 4 hydrogens should be at 3.024
        h_masses = result[methane['h_mask']]
        np.testing.assert_allclose(h_masses, 3.024, atol=1e-3)

    def test_heavy_atom_decreases(self, methane):
        """Carbon mass should decrease by 4 × (3.024 - 1.008)."""
        config = HMRConfig(enabled=True, target_h_mass=3.024)
        result = repartition_masses(
            methane['masses'], methane['bond_pairs'],
            methane['h_mask'], methane['atom_mask'],
            config,
        )
        expected_c_mass = 12.011 - 4 * (3.024 - 1.008)
        np.testing.assert_allclose(float(result[0]), expected_c_mass, atol=1e-3)

    def test_min_heavy_mass_guard(self):
        """Heavy atom should not go below min_heavy_mass."""
        # Make a case where carbon can't afford to give mass to all H
        masses = jnp.array([4.0, 1.008, 1.008, 1.008, 1.008])  # Light "carbon"
        bond_pairs = jnp.array([[0, 1], [0, 2], [0, 3], [0, 4]])
        h_mask = jnp.array([False, True, True, True, True])
        atom_mask = jnp.ones(5, dtype=bool)
        config = HMRConfig(enabled=True, target_h_mass=3.024, min_heavy_mass=1.5)

        result = repartition_masses(masses, bond_pairs, h_mask, atom_mask, config)
        # Heavy atom should be >= min_heavy_mass
        assert float(result[0]) >= 1.5 - 1e-6

    def test_ghost_atoms_ignored(self, with_ghosts):
        """Ghost atoms should not participate in repartitioning."""
        config = HMRConfig(enabled=True, target_h_mass=3.024)
        result = repartition_masses(
            with_ghosts['masses'], with_ghosts['bond_pairs'],
            with_ghosts['h_mask'], with_ghosts['atom_mask'],
            config,
        )
        # Ghost masses should be unchanged
        np.testing.assert_allclose(float(result[5]), 1.008, atol=1e-6)
        np.testing.assert_allclose(float(result[6]), 12.011, atol=1e-6)

    def test_water(self, water):
        """Water O-H repartitioning."""
        config = HMRConfig(enabled=True, target_h_mass=3.024)
        result = repartition_masses(
            water['masses'], water['bond_pairs'],
            water['h_mask'], water['atom_mask'],
            config,
        )
        # Both H should be 3.024
        np.testing.assert_allclose(float(result[1]), 3.024, atol=1e-3)
        np.testing.assert_allclose(float(result[2]), 3.024, atol=1e-3)
        # Total mass conserved
        np.testing.assert_allclose(
            float(jnp.sum(result)),
            float(jnp.sum(water['masses'])),
            atol=1e-3,
        )


class TestTimestep:
    """Test timestep recommendations."""

    def test_disabled(self):
        """Without HMR, 2 fs timestep."""
        assert compute_hmr_timestep(None) == 0.002

    def test_enabled_3x(self):
        """With 3× H mass, 4 fs timestep."""
        config = HMRConfig(enabled=True, target_h_mass=3.024)
        assert compute_hmr_timestep(config) == 0.004

    def test_enabled_2x(self):
        """With 2× H mass, 3 fs timestep."""
        config = HMRConfig(enabled=True, target_h_mass=2.016)
        assert compute_hmr_timestep(config) == 0.003


class TestReport:
    """Test diagnostic reports."""

    def test_report_keys(self, methane):
        """Report should have expected keys."""
        config = HMRConfig(enabled=True, target_h_mass=3.024)
        new_masses = repartition_masses(
            methane['masses'], methane['bond_pairs'],
            methane['h_mask'], methane['atom_mask'],
            config,
        )
        report = report_hmr(
            methane['masses'], new_masses,
            methane['h_mask'], methane['atom_mask'],
        )
        expected_keys = {
            'n_atoms', 'n_hydrogen', 'n_heavy',
            'total_mass_original', 'total_mass_new',
            'avg_h_mass_original', 'avg_h_mass_new',
            'min_heavy_mass', 'mass_conserved',
        }
        assert set(report.keys()) == expected_keys

    def test_report_conservation(self, methane):
        """Report should confirm mass conservation."""
        config = HMRConfig(enabled=True, target_h_mass=3.024)
        new_masses = repartition_masses(
            methane['masses'], methane['bond_pairs'],
            methane['h_mask'], methane['atom_mask'],
            config,
        )
        report = report_hmr(
            methane['masses'], new_masses,
            methane['h_mask'], methane['atom_mask'],
        )
        assert report['mass_conserved']
        assert report['n_hydrogen'] == 4
        assert report['n_heavy'] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
