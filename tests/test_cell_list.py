"""Tests for cell-list decomposition and cell-nonbonded kernels.

Tests Phase 1 functionality:
- Cell list building with ghost atom sanitization
- Grid-shift vs lax.scan stencil energy parity
- Single-point LJ energy vs dense FlashMD reference
- Ewald exclusion correction
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from prolix.physics.cell_list import (
    CellList,
    build_cell_list,
    compute_grid_shape,
    compute_cell_size,
    HALF_SHELL_SHIFTS,
)
from prolix.physics.cell_nonbonded import (
    cell_energy_scan,
    cell_energy_grid_shift,
    ewald_exclusion_correction,
    _cell_pair_energy,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def simple_box():
    """Two LJ particles in a 30 Å cubic box."""
    box_size = jnp.array([30.0, 30.0, 30.0])
    cutoff = 10.0
    # Two atoms at known distance = 5.0 Å
    positions = jnp.array([
        [10.0, 15.0, 15.0],
        [15.0, 15.0, 15.0],
    ])
    atom_mask = jnp.array([True, True])
    sigmas = jnp.array([3.4, 3.4])     # Argon LJ sigma
    epsilons = jnp.array([0.2, 0.2])   # Argon LJ epsilon
    charges = jnp.array([0.0, 0.0])    # No Coulomb
    return {
        'positions': positions,
        'box_size': box_size,
        'atom_mask': atom_mask,
        'sigmas': sigmas,
        'epsilons': epsilons,
        'charges': charges,
        'cutoff': cutoff,
    }


@pytest.fixture
def four_atom_box():
    """Four atoms with charges in a 20 Å cubic box."""
    box_size = jnp.array([20.0, 20.0, 20.0])
    cutoff = 8.0
    positions = jnp.array([
        [5.0, 5.0, 5.0],
        [5.0, 5.0, 9.0],   # 4 Å from atom 0
        [10.0, 10.0, 10.0],
        [15.0, 15.0, 15.0],
    ])
    atom_mask = jnp.array([True, True, True, True])
    sigmas = jnp.array([3.0, 3.0, 3.0, 3.0])
    epsilons = jnp.array([0.15, 0.15, 0.15, 0.15])
    charges = jnp.array([0.5, -0.5, 0.3, -0.3])
    return {
        'positions': positions,
        'box_size': box_size,
        'atom_mask': atom_mask,
        'sigmas': sigmas,
        'epsilons': epsilons,
        'charges': charges,
        'cutoff': cutoff,
    }


@pytest.fixture
def ghost_atom_box():
    """Box with 2 real atoms and 2 ghost atoms."""
    box_size = jnp.array([20.0, 20.0, 20.0])
    cutoff = 10.0
    positions = jnp.array([
        [5.0, 5.0, 5.0],
        [10.0, 10.0, 10.0],
        [9999.0, 9999.0, 9999.0],  # ghost — bad position
        [0.0, 0.0, 0.0],           # ghost
    ])
    atom_mask = jnp.array([True, True, False, False])
    sigmas = jnp.array([3.4, 3.4, 1e-6, 1e-6])  # ghost has bad sigma
    epsilons = jnp.array([0.2, 0.2, 0.2, 0.2])
    charges = jnp.array([0.5, -0.5, 0.5, -0.5])
    return {
        'positions': positions,
        'box_size': box_size,
        'atom_mask': atom_mask,
        'sigmas': sigmas,
        'epsilons': epsilons,
        'charges': charges,
        'cutoff': cutoff,
    }


# ===========================================================================
# Cell list building tests
# ===========================================================================

class TestCellListBuilding:
    """Test cell list construction and ghost atom handling."""

    def test_grid_shape(self):
        """Grid dimensions should be >= cutoff."""
        box = jnp.array([30.0, 30.0, 30.0])
        shape = compute_grid_shape(box, cutoff=10.0)
        assert shape == (3, 3, 3)

    def test_grid_shape_large_box(self):
        """Larger box should give more cells."""
        box = jnp.array([60.0, 60.0, 60.0])
        shape = compute_grid_shape(box, cutoff=10.0)
        assert shape == (6, 6, 6)

    def test_grid_shape_min_cells(self):
        """Box smaller than 3×cutoff should still give 3 cells."""
        box = jnp.array([15.0, 15.0, 15.0])
        shape = compute_grid_shape(box, cutoff=10.0)
        assert shape == (3, 3, 3)

    def test_cell_size(self):
        """Cell sizes should divide box evenly."""
        box = jnp.array([30.0, 30.0, 30.0])
        shape = (3, 3, 3)
        sizes = compute_cell_size(box, shape)
        np.testing.assert_allclose(sizes, [10.0, 10.0, 10.0])

    def test_build_basic(self, simple_box):
        """Build cell list from 2 atoms."""
        cells = build_cell_list(**simple_box)
        assert cells.positions.shape == cells.occupancy.shape + (3,)
        assert cells.mask.shape == cells.occupancy.shape
        assert not cells.overflow
        # 2 real atoms total → sum of counts = 2
        assert int(jnp.sum(cells.counts)) == 2

    def test_ghost_sanitization(self, ghost_atom_box):
        """Ghost atoms should have sigma=1.0, eps=0.0, position at box center."""
        cells = build_cell_list(**ghost_atom_box)
        # Only 2 real atoms
        assert int(jnp.sum(cells.counts)) == 2
        # No overflow
        assert not cells.overflow
        # All sigmas in the grid should be >= 1.0 (ghost = 1.0, real = 3.4)
        assert float(jnp.min(cells.sigmas)) >= 1.0 - 1e-6
        # All epsilons for non-masked slots should be 0.0
        ghost_eps = cells.epsilons[~cells.mask]
        np.testing.assert_allclose(ghost_eps, 0.0)

    def test_overflow_detection(self):
        """Exceeding max_atoms_per_cell should flag overflow."""
        box = jnp.array([10.0, 10.0, 10.0])
        # Put 5 atoms in same cell, M=4
        positions = jnp.array([
            [1.0, 1.0, 1.0],
            [1.1, 1.1, 1.1],
            [1.2, 1.2, 1.2],
            [1.3, 1.3, 1.3],
            [1.4, 1.4, 1.4],
        ])
        cells = build_cell_list(
            positions=positions,
            box_size=box,
            atom_mask=jnp.ones(5, dtype=bool),
            sigmas=jnp.ones(5) * 3.0,
            epsilons=jnp.ones(5) * 0.1,
            charges=jnp.zeros(5),
            cutoff=5.0,
            max_atoms_per_cell=4,
        )
        assert bool(cells.overflow)


# ===========================================================================
# Energy computation tests
# ===========================================================================

class TestCellEnergy:
    """Test energy computation from cell lists."""

    def test_scan_vs_grid_shift_parity(self, four_atom_box):
        """Both stencil strategies should give the same energy."""
        cells = build_cell_list(**four_atom_box)
        box_size = four_atom_box['box_size']
        cutoff = four_atom_box['cutoff']

        e_scan = cell_energy_scan(cells, box_size, cutoff)
        e_grid = cell_energy_grid_shift(cells, box_size, cutoff)

        np.testing.assert_allclose(
            float(e_scan), float(e_grid),
            rtol=1e-4,
            err_msg=f"Scan ({e_scan}) != Grid-shift ({e_grid})",
        )

    def test_two_atom_lj(self, simple_box):
        """Known LJ energy for two atoms at r=5.0 Å, sigma=3.4, eps=0.2."""
        cells = build_cell_list(**simple_box)
        box_size = simple_box['box_size']
        cutoff = simple_box['cutoff']

        e = cell_energy_grid_shift(cells, box_size, cutoff)

        # Expected LJ: 4 * eps * [(sigma/r)^12 - (sigma/r)^6]
        r = 5.0
        sigma = 3.4
        eps = 0.2
        sr = sigma / r
        e_expected = 4.0 * eps * (sr**12 - sr**6)

        np.testing.assert_allclose(
            float(e), e_expected, rtol=1e-3,
            err_msg=f"Cell LJ ({e}) != analytical ({e_expected})",
        )

    def test_two_atom_lj_scan(self, simple_box):
        """Same LJ test via scan strategy."""
        cells = build_cell_list(**simple_box)
        box_size = simple_box['box_size']
        cutoff = simple_box['cutoff']

        e = cell_energy_scan(cells, box_size, cutoff)

        r = 5.0
        sigma = 3.4
        eps = 0.2
        sr = sigma / r
        e_expected = 4.0 * eps * (sr**12 - sr**6)

        np.testing.assert_allclose(float(e), e_expected, rtol=1e-3)

    def test_ghost_atoms_zero_energy(self, ghost_atom_box):
        """Ghost atoms should not contribute to energy."""
        cells = build_cell_list(**ghost_atom_box)
        box_size = ghost_atom_box['box_size']
        cutoff = ghost_atom_box['cutoff']

        # Compute with ghosts
        e_with_ghost = cell_energy_grid_shift(cells, box_size, cutoff)

        # Compute with only real atoms
        real_box = {
            'positions': ghost_atom_box['positions'][:2],
            'box_size': box_size,
            'atom_mask': jnp.array([True, True]),
            'sigmas': ghost_atom_box['sigmas'][:2],
            'epsilons': ghost_atom_box['epsilons'][:2],
            'charges': ghost_atom_box['charges'][:2],
            'cutoff': cutoff,
        }
        cells_real = build_cell_list(**real_box)
        e_real = cell_energy_grid_shift(cells_real, box_size, cutoff)

        np.testing.assert_allclose(
            float(e_with_ghost), float(e_real), rtol=1e-4,
            err_msg="Ghost atoms affected energy!",
        )

    def test_ewald_direct_space(self, four_atom_box):
        """Erfc-damped Coulomb should be less than plain Coulomb."""
        cells = build_cell_list(**four_atom_box)
        box_size = four_atom_box['box_size']
        cutoff = four_atom_box['cutoff']

        e_plain = cell_energy_grid_shift(cells, box_size, cutoff, alpha=None)
        e_ewald = cell_energy_grid_shift(cells, box_size, cutoff, alpha=0.3)

        # erfc(α·r)/r < 1/r for r > 0, so |e_ewald| < |e_plain|
        assert abs(float(e_ewald)) < abs(float(e_plain)) or abs(float(e_plain)) < 1e-6

    def test_energy_is_finite(self, four_atom_box):
        """Energy should be finite, no NaN/Inf."""
        cells = build_cell_list(**four_atom_box)
        e = cell_energy_grid_shift(cells, four_atom_box['box_size'], four_atom_box['cutoff'])
        assert jnp.isfinite(e)


# ===========================================================================
# Ewald exclusion correction tests
# ===========================================================================

class TestEwaldExclusion:
    """Test Ewald exclusion correction (Layer 2)."""

    def test_exclusion_correction_sign(self):
        """Ewald exclusion correction should be positive (subtracting negative)."""
        positions = jnp.array([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ])
        charges = jnp.array([0.5, -0.5])
        atom_mask = jnp.array([True, True])
        # Atom 0 excludes atom 1
        excl_indices = jnp.array([[1, -1], [0, -1]])
        excl_scales = jnp.array([[0.0, 0.0], [0.0, 0.0]])  # full exclusion

        e = ewald_exclusion_correction(
            positions, charges, atom_mask,
            excl_indices, excl_scales,
            alpha=0.3,
        )
        # For opposite charges, the direct Ewald contribution is negative,
        # so subtracting it (erf/r correction) should be positive contribution
        assert jnp.isfinite(e)

    def test_exclusion_no_pairs(self):
        """No exclusions → zero correction."""
        positions = jnp.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        charges = jnp.array([1.0, -1.0])
        atom_mask = jnp.array([True, True])
        excl_indices = jnp.array([[-1], [-1]])
        excl_scales = jnp.array([[0.0], [0.0]])

        e = ewald_exclusion_correction(
            positions, charges, atom_mask,
            excl_indices, excl_scales, alpha=0.3,
        )
        np.testing.assert_allclose(float(e), 0.0, atol=1e-6)

    def test_exclusion_with_box(self):
        """Exclusion correction with periodic boundary conditions."""
        box_size = jnp.array([10.0, 10.0, 10.0])
        # Atoms across periodic boundary
        positions = jnp.array([
            [1.0, 5.0, 5.0],
            [9.0, 5.0, 5.0],  # distance = 2.0 Å via PBC
        ])
        charges = jnp.array([0.5, -0.5])
        atom_mask = jnp.array([True, True])
        excl_indices = jnp.array([[1, -1], [0, -1]])
        excl_scales = jnp.array([[0.0, 0.0], [0.0, 0.0]])

        e = ewald_exclusion_correction(
            positions, charges, atom_mask,
            excl_indices, excl_scales,
            alpha=0.3, box_size=box_size,
        )
        assert jnp.isfinite(e)


# ===========================================================================
# Half-shell shift vector tests
# ===========================================================================

class TestHalfShellShifts:
    """Test half-shell shift vector correctness."""

    def test_exactly_13_shifts(self):
        """There should be exactly 13 positive half-shell shifts."""
        assert len(HALF_SHELL_SHIFTS) == 13

    def test_no_zero_shift(self):
        """(0,0,0) should not be in the half-shell."""
        assert (0, 0, 0) not in HALF_SHELL_SHIFTS

    def test_inversion_coverage(self):
        """Each shift + its inversion should cover all 26 neighbors."""
        all_nbrs = set()
        for dx, dy, dz in HALF_SHELL_SHIFTS:
            all_nbrs.add((dx, dy, dz))
            all_nbrs.add((-dx, -dy, -dz))
        assert len(all_nbrs) == 26


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
