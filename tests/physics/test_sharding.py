"""Tests for multi-GPU sharding utilities."""

import jax.numpy as jnp
import pytest

from prolix.physics.sharding import pad_system_for_pmap, unpad_results


class TestPadSystemForPmap:
    """Tests for system padding to pmap-compatible sizes."""

    def test_pad_to_divisible_by_2(self):
        """Odd atom count should be padded to next even number."""
        positions = jnp.ones((5, 3))
        charges = jnp.ones(5)
        atom_mask = jnp.ones(5, dtype=jnp.bool_)

        result = pad_system_for_pmap(
            positions=positions,
            charges=charges,
            atom_mask=atom_mask,
            num_devices=2,
        )

        assert result.positions.shape == (6, 3)
        assert result.charges.shape == (6,)
        assert result.atom_mask.shape == (6,)
        # Ghost atoms should have zero charge
        assert result.charges[5] == 0.0
        # Ghost atoms should be masked out
        assert result.atom_mask[5] == False  # noqa: E712

    def test_no_pad_when_already_divisible(self):
        """Even atom count should not be padded."""
        positions = jnp.ones((6, 3))
        charges = jnp.ones(6)
        atom_mask = jnp.ones(6, dtype=jnp.bool_)

        result = pad_system_for_pmap(
            positions=positions,
            charges=charges,
            atom_mask=atom_mask,
            num_devices=2,
        )

        assert result.positions.shape == (6, 3)
        assert result.n_real == 6
        assert result.n_padded == 6

    def test_ghost_positions_far_away(self):
        """Ghost atom positions should be placed far from real atoms."""
        positions = jnp.zeros((3, 3))
        charges = jnp.ones(3)
        atom_mask = jnp.ones(3, dtype=jnp.bool_)

        result = pad_system_for_pmap(
            positions=positions,
            charges=charges,
            atom_mask=atom_mask,
            num_devices=2,
        )

        # Ghost atom at index 3 should be far away (>= 9999 A)
        ghost_pos = result.positions[3]
        assert jnp.linalg.norm(ghost_pos) >= 9999.0

    def test_sigmas_epsilons_padded_safely(self):
        """LJ parameters for ghost atoms should be near-zero to avoid interactions."""
        positions = jnp.ones((3, 3))
        charges = jnp.ones(3)
        sigmas = jnp.ones(3) * 3.0
        epsilons = jnp.ones(3) * 0.1
        atom_mask = jnp.ones(3, dtype=jnp.bool_)

        result = pad_system_for_pmap(
            positions=positions,
            charges=charges,
            atom_mask=atom_mask,
            sigmas=sigmas,
            epsilons=epsilons,
            num_devices=2,
        )

        # Ghost atom epsilon should be 0 (no LJ interactions)
        assert result.epsilons[3] == 0.0

    def test_exclusion_mask_padded(self):
        """Exclusion mask (N, N) should be padded with False for ghost rows/cols."""
        positions = jnp.ones((3, 3))
        charges = jnp.ones(3)
        atom_mask = jnp.ones(3, dtype=jnp.bool_)
        exclusion_mask = jnp.ones((3, 3), dtype=jnp.bool_)

        result = pad_system_for_pmap(
            positions=positions,
            charges=charges,
            atom_mask=atom_mask,
            exclusion_mask=exclusion_mask,
            num_devices=2,
        )

        assert result.exclusion_mask.shape == (4, 4)
        # Ghost row/col should be False
        assert result.exclusion_mask[3, 0] == False  # noqa: E712
        assert result.exclusion_mask[0, 3] == False  # noqa: E712

    def test_power_of_2_padding(self):
        """When power_of_2=True, pad to next power of 2."""
        positions = jnp.ones((5, 3))
        charges = jnp.ones(5)
        atom_mask = jnp.ones(5, dtype=jnp.bool_)

        result = pad_system_for_pmap(
            positions=positions,
            charges=charges,
            atom_mask=atom_mask,
            num_devices=2,
            power_of_2=True,
        )

        assert result.positions.shape[0] == 8  # next power of 2
        assert result.n_real == 5


class TestUnpadResults:
    """Tests for stripping ghost atoms from outputs."""

    def test_unpad_energy_unchanged(self):
        """Energy scalar should pass through unchanged."""
        energy = jnp.array(42.0)
        result = unpad_results(energy=energy, n_real=5, n_padded=8)
        assert result.energy == 42.0

    def test_unpad_forces(self):
        """Forces should be trimmed to real atom count."""
        forces = jnp.ones((8, 3))
        result = unpad_results(forces=forces, n_real=5, n_padded=8)
        assert result.forces.shape == (5, 3)

    def test_unpad_born_radii(self):
        """Born radii should be trimmed to real atom count."""
        born_radii = jnp.ones(8)
        result = unpad_results(born_radii=born_radii, n_real=5, n_padded=8)
        assert result.born_radii.shape == (5,)


from prolix.physics.sharding import shard_for_pmap, unshard_from_pmap


class TestShardForPmap:
    """Tests for splitting padded arrays across devices."""

    def test_shard_positions(self):
        """Positions (8, 3) with 2 devices -> (2, 4, 3)."""
        positions = jnp.arange(24).reshape(8, 3).astype(jnp.float32)
        sharded = shard_for_pmap(positions, num_devices=2)
        assert sharded.shape == (2, 4, 3)
        # First device gets atoms 0-3
        assert jnp.allclose(sharded[0], positions[:4])
        # Second device gets atoms 4-7
        assert jnp.allclose(sharded[1], positions[4:])

    def test_shard_1d(self):
        """1D array (8,) with 2 devices -> (2, 4)."""
        charges = jnp.ones(8)
        sharded = shard_for_pmap(charges, num_devices=2)
        assert sharded.shape == (2, 4)

    def test_shard_requires_divisible(self):
        """Should raise if N is not divisible by num_devices."""
        arr = jnp.ones(7)
        with pytest.raises(ValueError, match="divisible"):
            shard_for_pmap(arr, num_devices=2)


class TestUnshardFromPmap:
    """Tests for reassembling sharded outputs."""

    def test_unshard_2d(self):
        """(2, 4, 3) -> (8, 3)."""
        sharded = jnp.ones((2, 4, 3))
        result = unshard_from_pmap(sharded)
        assert result.shape == (8, 3)

    def test_unshard_1d(self):
        """(2, 4) -> (8,)."""
        sharded = jnp.ones((2, 4))
        result = unshard_from_pmap(sharded)
        assert result.shape == (8,)

    def test_unshard_scalar_sum(self):
        """(2,) scalar per device -> sum to single scalar."""
        sharded = jnp.array([3.0, 4.0])
        result = unshard_from_pmap(sharded, reduce="sum")
        assert jnp.allclose(result, 7.0)


import jax
from prolix.physics.sharding import sharded_coulomb_energy


class TestShardedCoulomb:
    """Tests that sharded Coulomb matches dense single-GPU Coulomb."""

    def test_parity_4_atoms(self):
        """Sharded 2-GPU Coulomb should match dense N^2 Coulomb."""
        key = jax.random.PRNGKey(0)
        positions = jax.random.normal(key, (4, 3)) * 5.0
        charges = jnp.array([0.5, -0.3, 0.2, -0.4])

        # Dense reference (single GPU)
        dr = positions[:, None, :] - positions[None, :, :]
        dist = jnp.sqrt(jnp.sum(dr**2, axis=-1) + 1e-12)
        dist_safe = dist + jnp.eye(4) * 1e6
        q_ij = charges[:, None] * charges[None, :]
        COULOMB_CONSTANT = 332.0637
        e_dense = 0.5 * COULOMB_CONSTANT * jnp.sum(q_ij / dist_safe)

        # Sharded (2 GPUs)
        atom_mask = jnp.ones(4, dtype=jnp.bool_)
        e_sharded = sharded_coulomb_energy(positions, charges, atom_mask, num_devices=2)

        assert jnp.allclose(e_dense, e_sharded, atol=1e-4), (
            f"Dense={e_dense:.6f} vs Sharded={e_sharded:.6f}"
        )

    def test_ghost_atoms_no_contribution(self):
        """Ghost atoms (mask=False) should not contribute to energy."""
        positions = jnp.array([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [9999.0, 9999.0, 9999.0],  # ghost
            [9999.0, 9999.0, 9999.0],  # ghost
        ])
        charges = jnp.array([1.0, -1.0, 0.0, 0.0])
        atom_mask = jnp.array([True, True, False, False])

        e_full = sharded_coulomb_energy(positions, charges, atom_mask, num_devices=2)

        # Reference: just 2 real atoms
        COULOMB_CONSTANT = 332.0637
        e_ref_exact = COULOMB_CONSTANT * (1.0 * -1.0) / 3.0

        assert jnp.allclose(e_full, e_ref_exact, atol=1e-3)
