"""Tests for MolecularBundle.from_system_dict factory."""

import warnings

import jax.numpy as jnp
import pytest

from prolix.types.bundles import MolecularBundle


class TestFromSystemDict:
    """Test MolecularBundle.from_system_dict classmethod."""

    def test_from_system_dict_warns(self):
        """Assert DeprecationWarning is emitted when calling from_system_dict."""
        # Minimal valid dict for 3-atom water
        d = {
            "positions": jnp.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
            "masses": jnp.array([16.0, 1.0, 1.0]),
            "charges": jnp.array([-0.834, 0.417, 0.417]),
            "sigmas": jnp.array([3.16, 2.65, 2.65]),
            "epsilons": jnp.array([0.16, 0.05, 0.05]),
            "radii": jnp.array([1.5, 1.2, 1.2]),
            "scaled_radii": jnp.array([1.2, 1.0, 1.0]),
            "bonds": jnp.zeros((0, 2), dtype=jnp.int32),
            "bond_params": jnp.zeros((0, 2), dtype=jnp.float32),
            "angles": jnp.zeros((0, 3), dtype=jnp.int32),
            "angle_params": jnp.zeros((0, 2), dtype=jnp.float32),
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bundle = MolecularBundle.from_system_dict(d)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()

    def test_from_system_dict_roundtrip(self):
        """Test roundtrip: dict -> MolecularBundle returns correct bundle."""
        # Minimal valid dict for 3-atom water
        positions = jnp.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
        d = {
            "positions": positions,
            "masses": jnp.array([16.0, 1.0, 1.0]),
            "charges": jnp.array([-0.834, 0.417, 0.417]),
            "sigmas": jnp.array([3.16, 2.65, 2.65]),
            "epsilons": jnp.array([0.16, 0.05, 0.05]),
            "radii": jnp.array([1.5, 1.2, 1.2]),
            "scaled_radii": jnp.array([1.2, 1.0, 1.0]),
            "bonds": jnp.zeros((0, 2), dtype=jnp.int32),
            "bond_params": jnp.zeros((0, 2), dtype=jnp.float32),
            "angles": jnp.zeros((0, 3), dtype=jnp.int32),
            "angle_params": jnp.zeros((0, 2), dtype=jnp.float32),
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bundle = MolecularBundle.from_system_dict(d)

        # Verify the bundle is a MolecularBundle
        assert isinstance(bundle, MolecularBundle)

        # Verify positions shape: should be padded to bucket size
        assert bundle.positions.shape[0] >= 3
        assert bundle.positions.shape[1] == 3

        # Verify n_atoms
        assert bundle.n_atoms == 3

        # Verify the first 3 positions match (rest are padded)
        assert jnp.allclose(bundle.positions[:3], positions)

        # Verify atom_mask: first 3 should be True, rest False
        assert jnp.sum(bundle.atom_mask) == 3

    def test_from_system_dict_invalid_key(self):
        """Test graceful error for missing required key."""
        # Missing 'positions' key
        d = {
            "masses": jnp.array([16.0, 1.0, 1.0]),
            "charges": jnp.array([-0.834, 0.417, 0.417]),
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(AttributeError):
                MolecularBundle.from_system_dict(d)

    def test_from_system_dict_boundary_condition(self):
        """Test that boundary_condition is passed through correctly."""
        d = {
            "positions": jnp.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
            "masses": jnp.array([16.0, 1.0, 1.0]),
            "charges": jnp.array([-0.834, 0.417, 0.417]),
            "sigmas": jnp.array([3.16, 2.65, 2.65]),
            "epsilons": jnp.array([0.16, 0.05, 0.05]),
            "radii": jnp.array([1.5, 1.2, 1.2]),
            "scaled_radii": jnp.array([1.2, 1.0, 1.0]),
            "bonds": jnp.zeros((0, 2), dtype=jnp.int32),
            "bond_params": jnp.zeros((0, 2), dtype=jnp.float32),
            "angles": jnp.zeros((0, 3), dtype=jnp.int32),
            "angle_params": jnp.zeros((0, 2), dtype=jnp.float32),
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bundle_periodic = MolecularBundle.from_system_dict(d, boundary_condition="periodic")
            bundle_free = MolecularBundle.from_system_dict(d, boundary_condition="free")

        # Verify that boundary_condition is stored in shape_spec
        assert bundle_periodic.shape_spec.boundary_condition == "periodic"
        assert bundle_free.shape_spec.boundary_condition == "free"
