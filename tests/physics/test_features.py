"""Tests for physics-based node features."""
import chex
import jax
import jax.numpy as jnp
import pytest
from proxide.core.containers import Protein
from proxide.physics.features import (
    compute_electrostatic_features_batch,
    compute_electrostatic_node_features,
)


@pytest.mark.parametrize("jit_compile", [True, False], ids=["jit", "eager"])
def test_compute_electrostatic_node_features_shape(
    pqr_protein: Protein, jit_compile,
):
    """Test that the computed features have the correct shape."""
    fn = compute_electrostatic_node_features
    if jit_compile:
        fn = jax.jit(fn)

    features = fn(pqr_protein)
    n_residues = pqr_protein.coordinates.shape[0]
    chex.assert_shape(features, (n_residues, 5))
    chex.assert_tree_all_finite(features)


def test_compute_electrostatic_node_features_no_charges(
    pqr_protein: Protein,
):
    """Test that a ValueError is raised if protein has no charges."""
    protein_no_charges = pqr_protein.replace(charges=None)
    with pytest.raises(ValueError, match="must have charges"):
        compute_electrostatic_node_features(protein_no_charges)


def test_compute_electrostatic_node_features_no_full_coordinates(
    pqr_protein: Protein,
):
    """Test that a ValueError is raised if protein has no full_coordinates."""
    protein_no_full_coords = pqr_protein.replace(full_coordinates=None)
    with pytest.raises(ValueError, match="must have full_coordinates"):
        compute_electrostatic_node_features(protein_no_full_coords)


@pytest.mark.parametrize("jit_compile", [True, False], ids=["jit", "eager"])
def test_compute_electrostatic_node_features_jittable(
    pqr_protein: Protein, jit_compile,
):
    """Test that the feature computation can be JIT compiled."""
    fn = compute_electrostatic_node_features
    if jit_compile:
        fn = jax.jit(fn)
    features = fn(pqr_protein)
    chex.assert_tree_all_finite(features)


@pytest.mark.parametrize("jit_compile", [True, False], ids=["jit", "eager"])
def test_compute_electrostatic_features_batch_shape(
    pqr_protein: Protein, jit_compile,
):
    """Test that the batched features have the correct shape."""
    proteins = [pqr_protein, pqr_protein]

    fn = compute_electrostatic_features_batch
    if jit_compile:
        fn = jax.jit(fn)

    features, mask = fn(proteins)
    n_residues = pqr_protein.coordinates.shape[0]
    chex.assert_shape(features, (2, n_residues, 5))
    chex.assert_shape(mask, (2, n_residues))
    chex.assert_trees_all_close(mask, jnp.ones_like(mask))


@pytest.mark.parametrize("jit_compile", [True, False], ids=["jit", "eager"])
def test_compute_electrostatic_features_batch_padding(
    pqr_protein: Protein, jit_compile,
):
    """Test that padding is applied correctly."""
    proteins = [pqr_protein]

    n_residues = pqr_protein.coordinates.shape[0]
    max_length = n_residues + 10

    fn = compute_electrostatic_features_batch
    if jit_compile:
        fn = jax.jit(fn, static_argnames=["max_length"])

    features, mask = fn(proteins, max_length=max_length)
    chex.assert_shape(features, (1, max_length, 5))
    chex.assert_shape(mask, (1, max_length))
    chex.assert_trees_all_close(jnp.sum(mask), n_residues)


def test_compute_electrostatic_features_batch_empty_list():
    """Test that an empty list of proteins raises a ValueError."""
    with pytest.raises(ValueError, match="Must provide at least one protein"):
        compute_electrostatic_features_batch([])


def test_compute_electrostatic_features_batch_max_length_too_small(
    pqr_protein: Protein,
):
    """Test that a small max_length raises a ValueError."""
    proteins = [pqr_protein]
    max_length = pqr_protein.coordinates.shape[0] - 1
    with pytest.raises(ValueError, match="is less than longest sequence"):
        compute_electrostatic_features_batch(proteins, max_length=max_length)


def test_compute_electrostatic_node_features_thermal_mode(
    pqr_protein: Protein,
):
    """Test that thermal mode works correctly."""
    # Use a dummy key for noise
    key = jax.random.key(0)

    # Mode: direct
    sigma_direct = 1.0
    features_direct = compute_electrostatic_node_features(
        pqr_protein,
        noise_scale=sigma_direct,
        noise_mode="direct",
        key=key,
    )

    # Mode: thermal
    # Calculate T such that sigma is roughly 1.0
    # sigma = sqrt(0.5 * R * T) => sigma^2 = 0.5 * R * T => T = sigma^2 / (0.5 * R)
    from proxide.physics.constants import BOLTZMANN_KCAL

    t = 1.0 / (0.5 * BOLTZMANN_KCAL)

    features_temp = compute_electrostatic_node_features(
        pqr_protein,
        noise_scale=t,
        noise_mode="thermal",
        key=key,
    )

    # The values should be close (floating point differences)
    chex.assert_trees_all_close(features_direct, features_temp, atol=1e-5)
