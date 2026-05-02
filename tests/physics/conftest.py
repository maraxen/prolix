import pytest
import jax.numpy as jnp
import numpy as np

@pytest.fixture
def simple_positions():
    """Fixture for simple particle positions (4 atoms)."""
    return jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=jnp.float32)

@pytest.fixture
def simple_charges():
    """Fixture for simple particle charges (4 atoms)."""
    return jnp.array([1.0, -1.0, 0.5, -0.5], dtype=jnp.float32)

@pytest.fixture
def backbone_positions_single_residue():
    """Fixture for backbone positions of a single residue (N, CA, C, O, CB).
    Shape (1, 5, 3).
    """
    return jnp.array([[
        [0.0, 0.0, 0.0],  # N
        [1.45, 0.0, 0.0], # CA
        [2.0, 1.4, 0.0],  # C
        [1.3, 2.4, 0.0],  # O
        [1.5, -1.0, 1.0]  # CB
    ]], dtype=jnp.float32)

@pytest.fixture
def backbone_positions_multi_residue():
    """Fixture for backbone positions of three residues.
    Shape (3, 5, 3).
    """
    res1 = np.array([
        [0.0, 0.0, 0.0], [1.45, 0.0, 0.0], [2.0, 1.4, 0.0], [1.3, 2.4, 0.0], [1.5, -1.0, 1.0]
    ])
    res2 = res1 + np.array([3.8, 0.0, 0.0])
    res3 = res1 + np.array([7.6, 0.0, 0.0])
    return jnp.array(np.stack([res1, res2, res3]), dtype=jnp.float32)
