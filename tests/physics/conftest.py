import pytest
import jax
import jax.numpy as jnp
import numpy as np

# Import fixtures from fixtures_openmm_parity for automatic discovery
try:
    from .fixtures_openmm_parity import ala_dip_reference
except ImportError:
    pass


@pytest.fixture
def regression_pme_params():
    """Shared PME knobs for OpenMM ↔ Prolix explicit-solvent parity tests."""
    from prolix.physics.regression_explicit_pme import REGRESSION_EXPLICIT_PME

    return dict(REGRESSION_EXPLICIT_PME)


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

@pytest.fixture(params=[jnp.float32, jnp.float64])
def dtype_mode(request):
    """Function-scoped dtype mode fixture.
    Parametrizes tests over f32 and f64. Saves/restores jax_enable_x64."""
    dtype = request.param
    x64_before = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", dtype == jnp.float64)
    yield dtype
    jax.config.update("jax_enable_x64", x64_before)

@pytest.fixture(scope="session", autouse=True)
def _enable_x64_for_physics_tests():
    """Session-scoped fixture to enable x64 precision for physics tests.

    This allows all fixtures and tests in this package to use float64 without
    per-function configuration. Avoids repetition of jax.config.update() calls.
    """
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)
