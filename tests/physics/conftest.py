"""Shared fixtures for physics tests."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from proxide.core.containers import Protein
from proxide import OutputSpec, parse_structure


@pytest.fixture(autouse=True)
def _enable_x64():
  """Enable float64 for physics tests via a proper fixture, avoiding global contamination."""
  jax.config.update("jax_enable_x64", True)
  yield
  # Note: JAX x64 flag is global and cannot be easily reverted once set.
  # We leave it True here; tests that require float32 should set it explicitly.


@pytest.fixture
def simple_charges() -> jax.Array:
  """Simple charge distribution for testing."""
  return jnp.array([1.0, -1.0, 0.5, -0.5])


@pytest.fixture
def simple_positions() -> jax.Array:
  """Simple atomic positions for testing."""
  return jnp.array(
    [
      [0.0, 0.0, 0.0],
      [5.0, 0.0, 0.0],
      [0.0, 5.0, 0.0],
      [5.0, 5.0, 0.0],
    ]
  )


@pytest.fixture
def backbone_positions_single_residue() -> jax.Array:
  """Backbone positions for a single idealized residue [N, CA, C, O, CB]."""
  return jnp.array(
    [
      [
        [0.0, 0.0, 0.0],  # N
        [1.5, 0.0, 0.0],  # CA
        [2.5, 1.0, 0.0],  # C
        [2.5, 2.0, 0.0],  # O
        [1.5, 0.0, 1.5],  # CB (perpendicular to backbone plane)
      ]
    ]
  )


@pytest.fixture
def backbone_charges_single_residue() -> jax.Array:
  """Backbone charges for a single residue [N, CA, C, O, CB]."""
  return jnp.array([[-0.3, 0.1, 0.5, -0.5, 0.2]])


@pytest.fixture
def backbone_positions_multi_residue() -> jax.Array:
  """Backbone positions for multiple residues."""
  # Create 3 residues in a simple extended conformation
  residue1 = jnp.array(
    [
      [0.0, 0.0, 0.0],
      [1.5, 0.0, 0.0],
      [2.5, 1.0, 0.0],
      [2.5, 2.0, 0.0],
      [1.5, 0.0, 1.5],
    ]
  )

  residue2 = jnp.array(
    [
      [3.8, 1.5, 0.0],
      [5.3, 1.5, 0.0],
      [6.3, 2.5, 0.0],
      [6.3, 3.5, 0.0],
      [5.3, 1.5, 1.5],
    ]
  )

  residue3 = jnp.array(
    [
      [7.6, 3.0, 0.0],
      [9.1, 3.0, 0.0],
      [10.1, 4.0, 0.0],
      [10.1, 5.0, 0.0],
      [9.1, 3.0, 1.5],
    ]
  )

  return jnp.stack([residue1, residue2, residue3])


@pytest.fixture
def backbone_charges_multi_residue() -> jax.Array:
  """Backbone charges for multiple residues."""
  # Charges for N, CA, C, O, CB
  residue_charges = jnp.array([-0.3, 0.1, 0.5, -0.5, 0.2])
  return jnp.tile(residue_charges, (3, 1))  # 3 residues


@pytest.fixture
def lj_parameters() -> dict[str, jax.Array]:
  """Simple LJ parameters for testing."""
  return {
    "sigma": jnp.array([3.5, 3.0, 2.5, 3.2]),
    "epsilon": jnp.array([0.1, 0.15, 0.08, 0.12]),
  }


@pytest.fixture
def temp_ff_dir(tmp_path: Path) -> Path:
  """Create a temporary directory for force field files."""
  ff_dir = tmp_path / "force_fields"
  ff_dir.mkdir()
  return ff_dir


# Force field path for parse_structure (needed to get Atom37 format)
_FF_PATH = (
  Path(__file__).parent.parent.parent.parent
  / "proxide"
  / "src"
  / "proxide"
  / "assets"
  / "protein.ff19SB.xml"
)

# Test PDB data directory
_DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def pqr_protein() -> Protein:
  """Load a protein with mock charges for physics feature tests.

  Uses parse_structure with parameterize_md=True to get Atom37 format
  (required by compute_backbone_coordinates), then overrides charges
  to match full_coordinates length for correct broadcasting.
  """
  pdb_path = _DATA_DIR / "1uao.pdb"

  spec = OutputSpec()
  spec.parameterize_md = True
  spec.add_hydrogens = False
  spec.force_field = str(_FF_PATH)

  protein = parse_structure(str(pdb_path), spec)
  assert protein.full_coordinates is not None, "Protein must have full_coordinates"
  assert protein.coordinates.ndim == 3, "Must be Atom37 format (N_res, 37, 3)"

  # Override charges to match full_coordinates count (N_res * 37)
  # Needed because parameterization assigns charges to real atoms only,
  # but full_coordinates includes Atom37 padding slots.
  # Also clear non-JIT-safe string/list fields to prevent tracer errors.
  n_atoms = protein.full_coordinates.reshape(-1, 3).shape[0]
  mock_charges = jnp.linspace(-0.5, 0.5, n_atoms)
  protein = protein.replace(
    charges=mock_charges,
    source=None,
    format=None,
    atom_names=None,
    atom_types=None,
    res_names=None,
    chain_ids=None,
    elements=None,
  )
  return protein
