"""Tests for MolecularBundle.from_pdb factory classmethod.

Verifies that:
- from_pdb loads PDB files correctly
- File validation (existence, forcefield support) works
- Error handling covers missing files and unsupported forcefields
"""

from pathlib import Path
import tempfile

import pytest
import jax.numpy as jnp

from prolix.types.bundles import MolecularBundle


def test_bundle_from_pdb_wrong_path_raises():
    """from_pdb raises FileNotFoundError for nonexistent PDB files."""
    with pytest.raises(FileNotFoundError, match="PDB not found"):
        MolecularBundle.from_pdb("/nonexistent/path/to/molecule.pdb")


def test_bundle_from_pdb_bad_forcefield_raises():
    """from_pdb raises ValueError for unsupported forcefields."""
    # Create a temporary valid PDB file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        # Minimal valid PDB format
        f.write("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n")
        f.write("END\n")
        temp_pdb = f.name

    try:
        with pytest.raises(ValueError, match="Unsupported forcefield"):
            MolecularBundle.from_pdb(temp_pdb, forcefield="unsupported_ff")
    finally:
        Path(temp_pdb).unlink()


@pytest.mark.skipif(
    _is_parmed_available := (
        lambda: (
            __import__("importlib.util").util.find_spec("parmed") is not None
        )
    )() is False,
    reason="parmed not installed"
)
def test_bundle_from_pdb_returns_bundle():
    """from_pdb loads a valid PDB and returns a MolecularBundle."""
    # Create a minimal valid PDB file with a single alanine residue
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
        # Standard PDB format for a 3-atom system
        f.write("REMARK   1  CREATED WITH OPEN BABEL\n")
        f.write("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n")
        f.write("ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n")
        f.write("ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C\n")
        f.write("END\n")
        temp_pdb = f.name

    try:
        result = MolecularBundle.from_pdb(temp_pdb, forcefield="amber14")
        assert isinstance(result, MolecularBundle)
        assert jnp.all(jnp.isfinite(result.positions))
        assert result.n_atoms > 0
    finally:
        Path(temp_pdb).unlink()


def _is_parmed_available() -> bool:
    """Check if parmed is available."""
    try:
        import parmed  # noqa: F401
        return True
    except ImportError:
        return False
