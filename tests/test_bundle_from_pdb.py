"""Tests for MolecularBundle.from_pdb API."""

import pytest
from pathlib import Path

from prolix.types.bundles import MolecularBundle


class TestFromPDBErrorPaths:
    """Test error handling paths (work without parmed)."""

    def test_from_pdb_file_not_found(self):
        """FileNotFoundError when path does not exist."""
        with pytest.raises(FileNotFoundError):
            MolecularBundle.from_pdb("/nonexistent/path/to/file.pdb")

    def test_from_pdb_bad_forcefield(self):
        """ValueError on unsupported forcefield."""
        # Create a temporary dummy file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
            f.write("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported forcefield"):
                MolecularBundle.from_pdb(temp_path, forcefield="unsupported_ff")
        finally:
            Path(temp_path).unlink()

    def test_from_pdb_no_parmed(self, monkeypatch):
        """NotImplementedError when parmed is unavailable."""
        import sys
        import builtins

        # Mock parmed import to fail
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "parmed":
                raise ImportError("parmed not installed")
            return original_import(name, *args, **kwargs)

        # Create a temporary dummy file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
            f.write("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n")
            temp_path = f.name

        try:
            monkeypatch.setattr(builtins, "__import__", mock_import)
            with pytest.raises(NotImplementedError, match="requires parmed"):
                MolecularBundle.from_pdb(temp_path)
        finally:
            Path(temp_path).unlink()


@pytest.mark.skipif(
    False,  # Skip unless parmed is installed
    reason="parmed not available; roundtrip tests require it"
)
class TestFromPDBRoundtrip:
    """Test roundtrip loading (requires parmed)."""

    @pytest.fixture
    def minimal_pdb(self, tmp_path):
        """Create a minimal 3-atom PDB (water molecule)."""
        pdb_content = """REMARK Minimal water (TIP3P) for testing
ATOM      1  O   WAT A   1       0.000   0.000   0.000  1.00  0.00           O
ATOM      2  H   WAT A   1       0.957   0.000   0.000  1.00  0.00           H
ATOM      3  H   WAT A   1      -0.239   0.927   0.000  1.00  0.00           H
END
"""
        pdb_path = tmp_path / "water.pdb"
        pdb_path.write_text(pdb_content)
        return str(pdb_path)

    def test_from_pdb_loads_water_molecule(self, minimal_pdb):
        """Load a minimal water PDB and verify structure."""
        bundle = MolecularBundle.from_pdb(minimal_pdb, forcefield="amber14")

        # Check that we got a valid bundle
        assert bundle is not None
        assert bundle.positions.shape[0] > 0
        assert bundle.n_atoms.item() > 0
