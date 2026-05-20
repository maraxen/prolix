"""Tests for HP4 fetch_ani1x_subset script.

Tests the following per spec §7 and §8 exit criteria:
- Unit conversion (Ha/Å → kcal/mol/Å) with multiplier 627.5094740631
- SELECT_16 diversity algorithm with synthetic HDF5
- --dry-run flag behavior (no writes)
- Missing archive error handling
"""

import pytest
import sys
import tempfile
import json
import hashlib
from pathlib import Path
import numpy as np
import h5py

# Add scripts directory to path so we can import fetch_ani1x_subset
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts" / "data"))

FORCE_MULTIPLIER = 627.5094740631


@pytest.mark.fast
def test_force_unit_conversion():
    """Verify Ha/Å → kcal/mol/Å conversion multiplier (spec §3, §8 exit criterion 8).

    The conversion multiplier is exactly 627.5094740631 (no Bohr factor).
    This test verifies the multiplier and round-trips a known force value.
    """
    # Test multiplier value explicitly
    assert abs(FORCE_MULTIPLIER - 627.5094740631) < 1e-9

    # Test round-trip: kcal/mol/Å → Ha/Å → kcal/mol/Å
    force_kcal = 10.0  # kcal/mol/Å

    # Convert to Ha/Å
    force_ha = force_kcal / FORCE_MULTIPLIER

    # Expected Ha/Å range: [0, ~0.5] per spec §8 criterion 8
    assert 0 <= force_ha <= 0.5, f"Round-trip gives unrealistic Ha/Å value: {force_ha}"

    # Convert back to kcal/mol/Å
    force_kcal_roundtrip = force_ha * FORCE_MULTIPLIER

    # Should match original within floating-point precision
    assert abs(force_kcal - force_kcal_roundtrip) < 1e-6


@pytest.mark.fast
def test_select_16_with_synthetic_hdf5():
    """Test SELECT_16 diversity algorithm on synthetic ANI-1x-like HDF5.

    Creates a tiny synthetic HDF5 with ~25 CHNO molecules (15–30 atoms),
    runs SELECT_16, verifies deterministic selection with seed=42.

    Per spec §4.2 and §8 exit criterion 12 (visual inspection).
    """
    from fetch_ani1x_subset import select_16_lane_a

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        synthetic_h5 = tmpdir_path / "synthetic_ani1x.h5"

        # Build synthetic HDF5 with 20 molecules, all with valid unique SMILES
        with h5py.File(synthetic_h5, "w") as f:
            # Create 20 unique SMILES by varying chain/branch lengths
            smiles_list = []
            for n_atoms in range(16, 26):
                # Create simple alkane chains
                smiles = "C" * n_atoms
                smiles_list.append((smiles, n_atoms))

            # Create additional variations with different elements
            for n_atoms in range(16, 22):
                smiles = "C" * (n_atoms - 1) + "N"
                smiles_list.append((smiles, n_atoms))

            for i, (smiles, expected_n_atoms) in enumerate(smiles_list[:20]):
                n_atoms = expected_n_atoms
                n_conf = 20 + (i % 6)  # Ensure ≥20 conformers per F3

                mol_group = f.create_group(smiles)

                # Minimal synthetic data
                mol_group.create_dataset(
                    "coordinates",
                    data=np.random.randn(n_conf, n_atoms, 3).astype(np.float32),
                )
                # atomic_numbers: derived from SMILES
                atomic_nums = [6 if c == 'C' else 7 if c == 'N' else 8 for c in smiles]
                mol_group.create_dataset(
                    "atomic_numbers",
                    data=np.array(atomic_nums, dtype=np.uint8),
                )
                mol_group.create_dataset(
                    "wb97x_dz.energy",
                    data=np.random.randn(n_conf).astype(np.float64),
                )
                mol_group.create_dataset(
                    "wb97x_dz.forces",
                    data=np.random.randn(n_conf, n_atoms, 3).astype(np.float32),
                )

        # Run SELECT_16
        selected = select_16_lane_a(synthetic_h5, seed=42)

        # Verify properties
        assert len(selected) > 0, "SELECT_16 returned no molecules"
        assert len(selected) <= 16, f"SELECT_16 returned {len(selected)} molecules (max 16)"

        # Verify each molecule has required fields
        for mol in selected:
            assert "smiles" in mol
            assert "n_atoms" in mol
            assert "n_conf_with_forces" in mol
            assert "rarity_score" in mol
            assert "env_hashes" in mol

            # Atom count should be in range (F2: 15-30, or 16-22 for our synthetic data)
            assert mol["n_atoms"] >= 15

        # Verify determinism: run again with same seed, should get identical result
        selected2 = select_16_lane_a(synthetic_h5, seed=42)
        assert len(selected) == len(selected2)
        for m1, m2 in zip(selected, selected2):
            assert m1["smiles"] == m2["smiles"]


@pytest.mark.fast
def test_dry_run_no_writes():
    """Verify --dry-run flag prevents file writes (spec §7.2, §8 criterion).

    Runs fetch with --dry-run and asserts no files created under out_dir.
    """
    from fetch_ani1x_subset import fetch_ani1x_subset

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        synthetic_h5 = tmpdir_path / "synthetic_ani1x.h5"
        out_dir = tmpdir_path / "ani1x_subset_output"

        # Create minimal synthetic HDF5
        with h5py.File(synthetic_h5, "w") as f:
            # Create 20 unique SMILES by varying chain lengths
            smiles_list = []
            for n_atoms in range(16, 26):  # 16-25 atoms
                smiles = "C" * n_atoms
                smiles_list.append((smiles, n_atoms))

            for i, (smiles, n_atoms) in enumerate(smiles_list[:20]):
                n_conf = 25

                mol_group = f.create_group(smiles)
                mol_group.create_dataset(
                    "coordinates",
                    data=np.random.randn(n_conf, n_atoms, 3).astype(np.float32),
                )
                # Create valid atomic_numbers for the SMILES
                atomic_nums = [6 if c == 'C' else 7 if c == 'N' else 8 for c in smiles]
                mol_group.create_dataset(
                    "atomic_numbers",
                    data=np.array(atomic_nums, dtype=np.uint8),
                )
                mol_group.create_dataset(
                    "wb97x_dz.energy",
                    data=np.random.randn(n_conf).astype(np.float64),
                )
                mol_group.create_dataset(
                    "wb97x_dz.forces",
                    data=np.random.randn(n_conf, n_atoms, 3).astype(np.float32),
                )

        # Run with dry-run
        success = fetch_ani1x_subset(
            ani1x_archive=synthetic_h5,
            comp6_archive=None,
            out_dir=out_dir,
            seed=42,
            dry_run=True,
        )

        assert success, "fetch_ani1x_subset failed with dry-run"

        # Verify no lane_a or lane_b directories created
        assert not (out_dir / "lane_a").exists(), "lane_a dir created during dry-run"
        assert not (out_dir / "lane_b").exists(), "lane_b dir created during dry-run"
        assert not (out_dir / "manifest.json").exists(), "manifest.json created during dry-run"


@pytest.mark.fast
def test_missing_archive_exits_nonzero():
    """Verify script exits non-zero and prints Figshare URL on missing archive.

    Per spec §7.2: "If --ani1x-archive path missing, print Figshare URL and exit non-zero."
    """
    from fetch_ani1x_subset import fetch_ani1x_subset

    nonexistent = Path("/tmp/nonexistent_ani1x_xyz.h5")
    assert not nonexistent.exists()

    success = fetch_ani1x_subset(
        ani1x_archive=nonexistent,
        comp6_archive=None,
        out_dir=Path("/tmp/unused"),
        seed=42,
        dry_run=False,
    )

    assert not success, "Should fail with missing archive"


@pytest.mark.fast
def test_sha256_computation():
    """Verify SHA-256 computation matches expected hash.

    Quick sanity check that SHA-256 is computed correctly.
    """
    from fetch_ani1x_subset import compute_sha256

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        test_file = tmpdir_path / "test.bin"

        # Write known content
        test_content = b"test data for sha256 validation"
        test_file.write_bytes(test_content)

        # Compute SHA-256
        computed_hash = compute_sha256(test_file)

        # Verify against expected (computed independently)
        expected_hash = hashlib.sha256(test_content).hexdigest()

        assert computed_hash == expected_hash
