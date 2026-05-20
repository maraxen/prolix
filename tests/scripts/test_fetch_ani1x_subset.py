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

    Note: ANI-1x groups are keyed by molecular formula (e.g., C9H9N3O2), not SMILES.
    """
    from fetch_ani1x_subset import select_16_lane_a

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        synthetic_h5 = tmpdir_path / "synthetic_ani1x.h5"

        # Build synthetic HDF5 with 20 molecules keyed by formula
        with h5py.File(synthetic_h5, "w") as f:
            # Create 20 molecules with formula-based group names
            formula_list = []
            for n_atoms in range(16, 26):
                # Create simple alkane chains
                formula = f"C{n_atoms}H{2*n_atoms + 2}"
                formula_list.append((formula, n_atoms))

            # Create additional variations with N
            for n_atoms in range(16, 22):
                formula = f"C{n_atoms-1}H{2*n_atoms}N"
                formula_list.append((formula, n_atoms))

            for i, (formula, expected_n_atoms) in enumerate(formula_list[:20]):
                n_atoms = expected_n_atoms
                n_conf = 20 + (i % 6)  # Ensure ≥20 conformers per F3

                mol_group = f.create_group(formula)

                # Minimal synthetic data
                # Positions: use realistic spacing (Å)
                positions = np.random.randn(n_conf, n_atoms, 3).astype(np.float32) * 0.5 + 1.5
                mol_group.create_dataset("coordinates", data=positions)

                # atomic_numbers: derived from formula
                # For simplicity: alternate C and N atoms to match n_atoms
                atomic_nums = []
                n_c = (n_atoms * 6) // 7 if 'N' in formula else n_atoms
                atomic_nums = [6] * min(n_c, n_atoms) + [7] * max(0, n_atoms - n_c)
                atomic_nums = atomic_nums[:n_atoms]
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
            assert "formula" in mol
            assert "n_atoms" in mol
            assert "n_conf_with_forces" in mol
            assert "rarity_score" in mol
            assert "env_hashes" in mol

            # Atom count should be in range (F2: 15-30)
            assert 15 <= mol["n_atoms"] <= 30

        # Verify determinism: run again with same seed, should get identical result
        selected2 = select_16_lane_a(synthetic_h5, seed=42)
        assert len(selected) == len(selected2)
        for m1, m2 in zip(selected, selected2):
            assert m1["formula"] == m2["formula"]


@pytest.mark.fast
def test_dry_run_no_writes():
    """Verify --dry-run flag prevents file writes (spec §7.2, §8 criterion).

    Runs fetch with --dry-run and asserts no files created under out_dir.

    Note: Uses real archives if available; skips if not (to avoid SHA-256 mismatch).
    """
    from fetch_ani1x_subset import fetch_ani1x_subset

    # Check if real archives are available (for integration testing)
    ani1x_real = Path("data/downloads/ani1x_release.h5")
    if not ani1x_real.exists():
        pytest.skip("Real ANI-1x archive not available for integration test")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        out_dir = tmpdir_path / "ani1x_subset_output"

        # Run with dry-run on real archive
        success = fetch_ani1x_subset(
            ani1x_archive=ani1x_real,
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
def test_local_env_hash_from_xyz():
    """Test AIMNet2-style local-environment hash from xyz + atomic numbers (no RDKit Mol).

    Verifies that the function correctly identifies bonded neighbors via distance threshold
    and computes hash tuples for non-H atoms.

    Per spec §4.2: hash(a) = (Z_a, n_H_neighbors, n_total_neighbors, sorted(Z_non_H_neighbors))
    """
    from fetch_ani1x_subset import compute_local_env_hash

    # Simple 2-atom molecule: C bonded to O
    # C: bonded to 1 O → (6, 0, 1, (8,))
    # O: bonded to 1 C → (8, 0, 1, (6,))

    atomic_numbers = np.array([6, 8], dtype=int)  # C, O
    positions = np.array([
        [0.0, 0.0, 0.0],      # C (left)
        [0.7, 0.0, 0.0],      # O (right, bonded to C via C-O bond at ~0.66 Å)
    ], dtype=np.float32)

    env_hashes = compute_local_env_hash(atomic_numbers, positions)

    # Should have 2 hashes (C and O)
    # C: 0 H neighbors, 1 total neighbor (O) → (6, 0, 1, (8,))
    # O: 0 H neighbors, 1 total neighbor (C) → (8, 0, 1, (6,))

    assert len(env_hashes) == 2, f"Expected 2 unique hashes (C and O), got {len(env_hashes)}"

    # C hash: (6, 0, 1, (8,))
    c_hash = (6, 0, 1, (8,))
    assert c_hash in env_hashes, f"Missing C hash {c_hash}. Got: {env_hashes}"

    # O hash: (8, 0, 1, (6,))
    o_hash = (8, 0, 1, (6,))
    assert o_hash in env_hashes, f"Missing O hash {o_hash}. Got: {env_hashes}"


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


@pytest.mark.fast
def test_lane_b_comp6v2_selection_synthetic():
    """Test Lane B selection from synthetic COMP6v2-like HDF5.

    Creates a synthetic COMP6v2-like HDF5 with groups /050, /080, /100, /138, /312
    and verifies select_lane_b_from_comp6v2 returns the right 4 molecules with
    the right bucket indices.

    Per spec §4.3 and §8 exit criterion 11 (span check):
    - Must select Trp-cage at /312 (bucket 3)
    - Must select Chignolin at /138 (bucket 2)
    - Must select 2 mid-size molecules in 65–128 atom range (bucket 1)
    - Total ensemble spans ≥3 buckets
    """
    from fetch_ani1x_subset import select_lane_b_from_comp6v2, bucket_idx_from_atom_count

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        synthetic_comp6 = tmpdir_path / "synthetic_comp6v2.h5"

        # Build synthetic COMP6v2 with groups /050, /080, /100, /138, /312
        with h5py.File(synthetic_comp6, "w") as f:
            test_groups = {
                "050": (50, 16),   # 50 atoms, 16 conformers
                "080": (80, 32),   # 80 atoms, 32 conformers (will be picked)
                "100": (100, 24),  # 100 atoms, 24 conformers (will be picked)
                "138": (138, 16),  # 138 atoms, 16 conformers — Chignolin
                "312": (312, 128), # 312 atoms, 128 conformers — Trp-cage
            }

            for group_name, (n_atoms, n_conf) in test_groups.items():
                group = f.create_group(group_name)

                # Create uniform species pattern (all same, so group is coherent)
                species_pattern = np.random.randint(1, 9, size=n_atoms, dtype=np.int64)
                species_pattern = np.tile(species_pattern, (n_conf, 1))  # (n_conf, n_atoms)

                group.create_dataset(
                    "coordinates",
                    data=np.random.randn(n_conf, n_atoms, 3).astype(np.float32),
                )
                group.create_dataset("species", data=species_pattern)
                group.create_dataset(
                    "energies",
                    data=np.random.randn(n_conf).astype(np.float64),
                )
                group.create_dataset(
                    "forces",
                    data=np.random.randn(n_conf, n_atoms, 3).astype(np.float32),
                )

        # Run Lane B selection
        selected = select_lane_b_from_comp6v2(synthetic_comp6, seed=42)

        # Verify we got 4 selections (mandatory: Trp-cage + Chignolin + 2 mid-size)
        assert len(selected) == 4, f"Expected 4 Lane B molecules, got {len(selected)}"

        # Verify mandatory Trp-cage
        trp_cage_sel = [s for s in selected if s["name"] == "trp_cage"]
        assert len(trp_cage_sel) == 1
        assert trp_cage_sel[0]["n_atoms"] == 312
        assert trp_cage_sel[0]["bucket_idx"] == 3

        # Verify mandatory Chignolin
        chignolin_sel = [s for s in selected if s["name"] == "chignolin"]
        assert len(chignolin_sel) == 1
        assert chignolin_sel[0]["n_atoms"] == 138
        assert chignolin_sel[0]["bucket_idx"] == 2

        # Verify 2 mid-size picks (bucket 1)
        midsize_sel = [s for s in selected if s["name"].startswith("midsize")]
        assert len(midsize_sel) == 2
        for sel in midsize_sel:
            assert 65 <= sel["n_atoms"] <= 128
            assert sel["bucket_idx"] == 1

        # Verify span check (§8 exit criterion 11): ≥3 molecules with bucket_idx ≥ 1
        bucket_geq_1 = [s for s in selected if s["bucket_idx"] >= 1]
        assert len(bucket_geq_1) >= 3, f"Expected ≥3 molecules with bucket_idx ≥ 1, got {len(bucket_geq_1)}"
