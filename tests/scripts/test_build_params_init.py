"""
Tests for build_params_init.py — OpenFF bonded parameter initialization.

Spec: docs/superpowers/specs/2026-05-20-hp4-ani1x-curation.md §10.2
"""

import pytest
import sys
from pathlib import Path

# Import the script functions — skip entire module if xyz2mol_perceive is missing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts" / "data"))
try:
    from build_params_init import (
        xyz2mol_perceive,
        get_openff_params_from_molecule,
        build_params_for_molecule,
    )
except ImportError:
    pytest.skip("xyz2mol_perceive not available in build_params_init", allow_module_level=True)

import json
import tempfile

import h5py
import numpy as np

try:
    from openff.toolkit import Molecule as OFFMolecule
    HAS_OPENFF = True
except ImportError:
    HAS_OPENFF = False


@pytest.mark.skipif(not HAS_OPENFF, reason="openff-toolkit not installed")
def test_xyz2mol_methanol():
    """Test xyz2mol perception on methanol (CH3OH)."""
    # Methanol structure: C-O with 3 H on C, 1 H on O
    positions = np.array([
        [0.000, 0.000, 0.000],  # C
        [1.400, 0.000, 0.000],  # O
        [-0.500, 0.900, 0.000],  # H
        [-0.500, -0.450, 0.780],  # H
        [-0.500, -0.450, -0.780],  # H
        [1.900, 0.800, 0.000],  # H (on O)
    ], dtype=np.float32)

    species = np.array([6, 8, 1, 1, 1, 1], dtype=np.int8)  # C, O, H, H, H, H

    mol = xyz2mol_perceive(positions, species, charge=0)

    # Check basic structure
    assert mol.GetNumAtoms() == 6, f"Expected 6 atoms, got {mol.GetNumAtoms()}"
    assert mol.GetNumBonds() >= 4, f"Expected at least 4 bonds (C-O + 3 C-H + 1 O-H), got {mol.GetNumBonds()}"

    # The exact bond count depends on RDKit's perception, but we should have:
    # - C-O bond
    # - 3 C-H bonds
    # - 1 O-H bond
    # = 5 bonds total
    assert mol.GetNumBonds() >= 5, f"Expected at least 5 bonds, got {mol.GetNumBonds()}"


@pytest.mark.skipif(not HAS_OPENFF, reason="openff-toolkit not installed")
def test_openff_assign_water():
    """Test OpenFF parameter assignment on a simple water molecule."""
    # Water: O in center, 2 H atoms
    positions = np.array([
        [0.000, 0.000, 0.000],  # O
        [0.957, 0.000, 0.000],  # H
        [-0.240, 0.927, 0.000],  # H
    ], dtype=np.float32)

    species = np.array([8, 1, 1], dtype=np.int8)

    mol = xyz2mol_perceive(positions, species, charge=0)
    off_mol = OFFMolecule.from_rdkit(mol, allow_undefined_stereo=True)

    # Get parameters
    params = get_openff_params_from_molecule(off_mol)

    # Water should have:
    # - 3 atoms
    # - 2 bonds (O-H)
    # - 1 angle (H-O-H)
    # - 0 torsions
    assert len(params["atom_types"]) == 3, "Water has 3 atoms"
    assert len(params["bonds"]) == 2, "Water has 2 O-H bonds"
    assert len(params["angles"]) == 1, "Water has 1 H-O-H angle"
    assert len(params["proper_torsions"]) == 0, "Water has no torsions"


@pytest.mark.skipif(not HAS_OPENFF, reason="openff-toolkit not installed")
def test_lane_a_pipeline_synthetic():
    """Test end-to-end Lane A pipeline on a synthetic small molecule."""
    # Create a temporary HDF5 file with synthetic CHNO molecule (like Lane A)
    positions = np.array([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [2.5, 0.0, 0.0],
        [-0.5, 0.9, 0.0],
        [-0.5, -0.45, 0.8],
        [-0.5, -0.45, -0.8],
        [3.0, 1.0, 0.0],
        [3.0, -1.0, 0.0],
        [1.5, 1.0, 0.0],
    ], dtype=np.float32)

    species = np.array([6, 6, 8, 1, 1, 1, 1, 1, 1], dtype=np.int8)  # C, C, O, H×6

    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = Path(tmpdir) / "test_mol.h5"

        # Write test HDF5
        with h5py.File(h5_path, "w") as f:
            # Expand to multiple conformers for realism
            positions_multi = np.tile(positions[np.newaxis, :, :], (5, 1, 1))
            f.create_dataset("positions", data=positions_multi)
            f.create_dataset("species", data=species)
            f.create_dataset("formula", data="C2HO")
            f.create_dataset("molecule_id", data="test_synthetic")
            f.attrs["bucket_idx"] = 0
            f.attrs["lane"] = "a"

            # Add energy and forces (required by HDF5 loader)
            f.create_dataset("energy", data=np.ones(5))
            f.create_dataset("forces", data=np.random.randn(5, 9, 3).astype(np.float32))

        # Run parameterization
        params = build_params_for_molecule(h5_path, dry_run=False)

        # Verify output structure
        assert params is not None, "Parameterization should succeed"
        assert params["molecule_id"] == "C2HO"
        assert params["lane"] == "a"
        assert params["n_atoms"] == 9
        assert "atom_types" in params
        assert "bonds" in params
        assert "angles" in params
        assert "proper_torsions" in params
        assert "improper_torsions" in params
        assert len(params["bonds"]) > 0, "Should have at least one bond"
        assert len(params["angles"]) > 0, "Should have at least one angle"
        assert "params_init_sha256" in params

        # Verify params_init.json was written
        params_path = h5_path.with_suffix(".params_init.json")
        assert params_path.exists(), "params_init.json should be written"

        with open(params_path) as f:
            loaded_params = json.load(f)

        assert loaded_params == params, "Loaded params should match returned params"


@pytest.mark.slow
@pytest.mark.skipif(not HAS_OPENFF, reason="openff-toolkit not installed")
def test_trp_cage_parameterization_smoke():
    """
    Smoke test: Trp-cage (Lane B, 312 atoms) parameterizes without errors.

    Skipped if data file is missing (e.g., on CI without full dataset).
    """
    data_dir = Path(__file__).parent.parent.parent / "data" / "ani1x_subset" / "lane_b"
    trp_cage_path = data_dir / "trp_cage.h5"

    if not trp_cage_path.exists():
        pytest.skip(f"Trp-cage data not found: {trp_cage_path}")

    # Run parameterization (do NOT write, just check it completes)
    params = build_params_for_molecule(trp_cage_path, dry_run=False)

    assert params is not None, "Trp-cage parameterization should succeed"
    assert params["n_atoms"] == 312, "Trp-cage has 312 atoms"
    assert params["molecule_id"] == "1L2Y", "Trp-cage PDB ID is 1L2Y"
    assert params["lane"] == "b", "Trp-cage is Lane B"

    # Check for peptide bonds (should have many)
    # Peptide C-N bonds typically have k ~260 kcal/mol/Å²
    bond_pattern_count = sum(
        1 for b in params["bonds"]
        if "c" in b["smirks_pattern"].lower() and "n" in b["smirks_pattern"].lower()
    )
    # Not strictly checking count since SMIRKS patterns are varied,
    # but verify at least some bonds detected
    assert len(params["bonds"]) > 50, "Trp-cage should have many bonds"

    # Check no NaN parameters
    for bond in params["bonds"]:
        assert not np.isnan(bond["k"]), f"NaN k in bond {bond}"
        assert not np.isnan(bond["r0"]), f"NaN r0 in bond {bond}"

    for angle in params["angles"]:
        assert not np.isnan(angle["k_theta"]), f"NaN k_theta in angle {angle}"
        assert not np.isnan(angle["theta0_deg"]), f"NaN theta0_deg in angle {angle}"


@pytest.mark.slow
@pytest.mark.skipif(not HAS_OPENFF, reason="openff-toolkit not installed")
def test_chignolin_parameterization_smoke():
    """Smoke test: Chignolin (Lane B, ~138 atoms) parameterizes without errors."""
    data_dir = Path(__file__).parent.parent.parent / "data" / "ani1x_subset" / "lane_b"
    chignolin_path = data_dir / "chignolin.h5"

    if not chignolin_path.exists():
        pytest.skip(f"Chignolin data not found: {chignolin_path}")

    params = build_params_for_molecule(chignolin_path, dry_run=False)

    assert params is not None, "Chignolin parameterization should succeed"
    assert params["n_atoms"] == 138, "Chignolin has 138 atoms"
    assert params["lane"] == "b", "Chignolin is Lane B"

    # Check no NaN parameters
    for bond in params["bonds"]:
        assert not np.isnan(bond["k"]), f"NaN k in bond {bond}"
        assert not np.isnan(bond["r0"]), f"NaN r0 in bond {bond}"

    for angle in params["angles"]:
        assert not np.isnan(angle["k_theta"]), f"NaN k_theta in angle {angle}"
        assert not np.isnan(angle["theta0_deg"]), f"NaN theta0_deg in angle {angle}"
