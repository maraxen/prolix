"""Tests for parse_structure with MD parameterization.

This module tests that the Rust-based parse_structure function correctly
returns parameterized Protein objects with all required MD fields.
"""

from pathlib import Path

import jax.numpy as jnp
import pytest
from proxide.core.containers import Protein
from proxide.io.parsing.rust import OutputSpec, parse_structure

# Path to test data and force field
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "pdb"
FF_PATH = Path(__file__).parent.parent.parent / "proxide" / "src" / "proxide" / "assets" / "protein.ff19SB.xml"


@pytest.fixture
def parameterized_protein() -> Protein:
    """Load a protein with MD parameterization enabled."""
    pdb_path = DATA_DIR / "1CRN.pdb"

    spec = OutputSpec()
    spec.parameterize_md = True
    spec.force_field = str(FF_PATH)
    spec.add_hydrogens = True

    return parse_structure(str(pdb_path), spec)


def test_parse_structure_returns_protein(parameterized_protein: Protein):
    """Test that parse_structure with MD params returns a Protein."""
    assert isinstance(parameterized_protein, Protein)


def test_parse_structure_has_charges(parameterized_protein: Protein):
    """Test that parameterized protein has charges assigned."""
    assert parameterized_protein.charges is not None
    assert len(parameterized_protein.charges.shape) == 1
    # 1CRN has ~550 atoms with hydrogens
    assert parameterized_protein.charges.shape[0] > 0


def test_parse_structure_has_lj_params(parameterized_protein: Protein):
    """Test that parameterized protein has LJ parameters (sigmas, epsilons)."""
    assert parameterized_protein.sigmas is not None
    assert parameterized_protein.epsilons is not None
    assert parameterized_protein.sigmas.shape == parameterized_protein.charges.shape
    assert parameterized_protein.epsilons.shape == parameterized_protein.charges.shape


def test_parse_structure_has_bonds(parameterized_protein: Protein):
    """Test that parameterized protein has bonds and bond parameters."""
    assert parameterized_protein.bonds is not None
    assert parameterized_protein.bond_params is not None
    assert parameterized_protein.bonds.shape[1] == 2  # (N_bonds, 2)
    assert parameterized_protein.bond_params.shape[0] == parameterized_protein.bonds.shape[0]
    assert parameterized_protein.bond_params.shape[1] == 2  # [length, k]


def test_parse_structure_has_angles(parameterized_protein: Protein):
    """Test that parameterized protein has angles and angle parameters."""
    assert parameterized_protein.angles is not None
    assert parameterized_protein.angle_params is not None
    assert parameterized_protein.angles.shape[1] == 3  # (N_angles, 3)
    assert parameterized_protein.angle_params.shape[0] == parameterized_protein.angles.shape[0]
    assert parameterized_protein.angle_params.shape[1] == 2  # [theta0, k]


def test_parse_structure_has_dihedrals(parameterized_protein: Protein):
    """Test that parameterized protein has dihedrals and dihedral parameters."""
    assert parameterized_protein.proper_dihedrals is not None
    assert parameterized_protein.dihedral_params is not None
    assert parameterized_protein.proper_dihedrals.shape[1] == 4  # (N_dihedrals, 4)
    assert parameterized_protein.dihedral_params.shape[0] == parameterized_protein.proper_dihedrals.shape[0]
    assert parameterized_protein.dihedral_params.shape[1] == 3  # [periodicity, phase, k]


def test_charges_are_reasonable(parameterized_protein: Protein):
    """Test that charges are in a reasonable range for proteins."""
    charges = parameterized_protein.charges
    # Protein partial charges should typically be between -2 and +2
    assert jnp.all(charges >= -3.0), "Found unreasonably negative charges"
    assert jnp.all(charges <= 3.0), "Found unreasonably positive charges"


def test_bond_lengths_are_physical(parameterized_protein: Protein):
    """Test that predicted bond lengths are in a physical range."""
    bond_params = parameterized_protein.bond_params
    lengths = bond_params[:, 0]  # First column is equilibrium length
    # Rust returns lengths in nm: 0.09-0.2 nm = 0.9-2.0 Angstroms
    assert jnp.all(lengths > 0.05), "Bond length too short"
    assert jnp.all(lengths < 0.3), "Bond length too long"


def test_lj_sigmas_are_physical(parameterized_protein: Protein):
    """Test that LJ sigma values are in a physical range."""
    sigmas = parameterized_protein.sigmas
    # Rust returns sigmas in nm: 0.1-0.4 nm = 1.0-4.0 Angstroms
    # Hydrogens may have sigma=0 (they are virtual sites or excluded)
    heavy_sigmas = sigmas[sigmas > 0]
    assert jnp.all(heavy_sigmas > 0.1), "Sigma too small for heavy atoms"
    assert jnp.all(heavy_sigmas < 0.5), "Sigma too large"
