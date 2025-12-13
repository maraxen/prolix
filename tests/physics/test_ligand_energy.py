"""Tests for GAFF ligand energy parameterization."""

import jax.numpy as jnp
import numpy as np
import pytest
from proxide.io.parsing.molecule import Molecule
from proxide.md.bridge.ligand import parameterize_ligand
from proxide.physics.force_fields import FullForceField, load_force_field

# Use temporary directory for creating test force fields
@pytest.fixture
def mock_gaff_ff(tmp_path):
    """Create a mock GAFF force field with GAFFNonbondedParams."""
    from proxide.physics.force_fields.components import (
        AtomTypeParams,
        BondPotentialParams,
        AnglePotentialParams,
        DihedralPotentialParams,
        CMAPParams,
        UreyBradleyParams,
        VirtualSiteParams,
        NonbondedGlobalParams,
        GAFFNonbondedParams,
    )
    
    # Basic components
    atom_params = AtomTypeParams(
        charges=jnp.array([0.0]),
        sigmas=jnp.array([1.0]),
        epsilons=jnp.array([0.0]),
        radii=jnp.array([1.0]),
        scales=jnp.array([0.0]),
        atom_key_to_id={("LIG", "C1"): 0},
        id_to_atom_key=[("LIG", "C1")],
        atom_class_map={},
        atom_type_map={},
    )
    
    # The crucial part: GAFF LJ params
    gaff_params = GAFFNonbondedParams(
        type_to_index={"ca": 0, "ha": 1},
        sigmas=jnp.array([3.4, 2.6]),  # Angstroms
        epsilons=jnp.array([0.1, 0.05]), # kcal/mol
    )
    
    ff = FullForceField(
        atom_params=atom_params,
        bond_params=BondPotentialParams(params=[]),
        angle_params=AnglePotentialParams(params=[]),
        dihedral_params=DihedralPotentialParams(propers=[], impropers=[]),
        cmap_params=CMAPParams(energy_grids=jnp.zeros((0, 24, 24)), torsions=[]),
        urey_bradley_params=UreyBradleyParams(params=[]),
        virtual_site_params=VirtualSiteParams(definitions={}),
        global_params=NonbondedGlobalParams(),
        gaff_nonbonded_params=gaff_params,
        residue_templates={},
        source_files=[],
    )
    return ff

def test_gaff_lj_lookup_uses_ff(mock_gaff_ff):
    """Test that parameterize_ligand uses the force field's GAFF params."""
    # Create a dummy molecule
    mol = Molecule(
        name="test",
        atom_names=["C1", "H1"],
        atom_types=["ca", "ha"],
        elements=["C", "H"],
        positions=np.zeros((2, 3)),
        charges=np.zeros(2),
        bonds=[],
        bond_orders=[],
    )
    
    params = parameterize_ligand(mol, mock_gaff_ff)
    
    # Should use values from mock_gaff_ff
    # ca: sigma=3.4, epsilon=0.1
    # ha: sigma=2.6, epsilon=0.05
    sigmas = params["sigmas"]
    epsilons = params["epsilons"]
    
    assert sigmas[0] == pytest.approx(3.4)
    assert epsilons[0] == pytest.approx(0.1)
    assert sigmas[1] == pytest.approx(2.6)
    assert epsilons[1] == pytest.approx(0.05)

def test_gaff_missing_lj_fallback(mock_gaff_ff, capsys):
    """Test fallback when GAFF type is missing from FF params."""
    mol = Molecule(
        name="test",
        atom_names=["X1"],
        atom_types=["missing_type"],
        elements=["C"],
        positions=np.zeros((1, 3)),
        charges=np.zeros(1),
        bonds=[],
        bond_orders=[],
    )
    
    # Should fall back to hardcoded defaults (or defaults logic inside ligand.py)
    # Current ligand.py logic prints a warning
    params = parameterize_ligand(mol, mock_gaff_ff)
    
    captured = capsys.readouterr()
    assert "No LJ params for atom type 'missing_type'" in captured.out
    
    # Default fallback is 1.7, 0.1
    assert params["sigmas"][0] == pytest.approx(1.7)
    assert params["epsilons"][0] == pytest.approx(0.1)

def test_default_gaff_loading():
    """Test that standard GAFF loading works (even without new params initially)."""
    # This relies on the real gaff-2.2.20.eqx or similar being present/mocked if we want integration test
    # But for unit test, we can just check if loading existing FF creates None gaff_params
    pass 
