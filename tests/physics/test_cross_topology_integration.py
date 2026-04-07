"""Integration tests verifying parity between standard and batched padding pipelines."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax_md import space

from prolix import pad_protein, collate_batch, make_batched_energy_fn
from prolix.physics.system import make_energy_fn
from proxide import OutputSpec, parse_structure

@pytest.fixture
def test_protein():
    from pathlib import Path
    pdb_path = Path(__file__).parent.parent.parent.parent / "proxide" / "tests" / "data" / "1uao.pdb"
    ff_path = Path(__file__).parent.parent.parent.parent / "proxide" / "src" / "proxide" / "assets" / "protein.ff19SB.xml"
    spec = OutputSpec()
    spec.parameterize_md = True
    spec.force_field = str(ff_path)
    # 1uao is a simple valid test structure
    protein = parse_structure(str(pdb_path), spec)
    return protein


def test_batched_bonded_parity(test_protein):
    """
    Verify that bonded terms calculated by the batched padded energy function
    match the canonical system.make_energy_fn exactly.
    """
    displacement_fn, _ = space.free()
    
    # 1. Canonical Energy
    # make_energy_fn automatically uses test_protein
    canonical_fns = make_energy_fn(
        displacement_fn,
        system=test_protein,
        implicit_solvent=False,  # just checking bonded currently
        use_pbc=False,
        return_decomposed=True,
    )
    
    pos = jnp.asarray(test_protein.coordinates).reshape(-1, 3)
    
    e_bond_canon = canonical_fns["bond"](pos)
    e_angle_canon = canonical_fns["angle"](pos)
    e_dih_canon = canonical_fns["dihedral"](pos)
    
    # 2. Batched Energy
    # Setup padding
    padded = pad_protein(
        test_protein,
        target_atoms=7000,
    )
    
    batch = collate_batch([padded])  # batch size 1
    
    # In batched_energy.py, _single_padded_energy is private but we can
    # test its bonded components directly or just test the full energy 
    # without exclusions for now.
    from prolix.batched_energy import _bond_energy_masked, _angle_energy_masked, _dihedral_energy_masked
    
    pad_pos = batch.positions[0]
    
    e_bond_batch = _bond_energy_masked(
        pad_pos, batch.bonds[0], batch.bond_params[0], batch.bond_mask[0], displacement_fn
    )
    
    e_angle_batch = _angle_energy_masked(
        pad_pos, batch.angles[0], batch.angle_params[0], batch.angle_mask[0], displacement_fn
    )
    
    e_dih_batch = _dihedral_energy_masked(
        pad_pos, batch.dihedrals[0], batch.dihedral_params[0], batch.dihedral_mask[0], displacement_fn
    )
    
    np.testing.assert_allclose(e_bond_batch, e_bond_canon, rtol=1e-5)
    np.testing.assert_allclose(e_angle_batch, e_angle_canon, rtol=1e-5)
    np.testing.assert_allclose(e_dih_batch, e_dih_canon, rtol=1e-5)
