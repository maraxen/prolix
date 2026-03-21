"""Test pad-and-batch logic for cross-topology batching."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from proxide.core.containers import Protein

from prolix.padding import (
    ATOM_BUCKETS,
    PaddedSystem,
    bucket_proteins,
    collate_batch,
    pad_protein,
    select_bucket,
)


@pytest.fixture
def dummy_protein_small() -> Protein:
    """A small dummy protein with 10 atoms."""
    return Protein(
        coordinates=np.random.randn(10, 3),
        aatype=np.zeros(10, dtype=np.int32),
        residue_index=np.zeros(10, dtype=np.int32),
        chain_index=np.zeros(10, dtype=np.int32),
        charges=np.linspace(-0.5, 0.5, 10),
        sigmas=np.ones(10) * 3.0,
        epsilons=np.ones(10) * 0.1,
        radii=np.ones(10) * 1.2,
        masses=np.ones(10) * 12.0,
        bonds=np.array([[0, 1], [1, 2], [2, 3]]),
        bond_params=np.array([[1.5, 300.0]] * 3),
        angles=np.array([[0, 1, 2], [1, 2, 3]]),
        angle_params=np.array([[2.0, 100.0]] * 2),
    )


@pytest.fixture
def dummy_protein_large() -> Protein:
    """A larger dummy protein with 5000 atoms."""
    return Protein(
        coordinates=np.random.randn(5000, 3),
        aatype=np.zeros(5000, dtype=np.int32),
        residue_index=np.zeros(5000, dtype=np.int32),
        chain_index=np.zeros(5000, dtype=np.int32),
        charges=np.zeros(5000),
        sigmas=np.ones(5000) * 3.0,
        epsilons=np.ones(5000) * 0.1,
        bonds=np.zeros((4000, 2), dtype=np.int32),
        bond_params=np.zeros((4000, 2)),
    )


def test_select_bucket():
    assert select_bucket(10) == ATOM_BUCKETS[0]
    assert select_bucket(2048) == 2048
    assert select_bucket(2049) == 2816  # new tighter bucket
    assert select_bucket(2816) == 2816
    assert select_bucket(2817) == 3072
    assert select_bucket(4000) == 4096
    assert select_bucket(4096) == 4096
    assert select_bucket(4097) == 5120  # was 8192, now 5120
    
    with pytest.raises(ValueError, match="exceeds maximum bucket size"):
        select_bucket(1000000)


def test_pad_protein_shapes_and_masks(dummy_protein_small: Protein):
    target_atoms = 100
    target_bonds = 5
    
    padded = pad_protein(
        dummy_protein_small, 
        target_atoms=target_atoms, 
        target_bonds=target_bonds,
        target_angles=5,
        target_dihedrals=5,
        target_impropers=5,
        target_cmaps=0
    )
    
    assert padded.positions.shape == (target_atoms, 3)
    assert padded.charges.shape == (target_atoms,)
    assert padded.bonds.shape == (target_bonds, 2)
    assert padded.bond_params.shape == (target_bonds, 2)
    
    # Check atom mask
    assert padded.n_real_atoms == 10
    assert jnp.sum(padded.atom_mask) == 10
    assert bool(padded.atom_mask[9]) is True
    assert bool(padded.atom_mask[10]) is False
    
    # Check bond mask
    assert jnp.sum(padded.bond_mask) == 3
    assert bool(padded.bond_mask[2]) is True
    assert bool(padded.bond_mask[3]) is False


def test_ghost_atom_values(dummy_protein_small: Protein):
    padded = pad_protein(dummy_protein_small, target_atoms=20)
    
    # Check ghost items (index 10-19)
    ghost_pos = padded.positions[10:]
    np.testing.assert_allclose(ghost_pos, 9999.0)
    
    ghost_charges = padded.charges[10:]
    np.testing.assert_allclose(ghost_charges, 0.0)
    
    ghost_sigmas = padded.sigmas[10:]
    np.testing.assert_allclose(ghost_sigmas, 1e-6)
    
    ghost_epsilons = padded.epsilons[10:]
    np.testing.assert_allclose(ghost_epsilons, 0.0)
    
    ghost_radii = padded.radii[10:]
    np.testing.assert_allclose(ghost_radii, 1.5)


def test_bonded_padding_zero_energy(dummy_protein_small: Protein):
    padded = pad_protein(
        dummy_protein_small, 
        target_atoms=20, 
        target_bonds=10,
        target_angles=10
    )
    
    # Real bonds
    assert padded.bonds[0, 0] == 0
    assert padded.bonds[0, 1] == 1
    assert padded.bond_params[0, 1] == 300.0  # k
    
    # Padded bonds should point to 0,0 with k=0
    assert padded.bonds[5, 0] == 0
    assert padded.bonds[5, 1] == 0
    assert padded.bond_params[5, 1] == 0.0


def test_bucket_proteins_grouping(dummy_protein_small: Protein, dummy_protein_large: Protein):
    proteins = [dummy_protein_small, dummy_protein_large, dummy_protein_small]
    
    groups = bucket_proteins(proteins, buckets=(4096, 8192))
    
    # Small proteins into 4096
    assert 4096 in groups
    assert len(groups[4096]) == 2
    assert groups[4096][0].n_real_atoms == 10
    assert groups[4096][0].n_padded_atoms == 4096
    
    # Large protein into 8192
    assert 8192 in groups
    assert len(groups[8192]) == 1
    assert groups[8192][0].n_real_atoms == 5000
    assert groups[8192][0].n_padded_atoms == 8192
    
    # Verify the two small proteins were padded to the SAME max bonds length
    p1 = groups[4096][0]
    p2 = groups[4096][1]
    
    assert p1.bonds.shape[0] == p2.bonds.shape[0]


def test_collate_batch(dummy_protein_small: Protein):
    # Two identical proteins
    p1 = pad_protein(dummy_protein_small, 50, target_bonds=10)
    p2 = pad_protein(dummy_protein_small, 50, target_bonds=10)
    
    batch = collate_batch([p1, p2])
    
    # Check leading batch dimension
    assert batch.positions.shape == (2, 50, 3)
    assert batch.charges.shape == (2, 50)
    assert batch.bonds.shape == (2, 10, 2)
    assert batch.bond_mask.shape == (2, 10)
    
    # Check eqx.field static properties are NOT stacked
    assert isinstance(batch.bucket_size, int)
    assert batch.bucket_size == 50
    assert batch.n_real_atoms.shape == (2,)
