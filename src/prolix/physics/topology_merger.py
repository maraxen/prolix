"""Utilities for merging protein and solvent topologies.

This module provides functions to combine protein and solvent parameters
into a single unified SystemParams representation, ensuring correct
intramolecular exclusions and bonded terms for water.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from flax import struct
from proxide.core.containers import Protein
from prolix.physics.water_models import WaterModelParams, WaterModelType, get_water_params
from prolix.physics.neighbor_list import ExclusionSpec

@struct.dataclass
class MergedTopology:
    """Container for merged protein and water parameters."""
    positions: jnp.ndarray
    charges: jnp.ndarray
    masses: jnp.ndarray
    sigmas: jnp.ndarray
    epsilons: jnp.ndarray
    bonds: jnp.ndarray
    bond_params: jnp.ndarray
    angles: jnp.ndarray
    angle_params: jnp.ndarray
    proper_dihedrals: jnp.ndarray | None  # Optional for pure water systems
    dihedral_params: jnp.ndarray | None
    impropers: jnp.ndarray | None
    improper_params: jnp.ndarray | None
    cmap_torsions: jnp.ndarray | None
    cmap_indices: jnp.ndarray | None
    cmap_energy_grids: jnp.ndarray | None
    radii: jnp.ndarray
    scaled_radii: jnp.ndarray
    element_ids: jnp.ndarray
    box_size: jnp.ndarray | None
    exclusion_spec: ExclusionSpec
    water_indices: jnp.ndarray  # (N_waters, 3)
    water_model: WaterModelType
    protein_atom_mask: jnp.ndarray
    water_atom_mask: jnp.ndarray
    n_protein_atoms: int
    n_water_atoms: int


def merge_solvated_topology(
    protein: Protein,
    water_positions: jnp.ndarray,
    model_type: WaterModelType | str = WaterModelType.OPC3,
    box_size: jnp.ndarray | None = None,
) -> MergedTopology:
    """Merge protein and solvent into a single system topology.

    Args:
        protein: The protein structure and its parameters.
        water_positions: (N_waters * 3, 3) matrix of water coordinates.
        model_type: The water model to use (TIP3P or OPC3).
        box_size: Optional (3,) array defining the periodic simulation box.

    Returns:
        A MergedTopology instance.
    """
    if isinstance(model_type, str):
        model_type = WaterModelType(model_type)

    params = get_water_params(model_type)
    n_protein = len(protein.full_coordinates)
    n_waters = len(water_positions) // 3

    # 1. Merge basic parameters
    charges = jnp.concatenate([
        jnp.asarray(protein.charges) if protein.charges is not None else jnp.zeros(n_protein),
        jnp.tile(jnp.array([params.charge_O, params.charge_H, params.charge_H]), n_waters)
    ])

    masses = jnp.concatenate([
        jnp.asarray(protein.masses) if protein.masses is not None else jnp.ones(n_protein) * 12.011,
        jnp.tile(jnp.array([15.999, 1.008, 1.008]), n_waters)
    ])

    sigmas = jnp.concatenate([
        jnp.asarray(protein.sigmas) if protein.sigmas is not None else jnp.ones(n_protein) * 3.0,
        jnp.tile(jnp.array([params.sigma_O, 1.0e-6, 1.0e-6]), n_waters)
    ])

    epsilons = jnp.concatenate([
        jnp.asarray(protein.epsilons) if protein.epsilons is not None else jnp.zeros(n_protein),
        jnp.tile(jnp.array([params.epsilon_O, 0.0, 0.0]), n_waters)
    ])

    elem_ids = getattr(protein, 'element_ids', None)
    # Water has default element IDs (O: 8, H: 1)
    element_ids = jnp.concatenate([
        jnp.asarray(elem_ids) if elem_ids is not None else jnp.zeros(n_protein, dtype=jnp.int32),
        jnp.tile(jnp.array([8, 1, 1], dtype=jnp.int32), n_waters)
    ])

    radii = jnp.concatenate([
        protein.radii if hasattr(protein, 'radii') else jnp.zeros(n_protein),
        jnp.tile(jnp.array([0.15, 0.12, 0.12]), n_waters)  # Default approximate VdW radii for water
    ])

    scaled_radii = jnp.concatenate([
        protein.scaled_radii if hasattr(protein, 'scaled_radii') else jnp.zeros(n_protein),
        jnp.tile(jnp.array([0.15, 0.12, 0.12]), n_waters)
    ])

    # 2. Extract protein bonded terms
    bonds = protein.bonds
    bond_params = protein.bond_params
    angles = protein.angles
    angle_params = protein.angle_params
    
    # New bonded terms
    proper_dihedrals = getattr(protein, 'proper_dihedrals', getattr(protein, 'dihedrals', None))
    dihedral_params = getattr(protein, 'dihedral_params', None)
    impropers = getattr(protein, 'impropers', None)
    improper_params = getattr(protein, 'improper_params', None)
    cmap_torsions = getattr(protein, 'cmap_torsions', None)
    cmap_indices = getattr(protein, 'cmap_indices', None)
    cmap_energy_grids = getattr(protein, 'cmap_energy', getattr(protein, 'cmap_energy_grids', None))

    # 3. Add water bonded terms
    water_indices = jnp.arange(n_waters * 3).reshape(n_waters, 3) + n_protein
    
    # O-H1 and O-H2 bonds
    water_bonds = jnp.stack([
        water_indices[:, [0, 1]], 
        water_indices[:, [0, 2]]
    ], axis=1).reshape(-1, 2)
    
    water_bond_params = jnp.tile(
        jnp.array([params.r_OH, params.k_bond]), (n_waters * 2, 1)
    )

    # H-O-H angles
    water_angles = water_indices[:, [1, 0, 2]]
    water_angle_params = jnp.tile(
        jnp.array([params.theta_HOH, params.k_angle]), (n_waters, 1)
    )

    # Combine bonded terms
    merged_bonds = jnp.concatenate([bonds, water_bonds]) if bonds is not None else water_bonds
    merged_bond_params = jnp.concatenate([bond_params, water_bond_params]) if bonds is not None else water_bond_params
    
    merged_angles = jnp.concatenate([angles, water_angles]) if angles is not None else water_angles
    merged_angle_params = jnp.concatenate([angle_params, water_angle_params]) if angles is not None else water_angle_params

    # 4. Exclusions
    from prolix.utils.topology import find_bonded_exclusions
    n_total = n_protein + n_waters * 3
    exclusions = find_bonded_exclusions(merged_bonds, n_total)
    
    # 1-4 scaling (water has none, so we reuse protein scales)
    c14 = protein.coulomb14scale if protein.coulomb14scale is not None else 0.83333333
    l14 = protein.lj14scale if protein.lj14scale is not None else 0.5
    
    exclusion_spec = ExclusionSpec(
        idx_12_13=jnp.concatenate([exclusions.idx_12, exclusions.idx_13], axis=0),
        idx_14=exclusions.idx_14,
        scale_14_elec=c14,
        scale_14_vdw=l14,
        n_atoms=n_total
    )

    return MergedTopology(
        positions=jnp.concatenate([protein.full_coordinates, water_positions]),
        charges=charges,
        masses=masses,
        sigmas=sigmas,
        epsilons=epsilons,
        bonds=merged_bonds,
        bond_params=merged_bond_params,
        angles=merged_angles,
        angle_params=merged_angle_params,
        proper_dihedrals=proper_dihedrals,
        dihedral_params=dihedral_params,
        impropers=impropers,
        improper_params=improper_params,
        cmap_torsions=cmap_torsions,
        cmap_indices=cmap_indices,
        cmap_energy_grids=cmap_energy_grids,
        radii=radii,
        scaled_radii=scaled_radii,
        element_ids=element_ids,
        box_size=box_size,
        exclusion_spec=exclusion_spec,
        water_indices=water_indices,
        water_model=model_type,
        protein_atom_mask=jnp.concatenate([jnp.ones(n_protein, dtype=bool), jnp.zeros(n_waters * 3, dtype=bool)]),
        water_atom_mask=jnp.concatenate([jnp.zeros(n_protein, dtype=bool), jnp.ones(n_waters * 3, dtype=bool)]),
        n_protein_atoms=n_protein,
        n_water_atoms=n_waters * 3,
    )
