"""Utilities for merging protein and solvent topologies.

This module provides functions to combine protein and solvent parameters
into a single unified SystemParams representation, ensuring correct
intramolecular exclusions and bonded terms for water.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from flax import struct
from proxide.core.containers import Protein

from prolix.physics.neighbor_list import ExclusionSpec
from prolix.physics.water_models import WaterModelType, get_water_params

if TYPE_CHECKING:
    from prolix.physics.solvation import SolventTopology

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
    urey_bradley_bonds: jnp.ndarray | None
    urey_bradley_params: jnp.ndarray | None
    radii: jnp.ndarray
    scaled_radii: jnp.ndarray
    element_ids: jnp.ndarray
    is_backbone: jnp.ndarray
    is_heavy: jnp.ndarray
    is_hydrogen: jnp.ndarray
    box_size: jnp.ndarray | None
    exclusion_spec: ExclusionSpec
    water_indices: jnp.ndarray  # (N_waters, 3)
    water_model: WaterModelType
    atom_mask: jnp.ndarray
    protein_atom_mask: jnp.ndarray
    water_atom_mask: jnp.ndarray
    n_protein_atoms: int
    n_water_atoms: int


def merge_solvated_topology(
    protein: Protein,
    solvent_topo: SolventTopology,
    model_type: WaterModelType | str = WaterModelType.OPC3,
    box_size: Array | None = None,
    merged_positions: jnp.ndarray | None = None,
) -> MergedTopology:
    """Merge protein and solvent into a single system topology.

    Args:
        protein: The protein structure and its parameters.
        solvent_topo: Structured SolventTopology containing pre-parameterized waters and ions.
        model_type: The water model being used.
        box_size: Optional (3,) array defining the periodic simulation box.

    Returns:
        A MergedTopology instance.
    """
    if isinstance(model_type, str):
        model_type = WaterModelType(model_type)

    params = get_water_params(model_type)
    n_protein = len(protein.full_coordinates)
    
    # Real atom mask from protein (handles missing atoms in PDB)
    # Use full_atom_mask (flat, matches full_coordinates) not atom_mask (N_res, 37)
    p_mask = getattr(protein, "full_atom_mask", None)
    if p_mask is None:
        p_mask = getattr(protein, "atom_mask", None)
    if p_mask is None:
        p_mask = jnp.ones(n_protein, dtype=bool)
    else:
        p_mask = jnp.asarray(p_mask, dtype=bool).ravel()

    # 1. Merge basic parameters
    charges = jnp.concatenate([
        jnp.asarray(protein.charges) if protein.charges is not None else jnp.zeros(n_protein),
        solvent_topo.charges
    ])

    masses = jnp.concatenate([
        jnp.asarray(protein.masses) if protein.masses is not None else jnp.ones(n_protein) * 12.011,
        solvent_topo.masses
    ])

    sigmas = jnp.concatenate([
        jnp.asarray(protein.sigmas) if protein.sigmas is not None else jnp.ones(n_protein) * 3.0,
        solvent_topo.sigmas
    ])

    epsilons = jnp.concatenate([
        jnp.asarray(protein.epsilons) if protein.epsilons is not None else jnp.zeros(n_protein),
        solvent_topo.epsilons
    ])

    elem_ids = getattr(protein, "element_ids", None)
    element_ids = jnp.concatenate([
        jnp.asarray(elem_ids) if elem_ids is not None else jnp.zeros(n_protein, dtype=jnp.int32),
        solvent_topo.element_ids
    ])

    radii = jnp.concatenate([
        protein.radii if getattr(protein, "radii", None) is not None else jnp.zeros(n_protein),
        jnp.where(solvent_topo.is_hydrogen, 0.12, 0.15)  # Heuristic for implicit radii in solvent
    ])

    scaled_radii = jnp.concatenate([
        protein.scaled_radii if getattr(protein, "scaled_radii", None) is not None else jnp.zeros(n_protein),
        jnp.where(solvent_topo.is_hydrogen, 0.12, 0.15)
    ])

    is_backbone = jnp.concatenate([
        jnp.asarray(protein.is_backbone) if getattr(protein, "is_backbone", None) is not None else jnp.zeros(n_protein, dtype=bool),
        jnp.zeros(len(solvent_topo.charges), dtype=bool)
    ])

    is_hydrogen = jnp.concatenate([
        jnp.asarray(protein.is_hydrogen) if getattr(protein, "is_hydrogen", None) is not None else jnp.zeros(n_protein, dtype=bool),
        solvent_topo.is_hydrogen
    ])

    is_heavy = jnp.concatenate([
        jnp.asarray(protein.is_heavy) if getattr(protein, "is_heavy", None) is not None else jnp.zeros(n_protein, dtype=bool),
        ~solvent_topo.is_hydrogen
    ])

    # 2. Extract protein bonded terms
    bonds = protein.bonds
    bond_params = protein.bond_params
    angles = protein.angles
    angle_params = protein.angle_params
    
    # New bonded terms
    proper_dihedrals = getattr(protein, "proper_dihedrals", getattr(protein, "dihedrals", None))
    dihedral_params = getattr(protein, "dihedral_params", None)
    impropers = getattr(protein, "impropers", None)
    improper_params = getattr(protein, "improper_params", None)
    cmap_torsions = getattr(protein, "cmap_torsions", None)
    cmap_indices = getattr(protein, "cmap_indices", None)
    cmap_energy_grids = getattr(protein, "cmap_energy", getattr(protein, "cmap_energy_grids", None))
    urey_bradley_bonds = getattr(protein, "urey_bradley_bonds", None)
    urey_bradley_params = getattr(protein, "urey_bradley_params", None)

    # 3. Add water bonded terms
    water_indices = solvent_topo.water_indices + n_protein
    n_waters = solvent_topo.n_waters
    
    # O-H1 and O-H2 bonds
    water_bonds = jnp.stack([
        water_indices[:, [0, 1]],
        water_indices[:, [0, 2]]
    ], axis=1).reshape(-1, 2)
    
    water_bond_params = jnp.tile(
        jnp.array([params.r_OH, params.k_bond]), (n_waters * 2, 1)
    )

    # H1-O-H2 angle
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
    n_total = n_protein + len(solvent_topo.charges)
    exclusions = find_bonded_exclusions(merged_bonds, n_total)
    
    # 1-4 scaling (water has none, so we reuse protein scales)
    c14 = protein.coulomb14scale if protein.coulomb14scale is not None else 0.83333333
    l14 = protein.lj14scale if protein.lj14scale is not None else 0.5
    
    merged_excl = ExclusionSpec(
        idx_12_13=jnp.concatenate([exclusions.idx_12, exclusions.idx_13], axis=0),
        idx_14=exclusions.idx_14,
        scale_14_elec=c14,
        scale_14_vdw=l14,
        n_atoms=n_total
    )

    solvent_mask = jnp.ones(len(solvent_topo.charges), dtype=bool)

    return MergedTopology(
        positions=merged_positions if merged_positions is not None else jnp.concatenate([protein.full_coordinates, solvent_topo.positions]),
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
        urey_bradley_bonds=urey_bradley_bonds,
        urey_bradley_params=urey_bradley_params,
        radii=radii,
        scaled_radii=scaled_radii,
        element_ids=element_ids,
        is_backbone=is_backbone,
        is_heavy=is_heavy,
        is_hydrogen=is_hydrogen,
        box_size=jnp.asarray(box_size) if box_size is not None else None,
        exclusion_spec=merged_excl if merged_excl is not None else protein.exclusion_spec,
        water_indices=water_indices,
        water_model=model_type,
        atom_mask=jnp.concatenate([p_mask, solvent_mask]),
        protein_atom_mask=jnp.concatenate([p_mask, jnp.zeros(len(solvent_topo.charges), dtype=bool)]),
        water_atom_mask=jnp.concatenate([
            jnp.zeros(n_protein, dtype=bool),
            jnp.zeros(len(solvent_topo.charges), dtype=bool).at[solvent_topo.water_indices.flatten()].set(True)
        ]),
        n_protein_atoms=n_protein,
        n_water_atoms=len(solvent_topo.charges),
    )
