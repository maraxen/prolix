"""Backward-compatibility shim for the SystemParams → Protein migration.

Deprecated: Use proxide.io.parsing.backend.parse_structure with
OutputSpec(parameterize_md=True) instead.
"""
from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import jax.numpy as jnp
from proxide.core.containers import Protein

if TYPE_CHECKING:
    from proxide.types import SystemParams


def system_params_to_protein(params: SystemParams) -> Protein:
    """Convert a legacy SystemParams dict to a Protein dataclass.

    This is a one-way compatibility bridge. Emits DeprecationWarning.
    """
    warnings.warn(
        "system_params_to_protein() is deprecated. Use "
        "proxide.io.parsing.backend.parse_structure(spec=OutputSpec("
        "parameterize_md=True)) instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    def _get(key, default=None):
        return params.get(key, default)

    # Required fields
    charges = jnp.asarray(params["charges"])
    bonds = jnp.asarray(params["bonds"])
    bond_params = jnp.asarray(params["bond_params"])
    angles = jnp.asarray(params["angles"])
    angle_params = jnp.asarray(params["angle_params"])
    sigmas = jnp.asarray(params["sigmas"])
    epsilons = jnp.asarray(params["epsilons"])

    # Renamed fields
    proper_dihedrals = jnp.asarray(params["dihedrals"]) if _get("dihedrals") is not None else None
    dihedral_params = jnp.asarray(params["dihedral_params"]) if _get("dihedral_params") is not None else None
    impropers = jnp.asarray(params["impropers"]) if _get("impropers") is not None else None
    improper_params = jnp.asarray(params["improper_params"]) if _get("improper_params") is not None else None
    radii = jnp.asarray(params["gb_radii"]) if _get("gb_radii") is not None else None

    # Handle Masses (W2)
    masses_raw = _get("masses")
    masses = jnp.asarray(masses_raw) if masses_raw is not None else jnp.ones_like(charges)

    # Constraint fields (C2)
    constrained_bonds = jnp.asarray(_get("constrained_bonds")) if _get("constrained_bonds") is not None else None
    constrained_bond_lengths = jnp.asarray(_get("constrained_bond_lengths")) if _get("constrained_bond_lengths") is not None else None

    # Optional fields
    gbsa_scales = jnp.asarray(_get("gbsa_scales")) if _get("gbsa_scales") is not None else None
    scaled_radii = jnp.asarray(_get("scaled_radii")) if _get("scaled_radii") is not None else None
    exclusion_mask = jnp.asarray(_get("exclusion_mask")) if _get("exclusion_mask") is not None else None
    scale_matrix_vdw = jnp.asarray(_get("scale_matrix_vdw")) if _get("scale_matrix_vdw") is not None else None
    scale_matrix_elec = jnp.asarray(_get("scale_matrix_elec")) if _get("scale_matrix_elec") is not None else None
    urey_bradley_bonds = jnp.asarray(_get("urey_bradley_bonds")) if _get("urey_bradley_bonds") is not None else None
    urey_bradley_params = jnp.asarray(_get("urey_bradley_params")) if _get("urey_bradley_params") is not None else None
    cmap_torsions = jnp.asarray(_get("cmap_torsions")) if _get("cmap_torsions") is not None else None
    cmap_energy_grids = jnp.asarray(_get("cmap_energy_grids")) if _get("cmap_energy_grids") is not None else None
    cmap_indices = jnp.asarray(_get("cmap_indices")) if _get("cmap_indices") is not None else None
    virtual_site_def = jnp.asarray(_get("virtual_site_def")) if _get("virtual_site_def") is not None else None
    virtual_site_params = jnp.asarray(_get("virtual_site_params")) if _get("virtual_site_params") is not None else None
    coulomb14scale = _get("coulomb14scale")
    lj14scale = _get("lj14scale")
    backbone_indices = jnp.asarray(_get("backbone_indices")) if _get("backbone_indices") is not None else None

    # Sentinel structural fields (S1)
    sentinel_coords = jnp.zeros((1, 37, 3), dtype=jnp.float32)
    sentinel_aatype = jnp.zeros((1,), dtype=jnp.int8)
    sentinel_residue_index = jnp.zeros((1,), dtype=jnp.int32)
    sentinel_chain_index = jnp.zeros((1,), dtype=jnp.int32)

    return Protein(
        coordinates=sentinel_coords, aatype=sentinel_aatype,
        residue_index=sentinel_residue_index, chain_index=sentinel_chain_index,
        charges=charges, masses=masses, bonds=bonds, bond_params=bond_params,
        constrained_bonds=constrained_bonds, constrained_bond_lengths=constrained_bond_lengths,
        angles=angles, angle_params=angle_params, proper_dihedrals=proper_dihedrals,
        dihedral_params=dihedral_params, impropers=impropers, improper_params=improper_params,
        sigmas=sigmas, epsilons=epsilons, radii=radii, gbsa_scales=gbsa_scales,
        scaled_radii=scaled_radii, exclusion_mask=exclusion_mask,
        scale_matrix_vdw=scale_matrix_vdw, scale_matrix_elec=scale_matrix_elec,
        urey_bradley_bonds=urey_bradley_bonds, urey_bradley_params=urey_bradley_params,
        cmap_torsions=cmap_torsions, cmap_energy_grids=cmap_energy_grids, cmap_indices=cmap_indices,
        virtual_site_def=virtual_site_def, virtual_site_params=virtual_site_params,
        coulomb14scale=coulomb14scale, lj14scale=lj14scale, backbone_indices=backbone_indices,
    )
