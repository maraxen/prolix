"""Standardized PyTree types for Prolix physics.

These types ensure that all parameters passed to the MD engine are JAX-compatible
and follow a strictly static signature, unblocking StableHLO/WASM export.
"""

from typing import Any, TypeVar

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array as ArrayType

T = TypeVar("T", bound="PhysicsSystem")

class PhysicsSystem(eqx.Module):
    """A protein system padded to a fixed atom count for vmap compatibility."""

    # Per-atom arrays (all shape: (N_padded, ...))
    positions: ArrayType        # (N_padded, 3)
    charges: ArrayType          # (N_padded,)
    sigmas: ArrayType           # (N_padded,)
    epsilons: ArrayType         # (N_padded,)
    radii: ArrayType            # (N_padded,)   — GB radii
    scaled_radii: ArrayType     # (N_padded,)   — OBC scaling factors
    masses: ArrayType           # (N_padded,)
    element_ids: ArrayType      # (N_padded,) int — atomic number (1=H, 6=C, 7=N, 8=O, 16=S)
    atom_mask: ArrayType        # (N_padded,) bool — True for real atoms
    is_hydrogen: ArrayType      # (N_padded,) bool — True for hydrogen atoms
    is_backbone: ArrayType      # (N_padded,) bool — True for backbone atoms (N, CA, C, O)
    is_heavy: ArrayType         # (N_padded,) bool — True for real non-hydrogen atoms
    protein_atom_mask: ArrayType # (N_padded,) bool — True for protein atoms
    water_atom_mask: ArrayType   # (N_padded,) bool — True for water atoms

    # Bonded term arrays (padded to max per bucket)
    bonds: ArrayType                                     # (N_bonds_padded, 2) int
    bond_params: ArrayType      # (N_bonds_padded, 2) float
    bond_mask: ArrayType        # (N_bonds_padded,) bool

    angles: ArrayType                                    # (N_angles_padded, 3) int
    angle_params: ArrayType     # (N_angles_padded, 2) float
    angle_mask: ArrayType       # (N_angles_padded,) bool

    dihedrals: ArrayType                                 # (N_dih_padded, 4) int
    dihedral_params: ArrayType  # (N_dih_padded, 3) float
    dihedral_mask: ArrayType    # (N_dih_padded,) bool

    impropers: ArrayType                                 # (N_imp_padded, 4) int
    improper_params: ArrayType  # (N_imp_padded, 3) float
    improper_mask: ArrayType    # (N_imp_padded,) bool

    # Urey-Bradley
    urey_bradley_bonds: ArrayType | None = None # (N_ub_padded, 2) int
    urey_bradley_params: ArrayType | None = None # (N_ub_padded, 2) float
    urey_bradley_mask: ArrayType | None = None   # (N_ub_padded,) bool

    # CMAP (optional)
    cmap_torsions: ArrayType | None = None               # (N_cmap_padded, 5) int
    cmap_indices: ArrayType | None = None                # (N_cmap_padded,) int
    cmap_mask: ArrayType | None = None       # (N_cmap_padded,) bool
    cmap_coeffs: ArrayType | None = None     # (N_maps, G, G, 16) — shared across batch

    # Non-bonded exclusions — sparse per-atom arrays
    excl_indices: ArrayType = None                       # (N_padded, max_excl) int32 — excluded atom indices, -1 = unused
    excl_scales_vdw: ArrayType | None = None    # (N_padded, max_excl) float32 — LJ scale (0.0 or 0.5 or 1.0)
    excl_scales_elec: ArrayType | None = None   # (N_padded, max_excl) float32 — elec scale (0.0 or 1/1.2 or 1.0)

    # RATTLE/SHAKE constraints — X-H bond pairs with target lengths
    constraint_pairs: ArrayType = None                   # (N_constr_padded, 2) int — atom indices for constrained bonds
    constraint_lengths: ArrayType | None = None   # (N_constr_padded,) float — equilibrium bond lengths (Å)
    constraint_mask: ArrayType | None = None      # (N_constr_padded,) bool — True for real constraints

    # Metadata
    n_real_atoms: ArrayType | None = None
    n_padded_atoms: int | ArrayType = eqx.field(static=True, default=0)
    bucket_size: int | ArrayType = eqx.field(static=True, default=0)

    # Water molecule indices for SETTLE
    water_indices: ArrayType | None = None               # (N_waters_padded, 3) int
    water_mask: ArrayType | None = None      # (N_waters_padded,) bool
    box_size: ArrayType | None = eqx.field(static=True, default=None) # (3,) static for PME grid shapes
    pme_alpha: float = eqx.field(static=True, default=0.0)
    pme_grid_points: int = eqx.field(static=True, default=64)
    nonbonded_cutoff: float = eqx.field(static=True, default=9.0)

    # Precomputed dense exclusion matrices
    dense_excl_scale_vdw: ArrayType | None = None   # (N_padded, N_padded) float32
    dense_excl_scale_elec: ArrayType | None = None  # (N_padded, N_padded) float32

    def replace(self: T, **kwargs) -> T:
        """Return a copy with specified fields replaced."""
        return eqx.tree_at(
            lambda s: tuple(getattr(s, k) for k in kwargs.keys()),
            self,
            tuple(kwargs.values()),
        )

    def __replace__(self: T, **kwargs) -> T:
        """Python 3.13+ compatibility."""
        return self.replace(**kwargs)

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        positions: ArrayType,
        box_size: ArrayType | None = None,
        cutoff_distance: float = 9.0,
    ) -> "PhysicsSystem":
        """Creates a PhysicsSystem from a dictionary (legacy test format)."""
        n_atoms = positions.shape[0]
        
        # JAX arrays don't support truthiness for non-scalars; use explicit None checks
        dihedrals = d.get("dihedrals")
        if dihedrals is None:
            dihedrals = d.get("proper_dihedrals")
            
        return cls(
            positions=positions,
            charges=jnp.asarray(d["charges"]),
            sigmas=jnp.asarray(d["sigmas"]),
            epsilons=jnp.asarray(d["epsilons"]),
            masses=jnp.asarray(d.get("masses", jnp.ones(n_atoms))),
            box_size=box_size,
            bonds=d.get("bonds"),
            bond_params=d.get("bond_params"),
            angles=d.get("angles"),
            angle_params=d.get("angle_params"),
            dihedrals=dihedrals,
            dihedral_params=d.get("dihedral_params"),
            impropers=d.get("impropers"),
            improper_params=d.get("improper_params"),
            urey_bradley_bonds=d.get("urey_bradley_bonds"),
            urey_bradley_params=d.get("urey_bradley_params"),
            # Mandatory fields with defaults
            radii=d.get("radii", jnp.ones(n_atoms)),
            scaled_radii=d.get("scaled_radii", jnp.ones(n_atoms)),
            element_ids=d.get("element_ids", jnp.zeros(n_atoms, dtype=jnp.int32)),
            atom_mask=d.get("atom_mask", jnp.ones(n_atoms, dtype=bool)),
            is_hydrogen=d.get("is_hydrogen", jnp.zeros(n_atoms, dtype=bool)),
            is_backbone=d.get("is_backbone", jnp.zeros(n_atoms, dtype=bool)),
            is_heavy=d.get("is_heavy", jnp.ones(n_atoms, dtype=bool)),
            protein_atom_mask=d.get("protein_atom_mask", jnp.ones(n_atoms, dtype=bool)),
            water_atom_mask=d.get("water_atom_mask", jnp.zeros(n_atoms, dtype=bool)),
            bond_mask=d.get("bond_mask", jnp.ones(d.get("bonds", jnp.zeros((0,2))).shape[0], dtype=bool)),
            angle_mask=d.get("angle_mask", jnp.ones(d.get("angles", jnp.zeros((0,3))).shape[0], dtype=bool)),
            dihedral_mask=d.get("dihedral_mask", jnp.ones(dihedrals.shape[0] if dihedrals is not None else 0, dtype=bool)),
            improper_mask=d.get("improper_mask", jnp.ones(d.get("impropers", jnp.zeros((0,4))).shape[0], dtype=bool)),
            urey_bradley_mask=d.get("urey_bradley_mask", jnp.ones(d.get("urey_bradley_bonds", jnp.zeros((0,2))).shape[0], dtype=bool)),
            excl_indices=d.get("excl_indices"),
            excl_scales_vdw=d.get("excl_scales_vdw"),
            excl_scales_elec=d.get("excl_scales_elec"),
            dense_excl_scale_vdw=d.get("exclusion_mask"),
            dense_excl_scale_elec=d.get("exclusion_mask"),
            nonbonded_cutoff=cutoff_distance,
        )


class EnergyParams(eqx.Module):
    """Container for energy function parameters."""
    params: dict[str, Any]

    @property
    def charges(self) -> ArrayType:
        return self.params["charges"]

    @property
    def sigmas(self) -> ArrayType:
        return self.params["sigmas"]

    @property
    def epsilons(self) -> ArrayType:
        return self.params["epsilons"]


class IntegratorParams(eqx.Module):
    """Parameters for MD integration steps."""
    dt: float | ArrayType
    kT: float | ArrayType
    gamma: float | ArrayType
    energy_params: EnergyParams
    water_indices: ArrayType = eqx.field(default_factory=lambda: jnp.zeros((0, 3), dtype=jnp.int32))
    constraint_dofs: ArrayType = eqx.field(default_factory=lambda: jnp.zeros((0,), dtype=jnp.int32))
    molecule_indices: ArrayType = eqx.field(default_factory=lambda: jnp.zeros((0,), dtype=jnp.int32))
    box: ArrayType = eqx.field(default_factory=lambda: jnp.zeros((3,)))
    positions_old: ArrayType = eqx.field(default_factory=lambda: jnp.zeros((0, 3)))
    n_dof: float | ArrayType = 0.0
    target_pressure_bar: float = 0.0
    barostat_interval: int = 1
    virtual_site_params: Any = None
