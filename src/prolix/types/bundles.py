"""MolecularBundle: Typed host-to-device boundary with bucketed dynamic topology.

Core design:
- MolecularBundle: eqx.Module with concrete arrays, no Optional fields
- MolecularShapeSpec: Frozen dataclass, hashable, serves as JIT cache key
- Topology arrays are DYNAMIC (not static=True) to avoid O(n) recompilations
- shape_spec is the ONLY static=True field — carries plain Python scalars
- displacement_fn is NOT stored; reconstructed from boundary_condition flag
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

# Bucketed array size thresholds: systems are padded to the smallest bucket
# that fits their actual count. This enables XLA to cache on bucket size,
# not distinct topology.
ATOM_BUCKETS = (256, 1_024, 5_000, 25_000, 60_000)
BOND_BUCKETS = (256, 1_024, 5_000, 25_000)
ANGLE_BUCKETS = (256, 1_024, 5_000, 25_000)
DIHEDRAL_BUCKETS = (512, 2_048, 10_000, 50_000)
WATER_BUCKETS = (16, 128, 1_024, 8_000)
EXCL_BUCKETS = (512, 2_048, 10_000, 50_000)
CMAP_BUCKETS = (16, 128, 512)
EXCEPTION_BUCKETS = (512, 2_048, 10_000, 50_000)


@dataclass(frozen=True)
class MolecularShapeSpec:
    """Hashable static descriptor — the JIT cache key for MolecularBundle.

    All fields are plain Python scalars, enabling deterministic hashing.
    boundary_condition is used by factories to reconstruct displacement_fn
    without closure capture.

    Attributes:
        n_atoms: True count of atoms (real entries in padded arrays)
        n_bonds: True count of bond terms
        n_angles: True count of angle terms
        n_dihedrals: True count of proper dihedral terms
        n_impropers: True count of improper dihedral terms
        n_urey_bradley: True count of Urey-Bradley 1-3 terms
        n_waters: True count of SETTLE water molecules
        n_excl: True count of nonbonded exclusion pairs
        n_cmap: True count of CMAP cross-terms
        n_exception_pairs: True count of 1-4 exception pairs
        has_pbc: Whether periodic boundary conditions are enabled
        has_implicit_solvent: Whether implicit solvent is present
        boundary_condition: "free" or "periodic" — used to reconstruct displacement_fn
    """

    n_atoms: int
    n_bonds: int
    n_angles: int
    n_dihedrals: int
    n_impropers: int
    n_urey_bradley: int
    n_waters: int
    n_excl: int
    n_cmap: int
    n_exception_pairs: int
    has_pbc: bool
    has_implicit_solvent: bool
    boundary_condition: Literal["free", "periodic"] = "periodic"


class MolecularBundle(eqx.Module):
    """Flat host-to-device boundary: all fields concrete, no Optional.

    Topology arrays are padded to bucket sizes (ATOM_BUCKETS, BOND_BUCKETS, etc.).
    Real entries are identified by mask arrays (atom_mask, bond_mask, etc.).

    Design constraints:
    - All topology arrays are DYNAMIC (not static=True) to avoid XLA recompilation
      per distinct topology. Instead, XLA caches on bucketed size.
    - shape_spec is the ONLY eqx.field(static=True) — it carries bucketed shapes
      as plain Python ints, serving as the JIT cache key.
    - displacement_fn is NOT stored — it's reconstructed from shape_spec.boundary_condition
      inside energy and integrator factories.
    - No Optional fields: all arrays are concrete and pre-allocated.
    """

    # Per-atom arrays (padded to ATOM_BUCKETS)
    positions: Float[Array, "N 3"]
    charges: Float[Array, "N"]
    sigmas: Float[Array, "N"]
    epsilons: Float[Array, "N"]
    radii: Float[Array, "N"]
    scaled_radii: Float[Array, "N"]
    atom_mask: Bool[Array, "N"]

    # Periodic boundary condition box (zero array when has_pbc=False)
    box: Float[Array, "3 3"]

    # Bond terms (padded to BOND_BUCKETS)
    bond_idx: Int[Array, "B 2"]
    bond_params: Float[Array, "B 2"]
    bond_mask: Bool[Array, "B"]

    # Angle terms (padded to ANGLE_BUCKETS)
    angle_idx: Int[Array, "A 3"]
    angle_params: Float[Array, "A 2"]
    angle_mask: Bool[Array, "A"]

    # Proper dihedral terms (padded to DIHEDRAL_BUCKETS)
    dihedral_idx: Int[Array, "D 4"]
    dihedral_params: Float[Array, "D 4"]
    dihedral_mask: Bool[Array, "D"]

    # Improper dihedral terms (padded to DIHEDRAL_BUCKETS)
    improper_idx: Int[Array, "I 4"]
    improper_params: Float[Array, "I 3"]
    improper_mask: Bool[Array, "I"]
    improper_is_periodic: Bool[Array, ""]

    # Urey-Bradley 1-3 interaction terms (padded to ANGLE_BUCKETS)
    urey_bradley_idx: Int[Array, "U 3"]
    urey_bradley_params: Float[Array, "U 2"]
    urey_bradley_mask: Bool[Array, "U"]

    # CMAP cross-term tables (padded to CMAP_BUCKETS)
    cmap_torsion_idx: Int[Array, "CM 8"]
    cmap_energy_grids: Float[Array, "CM G G"]
    cmap_mask: Bool[Array, "CM"]

    # SETTLE rigid water molecules (padded to WATER_BUCKETS)
    water_indices: Int[Array, "W 3"]
    water_mask: Bool[Array, "W"]

    # Nonbonded exclusion pairs (padded to EXCL_BUCKETS)
    excl_indices: Int[Array, "E 2"]
    excl_scales_vdw: Float[Array, "E"]
    excl_scales_elec: Float[Array, "E"]
    excl_mask: Bool[Array, "E"]

    # 1-4 exception pairs (special LJ/Coulomb scaling, padded to EXCEPTION_BUCKETS)
    exception_pairs: Int[Array, "X 2"]
    exception_sigmas: Float[Array, "X"]
    exception_epsilons: Float[Array, "X"]
    exception_chargeprods: Float[Array, "X"]
    exception_mask: Bool[Array, "X"]

    # Nonbonded computation parameters
    pme_alpha: Float[Array, ""]
    cutoff_distance: Float[Array, ""]

    # Static shape descriptor (the ONLY static=True field)
    shape_spec: MolecularShapeSpec = eqx.field(static=True)
