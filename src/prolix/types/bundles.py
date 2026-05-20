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
#
# HP4 (260520) — Refined ATOM_BUCKETS ladder to enable cross-bucket evidence for §7.1 figure.
# Previously (256, 1024, ...) all ANI-1x + most COMP6 molecules fit in bucket[0].
# New ladder (64, 128, 256, 1024, ...) allows:
# - Small molecules (10–64 atoms) → bucket 0
# - Dipeptides (65–128 atoms) → bucket 1
# - Larger peptides (129–256 atoms) → bucket 2
# - Large proteins (257–1024 atoms, e.g. Trp-cage 312) → bucket 3
# - Allows Lane B ensemble to span 4 distinct buckets (cross-bucket heterogeneity).
# Bonded ladders also refined to small-molecule scale; same proportional principle.
ATOM_BUCKETS = (64, 128, 256, 1_024, 5_000, 25_000, 60_000)
# Bonded ladders refined: prepend small-molecule entries (HP3 design forward-compatible)
BOND_BUCKETS = (16, 64, 256, 1_024, 5_000, 25_000)
ANGLE_BUCKETS = (32, 128, 256, 1_024, 5_000, 25_000)
DIHEDRAL_BUCKETS = (32, 128, 512, 2_048, 10_000, 50_000)
# Water / exclusion / exception: proportionally finer, capped at original scale
WATER_BUCKETS = (8, 16, 128, 1_024, 8_000)
EXCL_BUCKETS = (32, 128, 512, 2_048, 10_000, 50_000)
CMAP_BUCKETS = (8, 16, 128, 512)
EXCEPTION_BUCKETS = (32, 128, 512, 2_048, 10_000, 50_000)


def _bucket_idx(n: int, ladder: tuple[int, ...]) -> int:
    """Return the index of the smallest bucket >= n.

    Args:
        n: Actual count (e.g., number of atoms)
        ladder: Sorted tuple of bucket thresholds (e.g., ATOM_BUCKETS)

    Returns:
        Index of smallest bucket containing n. Asserts if n exceeds all buckets.

    Example:
        _bucket_idx(100, (256, 1024, 5000)) -> 0  (256 is at index 0)
        _bucket_idx(500, (256, 1024, 5000)) -> 1  (1024 is at index 1)
    """
    for i, bucket in enumerate(ladder):
        if n <= bucket:
            return i
    # Overflow: n exceeds largest bucket
    assert False, f"Count {n} exceeds all buckets {ladder}"


@dataclass(frozen=True)
class MolecularShapeSpec:
    """Hashable static descriptor — the JIT cache key for MolecularBundle.

    All fields are plain Python scalars. CRITICAL: bucket indices (not real counts)
    enable identical hashing for two systems with different real counts but same
    bucket sizes. This is required for Claim 1 (heterogeneous batch substrate).

    Attributes:
        atom_bucket_idx: Index into ATOM_BUCKETS (all systems in same bucket share key)
        bond_bucket_idx: Index into BOND_BUCKETS
        angle_bucket_idx: Index into ANGLE_BUCKETS
        dihedral_bucket_idx: Index into DIHEDRAL_BUCKETS
        water_bucket_idx: Index into WATER_BUCKETS
        excl_bucket_idx: Index into EXCL_BUCKETS
        cmap_bucket_idx: Index into CMAP_BUCKETS
        exception_bucket_idx: Index into EXCEPTION_BUCKETS
        has_pbc: Whether periodic boundary conditions are enabled
        has_implicit_solvent: Whether implicit solvent is present
        boundary_condition: "free" or "periodic" — used to reconstruct displacement_fn
    """

    atom_bucket_idx: int
    bond_bucket_idx: int
    angle_bucket_idx: int
    dihedral_bucket_idx: int
    water_bucket_idx: int
    excl_bucket_idx: int
    cmap_bucket_idx: int
    exception_bucket_idx: int
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
    - shape_spec is the ONLY eqx.field(static=True) — it carries bucket indices
      (coarse) as plain Python ints, serving as the JIT cache key. Two systems with
      different real counts but same bucket size produce identical shape_spec (enabling
      Claim 1: heterogeneous batch substrate).
    - Real per-system counts are stored as dynamic fields (n_atoms, n_bonds, etc.)
      using atom_mask, bond_mask, etc. to identify real vs padded entries.
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
    n_atoms: Int[Array, ""]  # Real count (derived from atom_mask.sum())

    # Periodic boundary condition box (zero array when has_pbc=False)
    box: Float[Array, "3 3"]

    # Bond terms (padded to BOND_BUCKETS)
    bond_idx: Int[Array, "B 2"]
    bond_params: Float[Array, "B 2"]
    bond_mask: Bool[Array, "B"]
    n_bonds: Int[Array, ""]  # Real count (derived from bond_mask.sum())

    # Angle terms (padded to ANGLE_BUCKETS)
    angle_idx: Int[Array, "A 3"]
    angle_params: Float[Array, "A 2"]
    angle_mask: Bool[Array, "A"]
    n_angles: Int[Array, ""]  # Real count

    # Proper dihedral terms (padded to DIHEDRAL_BUCKETS)
    dihedral_idx: Int[Array, "D 4"]
    dihedral_params: Float[Array, "D 4"]
    dihedral_mask: Bool[Array, "D"]
    n_dihedrals: Int[Array, ""]  # Real count

    # Improper dihedral terms (padded to DIHEDRAL_BUCKETS)
    improper_idx: Int[Array, "I 4"]
    improper_params: Float[Array, "I 3"]
    improper_mask: Bool[Array, "I"]
    improper_is_periodic: Bool[Array, ""]
    n_impropers: Int[Array, ""]  # Real count

    # Urey-Bradley 1-3 interaction terms (padded to ANGLE_BUCKETS)
    urey_bradley_idx: Int[Array, "U 3"]
    urey_bradley_params: Float[Array, "U 2"]
    urey_bradley_mask: Bool[Array, "U"]
    n_urey_bradley: Int[Array, ""]  # Real count

    # CMAP cross-term tables (padded to CMAP_BUCKETS)
    cmap_torsion_idx: Int[Array, "CM 8"]
    cmap_energy_grids: Float[Array, "CM G G"]
    cmap_mask: Bool[Array, "CM"]
    n_cmap: Int[Array, ""]  # Real count

    # SETTLE rigid water molecules (padded to WATER_BUCKETS)
    water_indices: Int[Array, "W 3"]
    water_mask: Bool[Array, "W"]
    n_waters: Int[Array, ""]  # Real count

    # Nonbonded exclusion pairs (padded to EXCL_BUCKETS)
    excl_indices: Int[Array, "E 2"]
    excl_scales_vdw: Float[Array, "E"]
    excl_scales_elec: Float[Array, "E"]
    excl_mask: Bool[Array, "E"]
    n_excl: Int[Array, ""]  # Real count

    # 1-4 exception pairs (special LJ/Coulomb scaling, padded to EXCEPTION_BUCKETS)
    exception_pairs: Int[Array, "X 2"]
    exception_sigmas: Float[Array, "X"]
    exception_epsilons: Float[Array, "X"]
    exception_chargeprods: Float[Array, "X"]
    exception_mask: Bool[Array, "X"]
    n_exception_pairs: Int[Array, ""]  # Real count

    # Nonbonded computation parameters
    pme_alpha: Float[Array, ""]
    cutoff_distance: Float[Array, ""]

    # Static shape descriptor (the ONLY static=True field)
    shape_spec: MolecularShapeSpec = eqx.field(static=True)
