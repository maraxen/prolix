"""Host-side factory for constructing FittingBundle with field resolution.

This module provides the single entry point for bundle construction from raw
arrays, following the tev_design pattern. All Optional fields are resolved to
concrete zero-filled arrays before the JIT boundary, ensuring clean separation
of host-side setup from JIT-traced computation.

Type annotations use jaxtyping (Float, Int, Bool, Array) without runtime
@jaxtyped decorators (type-annotation only, matching tev_design pattern).
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from prolix.fitting.bundles import ConformerBundle, FittingBundle
from prolix.fitting.params import BondedParams
from prolix.fitting.topology import BondedTopology


def build_fitting_bundle(
    positions_all: Float[Array, "N_conf n_atoms 3"] | Float[Array, "n_atoms 3"],
    forces_all: Float[Array, "N_conf n_atoms 3"] | Float[Array, "n_atoms 3"],
    energies_all: Float[Array, "N_conf"] | Float[Array, ""],
    params: BondedParams,
    topology: BondedTopology,
    *,
    atom_mask: Bool[Array, "n_atoms"] | None = None,
    box: Float[Array, "3 3"] | None = None,
    n_conf_real: int | None = None,
) -> FittingBundle:
    """Host-side factory for FittingBundle construction.

    Normalizes input shapes, resolves Optional fields to concrete arrays,
    and validates invariants before assembly. This is the single entry point
    for constructing FittingBundle; callers should never manually construct
    ConformerBundle or pass None fields to JIT boundaries.

    Shape normalization (mirrors tev_design bundle_builder.py:46-67):
    - If positions_all.ndim == 2 (single conformer), add leading axis → (1, n_atoms, 3)
    - If positions_all.ndim == 3, accept as-is → (N_conf, n_atoms, 3)
    - Same for forces_all
    - If energies_all.ndim == 0 (scalar), expand → (1,)
    - If energies_all.ndim == 1, accept as-is → (N_conf,)

    Optional field resolution:
    - atom_mask=None → jnp.ones(n_atoms, dtype=jnp.bool_)  [all real, no padding]
    - box=None → jnp.zeros((3, 3), dtype=jnp.float32)  [vacuum sentinel]
    - n_conf_real=None → positions_all.shape[0]  [all conformers are real]

    Static field discipline:
    - n_conf_real and n_atoms are Python ints (or convertible to int)
    - These are passed to ConformerBundle as eqx.field(static=True)
    - Never traced JAX arrays

    Args:
        positions_all: Atomic coordinates. Shape (n_atoms, 3) or (N_conf, n_atoms, 3).
        forces_all: Reference forces (e.g., from QM). Same shape as positions_all.
        energies_all: Reference energies. Shape (N_conf,) or scalar.
        params: BondedParams instance (trainable bonded parameters).
        topology: BondedTopology instance (static connectivity).
        atom_mask: Binary mask (True=real, False=padding). Default all-ones.
        box: Simulation box matrix (3, 3). Default zero matrix (vacuum).
        n_conf_real: Number of real conformers in buffer. Default = positions_all.shape[0].

    Returns:
        FittingBundle ready for JIT-compiled training.

    Raises:
        ValueError: If shapes are incompatible or invariants violated.
    """

    # ===== SHAPE NORMALIZATION =====

    # Normalize positions_all: (n_atoms, 3) → (1, n_atoms, 3) or keep (N_conf, n_atoms, 3)
    if positions_all.ndim == 2:
        positions_all = positions_all[None, ...]  # Add batch axis
    elif positions_all.ndim != 3:
        raise ValueError(
            f"positions_all must be 2D (n_atoms, 3) or 3D (N_conf, n_atoms, 3), "
            f"got shape {positions_all.shape} with ndim={positions_all.ndim}"
        )

    # Normalize forces_all: same shape as positions_all
    if forces_all.ndim == 2:
        forces_all = forces_all[None, ...]
    elif forces_all.ndim != 3:
        raise ValueError(
            f"forces_all must be 2D (n_atoms, 3) or 3D (N_conf, n_atoms, 3), "
            f"got shape {forces_all.shape} with ndim={forces_all.ndim}"
        )

    # Normalize energies_all: scalar → (1,) or keep (N_conf,)
    if energies_all.ndim == 0:
        energies_all = energies_all[None]
    elif energies_all.ndim != 1:
        raise ValueError(
            f"energies_all must be 1D (N_conf,) or scalar, "
            f"got shape {energies_all.shape} with ndim={energies_all.ndim}"
        )

    # After normalization, all three should be (N_conf, ...)
    N_conf, n_atoms = positions_all.shape[0], positions_all.shape[1]

    # ===== VALIDATION INVARIANTS =====

    # Check shapes match
    if positions_all.shape != forces_all.shape:
        raise ValueError(
            f"positions_all and forces_all must have identical shapes after normalization. "
            f"Got positions_all={positions_all.shape}, forces_all={forces_all.shape}"
        )

    if energies_all.shape[0] != N_conf:
        raise ValueError(
            f"energies_all.shape[0] ({energies_all.shape[0]}) must equal N_conf ({N_conf})"
        )

    # ===== OPTIONAL FIELD RESOLUTION =====

    # atom_mask: default to all-ones (no padding)
    if atom_mask is None:
        atom_mask = jnp.ones(n_atoms, dtype=jnp.bool_)
    else:
        if atom_mask.shape[0] != n_atoms:
            raise ValueError(
                f"atom_mask.shape[0] ({atom_mask.shape[0]}) must equal n_atoms ({n_atoms})"
            )

    # box: default to zero matrix (vacuum sentinel)
    if box is None:
        box = jnp.zeros((3, 3), dtype=jnp.float32)
    else:
        if box.shape != (3, 3):
            raise ValueError(
                f"box must have shape (3, 3), got shape {box.shape}"
            )

    # n_conf_real: default to all conformers are real
    if n_conf_real is None:
        n_conf_real = N_conf
    else:
        if n_conf_real > N_conf:
            raise ValueError(
                f"n_conf_real ({n_conf_real}) cannot exceed buffer size N_conf ({N_conf})"
            )
        if n_conf_real < 1:
            raise ValueError(
                f"n_conf_real must be ≥ 1, got {n_conf_real}"
            )

    # ===== STATIC FIELD DISCIPLINE =====

    # Ensure n_atoms and n_conf_real are Python ints (not traced JAX arrays)
    n_atoms = int(n_atoms)
    n_conf_real = int(n_conf_real)

    # ===== ASSEMBLY =====

    # Build ConformerBundle (per-mol, no batch axis)
    conformer_bundle = ConformerBundle(
        positions=positions_all,
        forces_ref=forces_all,
        energies_ref=energies_all,
        atom_mask=atom_mask,
        n_conf=n_conf_real,
        n_atoms=n_atoms,
    )

    # Build and return FittingBundle
    fitting_bundle = FittingBundle(
        conformers=conformer_bundle,
        params=params,
        topology=topology,
        box=box,
    )

    return fitting_bundle
