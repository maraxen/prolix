"""MolecularBundle-backed energy and displacement helpers for EnsemblePlan.run().

JIT contract
------------
``bundle.n_atoms``, ``bundle.n_waters``, etc. are **dynamic scalar arrays**. Never
call ``int()`` on them inside ``jax.vmap`` / ``jit`` — that forces concretization
and extra recompiles.

Patterns:
- **Static batch prefix** (``integration_prefix: int``): set once on the host when
  ``can_jit_vmap_n_mols`` holds (all bundles share the same ``n_atoms``). Passed
  as a closure constant into vmap so XLA sees a static slice, not a traced length.
- **Bucket size** (``atom_bucket_size``): from ``shape_spec`` only — compile-time
  constant per bucket, used for padding checks and planner axes, not for
  ``int(bundle.n_atoms)`` inside traced code.
- **Trajectory trim**: ``trim_trajectory_positions`` uses ``int(n_atoms)`` on the
  **host** after integration (or in ``unstack_trajectories``), never mid-trace.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax_md import space

from prolix.batched_energy import (
    _angle_energy_masked,
    _bond_energy_masked,
    _dihedral_energy_masked,
)
from prolix.types.bundles import ATOM_BUCKETS, WATER_BUCKETS, MolecularBundle


def atom_bucket_size(bundle: MolecularBundle) -> int:
    """Static padded atom count from shape_spec (compile-time constant per bucket)."""
    return ATOM_BUCKETS[bundle.shape_spec.atom_bucket_idx]


def water_bucket_size(bundle: MolecularBundle) -> int:
    """Static padded water slot count from shape_spec."""
    return WATER_BUCKETS[bundle.shape_spec.water_bucket_idx]


def positions_with_prefix(bundle: MolecularBundle, prefix: int) -> jnp.ndarray:
    """Prefix slice of positions; ``prefix`` must be a host static int (not traced)."""
    return bundle.positions[:prefix]


def unit_masses(prefix: int, dtype) -> jnp.ndarray:
    """Unit masses for ``prefix`` atoms; ``prefix`` is host static."""
    return jnp.ones(prefix, dtype=dtype)


def trim_trajectory_positions(
    positions: jnp.ndarray, bundle: MolecularBundle
) -> jnp.ndarray:
    """Trim ``(steps, bucket, 3)`` or ``(steps, n, 3)`` to real atoms (host)."""
    n_real = int(jnp.asarray(bundle.n_atoms))
    return positions[:, :n_real, ...]


def as_integration_scalars(
    dt: float | jnp.ndarray,
    kT: float | jnp.ndarray,
    *,
    dtype,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Promote host floats to JAX scalars for traced integration (no recompile on change)."""
    return (
        jnp.asarray(dt, dtype=dtype),
        jnp.asarray(kT, dtype=dtype),
    )


def displacement_fn_for_bundle(
    bundle: MolecularBundle,
) -> tuple[space.DisplacementFn, space.ShiftFn]:
    """Reconstruct JAX-MD displacement and shift from bundle shape_spec."""
    spec = bundle.shape_spec
    if spec.boundary_condition == "periodic" and spec.has_pbc:
        box_vec = jnp.diag(bundle.box)
        return space.periodic(box_vec)
    return space.free()


def tip3p_masses(n_waters: int, dtype=jnp.float64) -> jnp.ndarray:
    """TIP3P O-H-H masses for ``n_waters`` molecules (host ``n_waters``)."""
    per = jnp.array([15.999, 1.008, 1.008], dtype=dtype)
    return jnp.tile(per, n_waters)


def masses_for_bundle(bundle: MolecularBundle) -> jnp.ndarray:
    """Real-prefix masses for host-side single-system runs."""
    n_atoms = int(jnp.asarray(bundle.n_atoms))
    n_waters = int(jnp.asarray(bundle.n_waters))
    if n_waters > 0 and n_atoms == 3 * n_waters:
        return tip3p_masses(n_waters, dtype=bundle.positions.dtype)
    return jnp.ones(n_atoms, dtype=bundle.positions.dtype)


def active_positions(bundle: MolecularBundle) -> jnp.ndarray:
    """Real atom positions (host path)."""
    n_atoms = int(jnp.asarray(bundle.n_atoms))
    return bundle.positions[:n_atoms]


def water_indices_for_integration(bundle: MolecularBundle) -> jnp.ndarray | None:
    """Water indices for SETTLE when the bundle carries waters (host check)."""
    if int(jnp.asarray(bundle.n_waters)) == 0:
        return None
    return bundle.water_indices[: int(jnp.asarray(bundle.n_waters))]


def bonded_energy_fn_from_bundle(
    bundle: MolecularBundle,
) -> Callable[..., jnp.ndarray]:
    """Bonded-only energy callable compatible with settle_langevin."""
    disp_fn, _ = displacement_fn_for_bundle(bundle)

    def energy_fn(positions: jnp.ndarray, **kwargs: object) -> jnp.ndarray:
        del kwargs
        r = positions
        mask_dtype = r.dtype
        e = jnp.array(0.0, dtype=r.dtype)
        e = e + _bond_energy_masked(
            r,
            bundle.bond_idx,
            bundle.bond_params,
            bundle.bond_mask.astype(mask_dtype),
            disp_fn,
        )
        e = e + _angle_energy_masked(
            r,
            bundle.angle_idx,
            bundle.angle_params,
            bundle.angle_mask.astype(mask_dtype),
            disp_fn,
        )
        dih_params = bundle.dihedral_params[..., :3]
        e = e + _dihedral_energy_masked(
            r,
            bundle.dihedral_idx,
            dih_params,
            bundle.dihedral_mask.astype(mask_dtype),
            disp_fn,
        )
        e = e + _dihedral_energy_masked(
            r,
            bundle.improper_idx,
            bundle.improper_params,
            bundle.improper_mask.astype(mask_dtype),
            disp_fn,
        )
        return e

    return energy_fn
