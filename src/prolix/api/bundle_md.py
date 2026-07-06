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
    single_padded_energy,
)
from prolix.types.bundles import ATOM_BUCKETS, WATER_BUCKETS, MolecularBundle
from prolix.typing import PhysicsSystem


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
    """Real-prefix masses for host-side single-system runs.

    Reads the bundle's own ``masses`` field directly — ``make_bundle_from_system``
    (the sole production constructor) always populates it (real per-atom masses
    from the source system, or a unit-mass default when none was supplied), so
    no runtime fallback is needed. Branching on array *values* here (e.g. "are
    all masses 1.0?") would raise ``TracerBoolConversionError`` under
    ``energy_fn_from_bundle``'s ``jit`` trace — this function must stay a pure
    slice, never a data-dependent Python branch.
    """
    n_atoms = int(jnp.asarray(bundle.n_atoms))
    return bundle.masses[:n_atoms]


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
    return _bonded_energy_core(bundle)


def _bonded_energy_core(bundle: MolecularBundle) -> Callable[..., jnp.ndarray]:
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


def _dense_excl_matrices_from_bundle(
    bundle: MolecularBundle,
    n_atoms: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build (N, N) exclusion scale matrices from bundle pair lists (host-side)."""
    import numpy as np

    dense_vdw = np.ones((n_atoms, n_atoms), dtype=np.float32)
    dense_elec = np.ones((n_atoms, n_atoms), dtype=np.float32)
    n_excl = int(jnp.asarray(bundle.n_excl))
    if n_excl <= 0:
        return jnp.asarray(dense_vdw), jnp.asarray(dense_elec)

    pairs = np.asarray(bundle.excl_indices[:n_excl])
    active = np.asarray(bundle.excl_mask[:n_excl], dtype=bool)
    sv = np.asarray(bundle.excl_scales_vdw[:n_excl], dtype=np.float32)
    se = np.asarray(bundle.excl_scales_elec[:n_excl], dtype=np.float32)
    for k in range(n_excl):
        if not active[k]:
            continue
        i, j = int(pairs[k, 0]), int(pairs[k, 1])
        if i < 0 or j < 0 or i >= n_atoms or j >= n_atoms:
            continue
        dense_vdw[i, j] = dense_vdw[j, i] = float(sv[k])
        dense_elec[i, j] = dense_elec[j, i] = float(se[k])
    return jnp.asarray(dense_vdw), jnp.asarray(dense_elec)


def physics_system_from_bundle(
    bundle: MolecularBundle,
    positions: jnp.ndarray,
) -> PhysicsSystem:
    """Reconstruct a PhysicsSystem view for ``single_padded_energy`` (host prefix ``N``)."""
    n = int(positions.shape[0])
    n_real = int(jnp.asarray(bundle.n_atoms))

    def _slice1(arr):
        return arr[:n]

    def _slice_mask(mask, count: int):
        return mask[:count]

    nb = int(jnp.asarray(bundle.n_bonds))
    na = int(jnp.asarray(bundle.n_angles))
    nd = int(jnp.asarray(bundle.n_dihedrals))
    ni = int(jnp.asarray(bundle.n_impropers))
    nub = int(jnp.asarray(bundle.n_urey_bradley))
    nw = int(jnp.asarray(bundle.n_waters))

    empty_ub_b = jnp.zeros((0, 3), dtype=jnp.int32)
    empty_ub_p = jnp.zeros((0, 2), dtype=jnp.float32)
    empty_ub_m = jnp.zeros(0, dtype=bool)

    dih_params = bundle.dihedral_params
    if dih_params.ndim == 3:
        flat_dih = dih_params.reshape(-1, dih_params.shape[-1])
    else:
        flat_dih = dih_params[..., :3]

    imp_params = bundle.improper_params
    if imp_params.ndim == 3:
        flat_imp = imp_params.reshape(-1, imp_params.shape[-1])
    else:
        flat_imp = imp_params[..., :3]

    box_size = None
    if bundle.shape_spec.has_pbc:
        box_size = jnp.diag(bundle.box).astype(positions.dtype)

    dense_vdw, dense_elec = _dense_excl_matrices_from_bundle(bundle, n)

    return PhysicsSystem(
        positions=positions,
        charges=_slice1(bundle.charges),
        sigmas=_slice1(bundle.sigmas),
        epsilons=_slice1(bundle.epsilons),
        radii=_slice1(bundle.radii),
        scaled_radii=_slice1(bundle.scaled_radii),
        masses=masses_for_bundle(bundle)[:n],
        element_ids=jnp.zeros(n, dtype=jnp.int32),
        atom_mask=_slice1(bundle.atom_mask),
        is_hydrogen=jnp.zeros(n, dtype=bool),
        is_backbone=jnp.zeros(n, dtype=bool),
        is_heavy=_slice1(bundle.atom_mask),
        protein_atom_mask=jnp.zeros(n, dtype=bool),
        water_atom_mask=jnp.zeros(n, dtype=bool),
        bonds=bundle.bond_idx[:nb],
        bond_params=bundle.bond_params[:nb],
        bond_mask=_slice_mask(bundle.bond_mask, nb),
        angles=bundle.angle_idx[:na],
        angle_params=bundle.angle_params[:na],
        angle_mask=_slice_mask(bundle.angle_mask, na),
        dihedrals=bundle.dihedral_idx[:nd],
        dihedral_params=flat_dih[:nd],
        dihedral_mask=_slice_mask(bundle.dihedral_mask, nd),
        impropers=bundle.improper_idx[:ni],
        improper_params=flat_imp[:ni] if ni else jnp.zeros((0, 3), dtype=flat_imp.dtype),
        improper_mask=_slice_mask(bundle.improper_mask, ni) if ni else jnp.zeros(0, dtype=bool),
        urey_bradley_bonds=bundle.urey_bradley_idx[:nub] if nub else empty_ub_b,
        urey_bradley_params=bundle.urey_bradley_params[:nub] if nub else empty_ub_p,
        urey_bradley_mask=_slice_mask(bundle.urey_bradley_mask, nub) if nub else empty_ub_m,
        water_indices=bundle.water_indices[:nw] if nw else None,
        water_mask=_slice_mask(bundle.water_mask, nw) if nw else None,
        n_real_atoms=jnp.array(min(n, n_real), dtype=jnp.int32),
        n_padded_atoms=n,
        box_size=box_size,
        pme_alpha=float(jnp.asarray(bundle.pme_alpha)),
        pme_grid_points=64,
        nonbonded_cutoff=float(jnp.asarray(bundle.cutoff_distance)),
        dense_excl_scale_vdw=dense_vdw,
        dense_excl_scale_elec=dense_elec,
    )


def energy_fn_from_bundle(
    bundle: MolecularBundle,
    *,
    include_nonbonded: bool = True,
) -> Callable[..., jnp.ndarray]:
    """Total energy from bundle fields (bonded + optional nonbonded via ``single_padded_energy``)."""
    if not include_nonbonded:
        return bonded_energy_fn_from_bundle(bundle)

    disp_fn, _ = displacement_fn_for_bundle(bundle)

    def energy_fn(positions: jnp.ndarray, **kwargs: object) -> jnp.ndarray:
        del kwargs
        sys = physics_system_from_bundle(bundle, positions)
        return single_padded_energy(sys, disp_fn, implicit_solvent=False)

    return energy_fn
