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

import jax
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

# Match prolix.physics.bonded.make_exception_pair_energy_fn default (AKMA).
_EXCEPTION_COULOMB = 332.0637


def _exception_energy_masked(
    r: jnp.ndarray,
    pairs: jnp.ndarray,
    sigmas: jnp.ndarray,
    epsilons: jnp.ndarray,
    chargeprods: jnp.ndarray,
    mask: jnp.ndarray,
    displacement_fn: space.DisplacementFn,
) -> jnp.ndarray:
    """1-4 exception LJ+Coulomb with per-pair mask (padded EXCEPTION_BUCKETS)."""
    i = pairs[:, 0].astype(jnp.int32)
    j = pairs[:, 1].astype(jnp.int32)
    n = r.shape[0]
    # Safe gather for pad slots (mask zeros their contribution).
    i_safe = jnp.clip(i, 0, jnp.maximum(n - 1, 0))
    j_safe = jnp.clip(j, 0, jnp.maximum(n - 1, 0))
    ri = r[i_safe]
    rj = r[j_safe]
    dr = jax.vmap(displacement_fn)(ri, rj)
    dist = jnp.sqrt(jnp.sum(dr**2, axis=-1) + jnp.float32(1e-12))
    inv_r = sigmas / dist
    inv_r6 = inv_r**6
    e_lj = jnp.float32(4.0) * epsilons * (inv_r6**2 - inv_r6)
    e_coul = jnp.float32(_EXCEPTION_COULOMB) * chargeprods / dist
    return jnp.sum((e_lj + e_coul) * mask.astype(r.dtype))


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
    always populates it (real per-atom masses or unit-mass default). Branching on
    array *values* here would raise ``TracerBoolConversionError`` under
    ``energy_fn_from_bundle``'s ``jit`` trace — this function must stay a pure
    slice (XR-A2A3 A1).
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


def _host_float(x: object, default: float) -> float:
    """Concrete float for PhysicsSystem static fields; default when traced."""
    try:
        return float(jnp.asarray(x))
    except (TypeError, ValueError, jax.errors.ConcretizationTypeError):
        return default


def _dense_excl_matrices_from_bundle(
    bundle: MolecularBundle,
    n_atoms: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build (N, N) exclusion scale matrices from padded pair lists (JIT-safe).

    ``n_atoms`` must be a host-static int (positions prefix / bucket slice).
    Invalid or masked pairs leave the corresponding entries at 1.0.
    """
    dense_vdw = jnp.ones((n_atoms, n_atoms), dtype=jnp.float32)
    dense_elec = jnp.ones((n_atoms, n_atoms), dtype=jnp.float32)
    pairs = bundle.excl_indices
    i = pairs[:, 0]
    j = pairs[:, 1]
    valid = (
        bundle.excl_mask
        & (i >= 0)
        & (j >= 0)
        & (i < n_atoms)
        & (j < n_atoms)
    )
    i_safe = jnp.clip(i, 0, max(n_atoms - 1, 0))
    j_safe = jnp.clip(j, 0, max(n_atoms - 1, 0))
    sv = jnp.where(valid, bundle.excl_scales_vdw.astype(jnp.float32), 1.0)
    se = jnp.where(valid, bundle.excl_scales_elec.astype(jnp.float32), 1.0)
    # Keep ones for invalid slots: only write when valid.
    dense_vdw = dense_vdw.at[i_safe, j_safe].set(
        jnp.where(valid, sv, dense_vdw[i_safe, j_safe])
    )
    dense_vdw = dense_vdw.at[j_safe, i_safe].set(
        jnp.where(valid, sv, dense_vdw[j_safe, i_safe])
    )
    dense_elec = dense_elec.at[i_safe, j_safe].set(
        jnp.where(valid, se, dense_elec[i_safe, j_safe])
    )
    dense_elec = dense_elec.at[j_safe, i_safe].set(
        jnp.where(valid, se, dense_elec[j_safe, i_safe])
    )
    return dense_vdw, dense_elec


def physics_system_from_bundle(
    bundle: MolecularBundle,
    positions: jnp.ndarray,
) -> PhysicsSystem:
    """Reconstruct a PhysicsSystem view for ``single_padded_energy``.

    ``positions.shape[0]`` must be host-static (integration prefix or host trim).
    Topology uses padded arrays + masks — never ``int(bundle.n_*)`` — so this
    path is safe under ``jax.vmap`` / stacked N_MOLS dispatch.
    """
    n = int(positions.shape[0])
    n_real = jnp.asarray(bundle.n_atoms, dtype=jnp.int32)

    def _slice1(arr):
        return arr[:n]

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

    # Static PhysicsSystem floats: concrete on host; defaults when traced under vmap.
    # Free-space (no PBC) never takes the PME branch; cutoff default matches AKMA tip3p.
    pme_alpha = _host_float(bundle.pme_alpha, 0.0 if not bundle.shape_spec.has_pbc else 0.3)
    nonbonded_cutoff = _host_float(bundle.cutoff_distance, 9.0)

    return PhysicsSystem(
        positions=positions,
        charges=_slice1(bundle.charges),
        sigmas=_slice1(bundle.sigmas),
        epsilons=_slice1(bundle.epsilons),
        radii=_slice1(bundle.radii),
        scaled_radii=_slice1(bundle.scaled_radii),
        masses=bundle.masses[:n],
        element_ids=jnp.zeros(n, dtype=jnp.int32),
        atom_mask=_slice1(bundle.atom_mask),
        is_hydrogen=jnp.zeros(n, dtype=bool),
        is_backbone=jnp.zeros(n, dtype=bool),
        is_heavy=_slice1(bundle.atom_mask),
        protein_atom_mask=jnp.zeros(n, dtype=bool),
        water_atom_mask=jnp.zeros(n, dtype=bool),
        bonds=bundle.bond_idx,
        bond_params=bundle.bond_params,
        bond_mask=bundle.bond_mask,
        angles=bundle.angle_idx,
        angle_params=bundle.angle_params,
        angle_mask=bundle.angle_mask,
        dihedrals=bundle.dihedral_idx,
        dihedral_params=flat_dih,
        dihedral_mask=bundle.dihedral_mask,
        impropers=bundle.improper_idx,
        improper_params=flat_imp,
        improper_mask=bundle.improper_mask,
        urey_bradley_bonds=bundle.urey_bradley_idx,
        urey_bradley_params=bundle.urey_bradley_params,
        urey_bradley_mask=bundle.urey_bradley_mask,
        water_indices=bundle.water_indices,
        water_mask=bundle.water_mask,
        n_real_atoms=jnp.minimum(jnp.asarray(n, dtype=jnp.int32), n_real),
        n_padded_atoms=n,
        box_size=box_size,
        pme_alpha=pme_alpha,
        pme_grid_points=64,
        nonbonded_cutoff=nonbonded_cutoff,
        dense_excl_scale_vdw=dense_vdw,
        dense_excl_scale_elec=dense_elec,
    )


def energy_fn_from_bundle(
    bundle: MolecularBundle,
    *,
    include_nonbonded: bool = True,
) -> Callable[..., jnp.ndarray]:
    """Total energy from bundle fields (bonded + optional nonbonded via ``single_padded_energy``).

    Includes AMBER 1-4 ``exception_*`` pair energy when present (XR-PARITY-OMM-PROTEIN).
    """
    if not include_nonbonded:
        return bonded_energy_fn_from_bundle(bundle)

    disp_fn, _ = displacement_fn_for_bundle(bundle)

    def energy_fn(positions: jnp.ndarray, **kwargs: object) -> jnp.ndarray:
        del kwargs
        sys = physics_system_from_bundle(bundle, positions)
        e = single_padded_energy(sys, disp_fn, implicit_solvent=False)
        e = e + _exception_energy_masked(
            positions,
            bundle.exception_pairs,
            bundle.exception_sigmas,
            bundle.exception_epsilons,
            bundle.exception_chargeprods,
            bundle.exception_mask,
            disp_fn,
        )
        return e

    return energy_fn
