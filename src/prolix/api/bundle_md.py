"""MolecularBundle-backed energy and displacement helpers for EnsemblePlan.run().

JIT contract
------------
``bundle.n_atoms``, ``bundle.n_waters``, etc. are **dynamic scalar arrays**. Never
call ``int()`` on them inside ``jax.vmap`` / ``jit`` — that forces concretization
and extra recompiles.

Patterns:
- **Static batch prefix** (``integration_prefix: int``): host-static **atom
  bucket** length when ``can_jit_vmap_n_mols`` holds (same ``shape_spec`` /
  padded shapes; real ``n_atoms`` may differ). Passed as a closure constant
  into vmap so XLA sees a static slice, not a traced length.
- **Bucket size** (``atom_bucket_size``): from ``shape_spec`` only — compile-time
  constant per bucket.
- **Masses on stacked path**: ``masses_with_prefix`` (real ``bundle.masses``),
  never unit masses.
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
    single_padded_force,
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


def masses_with_prefix(bundle: MolecularBundle, prefix: int) -> jnp.ndarray:
    """Prefix slice of real ``bundle.masses``; ``prefix`` is host static (atom bucket)."""
    return bundle.masses[:prefix]


def unit_masses(prefix: int, dtype) -> jnp.ndarray:
    """Unit masses for ``prefix`` atoms; ``prefix`` is host static.

    Prefer ``masses_with_prefix`` on EnsemblePlan stacked paths.
    """
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


# B1's only periodic shape class today (4-water TIP3P) uses a 30x30x30 Å box
# (scripts/benchmarks/b1_init_exec.py:_four_water_bundle). Matches the same
# scoped-default philosophy as pme_alpha's 0.34 fallback below -- not a
# general solution, but correct for every periodic system this codebase
# currently constructs. See _host_box_size's docstring for why a fallback
# is needed at all.
#
# Plain Python tuple, NOT a module-level jnp.array: a single shared JAX array
# object referenced as a fallback across many different jax.jit trace
# contexts (e.g. repeated EnsemblePlan.run() calls under vmap, as happens in
# _run_stacked_dispatch) can trigger "leaked tracer" errors from a later
# trace picking up a stale reference tied to an earlier, already-exited
# trace. Constructing a fresh jnp.array from this tuple inside
# _host_box_size on every call avoids any possibility of that -- found while
# testing solvate_protein_to_bundle under real stacked dispatch (2+
# heterogeneous solvated proteins), where it reproduces reliably; B1's
# water class never triggered it, likely because its box_size fallback
# path is exercised far less variably (single topology, not a genuine mix).
_DEFAULT_PBC_BOX_SIZE = (30.0, 30.0, 30.0)


def _host_box_size(bundle: MolecularBundle, default: tuple[float, float, float]) -> jnp.ndarray:
    """Concrete (3,) box size for PhysicsSystem's static box_size field.

    ``MolecularBundle.box`` is deliberately a DYNAMIC field (see
    ``MolecularBundle``'s docstring: "All topology arrays are DYNAMIC ... to
    avoid XLA recompilation per distinct topology") -- but ``PhysicsSystem.
    box_size`` is ``eqx.field(static=True)``, since real FFT grid dimensions
    genuinely depend on its concrete value. Under EnsemblePlan's stacked/
    vmapped N_MOLS dispatch (B1's real production path for any B > 1), the
    per-bundle ``bundle.box`` a caller sees inside the vmapped closure is a
    BatchTracer, not a concrete array, regardless of whether every replica in
    the stack is physically identical (which they are, for a shape-class
    group built by tiling one template -- vmap doesn't know that a priori).

    Before this fix, ``box_size`` was derived unconditionally
    (``jnp.diag(bundle.box)``) with no fallback -- assigning a live tracer
    into a field declared ``static=True`` (silently, no error at
    construction time; Equinox just warns). Downstream, ``jax.grad``'s
    ``jax.eval_shape`` probe caught the resulting ``ConcretizationTypeError``
    correctly (that was the intended, documented case), but the SAME
    exception fires on every REAL evaluation too (not just the probe) once
    box_size is a real tracer -- and ``single_padded_energy``'s except clause
    (necessarily broad, to catch the probe) silently swallowed it every time,
    meaning PME reciprocal-space energy never actually executed for any
    multi-bundle stacked dispatch. Confirmed via 4 independent tests
    (research record 260717_pme_reciprocal_silently_disabled_under_stacked_dispatch):
    real ``EnsemblePlan.run()`` trajectories were bit-identical regardless of
    ``pme_alpha`` whenever dispatched through the stacked (B > 1) path, while
    single-bundle dispatch and hand-wired settle_langevin calls both showed
    genuine PME sensitivity.

    Mirrors the existing ``_host_float`` fallback pattern (already used for
    ``pme_alpha``/``nonbonded_cutoff`` below) rather than introducing a new
    approach -- same accepted tradeoff, same limitation: this is correct for
    every periodic system this codebase currently constructs, not a general
    fix for arbitrarily-varying box sizes within one stacked group.
    """
    diag = jnp.diag(jnp.asarray(bundle.box))
    try:
        # jnp.diag itself never raises under tracing (it's a normal traced op)
        # -- force concretization per-element via float(), mirroring
        # _host_float, so a traced bundle.box is actually detected here
        # rather than silently propagating as a "static" tracer.
        return jnp.array([float(diag[0]), float(diag[1]), float(diag[2])])
    except (TypeError, ValueError, jax.errors.ConcretizationTypeError):
        return jnp.array(default)


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
        box_size = _host_box_size(bundle, _DEFAULT_PBC_BOX_SIZE).astype(positions.dtype)

    dense_vdw, dense_elec = _dense_excl_matrices_from_bundle(bundle, n)

    # Padding-safe pair-list exclusions for the PME reciprocal-space correction
    # (_pme_reciprocal_and_corrections in batched_energy.py) -- the dense direct-
    # space path above doesn't need this (it already has dense_vdw/dense_elec),
    # but the PME correction requires a per-pair (i, j) list, and MUST be
    # precomputed here (bundle-construction time, always concrete/host-side) --
    # unlike box_size/pme_alpha above, there's no viable trace-time fallback:
    # the natural way to derive per-pair exclusions from raw bonds
    # (topology.find_bonded_exclusions) is pure Python/numpy graph traversal,
    # fundamentally unable to run under jax.vmap/jit tracing at all. Reusing
    # bundle.excl_indices/excl_scales_elec here (already correctly populated
    # host-side at bundle construction, already vmap-safe) avoids needing that
    # traversal a second time. Padding rows are redirected to index `n`
    # (guaranteed out-of-bounds, safely excluded by any `< n_atoms` bounds
    # check) rather than left as whatever sentinel bundle.excl_mask marks --
    # same defensive pattern as _scatter_water_target's padding-row redirect
    # for SETTLE (B1-SETTLE-STACK, .praxia/docs/specs/260715_b1-settle-stack.md).
    excl_pair_indices = jnp.where(
        bundle.excl_mask[:, None], bundle.excl_indices, n
    )

    # Per-atom-row exclusions (debt 765) for NL/flash kernels -- distinct from
    # the pair-list excl_pair_indices above, which only the PME exclusion
    # correction reads. Populated at bundle-construction time
    # (make_bundle_from_system, host-side, debt 765) -- None if the bundle
    # was built without an exclusion_spec (no NL/flash kernel support for
    # that bundle yet; matches every existing None-check consumer).
    excl_dense_indices = (
        _slice1(bundle.excl_dense_indices)
        if bundle.excl_dense_indices is not None
        else None
    )
    excl_dense_scales_vdw = (
        _slice1(bundle.excl_dense_scales_vdw)
        if bundle.excl_dense_scales_vdw is not None
        else None
    )
    excl_dense_scales_elec = (
        _slice1(bundle.excl_dense_scales_elec)
        if bundle.excl_dense_scales_elec is not None
        else None
    )

    # Static PhysicsSystem floats: concrete on host; defaults when traced under vmap.
    # Free-space (no PBC) never takes the PME branch; cutoff default matches AKMA tip3p.
    # PBC fallback is 0.34, matching prolix.physics.system.make_energy_fn's canonical
    # default_pme_alpha (system.py:123) -- NOT an arbitrary choice. bundle.pme_alpha is
    # a per-bundle scalar that becomes a traced value under vmap (e.g. B1's stacked/
    # N_MOLS dispatch, replicas batched via vmap), so _host_float's exception-based
    # fallback fires on every such call, silently discarding whatever concrete alpha
    # the bundle was actually constructed with. Before this fix the fallback was 0.3,
    # which silently disagreed with the 0.34 callers (e.g. scripts/benchmarks/
    # b1_init_exec.py's _four_water_bundle) explicitly configure -- found via
    # B1-NONBONDED-PARITY code review (.praxia/docs/specs/260715_b1-nonbonded-parity.md):
    # a parity test validated 0.34 through a non-vmapped single-bundle call, which does
    # not exercise this fallback at all, so it did not catch that the real vmapped B1
    # dispatch would silently use a different (and disagreeing-with-OpenMM) alpha.
    pme_alpha = _host_float(bundle.pme_alpha, 0.0 if not bundle.shape_spec.has_pbc else 0.34)
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
        excl_indices=excl_dense_indices,
        excl_scales_vdw=excl_dense_scales_vdw,
        excl_scales_elec=excl_dense_scales_elec,
        excl_pair_indices=excl_pair_indices,
        excl_pair_scales_vdw=bundle.excl_scales_vdw,
        excl_pair_scales_elec=bundle.excl_scales_elec,
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
        # `neighbor` (debt 760's NL path, see single_padded_energy's docstring)
        # is the only kwarg this energy function understands -- forward it
        # through. Everything else is dropped, matching prior behavior (this
        # closure is called via jax_md's canonicalize_force/eval_shape probing,
        # which may pass conventional extra kwargs like `t`/`box` this function
        # doesn't use).
        neighbor = kwargs.pop("neighbor", None)
        del kwargs
        sys = physics_system_from_bundle(bundle, positions)
        e = single_padded_energy(sys, disp_fn, implicit_solvent=False, neighbor=neighbor)
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


def force_fn_from_bundle(bundle: MolecularBundle) -> Callable[..., jnp.ndarray]:
    """Analytical/FlashMD forces from bundle fields (debt 761).

    Drop-in force-returning alternative to ``energy_fn_from_bundle`` for
    ``settle_langevin``'s ``energy_or_force_fn`` parameter (auto-detected via
    ``jax.eval_shape``'s output shape in ``_make_settle_compatible_force_fn`` /
    ``make_force_fn_like_canonicalize`` -- a force-shaped ``(N, 3)`` return
    skips the dense-energy autodiff pass entirely). Uses
    ``single_padded_force(..., use_flash=True)`` -- FlashMD's tiled,
    checkpointed direct-space kernel (``flash_explicit_forces``) -- instead
    of ``jax.grad`` through ``single_padded_energy``'s dense O(N^2) energy.
    The AMBER 1-4 exception term (absent from ``single_padded_force`` itself)
    is added back via a cheap ``jax.grad`` pass, mirroring
    ``energy_fn_from_bundle``'s own handling, so this is a faithful,
    physics-complete substitute -- not merely the non-bonded term.

    Verified (2026-07-19) to float32 precision (rel diff ~1.3e-5) against
    ``-jax.grad(energy_fn_from_bundle(bundle))`` on real periodic,
    PME-solvated 1VII/2GB1 bundles (see
    ``tests/physics/test_flash_dense_parity.py`` and
    ``.praxia/docs/decisions/260717_b1-connect-existing-engines-scope.md``).
    Raises for bundle shapes outside that verified scope rather than silently
    using an unvalidated path: implicit-solvent (``flash_explicit_forces``/
    ``flash_nonbonded_forces`` have no GB term at all) and non-periodic/vacuum
    bundles (the ``flash_nonbonded_forces`` branch was never benchmarked or
    parity-tested).
    """
    if bundle.shape_spec.has_implicit_solvent:
        raise ValueError(
            "force_fn_from_bundle (debt 761 FlashMD path) does not support "
            "implicit-solvent (GB) bundles -- flash_explicit_forces/"
            "flash_nonbonded_forces have no GB term. Use energy_fn_from_bundle "
            "(autodiff path) for implicit-solvent systems."
        )
    if not bundle.shape_spec.has_pbc:
        raise ValueError(
            "force_fn_from_bundle (debt 761 FlashMD path) is only verified for "
            "periodic explicit-solvent/PME bundles -- the vacuum "
            "flash_nonbonded_forces branch was not benchmarked or "
            "parity-tested. Use energy_fn_from_bundle (autodiff path) for "
            "vacuum/non-periodic systems."
        )

    disp_fn, _ = displacement_fn_for_bundle(bundle)

    def force_fn(positions: jnp.ndarray, **kwargs: object) -> jnp.ndarray:
        # single_padded_force has no neighbor-list branch (debt 761 is scoped
        # to the dense flash path only) -- callers must not combine
        # use_flash_forces with use_neighbor_list (EnsemblePlan.run() raises
        # before this is ever reached with a real NL request), so any
        # incoming kwargs here are jax_md probing artifacts
        # (eval_shape/canonicalize_force), matching energy_fn_from_bundle's
        # own "everything else is dropped" contract.
        del kwargs
        sys = physics_system_from_bundle(bundle, positions)
        f = single_padded_force(
            sys, disp_fn, implicit_solvent=False, explicit_solvent=True, use_flash=True,
        )
        f_exception = -jax.grad(
            lambda r: _exception_energy_masked(
                r,
                bundle.exception_pairs,
                bundle.exception_sigmas,
                bundle.exception_epsilons,
                bundle.exception_chargeprods,
                bundle.exception_mask,
                disp_fn,
            )
        )(positions)
        return f + f_exception

    return force_fn
