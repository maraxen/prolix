"""End-to-end multi-bucket, multi-system, and multi-GPU dynamics benchmark & accuracy audit (Pallas v2.4).

Profiles and audits:
1. Ragged-bucket ghost-atom filtering & compaction speedup:
   - Compares un-filtered NL / cell-prune (v2.3) vs ghost-free cell-prune + compacted NL + CTA-level empty-tile skips (v2.4)
     for a 6,658-atom solvated ubiquitin-scale system in an 8,192 bucket and a 25,000 ragged bucket.
2. SETTLE O-step + velocity projection speedup:
   - Measures fused single-pass water COM/inertia tensor + direct dv_tuple velocity projection on 7,023 rigid waters (B=1, 4, 8).
3. End-to-end EnsemblePlan.run(run_mode="inference", use_neighbor_list=True, use_pallas=True, use_pallas_pme=True)
   across bucket sizes (N_ATOMS) and batch sizes (N_MOLS / B):
   - Homogeneous 23,558-atom solvated DHFR (25,000 bucket) at B = 1, 2, 4, 8 (cold vs warm _STACKED_INFERENCE_CACHE).
   - Homogeneous 6,658-atom solvated protein (8,192 bucket) at B = 1, 4, 8.
   - Heterogeneous same-bucket systems (distinct atom counts 4,200 + 6,658 in shared 8,192 bucket).
   - Heterogeneous multi-bucket systems ([4200-atom, 6658-atom in 8192 bucket] + [DHFR WT, DHFR mutant in 25000 bucket]).
   - DHFR wild-type + mutant variant (perturbed charges & LJ sigmas/epsilons in shared 25,000 bucket).
4. Numerical & physical accuracy audit against ground truth:
   - Pallas v2.4 vs XLA reference energy & force parity.
   - Ragged-bucket padding invariance (8,192 bucket vs 25,000 bucket on identical 6,658 real atoms).
   - Stacked multi-replica batching parity vs independent single-system runs.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import xtrax.profiling as xp

import prolix.physics.neighbor_list as nl_mod
import scripts.experiments.bench_dhfr_parity as B_dhfr
from prolix.api.bundle_md import displacement_fn_for_bundle, energy_fn_from_bundle
from prolix.api.ensemble_plan import (
    _SINGLE_INFERENCE_CACHE,
    _STACKED_INFERENCE_CACHE,
    EnsemblePlan,
    _encode_bundle_nl,
)
from prolix.pallas_kernels import pallas_nonbonded_nl, resolve_pallas_compiler_params


def _probe_ms(fn, *args, n_steady: int = 3, n_wall: int = 7) -> tuple[float, dict[str, float]]:
    res = xp.probe_unit(fn, args, n_steady=n_steady, n_wall=n_wall)
    metrics = res.as_metrics()
    # Run an un-traced steady-state wall timer loop so XLA trace-writer overhead
    # on Blackwell (sm_120) does not inflate sub-millisecond kernel timings.
    n_repeats = max(20, n_wall * 5)
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        out = fn(*args)
    jax.block_until_ready(out)
    untraced_ms = ((time.perf_counter() - t0) / n_repeats) * 1000.0
    return float(untraced_ms), metrics


def _make_truncated_solvated_bundle(
    dhfr_bundle,
    n_real_target: int = 6658,
    bucket_atoms: int = 8192,
    bucket_waters: int = 2048,
    seed_shift: float = 0.0,
    charge_perturbation: float = 0.0,
):
    """Construct a valid periodic solvated MolecularBundle with ``n_real_target`` real atoms in ``bucket_atoms``."""
    # DHFR has 2,489 protein atoms (0..2488) followed by 7,023 waters (3 atoms each).
    n_prot = 2489
    n_waters_keep = max(1, (n_real_target - n_prot) // 3)
    n_waters_keep = min(n_waters_keep, int(jnp.sum(dhfr_bundle.water_mask)), bucket_waters)
    n_real = n_prot + 3 * n_waters_keep
    assert n_real <= bucket_atoms

    def _resize_1d(arr, target_len, pad_val):
        cur = arr[:n_real]
        if target_len > n_real:
            return jnp.pad(cur, (0, target_len - n_real), constant_values=pad_val)
        return cur[:target_len]

    def _resize_2d(arr, target_len, pad_val):
        cur = arr[:n_real]
        if target_len > n_real:
            return jnp.pad(cur, ((0, target_len - n_real), (0, 0)), constant_values=pad_val)
        return cur[:target_len]

    pos_real = dhfr_bundle.positions[:n_real] + jnp.asarray(seed_shift, dtype=dhfr_bundle.positions.dtype)
    box_vec = jnp.diag(dhfr_bundle.box)
    pos_real = jnp.mod(pos_real, box_vec[None, :])
    w_real_init = dhfr_bundle.water_indices[:n_waters_keep]
    if n_waters_keep > 0:
        ow_i = w_real_init[:, 0]
        r_ow = pos_real[ow_i]
        for h_col in (1, 2):
            h_i = w_real_init[:, h_col]
            dr = pos_real[h_i] - r_ow
            dr = dr - box_vec[None, :] * jnp.round(dr / box_vec[None, :])
            pos_real = pos_real.at[h_i].set(r_ow + dr)
    # Ghost lattice positions inside box
    n_ghost = bucket_atoms - n_real
    if n_ghost > 0:
        grid_g = int(np.ceil(n_ghost ** (1.0 / 3.0)))
        lin = jnp.linspace(1.0, float(box_vec[0]) - 1.0, grid_g, dtype=pos_real.dtype)
        gx, gy, gz = jnp.meshgrid(lin, lin, lin, indexing="ij")
        ghost_pos = jnp.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], axis=-1)[:n_ghost]
        pos_padded = jnp.concatenate([pos_real, ghost_pos], axis=0)
    else:
        pos_padded = pos_real

    atom_mask = jnp.arange(bucket_atoms) < n_real
    charges = _resize_1d(dhfr_bundle.charges, bucket_atoms, 0.0)
    if charge_perturbation != 0.0:
        # Neutral dipole perturbation on first 20 protein atoms
        delta_q = jnp.zeros_like(charges)
        delta_q = delta_q.at[:10].add(charge_perturbation)
        delta_q = delta_q.at[10:20].add(-charge_perturbation)
        charges = jnp.where(atom_mask, charges + delta_q, 0.0)

    sigmas = _resize_1d(dhfr_bundle.sigmas, bucket_atoms, 1.0)
    epsilons = _resize_1d(dhfr_bundle.epsilons, bucket_atoms, 0.0)
    masses = _resize_1d(dhfr_bundle.masses, bucket_atoms, 1.0)
    radii = _resize_1d(dhfr_bundle.radii, bucket_atoms, 1.5)
    scaled_radii = _resize_1d(dhfr_bundle.scaled_radii, bucket_atoms, 1.2)

    # Filter bonds/angles/dihedrals/impropers/exceptions so all atom indices < n_real
    def _filter_bonded(idx_arr, param_arr, mask_arr):
        valid = jnp.asarray(mask_arr, dtype=bool) & jnp.all(idx_arr < n_real, axis=-1)
        safe_idx = jnp.where(valid[:, None], idx_arr, 0)
        return safe_idx, param_arr, valid

    b_idx, b_par, b_msk = _filter_bonded(
        dhfr_bundle.bond_idx, dhfr_bundle.bond_params, dhfr_bundle.bond_mask
    )
    a_idx, a_par, a_msk = _filter_bonded(
        dhfr_bundle.angle_idx, dhfr_bundle.angle_params, dhfr_bundle.angle_mask
    )
    d_idx, d_par, d_msk = _filter_bonded(
        dhfr_bundle.dihedral_idx, dhfr_bundle.dihedral_params, dhfr_bundle.dihedral_mask
    )
    i_idx, i_par, i_msk = _filter_bonded(
        dhfr_bundle.improper_idx, dhfr_bundle.improper_params, dhfr_bundle.improper_mask
    )
    ex_valid = jnp.asarray(dhfr_bundle.exception_mask, dtype=bool) & jnp.all(
        dhfr_bundle.exception_pairs < n_real, axis=-1
    )
    ex_pairs = jnp.where(ex_valid[:, None], dhfr_bundle.exception_pairs, 0)

    # Water indices & mask in bucket_waters
    w_real = dhfr_bundle.water_indices[:n_waters_keep]
    w_pad_n = bucket_waters - n_waters_keep
    if w_pad_n > 0:
        w_pad = jnp.zeros((w_pad_n, 3), dtype=w_real.dtype)
        w_idx = jnp.concatenate([w_real, w_pad], axis=0)
    else:
        w_idx = w_real[:bucket_waters]
    w_mask = jnp.arange(bucket_waters) < n_waters_keep

    # Exclusions dense table
    ei_raw = _resize_2d(dhfr_bundle.excl_dense_indices, bucket_atoms, -1)
    ei_valid = (ei_raw >= 0) & (ei_raw < n_real) & atom_mask[:, None]
    ei = jnp.where(ei_valid, ei_raw, -1)
    ev = jnp.where(ei_valid, _resize_2d(dhfr_bundle.excl_dense_scales_vdw, bucket_atoms, 1.0), 1.0)
    ee = jnp.where(ei_valid, _resize_2d(dhfr_bundle.excl_dense_scales_elec, bucket_atoms, 1.0), 1.0)

    new_spec = dataclasses.replace(
        dhfr_bundle.shape_spec,
        atom_bucket_idx=bucket_atoms,
        water_bucket_idx=bucket_waters,
        use_size_buckets=False,
        exact_n_atoms=bucket_atoms,
        exact_n_waters=bucket_waters,
    )
    return dataclasses.replace(
        dhfr_bundle,
        positions=pos_padded,
        masses=masses,
        charges=charges,
        sigmas=sigmas,
        epsilons=epsilons,
        radii=radii,
        scaled_radii=scaled_radii,
        atom_mask=atom_mask,
        n_atoms=jnp.asarray(n_real, dtype=dhfr_bundle.n_atoms.dtype),
        bond_idx=b_idx,
        bond_params=b_par,
        bond_mask=b_msk,
        angle_idx=a_idx,
        angle_params=a_par,
        angle_mask=a_msk,
        dihedral_idx=d_idx,
        dihedral_params=d_par,
        dihedral_mask=d_msk,
        improper_idx=i_idx,
        improper_params=i_par,
        improper_mask=i_msk,
        exception_pairs=ex_pairs,
        exception_mask=ex_valid,
        water_indices=w_idx,
        water_mask=w_mask,
        excl_dense_indices=ei,
        excl_dense_scales_vdw=ev,
        excl_dense_scales_elec=ee,
        shape_spec=new_spec,
    )


def _cast_bundle_precision(bundle, dtype):
    """Return a copy of ``bundle`` with all floating-point array leaves cast to ``dtype``."""
    def _cast_leaf(x):
        if isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(dtype)
        return x

    return jax.tree_util.tree_map(_cast_leaf, bundle)


def _make_heterogeneous_dhfr_mutant_screen(dhfr_bundle, n_variants: int) -> list:
    """Build ``n_variants`` heterogeneous DHFR mutant bundles in the 25,000-atom bucket.

    Combines:
    - Deletion/insertion solvation length variants (23,501..23,558 real atoms, distinct
      ``n_atoms``, ``atom_mask``, ``water_mask``, and ``excl_dense_indices`` per variant).
    - Point mutations: dipole-neutral partial charge perturbations, sidechain LJ
      ``sigmas``/``epsilons`` perturbations, and torsion barrier ``dihedral_params``
      perturbations across protein residues.
    """
    variants = []
    for b in range(n_variants):
        # Vary water count by up to 19 waters (57 atoms) so real_atoms spans 23,501..23,558
        waters_removed = (b * 3) % 20 if b > 0 else 0
        n_real_target = 23558 - 3 * waters_removed
        q_pert = 0.0 if b == 0 else 0.015 * ((b % 7) + 1) * (1.0 if (b % 2 == 0) else -1.0)
        seed_shift = 0.0 if b == 0 else 0.002 * ((b % 11) + 1)
        bun = _make_truncated_solvated_bundle(
            dhfr_bundle,
            n_real_target=n_real_target,
            bucket_atoms=25000,
            bucket_waters=8192,
            seed_shift=seed_shift,
            charge_perturbation=q_pert,
        )
        if b > 0:
            # Perturb LJ sigmas & epsilons on a variant-specific 60-atom protein window
            p_start = (b * 73) % 2400
            p_end = p_start + 60
            sig_scale = 1.0 + 0.004 * ((b % 5) - 2)
            eps_scale = 1.0 + 0.025 * ((b % 5) - 2)
            new_sig = bun.sigmas.at[p_start:p_end].multiply(sig_scale)
            new_eps = bun.epsilons.at[p_start:p_end].multiply(eps_scale)
            # Perturb torsion barrier heights on a variant-specific 120-dihedral window
            d_len = int(bun.dihedral_params.shape[0])
            d_start = (b * 131) % max(1, d_len - 120)
            tors_scale = 1.0 + 0.04 * ((b % 4) + 1)
            new_dih_p = bun.dihedral_params.at[d_start : d_start + 120, 2].multiply(tors_scale)
            bun = dataclasses.replace(
                bun,
                sigmas=new_sig,
                epsilons=new_eps,
                dihedral_params=new_dih_p,
            )
        variants.append(bun)
    return variants


def run_dhfr_25k_substep_profile(
    dhfr_bundle,
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16),
    quick: bool = False,
) -> dict:
    """Continuous sub-step ``xtrax.profiling.probe_unit`` decomposition on 23,558-atom DHFR (25k bucket, 64^3 PME).

    Isolates the exact ``ms/call`` and ``ms/step/replica`` of:
    (a) ``init_fn`` / ``fast_reseed_fn`` / ``jnp.isin`` water-mask setup latency,
    (b) ``pallas_baoab_pre_force`` + ``pallas_baoab_post_force`` (MK-46),
    (c) ``pallas_bonded_and_nonbonded_nl_fused`` (MK-48 direct-space + bonded megakernel, homog & hetero),
    (d) ``spme_spread_64`` + ``spme_fft_64`` + ``spme_gather_64`` (MK-49),
    (e) ``cell_bin_argsort`` vs ``pallas_cell_prune`` vs per-replica ``rebuild_mask`` (MK-47),
    (f) Pure 200-step ``lax.while_loop`` propagation vs total ``EnsemblePlan.run(n_steps=200)`` wall time.
    """
    import os

    from prolix.api.ensemble_dispatch import dispatch_n_steps_inference
    from prolix.pallas_kernels import (
        build_tile_pair_neighbor_list,
        pallas_bonded_and_nonbonded_nl_fused,
        pallas_nonbonded_tile_pairs,
    )
    from prolix.physics.pme import influence_function
    from prolix.physics.pme_pallas import (
        compute_pme_spatial_perm,
        gather_pallas,
        spread_charges_pallas,
    )
    from prolix.physics.settle import pallas_baoab_post_force, pallas_baoab_pre_force

    print("=== [0/3] DHFR 23,558-Atom (25k Bucket, 64^3 PME) Sub-Step xtrax.profiling Breakdown ===")
    n_steady = 1 if quick else 2
    n_wall = 2 if quick else 5
    n_steps_loop = 20 if quick else 200
    eff_batches = (1, 2) if quick else batch_sizes

    prev_gp = os.environ.get("PROLIX_PME_GRID_POINTS")
    os.environ["PROLIX_PME_GRID_POINTS"] = "64"
    _SINGLE_INFERENCE_CACHE.clear()
    _STACKED_INFERENCE_CACHE.clear()

    dhfr_wt = _cast_bundle_precision(
        _make_truncated_solvated_bundle(
            dhfr_bundle, n_real_target=23558, bucket_atoms=25000, bucket_waters=8192
        ),
        jnp.float32,
    )
    box_vec = jnp.diag(dhfr_wt.box).astype(jnp.float32)
    cell_size = 9.5
    grid_dims_nl = nl_mod._compute_static_grid_dims(box_vec, cell_size)

    # Pre-allocate base neighbor list for DHFR 25k
    disp_fn, _ = displacement_fn_for_bundle(dhfr_wt)
    nl_fn = nl_mod.make_neighbor_list_fn(
        disp_fn,
        box_vec,
        float(dhfr_wt.cutoff_distance),
        capacity_multiplier=1.25,
        dr_threshold=0.5,
        atom_mask=dhfr_wt.atom_mask,
    )
    nbr_wt = _encode_bundle_nl(nl_fn.allocate(dhfr_wt.positions), dhfr_wt)
    max_occ = int(nbr_wt.idx.shape[-1])
    tp_nl_wt = build_tile_pair_neighbor_list(
        nbr_wt.idx,
        excl_indices=dhfr_wt.excl_dense_indices,
        excl_scales_vdw=dhfr_wt.excl_dense_scales_vdw,
        excl_scales_elec=dhfr_wt.excl_dense_scales_elec,
        atom_mask=dhfr_wt.atom_mask,
        positions=dhfr_wt.positions,
        box=box_vec,
    )
    pme_perm_wt, pme_inv_perm_wt = compute_pme_spatial_perm(
        dhfr_wt.positions, box_vec, (64, 64, 64), atom_mask=dhfr_wt.atom_mask
    )
    G_cached_f32 = influence_function(
        (64, 64, 64), box_vec, float(dhfr_wt.pme_alpha), 4
    ).astype(jnp.float32)

    from prolix.api.ensemble_plan import _NLDispatchCarry

    _SINGLE_INFERENCE_CACHE.clear()
    _STACKED_INFERENCE_CACHE.clear()
    plan_1 = EnsemblePlan.from_bundle(dhfr_wt, use_pallas=True, use_pallas_pme=True)
    state0, apply_fn0, dt_s0, kT_s0 = plan_1._setup_integrator(
        dhfr_wt,
        2.0,
        0.6,
        42,
        integration_prefix=25000,
        dt_unit="fs",
        gamma=10.0,
        initial_neighbor=nbr_wt,
        use_pallas=True,
        use_pallas_pme=True,
        excl_mode="pre_encoded",
    )
    reseed_1 = plan_1._build_fast_reseed_fn(
        dhfr_wt, state0, integration_prefix=25000, gamma=10.0
    )
    _ = plan_1.run(
        n_steps=2, dt=2.0, kT=0.6, seed=10, run_mode="inference", use_neighbor_list=True, nl_update_every=20
    )
    ce1 = next(iter(_SINGLE_INFERENCE_CACHE.values()))
    nl_step_fn_1 = ce1["nl_step_fn"]

    per_batch_records: list[dict] = []
    try:
        for B in eff_batches:
            het_bundles = [
                _cast_bundle_precision(b, jnp.float32)
                for b in _make_heterogeneous_dhfr_mutant_screen(dhfr_bundle, B)
            ]
            hom_bundles = [dhfr_wt for _ in range(B)]

            pos_b = jnp.stack([b.positions for b in het_bundles], axis=0)
            q_b = jnp.stack([b.charges for b in het_bundles], axis=0)
            sig_b = jnp.stack([b.sigmas for b in het_bundles], axis=0)
            eps_b = jnp.stack([b.epsilons for b in het_bundles], axis=0)
            am_b = jnp.stack([b.atom_mask for b in het_bundles], axis=0)
            wm_b = jnp.stack([b.water_mask for b in het_bundles], axis=0)
            wi_b = jnp.stack([b.water_indices for b in het_bundles], axis=0)
            masses_b = jnp.stack([b.masses for b in het_bundles], axis=0)[:, :, None]
            ei_b = jnp.stack([b.excl_dense_indices for b in het_bundles], axis=0)
            ev_b = jnp.stack([b.excl_dense_scales_vdw for b in het_bundles], axis=0)
            ee_b = jnp.stack([b.excl_dense_scales_elec for b in het_bundles], axis=0)
            bx_b = jnp.broadcast_to(box_vec[None, :], (B, 3))
            nbr_idx_b = jnp.broadcast_to(nbr_wt.idx[None, :, :], (B, 25000, max_occ))

            # (a) init_fn / fast_reseed_fn / jnp.isin water mask latency
            seeds_u32 = jnp.arange(B, dtype=jnp.uint32) + jnp.uint32(42)
            jit_reseed_b = jax.jit(
                lambda s_vec: jax.vmap(lambda s: reseed_1(s, kT_s0))(s_vec)
            )
            ms_reseed_b, _ = _probe_ms(jit_reseed_b, seeds_u32, n_steady=n_steady, n_wall=n_wall)

            def _isin_water_mask_only(wi_in, wm_in):
                def _one(wi_1, wm_1):
                    idx_f = wi_1.reshape(-1)
                    mask_f = jnp.broadcast_to(wm_1[:, None], wi_1.shape).reshape(-1)
                    real_idx_f = jnp.where(mask_f, idx_f, -1)
                    return jnp.isin(jnp.arange(25000, dtype=jnp.int32), real_idx_f, assume_unique=False)
                return jax.vmap(_one)(wi_in, wm_in)

            jit_isin_b = jax.jit(_isin_water_mask_only)
            ms_isin_b, _ = _probe_ms(jit_isin_b, wi_b, wm_b, n_steady=n_steady, n_wall=n_wall)

            # (b) pallas_baoab_pre_force + pallas_baoab_post_force (MK-46)
            mom_b = jnp.zeros_like(pos_b)
            f_b = jnp.ones_like(pos_b) * jnp.float32(0.01)
            keys_b = jax.vmap(jax.random.PRNGKey)(seeds_u32)
            dt_f32 = jnp.float32(dt_s0)
            kT_f32 = jnp.float32(kT_s0)
            gamma_akma = 10.0 * 0.04888821

            jit_baoab_pre = jax.jit(
                lambda r, p, f, k: jax.vmap(
                    lambda r1, p1, f1, m1, k1, wi1, wm1, am1: pallas_baoab_pre_force(
                        r1, p1, f1, m1, k1, wi1, wm1, box_vec, dt_f32, kT_f32, gamma_akma, atom_mask=am1
                    )
                )(r, p, f, masses_b, k, wi_b, wm_b, am_b)
            )
            jit_baoab_post = jax.jit(
                lambda r, p, f: jax.vmap(
                    lambda r1, p1, f1, m1, wi1, wm1, am1: pallas_baoab_post_force(
                        r1, p1, f1, m1, wi1, wm1, dt_f32, atom_mask=am1
                    )
                )(r, p, f, masses_b, wi_b, wm_b, am_b)
            )
            ms_baoab_pre, _ = _probe_ms(jit_baoab_pre, pos_b, mom_b, f_b, keys_b, n_steady=n_steady, n_wall=n_wall)
            ms_baoab_post, _ = _probe_ms(jit_baoab_post, pos_b, mom_b, f_b, n_steady=n_steady, n_wall=n_wall)

            # (c) pallas_bonded_and_nonbonded_nl_fused (MK-48, homogeneous vs heterogeneous)
            from prolix.pallas_kernels import _infer_first_pure_water_block

            nbr_mkt_wt = getattr(nbr_wt, "max_k_tiles", None)
            if nbr_mkt_wt is None:
                nbr_mkt_wt = nl_mod._max_k_tiles_from_idx(nbr_wt.idx)
            nbr_mkt_b = jnp.broadcast_to(nbr_mkt_wt[None, :], (B, *nbr_mkt_wt.shape))
            dhfr_pw_blk = _infer_first_pure_water_block(
                int(dhfr_wt.positions.shape[0]),
                (
                    (dhfr_wt.bond_idx, dhfr_wt.bond_mask, 2),
                    (dhfr_wt.angle_idx, dhfr_wt.angle_mask, 3),
                    (dhfr_wt.dihedral_idx, dhfr_wt.dihedral_mask, 4),
                    (dhfr_wt.improper_idx, dhfr_wt.improper_mask, 4),
                    (dhfr_wt.exception_pairs, dhfr_wt.exception_mask, 2),
                ),
            )

            def _mk48_homog_grad(r_in):
                def _one(r1, n1, mkt1):
                    return pallas_bonded_and_nonbonded_nl_fused(
                        r1,
                        dhfr_wt.charges,
                        dhfr_wt.sigmas,
                        dhfr_wt.epsilons,
                        dhfr_wt.atom_mask,
                        n1,
                        dhfr_wt.excl_dense_indices,
                        dhfr_wt.excl_dense_scales_vdw,
                        dhfr_wt.excl_dense_scales_elec,
                        bonds=dhfr_wt.bond_idx,
                        bond_params=dhfr_wt.bond_params,
                        bond_mask=dhfr_wt.bond_mask,
                        angles=dhfr_wt.angle_idx,
                        angle_params=dhfr_wt.angle_params,
                        angle_mask=dhfr_wt.angle_mask,
                        dihedrals=dhfr_wt.dihedral_idx,
                        dihedral_params=dhfr_wt.dihedral_params,
                        dihedral_mask=dhfr_wt.dihedral_mask,
                        impropers=dhfr_wt.improper_idx,
                        improper_params=dhfr_wt.improper_params,
                        improper_mask=dhfr_wt.improper_mask,
                        exception_pairs=dhfr_wt.exception_pairs,
                        exception_sigmas=dhfr_wt.exception_sigmas,
                        exception_epsilons=dhfr_wt.exception_epsilons,
                        exception_chargeprods=dhfr_wt.exception_chargeprods,
                        exception_mask=dhfr_wt.exception_mask,
                        box=box_vec,
                        pme_alpha=float(dhfr_wt.pme_alpha),
                        cutoff=float(dhfr_wt.cutoff_distance),
                        excl_mode="pre_encoded",
                        max_k_tiles=mkt1,
                        first_pure_water_block=dhfr_pw_blk,
                    )
                return jax.vmap(jax.grad(_one))(r_in, nbr_idx_b, nbr_mkt_b)

            dih_p_b = jnp.stack([b.dihedral_params for b in het_bundles], axis=0)

            def _mk48_hetero_grad(r_in, q_in, sig_in, eps_in, am_in, dp_in):
                def _one(r1, q1, s1, e1, m1, dp1, n1, mkt1):
                    return pallas_bonded_and_nonbonded_nl_fused(
                        r1,
                        q1,
                        s1,
                        e1,
                        m1,
                        n1,
                        dhfr_wt.excl_dense_indices,
                        dhfr_wt.excl_dense_scales_vdw,
                        dhfr_wt.excl_dense_scales_elec,
                        bonds=dhfr_wt.bond_idx,
                        bond_params=dhfr_wt.bond_params,
                        bond_mask=dhfr_wt.bond_mask,
                        angles=dhfr_wt.angle_idx,
                        angle_params=dhfr_wt.angle_params,
                        angle_mask=dhfr_wt.angle_mask,
                        dihedrals=dhfr_wt.dihedral_idx,
                        dihedral_params=dp1,
                        dihedral_mask=dhfr_wt.dihedral_mask,
                        impropers=dhfr_wt.improper_idx,
                        improper_params=dhfr_wt.improper_params,
                        improper_mask=dhfr_wt.improper_mask,
                        exception_pairs=dhfr_wt.exception_pairs,
                        exception_sigmas=dhfr_wt.exception_sigmas,
                        exception_epsilons=dhfr_wt.exception_epsilons,
                        exception_chargeprods=dhfr_wt.exception_chargeprods,
                        exception_mask=dhfr_wt.exception_mask,
                        box=box_vec,
                        pme_alpha=float(dhfr_wt.pme_alpha),
                        cutoff=float(dhfr_wt.cutoff_distance),
                        excl_mode="pre_encoded",
                        max_k_tiles=mkt1,
                        first_pure_water_block=dhfr_pw_blk,
                    )
                return jax.vmap(jax.grad(_one))(
                    r_in, q_in, sig_in, eps_in, am_in, dih_p_b, nbr_idx_b, nbr_mkt_b
                )

            jit_mk48_hom = jax.jit(_mk48_homog_grad)
            jit_mk48_het = jax.jit(_mk48_hetero_grad)
            ms_mk48_hom, _ = _probe_ms(jit_mk48_hom, pos_b, n_steady=n_steady, n_wall=n_wall)
            ms_mk48_het, _ = _probe_ms(
                jit_mk48_het, pos_b, q_b, sig_b, eps_b, am_b, dih_p_b, n_steady=n_steady, n_wall=n_wall
            )

            # MK-51: 32x32 Tile-Pair Newton 3rd Law nonbonded kernel probe
            jit_mk51_tp = jax.jit(
                lambda r_in: jax.vmap(
                    lambda r1: pallas_nonbonded_tile_pairs(
                        r1,
                        dhfr_wt.charges,
                        dhfr_wt.sigmas,
                        dhfr_wt.epsilons,
                        tp_nl_wt,
                        atom_mask=dhfr_wt.atom_mask,
                        box=box_vec,
                        pme_alpha=float(dhfr_wt.pme_alpha),
                        cutoff=float(dhfr_wt.cutoff_distance),
                        return_forces=True,
                    )[1]
                )(r_in)
            )
            ms_mk51_tp, _ = _probe_ms(jit_mk51_tp, pos_b, n_steady=n_steady, n_wall=n_wall)

            # (d) SPME 64^3 breakdown: spread, rfftn+irfftn, gather
            jit_spme_spread = jax.jit(
                lambda r_in, q_in, m_in: jax.vmap(
                    lambda r1, q1, m1: spread_charges_pallas(r1, q1, m1, box_vec, (64, 64, 64))
                )(r_in, q_in, m_in)
            )
            q_grid_b = jit_spme_spread(pos_b, q_b, am_b)
            jax.block_until_ready(q_grid_b)

            def _spme_fft_only(q_mesh):
                q_hat = jnp.fft.rfftn(q_mesh, s=(64, 64, 64), axes=(-3, -2, -1))
                theta = jnp.fft.irfftn(q_hat * G_cached_f32, s=(64, 64, 64), axes=(-3, -2, -1))
                return theta * jnp.float32(64 * 64 * 64)

            jit_spme_fft = jax.jit(_spme_fft_only)
            theta_b = jit_spme_fft(q_grid_b)
            jax.block_until_ready(theta_b)

            jit_spme_gather = jax.jit(
                lambda r_in, q_in, m_in, th_in: jax.vmap(
                    lambda r1, q1, m1, th1: gather_pallas(
                        r1, q1, m1, box_vec, th1, (64, 64, 64), compute_dq=False
                    )[0]
                )(r_in, q_in, m_in, th_in)
            )
            ms_spme_spread, _ = _probe_ms(jit_spme_spread, pos_b, q_b, am_b, n_steady=n_steady, n_wall=n_wall)
            ms_spme_fft, _ = _probe_ms(jit_spme_fft, q_grid_b, n_steady=n_steady, n_wall=n_wall)
            ms_spme_gather, _ = _probe_ms(
                jit_spme_gather, pos_b, q_b, am_b, theta_b, n_steady=n_steady, n_wall=n_wall
            )
            jit_spme_spread_sorted = jax.jit(
                lambda r_in, q_in, m_in: spread_charges_pallas(
                    r_in, q_in, m_in, box_vec, (64, 64, 64), atom_perm=pme_perm_wt
                )
            )
            jit_spme_gather_sorted = jax.jit(
                lambda r_in, q_in, m_in, th_in: gather_pallas(
                    r_in,
                    q_in,
                    m_in,
                    box_vec,
                    th_in,
                    (64, 64, 64),
                    atom_perm=pme_perm_wt,
                    atom_inv_perm=pme_inv_perm_wt,
                    compute_dq=False,
                )[0]
            )
            ms_spme_spread_sorted, _ = _probe_ms(
                jit_spme_spread_sorted, pos_b, q_b, am_b, n_steady=n_steady, n_wall=n_wall
            )
            ms_spme_gather_sorted, _ = _probe_ms(
                jit_spme_gather_sorted, pos_b, q_b, am_b, theta_b, n_steady=n_steady, n_wall=n_wall
            )

            # (e) Cell-list binning (argsort vs MK-52 Pallas atomic) vs Pallas cell-prune vs single-replica rebuild_mask
            reb_all = jnp.ones((B,), dtype=jnp.bool_)
            reb_one = jnp.arange(B) == 0
            jit_cell_bin_argsort = jax.jit(
                lambda r_in, m_in, rb_in: nl_mod._build_cell_stencil_buffers_batched(
                    r_in,
                    bx_b,
                    cell_size,
                    64,
                    grid_dims_nl,
                    atom_mask_b=m_in,
                    rebuild_mask_b=rb_in,
                    interpret=True,
                )
            )
            jit_cell_bin_mk52 = jax.jit(
                lambda r_in, m_in, rb_in: nl_mod._build_cell_stencil_buffers_batched(
                    r_in,
                    bx_b,
                    cell_size,
                    64,
                    grid_dims_nl,
                    atom_mask_b=m_in,
                    rebuild_mask_b=rb_in,
                    interpret=(jax.default_backend() == "cpu"),
                )
            )
            precomp_excl_b = nl_mod.precompute_static_nl_exclusion_tables(
                ei_b, ev_b, ee_b, 25000
            )
            jit_nl_full_all = jax.jit(
                lambda r_in, m_in, rb_in: nl_mod._pallas_prune_and_encode_cell_list_batch(
                    B,
                    (True, True, True, True, True),
                    jnp.mod(r_in, bx_b[:, None, :]),
                    bx_b,
                    ei_b,
                    ev_b,
                    ee_b,
                    cell_size=cell_size,
                    cell_capacity=64,
                    max_occupancy=max_occ,
                    r_cutoff=cell_size,
                    encode_exclusions=True,
                    interpret=(jax.default_backend() == "cpu"),
                    grid_dims=grid_dims_nl,
                    atom_mask=m_in,
                    mask_batched=True,
                    rebuild_mask=rb_in,
                    old_idx=nbr_idx_b,
                    precomputed_excl=precomp_excl_b,
                    return_max_k_tiles=True,
                    old_max_k_tiles=nbr_mkt_b,
                )
            )
            ms_cell_bin_all, _ = _probe_ms(jit_cell_bin_argsort, pos_b, am_b, reb_all, n_steady=n_steady, n_wall=n_wall)
            ms_cell_bin_one, _ = _probe_ms(jit_cell_bin_argsort, pos_b, am_b, reb_one, n_steady=n_steady, n_wall=n_wall)
            ms_cell_bin_mk52_all, _ = _probe_ms(jit_cell_bin_mk52, pos_b, am_b, reb_all, n_steady=n_steady, n_wall=n_wall)
            ms_cell_bin_mk52_one, _ = _probe_ms(jit_cell_bin_mk52, pos_b, am_b, reb_one, n_steady=n_steady, n_wall=n_wall)
            ms_nl_all, _ = _probe_ms(jit_nl_full_all, pos_b, am_b, reb_all, n_steady=n_steady, n_wall=n_wall)
            ms_nl_one, _ = _probe_ms(jit_nl_full_all, pos_b, am_b, reb_one, n_steady=n_steady, n_wall=n_wall)

            # (f) Pure while_loop propagation vs total EnsemblePlan.run wall time (homog & hetero)
            _SINGLE_INFERENCE_CACHE.clear()
            _STACKED_INFERENCE_CACHE.clear()
            # Build 1-replica _nl_step_fn and pre-initialized carry to time pure while_loop without init_fn/reseed
            _ = plan_1.run(
                n_steps=2, dt=2.0, kT=0.6, seed=10, run_mode="inference", use_neighbor_list=True, nl_update_every=20
            )
            ce1 = next(iter(_SINGLE_INFERENCE_CACHE.values()))
            nl_step_fn_1 = ce1["nl_step_fn"]
            from prolix.api.ensemble_plan import _NLDispatchCarry

            state_b_init = jit_reseed_b(seeds_u32)
            nbr_b_init = jax.tree.map(
                lambda x: jnp.broadcast_to(x[None, ...], (B,) + x.shape) if isinstance(x, jnp.ndarray) else x,
                ce1["nbr0"],
            )
            carry_b_init = _NLDispatchCarry(
                state_b_init,
                nbr_b_init,
                jnp.zeros((B,), dtype=jnp.bool_),
                state_b_init.position,
            )
            n_steps_i32 = jnp.asarray(n_steps_loop, dtype=jnp.int32)
            jit_pure_while_b = jax.jit(
                lambda c_in, n_in: (
                    dispatch_n_steps_inference(nl_step_fn_1, jax.tree.map(lambda x: x[0], c_in), n_in)
                    if B == 1
                    else jax.vmap(lambda c1: dispatch_n_steps_inference(nl_step_fn_1, c1, n_in))(c_in)
                )
            )
            w_pure = jit_pure_while_b(carry_b_init, jnp.asarray(2, dtype=jnp.int32))
            jax.block_until_ready(w_pure.langevin_state.position)
            t0_pure = time.perf_counter()
            out_pure = jit_pure_while_b(carry_b_init, n_steps_i32)
            jax.block_until_ready(out_pure.langevin_state.position)
            pure_while_wall_ms = (time.perf_counter() - t0_pure) * 1000.0

            _SINGLE_INFERENCE_CACHE.clear()
            _STACKED_INFERENCE_CACHE.clear()
            plan_hom = EnsemblePlan.from_bundles(hom_bundles, use_pallas=True, use_pallas_pme=True)
            out_w_hom = plan_hom.run(
                n_steps=5, dt=2.0, kT=0.6, seed=10, run_mode="inference", use_neighbor_list=True, nl_update_every=20
            )
            if isinstance(out_w_hom, list):
                for t in out_w_hom:
                    jax.block_until_ready(t.positions)
            else:
                jax.block_until_ready(out_w_hom.positions)

            t0 = time.perf_counter()
            out_p_hom = plan_hom.run(
                n_steps=n_steps_loop,
                dt=2.0,
                kT=0.6,
                seed=42,
                run_mode="inference",
                use_neighbor_list=True,
                nl_update_every=20,
            )
            if isinstance(out_p_hom, list):
                for t in out_p_hom:
                    jax.block_until_ready(t.positions)
            else:
                jax.block_until_ready(out_p_hom.positions)
            total_hom_wall_ms = (time.perf_counter() - t0) * 1000.0

            _SINGLE_INFERENCE_CACHE.clear()
            _STACKED_INFERENCE_CACHE.clear()
            plan_het = EnsemblePlan.from_bundles(het_bundles, use_pallas=True, use_pallas_pme=True)
            out_w_het = plan_het.run(
                n_steps=5, dt=2.0, kT=0.6, seed=10, run_mode="inference", use_neighbor_list=True, nl_update_every=20
            )
            if isinstance(out_w_het, list):
                for t in out_w_het:
                    jax.block_until_ready(t.positions)
            else:
                jax.block_until_ready(out_w_het.positions)

            t0 = time.perf_counter()
            out_p_het = plan_het.run(
                n_steps=n_steps_loop,
                dt=2.0,
                kT=0.6,
                seed=42,
                run_mode="inference",
                use_neighbor_list=True,
                nl_update_every=20,
            )
            if isinstance(out_p_het, list):
                for t in out_p_het:
                    jax.block_until_ready(t.positions)
            else:
                jax.block_until_ready(out_p_het.positions)
            total_het_wall_ms = (time.perf_counter() - t0) * 1000.0

            est_step_sum_hom_ms = (
                ms_baoab_pre
                + ms_baoab_post
                + ms_mk48_hom
                + ms_spme_spread
                + ms_spme_fft
                + ms_spme_gather
                + (ms_nl_all / 20.0)
            )
            est_step_sum_het_ms = (
                ms_baoab_pre
                + ms_baoab_post
                + ms_mk48_het
                + ms_spme_spread
                + ms_spme_fft
                + ms_spme_gather
                + (ms_nl_all / 20.0)
            )

            rec = {
                "replicas_B": B,
                "n_steps_timed": n_steps_loop,
                "a_reseed_init_ms_call": ms_reseed_b,
                "a_reseed_init_ms_per_rep": ms_reseed_b / B,
                "a_isin_water_mask_ms_call": ms_isin_b,
                "b_baoab_pre_force_ms_call": ms_baoab_pre,
                "b_baoab_post_force_ms_call": ms_baoab_post,
                "b_baoab_total_ms_per_rep": (ms_baoab_pre + ms_baoab_post) / B,
                "c_megakernel_homog_ms_call": ms_mk48_hom,
                "c_megakernel_homog_ms_per_rep": ms_mk48_hom / B,
                "c_megakernel_hetero_ms_call": ms_mk48_het,
                "c_megakernel_hetero_ms_per_rep": ms_mk48_het / B,
                "c_tile_pairs_mk51_homog_ms_call": ms_mk51_tp,
                "c_tile_pairs_mk51_homog_ms_per_rep": ms_mk51_tp / B,
                "d_spme_spread_64_ms_call": ms_spme_spread,
                "d_spme_spread_sorted_mk53_ms_call": ms_spme_spread_sorted,
                "d_spme_fft_64_ms_call": ms_spme_fft,
                "d_spme_gather_64_ms_call": ms_spme_gather,
                "d_spme_gather_sorted_mk53_ms_call": ms_spme_gather_sorted,
                "d_spme_total_64_ms_call": ms_spme_spread + ms_spme_fft + ms_spme_gather,
                "d_spme_total_64_ms_per_rep": (ms_spme_spread + ms_spme_fft + ms_spme_gather) / B,
                "e_cell_bin_argsort_all_ms_call": ms_cell_bin_all,
                "e_cell_bin_argsort_1rep_ms_call": ms_cell_bin_one,
                "e_cell_bin_pallas_mk52_all_ms_call": ms_cell_bin_mk52_all,
                "e_cell_bin_pallas_mk52_1rep_ms_call": ms_cell_bin_mk52_one,
                "e_nl_rebuild_all_ms_call": ms_nl_all,
                "e_nl_rebuild_1rep_ms_call": ms_nl_one,
                "e_pallas_prune_only_all_ms_call": max(0.0, ms_nl_all - ms_cell_bin_mk52_all),
                "e_nl_amortized_20steps_ms_per_rep": (ms_nl_all / 20.0) / B,
                "f_sum_substeps_homog_ms_step_batch": est_step_sum_hom_ms,
                "f_sum_substeps_homog_ms_step_per_rep": est_step_sum_hom_ms / B,
                "f_sum_substeps_hetero_ms_step_batch": est_step_sum_het_ms,
                "f_sum_substeps_hetero_ms_step_per_rep": est_step_sum_het_ms / B,
                "f_pure_while_loop_homog_ms_step_batch": pure_while_wall_ms / n_steps_loop,
                "f_pure_while_loop_homog_ms_step_per_rep": (pure_while_wall_ms / n_steps_loop) / B,
                "f_plan_run_homog_ms_step_batch": total_hom_wall_ms / n_steps_loop,
                "f_plan_run_homog_ms_step_per_rep": (total_hom_wall_ms / n_steps_loop) / B,
                "f_plan_run_hetero_ms_step_batch": total_het_wall_ms / n_steps_loop,
                "f_plan_run_hetero_ms_step_per_rep": (total_het_wall_ms / n_steps_loop) / B,
                "f_non_step_overhead_vs_pure_while_homog_ms_total": total_hom_wall_ms - pure_while_wall_ms,
                "f_non_step_overhead_homog_ms_total": total_hom_wall_ms - est_step_sum_hom_ms * n_steps_loop,
                "f_non_step_overhead_hetero_ms_total": total_het_wall_ms - est_step_sum_het_ms * n_steps_loop,
            }
            per_batch_records.append(rec)
            print(
                f"  [Substep B={B:2d}] reseed={ms_reseed_b:6.2f}ms (isin={ms_isin_b:5.2f}ms) | "
                f"BAOAB={(ms_baoab_pre + ms_baoab_post)/B:5.3f}ms/rep | "
                f"MK48(hom/het)={ms_mk48_hom/B:5.3f}/{ms_mk48_het/B:5.3f}ms/rep | "
                f"SPME={(ms_spme_spread + ms_spme_fft + ms_spme_gather)/B:5.3f}ms/rep "
                f"(spr={ms_spme_spread/B:5.3f}, fft={ms_spme_fft/B:5.3f}, gat={ms_spme_gather/B:5.3f}) | "
                f"NL_reb(all/1rep)={ms_nl_all:5.2f}/{ms_nl_one:5.2f}ms (bin_arg={ms_cell_bin_all:5.2f}, bin_mk52={ms_cell_bin_mk52_all:5.2f}/{ms_cell_bin_mk52_one:5.2f}ms) | "
                f"while_loop={(pure_while_wall_ms/n_steps_loop)/B:5.3f}ms/step/rep | "
                f"plan.run(hom/het)={(total_hom_wall_ms/n_steps_loop)/B:5.3f}/{(total_het_wall_ms/n_steps_loop)/B:5.3f}ms/step/rep"
            )
    finally:
        if prev_gp is None:
            os.environ.pop("PROLIX_PME_GRID_POINTS", None)
        else:
            os.environ["PROLIX_PME_GRID_POINTS"] = prev_gp

    return {"dhfr_25k_substep_breakdown": per_batch_records}



def run_ragged_bucket_and_accuracy_audit(dhfr_bundle, quick: bool = False) -> dict:
    print("=== [1/3] Ragged-Bucket Ghost-Atom Filtering Speedup & Numerical Accuracy Audit ===")
    ubq_8k = _make_truncated_solvated_bundle(
        dhfr_bundle, n_real_target=6658, bucket_atoms=8192, bucket_waters=2048
    )
    ubq_25k = _make_truncated_solvated_bundle(
        dhfr_bundle, n_real_target=6658, bucket_atoms=25000, bucket_waters=8192
    )

    disp_8k, _ = displacement_fn_for_bundle(ubq_8k)
    box_vec = jnp.diag(ubq_8k.box)
    nl_fn_8k = nl_mod.make_neighbor_list_fn(
        disp_8k,
        box_vec,
        9.0,
        capacity_multiplier=1.25,
        dr_threshold=0.5,
        atom_mask=ubq_8k.atom_mask,
    )
    nbr_8k = _encode_bundle_nl(nl_fn_8k.allocate(ubq_8k.positions), ubq_8k)

    # Compare v2.3 (unfiltered ghost atoms in 25k bucket) vs v2.4 (ghost-filtered & compacted in 25k bucket)
    disp_25k, _ = displacement_fn_for_bundle(ubq_25k)
    nl_fn_25k_v23 = nl_mod.make_neighbor_list_fn(
        disp_25k,
        box_vec,
        9.0,
        capacity_multiplier=1.25,
        dr_threshold=0.5,
        atom_mask=None,
    )
    nbr_25k_v23_raw = nl_fn_25k_v23.allocate(ubq_25k.positions)
    nbr_25k_v23 = dataclasses.replace(
        nbr_25k_v23_raw,
        idx=nl_mod.encode_nl_exclusions(
            nbr_25k_v23_raw.idx,
            ubq_25k.excl_dense_indices,
            ubq_25k.excl_dense_scales_vdw,
            ubq_25k.excl_dense_scales_elec,
            n_atoms=25000,
            atom_mask=None,
        ),
    )

    nl_fn_25k_v24 = nl_mod.make_neighbor_list_fn(
        disp_25k,
        box_vec,
        9.0,
        capacity_multiplier=1.25,
        dr_threshold=0.5,
        atom_mask=ubq_25k.atom_mask,
    )
    nbr_25k_v24 = _encode_bundle_nl(nl_fn_25k_v24.allocate(ubq_25k.positions), ubq_25k)

    # Verify zero ghost indices in v2.4 real rows
    n_real = 6658
    idx_v24_np = np.asarray(nbr_25k_v24.idx[:n_real])
    base_v24_np = np.where(idx_v24_np == 25000, 25000, idx_v24_np & nl_mod._NB_INDEX_MASK)
    ghost_count_v24 = int(np.sum((base_v24_np >= n_real) & (base_v24_np < 25000)))
    idx_v23_np = np.asarray(nbr_25k_v23.idx[:n_real])
    base_v23_np = np.where(idx_v23_np == 25000, 25000, idx_v23_np & nl_mod._NB_INDEX_MASK)
    ghost_count_v23 = int(np.sum((base_v23_np >= n_real) & (base_v23_np < 25000)))
    assert ghost_count_v24 == 0, f"Expected 0 ghost neighbors in v2.4, got {ghost_count_v24}"

    # Energy & force parity: 8,192 bucket vs 25,000 bucket vs XLA reference
    e_fn_8k_pal = energy_fn_from_bundle(
        ubq_8k, build_dense_exclusions=False, use_pallas=True, use_pallas_pme=True, excl_mode="pre_encoded"
    )
    e_fn_8k_xla = energy_fn_from_bundle(
        ubq_8k, build_dense_exclusions=False, use_pallas=False, use_pallas_pme=False, excl_mode="table"
    )
    e_fn_25k_pal = energy_fn_from_bundle(
        ubq_25k, build_dense_exclusions=False, use_pallas=True, use_pallas_pme=True, excl_mode="pre_encoded"
    )

    vg_8k_pal = jax.jit(jax.value_and_grad(lambda r, n: e_fn_8k_pal(r, neighbor=n)))
    vg_8k_xla = jax.jit(jax.value_and_grad(lambda r, n: e_fn_8k_xla(r, neighbor=n)))
    vg_25k_pal = jax.jit(jax.value_and_grad(lambda r, n: e_fn_25k_pal(r, neighbor=n)))

    nbr_8k_raw = nl_fn_8k.allocate(ubq_8k.positions)
    e_pal_8k, g_pal_8k = vg_8k_pal(ubq_8k.positions, nbr_8k)
    e_xla_8k, g_xla_8k = vg_8k_xla(ubq_8k.positions, nbr_8k_raw)
    e_pal_25k, g_pal_25k = vg_25k_pal(ubq_25k.positions, nbr_25k_v24)

    energy_diff_pal_vs_xla = float(jnp.abs(e_pal_8k - e_xla_8k))
    force_max_diff_pal_vs_xla = float(jnp.max(jnp.abs(g_pal_8k[:n_real] - g_xla_8k[:n_real])))
    energy_diff_8k_vs_25k = float(jnp.abs(e_pal_8k - e_pal_25k))
    force_max_diff_8k_vs_25k = float(jnp.max(jnp.abs(g_pal_8k[:n_real] - g_pal_25k[:n_real])))

    n_steady = 1 if quick else 3
    n_wall = 2 if quick else 6

    ms_25k_v23, _ = _probe_ms(vg_25k_pal, ubq_25k.positions, nbr_25k_v23, n_steady=n_steady, n_wall=n_wall)
    ms_25k_v24, _ = _probe_ms(vg_25k_pal, ubq_25k.positions, nbr_25k_v24, n_steady=n_steady, n_wall=n_wall)
    ms_8k_v24, _ = _probe_ms(vg_8k_pal, ubq_8k.positions, nbr_8k, n_steady=n_steady, n_wall=n_wall)

    # Also profile cell-list prune kernel on ragged 25k bucket: v2.3 (atom_mask=None) vs v2.4 (atom_mask)
    cell_size = 9.5
    grid_dims = nl_mod._compute_static_grid_dims(box_vec, cell_size)
    max_occ_25k = int(nbr_25k_v24.idx.shape[-1])
    jit_prune_v23 = jax.jit(
        lambda p: nl_mod.pallas_prune_and_encode_cell_list(
            jnp.mod(p, box_vec),
            box_vec,
            ubq_25k.excl_dense_indices,
            ubq_25k.excl_dense_scales_vdw,
            ubq_25k.excl_dense_scales_elec,
            cell_size=cell_size,
            cell_capacity=64,
            max_occupancy=max_occ_25k,
            r_cutoff=9.5,
            encode_exclusions=True,
            interpret=(jax.default_backend() == "cpu"),
            grid_dims=grid_dims,
            atom_mask=None,
        )
    )
    jit_prune_v24 = jax.jit(
        lambda p: nl_mod.pallas_prune_and_encode_cell_list(
            jnp.mod(p, box_vec),
            box_vec,
            ubq_25k.excl_dense_indices,
            ubq_25k.excl_dense_scales_vdw,
            ubq_25k.excl_dense_scales_elec,
            cell_size=cell_size,
            cell_capacity=64,
            max_occupancy=max_occ_25k,
            r_cutoff=9.5,
            encode_exclusions=True,
            interpret=(jax.default_backend() == "cpu"),
            grid_dims=grid_dims,
            atom_mask=ubq_25k.atom_mask,
        )
    )
    ms_prune_v23, _ = _probe_ms(jit_prune_v23, ubq_25k.positions, n_steady=n_steady, n_wall=n_wall)
    ms_prune_v24, _ = _probe_ms(jit_prune_v24, ubq_25k.positions, n_steady=n_steady, n_wall=n_wall)

    # Also profile MK-36 (fused direct+bonded megakernel vs separate launches) and MK-38 (SPME compute_dq=False vs True)
    from prolix.pallas_kernels import (
        pallas_bonded_all_fused,
        pallas_bonded_and_nonbonded_nl_fused,
    )
    from prolix.physics.pme_pallas import gather_pallas

    jit_fused_mk36 = jax.jit(
        jax.value_and_grad(
            lambda p: pallas_bonded_and_nonbonded_nl_fused(
                p,
                ubq_8k.charges,
                ubq_8k.sigmas,
                ubq_8k.epsilons,
                ubq_8k.atom_mask,
                nbr_8k.idx,
                ubq_8k.excl_dense_indices,
                ubq_8k.excl_dense_scales_vdw,
                ubq_8k.excl_dense_scales_elec,
                bonds=ubq_8k.bond_idx,
                bond_params=ubq_8k.bond_params,
                bond_mask=ubq_8k.bond_mask,
                angles=ubq_8k.angle_idx,
                angle_params=ubq_8k.angle_params,
                angle_mask=ubq_8k.angle_mask,
                dihedrals=ubq_8k.dihedral_idx,
                dihedral_params=ubq_8k.dihedral_params,
                dihedral_mask=ubq_8k.dihedral_mask,
                impropers=ubq_8k.improper_idx,
                improper_params=ubq_8k.improper_params,
                improper_mask=ubq_8k.improper_mask,
                exception_pairs=ubq_8k.exception_pairs,
                exception_sigmas=ubq_8k.exception_sigmas,
                exception_epsilons=ubq_8k.exception_epsilons,
                exception_chargeprods=ubq_8k.exception_chargeprods,
                exception_mask=ubq_8k.exception_mask,
                box=box_vec,
                pme_alpha=float(ubq_8k.pme_alpha),
                cutoff=float(ubq_8k.cutoff_distance),
                excl_mode="pre_encoded",
            )
        )
    )
    jit_sep_v24 = jax.jit(
        jax.value_and_grad(
            lambda p: (
                pallas_bonded_all_fused(
                    p,
                    bonds=ubq_8k.bond_idx,
                    bond_params=ubq_8k.bond_params,
                    bond_mask=ubq_8k.bond_mask,
                    angles=ubq_8k.angle_idx,
                    angle_params=ubq_8k.angle_params,
                    angle_mask=ubq_8k.angle_mask,
                    dihedrals=ubq_8k.dihedral_idx,
                    dihedral_params=ubq_8k.dihedral_params,
                    dihedral_mask=ubq_8k.dihedral_mask,
                    impropers=ubq_8k.improper_idx,
                    improper_params=ubq_8k.improper_params,
                    improper_mask=ubq_8k.improper_mask,
                    exception_pairs=ubq_8k.exception_pairs,
                    exception_sigmas=ubq_8k.exception_sigmas,
                    exception_epsilons=ubq_8k.exception_epsilons,
                    exception_chargeprods=ubq_8k.exception_chargeprods,
                    exception_mask=ubq_8k.exception_mask,
                    box=box_vec,
                )
                + pallas_nonbonded_nl(
                    p,
                    ubq_8k.charges,
                    ubq_8k.sigmas,
                    ubq_8k.epsilons,
                    ubq_8k.atom_mask,
                    nbr_8k.idx,
                    ubq_8k.excl_dense_indices,
                    ubq_8k.excl_dense_scales_vdw,
                    ubq_8k.excl_dense_scales_elec,
                    box=box_vec,
                    pme_alpha=float(ubq_8k.pme_alpha),
                    cutoff=float(ubq_8k.cutoff_distance),
                    excl_mode="pre_encoded",
                )
            )
        )
    )
    ms_sep_8k, _ = _probe_ms(jit_sep_v24, ubq_8k.positions, n_steady=n_steady, n_wall=n_wall)
    ms_fused_8k, _ = _probe_ms(jit_fused_mk36, ubq_8k.positions, n_steady=n_steady, n_wall=n_wall)

    theta_dummy = jnp.ones((64, 64, 64), dtype=jnp.float32)
    jit_gather_dq_true = jax.jit(
        lambda p: gather_pallas(
            p, ubq_8k.charges, ubq_8k.atom_mask, box_vec, theta_dummy, (64, 64, 64), compute_dq=True
        )[0]
    )
    jit_gather_dq_false = jax.jit(
        lambda p: gather_pallas(
            p, ubq_8k.charges, ubq_8k.atom_mask, box_vec, theta_dummy, (64, 64, 64), compute_dq=False
        )[0]
    )
    ms_gather_dq_true, _ = _probe_ms(jit_gather_dq_true, ubq_8k.positions, n_steady=n_steady, n_wall=n_wall)
    ms_gather_dq_false, _ = _probe_ms(jit_gather_dq_false, ubq_8k.positions, n_steady=n_steady, n_wall=n_wall)

    # Profile MK-46 (Fused Pallas BAOAB + analytical Miyamoto-Kollman SETTLE vs pure-JAX BAOAB + SETTLE)
    from prolix.physics.settle import (
        _langevin_settle_vel,
        pallas_baoab_post_force,
        pallas_baoab_pre_force,
        settle_positions,
    )

    w_valid = jnp.asarray(ubq_8k.water_mask, dtype=jnp.bool_)
    w_idx_8k = jnp.asarray(ubq_8k.water_indices, dtype=jnp.int32)
    masses_8k = jnp.where(ubq_8k.atom_mask, ubq_8k.masses, 1.0).astype(jnp.float32)[:, None]
    inv_mass_8k = jnp.where(ubq_8k.atom_mask[:, None], 1.0 / masses_8k, 0.0).astype(jnp.float32)
    dt_val = jnp.float32(0.0409)
    kT_val = jnp.float32(0.6)
    gamma_val = 0.48888
    c1_val = jnp.float32(0.9918)
    c2_8k = jnp.sqrt((1.0 - c1_val * c1_val) * kT_val * masses_8k).astype(jnp.float32)
    r0_8k = ubq_8k.positions.astype(jnp.float32)
    p0_8k = jnp.zeros_like(r0_8k)
    f0_8k = g_pal_8k.astype(jnp.float32)
    noise_8k = jnp.ones_like(r0_8k) * jnp.float32(0.01)
    key_8k = jax.random.PRNGKey(123)
    box_f32 = box_vec.astype(jnp.float32)

    def _pallas_baoab_step(r, p, f, k):
        r1, p1, _ = pallas_baoab_pre_force(
            r, p, f, masses_8k, k, w_idx_8k, w_valid, box_f32, dt_val, kT_val, gamma_val, atom_mask=ubq_8k.atom_mask
        )
        return pallas_baoab_post_force(
            r1, p1, f, masses_8k, w_idx_8k, w_valid, dt_val, atom_mask=ubq_8k.atom_mask
        )

    jit_mk46_pallas = jax.jit(_pallas_baoab_step)

    def _xla_baoab_settle(r, p, f, n):
        m3 = ubq_8k.atom_mask[:, None]
        p1 = jnp.where(m3, p + 0.5 * dt_val * f, 0.0)
        r_half = jnp.where(m3, r + 0.5 * dt_val * p1 * inv_mass_8k, r)
        r_half_c = settle_positions(r, r_half, w_idx_8k, box=box_f32, water_mask=w_valid)
        v_half = (r_half_c - r) / (0.5 * dt_val)
        p_half = jnp.where(m3, v_half * masses_8k, 0.0)
        p_o = jnp.where(m3, c1_val * p_half + c2_8k * n, 0.0)
        p_o_c = _langevin_settle_vel(
            p_o, r_half_c, r_half_c, masses_8k, w_idx_8k, dt_val, 15.999, 1.008, water_mask=w_valid, atom_mask=ubq_8k.atom_mask
        )
        r_un = jnp.where(m3, r_half_c + 0.5 * dt_val * p_o_c * inv_mass_8k, r_half_c)
        r_new = settle_positions(r_half_c, r_un, w_idx_8k, box=box_f32, water_mask=w_valid)
        v_a2 = (r_new - r_half_c) / (0.5 * dt_val)
        p_a2 = jnp.where(m3, v_a2 * masses_8k, 0.0)
        p_b2 = jnp.where(m3, p_a2 + 0.5 * dt_val * f, 0.0)
        p_f = _langevin_settle_vel(
            p_b2, r_new, r_new, masses_8k, w_idx_8k, dt_val, 15.999, 1.008, water_mask=w_valid, atom_mask=ubq_8k.atom_mask
        )
        return jnp.where(m3, p_f, 0.0)

    jit_mk46_xla = jax.jit(_xla_baoab_settle)
    ms_mk46_pallas, _ = _probe_ms(jit_mk46_pallas, r0_8k, p0_8k, f0_8k, key_8k, n_steady=n_steady, n_wall=n_wall)
    ms_mk46_xla, _ = _probe_ms(jit_mk46_xla, r0_8k, p0_8k, f0_8k, noise_8k, n_steady=n_steady, n_wall=n_wall)

    res = {
        "n_real_atoms": n_real,
        "ghost_neighbors_in_real_rows_v23": ghost_count_v23,
        "ghost_neighbors_in_real_rows_v24": ghost_count_v24,
        "energy_diff_pallas_vs_xla_kcal_mol": energy_diff_pal_vs_xla,
        "force_max_diff_pallas_vs_xla_kcal_mol_A": force_max_diff_pal_vs_xla,
        "ragged_bucket_invariance_energy_diff_8k_vs_25k": energy_diff_8k_vs_25k,
        "ragged_bucket_invariance_force_max_diff_8k_vs_25k": force_max_diff_8k_vs_25k,
        "force_eval_25k_ragged_v23_unfiltered_ms": ms_25k_v23,
        "force_eval_25k_ragged_v24_filtered_ms": ms_25k_v24,
        "force_eval_25k_ragged_speedup_v24_vs_v23": ms_25k_v23 / max(ms_25k_v24, 1e-9),
        "force_eval_8k_compact_bucket_v25_ms": ms_8k_v24,
        "cell_prune_25k_ragged_v23_unfiltered_ms": ms_prune_v23,
        "cell_prune_25k_ragged_v24_filtered_ms": ms_prune_v24,
        "cell_prune_25k_ragged_speedup_v24_vs_v23": ms_prune_v23 / max(ms_prune_v24, 1e-9),
        "mk36_separate_direct_and_bonded_8k_ms": ms_sep_8k,
        "mk36_fused_direct_and_bonded_megakernel_8k_ms": ms_fused_8k,
        "mk36_speedup_fused_vs_separate": ms_sep_8k / max(ms_fused_8k, 1e-9),
        "mk38_pme_gather_compute_dq_true_ms": ms_gather_dq_true,
        "mk38_pme_gather_compute_dq_false_ms": ms_gather_dq_false,
        "mk38_speedup_compute_dq_false_vs_true": ms_gather_dq_true / max(ms_gather_dq_false, 1e-9),
        "mk46_xla_baoab_settle_8k_ms": ms_mk46_xla,
        "mk46_pallas_fused_baoab_settle_8k_ms": ms_mk46_pallas,
        "mk46_speedup_pallas_vs_xla": ms_mk46_xla / max(ms_mk46_pallas, 1e-9),
    }
    print(json.dumps(res, indent=2))
    return res


def _dump_bundle_to_npz(bundle, out_npz: str, pme_grid_points: int = 64) -> None:
    """Save MolecularBundle arrays to .npz for OpenMM System construction inside OPENMM_PYTHON."""
    n_real = int(bundle.n_atoms)
    box_vec_a = np.asarray(jnp.diag(bundle.box) if bundle.box.ndim == 2 else bundle.box, dtype=np.float64)
    np.savez(
        out_npz,
        n_real=np.int32(n_real),
        box_vec_a=box_vec_a,
        cutoff_a=np.float64(float(bundle.cutoff_distance)),
        pme_alpha=np.float64(float(bundle.pme_alpha)),
        pme_grid_points=np.int32(int(pme_grid_points)),
        positions_a=np.asarray(bundle.positions[:n_real], dtype=np.float64),
        masses=np.asarray(bundle.masses[:n_real], dtype=np.float64),
        charges=np.asarray(bundle.charges[:n_real], dtype=np.float64),
        sigmas=np.asarray(bundle.sigmas[:n_real], dtype=np.float64),
        epsilons=np.asarray(bundle.epsilons[:n_real], dtype=np.float64),
        water_indices=np.asarray(bundle.water_indices, dtype=np.int32),
        water_mask=np.asarray(bundle.water_mask, dtype=bool),
        bond_idx=np.asarray(bundle.bond_idx, dtype=np.int32),
        bond_params=np.asarray(bundle.bond_params, dtype=np.float64),
        bond_mask=np.asarray(bundle.bond_mask, dtype=bool),
        angle_idx=np.asarray(bundle.angle_idx, dtype=np.int32),
        angle_params=np.asarray(bundle.angle_params, dtype=np.float64),
        angle_mask=np.asarray(bundle.angle_mask, dtype=bool),
        dihedral_idx=np.asarray(bundle.dihedral_idx, dtype=np.int32),
        dihedral_params=np.asarray(bundle.dihedral_params, dtype=np.float64),
        dihedral_mask=np.asarray(bundle.dihedral_mask, dtype=bool),
        improper_idx=np.asarray(bundle.improper_idx, dtype=np.int32),
        improper_params=np.asarray(bundle.improper_params, dtype=np.float64),
        improper_mask=np.asarray(bundle.improper_mask, dtype=bool),
        excl_dense_indices=np.asarray(bundle.excl_dense_indices[:n_real], dtype=np.int32),
        exception_pairs=np.asarray(bundle.exception_pairs, dtype=np.int32),
        exception_sigmas=np.asarray(bundle.exception_sigmas, dtype=np.float64),
        exception_epsilons=np.asarray(bundle.exception_epsilons, dtype=np.float64),
        exception_chargeprods=np.asarray(bundle.exception_chargeprods, dtype=np.float64),
        exception_mask=np.asarray(bundle.exception_mask, dtype=bool),
    )


def run_openmm_cuda_benchmarks(
    bundles_by_name: dict,
    grid_points_by_name: dict[str, int] | None = None,
    n_steps: int = 200,
    dt_fs: float = 2.0,
) -> dict:
    """Benchmark OpenMM CUDA platform across system sizes (single & mixed precision)."""
    import glob
    import os
    import subprocess
    import tempfile

    print("=== [3/3] OpenMM CUDA Platform Head-to-Head Throughput Comparison ===")
    ext_omm_py = os.environ.get(
        "OPENMM_PYTHON",
        "/orcd/pool/007/maarxaru/local/share/mamba/envs/b1-openmm/bin/python",
    )
    if not os.path.isfile(ext_omm_py):
        ext_omm_py = sys.executable

    grid_points_by_name = grid_points_by_name or {}
    omm_results: dict = {}
    with tempfile.TemporaryDirectory() as tmpd:
        specs = []
        for name, bundle in bundles_by_name.items():
            npz_p = os.path.join(tmpd, f"{name}.npz")
            gp = int(grid_points_by_name.get(name, 64))
            _dump_bundle_to_npz(bundle, npz_p, pme_grid_points=gp)
            specs.append({"name": name, "npz": npz_p, "n_atoms": int(bundle.n_atoms), "pme_grid_points": gp})

        specs_json = os.path.join(tmpd, "specs.json")
        Path(specs_json).write_text(json.dumps(specs))

        worker_code = """
import json, os, sys, time, numpy as np, openmm, openmm.unit as u

def build_sys_from_npz(d):
    n_real = int(d['n_real'])
    a_to_nm = 0.1
    kcal_to_kj = 4.184
    box_vec_a = np.asarray(d['box_vec_a'], dtype=np.float64)
    pos_a = np.array(d['positions_a'], dtype=np.float64, copy=True)
    omm_sys = openmm.System()
    omm_sys.setDefaultPeriodicBoxVectors(
        openmm.Vec3(float(box_vec_a[0]) * a_to_nm, 0.0, 0.0),
        openmm.Vec3(0.0, float(box_vec_a[1]) * a_to_nm, 0.0),
        openmm.Vec3(0.0, 0.0, float(box_vec_a[2]) * a_to_nm),
    )
    for m in d['masses']:
        omm_sys.addParticle(float(m))
    r_oh_nm = 0.9572 * a_to_nm
    r_hh_nm = 1.5139 * a_to_nm
    for (ow, hw1, hw2), valid_w in zip(d['water_indices'], d['water_mask']):
        if valid_w and ow < n_real and hw1 < n_real and hw2 < n_real:
            ow_i, hw1_i, hw2_i = int(ow), int(hw1), int(hw2)
            for h_i in (hw1_i, hw2_i):
                dr = pos_a[h_i] - pos_a[ow_i]
                dr -= box_vec_a * np.round(dr / box_vec_a)
                pos_a[h_i] = pos_a[ow_i] + dr
            omm_sys.addConstraint(ow_i, hw1_i, r_oh_nm)
            omm_sys.addConstraint(ow_i, hw2_i, r_oh_nm)
            omm_sys.addConstraint(hw1_i, hw2_i, r_hh_nm)
    hb = openmm.HarmonicBondForce()
    for (ai, aj), (r0_b, k_b), valid_b in zip(d['bond_idx'], d['bond_params'], d['bond_mask']):
        if valid_b and ai < n_real and aj < n_real:
            hb.addBond(int(ai), int(aj), float(r0_b) * a_to_nm, float(k_b) * kcal_to_kj / (a_to_nm * a_to_nm))
    omm_sys.addForce(hb)
    ha = openmm.HarmonicAngleForce()
    for (ai, aj, ak), (th0_a, k_a), valid_a in zip(d['angle_idx'], d['angle_params'], d['angle_mask']):
        if valid_a and ai < n_real and aj < n_real and ak < n_real:
            ha.addAngle(int(ai), int(aj), int(ak), float(th0_a), float(k_a) * kcal_to_kj)
    omm_sys.addForce(ha)
    pt = openmm.PeriodicTorsionForce()
    for idx_k, prm_k, msk_k in (('dihedral_idx', 'dihedral_params', 'dihedral_mask'), ('improper_idx', 'improper_params', 'improper_mask')):
        idx_np, prm_np, msk_np = d[idx_k], d[prm_k], d[msk_k]
        if prm_np.ndim == 2 and prm_np.shape[-1] >= 3:
            for (ai, aj, ak, al), (per, phase, k_d), valid_d in zip(idx_np, prm_np, msk_np):
                if valid_d and max(ai, aj, ak, al) < n_real and int(round(float(per))) > 0:
                    pt.addTorsion(int(ai), int(aj), int(ak), int(al), int(round(float(per))), float(phase), float(k_d) * kcal_to_kj)
    omm_sys.addForce(pt)
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.PME)
    nb.setCutoffDistance(float(d['cutoff_a']) * a_to_nm)
    nb.setUseDispersionCorrection(False)
    g_pts = int(d['pme_grid_points'])
    nb.setPMEParameters(float(d['pme_alpha']) / a_to_nm, g_pts, g_pts, g_pts)
    for q_v, sig_v, eps_v in zip(d['charges'], d['sigmas'], d['epsilons']):
        nb.addParticle(float(q_v), float(sig_v) * a_to_nm, float(eps_v) * kcal_to_kj)
    seen_excl = {}
    ei_np = d['excl_dense_indices']
    for i in range(n_real):
        for j in ei_np[i]:
            if 0 <= int(j) < n_real and int(j) != i:
                pair = (min(i, int(j)), max(i, int(j)))
                seen_excl[pair] = (0.0, 0.1, 0.0)
    for (ai, aj), sig_e, eps_e, qp_e, valid_e in zip(d['exception_pairs'], d['exception_sigmas'], d['exception_epsilons'], d['exception_chargeprods'], d['exception_mask']):
        if valid_e and ai < n_real and aj < n_real and ai != aj:
            pair = (min(int(ai), int(aj)), max(int(ai), int(aj)))
            seen_excl[pair] = (float(qp_e), max(float(sig_e) * a_to_nm, 0.01), float(eps_e) * kcal_to_kj)
    for (p_i, p_j), (qp_v, sig_v, eps_v) in seen_excl.items():
        nb.addException(p_i, p_j, qp_v, sig_v, eps_v, True)
    omm_sys.addForce(nb)
    return omm_sys, pos_a * a_to_nm

specs = json.loads(open(sys.argv[1]).read())
n_steps = int(sys.argv[2])
dt_ps = float(sys.argv[3]) * 1e-3
ps_per_step = dt_ps
load_failures = list(openmm.Platform.getPluginLoadFailures())
avail = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]
if 'CUDA' not in avail:
    plug_dir = os.environ.get('OPENMM_PLUGIN_DIR', '')
    if plug_dir and os.path.isdir(plug_dir):
        openmm.Platform.loadPluginsFromDirectory(plug_dir)
        load_failures.extend(list(openmm.Platform.getPluginLoadFailures()))
        avail = [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())]
plat_name = 'CUDA' if 'CUDA' in avail else ('OpenCL' if 'OpenCL' in avail else 'CPU')
plat = openmm.Platform.getPlatformByName(plat_name)
res = {
    'platform': plat_name,
    'available_platforms': avail,
    'plugin_load_failures': load_failures,
    'systems': {},
}
for sp in specs:
    d = np.load(sp['npz'])
    sys_obj, pos_nm = build_sys_from_npz(d)
    sys_entry = {'n_atoms': sp['n_atoms'], 'pme_grid_points': int(d['pme_grid_points'])}
    for prec in ('mixed', 'single'):
        props = {'Precision': prec} if plat_name in ('CUDA', 'OpenCL') else {}
        integ = openmm.LangevinMiddleIntegrator(300.0*u.kelvin, 10.0/u.picosecond, dt_ps*u.picoseconds)
        ctx = openmm.Context(sys_obj, integ, plat, props)
        ctx.setPositions(pos_nm)
        ctx.setVelocitiesToTemperature(300.0*u.kelvin, 42)
        integ.step(25)
        _ = ctx.getState(getPositions=True)
        t0 = time.perf_counter()
        integ.step(n_steps)
        _ = ctx.getState(getPositions=True)
        wall_s = time.perf_counter() - t0
        ms_per_step = (wall_s / n_steps) * 1000.0
        ns_per_day = (n_steps * ps_per_step * 1e-3) / (wall_s / 86400.0)
        sys_entry[prec] = {
            'ms_per_step': ms_per_step,
            'ns_per_day': ns_per_day,
            'wall_s': wall_s,
            'n_steps': n_steps,
        }
        del ctx, integ
    res['systems'][sp['name']] = sys_entry
print(json.dumps(res))
"""
        env_sub = os.environ.copy()
        omm_root = os.path.dirname(os.path.dirname(ext_omm_py))
        cur_root = os.path.dirname(os.path.dirname(sys.executable))
        plug_dir = os.path.join(omm_root, "lib", "plugins")
        if os.path.isdir(plug_dir):
            env_sub["OPENMM_PLUGIN_DIR"] = plug_dir

        ld_candidates: list[str] = [
            os.path.join(omm_root, "lib"),
            plug_dir,
        ]
        for root_cand in (omm_root, cur_root, "/orcd/pool/008/so3_shared/marielle/projects/.venv"):
            for nv_lib in sorted(glob.glob(os.path.join(root_cand, "lib", "python*", "site-packages", "nvidia", "*", "lib"))):
                ld_candidates.append(nv_lib)
        for cuda_lib in sorted(glob.glob("/orcd/software/core/001/pkg/cuda/*/lib64"), reverse=True):
            ld_candidates.append(cuda_lib)
        for sys_lib in ("/usr/local/cuda/lib64", "/usr/lib64", "/usr/lib/x86_64-linux-gnu"):
            ld_candidates.append(sys_lib)

        valid_ld = [p for p in ld_candidates if p and os.path.isdir(p)]
        if env_sub.get("LD_LIBRARY_PATH"):
            valid_ld.append(env_sub["LD_LIBRARY_PATH"])
        env_sub["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(valid_ld))

        bin_candidates: list[str] = [os.path.join(omm_root, "bin")]
        for cuda_bin in sorted(glob.glob("/orcd/software/core/001/pkg/cuda/*/bin"), reverse=True):
            bin_candidates.append(cuda_bin)
        for root_cand in (omm_root, cur_root, "/orcd/pool/008/so3_shared/marielle/projects/.venv"):
            for nv_bin in sorted(glob.glob(os.path.join(root_cand, "lib", "python*", "site-packages", "nvidia", "*", "bin"))):
                bin_candidates.append(nv_bin)
        valid_bins = [p for p in bin_candidates if p and os.path.isdir(p)]
        if env_sub.get("PATH"):
            valid_bins.append(env_sub["PATH"])
        env_sub["PATH"] = ":".join(dict.fromkeys(valid_bins))

        for bdir in valid_bins:
            nvcc_p = os.path.join(bdir, "nvcc")
            if os.path.isfile(nvcc_p) and os.access(nvcc_p, os.X_OK):
                env_sub.setdefault("OPENMM_CUDA_COMPILER", nvcc_p)
                break

        proc = subprocess.run(
            [ext_omm_py, "-c", worker_code, specs_json, str(n_steps), str(dt_fs)],
            capture_output=True,
            text=True,
            check=False,
            env=env_sub,
        )
        if proc.returncode == 0:
            omm_results = json.loads(proc.stdout.strip().splitlines()[-1])
            print(json.dumps(omm_results, indent=2))
        else:
            omm_results = {"error": proc.stderr[-1200:], "stdout": proc.stdout[-400:]}
            print("OpenMM worker warning:", omm_results)
    return omm_results


def _run_homogeneous_scaling_sweep(
    bundle_f64,
    *,
    bucket_atoms: int,
    real_atoms: int,
    pme_grid_points: int,
    batch_sizes: tuple[int, ...],
    n_steps_warm: int,
    n_steps_prod: int,
    dt_fs: float,
    label: str,
) -> tuple[list[dict], list[dict]]:
    """Run homogeneous EnsemblePlan.run scaling across B in both single (float32) and mixed (float64) modes."""
    import os

    ps_per_step = dt_fs * 1e-3
    bundle_f64 = _cast_bundle_precision(bundle_f64, jnp.float64)
    bundle_f32 = _cast_bundle_precision(bundle_f64, jnp.float32)
    rows_single: list[dict] = []
    rows_mixed: list[dict] = []

    prev_gp = os.environ.get("PROLIX_PME_GRID_POINTS")
    os.environ["PROLIX_PME_GRID_POINTS"] = str(int(pme_grid_points))
    _SINGLE_INFERENCE_CACHE.clear()
    _STACKED_INFERENCE_CACHE.clear()
    try:
        for prec_mode, b_proto, target_rows in (
            ("single", bundle_f32, rows_single),
            ("mixed", bundle_f64, rows_mixed),
        ):
            _SINGLE_INFERENCE_CACHE.clear()
            _STACKED_INFERENCE_CACHE.clear()
            for B in batch_sizes:
                _SINGLE_INFERENCE_CACHE.clear()
                _STACKED_INFERENCE_CACHE.clear()
                bundles = [b_proto for _ in range(B)]
                plan = EnsemblePlan.from_bundles(bundles, use_pallas=True, use_pallas_pme=True)

                t0_cold = time.perf_counter()
                trajs_w = plan.run(
                    n_steps=n_steps_warm,
                    dt=dt_fs,
                    kT=0.6,
                    seed=10,
                    run_mode="inference",
                    use_neighbor_list=True,
                    nl_update_every=20,
                )
                jax.block_until_ready(
                    trajs_w[0].positions if isinstance(trajs_w, list) else trajs_w.positions
                )
                cold_s = time.perf_counter() - t0_cold

                t0_warm = time.perf_counter()
                trajs_p = plan.run(
                    n_steps=n_steps_prod,
                    dt=dt_fs,
                    kT=0.6,
                    seed=42,
                    run_mode="inference",
                    use_neighbor_list=True,
                    nl_update_every=20,
                )
                if isinstance(trajs_p, list):
                    for t in trajs_p:
                        jax.block_until_ready(t.positions)
                else:
                    jax.block_until_ready(trajs_p.positions)
                warm_s = time.perf_counter() - t0_warm
                if isinstance(trajs_p, list):
                    all_finite = all(bool(np.all(np.isfinite(np.asarray(t.positions)))) for t in trajs_p)
                else:
                    all_finite = bool(np.all(np.isfinite(np.asarray(trajs_p.positions))))

                assert all_finite, f"Non-finite trajectory on {label} ({prec_mode}) B={B}"
                step_ms = (warm_s / n_steps_prod) * 1000.0
                per_rep_step_ms = step_ms / B
                agg_ns_per_day = (B * n_steps_prod * ps_per_step * 1e-3) / (warm_s / 86400.0)
                row = {
                    "precision_mode": prec_mode,
                    "bucket_atoms": bucket_atoms,
                    "real_atoms": real_atoms,
                    "pme_grid_points": int(pme_grid_points),
                    "replicas_B": B,
                    "n_steps": n_steps_prod,
                    "cold_compile_and_warmup_s": cold_s,
                    "warm_wall_s": warm_s,
                    "ms_per_step_batch": step_ms,
                    "ms_per_step_per_replica": per_rep_step_ms,
                    "aggregate_ns_per_day": agg_ns_per_day,
                    "per_replica_ns_per_day": agg_ns_per_day / B,
                    "all_finite": all_finite,
                }
                target_rows.append(row)
                print(
                    f"{label} [{prec_mode:6s}, grid={pme_grid_points}^3] B={B:2d}: "
                    f"{step_ms:7.3f} ms/step ({per_rep_step_ms:6.3f} ms/step/rep) | "
                    f"agg={agg_ns_per_day:7.2f} ns/day"
                )
    finally:
        if prev_gp is None:
            os.environ.pop("PROLIX_PME_GRID_POINTS", None)
        else:
            os.environ["PROLIX_PME_GRID_POINTS"] = prev_gp

    return rows_single, rows_mixed


def run_e2e_ensemble_pipeline_benchmarks(dhfr_bundle, quick: bool = False) -> dict:
    import os

    print("=== [2/3] End-to-End Multi-Replica & Multi-Bucket EnsemblePlan.run() Scaling ===")
    _SINGLE_INFERENCE_CACHE.clear()
    _STACKED_INFERENCE_CACHE.clear()

    n_steps_warm = 5 if quick else 20
    n_steps_prod = 20 if quick else 200
    dt_fs = 2.0

    results: dict = {
        "dhfr_25k_homogeneous_scaling_single": [],
        "dhfr_25k_homogeneous_scaling_mixed": [],
        "dhfr_25k_homogeneous_scaling": [],
        "medium_8k_homogeneous_scaling_single": [],
        "medium_8k_homogeneous_scaling_mixed": [],
        "medium_8k_homogeneous_scaling": [],
        "small_4200_8k_homogeneous_scaling_single": [],
        "small_4200_8k_homogeneous_scaling_mixed": [],
        "small_4200_8k_homogeneous_scaling": [],
        "heterogeneous_same_bucket_8k": {},
        "dhfr_wt_plus_mutant_25k": {},
        "heterogeneous_multi_bucket_8k_plus_25k": {},
        "openmm_cuda_comparison": {},
    }

    # 1. Homogeneous 23,558-atom solvated DHFR (25,000 bucket, 64^3 PME grid) across B
    dhfr_wt_25k = _make_truncated_solvated_bundle(
        dhfr_bundle,
        n_real_target=23558,
        bucket_atoms=25000,
        bucket_waters=8192,
        charge_perturbation=0.0,
    )
    dhfr_batch_sizes = (1, 2) if quick else (1, 2, 4, 8, 16)
    dhfr_single, dhfr_mixed = _run_homogeneous_scaling_sweep(
        dhfr_wt_25k,
        bucket_atoms=25000,
        real_atoms=23558,
        pme_grid_points=64,
        batch_sizes=dhfr_batch_sizes,
        n_steps_warm=n_steps_warm,
        n_steps_prod=n_steps_prod,
        dt_fs=dt_fs,
        label="DHFR (23,558 real / 25k bucket)",
    )
    results["dhfr_25k_homogeneous_scaling_single"] = dhfr_single
    results["dhfr_25k_homogeneous_scaling_mixed"] = dhfr_mixed
    results["dhfr_25k_homogeneous_scaling"] = dhfr_single

    # 2. Homogeneous 6,658-atom solvated protein (8,192 bucket, 48^3 PME grid) across B
    ubq_8k = _make_truncated_solvated_bundle(
        dhfr_bundle, n_real_target=6658, bucket_atoms=8192, bucket_waters=2048
    )
    med_batch_sizes = (1, 2) if quick else (1, 2, 4, 8, 16, 32)
    med_single, med_mixed = _run_homogeneous_scaling_sweep(
        ubq_8k,
        bucket_atoms=8192,
        real_atoms=6658,
        pme_grid_points=48,
        batch_sizes=med_batch_sizes,
        n_steps_warm=n_steps_warm,
        n_steps_prod=n_steps_prod,
        dt_fs=dt_fs,
        label="Medium (6,658 real / 8k bucket)",
    )
    results["medium_8k_homogeneous_scaling_single"] = med_single
    results["medium_8k_homogeneous_scaling_mixed"] = med_mixed
    results["medium_8k_homogeneous_scaling"] = med_single

    # 2b. Homogeneous 4,199-atom solvated protein (8,192 bucket, 32^3 PME grid) across B
    sys_4200 = _make_truncated_solvated_bundle(
        dhfr_bundle, n_real_target=4200, bucket_atoms=8192, bucket_waters=2048, seed_shift=0.3
    )
    sm_single, sm_mixed = _run_homogeneous_scaling_sweep(
        sys_4200,
        bucket_atoms=8192,
        real_atoms=int(sys_4200.n_atoms),
        pme_grid_points=32,
        batch_sizes=med_batch_sizes,
        n_steps_warm=n_steps_warm,
        n_steps_prod=n_steps_prod,
        dt_fs=dt_fs,
        label=f"Small ({int(sys_4200.n_atoms)} real / 8k bucket)",
    )
    results["small_4200_8k_homogeneous_scaling_single"] = sm_single
    results["small_4200_8k_homogeneous_scaling_mixed"] = sm_mixed
    results["small_4200_8k_homogeneous_scaling"] = sm_single

    # 3. Heterogeneous same-bucket (4,199 real atoms + 6,658 real atoms in shared 8,192 bucket)
    os.environ["PROLIX_PME_GRID_POINTS"] = "48"
    _SINGLE_INFERENCE_CACHE.clear()
    _STACKED_INFERENCE_CACHE.clear()
    het_same_bundles = [_cast_bundle_precision(sys_4200, jnp.float32), _cast_bundle_precision(ubq_8k, jnp.float32)]
    plan_het_same = EnsemblePlan.from_bundles(het_same_bundles, use_pallas=True, use_pallas_pme=True)
    trajs_hs_w = plan_het_same.run(
        n_steps=n_steps_warm,
        dt=dt_fs,
        kT=0.6,
        seed=5,
        run_mode="inference",
        use_neighbor_list=True,
        nl_update_every=20,
    )
    for t in trajs_hs_w:
        jax.block_until_ready(t.positions)

    t0_hs = time.perf_counter()
    trajs_hs = plan_het_same.run(
        n_steps=n_steps_prod,
        dt=dt_fs,
        kT=0.6,
        seed=77,
        run_mode="inference",
        use_neighbor_list=True,
        nl_update_every=20,
    )
    for t in trajs_hs:
        jax.block_until_ready(t.positions)
    hs_s = time.perf_counter() - t0_hs
    assert trajs_hs[0].positions.shape[1] == int(sys_4200.n_atoms)
    assert trajs_hs[1].positions.shape[1] == int(ubq_8k.n_atoms)
    results["heterogeneous_same_bucket_8k"] = {
        "bucket_atoms": 8192,
        "pme_grid_points": 48,
        "system_real_atoms": [int(sys_4200.n_atoms), int(ubq_8k.n_atoms)],
        "n_steps": n_steps_prod,
        "ms_per_step_batch": (hs_s / n_steps_prod) * 1000.0,
        "ms_per_step_per_system": (hs_s / n_steps_prod) * 500.0,
        "all_finite": all(bool(jnp.all(jnp.isfinite(t.positions))) for t in trajs_hs),
    }

    # 4. DHFR wild-type + mutant variant in shared 25,000 bucket
    os.environ["PROLIX_PME_GRID_POINTS"] = "64"
    _SINGLE_INFERENCE_CACHE.clear()
    _STACKED_INFERENCE_CACHE.clear()
    dhfr_mutant = _make_truncated_solvated_bundle(
        dhfr_bundle,
        n_real_target=23558,
        bucket_atoms=25000,
        bucket_waters=8192,
        charge_perturbation=0.08,
    )
    plan_wt_mut = EnsemblePlan.from_bundles(
        [_cast_bundle_precision(dhfr_wt_25k, jnp.float32), _cast_bundle_precision(dhfr_mutant, jnp.float32)],
        use_pallas=True,
        use_pallas_pme=True,
    )
    trajs_wm_w = plan_wt_mut.run(
        n_steps=n_steps_warm,
        dt=dt_fs,
        kT=0.6,
        seed=100,
        run_mode="inference",
        use_neighbor_list=True,
        nl_update_every=20,
    )
    for t in trajs_wm_w:
        jax.block_until_ready(t.positions)

    t0_wm = time.perf_counter()
    trajs_wm = plan_wt_mut.run(
        n_steps=n_steps_prod,
        dt=dt_fs,
        kT=0.6,
        seed=200,
        run_mode="inference",
        use_neighbor_list=True,
        nl_update_every=20,
    )
    for t in trajs_wm:
        jax.block_until_ready(t.positions)
    wm_s = time.perf_counter() - t0_wm

    wt_mut_rmsd = float(
        jnp.sqrt(jnp.mean((trajs_wm[0].positions[:, :23558, :] - trajs_wm[1].positions[:, :23558, :]) ** 2))
    )
    assert wt_mut_rmsd > 0.0, "Expected DHFR WT and mutant trajectories to differ"
    results["dhfr_wt_plus_mutant_25k"] = {
        "bucket_atoms": 25000,
        "pme_grid_points": 64,
        "real_atoms": [23558, 23558],
        "n_steps": n_steps_prod,
        "ms_per_step_batch": (wm_s / n_steps_prod) * 1000.0,
        "ms_per_step_per_variant": (wm_s / n_steps_prod) * 500.0,
        "wt_vs_mutant_rmsd_A": wt_mut_rmsd,
        "all_finite": all(bool(jnp.all(jnp.isfinite(t.positions))) for t in trajs_wm),
    }

    # 4b. Heterogeneous DHFR mutant screen scaling (B = 1, 2, 4, 8, 16, 32 variants with point mutations + length deletions)
    screen_batch_sizes = (1, 2) if quick else (1, 2, 4, 8, 16, 32)
    ps_per_step = dt_fs * 1e-3
    screen_rows_single: list[dict] = []
    screen_rows_mixed: list[dict] = []
    os.environ["PROLIX_PME_GRID_POINTS"] = "64"
    for prec_mode, dtype_target, target_rows in (
        ("single", jnp.float32, screen_rows_single),
        ("mixed", jnp.float64, screen_rows_mixed),
    ):
        for B_scr in screen_batch_sizes:
            _SINGLE_INFERENCE_CACHE.clear()
            _STACKED_INFERENCE_CACHE.clear()
            het_scr_bundles = [
                _cast_bundle_precision(b, dtype_target)
                for b in _make_heterogeneous_dhfr_mutant_screen(dhfr_bundle, B_scr)
            ]
            plan_scr = EnsemblePlan.from_bundles(het_scr_bundles, use_pallas=True, use_pallas_pme=True)
            t0_c = time.perf_counter()
            tr_w = plan_scr.run(
                n_steps=n_steps_warm,
                dt=dt_fs,
                kT=0.6,
                seed=50,
                run_mode="inference",
                use_neighbor_list=True,
                nl_update_every=20,
            )
            if isinstance(tr_w, list):
                for t in tr_w:
                    jax.block_until_ready(t.positions)
            else:
                jax.block_until_ready(tr_w.positions)
            cold_scr_s = time.perf_counter() - t0_c

            t0_w = time.perf_counter()
            tr_p = plan_scr.run(
                n_steps=n_steps_prod,
                dt=dt_fs,
                kT=0.6,
                seed=99,
                run_mode="inference",
                use_neighbor_list=True,
                nl_update_every=20,
            )
            if isinstance(tr_p, list):
                for t in tr_p:
                    jax.block_until_ready(t.positions)
            else:
                jax.block_until_ready(tr_p.positions)
            warm_scr_s = time.perf_counter() - t0_w
            if isinstance(tr_p, list):
                all_fin = all(bool(np.all(np.isfinite(np.asarray(t.positions)))) for t in tr_p)
            else:
                all_fin = bool(np.all(np.isfinite(np.asarray(tr_p.positions))))
            assert all_fin, f"Non-finite trajectory on DHFR heterogeneous screen ({prec_mode}) B={B_scr}"
            step_ms_s = (warm_scr_s / n_steps_prod) * 1000.0
            per_var_ms = step_ms_s / B_scr
            agg_ns_day = (B_scr * n_steps_prod * ps_per_step * 1e-3) / (warm_scr_s / 86400.0)
            row_scr = {
                "precision_mode": prec_mode,
                "bucket_atoms": 25000,
                "real_atoms_min": min(int(b.n_atoms) for b in het_scr_bundles),
                "real_atoms_max": max(int(b.n_atoms) for b in het_scr_bundles),
                "pme_grid_points": 64,
                "variants_B": B_scr,
                "n_steps": n_steps_prod,
                "cold_compile_and_warmup_s": cold_scr_s,
                "warm_wall_s": warm_scr_s,
                "ms_per_step_batch": step_ms_s,
                "ms_per_step_per_variant": per_var_ms,
                "aggregate_ns_per_day": agg_ns_day,
                "per_variant_ns_per_day": agg_ns_day / B_scr,
                "all_finite": all_fin,
            }
            target_rows.append(row_scr)
            print(
                f"DHFR Hetero Mutant Screen [{prec_mode:6s}, grid=64^3] B={B_scr:2d}: "
                f"{step_ms_s:7.3f} ms/step ({per_var_ms:6.3f} ms/step/var) | "
                f"agg={agg_ns_day:7.2f} ns/day"
            )
    results["dhfr_25k_heterogeneous_mutant_screen_single"] = screen_rows_single
    results["dhfr_25k_heterogeneous_mutant_screen_mixed"] = screen_rows_mixed

    # 5. Heterogeneous multi-bucket ([sys_4200, ubq_8k] in 8k bucket + [dhfr_bundle, dhfr_mutant] in 25k bucket)
    os.environ["PROLIX_PME_GRID_POINTS"] = "8192:48,25000:64"
    multi_bucket_bundles = [
        _cast_bundle_precision(sys_4200, jnp.float32),
        _cast_bundle_precision(dhfr_wt_25k, jnp.float32),
        _cast_bundle_precision(ubq_8k, jnp.float32),
        _cast_bundle_precision(dhfr_mutant, jnp.float32),
    ]
    plan_mb = EnsemblePlan.from_bundles(multi_bucket_bundles, use_pallas=True, use_pallas_pme=True)
    trajs_mb_w = plan_mb.run(
        n_steps=n_steps_warm,
        dt=dt_fs,
        kT=0.6,
        seed=300,
        run_mode="inference",
        use_neighbor_list=True,
        nl_update_every=20,
    )
    for t in trajs_mb_w:
        jax.block_until_ready(t.positions)

    t0_mb = time.perf_counter()
    trajs_mb = plan_mb.run(
        n_steps=n_steps_prod,
        dt=dt_fs,
        kT=0.6,
        seed=400,
        run_mode="inference",
        use_neighbor_list=True,
        nl_update_every=20,
    )
    for t in trajs_mb:
        jax.block_until_ready(t.positions)
    mb_s = time.perf_counter() - t0_mb
    os.environ.pop("PROLIX_PME_GRID_POINTS", None)
    assert [int(t.positions.shape[1]) for t in trajs_mb] == [
        int(sys_4200.n_atoms),
        23558,
        int(ubq_8k.n_atoms),
        23558,
    ]
    results["heterogeneous_multi_bucket_8k_plus_25k"] = {
        "buckets": [8192, 25000, 8192, 25000],
        "real_atoms": [int(sys_4200.n_atoms), 23558, int(ubq_8k.n_atoms), 23558],
        "n_steps": n_steps_prod,
        "ms_per_step_total_4_systems": (mb_s / n_steps_prod) * 1000.0,
        "ms_per_step_per_system": (mb_s / n_steps_prod) * 250.0,
        "all_finite": all(bool(jnp.all(jnp.isfinite(t.positions))) for t in trajs_mb),
    }

    # 6. Head-to-head comparison against OpenMM CUDA across all 3 system sizes
    omm_raw = run_openmm_cuda_benchmarks(
        {
            "small_4200": sys_4200,
            "medium_6658": ubq_8k,
            "dhfr_23558": dhfr_wt_25k,
        },
        grid_points_by_name={
            "small_4200": 32,
            "medium_6658": 48,
            "dhfr_23558": 64,
        },
        n_steps=n_steps_prod,
        dt_fs=dt_fs,
    )
    if "systems" in omm_raw:
        for sys_key, single_rows, mixed_rows in (
            ("small_4200", results["small_4200_8k_homogeneous_scaling_single"], results["small_4200_8k_homogeneous_scaling_mixed"]),
            ("medium_6658", results["medium_8k_homogeneous_scaling_single"], results["medium_8k_homogeneous_scaling_mixed"]),
            ("dhfr_23558", results["dhfr_25k_homogeneous_scaling_single"], results["dhfr_25k_homogeneous_scaling_mixed"]),
        ):
            omm_s = omm_raw["systems"].get(sys_key, {})
            if "mixed" in omm_s and "single" in omm_s:
                omm_mixed_ns = omm_s["mixed"]["ns_per_day"]
                omm_single_ns = omm_s["single"]["ns_per_day"]
                for r in single_rows + mixed_rows:
                    r["openmm_cuda_mixed_ns_per_day"] = omm_mixed_ns
                    r["openmm_cuda_single_ns_per_day"] = omm_single_ns
                    r["speedup_vs_openmm_cuda_mixed"] = r["aggregate_ns_per_day"] / max(omm_mixed_ns, 1e-9)
                    r["speedup_vs_openmm_cuda_single"] = r["aggregate_ns_per_day"] / max(omm_single_ns, 1e-9)
        omm_dhfr = omm_raw["systems"].get("dhfr_23558", {})
        if "mixed" in omm_dhfr and "single" in omm_dhfr:
            omm_mixed_ns = omm_dhfr["mixed"]["ns_per_day"]
            omm_single_ns = omm_dhfr["single"]["ns_per_day"]
            for r in screen_rows_single + screen_rows_mixed:
                r["openmm_cuda_mixed_ns_per_day"] = omm_mixed_ns
                r["openmm_cuda_single_ns_per_day"] = omm_single_ns
                r["speedup_vs_openmm_cuda_mixed"] = r["aggregate_ns_per_day"] / max(omm_mixed_ns, 1e-9)
                r["speedup_vs_openmm_cuda_single"] = r["aggregate_ns_per_day"] / max(omm_single_ns, 1e-9)
    results["openmm_cuda_comparison"] = omm_raw
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/bench/261106_pallas_v31_e2e_ensemble_buckets.json"),
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--substep-only", action="store_true")
    parser.add_argument("--v33-diverse-panel", action="store_true")
    parser.add_argument("--axis", type=str, default="all")
    args = parser.parse_args()

    if args.v33_diverse_panel:
        import bench_v33_diverse_panel as B_v33

        axes = None if args.axis == "all" else [int(x.strip()) for x in args.axis.split(",") if x.strip()]
        payload = B_v33.run_v33_diverse_benchmark_panel(quick=args.quick, axes=axes)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {args.out}")
        return

    ff_path = B_dhfr._resolve_ff_path()
    sys_data = B_dhfr._build_dhfr_system_dict(ff_path)
    dhfr_bundle = B_dhfr._build_bundle_from_sys_data(sys_data)

    dev = jax.devices()[0]
    dev_kind = str(getattr(dev, "device_kind", str(dev)))
    cc = getattr(dev, "compute_capability", None)
    cp_small = resolve_pallas_compiler_params("nonbonded_nl", grid_blocks=128)
    cp_large = resolve_pallas_compiler_params("nonbonded_nl", grid_blocks=2048)

    substep_profile = run_dhfr_25k_substep_profile(dhfr_bundle, quick=args.quick)
    if args.substep_only:
        payload = {
            "meta": {
                "version": "pallas_v3.2_substep_profile",
                "device": str(dev),
                "device_kind": dev_kind,
                "compute_capability": cc,
                "jax_version": jax.__version__,
            },
            "dhfr_substep_profile": substep_profile,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {args.out}")
        return

    ragged_and_audit = run_ragged_bucket_and_accuracy_audit(dhfr_bundle, quick=args.quick)
    e2e_results = run_e2e_ensemble_pipeline_benchmarks(dhfr_bundle, quick=args.quick)

    payload = {
        "meta": {
            "version": "pallas_v3.2",
            "device": str(dev),
            "device_kind": dev_kind,
            "compute_capability": cc,
            "jax_version": jax.__version__,
            "compiler_params_small_grid": {
                "num_warps": cp_small.num_warps,
                "num_stages": cp_small.num_stages,
            },
            "compiler_params_large_grid": {
                "num_warps": cp_large.num_warps,
                "num_stages": cp_large.num_stages,
            },
        },
        "dhfr_substep_profile": substep_profile,
        "ragged_bucket_and_accuracy_audit": ragged_and_audit,
        "e2e_ensemble_pipeline": e2e_results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

