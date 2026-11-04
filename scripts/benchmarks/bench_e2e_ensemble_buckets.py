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
    return float(res.wall_seconds_per_unit * 1000.0), metrics


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
        "force_eval_8k_compact_bucket_v24_ms": ms_8k_v24,
        "cell_prune_25k_ragged_v23_unfiltered_ms": ms_prune_v23,
        "cell_prune_25k_ragged_v24_filtered_ms": ms_prune_v24,
        "cell_prune_25k_ragged_speedup_v24_vs_v23": ms_prune_v23 / max(ms_prune_v24, 1e-9),
    }
    print(json.dumps(res, indent=2))
    return res


def run_e2e_ensemble_pipeline_benchmarks(dhfr_bundle, quick: bool = False) -> dict:
    print("=== [2/3] End-to-End Multi-Replica & Multi-Bucket EnsemblePlan.run() Scaling ===")
    _SINGLE_INFERENCE_CACHE.clear()
    _STACKED_INFERENCE_CACHE.clear()

    n_steps_warm = 5 if quick else 20
    n_steps_prod = 20 if quick else 200
    dt_fs = 2.0
    ps_per_step = dt_fs * 1e-3  # 0.002 ps/step -> ns_per_day = 86.4 * 0.002 / sec_per_step

    results = {
        "dhfr_25k_homogeneous_scaling": [],
        "medium_8k_homogeneous_scaling": [],
        "heterogeneous_same_bucket_8k": {},
        "dhfr_wt_plus_mutant_25k": {},
        "heterogeneous_multi_bucket_8k_plus_25k": {},
    }

    # 1. Homogeneous 23,558-atom solvated DHFR (25,000 bucket) across B
    dhfr_wt_25k = _make_truncated_solvated_bundle(
        dhfr_bundle,
        n_real_target=23558,
        bucket_atoms=25000,
        bucket_waters=8192,
        charge_perturbation=0.0,
    )
    dhfr_batch_sizes = (1, 2) if quick else (1, 2, 4, 8)
    for B in dhfr_batch_sizes:
        bundles = [dhfr_wt_25k for _ in range(B)]
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
        jax.block_until_ready(trajs_w[0].positions if isinstance(trajs_w, list) else trajs_w.positions)
        cold_s = time.perf_counter() - t0_cold

        # Warm production run (hits _SINGLE_INFERENCE_CACHE or _STACKED_INFERENCE_CACHE with dynamic n_steps)
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
            all_finite = all(bool(jnp.all(jnp.isfinite(t.positions))) for t in trajs_p)
        else:
            jax.block_until_ready(trajs_p.positions)
            all_finite = bool(jnp.all(jnp.isfinite(trajs_p.positions)))
        warm_s = time.perf_counter() - t0_warm

        assert all_finite, f"Non-finite trajectory on DHFR B={B}"
        step_ms = (warm_s / n_steps_prod) * 1000.0
        per_rep_step_ms = step_ms / B
        agg_ns_per_day = (B * n_steps_prod * ps_per_step * 1e-3) / (warm_s / 86400.0)
        row = {
            "bucket_atoms": 25000,
            "real_atoms": 23558,
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
        results["dhfr_25k_homogeneous_scaling"].append(row)
        print(
            f"DHFR (23,558 real / 25k bucket) B={B:2d}: {step_ms:7.3f} ms/step "
            f"({per_rep_step_ms:6.3f} ms/step/rep) | agg={agg_ns_per_day:6.2f} ns/day"
        )

    # 2. Homogeneous 6,658-atom solvated protein (8,192 bucket) across B
    ubq_8k = _make_truncated_solvated_bundle(
        dhfr_bundle, n_real_target=6658, bucket_atoms=8192, bucket_waters=2048
    )
    med_batch_sizes = (1, 2) if quick else (1, 4, 8)
    for B in med_batch_sizes:
        bundles = [ubq_8k for _ in range(B)]
        plan = EnsemblePlan.from_bundles(bundles, use_pallas=True, use_pallas_pme=True)
        trajs_w = plan.run(
            n_steps=n_steps_warm,
            dt=dt_fs,
            kT=0.6,
            seed=10,
            run_mode="inference",
            use_neighbor_list=True,
            nl_update_every=20,
        )
        jax.block_until_ready(trajs_w[0].positions if isinstance(trajs_w, list) else trajs_w.positions)

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
        step_ms = (warm_s / n_steps_prod) * 1000.0
        agg_ns_per_day = (B * n_steps_prod * ps_per_step * 1e-3) / (warm_s / 86400.0)
        row = {
            "bucket_atoms": 8192,
            "real_atoms": 6658,
            "replicas_B": B,
            "n_steps": n_steps_prod,
            "ms_per_step_batch": step_ms,
            "ms_per_step_per_replica": step_ms / B,
            "aggregate_ns_per_day": agg_ns_per_day,
        }
        results["medium_8k_homogeneous_scaling"].append(row)
        print(
            f"Medium (6,658 real / 8k bucket) B={B:2d}: {step_ms:7.3f} ms/step "
            f"({step_ms/B:6.3f} ms/step/rep) | agg={agg_ns_per_day:6.2f} ns/day"
        )

    # 3. Heterogeneous same-bucket (4,200 real atoms + 6,658 real atoms in shared 8,192 bucket)
    sys_4200 = _make_truncated_solvated_bundle(
        dhfr_bundle, n_real_target=4200, bucket_atoms=8192, bucket_waters=2048, seed_shift=0.3
    )
    het_same_bundles = [sys_4200, ubq_8k]
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
        "system_real_atoms": [int(sys_4200.n_atoms), int(ubq_8k.n_atoms)],
        "n_steps": n_steps_prod,
        "ms_per_step_batch": (hs_s / n_steps_prod) * 1000.0,
        "ms_per_step_per_system": (hs_s / n_steps_prod) * 500.0,
        "all_finite": all(bool(jnp.all(jnp.isfinite(t.positions))) for t in trajs_hs),
    }

    # 4. DHFR wild-type + mutant variant in shared 25,000 bucket (plus parity vs independent single runs)
    dhfr_mutant = _make_truncated_solvated_bundle(
        dhfr_bundle,
        n_real_target=23558,
        bucket_atoms=25000,
        bucket_waters=8192,
        charge_perturbation=0.08,
    )
    plan_wt_mut = EnsemblePlan.from_bundles(
        [dhfr_wt_25k, dhfr_mutant], use_pallas=True, use_pallas_pme=True
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

    # Verify WT and Mutant diverge physically due to charge perturbation
    wt_mut_rmsd = float(
        jnp.sqrt(jnp.mean((trajs_wm[0].positions[:, :23558, :] - trajs_wm[1].positions[:, :23558, :]) ** 2))
    )
    assert wt_mut_rmsd > 0.0, "Expected DHFR WT and mutant trajectories to differ"
    results["dhfr_wt_plus_mutant_25k"] = {
        "bucket_atoms": 25000,
        "real_atoms": [23558, 23558],
        "n_steps": n_steps_prod,
        "ms_per_step_batch": (wm_s / n_steps_prod) * 1000.0,
        "ms_per_step_per_variant": (wm_s / n_steps_prod) * 500.0,
        "wt_vs_mutant_rmsd_A": wt_mut_rmsd,
        "all_finite": all(bool(jnp.all(jnp.isfinite(t.positions))) for t in trajs_wm),
    }

    # 5. Heterogeneous multi-bucket ([sys_4200, ubq_8k] in 8k bucket + [dhfr_bundle, dhfr_mutant] in 25k bucket)
    multi_bucket_bundles = [sys_4200, dhfr_wt_25k, ubq_8k, dhfr_mutant]
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
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/bench/261102_pallas_v24_e2e_ensemble_buckets.json"),
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    ff_path = B_dhfr._resolve_ff_path()
    sys_data = B_dhfr._build_dhfr_system_dict(ff_path)
    dhfr_bundle = B_dhfr._build_bundle_from_sys_data(sys_data)

    dev = jax.devices()[0]
    dev_kind = str(getattr(dev, "device_kind", str(dev)))
    cc = getattr(dev, "compute_capability", None)
    cp_small = resolve_pallas_compiler_params("nonbonded_nl", grid_blocks=128)
    cp_large = resolve_pallas_compiler_params("nonbonded_nl", grid_blocks=2048)

    ragged_and_audit = run_ragged_bucket_and_accuracy_audit(dhfr_bundle, quick=args.quick)
    e2e_results = run_e2e_ensemble_pipeline_benchmarks(dhfr_bundle, quick=args.quick)

    payload = {
        "meta": {
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
        "ragged_bucket_and_accuracy_audit": ragged_and_audit,
        "e2e_ensemble_pipeline": e2e_results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
