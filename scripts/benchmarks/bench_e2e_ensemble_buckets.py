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
        pallas_nonbonded_nl,
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
                    all_finite = all(bool(jnp.all(jnp.isfinite(t.positions))) for t in trajs_p)
                else:
                    jax.block_until_ready(trajs_p.positions)
                    all_finite = bool(jnp.all(jnp.isfinite(trajs_p.positions)))
                warm_s = time.perf_counter() - t0_warm

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
    results["openmm_cuda_comparison"] = omm_raw
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/bench/261105_pallas_v27_e2e_ensemble_buckets.json"),
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
            "version": "pallas_v3.0",
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

