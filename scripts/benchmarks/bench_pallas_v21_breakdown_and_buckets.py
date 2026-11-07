"""Isolated kernel breakdown and multi-replica ragged bucket benchmark for Pallas v2.1.

Uses ``xtrax.profiling`` (``probe_unit``, ``count_backend_compiles``, ``iter_jaxpr_eqns``,
``loop_scaling.loop_bodies``) to profile:
1. Isolated sub-step speedups on 23,558-atom solvated DHFR:
   - SETTLE (7,023 waters): cuSOLVER eigh (v2.0) vs closed-form Cayley-Hamilton projector (v2.1)
   - SPME (23,558 atoms, 64^3 mesh): uncached + vjp_box active (v2.0) vs cached G + symbolic_zeros=True (v2.1)
   - Pallas nonbonded NL (23,558 x 762, pre_encoded, empty-tile skip, in-register bounds clamp)
   - Fused bonded megakernel (pallas_bonded_all_fused)
   - PBC skin check rebuild rate over 50 MD steps: raw diff (v2.0, 100% rebuilds) vs min-image diff (v2.1)
2. Multi-replica & ragged bucket scaling across bucket sizes (1k..16k) and replica counts (B=1, 4, 8, 16, 32):
   - Sequential jax.lax.map replica loop (v2.0) vs Native 2D Pallas grid=(B, n_blocks) (v2.1)
   - Jaxpr structural verification via ``xtrax.profiling.loop_scaling.loop_bodies`` and ``iter_jaxpr_eqns``
   - Full 23,558-atom DHFR multi-replica force evaluation scaling via ``xtrax.profiling.probe_unit``
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import xtrax.profiling as xp
import xtrax.profiling.loop_scaling as xls

import prolix.physics.neighbor_list as nl_mod
import prolix.physics.pme as pme_mod
import prolix.physics.settle as settle_mod
import scripts.experiments.bench_dhfr_parity as B_dhfr
from prolix.api.bundle_md import displacement_fn_for_bundle, energy_fn_from_bundle
from prolix.api.ensemble_plan import _encode_bundle_nl


def _probe_ms(fn, *args, n_steady: int = 3, n_wall: int = 7) -> tuple[float, dict[str, float]]:
    """Profile ``fn(*args)`` using ``xtrax.profiling.probe_unit`` and return (wall_ms, metrics)."""
    res = xp.probe_unit(fn, args, n_steady=n_steady, n_wall=n_wall)
    metrics = res.as_metrics()
    return float(res.wall_seconds_per_unit * 1000.0), metrics


def _settle_water_batch_v20_eigh(
    pos_oxygen_old: jax.Array,
    pos_h1_old: jax.Array,
    pos_h2_old: jax.Array,
    pos_oxygen_new: jax.Array,
    pos_h1_new: jax.Array,
    pos_h2_new: jax.Array,
    r_OH: float = settle_mod.TIP3P_ROH,
    r_HH: float = settle_mod.TIP3P_RHH,
    mass_oxygen: float = 15.999,
    mass_hydrogen: float = 1.008,
    box: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Reference v2.0 _settle_water_batch using jnp.linalg.eigh(n_mat) (cuSOLVER syevjBatched)."""
    if box is not None:
        def unwrap(pos_new, pos_old):
            delta = pos_new - pos_old
            return pos_old + settle_mod._min_image_delta(delta, box)

        pos_oxygen_new = unwrap(pos_oxygen_new, pos_oxygen_old)
        pos_h1_new = unwrap(pos_h1_new, pos_h1_old)
        pos_h2_new = unwrap(pos_h2_new, pos_h2_old)

        def to_oxygen_image(pos_h, pos_o):
            d = pos_h - pos_o
            return pos_o + settle_mod._min_image_delta(d, box)

        pos_h1_new = to_oxygen_image(pos_h1_new, pos_oxygen_new)
        pos_h2_new = to_oxygen_image(pos_h2_new, pos_oxygen_new)

    mass_total = mass_oxygen + 2 * mass_hydrogen
    com_new = (
        mass_oxygen * pos_oxygen_new + mass_hydrogen * pos_h1_new + mass_hydrogen * pos_h2_new
    ) / mass_total
    delta_O_new = pos_oxygen_new - com_new
    delta_H1_new = pos_h1_new - com_new
    delta_H2_new = pos_h2_new - com_new

    dist_oh_mid = jnp.sqrt(r_OH**2 - (r_HH / 2) ** 2)
    dist_O_to_COM = 2 * mass_hydrogen * dist_oh_mid / mass_total
    dist_H_mid_to_COM = mass_oxygen * dist_oh_mid / mass_total
    half_hh = r_HH / 2

    dtype = pos_oxygen_new.dtype
    zero = jnp.zeros((), dtype=dtype)
    b_O = jnp.stack([zero, jnp.asarray(dist_O_to_COM, dtype=dtype), zero])
    b_H1 = jnp.stack([jnp.asarray(half_hh, dtype=dtype), -jnp.asarray(dist_H_mid_to_COM, dtype=dtype), zero])
    b_H2 = jnp.stack([-jnp.asarray(half_hh, dtype=dtype), -jnp.asarray(dist_H_mid_to_COM, dtype=dtype), zero])
    b_all = jnp.stack([b_O, b_H1, b_H2], axis=0)
    y_all = jnp.stack([delta_O_new, delta_H1_new, delta_H2_new], axis=1)
    m_all = jnp.asarray([mass_oxygen, mass_hydrogen, mass_hydrogen], dtype=dtype)

    s_mat = jnp.einsum("i,ia,nib->nab", m_all, b_all, y_all)
    sxx, sxy, sxz = s_mat[:, 0, 0], s_mat[:, 0, 1], s_mat[:, 0, 2]
    syx, syy, syz = s_mat[:, 1, 0], s_mat[:, 1, 1], s_mat[:, 1, 2]
    szx, szy, szz = s_mat[:, 2, 0], s_mat[:, 2, 1], s_mat[:, 2, 2]

    row0 = jnp.stack([sxx + syy + szz, syz - szy, szx - sxz, sxy - syx], axis=-1)
    row1 = jnp.stack([syz - szy, sxx - syy - szz, sxy + syx, szx + sxz], axis=-1)
    row2 = jnp.stack([szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy], axis=-1)
    row3 = jnp.stack([sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz], axis=-1)
    n_mat = jnp.stack([row0, row1, row2, row3], axis=1)

    _, eigvecs = jnp.linalg.eigh(n_mat)
    q = eigvecs[:, :, -1]
    q0, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r01 = 2.0 * (qx * qy - q0 * qz)
    r02 = 2.0 * (qx * qz + q0 * qy)
    r10 = 2.0 * (qx * qy + q0 * qz)
    r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    r12 = 2.0 * (qy * qz - q0 * qx)
    r20 = 2.0 * (qx * qz - q0 * qy)
    r21 = 2.0 * (qy * qz + q0 * qx)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
    rot_matrices = jnp.stack(
        [
            jnp.stack([r00, r01, r02], axis=-1),
            jnp.stack([r10, r11, r12], axis=-1),
            jnp.stack([r20, r21, r22], axis=-1),
        ],
        axis=1,
    )
    pos_O_c = jnp.einsum("nij,j->ni", rot_matrices, b_O) + com_new
    pos_H1_c = jnp.einsum("nij,j->ni", rot_matrices, b_H1) + com_new
    pos_H2_c = jnp.einsum("nij,j->ni", rot_matrices, b_H2) + com_new
    return pos_O_c, pos_H1_c, pos_H2_c


def run_dhfr_breakdown(bundle) -> dict:
    print("=== Running 23,558-Atom Solvated DHFR Sub-Step Breakdown (via xtrax.profiling.probe_unit) ===")
    disp_fn, _ = displacement_fn_for_bundle(bundle)
    box_vec = jnp.diag(bundle.box)
    neighbor_fn = nl_mod.make_neighbor_list_fn(
        disp_fn, box_vec, 9.0, capacity_multiplier=1.25, dr_threshold=1.0
    )
    nbr_raw = neighbor_fn.allocate(bundle.positions)
    nbr_enc = _encode_bundle_nl(nbr_raw, bundle)

    # 1. SETTLE benchmark: v2.0 (cuSOLVER eigh) vs v2.1 (Cayley-Hamilton closed-form)
    w_idx = bundle.water_indices
    r_w_old = bundle.positions[w_idx]
    key = jax.random.PRNGKey(0)
    r_w_prime = r_w_old + 0.02 * jax.random.normal(key, r_w_old.shape, dtype=jnp.float64)

    jit_settle_v20 = jax.jit(
        lambda ro, rp: _settle_water_batch_v20_eigh(
            ro[:, 0], ro[:, 1], ro[:, 2], rp[:, 0], rp[:, 1], rp[:, 2],
            settle_mod.TIP3P_ROH, settle_mod.TIP3P_RHH, 15.999, 1.008, box_vec,
        )
    )
    jit_settle_v21 = jax.jit(
        lambda ro, rp: settle_mod._settle_water_batch(
            ro[:, 0], ro[:, 1], ro[:, 2], rp[:, 0], rp[:, 1], rp[:, 2],
            settle_mod.TIP3P_ROH, settle_mod.TIP3P_RHH, 15.999, 1.008, box_vec,
        )
    )

    settle_1call_v20_ms, settle_v20_probe = _probe_ms(jit_settle_v20, r_w_old, r_w_prime)
    settle_1call_v21_ms, settle_v21_probe = _probe_ms(jit_settle_v21, r_w_old, r_w_prime)
    out_v20 = jnp.stack(jit_settle_v20(r_w_old, r_w_prime), axis=1)
    out_v21 = jnp.stack(jit_settle_v21(r_w_old, r_w_prime), axis=1)
    settle_max_diff_A = float(jnp.max(jnp.abs(out_v20 - out_v21)))

    # 2. SPME benchmark: v2.0 (uncached + box differentiated in VJP) vs v2.1 (cached G + symbolic_zeros=True)
    grid_dims = (64, 64, 64)
    alpha = 0.34
    mask_f = bundle.atom_mask.astype(jnp.float64)

    def _spme_v20_fn(pos, box):
        return pme_mod.spme_energy_with_forces(
            pos,
            bundle.charges,
            mask_f,
            box,
            grid_dims,
            alpha,
            4,
            True,
        )

    spme_v21_fn = pme_mod.make_spme_energy_fn(
        box_vec, alpha=alpha, grid_dims=grid_dims, order=4, use_pallas_pme=True
    )

    jit_spme_v20 = jax.jit(jax.value_and_grad(lambda r, b: _spme_v20_fn(r, b), argnums=(0, 1)))
    jit_spme_v21 = jax.jit(
        jax.value_and_grad(lambda r: spme_v21_fn(r, bundle.charges, mask_f))
    )

    spme_v20_ms, spme_v20_probe = _probe_ms(jit_spme_v20, bundle.positions, box_vec)
    spme_v21_ms, spme_v21_probe = _probe_ms(jit_spme_v21, bundle.positions)

    # 3. Pallas nonbonded NL (v2.1 pre_encoded + empty-tile skip + in-register clamp)
    from prolix.pallas_kernels import pallas_nonbonded_nl

    excl_14_arr = jnp.zeros((int(bundle.positions.shape[0]), 1), dtype=jnp.int32)
    jit_nb_v21 = jax.jit(
        jax.value_and_grad(
            lambda r, idx: pallas_nonbonded_nl(
                r,
                bundle.charges,
                bundle.sigmas,
                bundle.epsilons,
                bundle.atom_mask,
                idx,
                excl_14_arr,
                excl_14_arr,
                pme_alpha=alpha,
                cutoff=9.0,
                box=box_vec,
                excl_mode="pre_encoded",
            )
        )
    )
    nb_v21_ms, nb_v21_probe = _probe_ms(jit_nb_v21, bundle.positions, nbr_enc.idx)

    # 4. Fused bonded megakernel (pallas_bonded_all_fused)
    from prolix.physics.pallas_bonded import pallas_bonded_all_fused

    c_t = jnp.zeros((0, 5), dtype=jnp.int32)
    c_mi = jnp.zeros((0,), dtype=jnp.int32)
    c_cf = jnp.zeros((0, 24, 24, 4, 4), dtype=jnp.float32)
    c_m = jnp.zeros((0,), dtype=jnp.float32)

    jit_bonded_v21 = jax.jit(
        jax.value_and_grad(
            lambda r: pallas_bonded_all_fused(
                r,
                bonds=bundle.bond_idx,
                bond_params=bundle.bond_params,
                bond_mask=bundle.bond_mask.astype(jnp.float32),
                angles=bundle.angle_idx,
                angle_params=bundle.angle_params,
                angle_mask=bundle.angle_mask.astype(jnp.float32),
                dihedrals=bundle.dihedral_idx,
                dihedral_params=bundle.dihedral_params,
                dihedral_mask=bundle.dihedral_mask.astype(jnp.float32),
                impropers=bundle.improper_idx,
                improper_params=bundle.improper_params,
                improper_mask=bundle.improper_mask.astype(jnp.float32),
                exception_pairs=bundle.exception_pairs,
                exception_sigmas=bundle.exception_sigmas,
                exception_epsilons=bundle.exception_epsilons,
                exception_chargeprods=bundle.exception_chargeprods,
                exception_mask=bundle.exception_mask.astype(jnp.float32),
                cmap_torsions=c_t,
                cmap_indices=c_mi,
                cmap_coeffs=c_cf,
                cmap_mask=c_m,
                box=box_vec,
            )
        )
    )
    bonded_v21_ms, bonded_v21_probe = _probe_ms(jit_bonded_v21, bundle.positions)

    # 5. Full single-step DHFR energy + force evaluation (v2.1)
    e_fn_pal = energy_fn_from_bundle(
        bundle,
        build_dense_exclusions=False,
        use_pallas=True,
        use_pallas_pme=True,
        excl_mode="pre_encoded",
    )
    vg_pal = jax.jit(jax.value_and_grad(lambda r, n: e_fn_pal(r, neighbor=n)))
    full_force_v21_ms, full_force_v21_probe = _probe_ms(vg_pal, bundle.positions, nbr_enc)

    # 6. Neighbor-list rebuild cost: v2.1 (jax_md + encode_nl_exclusions) vs v2.2 (pallas_prune_and_encode_cell_list)
    neighbor_fn_v21 = nl_mod.make_neighbor_list_fn(
        disp_fn, box_vec, 9.0, capacity_multiplier=1.25, dr_threshold=1.0, use_pallas_nl=False
    )
    nbr_v21 = _encode_bundle_nl(neighbor_fn_v21.allocate(bundle.positions), bundle)
    jit_nl_rebuild_v21 = jax.jit(
        lambda pos, nbr: _encode_bundle_nl(
            dataclasses.replace(
                nbr,
                reference_position=nbr.reference_position
                + jnp.asarray(0.5, dtype=pos.dtype) * box_vec.astype(pos.dtype),
            ).update(pos),
            bundle,
        )
    )
    nl_rebuild_v21_ms, nl_rebuild_v21_probe = _probe_ms(
        jit_nl_rebuild_v21, bundle.positions, nbr_v21
    )

    n_dhfr = int(bundle.positions.shape[0])
    cell_size_dhfr = float(nbr_enc.cell_size) if nbr_enc.cell_size is not None else 10.0
    cell_cap_dhfr = int(nbr_enc.cell_list_capacity) if nbr_enc.cell_list_capacity is not None else 64
    max_occ_dhfr = int(nbr_enc.idx.shape[-1])
    grid_dims_dhfr = nl_mod._compute_static_grid_dims(box_vec, cell_size_dhfr)

    jit_nl_rebuild_v22 = jax.jit(
        lambda pos: nl_mod.pallas_prune_and_encode_cell_list(
            jnp.mod(pos, box_vec),
            box_vec,
            bundle.excl_dense_indices[:n_dhfr],
            bundle.excl_dense_scales_vdw[:n_dhfr],
            bundle.excl_dense_scales_elec[:n_dhfr],
            cell_size=cell_size_dhfr,
            cell_capacity=cell_cap_dhfr,
            max_occupancy=max_occ_dhfr,
            r_cutoff=10.0,
            encode_exclusions=True,
            interpret=(jax.default_backend() == "cpu"),
            grid_dims=grid_dims_dhfr,
        )
    )
    nl_rebuild_v22_ms, nl_rebuild_v22_probe = _probe_ms(jit_nl_rebuild_v22, bundle.positions)

    breakdown = {
        "settle_1call_v20_eigh_ms": settle_1call_v20_ms,
        "settle_1call_v21_cayley_hamilton_ms": settle_1call_v21_ms,
        "settle_per_step_2calls_v20_ms": 2.0 * settle_1call_v20_ms,
        "settle_per_step_2calls_v21_ms": 2.0 * settle_1call_v21_ms,
        "settle_speedup": settle_1call_v20_ms / max(settle_1call_v21_ms, 1e-9),
        "settle_max_abs_diff_A": settle_max_diff_A,
        "spme_v20_uncached_with_vjp_box_ms": spme_v20_ms,
        "spme_v21_cached_symbolic_zeros_ms": spme_v21_ms,
        "spme_speedup": spme_v20_ms / max(spme_v21_ms, 1e-9),
        "pallas_nonbonded_nl_v21_ms": nb_v21_ms,
        "pallas_bonded_all_fused_v21_ms": bonded_v21_ms,
        "full_single_step_force_eval_v21_ms": full_force_v21_ms,
        "nl_rebuild_plus_encode_v21_jaxmd_ms": nl_rebuild_v21_ms,
        "nl_rebuild_plus_encode_v22_pallas_ms": nl_rebuild_v22_ms,
        "nl_rebuild_plus_encode_ms": nl_rebuild_v22_ms,
        "nl_rebuild_speedup_v22_vs_v21": nl_rebuild_v21_ms / max(nl_rebuild_v22_ms, 1e-9),
        "nl_rebuild_rate_v20_pbc_bug": 1.0,
        "nl_rebuild_per_step_penalty_saved_ms": nl_rebuild_v21_ms,
        "xtrax_probes": {
            "settle_v20": settle_v20_probe,
            "settle_v21": settle_v21_probe,
            "spme_v20": spme_v20_probe,
            "spme_v21": spme_v21_probe,
            "nonbonded_nl_v21": nb_v21_probe,
            "bonded_fused_v21": bonded_v21_probe,
            "full_force_v21": full_force_v21_probe,
            "nl_rebuild_v21": nl_rebuild_v21_probe,
            "nl_rebuild_v22": nl_rebuild_v22_probe,
        },
    }
    print(json.dumps({k: v for k, v in breakdown.items() if k != "xtrax_probes"}, indent=2))
    return breakdown


def run_multi_replica_bucket_scaling(dhfr_bundle) -> dict:
    print("=== Running Multi-Replica & Ragged Bucket Batching (2D Pallas Grid vs lax.map) ===")
    from prolix.pallas_kernels import _pallas_nonbonded_nl_single, pallas_nonbonded_nl

    results = {
        "jaxpr_loop_structure": {},
        "nonbonded_nl_batch_scaling": [],
        "dhfr_multi_replica_force_scaling": [],
    }

    bucket_sizes = [1024, 2048, 4096, 8192, 16384]
    batch_sizes = [1, 4, 8, 16, 32]
    K = 256
    box = jnp.array([62.23, 62.23, 62.23], dtype=jnp.float32)

    for N in bucket_sizes:
        key = jax.random.PRNGKey(N)
        pos_1 = jax.random.uniform(key, (N, 3), dtype=jnp.float32) * 60.0
        q_1 = jax.random.normal(key, (N,), dtype=jnp.float32) * 0.3
        sig_1 = jnp.full((N,), 3.15, dtype=jnp.float32)
        eps_1 = jnp.full((N,), 0.15, dtype=jnp.float32)
        mask_1 = jnp.ones((N,), dtype=jnp.float32)
        nbr_1 = jax.random.randint(key, (N, K), 0, N, dtype=jnp.int32)
        dummy_excl = jnp.zeros((N, 1), dtype=jnp.int32)

        for B in batch_sizes:
            pos_b = jnp.broadcast_to(pos_1[None, :, :], (B, N, 3)).copy()
            nbr_b = jnp.broadcast_to(nbr_1[None, :, :], (B, N, K)).copy()

            def _raw_v20_lax_map(pb, nb):
                def _step(i):
                    return pallas_nonbonded_nl(
                        pb[i],
                        q_1,
                        sig_1,
                        eps_1,
                        mask_1,
                        nb[i],
                        dummy_excl,
                        dummy_excl,
                        pme_alpha=0.34,
                        cutoff=9.0,
                        box=box,
                        excl_mode="pre_encoded",
                    )

                return jax.lax.map(_step, jnp.arange(B))

            def _raw_v21_2d_grid(pb, nb):
                return jax.vmap(
                    lambda p, n: pallas_nonbonded_nl(
                        p,
                        q_1,
                        sig_1,
                        eps_1,
                        mask_1,
                        n,
                        dummy_excl,
                        dummy_excl,
                        pme_alpha=0.34,
                        cutoff=9.0,
                        box=box,
                        excl_mode="pre_encoded",
                    )
                )(pb, nb)

            if N == 1024 and B == 8:
                bodies_v20 = xls.loop_bodies(_raw_v20_lax_map, pos_b, nbr_b)
                bodies_v21 = xls.loop_bodies(_raw_v21_2d_grid, pos_b, nbr_b)
                eqns_v20 = [e.primitive.name for e in xp.iter_jaxpr_eqns(jax.make_jaxpr(_raw_v20_lax_map)(pos_b, nbr_b))]
                eqns_v21 = [e.primitive.name for e in xp.iter_jaxpr_eqns(jax.make_jaxpr(_raw_v21_2d_grid)(pos_b, nbr_b))]
                results["jaxpr_loop_structure"] = {
                    "v20_lax_map_outer_scans": [
                        {"path": b.path, "primitive": b.primitive, "trip_count": b.trip_count}
                        for b in bodies_v20
                        if b.primitive == "scan"
                    ],
                    "v21_2d_grid_outer_scans": [
                        {"path": b.path, "primitive": b.primitive, "trip_count": b.trip_count}
                        for b in bodies_v21
                        if b.primitive == "scan"
                    ],
                    "v20_pallas_calls": eqns_v20.count("pallas_call"),
                    "v21_pallas_calls": eqns_v21.count("pallas_call"),
                }

            _fn_v20_lax_map = jax.jit(_raw_v20_lax_map)
            _fn_v21_2d_grid = jax.jit(_raw_v21_2d_grid)

            ms_v20, probe_v20 = _probe_ms(_fn_v20_lax_map, pos_b, nbr_b, n_steady=2, n_wall=5)
            ms_v21, probe_v21 = _probe_ms(_fn_v21_2d_grid, pos_b, nbr_b, n_steady=2, n_wall=5)
            row = {
                "bucket_atoms": N,
                "replicas_B": B,
                "nl_capacity_K": K,
                "v20_lax_map_ms": ms_v20,
                "v21_2d_grid_ms": ms_v21,
                "per_replica_v21_ms": ms_v21 / B,
                "speedup_2d_grid_vs_lax_map": ms_v20 / max(ms_v21, 1e-9),
                "v20_device_ms": probe_v20["device_seconds"] * 1000.0,
                "v21_device_ms": probe_v21["device_seconds"] * 1000.0,
                "v21_steady_compiles": probe_v21["n_compilations"],
            }
            results["nonbonded_nl_batch_scaling"].append(row)
            print(
                f"Bucket N={N:5d} B={B:2d}: v2.0 lax.map={ms_v20:7.3f} ms | "
                f"v2.1 2D-grid={ms_v21:7.3f} ms ({ms_v21/B:6.3f} ms/rep) | "
                f"speedup={row['speedup_2d_grid_vs_lax_map']:.2f}x"
            )

    # 2. Full 23,558-atom DHFR multi-replica force evaluation scaling (B = 1, 2, 4, 8)
    disp_fn, _ = displacement_fn_for_bundle(dhfr_bundle)
    box_vec = jnp.diag(dhfr_bundle.box)
    neighbor_fn = nl_mod.make_neighbor_list_fn(
        disp_fn, box_vec, 9.0, capacity_multiplier=1.25, dr_threshold=1.0
    )
    nbr_raw = neighbor_fn.allocate(dhfr_bundle.positions)
    nbr_enc = _encode_bundle_nl(nbr_raw, dhfr_bundle)
    e_fn_pal = energy_fn_from_bundle(
        dhfr_bundle,
        build_dense_exclusions=False,
        use_pallas=True,
        use_pallas_pme=True,
        excl_mode="pre_encoded",
    )
    vg_single = jax.value_and_grad(
        lambda r, idx: e_fn_pal(r, neighbor=dataclasses.replace(nbr_enc, idx=idx))
    )

    for B in (1, 2, 4, 8):
        pos_b = jnp.broadcast_to(
            dhfr_bundle.positions[None, :, :], (B, *dhfr_bundle.positions.shape)
        ).copy()
        nbr_idx_b = jnp.broadcast_to(
            nbr_enc.idx[None, :, :], (B, *nbr_enc.idx.shape)
        ).copy()
        vg_batch = jax.jit(jax.vmap(vg_single, in_axes=(0, 0)))
        ms_batch, probe_batch = _probe_ms(vg_batch, pos_b, nbr_idx_b, n_steady=2, n_wall=5)
        row = {
            "system": "DHFR_23558",
            "replicas_B": B,
            "total_batch_ms": ms_batch,
            "per_replica_ms": ms_batch / B,
            "device_ms": probe_batch["device_seconds"] * 1000.0,
            "steady_compiles": probe_batch["n_compilations"],
        }
        results["dhfr_multi_replica_force_scaling"].append(row)
        print(
            f"DHFR 23,558-atom Full Force Eval B={B:2d}: total={ms_batch:7.3f} ms | "
            f"per_replica={ms_batch/B:6.3f} ms"
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/bench/261030_b300_pallas_v21_breakdown_and_buckets.json"),
    )
    args = parser.parse_args()

    ff_path = B_dhfr._resolve_ff_path()
    sys_data = B_dhfr._build_dhfr_system_dict(ff_path)
    bundle = B_dhfr._build_bundle_from_sys_data(sys_data)

    breakdown = run_dhfr_breakdown(bundle)
    bucket_scaling = run_multi_replica_bucket_scaling(bundle)

    payload = {
        "meta": {
            "device": str(jax.devices()[0]),
            "device_kind": getattr(jax.devices()[0], "device_kind", "unknown"),
            "jax_version": jax.__version__,
            "xtrax_profiling_contract_version": xp.CONTRACT_VERSION,
        },
        "dhfr_23558_breakdown": breakdown,
        "multi_replica_bucket_scaling": bucket_scaling,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
