# ruff: noqa: E402
"""Pallas v3.5 Release Benchmark Ladder (`MK-65..74`): 6-Suite Multi-GPU Harness.

Implements the complete Pallas v3.5 release benchmark matrix with mandatory OpenMM `CUDA`
Anti-Fallback verification (`require_openmm_cuda_context`), multi-process `CUDA MPS` baselines
(`run_openmm_cuda_mps_replicas`), `xtrax.profiling.probe_unit` sub-step attribution, cold JIT
compile latency & XLA compile-count tracking, and GPU VRAM scaling:

- Suite A & B (`A`, `B`) — Official OpenMM `benchmark.py` + 8-Step Solvated/Holo Ladder (`2k -> 408k`):
  `5DFR_gbsa_implicit` (`MK-67`), `1UAO_2k`, `1VII_4k`, `1UBQ_7k`, `Holo_Lysozyme_15k`,
  `DHFR_24k` (`rf`, `60^3` PME, `64^3` PME, `hbonds`), `Holo_DHFR_24k`, `1AKE_45k`,
  `ApoA1_92k` (`rf`, `pme`), and ultra-large **`Cellulose_408k_pme` (`408,609` atoms)** (`MK-71`),
  measured in BOTH `single` and `mixed` precision (`MK-66`) alongside Cold JIT Compile Latency (s).
- Suite C (`C`) — Heterogeneous Congeneric (`MK-64/65`) & Cross-Size Ragged Queue (`MK-68`):
  1. `Holo-DHFR Congeneric Screen` (`B` ligand analogues, `stride=0` shared topology, `single` & `mixed`).
  2. `Cross-Size Ragged Multi-Target Screen` (`[1UAO_2k..Holo_DHFR_24k]`), verifying **1 XLA compile**
     across all 5 distinct molecular sizes + PME tier-mode ablation (`auto`, `grouped`, `uniform`).
- Suite D (`D`) — Frontend (`ProlixSimulation` / `ProlixBatchSimulation` vs `EnsemblePlan`) & `NPT` vs `NVT` (`MK-69`, `MK-74`):
  1. `ProlixSimulation` & `ProlixContext` (`single` & `mixed`) vs native `EnsemblePlan` vs OpenMM `CUDA`.
  2. `ProlixBatchSimulation` & `EnsemblePlan.from_openmm_systems` (`B = 1, 4, 8, 16`) vs `CUDA + MPS`.
  3. Dynamic-box `NPT` (`HBonds` + `MonteCarloBarostat` / `SCR`) vs `NVT` zero-sync throughput (`MK-74`).
- Suite E (`E`) — High-Batch `CUDA MPS` OOM-Cliff & Marginal VRAM Sweep (`B = 1, 4, 8, 16, 32, 64, 128`):
  Sweeps `1UAO_2k`, `1UBQ_7k`, and `DHFR_24k` up to `B = 128`, measuring Prolix peak & marginal
  VRAM/replica (`MB/rep`) vs OpenMM `CUDA + MPS` up to and beyond the `CUDA MPS` OOM/context cliff.
- Suite F (`F`) — Differentiable Moat (`jax.value_and_grad` `2k -> 92k` Atoms, `MK-73`):
  End-to-end forward + backward (`dE/dr, dE/dq, dE/dsigma, dE/depsilon`) ms/eval and peak VRAM (`MB`)
  across `5DFR_gbsa_implicit` (`2.5k`), `1UAO_2k`, `1UBQ_7k`, `Holo_Lysozyme_15k`, `DHFR_24k`,
  `1AKE_45k`, and `ApoA1_92k` (`O(N*K)` tile-pair custom VJPs — impossible in OpenMM `CUDA`).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

# Disable JAX 75% upfront VRAM preallocation so:
# (1) `dev.memory_stats()` reports exact live & peak Prolix GPU memory per batch B, and
# (2) OpenMM `CUDA + MPS` subprocesses receive full fair access to free GPU VRAM.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import prolix.physics.neighbor_list as nl_mod
import scripts.experiments.bench_dhfr_parity as b_dhfr
from prolix.api.bundle_md import displacement_fn_for_bundle
from prolix.api.ensemble_plan import (
    _SINGLE_INFERENCE_CACHE,
    _STACKED_INFERENCE_CACHE,
    _STACKED_JIT_FN_CACHE,
    EnsemblePlan,
    _encode_bundle_nl,
)
from prolix.openmm_xml_importer import parse_openmm_system_xml
from prolix.pallas_kernels import (
    build_tile_pair_neighbor_list,
    pallas_bonded_all_fused,
    pallas_nonbonded_tile_pairs,
)
from prolix.physics.neighbor_list import pair_list_to_dense_padded
from prolix.physics.pallas_gbsa_tile import (
    build_gbsa_tile_pair_list,
    pallas_gbsa_tile,
)
from prolix.physics.pme import (
    compute_openmm_pme_params,
    snap_fft_dim,
    spme_energy_with_forces,
)
from prolix.physics.system import (
    _mask,
    _pad_1d,
    _pad_2d,
    _pad_positions_with_ghost_lattice,
)
from prolix.types.bundles import MolecularBundle, MolecularShapeSpec
from scripts.benchmarks.bench_e2e_ensemble_buckets import (
    _cast_bundle_precision,
    _make_truncated_solvated_bundle,
)
from scripts.benchmarks.bench_v33_diverse_panel import (
    _ns_per_day,
    _re_pad_bundle_to_custom_bucket,
    build_apoa1_bundle,
    build_implicit_protein_bundle,
    build_solvated_1ake_bundle,
)
from scripts.benchmarks.bench_v34_system_ladder import (
    _backfill_provenance_in_record,
    _build_openmm_explicit_system,
    _profile_explicit_substeps,
    _time_openmm_cuda_single_and_mps,
    build_holo_protein_ligand_bundle,
)
from scripts.benchmarks.openmm_cuda_guard import (
    get_openmm_platform_inventory,
    require_openmm_cuda_context,
    run_openmm_cuda_mps_replicas,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_CONFIG: dict[str, bool] = {"allow_opencl_fallback": False}


def _clear_all_caches(*, clear_jax: bool = False) -> None:
    _SINGLE_INFERENCE_CACHE.clear()
    _STACKED_INFERENCE_CACHE.clear()
    _STACKED_JIT_FN_CACHE.clear()
    if clear_jax:
        jax.clear_caches()
    gc.collect()


def _jax_mem_mb() -> dict[str, float]:
    """Query current and peak GPU allocator usage in MiB from JAX device 0."""
    with contextlib.suppress(Exception):
        stats = jax.devices()[0].memory_stats()
        if stats:
            return {
                "bytes_in_use_mb": float(stats.get("bytes_in_use", 0)) / (1024.0 * 1024.0),
                "peak_bytes_in_use_mb": float(stats.get("peak_bytes_in_use", 0)) / (1024.0 * 1024.0),
            }
    return {"bytes_in_use_mb": 0.0, "peak_bytes_in_use_mb": 0.0}


def _filter_supported_kwargs(callable_obj: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Pass optional v3.5 kwargs (`nonbonded_engine`, `precision`, `pme_tier_mode`) if accepted."""
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _block_trajs(trajs: Any) -> None:
    if isinstance(trajs, (list, tuple)):
        for t in trajs:
            jax.block_until_ready(t.positions if hasattr(t, "positions") else t)
    elif hasattr(trajs, "positions"):
        jax.block_until_ready(trajs.positions)
    else:
        jax.block_until_ready(trajs)


def _make_plan_and_run_timed(
    bundles: list[MolecularBundle],
    *,
    n_warmup_steps: int,
    n_steps: int,
    dt_fs: float = 2.0,
    kT: float = 0.596,
    gamma_ps: float = 10.0,
    use_neighbor_list: bool = True,
    nl_update_every: int = 20,
    use_pallas: bool = True,
    use_pallas_pme: bool = True,
    nonbonded_engine: str | None = None,
    precision: str | None = None,
    pme_tier_mode: str | None = None,
    target_pressure_atm: float | None = None,
    barostat_every: int = 25,
) -> dict[str, Any]:
    """Construct `EnsemblePlan`, measure cold JIT compile + warmup wall time, and time steady `n_steps`."""
    _clear_all_caches()
    fb_kwargs: dict[str, Any] = {"use_pallas": use_pallas}
    if use_neighbor_list:
        fb_kwargs["use_pallas_pme"] = use_pallas_pme
    if nonbonded_engine is not None:
        fb_kwargs["nonbonded_engine"] = nonbonded_engine
    if precision is not None:
        fb_kwargs["precision"] = precision
    if pme_tier_mode is not None:
        fb_kwargs["pme_tier_mode"] = pme_tier_mode

    t_compile0 = time.perf_counter()
    plan = EnsemblePlan.from_bundles(
        bundles, **_filter_supported_kwargs(EnsemblePlan.from_bundles, fb_kwargs)
    )

    run_kwargs: dict[str, Any] = {
        "dt": dt_fs,
        "kT": kT,
        "gamma": gamma_ps,
        "run_mode": "inference",
        "use_pallas": use_pallas,
    }
    if use_neighbor_list:
        run_kwargs["use_neighbor_list"] = True
        run_kwargs["nl_update_every"] = nl_update_every
        run_kwargs["use_pallas_pme"] = use_pallas_pme
    if nonbonded_engine is not None:
        run_kwargs["nonbonded_engine"] = nonbonded_engine
    if precision is not None:
        run_kwargs["precision"] = precision
    if pme_tier_mode is not None:
        run_kwargs["pme_tier_mode"] = pme_tier_mode
    if target_pressure_atm is not None:
        run_kwargs["target_pressure"] = target_pressure_atm
        run_kwargs["pressure_atm"] = target_pressure_atm
        run_kwargs["barostat_interval"] = barostat_every
        run_kwargs["barostat_every"] = barostat_every
    filtered_run_kwargs = _filter_supported_kwargs(plan.run, run_kwargs)

    w_out = plan.run(n_steps=n_warmup_steps, seed=1, **filtered_run_kwargs)
    _block_trajs(w_out)
    cold_compile_s = time.perf_counter() - t_compile0
    jit_cache_entries = len(_SINGLE_INFERENCE_CACHE) + len(_STACKED_INFERENCE_CACHE)
    jit_fn_entries = len(_STACKED_JIT_FN_CACHE)

    t0 = time.perf_counter()
    out = plan.run(n_steps=n_steps, seed=2, **filtered_run_kwargs)
    _block_trajs(out)
    ms_batch = ((time.perf_counter() - t0) / n_steps) * 1000.0
    ms_rep = ms_batch / max(1, len(bundles))
    mem = _jax_mem_mb()
    return {
        "ms_batch": float(ms_batch),
        "ms_rep": float(ms_rep),
        "cold_compile_s": float(cold_compile_s),
        "xla_executable_count": max(1, int(jit_fn_entries or jit_cache_entries)),
        "bytes_in_use_mb": mem["bytes_in_use_mb"],
        "peak_bytes_in_use_mb": mem["peak_bytes_in_use_mb"],
    }


def _pad_1d_const(
    arr: Any, bucket: int, *, pad_value: float = 1.0, dtype: Any = jnp.float32
) -> jax.Array:
    """Pad a 1D array to ``bucket`` length using a constant ``pad_value``."""
    a1 = jnp.asarray(arr, dtype=dtype).reshape(-1)
    n_cur = int(a1.shape[0])
    if n_cur >= bucket:
        return a1[:bucket]
    return jnp.pad(a1, (0, bucket - n_cur), constant_values=pad_value).astype(dtype)


# ============================================================================
# 408,609-Atom `Cellulose_408k_pme` Builder (Official Amber20 or Periodic 408k)
# ============================================================================
def build_cellulose_408k_bundle(
    bucket_atoms: int = 409600,
    bucket_waters: int = 122880,
) -> tuple[MolecularBundle, Any, Any, str]:
    """Build the 408,609-atom ultra-large PME benchmark system (`Cellulose_408k_pme`).

    Prefers official `Amber20_Benchmark_Suite/PME/Topologies/Cellulose.prmtop` +
    `Cellulose.inpcrd` (`408,609` atoms) if cached locally or reachable; otherwise constructs a
    strictly matched `408,609`-atom periodic `ApoA1` 2x2x1 + TIP3P solvated assembly (`192x192x80`
    PME grid) for both Prolix and OpenMM `CUDA` so air-gapped compute nodes never fail.
    """
    import openmm as mm
    from openmm import app
    from openmm import unit as _omm_unit

    unit = cast("Any", _omm_unit)

    cand_dirs = [
        ROOT / "Amber20_Benchmark_Suite",
        ROOT / "data" / "Amber20_Benchmark_Suite",
        Path("/orcd/pool/008/so3_shared/marielle/Amber20_Benchmark_Suite"),
    ]
    prmtop_path: Path | None = None
    inpcrd_path: Path | None = None
    for d in cand_dirs:
        p_top = d / "PME" / "Topologies" / "Cellulose.prmtop"
        p_crd = d / "PME" / "Coordinates" / "Cellulose.inpcrd"
        if p_top.exists() and p_crd.exists():
            prmtop_path, inpcrd_path = p_top, p_crd
            break

    if prmtop_path is not None and inpcrd_path is not None:
        prmtop = app.AmberPrmtopFile(str(prmtop_path))
        inpcrd = app.AmberInpcrdFile(str(inpcrd_path))
        omm_sys = prmtop.createSystem(
            nonbondedMethod=app.PME,
            nonbondedCutoff=0.9 * unit.nanometer,
            constraints=app.HBonds,
            rigidWater=True,
        )
        if inpcrd.boxVectors is not None:
            omm_sys.setDefaultPeriodicBoxVectors(*inpcrd.boxVectors)
        for f in omm_sys.getForces():
            if isinstance(f, mm.NonbondedForce):
                f.setPMEParameters(0.34 * 10.0, 192, 192, 96)
        pos_nm = np.asarray(inpcrd.positions.value_in_unit(unit.nanometer), dtype=np.float32)
        pos_A = pos_nm * 10.0
        xml_str = mm.XmlSerializer.serialize(omm_sys)
        data = parse_openmm_system_xml(xml_str, positions=pos_A)
        n = int(data.n_atoms)
        a = max(bucket_atoms, ((n + 255) // 256) * 256)
        wb = max(bucket_waters, int(data.water_indices.shape[0]) + 256)
        ne = int(data.excl_indices.shape[0])
        eb = max(ne, 1)
        excl_idx, excl_vdw, excl_elec = pair_list_to_dense_padded(
            jnp.asarray(data.excl_indices, dtype=jnp.int32),
            jnp.asarray(data.excl_scales_vdw, dtype=jnp.float32),
            jnp.asarray(data.excl_scales_elec, dtype=jnp.float32),
            jnp.ones(ne, dtype=bool),
            a,
            max_exclusions=32,
        )
        water_set = set(np.asarray(data.water_indices).reshape(-1).tolist())
        b_np = np.asarray(data.bonds)
        b_keep = np.array([int(r[0]) not in water_set and int(r[1]) not in water_set for r in b_np], dtype=bool)
        bonds_f = jnp.asarray(b_np[b_keep], dtype=jnp.int32)
        bpar_f = jnp.asarray(np.asarray(data.bond_params)[b_keep], dtype=jnp.float32)
        a_np = np.asarray(data.angles)
        a_keep = np.array(
            [int(r[0]) not in water_set and int(r[1]) not in water_set and int(r[2]) not in water_set for r in a_np],
            dtype=bool,
        )
        angles_f = jnp.asarray(a_np[a_keep], dtype=jnp.int32)
        apar_f = jnp.asarray(np.asarray(data.angle_params)[a_keep], dtype=jnp.float32)
        box_vec = jnp.diag(data.box).astype(jnp.float32)
        pos_real = jnp.mod(jnp.asarray(data.positions, dtype=jnp.float32), box_vec[None, :])
        nb = max(int(bonds_f.shape[0]), 1)
        na = max(int(angles_f.shape[0]), 1)
        nd = max(int(data.dihedrals.shape[0]), 1)
        nw = int(data.water_indices.shape[0])
        nx = max(int(data.exception_pairs.shape[0]), 1)
        d_cols = int(data.dihedral_params.shape[1]) if data.dihedral_params.ndim == 2 else 3
        spec = MolecularShapeSpec(
            atom_bucket_idx=a,
            bond_bucket_idx=nb,
            angle_bucket_idx=na,
            dihedral_bucket_idx=nd,
            water_bucket_idx=wb,
            excl_bucket_idx=eb,
            cmap_bucket_idx=1,
            exception_bucket_idx=nx,
            has_pbc=True,
            has_implicit_solvent=False,
            boundary_condition="periodic",
            has_real_water=nw > 0,
            use_size_buckets=False,
            exact_n_atoms=a,
            exact_n_waters=wb,
            cmap_map_bucket_idx=0,
            constraint_bucket_idx=0,
            has_real_constraints=False,
            is_triclinic=False,
            has_real_cosolvents=False,
        )
        bun: MolecularBundle = cast("Any", MolecularBundle)(
            positions=_pad_positions_with_ghost_lattice(pos_real, a, box_vec, dtype=jnp.float32),
            masses=_pad_1d_const(data.masses, a, pad_value=1.0, dtype=jnp.float32),
            charges=_pad_1d(data.charges, a, dtype=jnp.float32),
            sigmas=_pad_1d_const(data.sigmas, a, pad_value=1.0, dtype=jnp.float32),
            epsilons=_pad_1d(data.epsilons, a, dtype=jnp.float32),
            radii=jnp.ones(a, dtype=jnp.float32),
            scaled_radii=jnp.ones(a, dtype=jnp.float32),
            atom_mask=_mask(n, a),
            n_atoms=jnp.array(n, dtype=jnp.int32),
            box=data.box.astype(jnp.float32),
            bond_idx=_pad_2d(bonds_f, nb, 2, dtype=jnp.int32),
            bond_params=_pad_2d(bpar_f, nb, 2, dtype=jnp.float32),
            bond_mask=_mask(int(bonds_f.shape[0]), nb),
            n_bonds=jnp.array(int(bonds_f.shape[0]), dtype=jnp.int32),
            angle_idx=_pad_2d(angles_f, na, 3, dtype=jnp.int32),
            angle_params=_pad_2d(apar_f, na, 2, dtype=jnp.float32),
            angle_mask=_mask(int(angles_f.shape[0]), na),
            n_angles=jnp.array(int(angles_f.shape[0]), dtype=jnp.int32),
            dihedral_idx=_pad_2d(data.dihedrals, nd, 4, dtype=jnp.int32),
            dihedral_params=_pad_2d(data.dihedral_params, nd, d_cols, dtype=jnp.float32),
            dihedral_mask=_mask(int(data.dihedrals.shape[0]), nd),
            n_dihedrals=jnp.array(int(data.dihedrals.shape[0]), dtype=jnp.int32),
            improper_idx=jnp.zeros((1, 4), dtype=jnp.int32),
            improper_params=jnp.zeros((1, 3), dtype=jnp.float32),
            improper_mask=jnp.zeros((1,), dtype=bool),
            improper_is_periodic=jnp.array(False, dtype=bool),
            n_impropers=jnp.array(0, dtype=jnp.int32),
            urey_bradley_idx=jnp.zeros((na, 3), dtype=jnp.int32),
            urey_bradley_params=jnp.zeros((na, 2), dtype=jnp.float32),
            urey_bradley_mask=jnp.zeros((na,), dtype=bool),
            n_urey_bradley=jnp.array(0, dtype=jnp.int32),
            cmap_torsion_idx=jnp.zeros((1, 8), dtype=jnp.int32),
            cmap_energy_grids=jnp.zeros((1, 24, 24), dtype=jnp.float32),
            cmap_mask=jnp.zeros((1,), dtype=bool),
            n_cmap=jnp.array(0, dtype=jnp.int32),
            water_indices=_pad_2d(data.water_indices, wb, 3, dtype=jnp.int32),
            water_mask=_mask(nw, wb),
            n_waters=jnp.array(nw, dtype=jnp.int32),
            excl_indices=_pad_2d(data.excl_indices, eb, 2, dtype=jnp.int32),
            excl_scales_vdw=_pad_1d(data.excl_scales_vdw, eb, dtype=jnp.float32),
            excl_scales_elec=_pad_1d(data.excl_scales_elec, eb, dtype=jnp.float32),
            excl_mask=_mask(ne, eb),
            n_excl=jnp.array(ne, dtype=jnp.int32),
            exception_pairs=_pad_2d(data.exception_pairs, nx, 2, dtype=jnp.int32),
            exception_sigmas=_pad_1d_const(data.exception_sigmas, nx, pad_value=1.0, dtype=jnp.float32),
            exception_epsilons=_pad_1d(data.exception_epsilons, nx, dtype=jnp.float32),
            exception_chargeprods=_pad_1d(data.exception_chargeprods, nx, dtype=jnp.float32),
            exception_mask=_mask(int(data.exception_pairs.shape[0]), nx),
            n_exception_pairs=jnp.array(int(data.exception_pairs.shape[0]), dtype=jnp.int32),
            pme_alpha=jnp.array(0.34, dtype=jnp.float32),
            cutoff_distance=jnp.array(9.0, dtype=jnp.float32),
            shape_spec=spec,
            excl_dense_indices=excl_idx,
            excl_dense_scales_vdw=excl_vdw,
            excl_dense_scales_elec=excl_elec,
            cmap_indices=jnp.zeros((1,), dtype=jnp.int32),
            cmap_coeffs=data.cmap_coeffs,
            constraint_pairs=jnp.zeros((0, 2), dtype=jnp.int32),
            constraint_lengths=jnp.zeros((0,), dtype=jnp.float32),
            constraint_mask=jnp.zeros((0,), dtype=bool),
            n_constraints=jnp.array(0, dtype=jnp.int32),
            water_r_OH=jnp.array(0.9572, dtype=jnp.float32),
            water_r_HH=jnp.array(1.5139, dtype=jnp.float32),
        )
        return bun, omm_sys, inpcrd.positions, "amber20_cellulose_prmtop"

    # Air-gapped fallback: 2x2x1 periodic ApoA1 + solvated slab = exact 408,609 real atoms
    apoa1_bun, _, _ = build_apoa1_bundle(
        bucket_atoms=100000, bucket_waters=24576, electrostatic_method="pme", include_constraints=False
    )
    n0 = int(apoa1_bun.n_atoms)  # 92,224
    box0 = np.diag(np.asarray(apoa1_bun.box, dtype=np.float32))  # [108.86, 108.86, 77.76]
    pos0 = np.asarray(apoa1_bun.positions[:n0], dtype=np.float32)
    q0 = np.asarray(apoa1_bun.charges[:n0], dtype=np.float32)
    sig0 = np.asarray(apoa1_bun.sigmas[:n0], dtype=np.float32)
    eps0 = np.asarray(apoa1_bun.epsilons[:n0], dtype=np.float32)
    m0 = np.asarray(apoa1_bun.masses[:n0], dtype=np.float32)

    shifts = [
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([box0[0], 0.0, 0.0], dtype=np.float32),
        np.array([0.0, box0[1], 0.0], dtype=np.float32),
        np.array([box0[0], box0[1], 0.0], dtype=np.float32),
    ]
    pos_4 = np.concatenate([pos0 + sh[None, :] for sh in shifts], axis=0)  # 368,896
    q_4 = np.tile(q0, 4)
    sig_4 = np.tile(sig0, 4)
    eps_4 = np.tile(eps0, 4)
    m_4 = np.tile(m0, 4)

    # Add 13,237 rigid TIP3P waters (39,711 atoms) + 2 Na+/Cl- ions in z-slab [79.5, 92.5] -> exact 408,609 atoms!
    n_target = 408609
    n_rem = n_target - int(pos_4.shape[0])  # 39,713 = 13,237 * 3 + 2
    n_extra_w = n_rem // 3  # 13,237
    n_ions = n_rem - 3 * n_extra_w  # 2
    lx_new, ly_new, lz_new = float(2.0 * box0[0]), float(2.0 * box0[1]), 96.0
    xs = np.linspace(2.5, lx_new - 2.5, 64, dtype=np.float32)
    ys = np.linspace(2.5, ly_new - 2.5, 64, dtype=np.float32)
    zs = np.linspace(float(box0[2]) + 2.2, lz_new - 2.2, 4, dtype=np.float32)
    centers = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)

    w_pos_extra = np.zeros((n_extra_w * 3, 3), dtype=np.float32)
    oh = 0.9572
    ang = np.deg2rad(104.52)
    h1_off = np.array([oh, 0.0, 0.0], dtype=np.float32)
    h2_off = np.array([oh * np.cos(ang), oh * np.sin(ang), 0.0], dtype=np.float32)
    c_w = centers[:n_extra_w]
    w_pos_extra[0::3] = c_w
    w_pos_extra[1::3] = c_w + h1_off[None, :]
    w_pos_extra[2::3] = c_w + h2_off[None, :]

    ion_pos = centers[n_extra_w : n_extra_w + n_ions]
    pos_all = np.concatenate([pos_4, w_pos_extra, ion_pos], axis=0)
    if pos_all.shape[0] != n_target:
        raise RuntimeError(f"Expected {n_target} atoms, got {pos_all.shape[0]}")

    q_w = np.tile(np.array([-0.834, 0.417, 0.417], dtype=np.float32), n_extra_w)
    sig_w = np.tile(np.array([3.1507, 1.0, 1.0], dtype=np.float32), n_extra_w)
    eps_w = np.tile(np.array([0.1521, 0.0, 0.0], dtype=np.float32), n_extra_w)
    m_w = np.tile(np.array([15.999, 1.008, 1.008], dtype=np.float32), n_extra_w)
    q_ion = np.array([1.0, -1.0][:n_ions], dtype=np.float32)
    sig_ion = np.full((n_ions,), 2.43, dtype=np.float32)
    eps_ion = np.full((n_ions,), 0.087, dtype=np.float32)
    m_ion = np.array([22.99, 35.45][:n_ions], dtype=np.float32)

    q_all = np.concatenate([q_4, q_w, q_ion])
    sig_all = np.concatenate([sig_4, sig_w, sig_ion])
    eps_all = np.concatenate([eps_4, eps_w, eps_ion])
    m_all = np.concatenate([m_4, m_w, m_ion])

    def _tile_idx(idx_arr: Any, msk_arr: Any) -> tuple[np.ndarray, int]:
        arr_np = np.asarray(idx_arr, dtype=np.int32)[np.asarray(msk_arr, dtype=bool)]
        reps = [arr_np + c * n0 for c in range(4)]
        cat = np.concatenate(reps, axis=0)
        return cat, int(cat.shape[0])

    def _tile_par(par_arr: Any, msk_arr: Any) -> np.ndarray:
        arr_np = np.asarray(par_arr, dtype=np.float32)[np.asarray(msk_arr, dtype=bool)]
        return np.tile(arr_np, (4, 1)) if arr_np.ndim == 2 else np.tile(arr_np, 4)

    b_cat, nb_tot = _tile_idx(apoa1_bun.bond_idx, apoa1_bun.bond_mask)
    bp_cat = _tile_par(apoa1_bun.bond_params, apoa1_bun.bond_mask)
    a_cat, na_tot = _tile_idx(apoa1_bun.angle_idx, apoa1_bun.angle_mask)
    ap_cat = _tile_par(apoa1_bun.angle_params, apoa1_bun.angle_mask)
    d_cat, nd_tot = _tile_idx(apoa1_bun.dihedral_idx, apoa1_bun.dihedral_mask)
    dp_cat = _tile_par(apoa1_bun.dihedral_params, apoa1_bun.dihedral_mask)
    ep_cat, nx_tot = _tile_idx(apoa1_bun.exception_pairs, apoa1_bun.exception_mask)
    es_cat = _tile_par(apoa1_bun.exception_sigmas, apoa1_bun.exception_mask)
    ee_cat = _tile_par(apoa1_bun.exception_epsilons, apoa1_bun.exception_mask)
    eq_cat = _tile_par(apoa1_bun.exception_chargeprods, apoa1_bun.exception_mask)

    w0_np = np.asarray(apoa1_bun.water_indices, dtype=np.int32)[np.asarray(apoa1_bun.water_mask, dtype=bool)]
    w_4 = np.concatenate([w0_np + c * n0 for c in range(4)], axis=0)
    base_extra = int(pos_4.shape[0])
    w_extra_idx = (base_extra + np.arange(n_extra_w * 3, dtype=np.int32)).reshape(n_extra_w, 3)
    w_all = np.concatenate([w_4, w_extra_idx], axis=0)
    nw_tot = int(w_all.shape[0])

    # Replicate dense exclusion tables by shifting non-self entries by c * n0
    if (
        apoa1_bun.excl_dense_indices is None
        or apoa1_bun.excl_dense_scales_vdw is None
        or apoa1_bun.excl_dense_scales_elec is None
    ):
        raise RuntimeError("ApoA1 bundle missing excl_dense_* arrays")
    ed_0 = np.asarray(apoa1_bun.excl_dense_indices[:n0], dtype=np.int32)
    sv_0 = np.asarray(apoa1_bun.excl_dense_scales_vdw[:n0], dtype=np.float32)
    se_0 = np.asarray(apoa1_bun.excl_dense_scales_elec[:n0], dtype=np.float32)
    max_ex = int(ed_0.shape[1])
    ed_pad = np.full((bucket_atoms, max_ex), -1, dtype=np.int32)
    sv_pad = np.ones((bucket_atoms, max_ex), dtype=np.float32)
    se_pad = np.ones((bucket_atoms, max_ex), dtype=np.float32)
    for c in range(4):
        sl = slice(c * n0, (c + 1) * n0)
        valid_e = ed_0 >= 0
        ed_pad[sl] = np.where(valid_e, ed_0 + c * n0, -1)
        sv_pad[sl] = sv_0
        se_pad[sl] = se_0
    # Extra TIP3P water intra-molecular exclusions (O-H1, O-H2, H1-H2)
    for w_i in range(n_extra_w):
        ow, h1, h2 = int(w_extra_idx[w_i, 0]), int(w_extra_idx[w_i, 1]), int(w_extra_idx[w_i, 2])
        ed_pad[ow, :2] = [h1, h2]
        sv_pad[ow, :2] = 0.0
        se_pad[ow, :2] = 0.0
        ed_pad[h1, :2] = [ow, h2]
        sv_pad[h1, :2] = 0.0
        se_pad[h1, :2] = 0.0
        ed_pad[h2, :2] = [ow, h1]
        sv_pad[h2, :2] = 0.0
        se_pad[h2, :2] = 0.0

    box_vec_408k = jnp.array([lx_new, ly_new, lz_new], dtype=jnp.float32)
    box_mat = jnp.diag(box_vec_408k)
    bb = ((nb_tot + 31) // 32) * 32
    ab = ((na_tot + 31) // 32) * 32
    db = ((nd_tot + 31) // 32) * 32
    xb = ((nx_tot + 31) // 32) * 32
    wb_pad = max(bucket_waters, ((nw_tot + 255) // 256) * 256)
    dp_cols = int(dp_cat.shape[1]) if dp_cat.ndim == 2 else 3

    spec_408k = dataclasses.replace(
        apoa1_bun.shape_spec,
        atom_bucket_idx=bucket_atoms,
        bond_bucket_idx=bb,
        angle_bucket_idx=ab,
        dihedral_bucket_idx=db,
        water_bucket_idx=wb_pad,
        excl_bucket_idx=1,
        cmap_bucket_idx=1,
        exception_bucket_idx=xb,
        has_pbc=True,
        has_implicit_solvent=False,
        has_real_water=nw_tot > 0,
        use_size_buckets=False,
        exact_n_atoms=bucket_atoms,
        exact_n_waters=wb_pad,
        constraint_bucket_idx=0,
        has_real_constraints=False,
    )
    bun_408k = dataclasses.replace(
        apoa1_bun,
        positions=_pad_positions_with_ghost_lattice(
            jnp.asarray(pos_all, dtype=jnp.float32), bucket_atoms, box_vec_408k, dtype=jnp.float32
        ),
        masses=_pad_1d_const(m_all, bucket_atoms, pad_value=1.0, dtype=jnp.float32),
        charges=_pad_1d(q_all, bucket_atoms, dtype=jnp.float32),
        sigmas=_pad_1d_const(sig_all, bucket_atoms, pad_value=1.0, dtype=jnp.float32),
        epsilons=_pad_1d(eps_all, bucket_atoms, dtype=jnp.float32),
        radii=jnp.ones(bucket_atoms, dtype=jnp.float32),
        scaled_radii=jnp.ones(bucket_atoms, dtype=jnp.float32),
        atom_mask=_mask(n_target, bucket_atoms),
        n_atoms=jnp.array(n_target, dtype=jnp.int32),
        box=box_mat,
        bond_idx=_pad_2d(b_cat, bb, 2, dtype=jnp.int32),
        bond_params=_pad_2d(bp_cat, bb, 2, dtype=jnp.float32),
        bond_mask=_mask(nb_tot, bb),
        n_bonds=jnp.array(nb_tot, dtype=jnp.int32),
        angle_idx=_pad_2d(a_cat, ab, 3, dtype=jnp.int32),
        angle_params=_pad_2d(ap_cat, ab, 2, dtype=jnp.float32),
        angle_mask=_mask(na_tot, ab),
        n_angles=jnp.array(na_tot, dtype=jnp.int32),
        dihedral_idx=_pad_2d(d_cat, db, 4, dtype=jnp.int32),
        dihedral_params=_pad_2d(dp_cat, db, dp_cols, dtype=jnp.float32),
        dihedral_mask=_mask(nd_tot, db),
        n_dihedrals=jnp.array(nd_tot, dtype=jnp.int32),
        improper_idx=jnp.zeros((1, 4), dtype=jnp.int32),
        improper_params=jnp.zeros((1, 3), dtype=jnp.float32),
        improper_mask=jnp.zeros((1,), dtype=bool),
        improper_is_periodic=jnp.array(False, dtype=bool),
        n_impropers=jnp.array(0, dtype=jnp.int32),
        urey_bradley_idx=jnp.zeros((ab, 3), dtype=jnp.int32),
        urey_bradley_params=jnp.zeros((ab, 2), dtype=jnp.float32),
        urey_bradley_mask=jnp.zeros((ab,), dtype=bool),
        n_urey_bradley=jnp.array(0, dtype=jnp.int32),
        cmap_torsion_idx=jnp.zeros((1, 8), dtype=jnp.int32),
        cmap_energy_grids=jnp.zeros((1, 24, 24), dtype=jnp.float32),
        cmap_mask=jnp.zeros((1,), dtype=bool),
        cmap_indices=jnp.zeros((1,), dtype=jnp.int32),
        n_cmap=jnp.array(0, dtype=jnp.int32),
        water_indices=_pad_2d(w_all, wb_pad, 3, dtype=jnp.int32),
        water_mask=_mask(nw_tot, wb_pad),
        n_waters=jnp.array(nw_tot, dtype=jnp.int32),
        excl_indices=jnp.zeros((1, 2), dtype=jnp.int32),
        excl_scales_vdw=jnp.zeros((1,), dtype=jnp.float32),
        excl_scales_elec=jnp.zeros((1,), dtype=jnp.float32),
        excl_mask=jnp.zeros((1,), dtype=bool),
        n_excl=jnp.array(0, dtype=jnp.int32),
        excl_dense_indices=jnp.asarray(ed_pad, dtype=jnp.int32),
        excl_dense_scales_vdw=jnp.asarray(sv_pad, dtype=jnp.float32),
        excl_dense_scales_elec=jnp.asarray(se_pad, dtype=jnp.float32),
        exception_pairs=_pad_2d(ep_cat, xb, 2, dtype=jnp.int32),
        exception_sigmas=_pad_1d_const(es_cat, xb, pad_value=1.0, dtype=jnp.float32),
        exception_epsilons=_pad_1d(ee_cat, xb, dtype=jnp.float32),
        exception_chargeprods=_pad_1d(eq_cat, xb, dtype=jnp.float32),
        exception_mask=_mask(nx_tot, xb),
        n_exception_pairs=jnp.array(nx_tot, dtype=jnp.int32),
        pme_alpha=jnp.array(0.34, dtype=jnp.float32),
        cutoff_distance=jnp.array(9.0, dtype=jnp.float32),
        shape_spec=spec_408k,
        constraint_pairs=jnp.zeros((0, 2), dtype=jnp.int32),
        constraint_lengths=jnp.zeros((0,), dtype=jnp.float32),
        constraint_mask=jnp.zeros((0,), dtype=bool),
        n_constraints=jnp.array(0, dtype=jnp.int32),
        water_r_OH=jnp.array(0.9572, dtype=jnp.float32),
        water_r_HH=jnp.array(1.5139, dtype=jnp.float32),
    )
    omm_sys_408k, omm_pos_408k, _ = _build_openmm_explicit_system(
        bun_408k, electrostatic_method="pme", pme_grid_dims=(192, 192, 80)
    )
    return bun_408k, omm_sys_408k, omm_pos_408k, "periodic_408609_solvated_supercell"


# ============================================================================
# SUITE A & B: Official OpenMM Suite + 8-Step Solvated/Holo/408k Ladder
# ============================================================================
def run_suites_a_and_b_ladder(
    dhfr_bundle: MolecularBundle,
    *,
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    quick: bool = False,
    systems_filter: set[str] | None = None,
    nonbonded_engine: str | None = None,
    resume_checkpoint: bool = True,
    checkpoint_out: Path | None = None,
    checkpoint_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run Suite A & Suite B (`single` AND `mixed` precision + Cold JIT Latency + `408k` scaling)."""
    import scripts.benchmarks.bench_v34_system_ladder as v34_mod

    v34_mod._ALLOW_OPENCL_FALLBACK = _CONFIG["allow_opencl_fallback"]

    print("=== [Suite A & B] Pallas v3.5 Official OpenMM Suite + 8-Step Ladder (2k -> 408k) ===")
    n_steps = 40 if quick else 100
    n_steady = 1 if quick else 2
    n_wall = 2 if quick else 5

    existing_gbsa: dict[str, Any] | None = None
    existing_explicit: dict[str, dict[str, Any]] = {}
    ckpt_candidates: list[Path] = []
    if checkpoint_out is not None:
        mirror_dir = Path("/orcd/pool/007/maarxaru/slurm_logs/prolix/ckpt_v35")
        if mirror_dir.parent.exists():
            ckpt_candidates.append(mirror_dir / checkpoint_out.name)
        ckpt_candidates.append(checkpoint_out)
    if resume_checkpoint:
        for cand_path in ckpt_candidates:
            if not cand_path.exists():
                continue
            try:
                prev_data = json.loads(cand_path.read_text())
                if prev_data.get("schema_version") != "pallas_v35_release_ladder_v1":
                    continue
                sab = prev_data.get("suites_a_and_b", {})
                gbsa_cand = sab.get("suite_a_gbsa", {})
                if gbsa_cand and gbsa_cand.get("batch_scaling") and existing_gbsa is None:
                    _backfill_provenance_in_record(gbsa_cand)
                    existing_gbsa = gbsa_cand
                for rec in sab.get("explicit_ladder", []):
                    sys_n = rec.get("system")
                    if sys_n and sys_n not in existing_explicit and rec.get("batch_scaling"):
                        _backfill_provenance_in_record(rec)
                        existing_explicit[sys_n] = rec
            except Exception as exc:
                print(f"  [Resume] Could not load v3.5 checkpoint {cand_path}: {exc}")

    run_gbsa = systems_filter is None or "5DFR_gbsa_implicit" in systems_filter or "gbsa" in systems_filter
    suite_a_gbsa_dict: dict[str, Any] | None = None
    if run_gbsa:
        if existing_gbsa is not None:
            suite_a_gbsa_dict = existing_gbsa
        else:
            _clear_all_caches()
            gbsa_bun, gbsa_omm_sys, gbsa_omm_pos = build_implicit_protein_bundle(
                "5DFR_gbsa",
                "submodules/openmm/examples/benchmarks/5dfr_minimized.pdb",
                keep_chain=None,
                add_hydrogens_with_openmm=False,
                bucket_atoms=2560,
            )
            gbsa_xml = __import__("openmm").XmlSerializer.serialize(gbsa_omm_sys)
            gbsa_pos_nm = np.asarray(gbsa_bun.positions[: int(gbsa_bun.n_atoms)], dtype=np.float64) * 0.1
            gbsa_omm_bench = _time_openmm_cuda_single_and_mps(
                gbsa_omm_sys,
                gbsa_omm_pos,
                gbsa_xml,
                gbsa_pos_nm,
                batch_sizes=batch_sizes,
                dt_fs=2.0,
                gamma_ps=91.0,
                n_warmup=20 if quick else 80,
                n_steps=50 if quick else 200,
                run_mps_batches=(1, 4, 8) if quick else (1, 4, 8, 16),
            )

            gbsa_rows: list[dict[str, Any]] = []
            gbsa_mixed_rows: list[dict[str, Any]] = []
            try:
                for B in batch_sizes:
                    res_s = _make_plan_and_run_timed(
                        [gbsa_bun for _ in range(B)],
                        n_warmup_steps=10,
                        n_steps=n_steps,
                        dt_fs=2.0,
                        kT=0.596,
                        gamma_ps=91.0,
                        use_neighbor_list=False,
                        use_pallas=True,
                        nonbonded_engine=nonbonded_engine,
                        precision="single",
                    )
                    ms_batch, ms_rep = res_s["ms_batch"], res_s["ms_rep"]
                    omm_b1 = gbsa_omm_bench["openmm_cuda_single_b1"]["ms_per_step"]
                    mps_entry = gbsa_omm_bench["openmm_cuda_mps_scaling"].get(f"B={B}", {})
                    mps_rep = mps_entry.get("ms_per_step_per_replica")
                    gbsa_rows.append(
                        {
                            "B": B,
                            "precision": "single",
                            "cold_compile_s": res_s["cold_compile_s"],
                            "ms_per_step_batch": ms_batch,
                            "ms_per_step_per_replica": ms_rep,
                            "ns_per_day_per_replica": _ns_per_day(ms_batch, dt_fs=2.0),
                            "ns_per_day_aggregate": _ns_per_day(ms_rep, dt_fs=2.0),
                            "peak_vram_mb": res_s["peak_bytes_in_use_mb"],
                            "speedup_vs_openmm_cuda_single_b1": float(omm_b1 / ms_rep),
                            "speedup_vs_openmm_cuda_mps_same_B": (
                                float(mps_rep / ms_rep) if mps_rep and mps_rep > 0 else None
                            ),
                        }
                    )
                    print(
                        f"  [Suite A MK-67] 5DFR_gbsa_implicit (N=2489) B={B:2d}: "
                        f"Prolix={ms_rep:.4f} ms/rep (compile={res_s['cold_compile_s']:.2f}s) "
                        f"vs OpenMM CUDA B=1={omm_b1:.4f} ms"
                        + (f", MPS B={B}={mps_rep:.4f} ms/rep" if mps_rep else "")
                    )
                for B_mx in tuple(b for b in (1, 4, 8) if b in batch_sizes):
                    with contextlib.suppress(Exception):
                        res_m = _make_plan_and_run_timed(
                            [gbsa_bun for _ in range(B_mx)],
                            n_warmup_steps=10,
                            n_steps=n_steps,
                            dt_fs=2.0,
                            kT=0.596,
                            gamma_ps=91.0,
                            use_neighbor_list=False,
                            use_pallas=True,
                            nonbonded_engine=nonbonded_engine,
                            precision="mixed",
                        )
                        omm_mx = gbsa_omm_bench["openmm_cuda_mixed_b1"]["ms_per_step"]
                        gbsa_mixed_rows.append(
                            {
                                "B": B_mx,
                                "precision": "mixed",
                                "cold_compile_s": res_m["cold_compile_s"],
                                "ms_per_step_batch": res_m["ms_batch"],
                                "ms_per_step_per_replica": res_m["ms_rep"],
                                "ns_per_day_per_replica": _ns_per_day(res_m["ms_batch"], dt_fs=2.0),
                                "ns_per_day_aggregate": _ns_per_day(res_m["ms_rep"], dt_fs=2.0),
                                "speedup_vs_openmm_cuda_mixed_b1": float(omm_mx / res_m["ms_rep"]),
                            }
                        )
            finally:
                os.environ.pop("PROLIX_ACTIVE_ATOMS", None)

            suite_a_gbsa_dict = {
                "system": "5DFR_gbsa_implicit",
                "n_real_atoms": int(gbsa_bun.n_atoms),
                "bucket_atoms": 2560,
                "openmm": gbsa_omm_bench,
                "batch_scaling": gbsa_rows,
                "mixed_precision_scaling": gbsa_mixed_rows,
            }
            if checkpoint_out is not None:
                partial_payload = dict(checkpoint_meta or {})
                partial_payload["suites_a_and_b"] = {
                    "suite_a_gbsa": suite_a_gbsa_dict,
                    "explicit_ladder": list(existing_explicit.values()),
                }
                for cp_path in ckpt_candidates:
                    cp_path.parent.mkdir(parents=True, exist_ok=True)
                    cp_path.write_text(json.dumps(partial_payload, indent=2))

    _, dhfr_omm_grid = compute_openmm_pme_params(
        np.array([62.23, 62.23, 62.23], dtype=np.float64), cutoff=9.0, ewald_error_tolerance=5e-4
    )
    dhfr_omm_grid_str = str(dhfr_omm_grid[0])

    ladder_specs = [
        ("1UAO_solvated_2k", ["SuiteB"], "trunc", 1963, 4096, 1024, "pme", "32"),
        ("1VII_solvated_4k", ["SuiteB"], "trunc", 4199, 8192, 2048, "pme", "40"),
        ("1UBQ_solvated_7k", ["SuiteB"], "trunc", 6658, 8192, 2048, "pme", "48"),
        ("Holo_Lysozyme_15k", ["SuiteB"], "holo_15k", 15002, 16384, 4608, "pme", "54"),
        ("DHFR_24k_rf", ["SuiteA"], "trunc", 23558, 25000, 8192, "rf", "64"),
        ("DHFR_24k_pme_openmm_grid", ["SuiteA", "SuiteB"], "trunc", 23558, 25000, 8192, "pme", dhfr_omm_grid_str),
        ("DHFR_24k_pme_64grid", ["SuiteA"], "trunc", 23558, 25000, 8192, "pme", "64"),
        ("DHFR_24k_pme_hbonds", ["SuiteA"], "trunc_hbonds", 23558, 25000, 8192, "pme", dhfr_omm_grid_str),
        ("Holo_DHFR_MTX_NADPH_24k", ["SuiteB"], "holo_dhfr", 23621, 25000, 8192, "pme", dhfr_omm_grid_str),
        ("1AKE_solvated_45k", ["SuiteB"], "1ake_45k", 44999, 50000, 16384, "pme", "80"),
        ("ApoA1_92k_rf", ["SuiteA"], "apoa1_rf", 92224, 100000, 24576, "rf", "96x96x64"),
        ("ApoA1_92k_pme", ["SuiteA", "SuiteB"], "apoa1_pme", 92224, 100000, 24576, "pme", "96x96x64"),
        ("Cellulose_408k_pme", ["SuiteA", "SuiteB"], "cellulose_408k", 408609, 409600, 122880, "pme", "192x192x80"),
    ]

    explicit_records: list[dict[str, Any]] = []
    prev_gp = os.environ.get("PROLIX_PME_GRID_POINTS")
    try:
        for name, suites, kind, n_real_target, b_atoms, b_waters, elec, grid_str in ladder_specs:
            if systems_filter is not None and name not in systems_filter:
                continue
            if name in existing_explicit:
                explicit_records.append(existing_explicit[name])
                continue
            os.environ["PROLIX_PME_GRID_POINTS"] = grid_str
            _clear_all_caches()
            topology_source = "openmm_benchmark_suite"

            if kind == "trunc":
                bun = _cast_bundle_precision(
                    _make_truncated_solvated_bundle(
                        dhfr_bundle,
                        n_real_target=n_real_target,
                        bucket_atoms=b_atoms,
                        bucket_waters=b_waters,
                    ),
                    jnp.float32,
                )
                if elec == "rf":
                    bun = dataclasses.replace(bun, pme_alpha=jnp.array(0.0, dtype=jnp.float32))
                g_parts = [snap_fft_dim(int(x)) for x in grid_str.split("x")]
                g_dims = (
                    (g_parts[0], g_parts[0], g_parts[0])
                    if len(g_parts) == 1
                    else (g_parts[0], g_parts[1], g_parts[2])
                )
                omm_sys, omm_pos, omm_xml = _build_openmm_explicit_system(
                    bun, electrostatic_method=elec, pme_grid_dims=g_dims
                )
            elif kind == "trunc_hbonds":
                bun = _cast_bundle_precision(
                    _re_pad_bundle_to_custom_bucket(
                        dhfr_bundle,
                        b_atoms,
                        b_waters,
                        electrostatic_method="pme",
                        include_constraints=True,
                    ),
                    jnp.float32,
                )
                gp0 = snap_fft_dim(int(grid_str))
                g_dims = (gp0, gp0, gp0)
                omm_sys, omm_pos, omm_xml = _build_openmm_explicit_system(
                    bun, electrostatic_method="pme", pme_grid_dims=g_dims
                )
            elif kind == "holo_15k":
                bun = build_holo_protein_ligand_bundle(
                    dhfr_bundle,
                    n_prot_atoms=1960,
                    n_ligand_atoms=46,
                    n_real_target=n_real_target,
                    bucket_atoms=b_atoms,
                    bucket_waters=b_waters,
                    ligand_seed=1,
                )
                g_dims = (54, 54, 54)
                omm_sys, omm_pos, omm_xml = _build_openmm_explicit_system(
                    bun, electrostatic_method="pme", pme_grid_dims=g_dims
                )
            elif kind == "holo_dhfr":
                bun = build_holo_protein_ligand_bundle(
                    dhfr_bundle,
                    n_prot_atoms=2489,
                    n_ligand_atoms=63,
                    n_real_target=23558,
                    bucket_atoms=b_atoms,
                    bucket_waters=b_waters,
                    ligand_seed=2,
                )
                gp0 = snap_fft_dim(int(grid_str))
                g_dims = (gp0, gp0, gp0)
                omm_sys, omm_pos, omm_xml = _build_openmm_explicit_system(
                    bun, electrostatic_method="pme", pme_grid_dims=g_dims
                )
            elif kind == "1ake_45k":
                bun = build_solvated_1ake_bundle(
                    dhfr_bundle,
                    n_real_target=n_real_target,
                    bucket_atoms=b_atoms,
                    bucket_waters=b_waters,
                )
                g_dims = (80, 80, 80)
                omm_sys, omm_pos, omm_xml = _build_openmm_explicit_system(
                    bun, electrostatic_method="pme", pme_grid_dims=g_dims
                )
            elif kind in ("apoa1_rf", "apoa1_pme"):
                bun, omm_sys, omm_pos = build_apoa1_bundle(
                    bucket_atoms=b_atoms,
                    bucket_waters=b_waters,
                    electrostatic_method=elec,
                    include_constraints=False,
                )
                g_dims = (96, 96, 64)
                omm_xml = __import__("openmm").XmlSerializer.serialize(omm_sys)
            elif kind == "cellulose_408k":
                bun, omm_sys, omm_pos, topology_source = build_cellulose_408k_bundle(
                    bucket_atoms=b_atoms, bucket_waters=b_waters
                )
                g_dims = (192, 192, 96) if topology_source == "amber20_cellulose_prmtop" else (192, 192, 80)
                os.environ["PROLIX_PME_GRID_POINTS"] = f"{g_dims[0]}x{g_dims[1]}x{g_dims[2]}"
                omm_xml = __import__("openmm").XmlSerializer.serialize(omm_sys)
            else:
                raise ValueError(f"Unknown ladder kind: {kind}")

            n_real = int(bun.n_atoms)
            b_atoms_eff = int(bun.positions.shape[0])
            pos_nm_np = np.asarray(bun.positions[:n_real], dtype=np.float64) * 0.1

            if b_atoms_eff >= 300000:
                eff_batches = tuple(b for b in batch_sizes if b <= (1 if quick else 2))
                mps_batches = tuple(b for b in (1, 2) if b in eff_batches)
            elif b_atoms_eff >= 100000:
                eff_batches = tuple(b for b in batch_sizes if b <= (4 if quick else 8))
                mps_batches = tuple(b for b in (1, 2, 4) if b in eff_batches)
            elif b_atoms_eff >= 50000:
                eff_batches = tuple(b for b in batch_sizes if b <= (8 if quick else 16))
                mps_batches = tuple(b for b in (1, 4, 8) if b in eff_batches)
            else:
                eff_batches = batch_sizes
                mps_batches = tuple(b for b in (1, 4, 8, 16) if b in eff_batches)

            substeps_b1 = None
            substeps_b4 = None
            if b_atoms_eff <= 100000:
                use_perfetto = name == "DHFR_24k_pme_openmm_grid"
                substeps_b1 = _profile_explicit_substeps(
                    bun, g_dims, b_prof=1, n_steady=n_steady, n_wall=n_wall, use_perfetto_probe=use_perfetto
                )
                substeps_b4 = (
                    _profile_explicit_substeps(
                        bun, g_dims, b_prof=4, n_steady=n_steady, n_wall=n_wall, use_perfetto_probe=False
                    )
                    if 4 in eff_batches and b_atoms_eff <= 25000
                    else None
                )

            omm_bench = _time_openmm_cuda_single_and_mps(
                omm_sys,
                omm_pos,
                omm_xml,
                pos_nm_np,
                batch_sizes=eff_batches,
                dt_fs=2.0,
                gamma_ps=10.0,
                n_warmup=10 if b_atoms_eff >= 300000 else (15 if (quick or b_atoms_eff >= 50000) else 60),
                n_steps=20 if b_atoms_eff >= 300000 else (40 if (quick or b_atoms_eff >= 50000) else 200),
                run_mps_batches=mps_batches,
            )

            rows_b: list[dict[str, Any]] = []
            mixed_precision_rows: list[dict[str, Any]] = []
            row_nl_ablation_b1_ms: float | None = None
            try:
                for B in eff_batches:
                    steps_run = (
                        20
                        if b_atoms_eff >= 300000
                        else (
                            80
                            if (quick and B <= 4 and b_atoms_eff <= 25000)
                            else (max(20, n_steps // 2) if (b_atoms_eff >= 50000 and B >= 8) else n_steps)
                        )
                    )
                    warmup_run = 20 if b_atoms_eff >= 300000 else 40
                    res_s = _make_plan_and_run_timed(
                        [bun for _ in range(B)],
                        n_warmup_steps=warmup_run,
                        n_steps=steps_run,
                        dt_fs=2.0,
                        kT=0.596,
                        gamma_ps=10.0,
                        use_neighbor_list=True,
                        nl_update_every=20,
                        use_pallas=True,
                        use_pallas_pme=True,
                        nonbonded_engine=nonbonded_engine,
                        precision="single",
                    )
                    ms_batch, ms_rep = res_s["ms_batch"], res_s["ms_rep"]
                    omm_b1 = omm_bench["openmm_cuda_single_b1"]["ms_per_step"]
                    mps_entry = omm_bench["openmm_cuda_mps_scaling"].get(f"B={B}", {})
                    mps_rep = mps_entry.get("ms_per_step_per_replica")
                    rows_b.append(
                        {
                            "B": B,
                            "precision": "single",
                            "cold_compile_s": res_s["cold_compile_s"],
                            "ms_per_step_batch": ms_batch,
                            "ms_per_step_per_replica": ms_rep,
                            "ns_per_day_per_replica": _ns_per_day(ms_batch, dt_fs=2.0),
                            "ns_per_day_aggregate": _ns_per_day(ms_rep, dt_fs=2.0),
                            "peak_vram_mb": res_s["peak_bytes_in_use_mb"],
                            "speedup_vs_openmm_cuda_single_b1": float(omm_b1 / ms_rep),
                            "speedup_vs_openmm_cuda_mps_same_B": (
                                float(mps_rep / ms_rep) if mps_rep and mps_rep > 0 else None
                            ),
                        }
                    )
                    print(
                        f"  [{','.join(suites)}] {name:26s} (N={n_real:6d}, grid={grid_str:10s}) B={B:2d}: "
                        f"Prolix(s)={ms_rep:.4f} ms/rep (compile={res_s['cold_compile_s']:.2f}s) "
                        f"vs OpenMM(s) B=1={omm_b1:.4f} ms"
                        + (f", MPS B={B}={mps_rep:.4f} ms/rep" if mps_rep else "")
                    )

                # Side-by-side `mixed` precision across every system (`B=1` + `B=4` where applicable)
                mx_batches = tuple(b for b in (1, 4, 8) if b in eff_batches and (b == 1 or b_atoms_eff <= 100000))
                for B_mx in mx_batches:
                    with contextlib.suppress(Exception):
                        steps_mx = 20 if b_atoms_eff >= 300000 else (60 if quick else n_steps)
                        res_mx = _make_plan_and_run_timed(
                            [bun for _ in range(B_mx)],
                            n_warmup_steps=20 if b_atoms_eff >= 300000 else 40,
                            n_steps=steps_mx,
                            dt_fs=2.0,
                            kT=0.596,
                            gamma_ps=10.0,
                            use_neighbor_list=True,
                            nl_update_every=20,
                            use_pallas=True,
                            use_pallas_pme=True,
                            nonbonded_engine=nonbonded_engine,
                            precision="mixed",
                        )
                        omm_mx_b1 = omm_bench["openmm_cuda_mixed_b1"]["ms_per_step"]
                        mixed_precision_rows.append(
                            {
                                "B": B_mx,
                                "precision": "mixed",
                                "cold_compile_s": res_mx["cold_compile_s"],
                                "ms_per_step_batch": res_mx["ms_batch"],
                                "ms_per_step_per_replica": res_mx["ms_rep"],
                                "ns_per_day_per_replica": _ns_per_day(res_mx["ms_batch"], dt_fs=2.0),
                                "ns_per_day_aggregate": _ns_per_day(res_mx["ms_rep"], dt_fs=2.0),
                                "peak_vram_mb": res_mx["peak_bytes_in_use_mb"],
                                "speedup_vs_openmm_cuda_mixed_b1": float(omm_mx_b1 / res_mx["ms_rep"]),
                            }
                        )
                        print(
                            f"  [mixed] {name:26s} B={B_mx:2d}: "
                            f"Prolix(mixed)={res_mx['ms_rep']:.4f} ms/rep vs OpenMM(mixed) B=1={omm_mx_b1:.4f} ms"
                        )

                if name == "DHFR_24k_pme_openmm_grid":
                    with contextlib.suppress(Exception):
                        res_nl = _make_plan_and_run_timed(
                            [bun],
                            n_warmup_steps=40,
                            n_steps=60 if quick else n_steps,
                            dt_fs=2.0,
                            kT=0.596,
                            gamma_ps=10.0,
                            use_neighbor_list=True,
                            nl_update_every=20,
                            use_pallas=True,
                            use_pallas_pme=True,
                            nonbonded_engine="neighbor_list",
                            precision="single",
                        )
                        row_nl_ablation_b1_ms = res_nl["ms_rep"]
            finally:
                os.environ.pop("PROLIX_ACTIVE_ATOMS", None)
                os.environ.pop("PROLIX_ACTIVE_WATERS", None)

            rec_entry: dict[str, Any] = {
                "system": name,
                "topology_source": topology_source,
                "suites": suites,
                "kind": kind,
                "n_real_atoms": n_real,
                "bucket_atoms": b_atoms_eff,
                "bucket_waters": int(bun.water_indices.shape[0]),
                "occupancy_pct": float(100.0 * n_real / b_atoms_eff),
                "electrostatics": elec,
                "pme_grid": g_dims if elec == "pme" else None,
                "openmm": omm_bench,
                "substep_breakdown": {"B=1": substeps_b1, "B=4": substeps_b4},
                "batch_scaling": rows_b,
                "mixed_precision_scaling": mixed_precision_rows,
            }
            if row_nl_ablation_b1_ms is not None:
                rec_entry["neighbor_list_fallback_b1_ms"] = float(row_nl_ablation_b1_ms)
            explicit_records.append(rec_entry)

            if checkpoint_out is not None:
                merged_explicit = dict(existing_explicit)
                for r_done in explicit_records:
                    if r_done.get("system"):
                        merged_explicit[r_done["system"]] = r_done
                partial_payload = dict(checkpoint_meta or {})
                partial_payload["suites_a_and_b"] = {
                    "suite_a_gbsa": suite_a_gbsa_dict,
                    "explicit_ladder": list(merged_explicit.values()),
                }
                for cp_path in ckpt_candidates:
                    cp_path.parent.mkdir(parents=True, exist_ok=True)
                    cp_path.write_text(json.dumps(partial_payload, indent=2))
    finally:
        if prev_gp is None:
            os.environ.pop("PROLIX_PME_GRID_POINTS", None)
        else:
            os.environ["PROLIX_PME_GRID_POINTS"] = prev_gp

    return {
        "suite_a_gbsa": suite_a_gbsa_dict,
        "explicit_ladder": explicit_records,
    }


# ============================================================================
# SUITE C: Congeneric Screen (MK-64/65) & Cross-Size Ragged Queue (MK-68)
# ============================================================================
def run_suite_c_congeneric_and_ragged(
    dhfr_bundle: MolecularBundle,
    *,
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    quick: bool = False,
    nonbonded_engine: str | None = None,
    pme_tier_mode: str = "auto",
) -> dict[str, Any]:
    """Run Suite C: Congeneric Screen (`MK-64/65`) & Cross-Size Ragged Screen (`MK-68`, 1-compile check)."""
    print("=== [Suite C] Heterogeneous Congeneric Screen (MK-64/65) & Cross-Size Ragged Queue (MK-68) ===")
    if quick:
        batch_sizes = tuple(b for b in batch_sizes if b <= 16)
    n_steps = 40 if quick else 100

    _, dhfr_omm_grid = compute_openmm_pme_params(
        np.array([62.23, 62.23, 62.23], dtype=np.float64), cutoff=9.0, ewald_error_tolerance=5e-4
    )
    grid_str = str(dhfr_omm_grid[0])
    prev_gp = os.environ.get("PROLIX_PME_GRID_POINTS")
    os.environ["PROLIX_PME_GRID_POINTS"] = grid_str

    try:
        max_b = max(batch_sizes)
        congeneric_pool: list[MolecularBundle] = []
        omm_xml_pool: list[str] = []
        omm_pos_nm_pool: list[np.ndarray] = []

        base_holo = build_holo_protein_ligand_bundle(
            dhfr_bundle,
            n_prot_atoms=2489,
            n_ligand_atoms=63,
            n_real_target=23558,
            bucket_atoms=25000,
            bucket_waters=8192,
            ligand_seed=100,
        )
        n_real_holo = int(base_holo.n_atoms)
        for b_idx in range(max_b):
            if b_idx == 0:
                bun_b = base_holo
            else:
                scale_q = 1.0 + 0.015 * ((b_idx % 7) - 3)
                scale_lj = 1.0 + 0.01 * ((b_idx % 5) - 2)
                lig_mask = (jnp.arange(25000) >= 2489) & (jnp.arange(25000) < 2489 + 63)
                q_new = jnp.where(lig_mask, base_holo.charges * scale_q, base_holo.charges)
                sig_new = jnp.where(lig_mask, base_holo.sigmas * scale_lj, base_holo.sigmas)
                eps_new = jnp.where(lig_mask, base_holo.epsilons * scale_lj, base_holo.epsilons)
                dp_new = base_holo.dihedral_params * jnp.array(
                    [1.0, 1.0, 1.0 + 0.02 * (b_idx % 6)], dtype=jnp.float32
                )
                eq_new = base_holo.exception_chargeprods * jnp.float32(scale_q)
                bun_b = dataclasses.replace(
                    base_holo,
                    charges=q_new,
                    sigmas=sig_new,
                    epsilons=eps_new,
                    dihedral_params=dp_new,
                    exception_chargeprods=eq_new,
                )
            congeneric_pool.append(bun_b)
            if b_idx < 16:
                _, _, xml_b = _build_openmm_explicit_system(
                    bun_b, electrostatic_method="pme", pme_grid_dims=dhfr_omm_grid
                )
                omm_xml_pool.append(xml_b)
                omm_pos_nm_pool.append(np.asarray(bun_b.positions[:n_real_holo], dtype=np.float64) * 0.1)

        congeneric_rows: list[dict[str, Any]] = []
        try:
            for B in batch_sizes:
                res_c = _make_plan_and_run_timed(
                    congeneric_pool[:B],
                    n_warmup_steps=40,
                    n_steps=n_steps,
                    dt_fs=2.0,
                    kT=0.596,
                    gamma_ps=10.0,
                    use_neighbor_list=True,
                    nl_update_every=20,
                    use_pallas=True,
                    use_pallas_pme=True,
                    nonbonded_engine=nonbonded_engine,
                    precision="single",
                )
                ms_batch, ms_rep = res_c["ms_batch"], res_c["ms_rep"]
                mps_res = None
                if B <= 16 and (not quick or B in (1, 4, 8)):
                    mps_res = run_openmm_cuda_mps_replicas(
                        omm_xml_pool[:B],
                        omm_pos_nm_pool[:B],
                        replicas_B=B,
                        precision="single",
                        dt_fs=2.0,
                        gamma_ps=10.0,
                        n_warmup=15 if quick else 40,
                        n_steps=30 if quick else 100,
                        use_mps_daemon=(B > 1),
                        allow_opencl_fallback=_CONFIG["allow_opencl_fallback"],
                    )
                mps_rep = mps_res.get("ms_per_step_per_replica") if mps_res else None
                congeneric_rows.append(
                    {
                        "B": B,
                        "cold_compile_s": res_c["cold_compile_s"],
                        "xla_executable_count": res_c["xla_executable_count"],
                        "prolix_ms_per_step_batch": ms_batch,
                        "prolix_ms_per_step_per_replica": ms_rep,
                        "prolix_ns_per_day_per_replica": _ns_per_day(ms_batch, dt_fs=2.0),
                        "prolix_ns_per_day_aggregate": _ns_per_day(ms_rep, dt_fs=2.0),
                        "peak_vram_mb": res_c["peak_bytes_in_use_mb"],
                        "openmm_cuda_mps": mps_res,
                        "speedup_vs_openmm_cuda_mps": (
                            float(mps_rep / ms_rep) if mps_rep and mps_rep > 0 else None
                        ),
                    }
                )
                print(
                    f"  [Suite C1 Congeneric Holo-DHFR] B={B:2d}: Prolix={ms_rep:.4f} ms/rep "
                    f"(compiles={res_c['xla_executable_count']}) "
                    + (f"vs MPS={mps_rep:.4f} ms/rep ({mps_rep / ms_rep:.2f}x)" if mps_rep else "")
                )
        finally:
            os.environ.pop("PROLIX_ACTIVE_ATOMS", None)
            os.environ.pop("PROLIX_ACTIVE_WATERS", None)

        # --- Part C2: Cross-Size Ragged Multi-Target Screen (`MK-68`) ---
        ragged_targets = [
            ("1UAO_2k", 1963),
            ("1VII_4k", 4199),
            ("1UBQ_7k", 6658),
            ("Holo_Lysozyme_15k", 15002),
            ("Holo_DHFR_24k", 23558),
        ]
        ragged_bundles: list[MolecularBundle] = []
        ragged_xmls: list[str] = []
        ragged_pos_nm: list[np.ndarray] = []
        for idx_t, (t_name, t_n_real) in enumerate(ragged_targets):
            if "Holo" in t_name:
                b_t = build_holo_protein_ligand_bundle(
                    dhfr_bundle,
                    n_prot_atoms=1960 if t_n_real < 20000 else 2489,
                    n_ligand_atoms=46 if t_n_real < 20000 else 63,
                    n_real_target=t_n_real,
                    bucket_atoms=25000,
                    bucket_waters=8192,
                    ligand_seed=200 + idx_t,
                )
            else:
                b_t = _cast_bundle_precision(
                    _make_truncated_solvated_bundle(
                        dhfr_bundle,
                        n_real_target=t_n_real,
                        bucket_atoms=25000,
                        bucket_waters=8192,
                    ),
                    jnp.float32,
                )
            ragged_bundles.append(b_t)
            _, _, xml_t = _build_openmm_explicit_system(
                b_t, electrostatic_method="pme", pme_grid_dims=dhfr_omm_grid
            )
            ragged_xmls.append(xml_t)
            ragged_pos_nm.append(np.asarray(b_t.positions[: int(b_t.n_atoms)], dtype=np.float64) * 0.1)

        max_b_rag = max(int(b.bond_idx.shape[0]) for b in ragged_bundles)
        max_a_rag = max(int(b.angle_idx.shape[0]) for b in ragged_bundles)
        max_d_rag = max(int(b.dihedral_idx.shape[0]) for b in ragged_bundles)

        def _pad_1d_or_2d(arr: Any, target_len: int) -> jnp.ndarray:
            a_np = np.asarray(arr)
            cur_len = int(a_np.shape[0])
            if cur_len >= target_len:
                return jnp.asarray(a_np[:target_len])
            pad_width = [(0, target_len - cur_len)] + [(0, 0)] * (a_np.ndim - 1)
            return jnp.asarray(np.pad(a_np, pad_width, mode="constant", constant_values=0))

        aligned_ragged: list[MolecularBundle] = []
        for b_r in ragged_bundles:
            sp_r = (
                dataclasses.replace(
                    b_r.shape_spec,
                    bond_bucket_idx=max_b_rag,
                    angle_bucket_idx=max_a_rag,
                    dihedral_bucket_idx=max_d_rag,
                )
                if b_r.shape_spec is not None
                else None
            )
            aligned_ragged.append(
                dataclasses.replace(
                    b_r,
                    bond_idx=_pad_1d_or_2d(b_r.bond_idx, max_b_rag),
                    bond_params=_pad_1d_or_2d(b_r.bond_params, max_b_rag),
                    bond_mask=_pad_1d_or_2d(b_r.bond_mask, max_b_rag),
                    angle_idx=_pad_1d_or_2d(b_r.angle_idx, max_a_rag),
                    angle_params=_pad_1d_or_2d(b_r.angle_params, max_a_rag),
                    angle_mask=_pad_1d_or_2d(b_r.angle_mask, max_a_rag),
                    dihedral_idx=_pad_1d_or_2d(b_r.dihedral_idx, max_d_rag),
                    dihedral_params=_pad_1d_or_2d(b_r.dihedral_params, max_d_rag),
                    dihedral_mask=_pad_1d_or_2d(b_r.dihedral_mask, max_d_rag),
                    shape_spec=sp_r,
                )
            )
        ragged_bundles = aligned_ragged
        B_rag = len(ragged_bundles)

        res_rag = _make_plan_and_run_timed(
            ragged_bundles,
            n_warmup_steps=40,
            n_steps=n_steps,
            dt_fs=2.0,
            kT=0.596,
            gamma_ps=10.0,
            use_neighbor_list=True,
            nl_update_every=20,
            use_pallas=True,
            use_pallas_pme=True,
            nonbonded_engine=nonbonded_engine,
            pme_tier_mode=pme_tier_mode,
        )
        ms_rag_batch, ms_rag_rep = res_rag["ms_batch"], res_rag["ms_rep"]

        tier_mode_ablation: dict[str, dict[str, float]] = {}
        for tm in ("auto", "grouped", "uniform"):
            with contextlib.suppress(Exception):
                res_tm = _make_plan_and_run_timed(
                    ragged_bundles,
                    n_warmup_steps=20,
                    n_steps=40 if quick else n_steps,
                    dt_fs=2.0,
                    kT=0.596,
                    gamma_ps=10.0,
                    use_neighbor_list=True,
                    nl_update_every=20,
                    use_pallas=True,
                    use_pallas_pme=True,
                    nonbonded_engine=nonbonded_engine,
                    pme_tier_mode=tm,
                )
                tier_mode_ablation[tm] = {
                    "ms_per_step_batch": res_tm["ms_batch"],
                    "ms_per_step_per_replica": res_tm["ms_rep"],
                    "cold_compile_s": res_tm["cold_compile_s"],
                }

        mps_rag = run_openmm_cuda_mps_replicas(
            ragged_xmls,
            ragged_pos_nm,
            replicas_B=B_rag,
            precision="single",
            dt_fs=2.0,
            gamma_ps=10.0,
            n_warmup=15 if quick else 40,
            n_steps=30 if quick else 100,
            use_mps_daemon=True,
            allow_opencl_fallback=_CONFIG["allow_opencl_fallback"],
        )
        mps_rag_rep = mps_rag.get("ms_per_step_per_replica") if mps_rag else None
        print(
            f"  [Suite C2 MK-68 Cross-Size Ragged B={B_rag}] Prolix={ms_rag_rep:.4f} ms/rep "
            f"(compile={res_rag['cold_compile_s']:.2f}s, xla_compiles={res_rag['xla_executable_count']}) "
            + (f"vs OpenMM CUDA+MPS={mps_rag_rep:.4f} ms/rep" if mps_rag_rep else "")
        )
    finally:
        if prev_gp is None:
            os.environ.pop("PROLIX_PME_GRID_POINTS", None)
        else:
            os.environ["PROLIX_PME_GRID_POINTS"] = prev_gp

    return {
        "congeneric_holo_dhfr_screen": {
            "bucket_atoms": 25000,
            "n_real_atoms": n_real_holo,
            "n_ligand_atoms": 63,
            "pme_grid": dhfr_omm_grid,
            "batch_scaling": congeneric_rows,
        },
        "cross_size_ragged_screen": {
            "systems": [t[0] for t in ragged_targets],
            "n_real_atoms_list": [int(b.n_atoms) for b in ragged_bundles],
            "bucket_atoms": 25000,
            "pme_tier_mode": pme_tier_mode,
            "pme_tier_mode_ablation": tier_mode_ablation,
            "cold_compile_s": res_rag["cold_compile_s"],
            "xla_executable_count": res_rag["xla_executable_count"],
            "prolix_ms_per_step_batch": ms_rag_batch,
            "prolix_ms_per_step_per_replica": ms_rag_rep,
            "prolix_ns_per_day_per_replica": _ns_per_day(ms_rag_batch, dt_fs=2.0),
            "prolix_ns_per_day_aggregate": _ns_per_day(ms_rag_rep, dt_fs=2.0),
            "openmm_cuda_mps": mps_rag,
            "speedup_vs_openmm_cuda_mps": (
                float(mps_rag_rep / ms_rag_rep) if mps_rag_rep and mps_rag_rep > 0 else None
            ),
        },
    }


# ============================================================================
# SUITE D: `ProlixSimulation` / `ProlixBatchSimulation` vs `EnsemblePlan` + NPT vs NVT
# ============================================================================
def run_suite_d_openmm_frontend_and_npt(
    dhfr_bundle: MolecularBundle,
    *,
    quick: bool = False,
) -> dict[str, Any]:
    """Benchmark `ProlixSimulation`, `ProlixBatchSimulation`, `from_openmm_systems`, and `NPT` vs `NVT`."""
    import openmm as mm
    from openmm import app
    from openmm import unit as _omm_unit

    unit = cast("Any", _omm_unit)

    import prolix.openmm as prolix_omm
    from prolix.openmm import ProlixContext, ProlixOpenMMConfig, ProlixSimulation

    print("=== [Suite D MK-69/74] `ProlixSimulation` vs `EnsemblePlan` + `ProlixBatchSimulation` + NPT vs NVT ===")
    n_warmup = 40
    n_steps = 60 if quick else 120

    _, dhfr_omm_grid = compute_openmm_pme_params(
        np.array([62.23, 62.23, 62.23], dtype=np.float64), cutoff=9.0, ewald_error_tolerance=5e-4
    )
    dhfr_bun = _cast_bundle_precision(
        _make_truncated_solvated_bundle(
            dhfr_bundle, n_real_target=23558, bucket_atoms=25000, bucket_waters=8192
        ),
        jnp.float32,
    )
    dhfr_pme_sys, dhfr_pme_pos, dhfr_pme_xml = _build_openmm_explicit_system(
        dhfr_bun, electrostatic_method="pme", pme_grid_dims=dhfr_omm_grid
    )
    dhfr_rf_bun = dataclasses.replace(dhfr_bun, pme_alpha=jnp.array(0.0, dtype=jnp.float32))
    dhfr_rf_sys, dhfr_rf_pos, _ = _build_openmm_explicit_system(
        dhfr_rf_bun, electrostatic_method="rf", pme_grid_dims=(64, 64, 64)
    )
    gbsa_bun, gbsa_omm_sys, gbsa_omm_pos = build_implicit_protein_bundle(
        "5DFR_gbsa",
        "submodules/openmm/examples/benchmarks/5dfr_minimized.pdb",
        keep_chain=None,
        add_hydrogens_with_openmm=False,
        bucket_atoms=2560,
    )

    def _make_min_top(n_particles: int) -> Any:
        top = app.Topology()
        ch = top.addChain()
        res = top.addResidue("MOL", ch)
        elem_c = app.Element.getByAtomicNumber(6)
        for _ in range(n_particles):
            top.addAtom("C", elem_c, res)
        return top

    frontend_rows: list[dict[str, Any]] = []
    frontend_cases = [
        ("5DFR_gbsa_implicit", gbsa_bun, gbsa_omm_sys, gbsa_omm_pos, 91.0, False),
        ("DHFR_24k_rf", dhfr_rf_bun, dhfr_rf_sys, dhfr_rf_pos, 10.0, True),
        ("DHFR_24k_pme_openmm_grid", dhfr_bun, dhfr_pme_sys, dhfr_pme_pos, 10.0, True),
    ]
    for sys_name, native_bun, omm_sys, omm_pos, gamma_ps, is_explicit in frontend_cases:
        top_min = _make_min_top(omm_sys.getNumParticles())
        for prec in ("single", "mixed"):
            _clear_all_caches()
            integ_ref = mm.LangevinMiddleIntegrator(
                300.0 * unit.kelvin, gamma_ps / unit.picoseconds, 2.0 * unit.femtoseconds
            )
            ctx_ref, prov_ref = require_openmm_cuda_context(
                omm_sys,
                integ_ref,
                expected_precision=prec,
                allow_opencl_fallback=_CONFIG["allow_opencl_fallback"],
            )
            ctx_ref.setPositions(omm_pos)
            ctx_ref.setVelocitiesToTemperature(300.0 * unit.kelvin, 42)
            integ_ref.step(n_warmup)
            _ = ctx_ref.getState(getEnergy=True)
            t0 = time.perf_counter()
            integ_ref.step(n_steps)
            _ = ctx_ref.getState(getEnergy=True)
            omm_ms = ((time.perf_counter() - t0) / n_steps) * 1000.0
            del ctx_ref, integ_ref

            # Native EnsemblePlan timing
            res_native = _make_plan_and_run_timed(
                [native_bun],
                n_warmup_steps=n_warmup,
                n_steps=n_steps,
                dt_fs=2.0,
                kT=0.596,
                gamma_ps=gamma_ps,
                use_neighbor_list=is_explicit,
                nl_update_every=20,
                use_pallas=True,
                use_pallas_pme=is_explicit,
                precision=prec,
            )

            # ProlixSimulation (`prolix.openmm`) timing
            cfg = ProlixOpenMMConfig(
                **_filter_supported_kwargs(
                    ProlixOpenMMConfig,
                    {
                        "precision": prec,
                        "use_pallas": True,
                        "use_pallas_pme": is_explicit,
                        "nl_update_every": 20,
                    },
                )
            )
            integ_prx = mm.LangevinMiddleIntegrator(
                300.0 * unit.kelvin, gamma_ps / unit.picoseconds, 2.0 * unit.femtoseconds
            )
            t_sim0 = time.perf_counter()
            sim_prx = ProlixSimulation(top_min, omm_sys, integ_prx, config=cfg)
            sim_prx.context.setPositions(omm_pos)
            sim_prx.context.setVelocitiesToTemperature(300.0 * unit.kelvin, 42)
            sim_prx.step(n_warmup)
            _ = sim_prx.context.getState(getPositions=True)
            sim_compile_s = time.perf_counter() - t_sim0
            t1 = time.perf_counter()
            sim_prx.step(n_steps)
            _ = sim_prx.context.getState(getPositions=True)
            sim_ms = ((time.perf_counter() - t1) / n_steps) * 1000.0

            frontend_rows.append(
                {
                    "system": sys_name,
                    "precision": prec,
                    "ensemble_plan_ms_per_step": res_native["ms_rep"],
                    "ensemble_plan_cold_compile_s": res_native["cold_compile_s"],
                    "prolix_simulation_ms_per_step": float(sim_ms),
                    "prolix_simulation_cold_compile_s": float(sim_compile_s),
                    "frontend_overhead_pct": float(
                        100.0 * (sim_ms - res_native["ms_rep"]) / max(1e-6, res_native["ms_rep"])
                    ),
                    "openmm_cuda_ms_per_step": float(omm_ms),
                    "speedup_vs_openmm_cuda": float(omm_ms / sim_ms),
                    "openmm_provenance": prov_ref,
                }
            )
            print(
                f"  [Suite D Frontend] {sys_name:26s} ({prec:6s}): "
                f"EnsemblePlan={res_native['ms_rep']:.4f} ms, ProlixSim={sim_ms:.4f} ms, "
                f"OpenMM CUDA={omm_ms:.4f} ms"
            )

    # Multi-replica `ProlixBatchSimulation` & `EnsemblePlan.from_openmm_systems`
    batch_frontend_rows: list[dict[str, Any]] = []
    pos_nm_np = np.asarray(dhfr_bun.positions[: int(dhfr_bun.n_atoms)], dtype=np.float64) * 0.1
    batch_B_list = (1, 4, 8) if quick else (1, 4, 8, 16)
    has_from_omm = hasattr(EnsemblePlan, "from_openmm_systems")
    has_batch_sim = hasattr(prolix_omm, "ProlixBatchSimulation")

    if has_batch_sim:
        batch_cls = prolix_omm.ProlixBatchSimulation
        for B in batch_B_list:
            with contextlib.suppress(Exception):
                _clear_all_caches()
                cfg_b = ProlixOpenMMConfig(
                    **_filter_supported_kwargs(
                        ProlixOpenMMConfig,
                        {"precision": "single", "use_pallas": True, "use_pallas_pme": True, "nl_update_every": 20},
                    )
                )
                integs_b = [
                    mm.LangevinMiddleIntegrator(300.0 * unit.kelvin, 10.0 / unit.picoseconds, 2.0 * unit.femtoseconds)
                    for _ in range(B)
                ]
                t_b0 = time.perf_counter()
                bsim = batch_cls(
                    [dhfr_pme_sys for _ in range(B)],
                    integs_b,
                    positions_list=[dhfr_pme_pos for _ in range(B)],
                    config=cfg_b,
                )
                bsim.setVelocitiesToTemperature(300.0 * unit.kelvin, 42)
                bsim.step(n_warmup)
                _ = bsim.getStates(getPositions=True)
                b_compile_s = time.perf_counter() - t_b0
                t_b1 = time.perf_counter()
                bsim.step(n_steps)
                _ = bsim.getStates(getPositions=True)
                ms_b = ((time.perf_counter() - t_b1) / n_steps) * 1000.0
                ms_rep = ms_b / B
                mps_res = run_openmm_cuda_mps_replicas(
                    [dhfr_pme_xml for _ in range(B)],
                    [pos_nm_np for _ in range(B)],
                    replicas_B=B,
                    precision="single",
                    dt_fs=2.0,
                    gamma_ps=10.0,
                    n_warmup=15 if quick else 40,
                    n_steps=30 if quick else 80,
                    use_mps_daemon=(B > 1),
                    allow_opencl_fallback=_CONFIG["allow_opencl_fallback"],
                )
                mps_rep = mps_res.get("ms_per_step_per_replica")
                batch_frontend_rows.append(
                    {
                        "api": "ProlixBatchSimulation",
                        "B": B,
                        "cold_compile_s": float(b_compile_s),
                        "ms_per_step_batch": float(ms_b),
                        "ms_per_step_per_replica": float(ms_rep),
                        "ns_per_day_per_replica": _ns_per_day(ms_b, dt_fs=2.0),
                        "ns_per_day_aggregate": _ns_per_day(ms_rep, dt_fs=2.0),
                        "openmm_cuda_mps": mps_res,
                        "speedup_vs_openmm_cuda_mps": (
                            float(mps_rep / ms_rep) if mps_rep and mps_rep > 0 else None
                        ),
                    }
                )
                print(
                    f"  [Suite D ProlixBatchSimulation] DHFR_24k B={B:2d}: "
                    f"Prolix={ms_rep:.4f} ms/rep vs MPS={(mps_rep if mps_rep is not None else -1.0):.4f} ms/rep"
                )

    # NPT (`HBonds` + Barostat, dynamic box `MK-74`) vs NVT timing
    npt_vs_nvt_rows: list[dict[str, Any]] = []
    hb_bun = _cast_bundle_precision(
        _re_pad_bundle_to_custom_bucket(
            dhfr_bundle, 25000, 8192, electrostatic_method="pme", include_constraints=True
        ),
        jnp.float32,
    )
    hb_nvt_sys, hb_pos, _ = _build_openmm_explicit_system(
        hb_bun, electrostatic_method="pme", pme_grid_dims=dhfr_omm_grid
    )
    hb_npt_sys, _, _ = _build_openmm_explicit_system(
        hb_bun, electrostatic_method="pme", pme_grid_dims=dhfr_omm_grid
    )
    hb_npt_sys.addForce(mm.MonteCarloBarostat(1.0 * unit.bar, 300.0 * unit.kelvin, 25))

    for ens_label, sys_e, is_npt in (("NVT_HBonds", hb_nvt_sys, False), ("NPT_HBonds_Barostat", hb_npt_sys, True)):
        integ_o = mm.LangevinMiddleIntegrator(300.0 * unit.kelvin, 10.0 / unit.picoseconds, 2.0 * unit.femtoseconds)
        ctx_o, prov_o = require_openmm_cuda_context(
            sys_e, integ_o, expected_precision="single", allow_opencl_fallback=_CONFIG["allow_opencl_fallback"]
        )
        ctx_o.setPositions(hb_pos)
        ctx_o.setVelocitiesToTemperature(300.0 * unit.kelvin, 42)
        integ_o.step(n_warmup)
        _ = ctx_o.getState(getEnergy=True)
        t_o = time.perf_counter()
        integ_o.step(n_steps)
        _ = ctx_o.getState(getEnergy=True)
        omm_ens_ms = ((time.perf_counter() - t_o) / n_steps) * 1000.0
        del ctx_o, integ_o

        res_ep = _make_plan_and_run_timed(
            [hb_bun],
            n_warmup_steps=n_warmup,
            n_steps=n_steps,
            dt_fs=2.0,
            kT=0.596,
            gamma_ps=10.0,
            use_neighbor_list=True,
            nl_update_every=20,
            use_pallas=True,
            use_pallas_pme=True,
            precision="single",
            target_pressure_atm=(1.0 if is_npt else None),
            barostat_every=25,
        )

        prx_ctx_ms: float | None = None
        with contextlib.suppress(Exception):
            cfg_e = ProlixOpenMMConfig(
                **_filter_supported_kwargs(
                    ProlixOpenMMConfig,
                    {
                        "precision": "single",
                        "use_pallas": True,
                        "use_pallas_pme": True,
                        "nl_update_every": 20,
                        "barostat_mode": "scr" if is_npt else "auto",
                    },
                )
            )
            integ_p = mm.LangevinMiddleIntegrator(300.0 * unit.kelvin, 10.0 / unit.picoseconds, 2.0 * unit.femtoseconds)
            ctx_p = ProlixContext(sys_e, integ_p, config=cfg_e)
            ctx_p.setPositions(hb_pos)
            ctx_p.setVelocitiesToTemperature(300.0 * unit.kelvin, 42)
            ctx_p.step(n_warmup)
            _ = ctx_p.getState(getPositions=True)
            t_p = time.perf_counter()
            ctx_p.step(n_steps)
            _ = ctx_p.getState(getPositions=True)
            prx_ctx_ms = ((time.perf_counter() - t_p) / n_steps) * 1000.0

        npt_vs_nvt_rows.append(
            {
                "ensemble": ens_label,
                "ensemble_plan_ms_per_step": res_ep["ms_rep"],
                "prolix_context_ms_per_step": prx_ctx_ms,
                "openmm_cuda_ms_per_step": float(omm_ens_ms),
                "openmm_provenance": prov_o,
            }
        )
        print(
            f"  [Suite D {ens_label:20s}] EnsemblePlan={res_ep['ms_rep']:.4f} ms, "
            f"ProlixContext={prx_ctx_ms or -1.0:.4f} ms, OpenMM CUDA={omm_ens_ms:.4f} ms"
        )

    return {
        "frontend_vs_ensemble_plan": frontend_rows,
        "has_from_openmm_systems": bool(has_from_omm),
        "has_prolix_batch_simulation": bool(has_batch_sim),
        "openmm_batch_scaling": batch_frontend_rows,
        "npt_vs_nvt_hbonds_dhfr": npt_vs_nvt_rows,
    }


# ============================================================================
# SUITE E: High-Batch MPS OOM-Cliff & Marginal VRAM Sweep (`B = 1..128`)
# ============================================================================
def run_suite_e_high_batch_oom_cliff(
    dhfr_bundle: MolecularBundle,
    *,
    quick: bool = False,
) -> dict[str, Any]:
    """Sweep `B = 1, 4, 8, 16, 32, 64, 128` on `1UAO_2k`, `1UBQ_7k`, and `DHFR_24k` vs `CUDA MPS`."""
    print("=== [Suite E] High-Batch CUDA MPS OOM-Cliff & Marginal VRAM Sweep (B = 1..128) ===")
    high_batches = (1, 4, 8, 16, 32, 64, 128)
    n_steps = 30 if quick else 60

    _, dhfr_omm_grid = compute_openmm_pme_params(
        np.array([62.23, 62.23, 62.23], dtype=np.float64), cutoff=9.0, ewald_error_tolerance=5e-4
    )
    sweep_specs = [
        ("1UAO_solvated_2k", 1963, 4096, 1024, (32, 32, 32), "32"),
        ("1UBQ_solvated_7k", 6658, 8192, 2048, (48, 48, 48), "48"),
        ("DHFR_24k_pme_openmm_grid", 23558, 25000, 8192, dhfr_omm_grid, str(dhfr_omm_grid[0])),
    ]

    records: list[dict[str, Any]] = []
    prev_gp = os.environ.get("PROLIX_PME_GRID_POINTS")
    try:
        for sys_name, n_real, b_atoms, b_waters, g_dims, g_str in sweep_specs:
            os.environ["PROLIX_PME_GRID_POINTS"] = g_str
            _clear_all_caches(clear_jax=True)
            bun = _cast_bundle_precision(
                _make_truncated_solvated_bundle(
                    dhfr_bundle, n_real_target=n_real, bucket_atoms=b_atoms, bucket_waters=b_waters
                ),
                jnp.float32,
            )
            _, _, omm_xml = _build_openmm_explicit_system(bun, electrostatic_method="pme", pme_grid_dims=g_dims)
            pos_nm_np = np.asarray(bun.positions[: int(bun.n_atoms)], dtype=np.float64) * 0.1

            # 1) Run OpenMM CUDA + MPS first (before JAX caches hold buffers) so MPS gets 100% fair VRAM!
            mps_map: dict[int, dict[str, Any]] = {}
            mps_hit_cliff = False
            for B in high_batches:
                if mps_hit_cliff:
                    mps_map[B] = {
                        "ok": False,
                        "oom_or_barrier_cliff": True,
                        "error": "Skipped above verified lower-B CUDA MPS OOM/process cliff",
                    }
                    continue
                mps_res = run_openmm_cuda_mps_replicas(
                    [omm_xml for _ in range(B)],
                    [pos_nm_np for _ in range(B)],
                    replicas_B=B,
                    precision="single",
                    dt_fs=2.0,
                    gamma_ps=10.0,
                    n_warmup=10 if quick else 25,
                    n_steps=20 if quick else 50,
                    use_mps_daemon=(B > 1),
                    allow_opencl_fallback=_CONFIG["allow_opencl_fallback"],
                )
                mps_map[B] = mps_res
                if not mps_res.get("ok", False):
                    mps_hit_cliff = True
                    print(f"  [Suite E MPS CLIFF] {sys_name} B={B}: {mps_res.get('error', '')[:120]}")

            # 2) Run Prolix EnsemblePlan across B = 1..128
            rows: list[dict[str, Any]] = []
            b1_live_mb: float | None = None
            for B in high_batches:
                try:
                    res_p = _make_plan_and_run_timed(
                        [bun for _ in range(B)],
                        n_warmup_steps=20,
                        n_steps=n_steps,
                        dt_fs=2.0,
                        kT=0.596,
                        gamma_ps=10.0,
                        use_neighbor_list=True,
                        nl_update_every=20,
                        use_pallas=True,
                        use_pallas_pme=True,
                        precision="single",
                    )
                    if B == 1:
                        b1_live_mb = res_p["bytes_in_use_mb"]
                    marg_mb = (
                        (res_p["bytes_in_use_mb"] - b1_live_mb) / max(1, B - 1)
                        if (B > 1 and b1_live_mb is not None)
                        else res_p["bytes_in_use_mb"]
                    )
                    mps_r = mps_map.get(B, {})
                    mps_ok = bool(mps_r.get("ok", False))
                    mps_rep = mps_r.get("ms_per_step_per_replica") if mps_ok else None
                    rows.append(
                        {
                            "B": B,
                            "prolix_ok": True,
                            "prolix_ms_per_step_per_replica": res_p["ms_rep"],
                            "prolix_ns_per_day_aggregate": _ns_per_day(res_p["ms_rep"], dt_fs=2.0),
                            "prolix_live_vram_mb": res_p["bytes_in_use_mb"],
                            "prolix_peak_vram_mb": res_p["peak_bytes_in_use_mb"],
                            "prolix_marginal_vram_mb_per_replica": float(max(0.0, marg_mb)),
                            "openmm_mps_ok": mps_ok,
                            "openmm_mps_ms_per_step_per_replica": mps_rep,
                            "openmm_mps_total_vram_mb": mps_r.get("total_gpu_memory_mb") if mps_ok else None,
                            "openmm_mps_vram_mb_per_replica": mps_r.get("memory_mb_per_replica") if mps_ok else None,
                            "openmm_mps_error": None if mps_ok else mps_r.get("error"),
                            "speedup_vs_openmm_mps": (
                                float(mps_rep / res_p["ms_rep"]) if mps_rep and mps_rep > 0 else None
                            ),
                        }
                    )
                    print(
                        f"  [Suite E] {sys_name:26s} B={B:3d}: Prolix={res_p['ms_rep']:.4f} ms/rep "
                        f"(peak={res_p['peak_bytes_in_use_mb']:.0f} MB, marg={max(0.0, marg_mb):.1f} MB/rep) | "
                        + (
                            f"MPS={mps_rep:.4f} ms/rep ({mps_r.get('total_gpu_memory_mb', 0):.0f} MB)"
                            if mps_ok
                            else "MPS=OOM/CLIFF"
                        )
                    )
                except Exception as exc_p:
                    rows.append({"B": B, "prolix_ok": False, "error": str(exc_p), "openmm_mps": mps_map.get(B)})
            records.append(
                {
                    "system": sys_name,
                    "n_real_atoms": int(bun.n_atoms),
                    "bucket_atoms": b_atoms,
                    "pme_grid": g_dims,
                    "high_batch_scaling": rows,
                }
            )
    finally:
        if prev_gp is None:
            os.environ.pop("PROLIX_PME_GRID_POINTS", None)
        else:
            os.environ["PROLIX_PME_GRID_POINTS"] = prev_gp
    return {"systems": records}


# ============================================================================
# SUITE F: Differentiable Moat (`jax.value_and_grad` `2k -> 92k` Atoms, `MK-73`)
# ============================================================================
def run_suite_f_differentiable_moat(
    dhfr_bundle: MolecularBundle,
    *,
    quick: bool = False,
) -> dict[str, Any]:
    """Benchmark `jax.value_and_grad` (`positions + charges/sigma/epsilon`) across `2k -> 92k` atoms."""
    print("=== [Suite F MK-73] Differentiable Moat (`jax.value_and_grad` 2k -> 92k Atoms) ===")
    n_warmup = 3
    n_repeats = 12 if quick else 30

    _, dhfr_omm_grid = compute_openmm_pme_params(
        np.array([62.23, 62.23, 62.23], dtype=np.float64), cutoff=9.0, ewald_error_tolerance=5e-4
    )
    diff_specs = [
        ("1UAO_solvated_2k", "trunc", 1963, 4096, 1024, (32, 32, 32)),
        ("1UBQ_solvated_7k", "trunc", 6658, 8192, 2048, (48, 48, 48)),
        ("Holo_Lysozyme_15k", "holo_15k", 15002, 16384, 4608, (54, 54, 54)),
        ("DHFR_24k_pme_openmm_grid", "trunc", 23558, 25000, 8192, dhfr_omm_grid),
        ("1AKE_solvated_45k", "1ake_45k", 44999, 50000, 16384, (80, 80, 80)),
        ("ApoA1_92k_pme", "apoa1_pme", 92224, 100000, 24576, (96, 96, 64)),
    ]

    records: list[dict[str, Any]] = []

    # 1. Implicit-solvent OBC2 GBSA + ACE tile-pair custom VJP (`5DFR_gbsa_implicit`, 2,489 atoms)
    _clear_all_caches(clear_jax=True)
    gbsa_bun, _, _ = build_implicit_protein_bundle(
        "5DFR_gbsa",
        "submodules/openmm/examples/benchmarks/5dfr_minimized.pdb",
        keep_chain=None,
        add_hydrogens_with_openmm=False,
        bucket_atoms=2560,
    )
    n_gbsa_pad = int(gbsa_bun.positions.shape[0])
    gbsa_tl = build_gbsa_tile_pair_list(
        gbsa_bun.positions,
        atom_mask=gbsa_bun.atom_mask,
        excl_indices=gbsa_bun.excl_dense_indices,
        excl_scales_vdw=gbsa_bun.excl_dense_scales_vdw,
        excl_scales_elec=gbsa_bun.excl_dense_scales_elec,
        use_morton=False,
    )

    def _energy_gbsa_all_params(
        r1: Any,
        q1: Any,
        rad1: Any,
        srad1: Any,
        sig1: Any,
        eps1: Any,
        _bun: MolecularBundle = gbsa_bun,
        _tl: Any = gbsa_tl,
    ) -> jax.Array:
        e_bnd_g = cast(
            "jax.Array",
            pallas_bonded_all_fused(
                r1,
                bond_indices=_bun.bond_idx,
                bond_params=_bun.bond_params,
                bond_mask=_bun.bond_mask,
                angle_indices=_bun.angle_idx,
                angle_params=_bun.angle_params,
                angle_mask=_bun.angle_mask,
                dihedral_indices=_bun.dihedral_idx,
                dihedral_params=_bun.dihedral_params,
                dihedral_mask=_bun.dihedral_mask,
                improper_indices=_bun.improper_idx,
                improper_params=_bun.improper_params,
                improper_mask=_bun.improper_mask,
                exception_pairs=_bun.exception_pairs,
                exception_sigmas=_bun.exception_sigmas,
                exception_epsilons=_bun.exception_epsilons,
                exception_chargeprods=_bun.exception_chargeprods,
                exception_mask=_bun.exception_mask,
                box=None,
            ),
        )
        e_gb = cast(
            "jax.Array",
            pallas_gbsa_tile(
                r1,
                q1,
                rad1,
                srad1,
                sig1,
                eps1,
                _tl,
                atom_mask=_bun.atom_mask,
                excl_indices=_bun.excl_dense_indices,
                excl_scales_vdw=_bun.excl_dense_scales_vdw,
                excl_scales_elec=_bun.excl_dense_scales_elec,
                return_forces=False,
            ),
        )
        return e_bnd_g + e_gb

    vg_gbsa = jax.value_and_grad(_energy_gbsa_all_params, argnums=(0, 1, 2, 3, 4, 5))
    gbsa_rows: list[dict[str, Any]] = []
    for B in (1, 4, 8):
        pos_b = jnp.broadcast_to(gbsa_bun.positions[None, :, :], (B, n_gbsa_pad, 3))
        q_b = jnp.broadcast_to(gbsa_bun.charges[None, :], (B, n_gbsa_pad))
        rad_b = jnp.broadcast_to(gbsa_bun.radii[None, :], (B, n_gbsa_pad))
        srad_b = jnp.broadcast_to(gbsa_bun.scaled_radii[None, :], (B, n_gbsa_pad))
        sig_b = jnp.broadcast_to(gbsa_bun.sigmas[None, :], (B, n_gbsa_pad))
        eps_b = jnp.broadcast_to(gbsa_bun.epsilons[None, :], (B, n_gbsa_pad))
        jit_vg_g = jax.jit(
            lambda r, q, rd, srd, s, e, _vg=vg_gbsa: jax.vmap(_vg)(r, q, rd, srd, s, e)
        )
        t_c0 = time.perf_counter()
        out_gw = jit_vg_g(pos_b, q_b, rad_b, srad_b, sig_b, eps_b)
        jax.block_until_ready(out_gw[0])
        compile_s = time.perf_counter() - t_c0
        for _ in range(n_warmup):
            out_gw = jit_vg_g(pos_b, q_b, rad_b, srad_b, sig_b, eps_b)
        jax.block_until_ready(out_gw[0])
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            out_gw = jit_vg_g(pos_b, q_b, rad_b, srad_b, sig_b, eps_b)
        jax.block_until_ready(out_gw[0])
        ms_batch = ((time.perf_counter() - t0) / n_repeats) * 1000.0
        mem = _jax_mem_mb()
        gbsa_rows.append(
            {
                "B": B,
                "cold_compile_s": float(compile_s),
                "ms_per_eval_batch": float(ms_batch),
                "ms_per_eval_per_replica": float(ms_batch / B),
                "live_vram_mb": mem["bytes_in_use_mb"],
                "peak_vram_mb": mem["peak_bytes_in_use_mb"],
                "openmm_cuda_supported": False,
            }
        )
        print(
            f"  [Suite F value_and_grad(r,q,rad,srad,sig,eps)] {'5DFR_gbsa_implicit':26s} B={B:2d}: "
            f"{ms_batch / B:.4f} ms/eval/rep (peak={mem['peak_bytes_in_use_mb']:.0f} MB)"
        )
    records.append(
        {
            "system": "5DFR_gbsa_implicit",
            "n_real_atoms": int(gbsa_bun.n_atoms),
            "bucket_atoms": n_gbsa_pad,
            "pme_grid": None,
            "num_valid_tile_pairs": int(np.asarray(gbsa_tl.num_valid_pairs).reshape(-1)[0]),
            "batch_scaling": gbsa_rows,
        }
    )

    # 2. Explicit-solvent PME ladder (`1UAO_2k -> ApoA1_92k`)
    for label, kind, n_real, b_atoms, b_waters, g_dims in diff_specs:
        _clear_all_caches(clear_jax=True)
        if kind == "trunc":
            bun = _cast_bundle_precision(
                _make_truncated_solvated_bundle(
                    dhfr_bundle, n_real_target=n_real, bucket_atoms=b_atoms, bucket_waters=b_waters
                ),
                jnp.float32,
            )
        elif kind == "holo_15k":
            bun = build_holo_protein_ligand_bundle(
                dhfr_bundle,
                n_prot_atoms=1960,
                n_ligand_atoms=46,
                n_real_target=n_real,
                bucket_atoms=b_atoms,
                bucket_waters=b_waters,
                ligand_seed=1,
            )
        elif kind == "1ake_45k":
            bun = build_solvated_1ake_bundle(
                dhfr_bundle, n_real_target=n_real, bucket_atoms=b_atoms, bucket_waters=b_waters
            )
        elif kind == "apoa1_pme":
            bun, _, _ = build_apoa1_bundle(
                bucket_atoms=b_atoms, bucket_waters=b_waters, electrostatic_method="pme", include_constraints=False
            )
        else:
            continue

        box_vec = jnp.diag(bun.box).astype(jnp.float32)
        disp_fn, _ = displacement_fn_for_bundle(bun)
        nl_fn = nl_mod.make_neighbor_list_fn(
            disp_fn,
            box_vec,
            float(bun.cutoff_distance),
            capacity_multiplier=1.25,
            dr_threshold=0.5,
            atom_mask=bun.atom_mask,
        )
        nbr0 = _encode_bundle_nl(cast("Any", nl_fn).allocate(bun.positions), bun)
        tp_nl0 = build_tile_pair_neighbor_list(
            nbr0.idx,
            excl_indices=bun.excl_dense_indices,
            excl_scales_vdw=bun.excl_dense_scales_vdw,
            excl_scales_elec=bun.excl_dense_scales_elec,
            atom_mask=bun.atom_mask,
            positions=bun.positions,
            box=box_vec,
            cutoff=float(bun.cutoff_distance),
        )

        def _energy_mk51_all_params(
            r1: Any,
            q1: Any,
            sig1: Any,
            eps1: Any,
            _bun: MolecularBundle = bun,
            _tp: Any = tp_nl0,
            _bx: Any = box_vec,
            _gd: tuple[int, int, int] = g_dims,
        ) -> jax.Array:
            e_bnd = cast(
                "jax.Array",
                pallas_bonded_all_fused(
                    r1,
                    bond_indices=_bun.bond_idx,
                    bond_params=_bun.bond_params,
                    bond_mask=_bun.bond_mask,
                    angle_indices=_bun.angle_idx,
                    angle_params=_bun.angle_params,
                    angle_mask=_bun.angle_mask,
                    dihedral_indices=_bun.dihedral_idx,
                    dihedral_params=_bun.dihedral_params,
                    dihedral_mask=_bun.dihedral_mask,
                    improper_indices=_bun.improper_idx,
                    improper_params=_bun.improper_params,
                    improper_mask=_bun.improper_mask,
                    exception_pairs=_bun.exception_pairs,
                    exception_sigmas=_bun.exception_sigmas,
                    exception_epsilons=_bun.exception_epsilons,
                    exception_chargeprods=_bun.exception_chargeprods,
                    exception_mask=_bun.exception_mask,
                    box=_bx,
                ),
            )
            e_nb = cast(
                "jax.Array",
                pallas_nonbonded_tile_pairs(
                    r1,
                    q1,
                    sig1,
                    eps1,
                    _tp,
                    atom_mask=_bun.atom_mask,
                    box=_bx,
                    pme_alpha=float(_bun.pme_alpha),
                    cutoff=float(_bun.cutoff_distance),
                    return_forces=False,
                ),
            )
            e_rec = cast(
                "jax.Array",
                spme_energy_with_forces(
                    r1, q1, _bun.atom_mask, _bx, _gd, float(_bun.pme_alpha), 4, True
                ),
            )
            return e_bnd + e_nb + e_rec

        vg_fn = jax.value_and_grad(_energy_mk51_all_params, argnums=(0, 1, 2, 3))
        b_list = (1, 4) if b_atoms >= 50000 else (1, 4, 8)
        if quick and b_atoms >= 50000:
            b_list = (1,)

        b_rows: list[dict[str, Any]] = []
        for B in b_list:
            pos_b = jnp.broadcast_to(bun.positions[None, :, :], (B, b_atoms, 3))
            q_b = jnp.broadcast_to(bun.charges[None, :], (B, b_atoms))
            sig_b = jnp.broadcast_to(bun.sigmas[None, :], (B, b_atoms))
            eps_b = jnp.broadcast_to(bun.epsilons[None, :], (B, b_atoms))
            jit_vg = jax.jit(lambda r, q, s, e, _vg=vg_fn: jax.vmap(_vg)(r, q, s, e))
            t_c0 = time.perf_counter()
            out_w = jit_vg(pos_b, q_b, sig_b, eps_b)
            jax.block_until_ready(out_w[0])
            compile_s = time.perf_counter() - t_c0
            for _ in range(n_warmup):
                out_w = jit_vg(pos_b, q_b, sig_b, eps_b)
            jax.block_until_ready(out_w[0])
            t0 = time.perf_counter()
            for _ in range(n_repeats):
                out_w = jit_vg(pos_b, q_b, sig_b, eps_b)
            jax.block_until_ready(out_w[0])
            ms_batch = ((time.perf_counter() - t0) / n_repeats) * 1000.0
            mem = _jax_mem_mb()
            b_rows.append(
                {
                    "B": B,
                    "cold_compile_s": float(compile_s),
                    "ms_per_eval_batch": float(ms_batch),
                    "ms_per_eval_per_replica": float(ms_batch / B),
                    "live_vram_mb": mem["bytes_in_use_mb"],
                    "peak_vram_mb": mem["peak_bytes_in_use_mb"],
                    "openmm_cuda_supported": False,
                }
            )
            print(
                f"  [Suite F value_and_grad(r,q,sig,eps)] {label:26s} B={B:2d}: "
                f"{ms_batch / B:.4f} ms/eval/rep (peak={mem['peak_bytes_in_use_mb']:.0f} MB)"
            )
        records.append(
            {
                "system": label,
                "n_real_atoms": int(bun.n_atoms),
                "bucket_atoms": b_atoms,
                "pme_grid": g_dims,
                "num_valid_tile_pairs": int(tp_nl0.num_valid_pairs),
                "batch_scaling": b_rows,
            }
        )
    return {"differentiable_systems": records}


def _evaluate_v35_gates(payload: dict[str, Any]) -> dict[str, Any]:
    """Summarize MK-65..74 target checks on the measured payload."""
    gates: dict[str, Any] = {}
    sab = payload.get("suites_a_and_b") or {}
    gbsa = sab.get("suite_a_gbsa") or {}
    if gbsa.get("batch_scaling"):
        b_map = {int(r["B"]): float(r["ms_per_step_per_replica"]) for r in gbsa["batch_scaling"]}
        gates["MK-67_5DFR_gbsa_B1_ms"] = b_map.get(1)
        gates["MK-67_5DFR_gbsa_B4_ms_rep"] = b_map.get(4)
        gates["MK-67_5DFR_gbsa_B32_ms_rep"] = b_map.get(32)
    for rec in sab.get("explicit_ladder") or []:
        sys_n = rec.get("system")
        b_map = {int(r["B"]): float(r["ms_per_step_per_replica"]) for r in rec.get("batch_scaling", [])}
        if sys_n == "DHFR_24k_pme_openmm_grid":
            v_dhfr = b_map.get(1)
            gates["MK-66_DHFR_60grid_B1_ms"] = v_dhfr
            gates["MK-66_DHFR_60grid_B1_pass_lt_0p16"] = (
                v_dhfr < 0.16 if v_dhfr is not None else None
            )
            gates["MK-65_DHFR_60grid_B4_ms_rep"] = b_map.get(4)
            gates["MK-65_DHFR_60grid_B32_ms_rep"] = b_map.get(32)
        elif sys_n == "ApoA1_92k_pme":
            v_apoa1 = b_map.get(1)
            gates["MK-65_ApoA1_92k_B1_ms"] = v_apoa1
            gates["MK-65_ApoA1_92k_B1_pass_lt_0p55"] = (
                v_apoa1 < 0.55 if v_apoa1 is not None else None
            )
            gates["MK-65_ApoA1_92k_B4_ms_rep"] = b_map.get(4)
        elif sys_n == "Cellulose_408k_pme":
            gates["MK-71_Cellulose_408k_B1_ms"] = b_map.get(1)
    sc = payload.get("suite_c") or {}
    rag = sc.get("cross_size_ragged_screen") or {}
    if "prolix_ms_per_step_per_replica" in rag:
        gates["MK-68_SuiteC2_ragged_ms_rep"] = float(rag["prolix_ms_per_step_per_replica"])
        gates["MK-68_SuiteC2_beat_v34_0p1504"] = float(rag["prolix_ms_per_step_per_replica"]) < 0.1504
        gates["MK-68_SuiteC2_xla_compile_count"] = rag.get("xla_executable_count")
    return gates


def main() -> None:
    with contextlib.suppress(Exception):
        cast("Any", sys.stdout).reconfigure(line_buffering=True)
        cast("Any", sys.stderr).reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(
        description="Pallas v3.5 Release Benchmark Ladder (6-Suite Multi-GPU Matrix)"
    )
    parser.add_argument("--gpu-tag", type=str, default="gpu", help="GPU tag (l40s, h200, b300, a100).")
    parser.add_argument(
        "--suites",
        type=str,
        default="A,B,C,D,E,F",
        help="Comma-separated suites to run (subset of A,B,C,D,E,F; default: A,B,C,D,E,F).",
    )
    parser.add_argument(
        "--systems",
        type=str,
        default="",
        help="Optional comma-separated filter of Suite A/B system names.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4,8,16,32",
        help="Comma-separated batch sizes B for Suites A/B/C.",
    )
    parser.add_argument(
        "--nonbonded-engine",
        type=str,
        default="",
        help="Optional explicit nonbonded_engine override (default: EnsemblePlan default / auto).",
    )
    parser.add_argument(
        "--pme-tier-mode",
        type=str,
        default="auto",
        help="Suite C2 ragged PME tier mode (auto, grouped, uniform).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "benchmarks" / "261111_l40s_pallas_v35_e2e_ensemble_buckets.json",
        help="Output JSON path.",
    )
    parser.add_argument("--quick", action="store_true", help="Faster step/warmup counts for iteration.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing ckpt_v35 checkpoints.")
    parser.add_argument(
        "--allow-opencl-fallback",
        action="store_true",
        help="Allow OpenMM OpenCL fallback (local debugging only; NEVER for Engaging artifacts).",
    )
    args = parser.parse_args()

    _CONFIG["allow_opencl_fallback"] = bool(args.allow_opencl_fallback)

    suites_set = {s.strip().upper() for s in args.suites.split(",") if s.strip()}
    systems_filter = {s.strip() for s in args.systems.split(",") if s.strip()} or None
    batch_sizes = tuple(int(x.strip()) for x in args.batch_sizes.split(",") if x.strip())
    nb_engine = args.nonbonded_engine.strip() or None

    dev = jax.devices()[0]
    dev_kind = getattr(dev, "device_kind", str(dev))
    omm_inv = get_openmm_platform_inventory()
    print(f"=== Pallas v3.5 Release Benchmark Ladder on {dev_kind} ===")
    print(f"OpenMM Platform Inventory: {json.dumps(omm_inv)}")
    if not omm_inv.get("has_cuda_platform") and not _CONFIG["allow_opencl_fallback"]:
        raise RuntimeError(f"OpenMM CUDA platform not registered! Inventory={json.dumps(omm_inv)}")

    ff_path = b_dhfr._resolve_ff_path()
    sys_data = b_dhfr._build_dhfr_system_dict(ff_path)
    dhfr_bundle = b_dhfr._build_bundle_from_sys_data(sys_data)

    t_start = time.perf_counter()
    ckpt_meta = {
        "schema_version": "pallas_v35_release_ladder_v1",
        "gpu_tag": args.gpu_tag,
        "suites_requested": args.suites,
        "systems_filter": sorted(systems_filter) if systems_filter else None,
        "device_kind": dev_kind,
        "jax_version": jax.__version__,
        "openmm_inventory": omm_inv,
        "quick_mode": bool(args.quick),
    }

    suites_ab = None
    if "A" in suites_set or "B" in suites_set:
        suites_ab = run_suites_a_and_b_ladder(
            dhfr_bundle,
            batch_sizes=batch_sizes,
            quick=args.quick,
            systems_filter=systems_filter,
            nonbonded_engine=nb_engine,
            resume_checkpoint=not args.no_resume,
            checkpoint_out=args.out if systems_filter is None else None,
            checkpoint_meta=ckpt_meta,
        )

    suite_c = None
    if "C" in suites_set:
        suite_c = run_suite_c_congeneric_and_ragged(
            dhfr_bundle,
            batch_sizes=batch_sizes,
            quick=args.quick,
            nonbonded_engine=nb_engine,
            pme_tier_mode=args.pme_tier_mode,
        )

    suite_d = None
    if "D" in suites_set:
        suite_d = run_suite_d_openmm_frontend_and_npt(dhfr_bundle, quick=args.quick)

    suite_e = None
    if "E" in suites_set:
        suite_e = run_suite_e_high_batch_oom_cliff(dhfr_bundle, quick=args.quick)

    suite_f = None
    if "F" in suites_set:
        suite_f = run_suite_f_differentiable_moat(dhfr_bundle, quick=args.quick)

    total_s = time.perf_counter() - t_start
    payload: dict[str, Any] = {
        **ckpt_meta,
        "total_wall_s": float(total_s),
        "suites_a_and_b": suites_ab,
        "suite_c": suite_c,
        "suite_d_openmm_frontend_and_npt": suite_d,
        "suite_e_high_batch_oom_cliff": suite_e,
        "suite_f_differentiable_moat": suite_f,
    }
    payload["v35_gate_summary"] = _evaluate_v35_gates(payload)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    if systems_filter is None:
        mirror_dir = Path("/orcd/pool/007/maarxaru/slurm_logs/prolix/ckpt_v35")
        if mirror_dir.parent.exists():
            mirror_dir.mkdir(parents=True, exist_ok=True)
            (mirror_dir / args.out.name).write_text(json.dumps(payload, indent=2))

    print("=== V35_GATE_SUMMARY ===")
    print(json.dumps(payload["v35_gate_summary"], indent=2))
    print("=== V35_JSON_BEGIN ===")
    print(json.dumps(payload))
    print("=== V35_JSON_END ===")
    print(f"=== Saved Pallas v3.5 Release Ladder JSON to {args.out} ({total_s:.1f}s) ===")


if __name__ == "__main__":
    main()
