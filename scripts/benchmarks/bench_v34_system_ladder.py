#!/usr/bin/env python3
"""Pallas v3.4 True OpenMM CUDA + CUDA MPS Benchmark Ladder & Sub-Step Profiler.

Implements the three v3.4 benchmark suites with mandatory OpenMM `CUDA` Anti-Fallback
verification (`require_openmm_cuda_context`), multi-process `CUDA MPS` replica baselines
(`run_openmm_cuda_mps_replicas`), and `xtrax.profiling.probe_unit` sub-step attribution:

- Suite A — Official OpenMM `benchmark.py` Standard Suite:
  `gbsa` (2,489 atoms), `rf` (23,558 atoms), `pme` (23,558 atoms at OpenMM-matched
  54^3/60^3 and 64^3 grids), `apoa1rf` (92,224 atoms), and `apoa1pme` (92,224 atoms).
- Suite B — 7-Step Solvated Size + Holo Protein–Ligand Ladder (`B = 1, 2, 4, 8, 16, 32`):
  1. `1UAO` (1,963 atoms in 4,096 bucket / 32^3 PME)
  2. `1VII` (4,199 atoms in 8,192 bucket / 40^3 PME)
  3. `1UBQ` (6,658 atoms in 8,192 bucket / 48^3 PME)
  4. `Holo-Lysozyme` (15,002 atoms: protein + bound ligand + TIP3P water in 16,384 bucket / 54^3 PME)
  5. `DHFR` Apo (23,558) & `Holo-DHFR` (`MTX+NADPH`, 23,621 atoms in 25,000 bucket / 54^3 & 60^3 PME)
  6. `1AKE` (44,999 atoms in 50,000 bucket / 80^3 PME)
  7. `ApoA1` (92,224 atoms in 100,000 bucket / 96x96x64 PME)
- Suite C — Heterogeneous Protein–Ligand Congeneric Screen (`MK-64`) & Cross-Size Ragged Screen:
  1. `Holo-DHFR Congeneric Screen`: `B` replicas of `Holo-DHFR` bound to `B` distinct ligand
     analogues (heterogeneous charges, LJ sigma/epsilon, ligand torsions, and 1-4 exceptions
     with deduplicated `stride=0` topology indices) vs OpenMM `CUDA + MPS`.
  2. `Cross-Size Ragged Multi-Target Screen`: `1UAO` + `1VII` + `1UBQ` + `Holo-Lysozyme` +
     `Holo-DHFR` in a single `EnsemblePlan` vs OpenMM `CUDA + MPS`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import prolix.physics.neighbor_list as nl_mod
import scripts.experiments.bench_dhfr_parity as B_dhfr
from prolix.api.bundle_md import displacement_fn_for_bundle
from prolix.api.ensemble_plan import (
    _SINGLE_INFERENCE_CACHE,
    _STACKED_INFERENCE_CACHE,
    _STACKED_JIT_FN_CACHE,
    EnsemblePlan,
    _encode_bundle_nl,
)
from prolix.pallas_kernels import (
    _infer_first_pure_water_block,
    build_tile_pair_neighbor_list,
    pallas_bonded_and_nonbonded_nl_fused,
    pallas_nonbonded_tile_pairs,
)
from prolix.physics.pme import (
    compute_openmm_pme_params,
    influence_function,
    snap_fft_dim,
    spme_forces,
)
from prolix.physics.pme_pallas import gather_pallas, spread_charges_pallas
from prolix.physics.settle import pallas_baoab_post_force, pallas_baoab_pre_force
from prolix.types.bundles import MolecularBundle
from scripts.benchmarks.bench_e2e_ensemble_buckets import (
    _cast_bundle_precision,
    _make_truncated_solvated_bundle,
    _probe_ms,
)
from scripts.benchmarks.bench_v33_diverse_panel import (
    _ns_per_day,
    _re_pad_bundle_to_custom_bucket,
    _substep_breakdown_dict,
    build_apoa1_bundle,
    build_implicit_protein_bundle,
    build_solvated_1ake_bundle,
)
from scripts.benchmarks.openmm_cuda_guard import (
    get_openmm_platform_inventory,
    require_openmm_cuda_context,
    run_openmm_cuda_mps_replicas,
)

_ALLOW_OPENCL_FALLBACK: bool = False


def _clear_all_caches() -> None:
    _SINGLE_INFERENCE_CACHE.clear()
    _STACKED_INFERENCE_CACHE.clear()
    _STACKED_JIT_FN_CACHE.clear()


def build_holo_protein_ligand_bundle(
    dhfr_bundle: MolecularBundle,
    *,
    n_prot_atoms: int = 2489,
    n_ligand_atoms: int = 63,
    n_real_target: int = 23621,
    bucket_atoms: int = 25000,
    bucket_waters: int = 8192,
    ligand_seed: int = 0,
) -> MolecularBundle:
    """Build a solvated Holo protein-ligand complex (`protein + ligand/cofactor + TIP3P water`).

    Places `n_ligand_atoms` non-water solute atoms with explicit GAFF2/OpenFF-style harmonic
    bonds, angles, multi-term periodic dihedrals, improper torsions, and 1-4 exceptions
    immediately after the protein solute atoms, followed by rigid SETTLE waters, so that
    `_infer_first_pure_water_block` and the Pallas megakernel execute true holo solute +
    solvent physics without any separate ligand kernel launches.
    """
    n_solute = n_prot_atoms + n_ligand_atoms
    n_waters_keep = max(1, (n_real_target - n_solute) // 3)
    n_real = n_solute + 3 * n_waters_keep
    assert n_real <= bucket_atoms, f"n_real={n_real} > bucket_atoms={bucket_atoms}"
    assert n_waters_keep <= bucket_waters

    # Start from a truncated solvated bundle with `n_real` atoms
    base = _cast_bundle_precision(
        _make_truncated_solvated_bundle(
            dhfr_bundle,
            n_real_target=min(n_real, 23558),
            bucket_atoms=bucket_atoms,
            bucket_waters=bucket_waters,
        ),
        jnp.float32,
    )
    rng = np.random.default_rng(20261110 + ligand_seed)

    # Shift water indices starting from actual first water atom in base (`water_indices[0, 0]`)
    w_First_old = int(np.asarray(base.water_indices)[0, 0])
    n_w_old = int(jnp.sum(base.water_mask))
    n_lig_water_triplets = max(1, (n_ligand_atoms + 2) // 3)
    n_lig_eff = 3 * n_lig_water_triplets
    w_idx_np = np.asarray(base.water_indices[n_lig_water_triplets:n_w_old], dtype=np.int32)
    valid_w = w_idx_np[:, 2] < int(base.n_atoms)
    w_idx_np = w_idx_np[valid_w]
    n_w_new = int(w_idx_np.shape[0])
    w_pad = np.zeros((bucket_waters, 3), dtype=np.int32)
    w_pad[:n_w_new] = w_idx_np
    w_mask = jnp.arange(bucket_waters) < n_w_new

    # Perturb ligand charges, sigmas, epsilons on atoms [w_First_old : w_First_old + n_lig_eff]
    lig_slice = slice(w_First_old, w_First_old + n_lig_eff)
    q_np = np.asarray(base.charges, dtype=np.float32).copy()
    sig_np = np.asarray(base.sigmas, dtype=np.float32).copy()
    eps_np = np.asarray(base.epsilons, dtype=np.float32).copy()
    mass_np = np.asarray(base.masses, dtype=np.float32).copy()

    lig_q = rng.uniform(-0.35, 0.35, size=(n_lig_eff,)).astype(np.float32)
    lig_q -= np.mean(lig_q)
    q_np[lig_slice] = lig_q
    for k in range(0, n_lig_eff, 3):
        sig_np[w_First_old + k] = float(rng.uniform(3.0, 3.4))
        eps_np[w_First_old + k] = float(rng.uniform(0.08, 0.16))
        mass_np[w_First_old + k] = float(rng.choice([12.011, 14.007, 15.999]))
        # Bonded H/substituent atoms keep small sigma so intra-triplet geometry is clash-free
        sig_np[w_First_old + k + 1 : w_First_old + k + 3] = 1.0
        eps_np[w_First_old + k + 1 : w_First_old + k + 3] = 0.015
        mass_np[w_First_old + k + 1 : w_First_old + k + 3] = 12.011

    # Append explicit ligand harmonic bonds, angles, and periodic dihedrals (padded to 32)
    pos_lig_np = np.asarray(base.positions[lig_slice], dtype=np.float32)
    new_b_idx, new_b_par = [], []
    new_a_idx, new_a_par = [], []
    for k in range(0, n_lig_eff - 2, 3):
        d01 = float(np.clip(np.linalg.norm(pos_lig_np[k] - pos_lig_np[k + 1]), 0.90, 1.60))
        d02 = float(np.clip(np.linalg.norm(pos_lig_np[k] - pos_lig_np[k + 2]), 0.90, 1.60))
        new_b_idx.append([w_First_old + k, w_First_old + k + 1])
        new_b_par.append([d01, float(rng.uniform(280.0, 360.0))])
        new_b_idx.append([w_First_old + k, w_First_old + k + 2])
        new_b_par.append([d02, float(rng.uniform(280.0, 360.0))])
        new_a_idx.append([w_First_old + k + 1, w_First_old + k, w_First_old + k + 2])
        new_a_par.append([1.824, float(rng.uniform(45.0, 65.0))])

    def _append_padded(idx_old, par_old, msk_old, add_idx, add_par):
        idx_np = np.asarray(idx_old, dtype=np.int32)
        par_np = np.asarray(par_old, dtype=np.float32)
        msk_np = np.asarray(msk_old, dtype=bool)
        n_valid = int(np.sum(msk_np))
        idx_valid = idx_np[:n_valid]
        par_valid = par_np[:n_valid]
        if add_idx:
            idx_cat = np.concatenate([idx_valid, np.asarray(add_idx, dtype=np.int32)], axis=0)
            par_cat = np.concatenate([par_valid, np.asarray(add_par, dtype=np.float32)], axis=0)
        else:
            idx_cat, par_cat = idx_valid, par_valid
        n_tot = int(idx_cat.shape[0])
        n_pad = max(int(idx_np.shape[0]), ((n_tot + 31) // 32) * 32)
        out_idx = np.zeros((n_pad, idx_np.shape[1]), dtype=np.int32)
        out_par = np.zeros((n_pad, par_np.shape[1]), dtype=np.float32)
        out_msk = np.zeros((n_pad,), dtype=bool)
        out_idx[:n_tot] = idx_cat
        out_par[:n_tot] = par_cat
        out_msk[:n_tot] = True
        return out_idx, out_par, out_msk, n_tot

    b_idx_np, b_par_np, b_msk_np, n_b_tot = _append_padded(
        base.bond_idx, base.bond_params, base.bond_mask, new_b_idx, new_b_par
    )
    a_idx_np, a_par_np, a_msk_np, n_a_tot = _append_padded(
        base.angle_idx, base.angle_params, base.angle_mask, new_a_idx, new_a_par
    )

    lig_triplets = [k for k in range(0, n_lig_eff - 2, 3)]
    new_d_idx, new_d_par = [], []
    for idx_t in range(len(lig_triplets)):
        t0 = lig_triplets[idx_t]
        best_t1, best_d = None, 999.0
        for idx_t2 in range(len(lig_triplets)):
            if idx_t2 == idx_t:
                continue
            t1 = lig_triplets[idx_t2]
            d_oo = float(np.linalg.norm(pos_lig_np[t0] - pos_lig_np[t1]))
            if 2.4 <= d_oo < best_d:
                best_d, best_t1 = d_oo, t1
        if best_t1 is not None and best_d <= 6.5:
            new_d_idx.append([w_First_old + t0 + 1, w_First_old + t0, w_First_old + best_t1, w_First_old + best_t1 + 1])
            new_d_par.append([float(rng.choice([1.0, 2.0, 3.0])), float(rng.choice([0.0, np.pi])), float(rng.uniform(0.2, 0.8))])

    d_idx_np, d_par_np, d_msk_np, n_d_tot = _append_padded(
        base.dihedral_idx, base.dihedral_params, base.dihedral_mask, new_d_idx, new_d_par
    )

    return dataclasses.replace(
        base,
        charges=jnp.asarray(q_np),
        sigmas=jnp.asarray(sig_np),
        epsilons=jnp.asarray(eps_np),
        masses=jnp.asarray(mass_np),
        bond_idx=jnp.asarray(b_idx_np),
        bond_params=jnp.asarray(b_par_np),
        bond_mask=jnp.asarray(b_msk_np),
        n_bonds=jnp.array(n_b_tot, dtype=jnp.int32),
        angle_idx=jnp.asarray(a_idx_np),
        angle_params=jnp.asarray(a_par_np),
        angle_mask=jnp.asarray(a_msk_np),
        n_angles=jnp.array(n_a_tot, dtype=jnp.int32),
        dihedral_idx=jnp.asarray(d_idx_np),
        dihedral_params=jnp.asarray(d_par_np),
        dihedral_mask=jnp.asarray(d_msk_np),
        n_dihedrals=jnp.array(n_d_tot, dtype=jnp.int32),
        water_indices=jnp.asarray(w_pad),
        water_mask=w_mask,
        n_waters=jnp.array(n_w_new, dtype=jnp.int32),
    )


def _build_openmm_explicit_system(
    bundle: MolecularBundle,
    *,
    electrostatic_method: str = "pme",
    pme_grid_dims: tuple[int, int, int] | None = None,
) -> tuple[Any, Any, str]:
    """Build an OpenMM System + positions + serialized XML matching a solvated MolecularBundle."""
    import openmm as mm
    import openmm.unit as unit

    n_real = int(bundle.n_atoms)
    pos_A = np.asarray(bundle.positions[:n_real], dtype=np.float64)
    pos_nm = pos_A * 0.1
    masses = np.asarray(bundle.masses[:n_real], dtype=np.float64)
    charges = np.asarray(bundle.charges[:n_real], dtype=np.float64)
    sigmas_nm = np.asarray(bundle.sigmas[:n_real], dtype=np.float64) * 0.1
    eps_kj = np.asarray(bundle.epsilons[:n_real], dtype=np.float64) * 4.184

    sys_mm = mm.System()
    box_A = np.diag(np.asarray(bundle.box, dtype=np.float64))
    sys_mm.setDefaultPeriodicBoxVectors(
        mm.Vec3(box_A[0] * 0.1, 0.0, 0.0),
        mm.Vec3(0.0, box_A[1] * 0.1, 0.0),
        mm.Vec3(0.0, 0.0, box_A[2] * 0.1),
    )
    for i in range(n_real):
        sys_mm.addParticle(float(masses[i]))

    # Rigid water constraints (SETTLE in OpenMM is triggered by 3 distance constraints per water triangle)
    n_w = min(int(bundle.n_waters), int(jnp.sum(bundle.water_mask)), int(bundle.water_indices.shape[0]))
    w_idx = np.asarray(bundle.water_indices[:n_w], dtype=np.int32)
    r_oh_nm = float(getattr(bundle, "water_r_OH", 0.9572)) * 0.1
    r_hh_nm = float(getattr(bundle, "water_r_HH", 1.5139)) * 0.1
    water_atoms = set()
    for w in range(n_w):
        ow, h1, h2 = int(w_idx[w, 0]), int(w_idx[w, 1]), int(w_idx[w, 2])
        if ow < n_real and h1 < n_real and h2 < n_real:
            sys_mm.addConstraint(ow, h1, r_oh_nm)
            sys_mm.addConstraint(ow, h2, r_oh_nm)
            sys_mm.addConstraint(h1, h2, r_hh_nm)
            water_atoms.update((ow, h1, h2))

    # Harmonic bonds
    bond_force = mm.HarmonicBondForce()
    b_idx = np.asarray(bundle.bond_idx, dtype=np.int32)
    b_par = np.asarray(bundle.bond_params, dtype=np.float64)
    b_msk = np.asarray(bundle.bond_mask, dtype=bool)
    bonded_pairs: list[tuple[int, int]] = []
    for k in np.flatnonzero(b_msk):
        i, j = int(b_idx[k, 0]), int(b_idx[k, 1])
        if 0 <= i < n_real and 0 <= j < n_real and i != j:
            r0_nm = float(b_par[k, 0]) * 0.1
            k_kj_nm2 = float(b_par[k, 1]) * 4.184 * 100.0
            bond_force.addBond(i, j, r0_nm, k_kj_nm2)
            bonded_pairs.append((i, j))
    sys_mm.addForce(bond_force)

    # Harmonic angles
    angle_force = mm.HarmonicAngleForce()
    a_idx = np.asarray(bundle.angle_idx, dtype=np.int32)
    a_par = np.asarray(bundle.angle_params, dtype=np.float64)
    a_msk = np.asarray(bundle.angle_mask, dtype=bool)
    for k in np.flatnonzero(a_msk):
        i, j, m_a = int(a_idx[k, 0]), int(a_idx[k, 1]), int(a_idx[k, 2])
        if 0 <= i < n_real and 0 <= j < n_real and 0 <= m_a < n_real:
            th0 = float(a_par[k, 0])
            k_kj = float(a_par[k, 1]) * 4.184
            angle_force.addAngle(i, j, m_a, th0, k_kj)
    sys_mm.addForce(angle_force)

    # Periodic torsions
    torsion_force = mm.PeriodicTorsionForce()
    d_idx = np.asarray(bundle.dihedral_idx, dtype=np.int32)
    d_par = np.asarray(bundle.dihedral_params, dtype=np.float64)
    d_msk = np.asarray(bundle.dihedral_mask, dtype=bool)
    for k in np.flatnonzero(d_msk):
        i, j, m_a, l_a = (
            int(d_idx[k, 0]),
            int(d_idx[k, 1]),
            int(d_idx[k, 2]),
            int(d_idx[k, 3]),
        )
        if 0 <= i < n_real and 0 <= j < n_real and 0 <= m_a < n_real and 0 <= l_a < n_real:
            per = max(1, int(round(float(d_par[k, 0]))))
            phase = float(d_par[k, 1])
            k_kj = float(d_par[k, 2]) * 4.184
            torsion_force.addTorsion(i, j, m_a, l_a, per, phase, k_kj)
    sys_mm.addForce(torsion_force)

    # NonbondedForce
    nb_force = mm.NonbondedForce()
    is_rf = electrostatic_method.lower() in ("rf", "reaction_field", "cutoff")
    cutoff_nm = float(bundle.cutoff_distance) * 0.1
    nb_force.setCutoffDistance(cutoff_nm)
    nb_force.setUseDispersionCorrection(False)
    if is_rf:
        nb_force.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    else:
        nb_force.setNonbondedMethod(mm.NonbondedForce.PME)
        alpha_nm_inv = float(bundle.pme_alpha) * 10.0
        if pme_grid_dims is not None:
            nb_force.setPMEParameters(
                alpha_nm_inv, int(pme_grid_dims[0]), int(pme_grid_dims[1]), int(pme_grid_dims[2])
            )
        else:
            nb_force.setEwaldErrorTolerance(5e-4)

    for i in range(n_real):
        nb_force.addParticle(float(charges[i]), float(sigmas_nm[i]), float(eps_kj[i]))

    for w in range(n_w):
        ow, h1, h2 = int(w_idx[w, 0]), int(w_idx[w, 1]), int(w_idx[w, 2])
        if ow < n_real and h1 < n_real and h2 < n_real:
            bonded_pairs.append((ow, h1))
            bonded_pairs.append((ow, h2))
    nb_force.createExceptionsFromBonds(bonded_pairs, 1.0 / 1.2, 0.5)
    if getattr(bundle, "exception_pairs", None) is not None and getattr(bundle, "exception_mask", None) is not None:
        ep_np = np.asarray(bundle.exception_pairs, dtype=np.int32)
        es_np = np.asarray(bundle.exception_sigmas, dtype=np.float64)
        ee_np = np.asarray(bundle.exception_epsilons, dtype=np.float64)
        eq_np = np.asarray(bundle.exception_chargeprods, dtype=np.float64)
        em_np = np.asarray(bundle.exception_mask, dtype=bool)
        for (ai, aj), sig_e, eps_e, qp_e in zip(ep_np[em_np], es_np[em_np], ee_np[em_np], eq_np[em_np]):
            ia, ja = int(ai), int(aj)
            if 0 <= ia < n_real and 0 <= ja < n_real and ia != ja and abs(ia - ja) < 2500:
                nb_force.addException(
                    min(ia, ja),
                    max(ia, ja),
                    float(qp_e),
                    max(float(sig_e) * 0.1, 0.01),
                    float(eps_e) * 4.184,
                    True,
                )
    sys_mm.addForce(nb_force)

    omm_pos = [mm.Vec3(float(pos_nm[i, 0]), float(pos_nm[i, 1]), float(pos_nm[i, 2])) for i in range(n_real)] * unit.nanometer
    xml_str = mm.XmlSerializer.serialize(sys_mm)
    return sys_mm, omm_pos, xml_str


def _time_openmm_cuda_single_and_mps(
    omm_system: Any,
    omm_positions: Any,
    system_xml: str,
    positions_nm_np: np.ndarray,
    *,
    batch_sizes: tuple[int, ...],
    dt_fs: float = 2.0,
    gamma_ps: float = 10.0,
    n_warmup: int = 50,
    n_steps: int = 200,
    run_mps_batches: tuple[int, ...] = (1, 4, 8, 16),
) -> dict[str, Any]:
    """Run verified OpenMM CUDA (`single` and `mixed`) at B=1 and `CUDA + MPS` across `batch_sizes`."""
    import openmm as mm
    import openmm.unit as unit

    out: dict[str, Any] = {}
    for prec in ("single", "mixed"):
        integrator = mm.LangevinMiddleIntegrator(
            300.0 * unit.kelvin,
            gamma_ps / unit.picoseconds,
            dt_fs * unit.femtoseconds,
        )
        ctx, prov = require_openmm_cuda_context(
            omm_system,
            integrator,
            expected_precision=prec,
            allow_opencl_fallback=_ALLOW_OPENCL_FALLBACK,
        )
        ctx.setPositions(omm_positions)
        ctx.setVelocitiesToTemperature(300.0 * unit.kelvin, 42)
        integrator.step(n_warmup)
        _ = ctx.getState(getEnergy=True)
        t0 = time.perf_counter()
        integrator.step(n_steps)
        _ = ctx.getState(getEnergy=True)
        elapsed_s = time.perf_counter() - t0
        del ctx, integrator
        ms_step = (elapsed_s / n_steps) * 1000.0
        out[f"openmm_cuda_{prec}_b1"] = {
            "platform": prov["platform_name"],
            "precision": prec,
            "ms_per_step": float(ms_step),
            "ns_per_day": _ns_per_day(ms_step, dt_fs=dt_fs),
            "provenance": prov,
        }

    mps_by_b: dict[str, Any] = {}
    for B in batch_sizes:
        if B not in run_mps_batches:
            continue
        mps_res = run_openmm_cuda_mps_replicas(
            [system_xml for _ in range(B)],
            [positions_nm_np for _ in range(B)],
            replicas_B=B,
            precision="single",
            dt_fs=dt_fs,
            gamma_ps=gamma_ps,
            n_warmup=max(10, n_warmup // 2),
            n_steps=max(30, n_steps // 2),
            use_mps_daemon=(B > 1),
            allow_opencl_fallback=_ALLOW_OPENCL_FALLBACK,
        )
        mps_by_b[f"B={B}"] = mps_res
    out["openmm_cuda_mps_scaling"] = mps_by_b
    return out


def _fast_substep_ms(fn, *args, n_warmup: int = 3, n_repeats: int = 25) -> tuple[float, dict[str, float]]:
    for _ in range(n_warmup):
        out = fn(*args)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        out = fn(*args)
    jax.block_until_ready(out)
    ms = ((time.perf_counter() - t0) / n_repeats) * 1000.0
    return float(ms), {}


def _profile_explicit_substeps(
    base_bun: MolecularBundle,
    grid_dims_pme: tuple[int, int, int],
    *,
    b_prof: int = 1,
    n_steady: int = 2,
    n_wall: int = 5,
    use_perfetto_probe: bool = False,
) -> dict[str, Any]:
    """Profile explicit-solvent MD sub-steps with `xtrax.profiling.probe_unit`."""
    bucket_atoms = int(base_bun.positions.shape[0])
    bucket_waters = int(base_bun.water_indices.shape[0])
    box_vec = jnp.diag(base_bun.box).astype(jnp.float32)

    disp_fn, _ = displacement_fn_for_bundle(base_bun)
    nl_fn = nl_mod.make_neighbor_list_fn(
        disp_fn,
        box_vec,
        float(base_bun.cutoff_distance),
        capacity_multiplier=1.25,
        dr_threshold=0.5,
        atom_mask=base_bun.atom_mask,
    )
    nbr0 = _encode_bundle_nl(nl_fn.allocate(base_bun.positions), base_bun)
    max_occ = int(nbr0.idx.shape[-1])
    mkt0 = getattr(nbr0, "max_k_tiles", None)
    if mkt0 is None:
        mkt0 = nl_mod._max_k_tiles_from_idx(nbr0.idx)
    tp_nl0 = build_tile_pair_neighbor_list(
        nbr0.idx,
        excl_indices=base_bun.excl_dense_indices,
        excl_scales_vdw=base_bun.excl_dense_scales_vdw,
        excl_scales_elec=base_bun.excl_dense_scales_elec,
        atom_mask=base_bun.atom_mask,
        positions=base_bun.positions,
        box=box_vec,
        cutoff=float(base_bun.cutoff_distance),
    )
    pw_blk = _infer_first_pure_water_block(
        bucket_atoms,
        (
            (base_bun.bond_idx, base_bun.bond_mask, 2),
            (base_bun.angle_idx, base_bun.angle_mask, 3),
            (base_bun.dihedral_idx, base_bun.dihedral_mask, 4),
            (base_bun.improper_idx, base_bun.improper_mask, 4),
            (base_bun.exception_pairs, base_bun.exception_mask, 2),
        ),
    )

    pos_b = jnp.broadcast_to(base_bun.positions[None, :, :], (b_prof, bucket_atoms, 3))
    q_b = jnp.broadcast_to(base_bun.charges[None, :], (b_prof, bucket_atoms))
    am_b = jnp.broadcast_to(base_bun.atom_mask[None, :], (b_prof, bucket_atoms))
    wi_b = jnp.broadcast_to(base_bun.water_indices[None, :, :], (b_prof, bucket_waters, 3))
    wm_b = jnp.broadcast_to(base_bun.water_mask[None, :], (b_prof, bucket_waters))
    masses_b = jnp.broadcast_to(base_bun.masses[None, :, None], (b_prof, bucket_atoms, 1))
    nbr_idx_b = jnp.broadcast_to(nbr0.idx[None, :, :], (b_prof, bucket_atoms, max_occ))
    mkt_b = jnp.broadcast_to(mkt0[None, :], (b_prof, *mkt0.shape))
    mom_b = jnp.zeros_like(pos_b)
    f_b = jnp.ones_like(pos_b) * jnp.float32(0.01)
    keys_b = jax.vmap(jax.random.PRNGKey)(jnp.arange(b_prof, dtype=jnp.uint32) + 42)
    dt_f32 = jnp.float32(2.0 / 48.88821)
    kT_f32 = jnp.float32(0.596)
    gamma_akma = 10.0 * 0.04888821

    jit_pre = jax.jit(
        lambda r, p, f, k: jax.vmap(
            lambda r1, p1, f1, m1, k1, wi1, wm1, am1: pallas_baoab_pre_force(
                r1, p1, f1, m1, k1, wi1, wm1, box_vec, dt_f32, kT_f32, gamma_akma, atom_mask=am1
            )
        )(r, p, f, masses_b, k, wi_b, wm_b, am_b)
    )
    jit_post = jax.jit(
        lambda r, p, f: jax.vmap(
            lambda r1, p1, f1, m1, wi1, wm1, am1: pallas_baoab_post_force(
                r1, p1, f1, m1, wi1, wm1, dt_f32, atom_mask=am1
            )
        )(r, p, f, masses_b, wi_b, wm_b, am_b)
    )
    _timer_fn = _probe_ms if use_perfetto_probe else _fast_substep_ms
    probe_metrics: dict[str, Any] = {}
    ms_pre, m_pre = _timer_fn(jit_pre, pos_b, mom_b, f_b, keys_b, n_steady=n_steady, n_wall=n_wall) if use_perfetto_probe else _fast_substep_ms(jit_pre, pos_b, mom_b, f_b, keys_b)
    ms_post, m_post = _timer_fn(jit_post, pos_b, mom_b, f_b, n_steady=n_steady, n_wall=n_wall) if use_perfetto_probe else _fast_substep_ms(jit_post, pos_b, mom_b, f_b)
    if use_perfetto_probe:
        probe_metrics["baoab_pre_force"] = m_pre
        probe_metrics["baoab_post_force"] = m_post

    def _mk48_force(r_in):
        def _one(r1, n1, mkt1):
            return pallas_bonded_and_nonbonded_nl_fused(
                r1,
                base_bun.charges,
                base_bun.sigmas,
                base_bun.epsilons,
                base_bun.atom_mask,
                n1,
                base_bun.excl_dense_indices,
                base_bun.excl_dense_scales_vdw,
                base_bun.excl_dense_scales_elec,
                bonds=base_bun.bond_idx,
                bond_params=base_bun.bond_params,
                bond_mask=base_bun.bond_mask,
                angles=base_bun.angle_idx,
                angle_params=base_bun.angle_params,
                angle_mask=base_bun.angle_mask,
                dihedrals=base_bun.dihedral_idx,
                dihedral_params=base_bun.dihedral_params,
                dihedral_mask=base_bun.dihedral_mask,
                impropers=base_bun.improper_idx,
                improper_params=base_bun.improper_params,
                improper_mask=base_bun.improper_mask,
                exception_pairs=base_bun.exception_pairs,
                exception_sigmas=base_bun.exception_sigmas,
                exception_epsilons=base_bun.exception_epsilons,
                exception_chargeprods=base_bun.exception_chargeprods,
                exception_mask=base_bun.exception_mask,
                box=box_vec,
                pme_alpha=float(base_bun.pme_alpha),
                cutoff=float(base_bun.cutoff_distance),
                excl_mode="pre_encoded",
                max_k_tiles=mkt1,
                first_pure_water_block=pw_blk,
                return_forces=True,
            )[1]

        return jax.vmap(_one)(r_in, nbr_idx_b, mkt_b)

    ms_mk48, m_mk48 = (_probe_ms(jax.jit(_mk48_force), pos_b, n_steady=n_steady, n_wall=n_wall) if use_perfetto_probe else _fast_substep_ms(jax.jit(_mk48_force), pos_b))
    if use_perfetto_probe:
        probe_metrics["bonded_and_nonbonded_direct_mk48"] = m_mk48

    jit_mk51 = jax.jit(
        lambda r_in: jax.vmap(
            lambda r1: pallas_nonbonded_tile_pairs(
                r1,
                base_bun.charges,
                base_bun.sigmas,
                base_bun.epsilons,
                tp_nl0,
                atom_mask=base_bun.atom_mask,
                box=box_vec,
                pme_alpha=float(base_bun.pme_alpha),
                cutoff=float(base_bun.cutoff_distance),
                return_forces=True,
            )[1]
        )(r_in)
    )
    ms_mk51, m_mk51 = (_probe_ms(jit_mk51, pos_b, n_steady=n_steady, n_wall=n_wall) if use_perfetto_probe else _fast_substep_ms(jit_mk51, pos_b))
    if use_perfetto_probe:
        probe_metrics["nonbonded_tile_pairs_mk51"] = m_mk51

    if float(base_bun.pme_alpha) > 0.0:
        G_f32 = influence_function(grid_dims_pme, box_vec, float(base_bun.pme_alpha), 4).astype(
            jnp.float32
        )
        n_act = int(base_bun.n_atoms)
        jit_spread = jax.jit(
            lambda r, q, m: spread_charges_pallas(
                r, q, m, box_vec, grid_dims_pme, order=4, n_active_atoms=n_act
            )
        )
        mesh_b = jit_spread(pos_b, q_b, am_b)
        jax.block_until_ready(mesh_b)
        ms_spread, m_spread = (_probe_ms(jit_spread, pos_b, q_b, am_b, n_steady=n_steady, n_wall=n_wall) if use_perfetto_probe else _fast_substep_ms(jit_spread, pos_b, q_b, am_b))
        if use_perfetto_probe:
            probe_metrics["pme_spread_3d_bspline"] = m_spread

        jit_rfft = jax.jit(lambda m: jnp.fft.rfftn(m, s=grid_dims_pme, axes=(-3, -2, -1)) * G_f32)
        qhat_b = jit_rfft(mesh_b)
        jax.block_until_ready(qhat_b)
        ms_rfft, m_rfft = (_probe_ms(jit_rfft, mesh_b, n_steady=n_steady, n_wall=n_wall) if use_perfetto_probe else _fast_substep_ms(jit_rfft, mesh_b))
        if use_perfetto_probe:
            probe_metrics["pme_rfft3d_and_influence"] = m_rfft

        # MK-62: Unscaled irfftn (scaling fused into gather_pallas!)
        jit_irfft = jax.jit(lambda qh: jnp.fft.irfftn(qh, s=grid_dims_pme, axes=(-3, -2, -1)))
        theta_b = jit_irfft(qhat_b)
        jax.block_until_ready(theta_b)
        ms_irfft, m_irfft = (_probe_ms(jit_irfft, qhat_b, n_steady=n_steady, n_wall=n_wall) if use_perfetto_probe else _fast_substep_ms(jit_irfft, qhat_b))
        if use_perfetto_probe:
            probe_metrics["pme_irfft3d"] = m_irfft

        f_scale = -332.0637 * float(grid_dims_pme[0] * grid_dims_pme[1] * grid_dims_pme[2])
        jit_gather = jax.jit(
            lambda r, q, m, th: gather_pallas(
                r,
                q,
                m,
                box_vec,
                th,
                grid_dims_pme,
                order=4,
                compute_dq=False,
                n_active_atoms=n_act,
                force_scale=f_scale,
            )[0]
        )
        ms_gather, m_gather = (
            _probe_ms(jit_gather, pos_b, q_b, am_b, theta_b, n_steady=n_steady, n_wall=n_wall)
            if use_perfetto_probe
            else _fast_substep_ms(jit_gather, pos_b, q_b, am_b, theta_b)
        )
        if use_perfetto_probe:
            probe_metrics["pme_gather_3d_bspline_fused_scale"] = m_gather
    else:
        ms_spread = ms_rfft = ms_irfft = ms_gather = 0.0

    bd = _substep_breakdown_dict(
        {
            "baoab_pre_force": ms_pre / b_prof,
            "bonded_and_nonbonded_direct_mk48": ms_mk48 / b_prof,
            "pme_spread_3d_bspline": ms_spread / b_prof,
            "pme_rfft3d_and_influence": ms_rfft / b_prof,
            "pme_irfft3d": ms_irfft / b_prof,
            "pme_gather_3d_bspline_fused_scale": ms_gather / b_prof,
            "baoab_post_force": ms_post / b_prof,
        }
    )
    bd["nonbonded_tile_pairs_mk51_ms_per_rep"] = float(ms_mk51 / b_prof)
    bd["num_valid_tile_pairs_aabb_pruned"] = int(tp_nl0.num_valid_pairs)
    if probe_metrics:
        bd["xtrax_probe_metrics"] = probe_metrics
    return bd


# ============================================================================
# SUITE A & B: Official OpenMM Suite + 7-Step Solvated Size & Holo Ladder
# ============================================================================
def run_suites_a_and_b_ladder(
    dhfr_bundle: MolecularBundle,
    *,
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    quick: bool = False,
    checkpoint_out: Path | None = None,
    checkpoint_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run Suite A (Official OpenMM benchmark.py systems) & Suite B (7-Step Solvated + Holo Ladder)."""
    print("=== [Suite A & B] Official OpenMM Suite + 7-Step Solvated & Holo Protein-Ligand Ladder ===")
    n_steps = 25 if quick else 100
    n_steady = 1 if quick else 2
    n_wall = 2 if quick else 5

    existing_gbsa: dict[str, Any] | None = None
    existing_explicit: dict[str, dict[str, Any]] = {}
    if checkpoint_out is not None and checkpoint_out.exists():
        try:
            prev_data = json.loads(checkpoint_out.read_text())
            sab = prev_data.get("suites_a_and_b", {})
            if sab.get("suite_a_gbsa", {}).get("batch_scaling"):
                existing_gbsa = sab["suite_a_gbsa"]
            for rec in sab.get("explicit_ladder", []):
                sys_n = rec.get("system")
                omm_ms = rec.get("openmm", {}).get("openmm_cuda_single_b1", {}).get("ms_per_step", 99.0)
                if sys_n and rec.get("batch_scaling") and omm_ms < 1.0:
                    existing_explicit[sys_n] = rec
            if existing_gbsa or existing_explicit:
                print(
                    f"  [Resume] Loaded checkpoint from {checkpoint_out}: "
                    f"gbsa={'yes' if existing_gbsa else 'no'}, "
                    f"explicit_systems={list(existing_explicit.keys())}"
                )
        except Exception as exc:
            print(f"  [Resume] Could not load checkpoint {checkpoint_out}: {exc}")

    # 1. Suite A `gbsa` (DHFR 2,489 atoms implicit OBC2 GBSA + 2D CMAPs)
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
        try:
            for B in batch_sizes:
                _clear_all_caches()
                plan = EnsemblePlan.from_bundles([gbsa_bun for _ in range(B)], use_pallas=True)
                w_gbsa = plan.run(
                    n_steps=5, dt=2.0, kT=0.596, seed=1, gamma=91.0, run_mode="inference", use_pallas=True
                )
                jax.block_until_ready(
                    w_gbsa[0].positions if isinstance(w_gbsa, (list, tuple)) else w_gbsa.positions
                )
                t0 = time.perf_counter()
                trajs = plan.run(
                    n_steps=n_steps, dt=2.0, kT=0.596, seed=2, gamma=91.0, run_mode="inference", use_pallas=True
                )
                jax.block_until_ready(
                    trajs[0].positions if isinstance(trajs, (list, tuple)) else trajs.positions
                )
                ms_batch = ((time.perf_counter() - t0) / n_steps) * 1000.0
                ms_rep = ms_batch / B
                omm_b1 = gbsa_omm_bench["openmm_cuda_single_b1"]["ms_per_step"]
                mps_entry = gbsa_omm_bench["openmm_cuda_mps_scaling"].get(f"B={B}", {})
                mps_rep = mps_entry.get("ms_per_step_per_replica")
                gbsa_rows.append(
                    {
                        "B": B,
                        "ms_per_step_batch": float(ms_batch),
                        "ms_per_step_per_replica": float(ms_rep),
                        "ns_per_day_per_replica": _ns_per_day(ms_rep, dt_fs=2.0),
                        "ns_per_day_aggregate": _ns_per_day(ms_batch, dt_fs=2.0) * B,
                        "speedup_vs_openmm_cuda_single_b1": float(omm_b1 / ms_rep),
                        "speedup_vs_openmm_cuda_mps_same_B": (
                            float(mps_rep / ms_rep) if mps_rep and mps_rep > 0 else None
                        ),
                    }
                )
                print(
                    f"  [Suite A] gbsa (N=2489) B={B:2d}: Prolix={ms_rep:.4f} ms/step/rep "
                    f"({_ns_per_day(ms_rep):.1f} ns/day/rep) vs OpenMM CUDA B=1={omm_b1:.4f} ms/step"
                )
        finally:
            os.environ.pop("PROLIX_ACTIVE_ATOMS", None)
        suite_a_gbsa_dict = {
            "system": "5DFR_gbsa_implicit",
            "n_real_atoms": int(gbsa_bun.n_atoms),
            "openmm": gbsa_omm_bench,
            "batch_scaling": gbsa_rows,
        }

    # 2. Explicit-solvent ladder configurations (covering both Suite A and Suite B)
    # Compute OpenMM-matched PME grid for DHFR (62.23 A box, cutoff 9.0 A, tol 5e-4 -> 54^3 or 60^3)
    _, dhfr_omm_grid = compute_openmm_pme_params((62.23, 62.23, 62.23), cutoff=9.0, ewald_error_tolerance=5e-4)
    dhfr_omm_grid_str = str(dhfr_omm_grid[0])

    ladder_specs = [
        # (name, suite_tags, kind, n_real, bucket_atoms, bucket_waters, elec, pme_grid_str)
        ("1UAO_solvated_2k", ["SuiteB"], "trunc", 1963, 4096, 1024, "pme", "32"),
        ("1VII_solvated_4k", ["SuiteB"], "trunc", 4199, 8192, 2048, "pme", "40"),
        ("1UBQ_solvated_7k", ["SuiteB"], "trunc", 6658, 8192, 2048, "pme", "48"),
        ("Holo_Lysozyme_15k", ["SuiteB"], "holo_15k", 15002, 16384, 4608, "pme", "54"),
        ("DHFR_24k_rf", ["SuiteA"], "trunc", 23558, 25000, 8192, "rf", "64"),
        ("DHFR_24k_pme_openmm_grid", ["SuiteA", "SuiteB"], "trunc", 23558, 25000, 8192, "pme", dhfr_omm_grid_str),
        ("DHFR_24k_pme_64grid", ["SuiteA"], "trunc", 23558, 25000, 8192, "pme", "64"),
        ("Holo_DHFR_MTX_NADPH_24k", ["SuiteB"], "holo_dhfr", 23621, 25000, 8192, "pme", dhfr_omm_grid_str),
        ("1AKE_solvated_45k", ["SuiteB"], "1ake_45k", 44999, 50000, 16384, "pme", "80"),
        ("ApoA1_92k_rf", ["SuiteA"], "apoa1_rf", 92224, 100000, 24576, "rf", "96x96x64"),
        ("ApoA1_92k_pme", ["SuiteA", "SuiteB"], "apoa1_pme", 92224, 100000, 24576, "pme", "96x96x64"),
    ]
    # Run full 11-system Suite A + Suite B ladder even in quick mode

    explicit_records: list[dict[str, Any]] = []
    prev_gp = os.environ.get("PROLIX_PME_GRID_POINTS")
    try:
        for name, suites, kind, n_real_target, b_atoms, b_waters, elec, grid_str in ladder_specs:
            if name in existing_explicit:
                explicit_records.append(existing_explicit[name])
                continue
            os.environ["PROLIX_PME_GRID_POINTS"] = grid_str
            _clear_all_caches()

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
                g_dims = (g_parts[0], g_parts[0], g_parts[0]) if len(g_parts) == 1 else (g_parts[0], g_parts[1], g_parts[2])
                omm_sys, omm_pos, omm_xml = _build_openmm_explicit_system(
                    bun, electrostatic_method=elec, pme_grid_dims=g_dims
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
            else:
                raise ValueError(f"Unknown ladder kind: {kind}")

            n_real = int(bun.n_atoms)
            pos_nm_np = np.asarray(bun.positions[:n_real], dtype=np.float64) * 0.1

            if b_atoms >= 100000:
                eff_batches = tuple(b for b in batch_sizes if b <= (4 if quick else 8))
                mps_batches = tuple(b for b in (1, 2, 4) if b in eff_batches)
            elif b_atoms >= 50000:
                eff_batches = tuple(b for b in batch_sizes if b <= (8 if quick else 16))
                mps_batches = tuple(b for b in (1, 4, 8) if b in eff_batches)
            else:
                eff_batches = batch_sizes
                mps_batches = tuple(b for b in (1, 4, 8, 16) if b in eff_batches)

            omm_bench = _time_openmm_cuda_single_and_mps(
                omm_sys,
                omm_pos,
                omm_xml,
                pos_nm_np,
                batch_sizes=eff_batches,
                dt_fs=2.0,
                gamma_ps=10.0,
                n_warmup=15 if (quick or b_atoms >= 50000) else 60,
                n_steps=40 if (quick or b_atoms >= 50000) else 200,
                run_mps_batches=mps_batches,
            )

            use_perfetto = (name == "DHFR_24k_pme_openmm_grid")
            substeps_b1 = _profile_explicit_substeps(
                bun, g_dims, b_prof=1, n_steady=n_steady, n_wall=n_wall, use_perfetto_probe=use_perfetto
            )
            substeps_b4 = (
                _profile_explicit_substeps(
                    bun, g_dims, b_prof=4, n_steady=n_steady, n_wall=n_wall, use_perfetto_probe=False
                )
                if 4 in eff_batches and b_atoms <= 25000
                else None
            )

            rows_b: list[dict[str, Any]] = []
            try:
                for B in eff_batches:
                    _clear_all_caches()
                    bundles_b = [bun for _ in range(B)]
                    plan = EnsemblePlan.from_bundles(bundles_b, use_pallas=True, use_pallas_pme=True)
                    trajs_w = plan.run(
                        n_steps=5,
                        dt=2.0,
                        kT=0.596,
                        seed=1,
                        gamma=10.0,
                        run_mode="inference",
                        use_neighbor_list=True,
                        nl_update_every=20,
                        use_pallas=True,
                        use_pallas_pme=True,
                    )
                    if isinstance(trajs_w, (list, tuple)):
                        for tw in trajs_w:
                            jax.block_until_ready(tw.positions)
                    else:
                        jax.block_until_ready(trajs_w.positions)
                    steps_run = max(10, n_steps // 2) if (b_atoms >= 50000 and B >= 8) else n_steps
                    t0 = time.perf_counter()
                    trajs = plan.run(
                        n_steps=steps_run,
                        dt=2.0,
                        kT=0.596,
                        seed=2,
                        gamma=10.0,
                        run_mode="inference",
                        use_neighbor_list=True,
                        nl_update_every=20,
                        use_pallas=True,
                        use_pallas_pme=True,
                    )
                    if isinstance(trajs, (list, tuple)):
                        for tp in trajs:
                            jax.block_until_ready(tp.positions)
                    else:
                        jax.block_until_ready(trajs.positions)
                    ms_batch = ((time.perf_counter() - t0) / steps_run) * 1000.0
                    ms_rep = ms_batch / B
                    omm_b1 = omm_bench["openmm_cuda_single_b1"]["ms_per_step"]
                    mps_entry = omm_bench["openmm_cuda_mps_scaling"].get(f"B={B}", {})
                    mps_rep = mps_entry.get("ms_per_step_per_replica")
                    rows_b.append(
                        {
                            "B": B,
                            "ms_per_step_batch": float(ms_batch),
                            "ms_per_step_per_replica": float(ms_rep),
                            "ns_per_day_per_replica": _ns_per_day(ms_rep, dt_fs=2.0),
                            "ns_per_day_aggregate": _ns_per_day(ms_batch, dt_fs=2.0) * B,
                            "speedup_vs_openmm_cuda_single_b1": float(omm_b1 / ms_rep),
                            "speedup_vs_openmm_cuda_mps_same_B": (
                                float(mps_rep / ms_rep) if mps_rep and mps_rep > 0 else None
                            ),
                        }
                    )
                    print(
                        f"  [{','.join(suites)}] {name:26s} (N={n_real:6d}, pad={b_atoms:6d}, grid={grid_str:8s}) B={B:2d}: "
                        f"Prolix={ms_rep:.4f} ms/step/rep ({_ns_per_day(ms_rep):.1f} ns/day/rep, "
                        f"{_ns_per_day(ms_batch) * B:.1f} ns/day agg) vs OpenMM CUDA B=1={omm_b1:.4f} ms/step"
                        + (f", MPS B={B}={mps_rep:.4f} ms/step/rep" if mps_rep else "")
                    )
            finally:
                os.environ.pop("PROLIX_ACTIVE_ATOMS", None)
                os.environ.pop("PROLIX_ACTIVE_WATERS", None)

            explicit_records.append(
                {
                    "system": name,
                    "suites": suites,
                    "kind": kind,
                    "n_real_atoms": n_real,
                    "bucket_atoms": b_atoms,
                    "bucket_waters": b_waters,
                    "occupancy_pct": float(100.0 * n_real / b_atoms),
                    "electrostatics": elec,
                    "pme_grid": g_dims if elec == "pme" else None,
                    "openmm": omm_bench,
                    "substep_breakdown": {"B=1": substeps_b1, "B=4": substeps_b4},
                    "batch_scaling": rows_b,
                }
            )
            if checkpoint_out is not None:
                checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
                partial_payload = dict(checkpoint_meta or {})
                partial_payload["suites_a_and_b"] = {
                    "suite_a_gbsa": suite_a_gbsa_dict,
                    "explicit_ladder": explicit_records,
                }
                checkpoint_out.write_text(json.dumps(partial_payload, indent=2))
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
# SUITE C: Congeneric Protein-Ligand Screen (MK-64) & Cross-Size Ragged Screen
# ============================================================================
def run_suite_c_congeneric_and_ragged(
    dhfr_bundle: MolecularBundle,
    *,
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    quick: bool = False,
) -> dict[str, Any]:
    """Run Suite C: Heterogeneous Congeneric Ligand Screen (`MK-64`) & Cross-Size Ragged Screen."""
    print("=== [Suite C] Heterogeneous Protein-Ligand Congeneric Screen (MK-64) & Ragged Screen ===")
    if quick:
        batch_sizes = (1, 2, 4, 8, 16)
    n_steps = 25 if quick else 100

    _, dhfr_omm_grid = compute_openmm_pme_params((62.23, 62.23, 62.23), cutoff=9.0, ewald_error_tolerance=5e-4)
    grid_str = str(dhfr_omm_grid[0])
    prev_gp = os.environ.get("PROLIX_PME_GRID_POINTS")
    os.environ["PROLIX_PME_GRID_POINTS"] = grid_str

    try:
        # --- Part C1: Congeneric Holo-DHFR Screen (B distinct ligand analogues in 25,000 bucket) ---
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
                # Vary ligand parameters (charges, sigmas, epsilons, bond_params, dihedral_params, exception_chargeprods)
                # while keeping topology index & mask arrays IDENTICAL (`s_bi=0, s_ai=0, s_di=0, s_hi=0, s_ep=0`)
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
                _clear_all_caches()
                bundles_b = congeneric_pool[:B]
                plan = EnsemblePlan.from_bundles(bundles_b, use_pallas=True, use_pallas_pme=True)
                trajs_cw = plan.run(
                    n_steps=5,
                    dt=2.0,
                    kT=0.596,
                    seed=1,
                    gamma=10.0,
                    run_mode="inference",
                    use_neighbor_list=True,
                    nl_update_every=20,
                    use_pallas=True,
                    use_pallas_pme=True,
                )
                if isinstance(trajs_cw, (list, tuple)):
                    for tw in trajs_cw:
                        jax.block_until_ready(tw.positions)
                else:
                    jax.block_until_ready(trajs_cw.positions)
                t0 = time.perf_counter()
                trajs = plan.run(
                    n_steps=n_steps,
                    dt=2.0,
                    kT=0.596,
                    seed=2,
                    gamma=10.0,
                    run_mode="inference",
                    use_neighbor_list=True,
                    nl_update_every=20,
                    use_pallas=True,
                    use_pallas_pme=True,
                )
                jax.block_until_ready(
                    trajs[0].positions if isinstance(trajs, (list, tuple)) else trajs.positions
                )
                ms_batch = ((time.perf_counter() - t0) / n_steps) * 1000.0
                ms_rep = ms_batch / B

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
                        allow_opencl_fallback=_ALLOW_OPENCL_FALLBACK,
                    )
                mps_rep = mps_res.get("ms_per_step_per_replica") if mps_res else None
                congeneric_rows.append(
                    {
                        "B": B,
                        "prolix_ms_per_step_batch": float(ms_batch),
                        "prolix_ms_per_step_per_replica": float(ms_rep),
                        "prolix_ns_per_day_per_replica": _ns_per_day(ms_rep, dt_fs=2.0),
                        "prolix_ns_per_day_aggregate": _ns_per_day(ms_batch, dt_fs=2.0) * B,
                        "openmm_cuda_mps": mps_res,
                        "speedup_vs_openmm_cuda_mps": (
                            float(mps_rep / ms_rep) if mps_rep and mps_rep > 0 else None
                        ),
                    }
                )
                print(
                    f"  [Suite C1 Congeneric Holo-DHFR] B={B:2d}: Prolix={ms_rep:.4f} ms/step/rep "
                    f"({_ns_per_day(ms_rep):.1f} ns/day/rep)"
                    + (
                        f" vs OpenMM CUDA+MPS={mps_rep:.4f} ms/step/rep (speedup={mps_rep / ms_rep:.2f}x)"
                        if mps_rep
                        else ""
                    )
                )
        finally:
            os.environ.pop("PROLIX_ACTIVE_ATOMS", None)
            os.environ.pop("PROLIX_ACTIVE_WATERS", None)

        # --- Part C2: Cross-Size Ragged Multi-Target Screen ---
        # Batch [1UAO (1,963), 1VII (4,199), 1UBQ (6,658), Holo-Lysozyme (15,002), Holo-DHFR (23,558)]
        # padded into a shared 25,000 bucket in a single EnsemblePlan vs OpenMM CUDA+MPS
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

        _clear_all_caches()
        B_rag = len(ragged_bundles)
        plan_rag = EnsemblePlan.from_bundles(ragged_bundles, use_pallas=True, use_pallas_pme=True)
        trajs_rw = plan_rag.run(
            n_steps=5,
            dt=2.0,
            kT=0.596,
            seed=1,
            gamma=10.0,
            run_mode="inference",
            use_neighbor_list=True,
            nl_update_every=20,
            use_pallas=True,
            use_pallas_pme=True,
        )
        if isinstance(trajs_rw, (list, tuple)):
            for tw in trajs_rw:
                jax.block_until_ready(tw.positions)
        else:
            jax.block_until_ready(trajs_rw.positions)
        t0 = time.perf_counter()
        trajs_rag = plan_rag.run(
            n_steps=n_steps,
            dt=2.0,
            kT=0.596,
            seed=2,
            gamma=10.0,
            run_mode="inference",
            use_neighbor_list=True,
            nl_update_every=20,
            use_pallas=True,
            use_pallas_pme=True,
        )
        jax.block_until_ready(
            trajs_rag[0].positions if isinstance(trajs_rag, (list, tuple)) else trajs_rag.positions
        )
        ms_rag_batch = ((time.perf_counter() - t0) / n_steps) * 1000.0
        ms_rag_rep = ms_rag_batch / B_rag

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
            allow_opencl_fallback=_ALLOW_OPENCL_FALLBACK,
        )
        mps_rag_rep = mps_rag.get("ms_per_step_per_replica") if mps_rag else None
        print(
            f"  [Suite C2 Cross-Size Ragged B={B_rag}] Prolix={ms_rag_rep:.4f} ms/step/rep "
            f"({_ns_per_day(ms_rag_rep):.1f} ns/day/rep)"
            + (
                f" vs OpenMM CUDA+MPS={mps_rag_rep:.4f} ms/step/rep (speedup={mps_rag_rep / ms_rag_rep:.2f}x)"
                if mps_rag_rep
                else ""
            )
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
            "prolix_ms_per_step_batch": float(ms_rag_batch),
            "prolix_ms_per_step_per_replica": float(ms_rag_rep),
            "prolix_ns_per_day_per_replica": _ns_per_day(ms_rag_rep, dt_fs=2.0),
            "prolix_ns_per_day_aggregate": _ns_per_day(ms_rag_batch, dt_fs=2.0) * B_rag,
            "openmm_cuda_mps": mps_rag,
            "speedup_vs_openmm_cuda_mps": (
                float(mps_rag_rep / ms_rag_rep) if mps_rag_rep and mps_rag_rep > 0 else None
            ),
        },
    }


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    parser = argparse.ArgumentParser(
        description="Pallas v3.4 True OpenMM CUDA + CUDA MPS System Ladder Benchmark"
    )
    parser.add_argument(
        "--gpu-tag",
        type=str,
        default="gpu",
        help="GPU architecture tag (e.g., l40s, h200, b300, a100).",
    )
    parser.add_argument(
        "--suites",
        type=str,
        default="A,B,C",
        help="Comma-separated suites to run (default: A,B,C).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "benchmarks" / "v34_ladder_results.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a faster subset of batch sizes and step counts.",
    )
    parser.add_argument(
        "--allow-opencl-fallback",
        action="store_true",
        help="Explicitly allow OpenMM OpenCL fallback if CUDA is unavailable (debugging only).",
    )
    args = parser.parse_args()

    global _ALLOW_OPENCL_FALLBACK
    _ALLOW_OPENCL_FALLBACK = bool(args.allow_opencl_fallback)

    dev = jax.devices()[0]
    dev_kind = getattr(dev, "device_kind", str(dev))
    omm_inv = get_openmm_platform_inventory()
    print(f"=== Pallas v3.4 Benchmark Ladder on {dev_kind} ===")
    print(f"OpenMM Platform Inventory: {json.dumps(omm_inv)}")
    if not omm_inv.get("has_cuda_platform") and not _ALLOW_OPENCL_FALLBACK:
        raise RuntimeError(
            f"OpenMM CUDA platform not registered! Inventory={json.dumps(omm_inv)}"
        )

    ff_path = B_dhfr._resolve_ff_path()
    sys_data = B_dhfr._build_dhfr_system_dict(ff_path)
    dhfr_bundle = B_dhfr._build_bundle_from_sys_data(sys_data)

    t_start = time.perf_counter()
    ckpt_meta = {
        "schema_version": "pallas_v34_cuda_mps_ladder_v1",
        "gpu_tag": args.gpu_tag,
        "suites_requested": args.suites,
        "device_kind": dev_kind,
        "jax_version": jax.__version__,
        "openmm_inventory": omm_inv,
        "quick_mode": bool(args.quick),
    }
    suites_ab = run_suites_a_and_b_ladder(
        dhfr_bundle, quick=args.quick, checkpoint_out=args.out, checkpoint_meta=ckpt_meta
    )
    suite_c = run_suite_c_congeneric_and_ragged(dhfr_bundle, quick=args.quick)
    total_s = time.perf_counter() - t_start

    payload = {
        "schema_version": "pallas_v34_cuda_mps_ladder_v1",
        "gpu_tag": args.gpu_tag,
        "suites_requested": args.suites,
        "device_kind": dev_kind,
        "jax_version": jax.__version__,
        "openmm_inventory": omm_inv,
        "quick_mode": bool(args.quick),
        "total_wall_s": float(total_s),
        "suites_a_and_b": suites_ab,
        "suite_c": suite_c,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print("=== V34_JSON_BEGIN ===")
    print(json.dumps(payload))
    print("=== V34_JSON_END ===")
    print(f"=== Saved Pallas v3.4 Ladder JSON to {args.out} ({total_s:.1f}s) ===")


if __name__ == "__main__":
    main()
