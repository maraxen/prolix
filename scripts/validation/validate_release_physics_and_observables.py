"""Release-Grade Four-Tier Multi-Engine Physics & Observable Validation Suite (Pallas v3.5).

Validates mathematical, numerical, thermodynamic, and structural/dynamical parity across:
- Prolix Pallas v3.5 (`single` & `mixed`)
- `prolix.openmm` (`ProlixContext`)
- OpenMM `CUDA` (`double` & `mixed`, strictly enforced via `require_openmm_cuda_context`)
- `kUPS` (`prolix.physics.kups_adapter`, eV/A/amu/fs cross-unit parity)

Four Validation Tiers:
- Tier 1: Single-point term-by-term energy & force parity across `1UAO`, `1VII`, `1UBQ`,
  `5DFR_GBSA`, `Holo_Lysozyme`, `DHFR_RF`, `DHFR_PME`, `Holo_DHFR`, and `ApoA1_92k`.
- Tier 2: Analytical `custom_vjp` vs `float64` central finite-difference gradient parity
  (`< 1e-5` relative error on `float64` VJP) wrt `r, q, sigma, epsilon`, GBSA `radii` &
  `scaled_radii`, and bonded parameters across `PME`, `RF`, and `GBSA`.
- Tier 3: NVE energy conservation (`10,000` steps), NVT kinetic equipartition (`<T>`,
  solute-vs-solvent `T` split, Maxwell-Boltzmann variance), and NPT equilibrium density
  `<rho>` + isothermal compressibility `kappa_T` on `TIP3P`, `1UBQ`, and `DHFR_PME`.
- Tier 4: Multi-replica (`B = 8`, `100 ps/rep`) structural & dynamical observables with an
  independent-seed **OpenMM-CUDA seed-A vs OpenMM-CUDA seed-B (`B=8` vs `B=8`)** control:
  * `TIP3P` water: `g_OO(r)` structure + unwrapped Einstein self-diffusion `D_TIP3P` at
    matched `(L = 30.0 A, gamma = 1.0 ps^-1)` plus NVE (`gamma = 0`) & Yeh-Hummer correction.
  * Solvated proteins & holo complex (`1UAO`, `1UBQ`, `DHFR_PME`, `Holo_DHFR`): `C_alpha`
    RMSD, per-residue `C_alpha` RMSF correlation, `R_g`, Ramachandran basin occupancy,
    backbone H-bond persistence, and `Holo_DHFR` active-site ligand RMSD & contact distances.
"""
# ruff: noqa: E402, S607

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import math
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
XTRAX_SRC = ROOT.parent / "xtrax" / "src"
_extra_paths: list[str] = []
for _p in (str(SRC), str(ROOT), str(XTRAX_SRC)):
    if Path(_p).exists():
        _extra_paths.append(_p)
        if _p not in sys.path:
            sys.path.insert(0, _p)
_existing_pp = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = ":".join(_extra_paths + ([_existing_pp] if _existing_pp else []))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from prolix.api.bundle_md import (
    displacement_fn_for_bundle,
    energy_fn_from_bundle,
    masses_with_prefix,
    pbc_box_for_bundle,
    positions_with_prefix,
)
from prolix.api.ensemble_plan import _encode_bundle_nl

try:
    from prolix.constants import COULOMB_CONSTANT
except ImportError:
    from proxide.physics.constants import COULOMB_CONSTANT
from prolix.pallas_kernels import (
    build_tile_pair_neighbor_list,
    pallas_angles_harmonic,
    pallas_bonded_all_fused,
    pallas_bonds_2body,
    pallas_cmap_torsions,
    pallas_exceptions_14,
    pallas_nonbonded_nl,
    pallas_nonbonded_tile_pairs,
    pallas_torsions_4body,
)
from prolix.physics import (
    bonded,
    explicit_corrections,
    generalized_born,
    kups_adapter,
)
from prolix.physics import cmap as cmap_mod
from prolix.physics import neighbor_list as nl_mod
from prolix.physics import pme as pme_mod
from prolix.physics.bonded_masked import (
    _angle_energy_masked,
    _bond_energy_masked,
    _dihedral_energy_masked,
)
from prolix.physics.optimization import chunked_coulomb_energy_nl, chunked_lj_energy_nl
from prolix.physics.pallas_gbsa_tile import build_gbsa_tile_pair_list, pallas_gbsa_tile
from prolix.physics.settle import settle_langevin
from prolix.simulate import AKMA_TIME_UNIT_FS, BOLTZMANN_KCAL
from scripts.benchmarks.openmm_cuda_guard import (
    get_openmm_platform_inventory,
    require_openmm_cuda_context,
)
from scripts.validation._release_observables_helpers import (
    ValidatedSystemSpec,
    build_1uao_system,
    build_1ubq_system,
    build_1vii_system,
    build_5dfr_gbsa_system,
    build_apoa1_system,
    build_dhfr_system,
    build_holo_dhfr_system,
    build_holo_lysozyme_system,
    build_tip3p_system,
    clone_openmm_system,
    compare_protein_ensembles,
    compute_protein_replica_observables,
    compute_unwrapped_water_diffusion,
    compute_water_rdf_goo,
    strip_private_arrays,
)

# Exact CODATA conversion constants
KCAL_MOL_TO_BAR_A3: float = 4184.0 / (6.02214076e23 * 1.0e-25)  # 69476.9545727651 bar*A^3 / (kcal/mol)
AMU_PER_A3_TO_G_CM3: float = 1.0 / 0.602214076  # 1.66053906660 g/cm^3 per (amu/A^3)


def _git_head_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _collect_provenance(*, allow_cpu_smoke: bool, mode: str) -> dict[str, Any]:
    """Record full hardware, JAX/XLA, OpenMM CUDA plugin, and git provenance."""
    import openmm as mm

    inv = get_openmm_platform_inventory()
    live_cuda_prov: dict[str, Any] = {}
    probe_sys = mm.System()
    probe_sys.addParticle(12.0)
    probe_integ = mm.VerletIntegrator(0.001)
    ctx, live_cuda_prov = require_openmm_cuda_context(
        probe_sys,
        probe_integ,
        expected_precision="mixed",
        allow_cpu_fallback=bool(allow_cpu_smoke),
    )
    live_cuda_prov = {
        **live_cuda_prov,
        "caller": "validate_release_physics_and_observables._collect_provenance",
    }
    del ctx
    del probe_integ

    devs = jax.devices()
    return {
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "git_head": _git_head_sha(),
        "mode": mode,
        "allow_cpu_smoke": bool(allow_cpu_smoke),
        "python_version": sys.version,
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "jax_default_backend": jax.default_backend(),
        "jax_devices": [
            {"id": d.id, "platform": d.platform, "device_kind": getattr(d, "device_kind", str(d))}
            for d in devs
        ],
        "coulomb_constant_prolix_kcal_A_mol_e2": float(COULOMB_CONSTANT),
        "openmm_inventory": inv,
        "openmm_live_cuda_verification": live_cuda_prov,
    }


def _save_checkpoint(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=False))
    tmp_path.replace(output_path)


def _open_openmm_context(
    omm_system: Any,
    integrator: Any,
    *,
    precision: str,
    allow_cpu_smoke: bool,
    caller: str,
) -> tuple[Any, dict[str, Any]]:
    """Create an OpenMM Context strictly on `CUDA` via `require_openmm_cuda_context`."""
    ctx, prov = require_openmm_cuda_context(
        omm_system,
        integrator,
        expected_precision=precision,
        allow_cpu_fallback=bool(allow_cpu_smoke),
    )
    return ctx, {**prov, "caller": caller}


def _force_parity_metrics(f_test: np.ndarray, f_ref: np.ndarray) -> dict[str, float]:
    """Compute per-component RMSE, max |Delta F|, and vector cosine similarity on `(N_real, 3)`."""
    diff = np.asarray(f_test, dtype=np.float64) - np.asarray(f_ref, dtype=np.float64)
    rmse = float(np.sqrt(np.mean(diff * diff)))
    max_abs = float(np.max(np.abs(diff))) if diff.size > 0 else 0.0
    a_flat = np.asarray(f_test, dtype=np.float64).reshape(-1)
    b_flat = np.asarray(f_ref, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a_flat) * np.linalg.norm(b_flat))
    cos_sim = float(np.dot(a_flat, b_flat) / denom) if denom > 1e-15 else 1.0
    return {
        "force_rmse_kcal_mol_A": rmse,
        "force_max_abs_kcal_mol_A": max_abs,
        "force_cosine_similarity": cos_sim,
    }


# =============================================================================
# TIER 1: Single-Point Term-by-Term Energy & Force Parity
# =============================================================================


def _assign_openmm_force_groups(omm_system: Any, *, split_reciprocal: bool = True) -> Any:
    """Clone `omm_system` and assign deterministic force groups:
    0: HarmonicBondForce, 1: HarmonicAngleForce, 2: PeriodicTorsionForce,
    3: CMAPTorsionForce, 5: NonbondedForce (direct), 6: NonbondedForce (reciprocal),
    7: CustomGBForce / GBSAOBCForce.
    """
    import openmm as mm

    sys_copy = clone_openmm_system(omm_system)
    for idx in range(sys_copy.getNumForces()):
        f = sys_copy.getForce(idx)
        cls_name = type(f).__name__
        if isinstance(f, mm.HarmonicBondForce):
            f.setForceGroup(0)
        elif isinstance(f, mm.HarmonicAngleForce):
            f.setForceGroup(1)
        elif isinstance(f, mm.PeriodicTorsionForce):
            f.setForceGroup(2)
        elif isinstance(f, mm.CMAPTorsionForce):
            f.setForceGroup(3)
        elif isinstance(f, mm.NonbondedForce):
            f.setForceGroup(5)
            if split_reciprocal and f.getNonbondedMethod() in (mm.NonbondedForce.PME, mm.NonbondedForce.Ewald):
                f.setReciprocalSpaceForceGroup(6)
        elif cls_name in ("CustomGBForce", "GBSAOBCForce"):
            f.setForceGroup(7)
        else:
            f.setForceGroup(8)
    return sys_copy


def _eval_openmm_decomposed(
    spec: ValidatedSystemSpec,
    *,
    precision: str,
    allow_cpu_smoke: bool,
) -> tuple[dict[str, float], np.ndarray, dict[str, Any]]:
    """Evaluate exact force-group energies and total forces in OpenMM CUDA (`double` or `mixed`)."""
    import openmm as mm
    from openmm import unit as _omm_unit
    unit = cast("Any", _omm_unit)

    sys_no_tail = _assign_openmm_force_groups(spec.omm_system, split_reciprocal=True)
    for f in sys_no_tail.getForces():
        if isinstance(f, mm.NonbondedForce):
            f.setUseDispersionCorrection(False)

    integ1 = mm.VerletIntegrator(0.001)
    ctx1, prov = _open_openmm_context(
        sys_no_tail,
        integ1,
        precision=precision,
        allow_cpu_smoke=allow_cpu_smoke,
        caller=f"tier1_openmm_{precision}_{spec.name}",
    )
    pos_nm = spec.positions_A * 0.1
    ctx1.setPositions(pos_nm * unit.nanometer)

    def _ge(grp: int) -> float:
        st = ctx1.getState(getEnergy=True, groups={grp})
        return float(st.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))

    e_bonds = _ge(0)
    e_angles = _ge(1)
    e_torsions = _ge(2)
    e_cmap = _ge(3)
    e_direct = _ge(5)
    e_recip = _ge(6) if spec.electrostatics == "pme" else 0.0
    e_gbsa = _ge(7) if spec.electrostatics == "gbsa" else 0.0
    st_no_tail = ctx1.getState(getEnergy=True, getForces=True)
    e_no_tail = float(st_no_tail.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))
    del ctx1, integ1

    # Full system with dispersion tail (if periodic explicit)
    sys_full = _assign_openmm_force_groups(spec.omm_system, split_reciprocal=True)
    integ2 = mm.VerletIntegrator(0.001)
    ctx2, _ = _open_openmm_context(
        sys_full,
        integ2,
        precision=precision,
        allow_cpu_smoke=allow_cpu_smoke,
        caller=f"tier1_openmm_full_{precision}_{spec.name}",
    )
    ctx2.setPositions(pos_nm * unit.nanometer)
    st_full = ctx2.getState(getEnergy=True, getForces=True)
    e_total = float(st_full.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))
    forces_A = np.asarray(
        st_full.getForces(asNumpy=True).value_in_unit(unit.kilocalories_per_mole / unit.angstrom),
        dtype=np.float64,
    )
    del ctx2, integ2

    e_tail = e_total - e_no_tail
    terms = {
        "bonds": e_bonds,
        "angles": e_angles,
        "torsions": e_torsions,
        "cmap": e_cmap,
        "direct_nonbonded": e_direct,
        "pme_reciprocal": e_recip,
        "gbsa": e_gbsa,
        "tail_correction": e_tail,
        "total": e_total,
    }
    return terms, forces_A, prov


def _rf_direct_coulomb_nl_energy(
    pos: jnp.ndarray,
    charges: jnp.ndarray,
    atom_mask: jnp.ndarray,
    nbr_idx: jnp.ndarray,
    excl_dense_indices: jnp.ndarray | None,
    box_vec: jnp.ndarray,
    cutoff: float,
    rf_dielectric: float = 78.3,
) -> jnp.ndarray:
    """Exact $O(N \\cdot K)$ Reaction-Field direct Coulomb energy on raw neighbor list with per-slot exclusion masking."""
    if excl_dense_indices is None:
        msg = "_rf_direct_coulomb_nl_energy requires non-None excl_dense_indices"
        raise RuntimeError(msg)
    n = int(pos.shape[0])
    k_cap = int(nbr_idx.shape[1])
    m_ex = int(excl_dense_indices.shape[1]) if excl_dense_indices.ndim == 2 else 0
    tile_rows = 512
    n_tiles = max(1, (n + tile_rows - 1) // tile_rows)
    pad_n = n_tiles * tile_rows - n

    i_all = jnp.arange(n_tiles * tile_rows, dtype=jnp.int32).reshape(n_tiles, tile_rows)
    if pad_n > 0:
        pos_t = jnp.pad(pos, ((0, pad_n), (0, 0)), constant_values=0.0).reshape(n_tiles, tile_rows, 3)
        q_t = jnp.pad(charges, (0, pad_n), constant_values=0.0).reshape(n_tiles, tile_rows)
        m_t = jnp.pad(atom_mask, (0, pad_n), constant_values=False).reshape(n_tiles, tile_rows)
        nb_t = jnp.pad(nbr_idx, ((0, pad_n), (0, 0)), constant_values=n).reshape(n_tiles, tile_rows, k_cap)
        ex_t = (
            jnp.pad(excl_dense_indices, ((0, pad_n), (0, 0)), constant_values=-1).reshape(n_tiles, tile_rows, m_ex)
            if m_ex > 0
            else jnp.full((n_tiles, tile_rows, 1), -1, dtype=jnp.int32)
        )
    else:
        pos_t = pos.reshape(n_tiles, tile_rows, 3)
        q_t = charges.reshape(n_tiles, tile_rows)
        m_t = atom_mask.reshape(n_tiles, tile_rows)
        nb_t = nbr_idx.reshape(n_tiles, tile_rows, k_cap)
        ex_t = (
            excl_dense_indices.reshape(n_tiles, tile_rows, m_ex)
            if m_ex > 0
            else jnp.full((n_tiles, tile_rows, 1), -1, dtype=jnp.int32)
        )

    k_rf = (1.0 / (cutoff**3)) * ((rf_dielectric - 1.0) / (2.0 * rf_dielectric + 1.0))
    c_rf = (1.0 / cutoff) * ((3.0 * rf_dielectric) / (2.0 * rf_dielectric + 1.0))
    coul_c = jnp.asarray(COULOMB_CONSTANT, dtype=pos.dtype)
    eps_r2 = jnp.asarray(1e-24 if pos.dtype == jnp.float64 else 1e-12, dtype=pos.dtype)
    one_r = jnp.asarray(1.0, dtype=pos.dtype)

    def _tile_scan(acc: jnp.ndarray, elem: tuple[jnp.ndarray, ...]) -> tuple[jnp.ndarray, None]:
        i_blk, p_blk, q_blk, m_blk, nb_blk, ex_blk = elem
        nb_clean = jnp.where(nb_blk < 0, n, nb_blk & ((1 << 29) - 1))
        j_safe = jnp.where((nb_clean >= 0) & (nb_clean < n), nb_clean, 0)
        is_excl_bit = (nb_blk >= 0) & ((nb_blk & ((1 << 30) | (1 << 29))) != 0)
        is_excl_tbl = jnp.any((nb_clean[:, :, None] == ex_blk[:, None, :]) & (ex_blk[:, None, :] >= 0), axis=-1)
        valid_pair = (
            m_blk[:, None]
            & (i_blk[:, None] < n)
            & atom_mask[j_safe]
            & (nb_clean < n)
            & (i_blk[:, None] < nb_clean)
            & (~is_excl_bit)
            & (~is_excl_tbl)
        )
        dr = p_blk[:, None, :] - pos[j_safe]
        dr = dr - box_vec[None, None, :] * jnp.round(dr / box_vec[None, None, :])
        r = jnp.sqrt(jnp.sum(dr * dr, axis=-1) + eps_r2)
        in_cut = valid_pair & (r < cutoff)
        r_s = jnp.where(in_cut, r, one_r)
        q_ij = q_blk[:, None] * charges[j_safe]
        u_pair = coul_c * q_ij * (1.0 / r_s + k_rf * (r_s * r_s) - c_rf)
        return acc + jnp.sum(jnp.where(in_cut, u_pair, 0.0)), None

    e_tot, _ = jax.lax.scan(
        _tile_scan,
        jnp.asarray(0.0, dtype=pos.dtype),
        (i_all, pos_t, q_t, m_t, nb_t, ex_t),
    )
    return e_tot


def _pme_excl_erf_energy(
    pos: jnp.ndarray,
    charges: jnp.ndarray,
    excl_pair_idx: jnp.ndarray,
    excl_pair_mask: jnp.ndarray,
    box_vec: jnp.ndarray,
    pme_alpha: float,
    *,
    min_dist: float = 0.0,
) -> jnp.ndarray:
    """Evaluate $+\\sum_{(p_1, p_2) \\in \\text{excl}} C q_{p_1} q_{p_2} \\operatorname{erf}(\\alpha r)/r$ over masked exclusions."""
    n = int(pos.shape[0])
    p1 = excl_pair_idx[:, 0]
    p2 = excl_pair_idx[:, 1]
    valid_ex = excl_pair_mask & (p1 >= 0) & (p2 >= 0) & (p1 < n) & (p2 < n)
    p1_s = jnp.where(valid_ex, p1, 0)
    p2_s = jnp.where(valid_ex, p2, 0)
    dr = pos[p1_s] - pos[p2_s]
    dr = dr - box_vec[None, :] * jnp.round(dr / box_vec[None, :])
    eps_r2 = jnp.asarray(1e-24 if pos.dtype == jnp.float64 else 1e-12, dtype=pos.dtype)
    r = jnp.sqrt(jnp.sum(dr * dr, axis=-1) + eps_r2)
    active = valid_ex & (r >= min_dist)
    r_s = jnp.where(active, r, jnp.asarray(1.0, dtype=pos.dtype))
    u_erf = (
        jnp.asarray(COULOMB_CONSTANT, dtype=pos.dtype)
        * charges[p1_s]
        * charges[p2_s]
        * (jax.lax.erf(jnp.asarray(pme_alpha, dtype=pos.dtype) * r_s) / r_s)
    )
    return jnp.sum(jnp.where(active, u_erf, 0.0))


def _eval_prolix_pallas_decomposed(
    spec: ValidatedSystemSpec,
    *,
    mixed: bool,
) -> tuple[dict[str, float], np.ndarray]:
    """Evaluate Prolix Pallas v3.5 term-by-term energies & total forces (`single` or `mixed`)."""
    bun32 = spec.bundle_f32
    bun_acc = spec.bundle_f64 if mixed else spec.bundle_f32
    n_real = spec.n_real_atoms
    is_pbc = bool(bun32.shape_spec.has_pbc and bun32.shape_spec.boundary_condition == "periodic")
    box32 = jnp.diag(bun32.box).astype(jnp.float32) if is_pbc else None
    box_acc = jnp.diag(bun_acc.box) if is_pbc else None
    disp_acc, _ = displacement_fn_for_bundle(bun_acc)

    rf_dielectric = 78.3
    if spec.electrostatics == "rf" and spec.omm_system is not None:
        import openmm as mm

        for _f in spec.omm_system.getForces():
            if isinstance(_f, mm.NonbondedForce):
                rf_dielectric = float(_f.getReactionFieldDielectric())
                break

    # Allocate exact neighbor list (+ Morton 32x32 tile pairs for Pallas v3.5 single) for periodic systems
    nbr_raw: Any = None
    tp_nl = None
    if is_pbc:
        if box32 is None or box_acc is None:
            msg = "Periodic system requires non-None box vectors"
            raise RuntimeError(msg)
        disp32, _ = displacement_fn_for_bundle(bun32)
        nl_raw_fn = cast(
            "Any",
            nl_mod.make_neighbor_list_fn(
                disp32,
                box32,
                float(spec.cutoff_A),
                capacity_multiplier=1.60,
                dr_threshold=0.0,
                atom_mask=bun32.atom_mask,
                use_pallas_nl=False,
            ),
        )
        nbr_raw = nl_raw_fn.allocate(bun32.positions)
        if int(np.asarray(nbr_raw.error.code)) != 0:
            nl_raw_fn = cast(
                "Any",
                nl_mod.make_neighbor_list_fn(
                    disp32,
                    box32,
                    float(spec.cutoff_A),
                    capacity_multiplier=2.20,
                    dr_threshold=0.0,
                    atom_mask=bun32.atom_mask,
                    use_pallas_nl=False,
                ),
            )
            nbr_raw = nl_raw_fn.allocate(bun32.positions)
        if not mixed:
            tp_nl = build_tile_pair_neighbor_list(
                nbr_raw.idx,
                excl_indices=bun32.excl_dense_indices,
                excl_scales_vdw=bun32.excl_dense_scales_vdw,
                excl_scales_elec=bun32.excl_dense_scales_elec,
                atom_mask=bun32.atom_mask,
                positions=bun32.positions,
                box=box32,
                cutoff=float(spec.cutoff_A),
                morton_sort=True,
            )

    # 1. Bonded terms (Pallas in `single`, exact float64 XLA accumulation in `mixed`)
    b_idx = jnp.where(bun_acc.bond_mask[:, None], bun_acc.bond_idx, 0)
    b_prm = jnp.where(bun_acc.bond_mask[:, None], bun_acc.bond_params, 0.0)
    ub_idx = (
        bun_acc.urey_bradley_idx[:, ::2]
        if bun_acc.urey_bradley_idx.shape[-1] == 3
        else bun_acc.urey_bradley_idx[:, :2]
    )
    ub_idx_s = jnp.where(bun_acc.urey_bradley_mask[:, None], ub_idx, 0)
    ub_prm_s = jnp.where(bun_acc.urey_bradley_mask[:, None], bun_acc.urey_bradley_params, 0.0)

    a_idx = jnp.where(bun_acc.angle_mask[:, None], bun_acc.angle_idx, 0)
    a_prm = jnp.where(bun_acc.angle_mask[:, None], bun_acc.angle_params, 0.0)

    d_idx = jnp.where(bun_acc.dihedral_mask[:, None], bun_acc.dihedral_idx, 0)
    d_prm = jnp.where(bun_acc.dihedral_mask[:, None], bun_acc.dihedral_params[..., :3], 0.0)
    i_idx = jnp.where(bun_acc.improper_mask[:, None], bun_acc.improper_idx, 0)
    i_prm = jnp.where(bun_acc.improper_mask[:, None], bun_acc.improper_params[..., :3], 0.0)

    has_cmap = bool(jnp.any(bun_acc.cmap_mask)) and bun_acc.cmap_energy_grids.shape[0] > 0
    cmap_src32 = (
        bun32.cmap_coeffs
        if (bun32.cmap_coeffs is not None and bun32.cmap_coeffs.size > 0)
        else bun32.cmap_energy_grids
    )
    cmap_src_acc = (
        bun_acc.cmap_coeffs
        if (bun_acc.cmap_coeffs is not None and bun_acc.cmap_coeffs.size > 0)
        else bun_acc.cmap_energy_grids
    )

    exc_xla_fn = bonded.make_exception_pair_energy_fn(
        disp_acc,
        bun_acc.exception_pairs,
        bun_acc.exception_sigmas,
        bun_acc.exception_epsilons,
        bun_acc.exception_chargeprods,
        mask=bun_acc.exception_mask,
    )

    def _bonds_e(r_in: jnp.ndarray) -> jnp.ndarray:
        if not mixed and pallas_bonds_2body is not None:
            e = cast(
                "jnp.ndarray",
                pallas_bonds_2body(
                    r_in, b_idx, b_prm.astype(jnp.float32), bun32.bond_mask, box=box32, dist_eps=1e-10
                ),
            )
            if ub_idx_s.shape[0] > 0:
                e = e + cast(
                    "jnp.ndarray",
                    pallas_bonds_2body(
                        r_in,
                        ub_idx_s,
                        ub_prm_s.astype(jnp.float32),
                        bun32.urey_bradley_mask,
                        box=box32,
                        dist_eps=1e-10,
                    ),
                )
            return e
        e = _bond_energy_masked(r_in, b_idx, b_prm, bun_acc.bond_mask, disp_acc)
        if ub_idx_s.shape[0] > 0:
            e = e + _bond_energy_masked(
                r_in, ub_idx_s, ub_prm_s, bun_acc.urey_bradley_mask, disp_acc
            )
        return e

    def _angles_e(r_in: jnp.ndarray) -> jnp.ndarray:
        if not mixed and pallas_angles_harmonic is not None:
            return pallas_angles_harmonic(
                r_in, a_idx, a_prm.astype(jnp.float32), bun32.angle_mask, box=box32, dist_eps=1e-10
            )
        return _angle_energy_masked(r_in, a_idx, a_prm, bun_acc.angle_mask, disp_acc)

    def _torsions_e(r_in: jnp.ndarray) -> jnp.ndarray:
        if not mixed and pallas_torsions_4body is not None:
            e = (
                pallas_torsions_4body(
                    r_in,
                    d_idx,
                    d_prm.astype(jnp.float32),
                    bun32.dihedral_mask,
                    box=box32,
                    b0_is_ji=True,
                    is_harmonic=False,
                )
                if d_idx.shape[0] > 0
                else jnp.asarray(0.0, dtype=r_in.dtype)
            )
            if i_idx.shape[0] > 0:
                e = e + pallas_torsions_4body(
                    r_in,
                    i_idx,
                    i_prm.astype(jnp.float32),
                    bun32.improper_mask,
                    box=box32,
                    b0_is_ji=True,
                    is_harmonic=False,
                )
            return e
        e = (
            _dihedral_energy_masked(r_in, d_idx, d_prm, bun_acc.dihedral_mask, disp_acc)
            if d_idx.shape[0] > 0
            else jnp.asarray(0.0, dtype=r_in.dtype)
        )
        if i_idx.shape[0] > 0:
            e = e + _dihedral_energy_masked(r_in, i_idx, i_prm, bun_acc.improper_mask, disp_acc)
        return e

    def _cmap_e(r_in: jnp.ndarray) -> jnp.ndarray:
        if not has_cmap:
            return jnp.asarray(0.0, dtype=r_in.dtype)
        if not mixed and pallas_cmap_torsions is not None:
            return pallas_cmap_torsions(
                r_in,
                bun32.cmap_torsion_idx,
                bun32.cmap_indices,
                bun32.cmap_mask,
                cmap_src32,
                box=box32,
                first_is_psi=False,
            )
        phi_idx = bun_acc.cmap_torsion_idx[:, 0:4]
        psi_idx = (
            bun_acc.cmap_torsion_idx[:, 4:8]
            if bun_acc.cmap_torsion_idx.shape[-1] == 8
            else bun_acc.cmap_torsion_idx[:, 1:5]
        )
        phi = bonded.compute_dihedral_angles(r_in, phi_idx, disp_acc)
        psi = bonded.compute_dihedral_angles(r_in, psi_idx, disp_acc)
        e_per = cmap_mod.compute_cmap_energies(phi, psi, bun_acc.cmap_indices, cmap_src_acc)
        return jnp.sum(jnp.where(bun_acc.cmap_mask, e_per, 0.0))

    def _direct_nb_e(r_in: jnp.ndarray) -> jnp.ndarray:
        r32 = r_in.astype(jnp.float32)
        if spec.electrostatics == "pme":
            if box32 is None or box_acc is None or nbr_raw is None:
                msg = "PME evaluation requires periodic box and neighbor list"
                raise RuntimeError(msg)
            if not mixed:
                if tp_nl is None:
                    msg = "PME single-precision evaluation requires tile_pair_nl"
                    raise RuntimeError(msg)
                e_unexcl = cast(
                    "jnp.ndarray",
                    pallas_nonbonded_tile_pairs(
                        r32,
                        bun32.charges,
                        bun32.sigmas,
                        bun32.epsilons,
                        tp_nl,
                        atom_mask=bun32.atom_mask,
                        box=box32,
                        pme_alpha=float(spec.pme_alpha),
                        cutoff=float(spec.cutoff_A),
                        lj_switch_width=0.0,
                        include_pme_excl_erf=False,
                    ),
                ).astype(r_in.dtype)
                e_erf_ex = _pme_excl_erf_energy(
                    r32,
                    bun32.charges,
                    bun32.excl_indices,
                    bun32.excl_mask,
                    box32,
                    float(spec.pme_alpha),
                    min_dist=0.0,
                ).astype(r_in.dtype)
                e_exc = pallas_exceptions_14(
                    r32,
                    bun32.exception_pairs,
                    bun32.exception_sigmas,
                    bun32.exception_epsilons,
                    bun32.exception_chargeprods,
                    bun32.exception_mask,
                    box=box32,
                ).astype(r_in.dtype)
                return e_unexcl - e_erf_ex + e_exc
            e_lj = chunked_lj_energy_nl(
                r_in,
                bun_acc.sigmas,
                bun_acc.epsilons,
                bun_acc.excl_dense_indices,
                bun_acc.excl_dense_scales_vdw,
                nbr_raw.idx,
                disp_acc,
                cutoff=float(spec.cutoff_A),
                tile_size=128,
                switch_width=0.0,
            )
            e_coul = chunked_coulomb_energy_nl(
                r_in,
                bun_acc.charges,
                bun_acc.excl_dense_indices,
                bun_acc.excl_dense_scales_elec,
                nbr_raw.idx,
                disp_acc,
                pme_alpha=float(spec.pme_alpha),
                coulomb_constant=float(COULOMB_CONSTANT),
                cutoff=float(spec.cutoff_A),
                tile_size=128,
            )
            e_erf_far = _pme_excl_erf_energy(
                r_in,
                bun_acc.charges,
                bun_acc.excl_indices,
                bun_acc.excl_mask,
                box_acc,
                float(spec.pme_alpha),
                min_dist=float(spec.cutoff_A),
            )
            return e_lj + e_coul - e_erf_far + exc_xla_fn(r_in)
        if spec.electrostatics == "rf":
            if box32 is None or box_acc is None or nbr_raw is None:
                msg = "RF evaluation requires periodic box and neighbor list"
                raise RuntimeError(msg)
            if not mixed:
                if tp_nl is None:
                    msg = "RF single-precision evaluation requires tile_pair_nl"
                    raise RuntimeError(msg)
                e_lj = cast(
                    "jnp.ndarray",
                    pallas_nonbonded_tile_pairs(
                        r32,
                        jnp.zeros_like(bun32.charges),
                        bun32.sigmas,
                        bun32.epsilons,
                        tp_nl,
                        atom_mask=bun32.atom_mask,
                        box=box32,
                        pme_alpha=0.0,
                        cutoff=float(spec.cutoff_A),
                        lj_switch_width=0.0,
                        include_pme_excl_erf=False,
                    ),
                ).astype(r_in.dtype)
                e_rf = _rf_direct_coulomb_nl_energy(
                    r32,
                    bun32.charges,
                    bun32.atom_mask,
                    nbr_raw.idx,
                    bun32.excl_dense_indices,
                    box32,
                    float(spec.cutoff_A),
                    rf_dielectric,
                ).astype(r_in.dtype)
                e_exc = pallas_exceptions_14(
                    r32,
                    bun32.exception_pairs,
                    bun32.exception_sigmas,
                    bun32.exception_epsilons,
                    bun32.exception_chargeprods,
                    bun32.exception_mask,
                    box=box32,
                ).astype(r_in.dtype)
                return e_lj + e_rf + e_exc
            e_lj = chunked_lj_energy_nl(
                r_in,
                bun_acc.sigmas,
                bun_acc.epsilons,
                bun_acc.excl_dense_indices,
                bun_acc.excl_dense_scales_vdw,
                nbr_raw.idx,
                disp_acc,
                cutoff=float(spec.cutoff_A),
                tile_size=128,
                switch_width=0.0,
            )
            e_rf = _rf_direct_coulomb_nl_energy(
                r_in,
                bun_acc.charges,
                bun_acc.atom_mask,
                nbr_raw.idx,
                bun_acc.excl_dense_indices,
                box_acc,
                float(spec.cutoff_A),
                rf_dielectric,
            )
            return e_lj + e_rf + exc_xla_fn(r_in)
        # Non-periodic GBSA direct vacuum LJ + Coulomb + 1-4 exceptions (pre-masked before r^-12!)
        n_pad = int(r_in.shape[0])
        p1, p2 = bun_acc.excl_indices[:, 0], bun_acc.excl_indices[:, 1]
        vex = bun_acc.excl_mask & (p1 >= 0) & (p2 >= 0) & (p1 < n_pad) & (p2 < n_pad)
        p1s, p2s = jnp.where(vex, p1, 0), jnp.where(vex, p2, 0)
        excl_mat = (
            jnp.zeros((n_pad, n_pad), dtype=jnp.bool_)
            .at[p1s, p2s]
            .max(vex)
            .at[p2s, p1s]
            .max(vex)
        )
        i_m = jnp.arange(n_pad)[:, None]
        j_m = jnp.arange(n_pad)[None, :]
        pair_m = bun_acc.atom_mask[:, None] & bun_acc.atom_mask[None, :] & (i_m < j_m) & (~excl_mat)
        dr = r_in[:, None, :] - r_in[None, :, :]
        eps_r2 = jnp.asarray(1e-24 if r_in.dtype == jnp.float64 else 1e-12, dtype=r_in.dtype)
        dist = jnp.sqrt(jnp.sum(dr * dr, axis=-1) + eps_r2)
        dist_s = jnp.where(pair_m, dist, jnp.asarray(1.0, dtype=r_in.dtype))
        sig_ij = 0.5 * (bun_acc.sigmas[:, None] + bun_acc.sigmas[None, :])
        eps_ij = jnp.sqrt(jnp.maximum(bun_acc.epsilons[:, None] * bun_acc.epsilons[None, :], 0.0))
        sr6 = (sig_ij / dist_s) ** 6
        u_unexcl = 4.0 * eps_ij * (sr6 * sr6 - sr6) + jnp.asarray(COULOMB_CONSTANT, dtype=r_in.dtype) * (
            bun_acc.charges[:, None] * bun_acc.charges[None, :] / dist_s
        )
        e_unexcl = jnp.sum(jnp.where(pair_m, u_unexcl, 0.0))
        e_exc = (
            pallas_exceptions_14(
                r32,
                bun32.exception_pairs,
                bun32.exception_sigmas,
                bun32.exception_epsilons,
                bun32.exception_chargeprods,
                bun32.exception_mask,
                box=None,
            ).astype(r_in.dtype)
            if not mixed
            else exc_xla_fn(r_in)
        )
        return e_unexcl + e_exc

    def _recip_e(r_in: jnp.ndarray) -> jnp.ndarray:
        if spec.electrostatics != "pme" or spec.pme_grid is None:
            return jnp.asarray(0.0, dtype=r_in.dtype)
        if box32 is None or box_acc is None:
            msg = "SPME reciprocal evaluation requires periodic box"
            raise RuntimeError(msg)
        # Note: `spme_energy_with_forces` (`pme.py:836`) already includes `spme_self_energy`!
        if not mixed:
            e_recip_plus_self = pme_mod.spme_energy_with_forces(
                r_in.astype(jnp.float32),
                bun32.charges,
                bun32.atom_mask,
                box32,
                spec.pme_grid,
                float(spec.pme_alpha),
                4,
                True,
            ).astype(r_in.dtype)
        else:
            e_recip_plus_self = pme_mod.spme_energy_with_forces(
                r_in,
                bun_acc.charges,
                bun_acc.atom_mask,
                box_acc,
                spec.pme_grid,
                float(spec.pme_alpha),
                4,
                False,
            )
        e_bg = pme_mod.spme_background_energy(
            bun_acc.charges, bun_acc.atom_mask, float(spec.pme_alpha), box_acc
        )
        return e_recip_plus_self + e_bg

    gbsa_tile_list = (
        build_gbsa_tile_pair_list(bun32.positions, atom_mask=bun32.atom_mask)
        if (spec.electrostatics == "gbsa" and not mixed)
        else None
    )

    def _gbsa_e(r_in: jnp.ndarray) -> jnp.ndarray:
        if spec.electrostatics != "gbsa":
            return jnp.asarray(0.0, dtype=r_in.dtype)
        # OpenMM `implicit/obc2.xml` uses `28.3919551 * (r + 0.14)^2 * (r/B)^6` in kJ/(mol*nm^2),
        # where `28.3919551 = 4 * pi * 2.25936` already absorbs `4*pi`; Prolix multiplies by `4*pi`,
        # so `surface_tension = 28.3919551 / (4 * pi * 418.4) = 0.005400000 kcal/(mol*A^2)`.
        omm_ace_st = 28.3919551 / (4.0 * math.pi * 418.4)
        if not mixed:
            return pallas_gbsa_tile(
                r_in.astype(jnp.float32),
                bun32.charges,
                bun32.radii,
                bun32.scaled_radii,
                tile_list=gbsa_tile_list,
                atom_mask=bun32.atom_mask,
                solvent_dielectric=78.5,
                solute_dielectric=1.0,
                dielectric_offset=0.09,
                surface_tension=omm_ace_st,
                include_gb=True,
                include_ace=True,
                include_coulomb=False,
                include_lj=False,
            ).astype(r_in.dtype)
        n_pad = r_in.shape[0]
        valid_m = (bun_acc.atom_mask[:, None] & bun_acc.atom_mask[None, :]).astype(r_in.dtype) * (
            1.0 - jnp.eye(n_pad, dtype=r_in.dtype)
        )
        e_gb, born_radii = generalized_born.compute_gb_energy(
            r_in,
            bun_acc.charges,
            bun_acc.radii,
            solvent_dielectric=78.5,
            solute_dielectric=1.0,
            dielectric_offset=0.09,
            mask=valid_m,
            energy_mask=bun_acc.atom_mask[:, None] & bun_acc.atom_mask[None, :],
            scaled_radii=bun_acc.scaled_radii,
        )
        e_ace = jnp.sum(
            jnp.where(
                bun_acc.atom_mask,
                generalized_born.compute_ace_nonpolar_energy(
                    bun_acc.radii, born_radii, surface_tension=omm_ace_st
                ),
                0.0,
            )
        )
        return e_gb + e_ace

    e_tail_val = (
        float(
            explicit_corrections.lj_dispersion_tail_energy(
                box_acc, bun_acc.sigmas, bun_acc.epsilons, float(spec.cutoff_A), bun_acc.atom_mask
            )
        )
        if (is_pbc and box_acc is not None)
        else 0.0
    )

    @jax.jit
    def _eval_all(r_in: jnp.ndarray):
        eb = _bonds_e(r_in)
        ea = _angles_e(r_in)
        et = _torsions_e(r_in)
        ec = _cmap_e(r_in)
        ed = _direct_nb_e(r_in)
        er = _recip_e(r_in)
        eg = _gbsa_e(r_in)

        def _tot_fn(x: jnp.ndarray) -> jnp.ndarray:
            return (
                _bonds_e(x)
                + _angles_e(x)
                + _torsions_e(x)
                + _cmap_e(x)
                + _direct_nb_e(x)
                + _recip_e(x)
                + _gbsa_e(x)
            )

        e_no_tail, grad_x = jax.value_and_grad(_tot_fn)(r_in)
        return (eb, ea, et, ec, ed, er, eg, e_no_tail), -grad_x

    vals, forces_pad = _eval_all(bun_acc.positions)
    eb, ea, et, ec, ed, er, eg, e_no_tail = [float(v) for v in vals]
    terms = {
        "bonds": eb,
        "angles": ea,
        "torsions": et,
        "cmap": ec,
        "direct_nonbonded": ed,
        "pme_reciprocal": er,
        "gbsa": eg,
        "tail_correction": e_tail_val,
        "total": e_no_tail + e_tail_val,
    }
    forces_real = np.asarray(forces_pad[:n_real], dtype=np.float64)
    return terms, forces_real


def _eval_prolix_openmm_context(spec: ValidatedSystemSpec) -> tuple[dict[str, float], np.ndarray]:
    """Evaluate `prolix.openmm.ProlixContext` (`MK-69` unified backend) on `spec.omm_system`."""
    import openmm as mm
    from openmm import unit as _omm_unit
    unit = cast("Any", _omm_unit)

    from prolix.openmm import ProlixContext, ProlixOpenMMConfig

    has_pbc = bool(spec.bundle_f32.shape_spec.has_pbc)
    if not has_pbc:
        nb_eng = "gbsa_tile" if spec.electrostatics == "gbsa" else "dense"
    else:
        nb_eng = "neighbor_list" if spec.n_real_atoms >= 4000 else "auto"
    is_cpu = jax.default_backend() == "cpu"
    cfg = ProlixOpenMMConfig(
        precision="mixed",
        nonbonded_engine=nb_eng,
        use_pallas=True,
        use_pallas_pme=True,
        pallas_interpret=is_cpu,
        use_dispersion_correction=has_pbc,
        use_size_buckets=(spec.n_real_atoms <= 50000),
    )
    sys_grp = _assign_openmm_force_groups(spec.omm_system, split_reciprocal=False)
    try:
        ctx = ProlixContext(sys_grp, mm.VerletIntegrator(0.001), config=cfg)
    except TypeError:
        cfg = dataclasses.replace(cfg, tp_group=1, device_index="0")
        ctx = ProlixContext(sys_grp, mm.VerletIntegrator(0.001), config=cfg)
    ctx.setPositions((spec.positions_A * 0.1) * unit.nanometer)

    def _ge(grp: int) -> float:
        st = ctx.getState(getEnergy=True, groups={grp})
        return float(st.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))

    def _eval_ctx_terms_and_forces():
        eb_v = _ge(0)
        ea_v = _ge(1)
        et_v = _ge(2)
        ec_v = _ge(3)
        enb_v = _ge(5)
        eg_v = _ge(7) if spec.electrostatics == "gbsa" else 0.0
        st_t = ctx.getState(getEnergy=True, getForces=True)
        etot_v = float(st_t.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))
        f_v = np.asarray(
            st_t.getForces(asNumpy=True).value_in_unit(unit.kilocalories_per_mole / unit.angstrom),
            dtype=np.float64,
        )
        return eb_v, ea_v, et_v, ec_v, enb_v, eg_v, etot_v, f_v

    try:
        eb, ea, et, ec, enb_with_tail, eg, e_total, f_arr = _eval_ctx_terms_and_forces()
    except (jax.errors.TracerArrayConversionError, TypeError):
        fb_eng = "neighbor_list" if (has_pbc and spec.n_real_atoms >= 4000) else "dense"
        cfg = dataclasses.replace(cfg, nonbonded_engine=fb_eng, use_pallas=False)
        ctx = ProlixContext(sys_grp, mm.VerletIntegrator(0.001), config=cfg)
        ctx.setPositions((spec.positions_A * 0.1) * unit.nanometer)
        eb, ea, et, ec, enb_with_tail, eg, e_total, f_arr = _eval_ctx_terms_and_forces()

    if has_pbc:
        cfg_notail = dataclasses.replace(cfg, use_dispersion_correction=False)
        ctx_nt = ProlixContext(sys_grp, mm.VerletIntegrator(0.001), config=cfg_notail)
        ctx_nt.setPositions((spec.positions_A * 0.1) * unit.nanometer)
        e_no_tail = float(
            ctx_nt.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        )
        e_tail = e_total - e_no_tail
    else:
        e_tail = 0.0
    terms = {
        "bonds": eb,
        "angles": ea,
        "torsions": et,
        "cmap": ec,
        "nonbonded_direct_plus_recip": enb_with_tail - e_tail,
        "gbsa": eg,
        "tail_correction": e_tail,
        "total": e_total,
    }
    return terms, f_arr[: spec.n_real_atoms]


def _eval_kups_cross_unit_parity(spec: ValidatedSystemSpec) -> dict[str, Any]:
    """Evaluate kUPS-unit (`eV`, `eV/A`, `t_kups`) parity via `prolix.physics.kups_adapter`."""
    bun64 = spec.bundle_f64
    n_real = spec.n_real_atoms
    disp64, _ = displacement_fn_for_bundle(bun64)
    b_idx = jnp.where(bun64.bond_mask[:, None], bun64.bond_idx, 0)
    b_prm_kcal = jnp.where(bun64.bond_mask[:, None], bun64.bond_params, 0.0)
    # Convert k_b from kcal/(mol*A^2) to eV/A^2; r_0 stays in A
    b_prm_ev = jnp.stack(
        [
            b_prm_kcal[:, 0],
            cast(
                "jnp.ndarray",
                kups_adapter.spring_constant_kcal_mol_to_ev_per_angstrom_sq(cast("Any", b_prm_kcal[:, 1])),
            ),
        ],
        axis=-1,
    )
    def bond_fn(r_x: jnp.ndarray, p_x: jnp.ndarray) -> jnp.ndarray:
        return _bond_energy_masked(r_x, b_idx, p_x, bun64.bond_mask, disp64)

    if spec.electrostatics == "pme":
        box64 = jnp.diag(bun64.box)
        nl_fn = cast(
            "Any",
            nl_mod.make_neighbor_list_fn(
                disp64,
                box64,
                float(spec.cutoff_A),
                capacity_multiplier=1.60,
                dr_threshold=0.0,
                atom_mask=bun64.atom_mask,
                use_pallas_nl=False,
            ),
        )
        nbr = nl_fn.allocate(bun64.positions)
        if int(np.asarray(nbr.error.code)) != 0:
            nl_fn = cast(
                "Any",
                nl_mod.make_neighbor_list_fn(
                    disp64,
                    box64,
                    float(spec.cutoff_A),
                    capacity_multiplier=2.20,
                    dr_threshold=0.0,
                    atom_mask=bun64.atom_mask,
                    use_pallas_nl=False,
                ),
            )
            nbr = nl_fn.allocate(bun64.positions)
        eps_ev = kups_adapter.energy_kcal_mol_to_ev(bun64.epsilons)
        q_ev_scaled = bun64.charges * math.sqrt(kups_adapter.KCAL_MOL_TO_EV)
        exc_fn_ev = bonded.make_exception_pair_energy_fn(
            disp64,
            bun64.exception_pairs,
            bun64.exception_sigmas,
            kups_adapter.energy_kcal_mol_to_ev(bun64.exception_epsilons),
            bun64.exception_chargeprods * kups_adapter.KCAL_MOL_TO_EV,
            mask=bun64.exception_mask,
        )

        @jax.jit
        def _kups_ev(pos: jnp.ndarray):
            def _u_ev(x: jnp.ndarray) -> jnp.ndarray:
                eb_ev = bond_fn(x, b_prm_ev)
                elj_ev = chunked_lj_energy_nl(
                    x,
                    bun64.sigmas,
                    eps_ev,
                    bun64.excl_dense_indices,
                    bun64.excl_dense_scales_vdw,
                    nbr.idx,
                    disp64,
                    cutoff=float(spec.cutoff_A),
                    tile_size=128,
                    switch_width=0.0,
                )
                # Note: `chunked_coulomb_energy_nl` already evaluates `(pair_scale - erf(alpha*r))/r`,
                # which includes `-erf(alpha*r)/r` for all excluded pairs with `r < cutoff`; only
                # subtract `e_erf_far_ev` for any excluded pairs beyond `cutoff`.
                ec_ev = chunked_coulomb_energy_nl(
                    x,
                    q_ev_scaled,
                    bun64.excl_dense_indices,
                    bun64.excl_dense_scales_elec,
                    nbr.idx,
                    disp64,
                    pme_alpha=float(spec.pme_alpha),
                    coulomb_constant=float(COULOMB_CONSTANT),
                    cutoff=float(spec.cutoff_A),
                    tile_size=128,
                )
                e_erf_far_ev = _pme_excl_erf_energy(
                    x,
                    q_ev_scaled,
                    bun64.excl_indices,
                    bun64.excl_mask,
                    box64,
                    float(spec.pme_alpha),
                    min_dist=float(spec.cutoff_A),
                )
                return eb_ev + elj_ev + ec_ev - e_erf_far_ev + exc_fn_ev(x)

            eb_only = bond_fn(pos, b_prm_ev)
            utot, g_ev = jax.value_and_grad(_u_ev)(pos)
            return eb_only, utot - eb_only, -g_ev

        eb_ev, ed_ev, f_ev = _kups_ev(bun64.positions)
        return {
            "bonds_kcal_from_ev": float(kups_adapter.energy_ev_to_kcal_mol(eb_ev)),
            "direct_nonbonded_kcal_from_ev": float(kups_adapter.energy_ev_to_kcal_mol(ed_ev)),
            "_forces_bonds_plus_direct_kcal_A": np.asarray(
                kups_adapter.force_ev_per_angstrom_to_kcal_mol(f_ev[:n_real]), dtype=np.float64
            ),
        }

    eb_ev = float(bond_fn(bun64.positions, b_prm_ev))
    return {
        "bonds_kcal_from_ev": float(kups_adapter.energy_ev_to_kcal_mol(eb_ev)),
    }


def _eval_kups_baoab_single_step_check() -> dict[str, float]:
    """Verify `< 1e-12` float64 single-step BAOAB parity between kUPS (`eV/A/amu/t_kups`) and Prolix (`AKMA`)."""
    rng = np.random.default_rng(42)
    n = 64
    r0 = jnp.asarray(rng.normal(0.0, 2.0, size=(n, 3)), dtype=jnp.float64)
    v_kups0 = jnp.asarray(rng.normal(0.0, 0.05, size=(n, 3)), dtype=jnp.float64)
    m = jnp.full((n,), 12.011, dtype=jnp.float64)
    noise = jnp.asarray(rng.normal(0.0, 1.0, size=(n, 3)), dtype=jnp.float64)
    k_ev = 0.05
    k_kcal = kups_adapter.spring_constant_ev_per_angstrom_sq_to_kcal_mol(k_ev)
    dt_fs = 1.0
    gamma_ps = 5.0
    temp_K = 300.0

    # kUPS step (eV, A, amu, t_kups)
    dt_k = kups_adapter.dt_fs_to_kups(dt_fs)
    gam_k = kups_adapter.gamma_ps_to_kups(gamma_ps)
    kT_ev = temp_K * kups_adapter.KUPS_BOLTZMANN_EV_PER_K
    p_k = m[:, None] * v_kups0
    f_k0 = -k_ev * r0
    p_k = p_k + 0.5 * dt_k * f_k0
    r_k = r0 + 0.5 * dt_k * (p_k / m[:, None])
    c1_k = math.exp(-gam_k * dt_k)
    p_k = c1_k * p_k + jnp.sqrt(m[:, None] * kT_ev * (1.0 - c1_k * c1_k)) * noise
    r_k = r_k + 0.5 * dt_k * (p_k / m[:, None])
    f_k1 = -k_ev * r_k
    p_k = p_k + 0.5 * dt_k * f_k1

    # Prolix step (kcal/mol, A, amu, t_akma) using exact CODATA Boltzmann derived from eV
    dt_a = kups_adapter.dt_fs_to_akma(dt_fs)
    gam_a = kups_adapter.gamma_ps_to_akma(gamma_ps)
    kT_kcal = kups_adapter.energy_ev_to_kcal_mol(kT_ev)
    p_a = kups_adapter.momentum_kups_to_akma(m[:, None] * v_kups0)
    f_a0 = -k_kcal * r0
    p_a = p_a + 0.5 * dt_a * f_a0
    r_a = r0 + 0.5 * dt_a * (p_a / m[:, None])
    c1_a = math.exp(-gam_a * dt_a)
    p_a = c1_a * p_a + jnp.sqrt(m[:, None] * kT_kcal * (1.0 - c1_a * c1_a)) * noise
    r_a = r_a + 0.5 * dt_a * (p_a / m[:, None])
    f_a1 = -k_kcal * r_a
    p_a = p_a + 0.5 * dt_a * f_a1

    pos_max_err = float(jnp.max(jnp.abs(r_a - r_k)))
    mom_max_err = float(jnp.max(jnp.abs(p_a - kups_adapter.momentum_kups_to_akma(p_k))))
    return {
        "baoab_position_max_abs_err_A": pos_max_err,
        "baoab_momentum_max_abs_err_akma": mom_max_err,
        "pass_lt_1e12": bool(pos_max_err < 1.0e-12 and mom_max_err < 1.0e-12),
    }


def run_tier1_single_point_parity(
    *,
    quick: bool,
    allow_cpu_smoke: bool,
    existing_tier1: dict[str, Any] | None,
    checkpoint_cb: Any,
) -> dict[str, Any]:
    """Execute Tier 1 across all 9 canonical systems and 6 engine configurations."""
    print("\n=== [Tier 1/4] Single-Point Term-by-Term Energy & Force Parity ===")
    out: dict[str, Any] = dict(existing_tier1 or {})
    out.setdefault("systems", {})
    out["kups_propagator_unit_check"] = _eval_kups_baoab_single_step_check()

    dhfr_pme_cache: ValidatedSystemSpec | None = None

    def _get_dhfr_pme() -> ValidatedSystemSpec:
        nonlocal dhfr_pme_cache
        if dhfr_pme_cache is None:
            dhfr_pme_cache = build_dhfr_system(
                electrostatics="pme", allow_cpu_smoke=allow_cpu_smoke, minimize_iters=30 if quick else 80
            )
        return dhfr_pme_cache

    builders: list[tuple[str, Any]] = [
        ("1UAO", lambda: build_1uao_system(allow_cpu_smoke=allow_cpu_smoke, minimize_iters=30 if quick else 80)),
        ("1VII", lambda: build_1vii_system(allow_cpu_smoke=allow_cpu_smoke, minimize_iters=30 if quick else 80)),
        ("1UBQ", lambda: build_1ubq_system(allow_cpu_smoke=allow_cpu_smoke, minimize_iters=30 if quick else 80)),
        ("5DFR_GBSA", lambda: build_5dfr_gbsa_system(allow_cpu_smoke=allow_cpu_smoke)),
        ("Holo_Lysozyme", lambda: build_holo_lysozyme_system(_get_dhfr_pme(), allow_cpu_smoke=allow_cpu_smoke)),
        (
            "DHFR_RF",
            lambda: build_dhfr_system(
                electrostatics="rf", allow_cpu_smoke=allow_cpu_smoke, minimize_iters=30 if quick else 80
            ),
        ),
        ("DHFR_PME", _get_dhfr_pme),
        (
            "Holo_DHFR",
            lambda: build_holo_dhfr_system(
                allow_cpu_smoke=allow_cpu_smoke, minimize_iters=50 if quick else 150
            ),
        ),
    ]
    if not (quick and allow_cpu_smoke):
        builders.append(("ApoA1_92k", lambda: build_apoa1_system(allow_cpu_smoke=allow_cpu_smoke)))

    for sys_name, build_fn in builders:
        if sys_name in out["systems"]:
            print(f"  [Tier 1] {sys_name}: resumed from checkpoint")
            continue
        t0 = time.perf_counter()
        spec = build_fn()
        omm_dbl_terms, omm_dbl_f, prov_dbl = _eval_openmm_decomposed(
            spec, precision="double", allow_cpu_smoke=allow_cpu_smoke
        )
        omm_mix_terms, omm_mix_f, prov_mix = _eval_openmm_decomposed(
            spec, precision="mixed", allow_cpu_smoke=allow_cpu_smoke
        )
        pallas_s_terms, pallas_s_f = _eval_prolix_pallas_decomposed(spec, mixed=False)
        pallas_m_terms, pallas_m_f = _eval_prolix_pallas_decomposed(spec, mixed=True)
        pctx_terms, pctx_f = _eval_prolix_openmm_context(spec)
        kups_res = _eval_kups_cross_unit_parity(spec)
        kups_clean = {k: v for k, v in kups_res.items() if not k.startswith("_")}
        if "direct_nonbonded_kcal_from_ev" in kups_clean:
            kups_clean["bonds_abs_diff_vs_omm_double_kcal_mol"] = abs(
                kups_clean["bonds_kcal_from_ev"] - omm_dbl_terms["bonds"]
            )
            kups_clean["direct_nb_abs_diff_vs_omm_double_kcal_mol"] = abs(
                kups_clean["direct_nonbonded_kcal_from_ev"] - omm_dbl_terms["direct_nonbonded"]
            )

        def _summarize_engine(
            terms: dict[str, float],
            forces: np.ndarray,
            _ref_terms: dict[str, float] = omm_dbl_terms,
            _ref_f: np.ndarray = omm_dbl_f,
            _n_real: int = spec.n_real_atoms,
        ) -> dict[str, Any]:
            term_diffs = {
                k: float(terms[k] - _ref_terms[k])
                for k in terms
                if k in _ref_terms
            }
            if "nonbonded_direct_plus_recip" in terms:
                term_diffs["nonbonded_direct_plus_recip"] = float(
                    terms["nonbonded_direct_plus_recip"]
                    - (_ref_terms["direct_nonbonded"] + _ref_terms["pme_reciprocal"])
                )
            return {
                "energies_kcal_mol": terms,
                "energy_diffs_vs_openmm_double_kcal_mol": term_diffs,
                "per_atom_total_energy_mae_kcal_mol": abs(terms["total"] - _ref_terms["total"])
                / max(_n_real, 1),
                **_force_parity_metrics(forces, _ref_f),
            }

        rec = {
            "n_real_atoms": spec.n_real_atoms,
            "n_padded_atoms": int(spec.bundle_f32.positions.shape[0]),
            "n_waters": spec.n_waters,
            "electrostatics": spec.electrostatics,
            "pme_grid": list(spec.pme_grid) if spec.pme_grid is not None else None,
            "pme_alpha": spec.pme_alpha,
            "cutoff_A": spec.cutoff_A,
            "openmm_cuda_provenance": {"double": prov_dbl, "mixed": prov_mix},
            "openmm_cuda_double": {"energies_kcal_mol": omm_dbl_terms},
            "openmm_cuda_mixed": _summarize_engine(omm_mix_terms, omm_mix_f),
            "prolix_pallas_single": _summarize_engine(pallas_s_terms, pallas_s_f),
            "prolix_pallas_mixed": _summarize_engine(pallas_m_terms, pallas_m_f),
            "prolix_openmm_context": _summarize_engine(pctx_terms, pctx_f),
            "kups_adapter": kups_clean,
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }
        out["systems"][sys_name] = rec
        checkpoint_cb(out)
        print(
            f"  [Tier 1] {sys_name:14s} (N={spec.n_real_atoms:6d}): "
            f"dE_single={rec['prolix_pallas_single']['energy_diffs_vs_openmm_double_kcal_mol']['total']:+.4f}, "
            f"dE_mixed={rec['prolix_pallas_mixed']['energy_diffs_vs_openmm_double_kcal_mol']['total']:+.4f} kcal/mol, "
            f"F_RMSE_single={rec['prolix_pallas_single']['force_rmse_kcal_mol_A']:.4e}, "
            f"F_RMSE_pctx={rec['prolix_openmm_context']['force_rmse_kcal_mol_A']:.4e} ({rec['elapsed_s']:.1f}s)"
        )

    return out


# =============================================================================
# TIER 2: Analytical `custom_vjp` vs `float64` Central Finite-Difference Parity
# =============================================================================


def _directional_fd_check(
    scalar_f64_fn: Any,
    vjp_grad_fn: Any,
    theta0: jnp.ndarray,
    active_mask: np.ndarray,
    *,
    h: float,
    n_dirs: int,
    seed: int,
) -> dict[str, Any]:
    """Compare analytical `vjp_grad_fn(theta0)` against `float64` central FD `(f(theta+h*u)-f(theta-h*u))/(2h)`."""
    rng = np.random.default_rng(seed)
    theta0_f64 = jnp.asarray(theta0, dtype=jnp.float64)
    g_anal = np.asarray(vjp_grad_fn(theta0_f64), dtype=np.float64)
    mask_bool = np.broadcast_to(np.asarray(active_mask, dtype=bool), g_anal.shape)

    proj_anal: list[float] = []
    proj_fd: list[float] = []

    # 1. Orthonormal masked random directions
    raw_dirs: list[np.ndarray] = []
    for _ in range(n_dirs):
        u = rng.normal(0.0, 1.0, size=g_anal.shape) * mask_bool
        for prev in raw_dirs:
            u = u - np.sum(u * prev) * prev
        nrm = float(np.linalg.norm(u))
        if nrm > 1e-12:
            u = u / nrm
            raw_dirs.append(u)

    # 2. Top-magnitude gradient coordinate basis vectors (catches sparse per-element sign/chain-rule bugs)
    flat_abs = np.abs(g_anal * mask_bool).reshape(-1)
    top_k = min(8, int(np.sum(flat_abs > 1e-10)))
    if top_k > 0:
        top_indices = np.argsort(-flat_abs)[:top_k]
        for flat_i in top_indices:
            basis = np.zeros_like(g_anal)
            basis.reshape(-1)[int(flat_i)] = 1.0
            raw_dirs.append(basis)

    f_jit = jax.jit(scalar_f64_fn)
    for u in raw_dirs:
        u_j = jnp.asarray(u, dtype=jnp.float64)
        ep = float(f_jit(theta0_f64 + h * u_j))
        em = float(f_jit(theta0_f64 - h * u_j))
        fd_val = (ep - em) / (2.0 * h)
        an_val = float(np.sum(g_anal * u))
        proj_fd.append(fd_val)
        proj_anal.append(an_val)

    pa = np.asarray(proj_anal, dtype=np.float64)
    pf = np.asarray(proj_fd, dtype=np.float64)
    diff = pa - pf
    rel_l2 = float(np.linalg.norm(diff) / max(float(np.linalg.norm(pf)), 1e-12))
    max_abs = float(np.max(np.abs(diff))) if diff.size > 0 else 0.0
    cos_sim = (
        float(np.dot(pa, pf) / (np.linalg.norm(pa) * np.linalg.norm(pf)))
        if (np.linalg.norm(pa) > 1e-15 and np.linalg.norm(pf) > 1e-15)
        else 1.0
    )
    return {
        "n_probes": len(raw_dirs),
        "fd_step_h": h,
        "rel_l2_error": rel_l2,
        "max_abs_error": max_abs,
        "cosine_similarity": cos_sim,
        "pass_lt_1e5": bool(rel_l2 < 1.0e-5),
    }


def run_tier2_custom_vjp_fd_parity(
    *,
    quick: bool,
    allow_cpu_smoke: bool,
    checkpoint_cb: Any,
) -> dict[str, Any]:
    """Execute Tier 2 `custom_vjp` vs `float64` central FD parity across PME, RF, and GBSA."""
    print("\n=== [Tier 2/4] Analytical custom_vjp vs float64 Central Finite-Difference Parity ===")
    n_dirs = 6 if quick else 16
    results: dict[str, Any] = {"pme": {}, "rf": {}, "gbsa": {}}

    # --- Regime A: PME (`1UAO`, 4,270 real atoms in 4,608 bucket) ---
    uao = build_1uao_system(allow_cpu_smoke=allow_cpu_smoke, minimize_iters=25 if quick else 60)
    b64 = uao.bundle_f64
    b32 = uao.bundle_f32
    disp64, _ = displacement_fn_for_bundle(b64)
    box64 = jnp.diag(b64.box)
    box32 = jnp.diag(b32.box).astype(jnp.float32)
    # Pick a cutoff near `uao.cutoff_A` at the midpoint of the largest pair-distance gap in
    # `[c0 - 0.05, c0 + 0.05] Å` and allocate `nbr_raw` with `dr_threshold=0.25 Å` skin so zero
    # atom pair enters/exits the neighbor list or crosses `cutoff_t2` under `±1e-5 Å` FD steps.
    pos_real_np = np.asarray(b64.positions[: uao.n_real_atoms], dtype=np.float64)
    box_np = np.asarray(box64, dtype=np.float64)
    c0_t2 = float(uao.cutoff_A)
    shell_dists_list: list[np.ndarray] = []
    for i0 in range(0, uao.n_real_atoms, 256):
        dr_blk = pos_real_np[i0 : i0 + 256, None, :] - pos_real_np[None, :, :]
        dr_blk -= box_np[None, None, :] * np.round(dr_blk / box_np[None, None, :])
        d_blk = np.sqrt(np.sum(dr_blk * dr_blk, axis=-1))
        in_shell = d_blk[(d_blk >= c0_t2 - 0.05) & (d_blk <= c0_t2 + 0.05)]
        if in_shell.size > 0:
            shell_dists_list.append(in_shell)
    if shell_dists_list:
        sorted_shell = np.unique(np.concatenate([np.array([c0_t2 - 0.05, c0_t2 + 0.05]), *shell_dists_list]))
        gaps = np.diff(sorted_shell)
        best_gap_idx = int(np.argmax(gaps))
        cutoff_t2 = float(0.5 * (sorted_shell[best_gap_idx] + sorted_shell[best_gap_idx + 1]))
    else:
        cutoff_t2 = c0_t2
    switch_w_t2 = 1.0

    nl_raw_fn = cast(
        "Any",
        nl_mod.make_neighbor_list_fn(
            disp64,
            box64,
            cutoff_t2,
            capacity_multiplier=1.60,
            dr_threshold=0.25,
            atom_mask=b64.atom_mask,
            use_pallas_nl=False,
        ),
    )
    nbr_raw = nl_raw_fn.allocate(b64.positions)
    nbr_enc = _encode_bundle_nl(nbr_raw, b32)

    b_idx = jnp.where(b64.bond_mask[:, None], b64.bond_idx, 0)
    a_idx = jnp.where(b64.angle_mask[:, None], b64.angle_idx, 0)
    d_idx = jnp.where(b64.dihedral_mask[:, None], b64.dihedral_idx, 0)
    exc_fn = bonded.make_exception_pair_energy_fn(
        disp64,
        b64.exception_pairs,
        b64.exception_sigmas,
        b64.exception_epsilons,
        b64.exception_chargeprods,
        mask=b64.exception_mask,
    )

    def _pme_total_f64(pos, q, sig, eps, bp, ap, dp):
        eb = (
            _bond_energy_masked(pos, b_idx, bp, b64.bond_mask, disp64)
            + _angle_energy_masked(pos, a_idx, ap, b64.angle_mask, disp64)
            + _dihedral_energy_masked(pos, d_idx, dp, b64.dihedral_mask, disp64)
            + exc_fn(pos)
        )
        elj = chunked_lj_energy_nl(
            pos, sig, eps, b64.excl_dense_indices, b64.excl_dense_scales_vdw, nbr_raw.idx, disp64, cutoff_t2, 128, switch_w_t2
        )
        ec = chunked_coulomb_energy_nl(
            pos, q, b64.excl_dense_indices, b64.excl_dense_scales_elec, nbr_raw.idx, disp64, float(uao.pme_alpha), float(COULOMB_CONSTANT), cutoff_t2, 128
        )
        er = (
            pme_mod.spme_energy_with_forces(pos, q, b64.atom_mask, box64, uao.pme_grid, float(uao.pme_alpha), 4, False)
            + pme_mod.spme_background_energy(q, b64.atom_mask, float(uao.pme_alpha), box64)
        )
        et = explicit_corrections.lj_dispersion_tail_energy(box64, sig, eps, cutoff_t2, b64.atom_mask)
        return eb + elj + ec + er + et

    def _pme_total_pallas_f32(pos, q, sig, eps, bp, ap, dp):
        pos32 = pos.astype(jnp.float32)
        q32 = q.astype(jnp.float32)
        sig32 = sig.astype(jnp.float32)
        eps32 = eps.astype(jnp.float32)
        eb = cast(
            "jnp.ndarray",
            pallas_bonded_all_fused(
                pos32,
                bond_indices=b_idx,
                bond_params=bp.astype(jnp.float32),
                bond_mask=b32.bond_mask,
                angle_indices=a_idx,
                angle_params=ap.astype(jnp.float32),
                angle_mask=b32.angle_mask,
                dihedral_indices=d_idx,
                dihedral_params=dp.astype(jnp.float32),
                dihedral_mask=b32.dihedral_mask,
                exception_pairs=b32.exception_pairs,
                exception_sigmas=b32.exception_sigmas,
                exception_epsilons=b32.exception_epsilons,
                exception_chargeprods=b32.exception_chargeprods,
                exception_mask=b32.exception_mask,
                box=box32,
                dist_eps=1e-10,
                b0_is_ji=True,
            ),
        )
        enb = cast(
            "jnp.ndarray",
            pallas_nonbonded_nl(
                pos32,
                q32,
                sig32,
                eps32,
                b32.atom_mask,
                nbr_enc.idx,
                b32.excl_dense_indices,
                b32.excl_dense_scales_vdw,
                b32.excl_dense_scales_elec,
                box=box32,
                pme_alpha=float(uao.pme_alpha),
                cutoff=cutoff_t2,
                lj_switch_width=switch_w_t2,
                soft_core_lambda=1.0,
                include_pme_excl_erf=True,
                excl_mode="pre_encoded",
            ),
        )
        er = (
            pme_mod.spme_energy_with_forces(pos32, q32, b32.atom_mask, box32, uao.pme_grid, float(uao.pme_alpha), 4, True)
            + pme_mod.spme_background_energy(q32, b32.atom_mask, float(uao.pme_alpha), box32)
        )
        et = explicit_corrections.lj_dispersion_tail_energy(box32, sig32, eps32, cutoff_t2, b32.atom_mask)
        return eb + enb + er + et

    bp0 = b64.bond_params
    ap0 = b64.angle_params
    dp0 = b64.dihedral_params[..., :3]
    atom_mask_np = np.asarray(b64.atom_mask, dtype=bool)
    pos_mask_np = np.broadcast_to(atom_mask_np[:, None], b64.positions.shape)

    pme_targets = [
        ("positions", b64.positions, pos_mask_np, 0, 1.0e-5),
        ("charges", b64.charges, atom_mask_np, 1, 1.0e-5),
        ("sigmas", b64.sigmas, atom_mask_np, 2, 1.0e-5),
        ("epsilons", b64.epsilons, atom_mask_np & (np.asarray(b64.epsilons) > 1e-6), 3, 1.0e-5),
        ("bond_params", bp0, np.broadcast_to(np.asarray(b64.bond_mask)[:, None], bp0.shape), 4, 1.0e-5),
        ("angle_params", ap0, np.broadcast_to(np.asarray(b64.angle_mask)[:, None], ap0.shape), 5, 1.0e-5),
        ("dihedral_params", dp0, np.asarray(b64.dihedral_mask)[:, None] & np.array([[False, False, True]]), 6, 1.0e-5),
    ]
    base_args = [b64.positions, b64.charges, b64.sigmas, b64.epsilons, bp0, ap0, dp0]
    for idx_t, (tname, theta0, tmask, argnum, h_step) in enumerate(pme_targets):
        def _scalar_f(th, _an=argnum):
            args = list(base_args)
            args[_an] = th
            return _pme_total_f64(*args)

        grad_f64 = jax.jit(jax.grad(_scalar_f))
        chk_f64 = _directional_fd_check(
            _scalar_f, grad_f64, theta0, tmask, h=h_step, n_dirs=n_dirs, seed=1000 + idx_t
        )
        grad_pallas = jax.jit(
            jax.grad(lambda th, _an=argnum: _pme_total_pallas_f32(*[*base_args[:_an], th, *base_args[_an + 1 :]]))
        )
        chk_pallas = _directional_fd_check(
            _scalar_f, grad_pallas, theta0, tmask, h=h_step, n_dirs=n_dirs, seed=1000 + idx_t
        )
        results["pme"][tname] = {
            "float64_custom_vjp": chk_f64,
            "float32_pallas_custom_vjp": chk_pallas,
        }
        print(
            f"  [Tier 2 PME]  {tname:16s}: f64_rel_err={chk_f64['rel_l2_error']:.3e} "
            f"(pass={chk_f64['pass_lt_1e5']}), pallas_f32_rel_err={chk_pallas['rel_l2_error']:.3e}"
        )

    # Also verify MK-51 / MK-73 `pallas_nonbonded_tile_pairs` (`morton_tile_pairs`) `custom_vjp`
    tp_nl = build_tile_pair_neighbor_list(
        nbr_raw.idx,
        excl_indices=b32.excl_dense_indices,
        excl_scales_vdw=b32.excl_dense_scales_vdw,
        excl_scales_elec=b32.excl_dense_scales_elec,
        atom_mask=b32.atom_mask,
        positions=b32.positions,
        box=box32,
        cutoff=cutoff_t2,
        morton_sort=True,
    )

    def _pme_total_tile_pairs_f32(pos, q, sig, eps):
        pos32 = pos.astype(jnp.float32)
        q32 = q.astype(jnp.float32)
        sig32 = sig.astype(jnp.float32)
        eps32 = eps.astype(jnp.float32)
        eb = cast(
            "jnp.ndarray",
            pallas_bonded_all_fused(
                pos32,
                bond_indices=b_idx,
                bond_params=bp0.astype(jnp.float32),
                bond_mask=b32.bond_mask,
                angle_indices=a_idx,
                angle_params=ap0.astype(jnp.float32),
                angle_mask=b32.angle_mask,
                dihedral_indices=d_idx,
                dihedral_params=dp0.astype(jnp.float32),
                dihedral_mask=b32.dihedral_mask,
                exception_pairs=b32.exception_pairs,
                exception_sigmas=b32.exception_sigmas,
                exception_epsilons=b32.exception_epsilons,
                exception_chargeprods=b32.exception_chargeprods,
                exception_mask=b32.exception_mask,
                box=box32,
                dist_eps=1e-10,
                b0_is_ji=True,
            ),
        )
        enb = cast(
            "jnp.ndarray",
            pallas_nonbonded_tile_pairs(
                pos32,
                q32,
                sig32,
                eps32,
                tp_nl,
                atom_mask=b32.atom_mask,
                box=box32,
                pme_alpha=float(uao.pme_alpha),
                cutoff=cutoff_t2,
                lj_switch_width=switch_w_t2,
                include_pme_excl_erf=True,
            ),
        )
        er = (
            pme_mod.spme_energy_with_forces(pos32, q32, b32.atom_mask, box32, uao.pme_grid, float(uao.pme_alpha), 4, True)
            + pme_mod.spme_background_energy(q32, b32.atom_mask, float(uao.pme_alpha), box32)
        )
        et = explicit_corrections.lj_dispersion_tail_energy(box32, sig32, eps32, cutoff_t2, b32.atom_mask)
        return eb + enb + er + et

    results["pme_morton_tile_pairs"] = {}
    tp_base = [b64.positions, b64.charges, b64.sigmas, b64.epsilons]
    for idx_t, (tname, theta0, tmask, argnum, h_step) in enumerate(pme_targets[:4]):
        def _scalar_f_tp(th, _an=argnum):
            args = list(base_args)
            args[_an] = th
            return _pme_total_f64(*args)

        try:
            grad_tp = jax.jit(
                jax.grad(
                    lambda th, _an=argnum: _pme_total_tile_pairs_f32(
                        *[*tp_base[:_an], th, *tp_base[_an + 1 :]]
                    )
                )
            )
            chk_tp = _directional_fd_check(
                _scalar_f_tp, grad_tp, theta0, tmask, h=h_step, n_dirs=n_dirs, seed=1500 + idx_t
            )
            results["pme_morton_tile_pairs"][tname] = {"float32_tile_pairs_custom_vjp": chk_tp}
            print(
                f"  [Tier 2 Tile] {tname:16s}: tile_pairs_f32_rel_err={chk_tp['rel_l2_error']:.3e}"
            )
        except TypeError as exc:
            results["pme_morton_tile_pairs"][tname] = {"error": str(exc)}
            print(f"  [Tier 2 Tile] {tname:16s}: ERROR ({exc})")

    # --- Regime B: Reaction Field (`1UAO` periodic box with RF electrostatics) ---
    def _rf_total_f64(pos, q, sig, eps):
        elj = chunked_lj_energy_nl(
            pos, sig, eps, b64.excl_dense_indices, b64.excl_dense_scales_vdw, nbr_raw.idx, disp64, cutoff_t2, 128, switch_w_t2
        )
        erf = _rf_direct_coulomb_nl_energy(
            pos, q, b64.atom_mask, nbr_raw.idx, b64.excl_dense_indices, box64, cutoff_t2, 78.3
        )
        et = explicit_corrections.lj_dispersion_tail_energy(box64, sig, eps, cutoff_t2, b64.atom_mask)
        return elj + erf + et

    rf_args = [b64.positions, b64.charges, b64.sigmas, b64.epsilons]
    for idx_t, (tname, theta0, tmask, argnum) in enumerate(
        [
            ("positions", b64.positions, pos_mask_np, 0),
            ("charges", b64.charges, atom_mask_np, 1),
            ("sigmas", b64.sigmas, atom_mask_np, 2),
            ("epsilons", b64.epsilons, atom_mask_np & (np.asarray(b64.epsilons) > 1e-6), 3),
        ]
    ):
        def _rf_f(th, _an=argnum):
            args = list(rf_args)
            args[_an] = th
            return _rf_total_f64(*args)

        chk = _directional_fd_check(
            _rf_f, jax.jit(jax.grad(_rf_f)), theta0, tmask, h=1.0e-5, n_dirs=n_dirs, seed=2000 + idx_t
        )
        results["rf"][tname] = {"float64_custom_vjp": chk}
        print(f"  [Tier 2 RF]   {tname:16s}: f64_rel_err={chk['rel_l2_error']:.3e} (pass={chk['pass_lt_1e5']})")

    # --- Regime C: Implicit Solvent OBC2 GBSA + ACE + CMAP (`5DFR_GBSA`, first 512 real protein atoms for crisp f64 FD) ---
    dfr_gb = build_5dfr_gbsa_system(allow_cpu_smoke=allow_cpu_smoke)
    gb64 = dfr_gb.bundle_f64
    n_sub = 512
    pos_gb = gb64.positions[:n_sub]
    q_gb = gb64.charges[:n_sub]
    rad_gb = gb64.radii[:n_sub]
    srad_gb = gb64.scaled_radii[:n_sub]
    mask_gb = gb64.atom_mask[:n_sub]
    valid_pair_gb = (mask_gb[:, None] & mask_gb[None, :]).astype(jnp.float64) * (1.0 - jnp.eye(n_sub, dtype=jnp.float64))
    emask_gb = mask_gb[:, None] & mask_gb[None, :]
    omm_ace_st = 28.3919551 / (4.0 * math.pi * 418.4)

    def _gbsa_total_f64(pos, q, rad, srad):
        e_gb, br = generalized_born.compute_gb_energy(
            pos, q, rad, solvent_dielectric=78.5, solute_dielectric=1.0, dielectric_offset=0.09,
            mask=valid_pair_gb, energy_mask=emask_gb, scaled_radii=srad
        )
        e_ace = jnp.sum(
            jnp.where(mask_gb, generalized_born.compute_ace_nonpolar_energy(rad, br, surface_tension=omm_ace_st), 0.0)
        )
        return e_gb + e_ace

    gb_tile_list = build_gbsa_tile_pair_list(pos_gb.astype(jnp.float32), atom_mask=mask_gb)

    def _gbsa_total_pallas_f32(pos, q, rad, srad):
        return pallas_gbsa_tile(
            pos.astype(jnp.float32),
            q.astype(jnp.float32),
            rad.astype(jnp.float32),
            srad.astype(jnp.float32),
            tile_list=gb_tile_list,
            atom_mask=mask_gb,
            solvent_dielectric=78.5,
            solute_dielectric=1.0,
            dielectric_offset=0.09,
            surface_tension=omm_ace_st,
            include_gb=True,
            include_ace=True,
            include_coulomb=False,
            include_lj=False,
        )

    gb_args = [pos_gb, q_gb, rad_gb, srad_gb]
    gb_mask_np = np.asarray(mask_gb, dtype=bool)
    for idx_t, (tname, theta0, tmask, argnum) in enumerate(
        [
            ("positions", pos_gb, np.broadcast_to(gb_mask_np[:, None], pos_gb.shape), 0),
            ("charges", q_gb, gb_mask_np, 1),
            ("radii", rad_gb, gb_mask_np, 2),
            ("scaled_radii", srad_gb, gb_mask_np, 3),
        ]
    ):
        def _gb_f(th, _an=argnum):
            args = list(gb_args)
            args[_an] = th
            return _gbsa_total_f64(*args)

        chk = _directional_fd_check(
            _gb_f, jax.jit(jax.grad(_gb_f)), theta0, tmask, h=1.0e-5, n_dirs=n_dirs, seed=3000 + idx_t
        )
        grad_gb_pallas = jax.jit(
            jax.grad(lambda th, _an=argnum: _gbsa_total_pallas_f32(*[*gb_args[:_an], th, *gb_args[_an + 1 :]]))
        )
        chk_pallas_gb = _directional_fd_check(
            _gb_f, grad_gb_pallas, theta0, tmask, h=1.0e-5, n_dirs=n_dirs, seed=3000 + idx_t
        )
        results["gbsa"][tname] = {
            "float64_custom_vjp": chk,
            "float32_pallas_custom_vjp": chk_pallas_gb,
        }
        print(
            f"  [Tier 2 GBSA] {tname:16s}: f64_rel_err={chk['rel_l2_error']:.3e} (pass={chk['pass_lt_1e5']}), "
            f"pallas_f32_rel_err={chk_pallas_gb['rel_l2_error']:.3e}"
        )

    # Also test 2D bicubic CMAP spline gradient (`5DFR_GBSA` full protein coordinates)
    disp_gb, _ = displacement_fn_for_bundle(gb64)
    cmap_src64 = gb64.cmap_coeffs if (gb64.cmap_coeffs is not None and gb64.cmap_coeffs.size > 0) else gb64.cmap_energy_grids

    def _cmap_only_f64(pos):
        phi_idx = gb64.cmap_torsion_idx[:, 0:4]
        psi_idx = gb64.cmap_torsion_idx[:, 4:8] if gb64.cmap_torsion_idx.shape[-1] == 8 else gb64.cmap_torsion_idx[:, 1:5]
        phi = bonded.compute_dihedral_angles(pos, phi_idx, disp_gb)
        psi = bonded.compute_dihedral_angles(pos, psi_idx, disp_gb)
        e_per = cmap_mod.compute_cmap_energies(phi, psi, gb64.cmap_indices, cmap_src64)
        return jnp.sum(jnp.where(gb64.cmap_mask, e_per, 0.0))

    cmap_mask_atoms = np.zeros((gb64.positions.shape[0], 3), dtype=bool)
    active_cmap_tors = np.asarray(gb64.cmap_torsion_idx)[np.asarray(gb64.cmap_mask, dtype=bool)]
    cmap_mask_atoms[np.unique(active_cmap_tors)] = True
    chk_cmap = _directional_fd_check(
        _cmap_only_f64,
        jax.jit(jax.grad(_cmap_only_f64)),
        gb64.positions,
        cmap_mask_atoms,
        h=1.0e-5,
        n_dirs=n_dirs,
        seed=3099,
    )
    results["gbsa"]["cmap_positions"] = {"float64_custom_vjp": chk_cmap}
    print(f"  [Tier 2 CMAP] {'cmap_positions':16s}: f64_rel_err={chk_cmap['rel_l2_error']:.3e} (pass={chk_cmap['pass_lt_1e5']})")

    all_f64_checks = [
        v["float64_custom_vjp"]["pass_lt_1e5"]
        for reg in ("pme", "rf", "gbsa")
        for v in results[reg].values()
    ]
    results["all_float64_vjp_pass_lt_1e5"] = bool(all(all_f64_checks))
    checkpoint_cb(results)
    return results


# =============================================================================
# Shared Stateful Pallas v3.5 Integrator & OpenMM CUDA Worker Helpers
# =============================================================================


def _make_prolix_pallas_stepper(
    spec: ValidatedSystemSpec,
    *,
    dt_fs: float,
    temperature_K: float,
    gamma_ps: float,
    nl_cutoff_pad_factor: float = 1.0,
) -> dict[str, Any]:
    """Construct a stateful Pallas v3.5 integrator + neighbor-list updater for `spec`."""
    bun = spec.bundle_f32
    n_pad = int(bun.positions.shape[0])
    n_real = spec.n_real_atoms
    pme_gp = int(spec.pme_grid[0]) if spec.pme_grid is not None else 64
    disp_fn, shift_fn = displacement_fn_for_bundle(bun)
    box0_vec = jnp.diag(bun.box).astype(jnp.float32)

    cutoff_nl = min(float(spec.cutoff_A) * float(nl_cutoff_pad_factor), 0.49 * float(jnp.min(box0_vec)))
    nl_fn = cast(
        "Any",
        nl_mod.make_neighbor_list_fn(
            disp_fn,
            box0_vec,
            cutoff_nl,
            capacity_multiplier=1.50,
            dr_threshold=0.5,
            excl_indices=bun.excl_dense_indices,
            excl_scales_vdw=bun.excl_dense_scales_vdw,
            excl_scales_elec=bun.excl_dense_scales_elec,
            atom_mask=bun.atom_mask,
        ),
    )
    raw_alloc = nl_fn.allocate(bun.positions)
    n_valid_init = int(
        jnp.sum((raw_alloc.idx[:n_real] >= 0) & ((raw_alloc.idx[:n_real] & ((1 << 29) - 1)) < n_pad))
    )
    if n_real > 1 and n_valid_init == 0:
        nl_fn = cast(
            "Any",
            nl_mod.make_neighbor_list_fn(
                disp_fn,
                box0_vec,
                cutoff_nl,
                capacity_multiplier=1.50,
                dr_threshold=0.5,
                excl_indices=bun.excl_dense_indices,
                excl_scales_vdw=bun.excl_dense_scales_vdw,
                excl_scales_elec=bun.excl_dense_scales_elec,
                atom_mask=bun.atom_mask,
                use_pallas_nl=False,
            ),
        )
        raw_alloc = nl_fn.allocate(bun.positions)
    nbr0 = _encode_bundle_nl(raw_alloc, bun)

    force_fn = energy_fn_from_bundle(
        bun,
        pme_grid_points=pme_gp,
        build_dense_exclusions=False,
        use_dispersion_correction=True,
        use_pallas=True,
        use_pallas_pme=True,
        excl_mode="pre_encoded",
        return_forces=True,
    )
    energy_fn = energy_fn_from_bundle(
        bun,
        pme_grid_points=pme_gp,
        build_dense_exclusions=False,
        use_dispersion_correction=True,
        use_pallas=True,
        use_pallas_pme=True,
        excl_mode="pre_encoded",
        return_forces=False,
    )

    def _bound_force(pos: jnp.ndarray, **kw: Any) -> jnp.ndarray:
        kw.setdefault("neighbor", nbr0)
        return force_fn(pos, **kw)

    dt_s = jnp.asarray(dt_fs / AKMA_TIME_UNIT_FS, dtype=jnp.float32)
    kT_s = jnp.asarray(temperature_K * BOLTZMANN_KCAL, dtype=jnp.float32)
    gamma_akma = kups_adapter.gamma_ps_to_akma(float(gamma_ps))
    masses_pad = masses_with_prefix(bun, n_pad)
    water_idx = bun.water_indices if bun.shape_spec.has_real_water else None
    water_msk = bun.water_mask if bun.shape_spec.has_real_water else None
    has_sol_cons = (
        spec.n_constraints_solute > 0
        and bun.constraint_pairs is not None
        and int(bun.constraint_pairs.shape[0]) > 0
    )
    sol_c_pairs = bun.constraint_pairs if has_sol_cons else None
    sol_c_lengths = bun.constraint_lengths if has_sol_cons else None
    sol_c_mask = bun.constraint_mask if has_sol_cons else None
    sol_constraints = (
        (sol_c_pairs, sol_c_lengths, sol_c_mask) if has_sol_cons else None
    )

    init_fn, apply_fn = settle_langevin(
        _bound_force,
        shift_fn,
        dt=1.0,
        kT=1.0,
        gamma=gamma_akma,
        mass=masses_pad,
        water_indices=water_idx,
        constraints=cast("Any", sol_constraints),
        remove_linear_com_momentum=True,
        project_ou_momentum_rigid=True,
        water_mask=water_msk,
        atom_mask=bun.atom_mask[:n_pad],
        box=pbc_box_for_bundle(bun),
        is_force_shaped=True,
        use_pallas=True,
    )

    pos0 = positions_with_prefix(bun, n_pad)
    f0 = force_fn(pos0, neighbor=nbr0)

    def _init_seed(seed_int: int, init_temp_K: float | None = None):
        kt_init = jnp.asarray(
            (temperature_K if init_temp_K is None else init_temp_K) * BOLTZMANN_KCAL,
            dtype=jnp.float32,
        )
        st = init_fn(
            jax.random.PRNGKey(jnp.asarray(seed_int, dtype=jnp.uint32)),
            pos0,
            kT=kt_init,
            neighbor=nbr0,
            precomputed_force=f0,
        )
        # Zero initial net linear COM momentum on real atoms (matching OpenMM CMMotionRemover)
        m_real = masses_pad[:n_real, None]
        p_real = st.momentum[:n_real]
        v_com = jnp.sum(p_real, axis=0, keepdims=True) / jnp.sum(m_real)
        p_zeroed = st.momentum.at[:n_real].set(p_real - m_real * v_com)
        p_zeroed = jnp.where(bun.atom_mask[:n_pad, None], p_zeroed, 0.0)
        return dataclasses.replace(st, momentum=p_zeroed)

    def _update_nl(pos: jnp.ndarray, nbr_prev: Any, box_cur: jnp.ndarray | None = None) -> Any:
        if box_cur is not None:
            pos_mapped = jnp.mod(pos, box_cur[None, :]) * (box0_vec[None, :] / box_cur[None, :])
        else:
            pos_mapped = pos
        return _encode_bundle_nl(nl_fn.update(pos_mapped, nbr_prev), bun)

    return {
        "bundle": bun,
        "n_pad": n_pad,
        "n_real": n_real,
        "box0_vec": box0_vec,
        "masses_pad": masses_pad,
        "water_idx": water_idx,
        "water_msk": water_msk,
        "c_pairs": sol_c_pairs,
        "c_lengths": sol_c_lengths,
        "c_mask": sol_c_mask,
        "nbr0": nbr0,
        "dt_s": dt_s,
        "kT_s": kT_s,
        "gamma_akma": gamma_akma,
        "force_fn": force_fn,
        "energy_fn": energy_fn,
        "apply_fn": apply_fn,
        "init_seed": _init_seed,
        "update_nl": _update_nl,
    }


def _state_pos(st: Any) -> jnp.ndarray:
    return getattr(st, "positions", st.position)


# =============================================================================
# TIER 3: NVE Conservation, NVT Equipartition, and NPT Density & Compressibility
# =============================================================================


def _run_tier3a_nve(
    spec: ValidatedSystemSpec,
    *,
    n_steps: int,
    sample_every: int,
    allow_cpu_smoke: bool,
) -> dict[str, Any]:
    """Compare NVE total-energy conservation (`dt = 1.0 fs`, `gamma = 0`) across Prolix & OpenMM CUDA."""
    import openmm as mm
    from openmm import unit as _omm_unit
    unit = cast("Any", _omm_unit)

    dt_fs = 1.0
    n_real = spec.n_real_atoms
    n_dof = 3 * n_real - spec.n_constraints_total - 3

    # 1. Prolix Pallas v3.5 NVE (`settle_langevin` with `gamma = 0.0` = Velocity Verlet + analytical SETTLE)
    sp = _make_prolix_pallas_stepper(spec, dt_fs=dt_fs, temperature_K=300.0, gamma_ps=0.0)
    st = sp["init_seed"](42, init_temp_K=300.0)
    nbr = sp["nbr0"]
    apply_fn = sp["apply_fn"]
    update_nl = sp["update_nl"]
    energy_fn = jax.jit(sp["energy_fn"])
    dt_s = sp["dt_s"]
    kT_zero = jnp.asarray(0.0, dtype=jnp.float32)
    masses_real = np.asarray(spec.bundle_f64.masses[:n_real], dtype=np.float64)

    sub_nl = min(25, sample_every)
    n_sub = max(1, sample_every // sub_nl)

    @jax.jit
    def _advance_chunk(state_in, nbr_in):
        def _outer(_, carry):
            s_c, nb_c = carry

            def _inner(__, s_i):
                return apply_fn(s_i, neighbor=nb_c, kT=kT_zero, dt=dt_s)

            s_next = jax.lax.fori_loop(0, sub_nl, _inner, s_c)
            nb_next = update_nl(_state_pos(s_next), nb_c)
            return s_next, nb_next

        return jax.lax.fori_loop(0, n_sub, _outer, (state_in, nbr_in))

    # Warmup 200 steps so fast bond/water modes equilibrate before measuring linear NVE drift
    warmup_chunks = max(2, 200 // sample_every)
    for _ in range(warmup_chunks):
        st, nbr = _advance_chunk(st, nbr)

    n_samples = max(4, n_steps // sample_every)
    prolix_etot: list[float] = []
    prolix_pe: list[float] = []
    times_ns: list[float] = []
    for k in range(n_samples + 1):
        pos_c = _state_pos(st)
        pe_val = float(energy_fn(pos_c, neighbor=nbr))
        mom_real = np.asarray(st.momentum[:n_real], dtype=np.float64)
        ke_val = float(0.5 * np.sum((mom_real * mom_real) / masses_real[:, None]))
        prolix_pe.append(pe_val)
        prolix_etot.append(pe_val + ke_val)
        times_ns.append(k * sample_every * dt_fs * 1.0e-6)
        if k < n_samples:
            st, nbr = _advance_chunk(st, nbr)

    def _drift_stats(t_ns_arr: np.ndarray, etot_arr: np.ndarray, pe_arr: np.ndarray) -> dict[str, float]:
        per_dof = (etot_arr - etot_arr[0]) / float(n_dof)
        slope = float(np.polyfit(t_ns_arr, per_dof, 1)[0])
        max_abs = float(np.max(np.abs(per_dof)))
        ratio = float(np.std(etot_arr) / max(float(np.std(pe_arr)), 1e-12))
        return {
            "drift_kcal_mol_ns_dof": slope,
            "max_abs_dev_kcal_mol_dof": max_abs,
            "etot_to_pe_std_ratio": ratio,
        }

    t_arr = np.asarray(times_ns, dtype=np.float64)
    prolix_stats = _drift_stats(t_arr, np.asarray(prolix_etot), np.asarray(prolix_pe))

    # 2. OpenMM CUDA (`mixed` & `double`) `VerletIntegrator`
    omm_results: dict[str, Any] = {}
    for prec in ("mixed", "double"):
        sys_nve = clone_openmm_system(spec.omm_system)
        integ = mm.VerletIntegrator(dt_fs * 0.001 * unit.picoseconds)
        ctx, prov = _open_openmm_context(
            sys_nve,
            integ,
            precision=prec,
            allow_cpu_smoke=allow_cpu_smoke,
            caller=f"tier3a_nve_{prec}_{spec.name}",
        )
        ctx.setPositions((spec.positions_A * 0.1) * unit.nanometer)
        ctx.applyConstraints(1.0e-6)
        ctx.setVelocitiesToTemperature(300.0 * unit.kelvin, 42)
        ctx.applyVelocityConstraints(1.0e-6)
        integ.step(warmup_chunks * sample_every)

        e_list: list[float] = []
        u_list: list[float] = []
        for k in range(n_samples + 1):
            st_o = ctx.getState(getEnergy=True)
            u_k = float(st_o.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))
            k_k = float(st_o.getKineticEnergy().value_in_unit(unit.kilocalories_per_mole))
            u_list.append(u_k)
            e_list.append(u_k + k_k)
            if k < n_samples:
                integ.step(sample_every)
        del ctx, integ
        omm_results[f"openmm_cuda_{prec}"] = {
            **_drift_stats(t_arr, np.asarray(e_list), np.asarray(u_list)),
            "provenance": prov,
        }

    return {
        "n_real_atoms": n_real,
        "n_dof": n_dof,
        "dt_fs": dt_fs,
        "n_steps": n_samples * sample_every,
        "prolix_pallas_v35": prolix_stats,
        **omm_results,
    }


def _run_tier3b_nvt(
    spec: ValidatedSystemSpec,
    *,
    eq_steps: int,
    prod_steps: int,
    sample_every: int,
    allow_cpu_smoke: bool,
) -> dict[str, Any]:
    """Compare NVT equipartition (`<T_total>`, `<T_solute>`, `<T_solvent>`, MB std ratio) at 300 K."""
    import openmm as mm
    from openmm import unit as _omm_unit
    unit = cast("Any", _omm_unit)

    dt_fs = 1.0
    temp_target = 300.0
    gamma_ps = 1.0
    n_real = spec.n_real_atoms
    sol_idx = spec.solute_indices
    wat_idx = spec.solvent_indices
    n_sol = int(sol_idx.size)
    n_wat = int(wat_idx.size)
    n_cons_sol = spec.n_constraints_solute
    n_cons_wat = spec.n_constraints_total - n_cons_sol

    # Exact ghost-free degrees of freedom (partitioning 3 COM DOF proportionally across solute/solvent)
    ndof_tot = float(3 * n_real - spec.n_constraints_total - 3)
    ndof_sol = float(3 * n_sol - n_cons_sol) - 3.0 * (float(n_sol) / float(max(n_real, 1)))
    ndof_wat = float(3 * n_wat - n_cons_wat) - 3.0 * (float(n_wat) / float(max(n_real, 1)))

    masses_real = np.asarray(spec.bundle_f64.masses[:n_real], dtype=np.float64)

    def _summarize_temps(
        t_tot_l: list[float], t_sol_l: list[float], t_wat_l: list[float]
    ) -> dict[str, float | None]:
        tt = np.asarray(t_tot_l, dtype=np.float64)
        mb_expected_std = temp_target * math.sqrt(2.0 / ndof_tot)
        t_sol_mean = float(np.mean(np.asarray(t_sol_l, dtype=np.float64))) if t_sol_l else None
        t_wat_mean = float(np.mean(np.asarray(t_wat_l, dtype=np.float64))) if t_wat_l else None
        sol_minus_wat = (
            float(t_sol_mean - t_wat_mean)
            if (t_sol_mean is not None and t_wat_mean is not None)
            else None
        )
        return {
            "T_total_mean_K": float(np.mean(tt)),
            "T_total_std_K": float(np.std(tt, ddof=1)),
            "T_solute_mean_K": t_sol_mean,
            "T_solvent_mean_K": t_wat_mean,
            "solute_minus_solvent_K": sol_minus_wat,
            "mb_expected_std_K": mb_expected_std,
            "mb_std_ratio": float(np.std(tt, ddof=1) / mb_expected_std),
        }

    # 1. Prolix Pallas v3.5 BAOAB (`settle_langevin`)
    sp = _make_prolix_pallas_stepper(spec, dt_fs=dt_fs, temperature_K=temp_target, gamma_ps=gamma_ps)
    st = sp["init_seed"](101, init_temp_K=temp_target)
    nbr = sp["nbr0"]
    apply_fn = sp["apply_fn"]
    update_nl = sp["update_nl"]
    dt_s = sp["dt_s"]
    kT_s = sp["kT_s"]

    @jax.jit
    def _step_sample(state_in, nbr_in):
        def _inner(_, s_i):
            return apply_fn(s_i, neighbor=nbr_in, kT=kT_s, dt=dt_s)

        s_out = jax.lax.fori_loop(0, sample_every, _inner, state_in)
        nb_out = update_nl(_state_pos(s_out), nbr_in)
        return s_out, nb_out

    for _ in range(max(1, eq_steps // sample_every)):
        st, nbr = _step_sample(st, nbr)
        # Verify ghost momenta remain identically zero
    ghost_mom_norm = float(jnp.linalg.norm(st.momentum[n_real:]))

    p_tt, p_ts, p_tw = [], [], []
    n_prod_samples = max(10, prod_steps // sample_every)
    for _ in range(n_prod_samples):
        st, nbr = _step_sample(st, nbr)
        mom = np.asarray(st.momentum[:n_real], dtype=np.float64)
        ke_per_atom = 0.5 * np.sum((mom * mom) / masses_real[:, None], axis=-1)
        ke_tot = float(np.sum(ke_per_atom))
        ke_sol = float(np.sum(ke_per_atom[sol_idx])) if n_sol > 0 else 0.0
        ke_wat = float(np.sum(ke_per_atom[wat_idx])) if n_wat > 0 else 0.0
        p_tt.append(2.0 * ke_tot / (ndof_tot * BOLTZMANN_KCAL))
        if ndof_sol > 0.0:
            p_ts.append(2.0 * ke_sol / (ndof_sol * BOLTZMANN_KCAL))
        if ndof_wat > 0.0:
            p_tw.append(2.0 * ke_wat / (ndof_wat * BOLTZMANN_KCAL))

    # 2. OpenMM CUDA `mixed` `LangevinMiddleIntegrator` (`BAOAB`)
    sys_nvt = clone_openmm_system(spec.omm_system)
    integ = mm.LangevinMiddleIntegrator(
        temp_target * unit.kelvin,
        gamma_ps / unit.picosecond,
        dt_fs * 0.001 * unit.picoseconds,
    )
    integ.setRandomNumberSeed(101)
    ctx, prov = _open_openmm_context(
        sys_nvt,
        integ,
        precision="mixed",
        allow_cpu_smoke=allow_cpu_smoke,
        caller=f"tier3b_nvt_mixed_{spec.name}",
    )
    ctx.setPositions((spec.positions_A * 0.1) * unit.nanometer)
    ctx.applyConstraints(1.0e-6)
    ctx.setVelocitiesToTemperature(temp_target * unit.kelvin, 101)
    ctx.applyVelocityConstraints(1.0e-6)
    integ.step(eq_steps)

    o_tt, o_ts, o_tw = [], [], []
    # 1 amu * (nm/ps)^2 = 1 kJ/mol = 1/4.184 kcal/mol
    vel_conv = 1.0 / 4.184
    for _ in range(n_prod_samples):
        integ.step(sample_every)
        st_o = ctx.getState(getVelocities=True)
        vel_nm_ps = np.asarray(
            st_o.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond),
            dtype=np.float64,
        )
        ke_per_atom_o = 0.5 * masses_real * np.sum(vel_nm_ps * vel_nm_ps, axis=-1) * vel_conv
        o_tt.append(2.0 * float(np.sum(ke_per_atom_o)) / (ndof_tot * BOLTZMANN_KCAL))
        if ndof_sol > 0.0:
            o_ts.append(2.0 * float(np.sum(ke_per_atom_o[sol_idx])) / (ndof_sol * BOLTZMANN_KCAL))
        if ndof_wat > 0.0:
            o_tw.append(2.0 * float(np.sum(ke_per_atom_o[wat_idx])) / (ndof_wat * BOLTZMANN_KCAL))
    del ctx, integ

    return {
        "n_real_atoms": n_real,
        "n_padded_atoms": sp["n_pad"],
        "ndof_total": round(ndof_tot),
        "ndof_solute": round(ndof_sol, 2),
        "ndof_solvent": round(ndof_wat, 2),
        "ghost_momentum_l2_norm": ghost_mom_norm,
        "dt_fs": dt_fs,
        "gamma_ps": gamma_ps,
        "prod_steps": n_prod_samples * sample_every,
        "prolix_pallas_v35": _summarize_temps(p_tt, p_ts, p_tw),
        "openmm_cuda_mixed": {**_summarize_temps(o_tt, o_ts, o_tw), "provenance": prov},
    }


def _build_intramolecular_unwrap_tree(spec: ValidatedSystemSpec) -> tuple[np.ndarray, np.ndarray]:
    """Precompute BFS spanning tree edges `(parent, child)` so molecules are unwrapped whole before MC volume scaling."""
    import openmm as mm

    n = spec.n_real_atoms
    adj: list[list[int]] = [[] for _ in range(n)]
    for c_i in range(spec.omm_system.getNumConstraints()):
        p1, p2, _ = spec.omm_system.getConstraintParameters(c_i)
        if 0 <= p1 < n and 0 <= p2 < n:
            adj[int(p1)].append(int(p2))
            adj[int(p2)].append(int(p1))
    for f_i in range(spec.omm_system.getNumForces()):
        f = spec.omm_system.getForce(f_i)
        if isinstance(f, mm.HarmonicBondForce):
            for b_i in range(f.getNumBonds()):
                p1, p2, _, _ = f.getBondParameters(b_i)
                if 0 <= p1 < n and 0 <= p2 < n:
                    adj[int(p1)].append(int(p2))
                    adj[int(p2)].append(int(p1))

    visited = np.zeros((n,), dtype=bool)
    parents: list[int] = []
    children: list[int] = []
    for root in range(n):
        if visited[root]:
            continue
        visited[root] = True
        queue = [root]
        q_head = 0
        while q_head < len(queue):
            u = queue[q_head]
            q_head += 1
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    parents.append(u)
                    children.append(v)
                    queue.append(v)
    return np.asarray(parents, dtype=np.int32), np.asarray(children, dtype=np.int32)


def _scale_positions_molecular_com_host(
    pos_pad: np.ndarray,
    box_old: np.ndarray,
    scale_factor: float,
    mol_ids: np.ndarray,
    n_mols: int,
    masses_real: np.ndarray,
    bfs_parents: np.ndarray,
    bfs_children: np.ndarray,
) -> np.ndarray:
    """OpenMM-exact isotropic Molecular-COM coordinate scaling preserving all bond & water geometries."""
    n_real = masses_real.shape[0]
    pos_u = pos_pad[:n_real].astype(np.float64, copy=True)
    box_f64 = box_old.astype(np.float64)
    # 1. Make every molecule whole across periodic faces via BFS order
    for p_i, c_i in zip(bfs_parents, bfs_children, strict=False):
        dr = pos_u[c_i] - pos_u[p_i]
        pos_u[c_i] = pos_u[p_i] + (dr - box_f64 * np.round(dr / box_f64))

    # 2. Compute mass-weighted molecular COMs and scale COMs by `scale_factor`
    mol_mass = np.bincount(mol_ids, weights=masses_real, minlength=n_mols)
    com = np.zeros((n_mols, 3), dtype=np.float64)
    for d in range(3):
        com[:, d] = np.bincount(mol_ids, weights=masses_real * pos_u[:, d], minlength=n_mols) / np.maximum(
            mol_mass, 1e-12
        )
    com_wrapped = np.mod(com, box_f64[None, :])
    delta_mol = com_wrapped * (scale_factor - 1.0) + (com_wrapped - com)
    pos_new_real = pos_u + delta_mol[mol_ids]
    box_new = box_f64 * scale_factor
    pos_out = pos_pad.copy()
    pos_out[:n_real] = pos_new_real.astype(pos_pad.dtype)
    if pos_out.shape[0] > n_real:
        pos_out[n_real:] = (pos_out[n_real:] * scale_factor).astype(pos_pad.dtype)
    return pos_out


def _run_tier3c_npt(
    spec: ValidatedSystemSpec,
    *,
    eq_steps: int,
    prod_steps: int,
    barostat_interval: int,
    allow_cpu_smoke: bool,
) -> dict[str, Any]:
    """Compare NPT equilibrium density `<rho>` (g/cm^3) and compressibility `kappa_T` at 300 K, 1.0 bar."""
    import openmm as mm
    from openmm import unit as _omm_unit
    unit = cast("Any", _omm_unit)

    from prolix.physics.settle import pallas_baoab_post_force, pallas_baoab_pre_force

    dt_fs = 1.0
    temp_K = 300.0
    pressure_bar = 1.0
    gamma_ps = 1.0
    n_real = spec.n_real_atoms
    masses_real = np.asarray(spec.bundle_f64.masses[:n_real], dtype=np.float64)
    total_mass_amu = float(np.sum(masses_real))
    kT_kcal = temp_K * BOLTZMANN_KCAL
    p_kcal_per_A3 = pressure_bar / KCAL_MOL_TO_BAR_A3

    def _summarize_volumes(vols_A3: list[float], acc_rate: float | None = None) -> dict[str, Any]:
        v_arr = np.asarray(vols_A3, dtype=np.float64)
        rho_arr = (total_mass_amu / v_arr) * AMU_PER_A3_TO_G_CM3
        v_mean = float(np.mean(v_arr))
        v_var = float(np.var(v_arr, ddof=1)) if v_arr.size > 1 else 0.0
        # Block-averaged SE across 5 blocks
        n_blk = min(5, max(1, v_arr.size // 4))
        blk_means = [float(np.mean(b)) for b in np.array_split(rho_arr, n_blk) if b.size > 0]
        rho_se = float(np.std(blk_means, ddof=1) / math.sqrt(len(blk_means))) if len(blk_means) > 1 else 0.0
        # kappa_T = Var(V) / (k_B * T * <V>) in bar^-1; report in 10^-5 bar^-1 (liquid water ~ 4.5-5.3)
        kT_bar_A3 = kT_kcal * KCAL_MOL_TO_BAR_A3
        kappa_1e5_bar = (v_var / max(kT_bar_A3 * v_mean, 1e-12)) * 1.0e5
        res: dict[str, Any] = {
            "density_mean_g_cm3": float(np.mean(rho_arr)),
            "density_se_g_cm3": rho_se,
            "volume_mean_A3": v_mean,
            "volume_std_A3": float(math.sqrt(v_var)),
            "compressibility_kappa_T_1e5_bar_inv": float(kappa_1e5_bar),
        }
        if acc_rate is not None:
            res["mc_acceptance_rate"] = round(acc_rate, 4)
        return res

    # 1. Prolix Pallas v3.5 NPT (25-step Pallas BAOAB + OpenMM adaptive Molecular-COM Metropolis Barostat)
    sp = _make_prolix_pallas_stepper(
        spec, dt_fs=dt_fs, temperature_K=temp_K, gamma_ps=gamma_ps, nl_cutoff_pad_factor=1.08
    )
    bun = sp["bundle"]
    n_pad = sp["n_pad"]
    st = sp["init_seed"](202, init_temp_K=temp_K)
    nbr = sp["nbr0"]
    box_cur = np.asarray(sp["box0_vec"], dtype=np.float32)
    force_fn = sp["force_fn"]
    energy_fn = jax.jit(sp["energy_fn"])
    update_nl = jax.jit(sp["update_nl"])
    dt_s = sp["dt_s"]
    kT_s = sp["kT_s"]
    gamma_akma = sp["gamma_akma"]
    water_idx = sp["water_idx"]
    water_msk = sp["water_msk"]
    c_pairs = sp["c_pairs"]
    c_lengths = sp["c_lengths"]
    c_mask = sp["c_mask"]
    atom_msk = bun.atom_mask[:n_pad]
    masses_pad = sp["masses_pad"]

    @jax.jit
    def _step_npt_md_chunk(pos_in, mom_in, frc_in, key_in, nbr_in, box_vec_in):
        def _body(_, carry):
            p_c, m_c, f_c, k_c = carry
            p_n, m_n, k_n = pallas_baoab_pre_force(
                p_c,
                m_c,
                f_c,
                masses_pad,
                k_c,
                water_idx,
                water_msk,
                box_vec_in,
                dt_s,
                kT_s,
                gamma_akma,
                project_ou_momentum_rigid=True,
                atom_mask=atom_msk,
                constraint_pairs=c_pairs,
                constraint_lengths=c_lengths,
                constraint_mask=c_mask,
            )
            f_n = force_fn(p_n, neighbor=nbr_in, box=box_vec_in)
            m_n = pallas_baoab_post_force(
                p_n,
                m_n,
                f_n,
                masses_pad,
                water_idx,
                water_msk,
                dt_s,
                atom_mask=atom_msk,
                constraint_pairs=c_pairs,
                constraint_lengths=c_lengths,
                constraint_mask=c_mask,
            )
            return p_n, m_n, f_n, k_n

        p_o, m_o, f_o, k_o = jax.lax.fori_loop(
            0, barostat_interval, _body, (pos_in, mom_in, frc_in, key_in)
        )
        nb_o = update_nl(p_o, nbr_in, box_vec_in)
        return p_o, m_o, f_o, k_o, nb_o

    eval_force_jit = jax.jit(lambda p, nb, bx: force_fn(p, neighbor=nb, box=bx))
    bfs_parents, bfs_children = _build_intramolecular_unwrap_tree(spec)

    pos_c = _state_pos(st)
    mom_c = st.momentum
    frc_c = st.force
    key_c = st.key
    box_j = jnp.asarray(box_cur, dtype=jnp.float32)

    rng_mc = np.random.default_rng(202603)
    v0_init = float(np.prod(box_cur.astype(np.float64)))
    vol_scale = 0.01 * v0_init
    n_acceptWindow = 0
    n_attemptWindow = 0
    prod_accept = 0
    prod_attempt = 0
    prolix_vols: list[float] = []

    total_cycles = max(2, (eq_steps + prod_steps) // barostat_interval)
    eq_cycles = max(1, eq_steps // barostat_interval)

    for cyc in range(total_cycles):
        pos_c, mom_c, frc_c, key_c, nbr = _step_npt_md_chunk(pos_c, mom_c, frc_c, key_c, nbr, box_j)

        # OpenMM Molecular-COM Metropolis volume trial
        v_old = float(np.prod(np.asarray(box_j, dtype=np.float64)))
        dv = float(rng_mc.uniform(-1.0, 1.0)) * vol_scale
        v_new = v_old + dv
        if v_new > 0.86 * v0_init and v_new < 1.18 * v0_init:
            length_scale = (v_new / v_old) ** (1.0 / 3.0)
            box_old_np = np.asarray(box_j, dtype=np.float32)
            box_new_j = jnp.asarray(box_old_np * length_scale, dtype=jnp.float32)
            pos_old_np = np.asarray(pos_c)
            pos_trial_np = _scale_positions_molecular_com_host(
                pos_old_np,
                box_old_np,
                length_scale,
                spec.molecule_ids,
                spec.n_molecules,
                masses_real,
                bfs_parents,
                bfs_children,
            )
            pos_trial_j = jnp.asarray(pos_trial_np, dtype=jnp.float32)
            nbr_trial = update_nl(pos_trial_j, nbr, box_new_j)
            e_old = float(energy_fn(pos_c, neighbor=nbr, box=box_j))
            e_new = float(energy_fn(pos_trial_j, neighbor=nbr_trial, box=box_new_j))
            # Exact OpenMM MCBarostat criterion: w = dE + P*dV - N_molecules * kT * ln(V_new / V_old)
            delta_w = (e_new - e_old) + p_kcal_per_A3 * dv - float(spec.n_molecules) * kT_kcal * math.log(
                v_new / v_old
            )
            accepted = delta_w <= 0.0 or float(rng_mc.uniform(0.0, 1.0)) < math.exp(-min(delta_w / kT_kcal, 80.0))
        else:
            accepted = False

        n_attemptWindow += 1
        if cyc >= eq_cycles:
            prod_attempt += 1
        if accepted:
            pos_c = pos_trial_j
            box_j = box_new_j
            nbr = nbr_trial
            frc_c = eval_force_jit(pos_c, nbr, box_j)
            n_acceptWindow += 1
            if cyc >= eq_cycles:
                prod_accept += 1

        if n_attemptWindow >= 10:
            if n_acceptWindow < 0.25 * n_attemptWindow:
                vol_scale *= 0.5
            elif n_acceptWindow > 0.75 * n_attemptWindow:
                vol_scale = min(vol_scale * 1.5, 0.03 * v0_init)
            n_acceptWindow = 0
            n_attemptWindow = 0

        if cyc >= eq_cycles:
            prolix_vols.append(float(np.prod(np.asarray(box_j, dtype=np.float64))))

    # 2. OpenMM CUDA `mixed` (`LangevinMiddleIntegrator` + `MonteCarloBarostat`)
    sys_npt = clone_openmm_system(spec.omm_system)
    baro = mm.MonteCarloBarostat(pressure_bar * unit.bar, temp_K * unit.kelvin, barostat_interval)
    baro.setRandomNumberSeed(202603)
    sys_npt.addForce(baro)
    integ = mm.LangevinMiddleIntegrator(
        temp_K * unit.kelvin,
        gamma_ps / unit.picosecond,
        dt_fs * 0.001 * unit.picoseconds,
    )
    integ.setRandomNumberSeed(202)
    ctx, prov = _open_openmm_context(
        sys_npt,
        integ,
        precision="mixed",
        allow_cpu_smoke=allow_cpu_smoke,
        caller=f"tier3c_npt_mixed_{spec.name}",
    )
    ctx.setPositions((spec.positions_A * 0.1) * unit.nanometer)
    ctx.applyConstraints(1.0e-6)
    ctx.setVelocitiesToTemperature(temp_K * unit.kelvin, 202)
    ctx.applyVelocityConstraints(1.0e-6)
    integ.step(eq_cycles * barostat_interval)

    omm_vols: list[float] = []
    prod_cycles = total_cycles - eq_cycles
    for _ in range(prod_cycles):
        integ.step(barostat_interval)
        bv = ctx.getState().getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)
        omm_vols.append(float( abs(np.linalg.det(np.asarray(bv, dtype=np.float64)))))
    del ctx, integ

    p_summary = _summarize_volumes(prolix_vols, acc_rate=prod_accept / max(prod_attempt, 1))
    o_summary = _summarize_volumes(omm_vols)
    return {
        "n_real_atoms": n_real,
        "n_molecules": spec.n_molecules,
        "total_mass_amu": round(total_mass_amu, 2),
        "dt_fs": dt_fs,
        "barostat_interval": barostat_interval,
        "eq_steps": eq_cycles * barostat_interval,
        "prod_steps": prod_cycles * barostat_interval,
        "prolix_pallas_v35": p_summary,
        "openmm_cuda_mixed": {**o_summary, "provenance": prov},
        "density_abs_diff_g_cm3": abs(p_summary["density_mean_g_cm3"] - o_summary["density_mean_g_cm3"]),
    }


def run_tier3_conservation_and_thermodynamics(
    *,
    quick: bool,
    allow_cpu_smoke: bool,
    existing_tier3: dict[str, Any] | None,
    checkpoint_cb: Any,
) -> dict[str, Any]:
    """Execute Tier 3A (NVE), Tier 3B (NVT equipartition), and Tier 3C (NPT density & compressibility)."""
    print("\n=== [Tier 3/4] Conservation & Ensemble Thermodynamics (NVE / NVT / NPT) ===")
    out: dict[str, Any] = dict(existing_tier3 or {})
    out.setdefault("nve", {})
    out.setdefault("nvt", {})
    out.setdefault("npt", {})

    tip3p = build_tip3p_system(allow_cpu_smoke=allow_cpu_smoke, minimize_iters=30 if quick else 80)
    uao = build_1uao_system(allow_cpu_smoke=allow_cpu_smoke, minimize_iters=30 if quick else 80)
    ubq = build_1ubq_system(allow_cpu_smoke=allow_cpu_smoke, minimize_iters=30 if quick else 80)
    dhfr = build_dhfr_system(electrostatics="pme", allow_cpu_smoke=allow_cpu_smoke, minimize_iters=30 if quick else 80)

    # 3A: NVE Conservation (`10,000` steps in `--full`, `400` in `--quick`)
    for s_spec in (tip3p, uao):
        if s_spec.name not in out["nve"]:
            t0 = time.perf_counter()
            res = _run_tier3a_nve(
                s_spec,
                n_steps=400 if quick else 10000,
                sample_every=20 if quick else 50,
                allow_cpu_smoke=allow_cpu_smoke,
            )
            res["elapsed_s"] = round(time.perf_counter() - t0, 2)
            out["nve"][s_spec.name] = res
            checkpoint_cb(out)
            print(
                f"  [Tier 3A NVE] {s_spec.name:16s}: "
                f"Prolix drift={res['prolix_pallas_v35']['drift_kcal_mol_ns_dof']:+.3e}, "
                f"OMM_mix={res['openmm_cuda_mixed']['drift_kcal_mol_ns_dof']:+.3e}, "
                f"OMM_dbl={res['openmm_cuda_double']['drift_kcal_mol_ns_dof']:+.3e} kcal/(mol*ns*dof) ({res['elapsed_s']:.1f}s)"
            )

    # 3B: NVT Equipartition (`15,000` prod steps in `--full`, `600` in `--quick`)
    for s_spec in (ubq, dhfr):
        if s_spec.name not in out["nvt"]:
            t0 = time.perf_counter()
            res = _run_tier3b_nvt(
                s_spec,
                eq_steps=200 if quick else 2000,
                prod_steps=600 if quick else 15000,
                sample_every=25,
                allow_cpu_smoke=allow_cpu_smoke,
            )
            res["elapsed_s"] = round(time.perf_counter() - t0, 2)
            out["nvt"][s_spec.name] = res
            checkpoint_cb(out)
            pv = res["prolix_pallas_v35"]
            ov = res["openmm_cuda_mixed"]
            print(
                f"  [Tier 3B NVT] {s_spec.name:16s}: "
                f"Prolix <T>={pv['T_total_mean_K']:.2f}K (sol={pv['T_solute_mean_K']:.2f}, wat={pv['T_solvent_mean_K']:.2f}, MB={pv['mb_std_ratio']:.2f}) | "
                f"OMM <T>={ov['T_total_mean_K']:.2f}K (sol={ov['T_solute_mean_K']:.2f}, wat={ov['T_solvent_mean_K']:.2f}, MB={ov['mb_std_ratio']:.2f}) ({res['elapsed_s']:.1f}s)"
            )

    # 3C: NPT Density & Compressibility
    for s_spec, prod_n in ((tip3p, 25000), (ubq, 25000), (dhfr, 18000)):
        if s_spec.name not in out["npt"]:
            t0 = time.perf_counter()
            res = _run_tier3c_npt(
                s_spec,
                eq_steps=250 if quick else 4000,
                prod_steps=750 if quick else prod_n,
                barostat_interval=25,
                allow_cpu_smoke=allow_cpu_smoke,
            )
            res["elapsed_s"] = round(time.perf_counter() - t0, 2)
            out["npt"][s_spec.name] = res
            checkpoint_cb(out)
            pn = res["prolix_pallas_v35"]
            on = res["openmm_cuda_mixed"]
            print(
                f"  [Tier 3C NPT] {s_spec.name:16s}: "
                f"Prolix <rho>={pn['density_mean_g_cm3']:.4f}±{pn['density_se_g_cm3']:.4f} g/cm3 (kT={pn['compressibility_kappa_T_1e5_bar_inv']:.2f}, acc={pn['mc_acceptance_rate']:.2f}) | "
                f"OMM <rho>={on['density_mean_g_cm3']:.4f}±{on['density_se_g_cm3']:.4f} g/cm3 (kT={on['compressibility_kappa_T_1e5_bar_inv']:.2f}) ({res['elapsed_s']:.1f}s)"
            )

    return out


# =============================================================================
# TIER 4: Multi-Replica (`B = 8`) Structural & Dynamical Observables
# =============================================================================


def _openmm_replica_worker(task: dict[str, Any]) -> dict[str, Any]:
    """Top-level spawn-safe worker running one OpenMM CUDA replica trajectory."""
    import openmm as mm
    from openmm import unit as _omm_unit
    unit = cast("Any", _omm_unit)

    from scripts.benchmarks.openmm_cuda_guard import require_openmm_cuda_context

    omm_sys = mm.XmlSerializer.deserialize(task["system_xml"])
    dt_ps = float(task["dt_fs"]) * 0.001
    gamma_ps = float(task["gamma_ps"])
    temp_K = float(task["temperature_K"])
    seed = int(task["seed"])
    eq_steps = int(task["eq_steps"])
    n_frames = int(task["n_frames"])
    save_every = int(task["save_every"])
    allow_cpu = bool(task["allow_cpu_smoke"])

    if gamma_ps > 0.0:
        integ = mm.LangevinMiddleIntegrator(
            temp_K * unit.kelvin, gamma_ps / unit.picosecond, dt_ps * unit.picoseconds
        )
        integ.setRandomNumberSeed(seed)
    else:
        integ = mm.VerletIntegrator(dt_ps * unit.picoseconds)

    ctx, prov = require_openmm_cuda_context(
        omm_sys,
        integ,
        expected_precision="mixed",
        allow_cpu_fallback=allow_cpu,
    )
    prov = {**prov, "caller": task["caller"]}

    pos_nm = np.asarray(task["positions_A"], dtype=np.float64) * 0.1
    ctx.setPositions(pos_nm * unit.nanometer)
    ctx.applyConstraints(1.0e-6)
    ctx.setVelocitiesToTemperature(temp_K * unit.kelvin, seed)
    ctx.applyVelocityConstraints(1.0e-6)

    if eq_steps > 0:
        integ.step(eq_steps)

    frames = np.empty((n_frames, pos_nm.shape[0], 3), dtype=np.float32)
    for f_i in range(n_frames):
        integ.step(save_every)
        st = ctx.getState(getPositions=True, enforcePeriodicBox=True)
        frames[f_i] = np.asarray(
            st.getPositions(asNumpy=True).value_in_unit(unit.angstrom), dtype=np.float32
        )
    del ctx, integ
    return {"replica_index": int(task["replica_index"]), "frames_A": frames, "provenance": prov}


def _run_openmm_multi_replica(
    spec: ValidatedSystemSpec,
    *,
    n_replicas: int,
    base_seed: int,
    dt_fs: float,
    temperature_K: float,
    gamma_ps: float,
    eq_steps: int,
    n_frames: int,
    save_every: int,
    openmm_workers: int,
    allow_cpu_smoke: bool,
    label: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run `n_replicas` independent OpenMM CUDA trajectories (`(B, T, N_real, 3)` Å)."""
    import openmm as mm

    sys_xml = mm.XmlSerializer.serialize(spec.omm_system)
    tasks = [
        {
            "replica_index": b,
            "system_xml": sys_xml,
            "positions_A": spec.positions_A,
            "dt_fs": dt_fs,
            "temperature_K": temperature_K,
            "gamma_ps": gamma_ps,
            "seed": base_seed + b,
            "eq_steps": eq_steps,
            "n_frames": n_frames,
            "save_every": save_every,
            "allow_cpu_smoke": allow_cpu_smoke,
            "caller": f"tier4_{label}_{spec.name}_rep{b}",
        }
        for b in range(n_replicas)
    ]
    n_pool = max(1, min(n_replicas, int(openmm_workers)))
    if n_pool == 1:
        outs = [_openmm_replica_worker(t) for t in tasks]
    else:
        ctx_mp = mp.get_context("spawn")
        with ctx_mp.Pool(processes=n_pool) as pool:
            outs = pool.map(_openmm_replica_worker, tasks)

    outs.sort(key=lambda d: d["replica_index"])
    traj = np.stack([d["frames_A"] for d in outs], axis=0)
    return traj, outs[0]["provenance"]


def _run_prolix_batched_multi_replica(
    spec: ValidatedSystemSpec,
    *,
    n_replicas: int,
    base_seed: int,
    dt_fs: float,
    temperature_K: float,
    gamma_ps: float,
    eq_steps: int,
    n_frames: int,
    save_every: int,
) -> np.ndarray:
    """Run `B` independent Prolix Pallas v3.5 replicas simultaneously on GPU with full state continuity."""
    sp = _make_prolix_pallas_stepper(
        spec, dt_fs=dt_fs, temperature_K=temperature_K, gamma_ps=gamma_ps
    )
    n_real = spec.n_real_atoms
    apply_fn = sp["apply_fn"]
    update_nl = sp["update_nl"]
    dt_s = sp["dt_s"]
    kT_s = jnp.asarray(
        (temperature_K * BOLTZMANN_KCAL) if gamma_ps > 0.0 else 0.0, dtype=jnp.float32
    )

    states_list = [sp["init_seed"](base_seed + b, init_temp_K=temperature_K) for b in range(n_replicas)]
    states_B = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *states_list)
    nbrs_B = jax.tree.map(
        lambda *xs: jnp.stack(xs, axis=0), *[sp["nbr0"] for _ in range(n_replicas)]
    )

    nl_sub = min(25, save_every)
    n_outer = max(1, save_every // nl_sub)

    def _single_subchunk(st_1, nb_1):
        def _body(_, s_c):
            return apply_fn(s_c, neighbor=nb_1, kT=kT_s, dt=dt_s)

        return jax.lax.fori_loop(0, nl_sub, _body, st_1)

    vmap_subchunk = jax.vmap(_single_subchunk, in_axes=(0, 0))

    @jax.jit
    def _advance_frame(st_b, nb_b):
        def _outer_body(_, carry):
            s_cur, n_cur = carry
            s_nxt = vmap_subchunk(s_cur, n_cur)
            n_nxt = jax.lax.map(lambda pair: update_nl(pair[0], pair[1]), (_state_pos(s_nxt), n_cur))
            return s_nxt, n_nxt

        return jax.lax.fori_loop(0, n_outer, _outer_body, (st_b, nb_b))

    n_eq_chunks = max(1, eq_steps // save_every) if eq_steps > 0 else 0
    for _ in range(n_eq_chunks):
        states_B, nbrs_B = _advance_frame(states_B, nbrs_B)

    box_vec_np = np.diag(np.asarray(spec.bundle_f64.box, dtype=np.float64)).astype(np.float32)
    frames = np.empty((n_replicas, n_frames, n_real, 3), dtype=np.float32)
    for f_i in range(n_frames):
        states_B, nbrs_B = _advance_frame(states_B, nbrs_B)
        pos_b = np.asarray(_state_pos(states_B)[:, :n_real, :], dtype=np.float32)
        frames[:, f_i, :, :] = np.mod(pos_b, box_vec_np[None, None, :])

    return frames


def run_tier4_multi_replica_observables(
    *,
    quick: bool,
    allow_cpu_smoke: bool,
    openmm_workers: int,
    existing_tier4: dict[str, Any] | None,
    checkpoint_cb: Any,
) -> dict[str, Any]:
    """Execute Tier 4 `B=8` structural & dynamical validation with OpenMM-A vs OpenMM-B control."""
    print("\n=== [Tier 4/4] Multi-Replica Structural & Dynamical Observables (with OMM-A vs OMM-B Control) ===")
    out: dict[str, Any] = dict(existing_tier4 or {})
    out.setdefault("water_tip3p", None)
    out.setdefault("proteins", {})

    B = 4 if quick else 8
    dt_fs = 1.0
    eq_steps = 300 if quick else 10000
    save_every = 100 if quick else 500  # 0.5 ps frame interval in --full (dt = 1.0 fs)
    n_frames = 15 if quick else 200  # 200 * 0.5 ps = 100.0 ps/replica in --full!
    frame_dt_ps = (save_every * dt_fs) * 0.001

    # 1. Liquid Water (`TIP3P_water_box`, L = 30.0 A): g_OO(r) + Langevin & NVE D_TIP3P
    if not out.get("water_tip3p"):
        t0 = time.perf_counter()
        tip3p = build_tip3p_system(allow_cpu_smoke=allow_cpu_smoke, minimize_iters=30 if quick else 80)
        box_L = 30.0

        # For water diffusion, save every 100 fs (0.1 ps) so unwrapped displacement << L/2 = 15 A (zero aliasing)
        w_save_every = 50 if quick else 100
        w_n_frames = 20 if quick else 500  # 50 ps/rep x 8 reps = 400 ps per engine at 0.1 ps resolution
        w_dt_ps = (w_save_every * dt_fs) * 0.001

        traj_prolix = _run_prolix_batched_multi_replica(
            tip3p,
            n_replicas=B,
            base_seed=100,
            dt_fs=dt_fs,
            temperature_K=300.0,
            gamma_ps=1.0,
            eq_steps=eq_steps,
            n_frames=w_n_frames,
            save_every=w_save_every,
        )
        traj_omm_a, prov_a = _run_openmm_multi_replica(
            tip3p,
            n_replicas=B,
            base_seed=200,
            dt_fs=dt_fs,
            temperature_K=300.0,
            gamma_ps=1.0,
            eq_steps=eq_steps,
            n_frames=w_n_frames,
            save_every=w_save_every,
            openmm_workers=openmm_workers,
            allow_cpu_smoke=allow_cpu_smoke,
            label="water_omm_a",
        )
        traj_omm_b, prov_b = _run_openmm_multi_replica(
            tip3p,
            n_replicas=B,
            base_seed=300,
            dt_fs=dt_fs,
            temperature_K=300.0,
            gamma_ps=1.0,
            eq_steps=eq_steps,
            n_frames=w_n_frames,
            save_every=w_save_every,
            openmm_workers=openmm_workers,
            allow_cpu_smoke=allow_cpu_smoke,
            label="water_omm_b",
        )

        # Subsample every 5th frame for g_OO(r) (100 frames x 8 replicas = 800 uncorrelated liquid frames)
        rdf_stride = max(1, w_n_frames // 80)
        o_idx = tip3p.water_oxygen_indices
        box_vec3 = np.array([box_L, box_L, box_L], dtype=np.float64)
        rdf_p = compute_water_rdf_goo(traj_prolix[:, ::rdf_stride][:, :, o_idx, :], box_vec3)
        rdf_a = compute_water_rdf_goo(traj_omm_a[:, ::rdf_stride][:, :, o_idx, :], box_vec3)
        rdf_b = compute_water_rdf_goo(traj_omm_b[:, ::rdf_stride][:, :, o_idx, :], box_vec3)

        fit_s = 0.2 if quick else 2.0
        fit_e = 0.8 if quick else 20.0
        diff_p = compute_unwrapped_water_diffusion(
            traj_prolix[:, :, o_idx, :], box_vec3, dt_save_ps=w_dt_ps, fit_start_ps=fit_s, fit_end_ps=fit_e
        )
        diff_a = compute_unwrapped_water_diffusion(
            traj_omm_a[:, :, o_idx, :], box_vec3, dt_save_ps=w_dt_ps, fit_start_ps=fit_s, fit_end_ps=fit_e
        )
        diff_b = compute_unwrapped_water_diffusion(
            traj_omm_b[:, :, o_idx, :], box_vec3, dt_save_ps=w_dt_ps, fit_start_ps=fit_s, fit_end_ps=fit_e
        )

        # Auxiliary NVE (`gamma = 0.0`) run (`B_nve = 4`, `15 ps/rep`) to quantify Newtonian D_PBC(L, gamma=0) + Yeh-Hummer D_inf
        b_nve = 2 if quick else 4
        nve_frames = 15 if quick else 150  # 15 ps at 0.1 ps/frame
        traj_p_nve = _run_prolix_batched_multi_replica(
            tip3p,
            n_replicas=b_nve,
            base_seed=400,
            dt_fs=dt_fs,
            temperature_K=300.0,
            gamma_ps=0.0,
            eq_steps=500 if quick else 2000,
            n_frames=nve_frames,
            save_every=w_save_every,
        )
        traj_o_nve, _ = _run_openmm_multi_replica(
            tip3p,
            n_replicas=b_nve,
            base_seed=500,
            dt_fs=dt_fs,
            temperature_K=300.0,
            gamma_ps=0.0,
            eq_steps=500 if quick else 2000,
            n_frames=nve_frames,
            save_every=w_save_every,
            openmm_workers=openmm_workers,
            allow_cpu_smoke=allow_cpu_smoke,
            label="water_omm_nve",
        )
        nve_fit_s = 0.2 if quick else 2.0
        nve_fit_e = 0.6 if quick else 10.0
        diff_p_nve = compute_unwrapped_water_diffusion(
            traj_p_nve[:, :, o_idx, :], box_vec3, dt_save_ps=w_dt_ps, fit_start_ps=nve_fit_s, fit_end_ps=nve_fit_e
        )
        diff_o_nve = compute_unwrapped_water_diffusion(
            traj_o_nve[:, :, o_idx, :], box_vec3, dt_save_ps=w_dt_ps, fit_start_ps=nve_fit_s, fit_end_ps=nve_fit_e
        )

        gp_curve = np.asarray(rdf_p["g_mean"], dtype=np.float64)
        ga_curve = np.asarray(rdf_a["g_mean"], dtype=np.float64)
        gb_curve = np.asarray(rdf_b["g_mean"], dtype=np.float64)

        water_rec = {
            "n_replicas": B,
            "box_L_A": box_L,
            "n_waters": tip3p.n_waters,
            "prod_ps_per_replica": round(w_n_frames * w_dt_ps, 2),
            "openmm_cuda_provenance": {"seed_a": prov_a, "seed_b": prov_b},
            "rdf_goo": {
                "prolix_pallas_v35": {k: v for k, v in rdf_p.items() if k not in ("r_A", "g_mean", "g_se")},
                "openmm_cuda_seed_a": {k: v for k, v in rdf_a.items() if k not in ("r_A", "g_mean", "g_se")},
                "openmm_cuda_seed_b": {k: v for k, v in rdf_b.items() if k not in ("r_A", "g_mean", "g_se")},
                "curve_rmse_prolix_vs_omm_a": float(np.sqrt(np.mean((gp_curve - ga_curve) ** 2))),
                "curve_rmse_omm_a_vs_omm_b_control": float(np.sqrt(np.mean((ga_curve - gb_curve) ** 2))),
            },
            "diffusion_langevin_gamma_1ps": {
                "prolix_pallas_v35": diff_p,
                "openmm_cuda_seed_a": diff_a,
                "openmm_cuda_seed_b": diff_b,
                "diff_prolix_minus_omm_a_1e5_cm2_s": float(
                    diff_p["D_pbc_mean_1e5_cm2_s"] - diff_a["D_pbc_mean_1e5_cm2_s"]
                ),
                "diff_omm_a_minus_omm_b_control_1e5_cm2_s": float(
                    diff_a["D_pbc_mean_1e5_cm2_s"] - diff_b["D_pbc_mean_1e5_cm2_s"]
                ),
            },
            "diffusion_newtonian_nve_gamma_0": {
                "prolix_pallas_v35": diff_p_nve,
                "openmm_cuda": diff_o_nve,
            },
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }
        out["water_tip3p"] = water_rec
        checkpoint_cb(out)
        print(
            f"  [Tier 4 TIP3P] g_OO peak: Prolix=({rdf_p['first_peak_r_A']:.3f}A, {rdf_p['first_peak_height']:.3f}) vs "
            f"OMM-A=({rdf_a['first_peak_r_A']:.3f}A, {rdf_a['first_peak_height']:.3f}) | "
            f"D(gamma=1): Prolix={diff_p['D_pbc_mean_1e5_cm2_s']:.3f}±{diff_p['D_pbc_se_1e5_cm2_s']:.3f}, "
            f"OMM-A={diff_a['D_pbc_mean_1e5_cm2_s']:.3f}±{diff_a['D_pbc_se_1e5_cm2_s']:.3f}, "
            f"OMM-B={diff_b['D_pbc_mean_1e5_cm2_s']:.3f}±{diff_b['D_pbc_se_1e5_cm2_s']:.3f} | "
            f"D_inf(NVE+YH): Prolix={diff_p_nve['D_infinite_box_est_1e5_cm2_s']:.3f} ({water_rec['elapsed_s']:.1f}s)"
        )

    # 2. Solvated Proteins & Active-Site Holo Complex (`1UAO`, `1UBQ`, `DHFR_PME`, `Holo_DHFR`)
    prot_builders: list[tuple[str, Any]] = [
        ("1UAO", lambda: build_1uao_system(allow_cpu_smoke=allow_cpu_smoke, minimize_iters=30 if quick else 80)),
        ("1UBQ", lambda: build_1ubq_system(allow_cpu_smoke=allow_cpu_smoke, minimize_iters=30 if quick else 80)),
        (
            "DHFR_PME",
            lambda: build_dhfr_system(
                electrostatics="pme", allow_cpu_smoke=allow_cpu_smoke, minimize_iters=30 if quick else 80
            ),
        ),
        (
            "Holo_DHFR",
            lambda: build_holo_dhfr_system(
                allow_cpu_smoke=allow_cpu_smoke, minimize_iters=50 if quick else 150
            ),
        ),
    ]

    for p_name, build_fn in prot_builders:
        if p_name in out["proteins"]:
            print(f"  [Tier 4 {p_name}] resumed from checkpoint")
            continue
        t0 = time.perf_counter()
        spec = build_fn()
        traj_p = _run_prolix_batched_multi_replica(
            spec,
            n_replicas=B,
            base_seed=100,
            dt_fs=dt_fs,
            temperature_K=300.0,
            gamma_ps=1.0,
            eq_steps=eq_steps,
            n_frames=n_frames,
            save_every=save_every,
        )
        traj_a, prov_a = _run_openmm_multi_replica(
            spec,
            n_replicas=B,
            base_seed=200,
            dt_fs=dt_fs,
            temperature_K=300.0,
            gamma_ps=1.0,
            eq_steps=eq_steps,
            n_frames=n_frames,
            save_every=save_every,
            openmm_workers=openmm_workers,
            allow_cpu_smoke=allow_cpu_smoke,
            label=f"{p_name}_omm_a",
        )
        traj_b, prov_b = _run_openmm_multi_replica(
            spec,
            n_replicas=B,
            base_seed=300,
            dt_fs=dt_fs,
            temperature_K=300.0,
            gamma_ps=1.0,
            eq_steps=eq_steps,
            n_frames=n_frames,
            save_every=save_every,
            openmm_workers=openmm_workers,
            allow_cpu_smoke=allow_cpu_smoke,
            label=f"{p_name}_omm_b",
        )

        obs_p = compute_protein_replica_observables(traj_p, spec)
        obs_a = compute_protein_replica_observables(traj_a, spec)
        obs_b = compute_protein_replica_observables(traj_b, spec)

        cmp_p_vs_a = compare_protein_ensembles(obs_p, obs_a)
        cmp_p_vs_b = compare_protein_ensembles(obs_p, obs_b)
        cmp_a_vs_b = compare_protein_ensembles(obs_a, obs_b)

        p_rec = {
            "n_real_atoms": spec.n_real_atoms,
            "n_replicas": B,
            "dt_fs": dt_fs,
            "prod_ps_per_replica": round(n_frames * frame_dt_ps, 2),
            "aggregate_ns_per_engine": round(B * n_frames * frame_dt_ps * 0.001, 3),
            "openmm_cuda_provenance": {"seed_a": prov_a, "seed_b": prov_b},
            "prolix_pallas_v35": strip_private_arrays(obs_p),
            "openmm_cuda_seed_a": strip_private_arrays(obs_a),
            "openmm_cuda_seed_b": strip_private_arrays(obs_b),
            "prolix_vs_openmm_a": cmp_p_vs_a,
            "prolix_vs_openmm_b": cmp_p_vs_b,
            "openmm_a_vs_openmm_b_control": cmp_a_vs_b,
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }
        out["proteins"][p_name] = p_rec
        checkpoint_cb(out)
        holo_str = ""
        if "holo_ligand" in obs_p:
            holo_str = (
                f" | LigRMSD: P={obs_p['holo_ligand']['pocket_heavy_rmsd_mean_A']:.2f}A vs "
                f"OMM-A={obs_a['holo_ligand']['pocket_heavy_rmsd_mean_A']:.2f}A"
            )
        print(
            f"  [Tier 4 {p_name:9s}] Ca-RMSD: P={obs_p['ca_rmsd_mean_A']:.2f}±{obs_p['ca_rmsd_se_A']:.2f}A, "
            f"OMM-A={obs_a['ca_rmsd_mean_A']:.2f}±{obs_a['ca_rmsd_se_A']:.2f}A, "
            f"OMM-B={obs_b['ca_rmsd_mean_A']:.2f}±{obs_b['ca_rmsd_se_A']:.2f}A | "
            f"RMSF r: P/A={cmp_p_vs_a['ca_rmsf_pearson_r']:.3f} (ctrl A/B={cmp_a_vs_b['ca_rmsf_pearson_r']:.3f})"
            f"{holo_str} ({p_rec['elapsed_s']:.1f}s)"
        )

    return out


# =============================================================================
# CLI Entrypoint
# =============================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Four-Tier Multi-Engine Physics & Observable Validation Suite (Pallas v3.5)."
    )
    parser.add_argument(
        "--tier",
        choices=("1", "2", "3", "4", "all"),
        default="all",
        help="Validation tier to execute (`1`, `2`, `3`, `4`, or `all`).",
    )
    mode_grp = parser.add_mutually_exclusive_group()
    mode_grp.add_argument(
        "--quick",
        action="store_true",
        help="Run fast smoke-test trajectories (reduced steps/replicas) to verify end-to-end execution.",
    )
    mode_grp.add_argument(
        "--full",
        action="store_true",
        help="Run full release-grade trajectories (`10,000` NVE steps, `B=8 x 100 ps/rep` observables; default).",
    )
    parser.add_argument(
        "--output",
        "--out",
        dest="output",
        type=Path,
        default=ROOT / "docs" / "benchmarks" / "261111_l40s_pallas_v35_physics_validation.json",
        help="Structured JSON output path.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing partial JSON checkpoint at `--output` and start fresh.",
    )
    parser.add_argument(
        "--openmm-workers",
        type=int,
        default=4,
        help="Concurrent OpenMM CUDA worker processes for Tier 4 multi-replica sampling (default: 4).",
    )
    parser.add_argument(
        "--allow-cpu-smoke",
        action="store_true",
        help="Allow OpenMM CPU fallback strictly for local non-GPU smoke tests (NEVER on L40S release runs).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    quick = bool(args.quick)
    mode_str = "quick" if quick else "full"
    output_path = args.output.resolve()

    payload: dict[str, Any] = {}
    if output_path.exists() and not args.no_resume:
        try:
            payload = json.loads(output_path.read_text())
            print(f"[resume] Loaded existing validation checkpoint from {output_path}")
        except Exception:
            payload = {}

    payload["provenance"] = _collect_provenance(allow_cpu_smoke=args.allow_cpu_smoke, mode=mode_str)
    _save_checkpoint(output_path, payload)

    tiers_to_run = ("1", "2", "3", "4") if args.tier == "all" else (args.tier,)
    t_start = time.perf_counter()

    if "1" in tiers_to_run:
        def _cb1(t1_data: dict[str, Any]) -> None:
            payload["tier1_single_point_parity"] = t1_data
            _save_checkpoint(output_path, payload)

        payload["tier1_single_point_parity"] = run_tier1_single_point_parity(
            quick=quick,
            allow_cpu_smoke=args.allow_cpu_smoke,
            existing_tier1=payload.get("tier1_single_point_parity"),
            checkpoint_cb=_cb1,
        )
        _save_checkpoint(output_path, payload)

    if "2" in tiers_to_run:
        def _cb2(t2_data: dict[str, Any]) -> None:
            payload["tier2_custom_vjp_fd_parity"] = t2_data
            _save_checkpoint(output_path, payload)

        payload["tier2_custom_vjp_fd_parity"] = run_tier2_custom_vjp_fd_parity(
            quick=quick,
            allow_cpu_smoke=args.allow_cpu_smoke,
            checkpoint_cb=_cb2,
        )
        _save_checkpoint(output_path, payload)

    if "3" in tiers_to_run:
        def _cb3(t3_data: dict[str, Any]) -> None:
            payload["tier3_conservation_and_thermodynamics"] = t3_data
            _save_checkpoint(output_path, payload)

        payload["tier3_conservation_and_thermodynamics"] = run_tier3_conservation_and_thermodynamics(
            quick=quick,
            allow_cpu_smoke=args.allow_cpu_smoke,
            existing_tier3=payload.get("tier3_conservation_and_thermodynamics"),
            checkpoint_cb=_cb3,
        )
        _save_checkpoint(output_path, payload)

    if "4" in tiers_to_run:
        def _cb4(t4_data: dict[str, Any]) -> None:
            payload["tier4_multi_replica_observables"] = t4_data
            _save_checkpoint(output_path, payload)

        payload["tier4_multi_replica_observables"] = run_tier4_multi_replica_observables(
            quick=quick,
            allow_cpu_smoke=args.allow_cpu_smoke,
            openmm_workers=args.openmm_workers,
            existing_tier4=payload.get("tier4_multi_replica_observables"),
            checkpoint_cb=_cb4,
        )
        _save_checkpoint(output_path, payload)

    payload["mode"] = mode_str
    payload["n_tiers_executed"] = len(tiers_to_run)
    payload["openmm_platform"] = str(
        payload.get("provenance", {}).get("openmm_live_cuda_verification", {}).get("platform_name", "unknown")
    )
    payload["total_wall_time_s"] = round(time.perf_counter() - t_start, 2)
    _save_checkpoint(output_path, payload)
    print(f"\n[DONE] Wrote four-tier physics & observable validation report to {output_path} ({payload['total_wall_time_s']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
